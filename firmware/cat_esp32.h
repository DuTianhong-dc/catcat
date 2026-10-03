#ifndef __CAT_ESP32_H
#define __CAT_ESP32_H
#include "stm32f4xx.h"

/* ============================================================
 *  ESP32-S3 通信链路驱动 (USART3 / PB10-PB11)
 *  协议见 通信协议.txt 第五章。
 *
 *  链路: 上位机 --BLE--> ESP32-S3 --UART--> 本板
 *  ESP32 只是透传桥, 协议全在本板实现。
 *  角色分工: 本板跑策略+控制; 上位机只发指令、收状态, 不参与控制环。
 * ============================================================ */

/* ==================== 通信参数 ==================== */
#define ESP32_BAUD        115200u   /* 8N1 */
#define ESP32_RX_BUF       256u     /* 环形接收缓冲(2026-09-29: 64→256, 为 OTA 大流量兜底) */
/* (原 ESP32_TX_BUF 96 已删, 2026-10-03 审查 T4: 从未被引用 —— 上行组包缓冲是
 *  esp32_send_up() 里的局部 f[64], 真要用再按需定义, 别留死常量。) */

/* ---- OTA 帧(策略在线更新; 帧格式与流程 → 通信协议.txt 第七章) ----
 * 下行第二类帧: CC | len | 数据(len) | ck —— 与 10B 控制帧并存, 首字节区分。
 * 本模块只负责"收帧边界 + 校验和"; 子类型分派与处理 → cat_polupd。 */
#define ESP32_OTA_HEAD      0xCCu
#define ESP32_OTA_MAX         96u   /* 单帧数据上限(BEGIN 帧数据 73B 为最大) */

/* ==================== 下行帧 (上位机 → 本板, 固定 10 字节) ====================
 *   AA 55 | vx[2] | wz[2] | flags[1] | pan[1] | tilt[1] | ck[1]
 *    0  1     2-3     4-5      6          7        8         9
 *  vx/wz: int16 小端, 实际值 = 原始值 / 1000.0 (m/s, rad/s)
 *  flags: 只有 bit0 = 使能 (0 = 停止运动, 原地站立; 1 = 运行)
 *    ⚠️ 注意"停止"≠"卸力": 卸力是 0x28 扭矩开关 = 0, 会让舵机断电、猫直接趴下。
 *       本位的 0 只表示"不执行新的移动指令", 舵机仍保持出力站住。
 *  ck   : (字节 2~8 累加) & 0xFF —— 不含帧头, 跟舵机同规则 */
#define ESP32_DL_HEAD0       0xAAu
#define ESP32_DL_HEAD1       0x55u
#define ESP32_DL_LEN           10u
#define ESP32_DL_CK_OFF        9u
#define ESP32_DL_CK_FIRST      2u   /* 校验和从哪一字节开始累加 */
#define ESP32_DL_SCALE     1000.0f  /* 定点数: 实际值 = 原始 / 1000 */

/* 上位机下行发送频率(2026-09-26 定)。
 * ⚠️ 固件【不】据此定时 —— 收是中断驱动的; 这个值只用于两件事:
 *    ① 推 DEADMAN 的下限  ② 日后验算下行缓冲够不够 */
#define ESP32_DL_HZ          100u

/* ==================== 上行帧 (本板 → 上位机) ====================
 *   5A A5 | len | type | 载荷 | ck
 *   len = 载荷长度; ck = (从字节 0 累加) & 0xFF —— 含帧头, 跟 IMU 同规则 */
#define ESP32_UP_HEAD0       0x5Au
#define ESP32_UP_HEAD1       0xA5u
#define ESP32_UP_TYPE_FULL   0x01u  /* 全量状态包(当前只有这一种) */
#define ESP32_UP_PAYLOAD      86u   /* 载荷字节数 */
#define ESP32_UP_LEN          91u   /* 整帧 = 2 帧头 + 1 len + 1 type + 86 载荷 + 1 ck */

/* 上报周期: 10Hz。它的倒数就是"状态多久刷新一次", 改这里要同步改上位机解析节奏 */
#define ESP32_UP_PERIOD_MS    100u

/* 失联保护(DEADMAN): 超过这个时间没收到有效下行帧, 就把指令归零(原地站立)。
 * 目的: 上位机断连时不能让猫"抱着最后一个指令一直走下去"。
 * ⚠️ 要大于【发送间隔 × 5】。发送间隔 = 1000/ESP32_DL_HZ = 10ms → 下限 50ms。
 *    现在取 1000ms: 100Hz 下相当于允许连丢 100 帧, 抗 BLE 抖动很宽;
 *    真断连时晚 1 秒停也无妨 —— 误触发的代价更大(猫走得好好的突然停)。 */
#define ESP32_DEADMAN_MS     1000u

/* 频率一改, 不成立的组合在这里直接编译报错(免得又忘了校) */
#if ESP32_DEADMAN_MS < (1000u / ESP32_DL_HZ * 5u)
#error "DEADMAN 太短, 会被正常抖动误触发(要 >= 发送间隔 x 5)"
#endif

/* ==================== 下行指令(解析结果, 供上层取用) ==================== */
typedef struct {
    float   vx;      /* m/s, 正 = 前进 */
    float   wz;      /* rad/s, 正 = 左转 */
    uint8_t enable;  /* 使能位: 0 = 停止运动(原地站立, 舵机保持出力) 1 = 运行 */
    int8_t  pan;     /* 云台水平, -127~127 映射 0~180 度 */
    int8_t  tilt;    /* 云台俯仰, 同上 */
} esp32_cmd_t;

/* ==================== 函数原型 ==================== */
/* ① 初始化 */
void esp32_init(void);              //配 USART3 + PB10/PB11(AF7) + 开接收中断

/* ② 收发 */
void esp32_data_recv(uint8_t byte);                              //中断里喂 1 个字节进环形缓冲
void esp32_data_send(const uint8_t *tx_buf, uint16_t tx_len);    //发一串字节
void esp32_send_up(uint8_t type, const uint8_t *payload, uint8_t len);  //组上行帧(5A A5..)并发送

/* ③ 解析 */
void esp32_poll(void);              //主循环里调: 排空缓冲喂状态机 + 判失联(DEADMAN)

/* ④ 取数 */
const esp32_cmd_t *esp32_get_cmd(void);   //取最新一条指令(返回指针, 只读)
uint8_t               esp32_alive(void);      //上位机是否还在连(未超 DEADMAN)
/* 取一帧已收好的 OTA 数据(首字节=子类型)。有则返回缓冲指针与长度, 无则 NULL。
 * 取走即释放 —— 处理完再收下一帧(主循环节奏足够)。 */
const uint8_t *esp32_ota_fetch(uint16_t *len);

/* ⚠️ 状态上报(组包)不在这里: 见 main.c 的 report_state()。
 *   它要同时取 IMU 和舵机的数据, 属于跨模块编排, 放本模块会让本模块反过来依赖 cat_imu。 */

#endif
