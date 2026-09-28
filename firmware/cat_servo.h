#ifndef __CAT_SERVO_H
#define __CAT_SERVO_H
#include "stm32f4xx.h"
/* ==================== 帧缓冲大小 ====================
 * 收发各一块 tx_frame[BUFFER] / rx_frame[BUFFER], 是"整帧"上限, 不是"参数"上限。
 * 128 的依据、以及再往上调时要连带核查哪两处类型, 见 舵机驱动说明.md(二)。 */
#define BUFFER      128
#define SERVO_COUNT 12              //总线上舵机颗数(ID 依次 1~SERVO_COUNT)

/* ==================== 协议常量 ==================== */
#define SCS_HEADER1   0xFF          //帧头第 1 字节
#define SCS_HEADER2   0xFF          //帧头第 2 字节
#define SCS_BROADCAST 0xFE          //广播 ID: 所有舵机都收, 但都不应答(除 PING)

/* 收应答超时(真微秒)。5ms 的实测推导、以及"开机首次 ping 需上层重试", 见 舵机驱动说明.md(三)。 */
#define SCS_TMO_US  5000u           //收应答超时 5ms

/* 应答帧 FF FF|ID|LEN|ERROR|数据…|CK 的固定开销与数据段起点。LEN = 数据个数 + 2, 见 说明.md(五)。 */
#define SCS_RESP_FIXED  6u          //应答固定开销(无数据时整帧就是这个长度)
#define SCS_RESP_DATA   5u          //数据段起始下标

/* 指令帧 FF FF|ID|LEN|指令|参数…|CK 的固定开销(字头2+ID1+LEN1+指令1+CK1)。
 * 参数区容量 = BUFFER - 它, 见 servo_com_pack_set()。 */
#define SCS_FRAME_FIXED 6u

/* 同步读的每组颗数。为何固定 4(12 颗一帧必丢尾), 见 舵机驱动说明.md(四)。
 * 不带 u 后缀: 与 SERVO_COUNT 同类型, 避免符号比较警告。 */
#define SCS_SYNC_GROUP  4

// 结构体、枚举、extern、函数原型
typedef struct{
uint8_t length;         //舵机指令包数据长度，等于待发送的参数 N 加上 2，即“N+2”，这个2包含指令以及校验和。
uint8_t instruction;    //舵机指令
uint8_t parameter[BUFFER-6];/*除指令另外需要补充的控制信息，参数最大支持双字节参数表示一个内存值，字节顺序参考舵机使
用手册内存控制表。*/
/* 校验和不放在结构体里了: 它由 servo_checksum() 在拼帧时就地算, 见 cat_servo.c */
}command_packet_t;

/* 指令码。来源: 舵机资料.txt 手册 4 章的指令总表, **12 条全收**(2026-09-24 脚本逐条比对过)。
 * 已有函数的: PING / READ / WRITE / SYNC_READ / SYNC_WRITE;
 * 余下的是补全收录, 暂未写对应函数。 */
typedef enum {
    SCS_PING        = 0x01,    /* 查询工作状态 */
    SCS_READ        = 0x02,    /* 读控制表 */
    SCS_WRITE       = 0x03,    /* 写控制表 */
    SCS_REG_WRITE   = 0x04,    /* 异步写(先存着, 等 ACTION 才动) */
    SCS_ACTION      = 0x05,    /* 触发异步写 */
    SCS_RECOVERY    = 0x06,    /* 参数恢复: 恢复除 ID 号以外的全部参数 */
    SCS_REBOOT      = 0x08,    /* 重启舵机 */
    SCS_BACKUP      = 0x09,    /* 参数备份(配 0x06 恢复用) */
    SCS_RESET       = 0x0A,    /* 状态重置: 清圈数 */
    SCS_CALIB       = 0x0B,    /* 位置校准: 当前位重校准为设定值(无参数=中位, 带 2B=指定值) */
    SCS_SYNC_READ   = 0x82,    /* 同步读(多舵机) */
    SCS_SYNC_WRITE  = 0x83     /* 同步写(多舵机) */
} scs_instr_t;

/* ==================== 内存表地址 ====================
 * 来源 xlsx 子表"磁编码SMS&STS", 逐行抄录。EPROM 0x00~0x27(改前须先解 0x37 锁) / SRAM 0x28~0x45。
 * 多字节量一律低字节在前。各量的类型与单位换算见 舵机驱动说明.md(七)。 */
typedef enum {
    /* ---------- EPROM 区 (掉电保存) ---------- */
    SCS_ADDR_FW_MAJOR      = 0x00,  /* [1B] 固件主版本号              只读 */
    SCS_ADDR_FW_MINOR      = 0x01,  /* [1B] 固件次版本号              只读 */
    SCS_ADDR_FW_END        = 0x02,  /* [1B] END                       只读 */
    SCS_ADDR_SERVO_MAJOR   = 0x03,  /* [1B] 舵机主版本号              只读 */
    SCS_ADDR_SERVO_MINOR   = 0x04,  /* [1B] 舵机次版本号              只读 */
    SCS_ADDR_ID            = 0x05,  /* [1B] ID 0~253 (254=广播) */
    SCS_ADDR_BAUD          = 0x06,  /* [1B] 0~7 → 1M/500k/250k/128k/115200/76800/57600/38400 */
    SCS_ADDR_RETURN_DELAY  = 0x07,  /* [1B] 返回延时, 单位 2us, 最大 508us (STS 无此功能) */
    SCS_ADDR_RESP_LEVEL    = 0x08,  /* [1B] 应答级别: 0=只有读/PING 应答, 1=所有指令都应答 */
    SCS_ADDR_ANGLE_MIN     = 0x09,  /* [2B] 最小角度限制 (步) */
    SCS_ADDR_ANGLE_MAX     = 0x0B,  /* [2B] 最大角度限制 (步) */
    SCS_ADDR_TEMP_MAX      = 0x0D,  /* [1B] 最高温度上限 (℃) */
    SCS_ADDR_VOLT_MAX      = 0x0E,  /* [1B] 最高输入电压 (0.1V) */
    SCS_ADDR_VOLT_MIN      = 0x0F,  /* [1B] 最低输入电压 (0.1V) */
    SCS_ADDR_TORQUE_MAX    = 0x10,  /* [2B] 最大扭矩 0~1000=0~100% (上电赋给 0x30) */
    SCS_ADDR_PHASE         = 0x12,  /* [1B] 相位(特殊功能字节, 勿乱改; 位定义见 SCS_PH_*) */
    SCS_ADDR_UNLOAD_COND   = 0x13,  /* [1B] 卸载条件(各保护开关; 位定义见 SCS_UL_*) */
    SCS_ADDR_LED_ALARM     = 0x14,  /* [1B] LED 报警条件(位定义同卸载条件) */
    SCS_ADDR_POS_P         = 0x15,  /* [1B] 位置环 P 比例系数 */
    SCS_ADDR_POS_D         = 0x16,  /* [1B] 位置环 D 微分系数 */
    SCS_ADDR_POS_I         = 0x17,  /* [1B] 位置环 I 积分系数 */
    SCS_ADDR_START_FORCE   = 0x18,  /* [1B] 最小启动力 (0.1%) */
    SCS_ADDR_INTEGRAL_LIM  = 0x19,  /* [1B] 积分限制值 (0=关闭) */
    SCS_ADDR_DEADZONE_CW   = 0x1A,  /* [1B] 顺时针不灵敏区 (步) */
    SCS_ADDR_DEADZONE_CCW  = 0x1B,  /* [1B] 逆时针不灵敏区 (步) */
    SCS_ADDR_CURR_PROT     = 0x1C,  /* [2B] 保护电流 (6.5mA/LSB, 最大 3250mA) */
    SCS_ADDR_RESOLUTION    = 0x1E,  /* [1B] 角度分辨率放大系数 1~3 (多圈控制) */
    SCS_ADDR_POS_CORRECT   = 0x1F,  /* [2B] 位置校正 (-2047~2047, BIT11 为方向) */
    SCS_ADDR_RUN_MODE      = 0x21,  /* [1B] 0 位置伺服 / 1 恒速 / 2 PWM开环 / 3 步进 */
    SCS_ADDR_PROT_TORQUE   = 0x22,  /* [1B] 保护扭矩 (1.0%) */
    SCS_ADDR_PROT_TIME     = 0x23,  /* [1B] 保护时间 (10ms/LSB) */
    SCS_ADDR_OVERLOAD_TQ   = 0x24,  /* [1B] 过载扭矩阈值 (1.0%) */
    SCS_ADDR_SPEED_P       = 0x25,  /* [1B] 速度闭环 P (模式1 用) */
    SCS_ADDR_OC_PROT_TIME  = 0x26,  /* [1B] 过流保护时间 (10ms/LSB) */
    SCS_ADDR_SPEED_I       = 0x27,  /* [1B] 速度闭环 I (模式1 用) */

    /* ---------- SRAM 区 (掉电丢失) ---------- */
    SCS_ADDR_TORQUE_EN     = 0x28,  /* [1B] 扭矩开关: 0=卸力 1=使能 128=把当前位置校正为2048 */
    SCS_ADDR_ACCEL         = 0x29,  /* [1B] 加速度 (100 步/s² /LSB) */
    SCS_ADDR_GOAL_POS      = 0x2A,  /* [2B] 目标位置 (步) */
    SCS_ADDR_RUN_TIME      = 0x2C,  /* [2B] 运行时间 (PWM开环模式用, 50~1000, BIT10 方向) */
    SCS_ADDR_RUN_SPEED     = 0x2E,  /* [2B] 运行速度 (步/s, 50步/s = 0.732 RPM) */
    SCS_ADDR_TORQUE_LIMIT  = 0x30,  /* [2B] 转矩限制 (上电由 0x10 赋值) */
    SCS_ADDR_LOCK          = 0x37,  /* [1B] 锁标志: 0=解锁(写EPROM掉电保存) 1=上锁(不保存) */
    SCS_ADDR_PRESENT_POS   = 0x38,  /* [2B] 当前位置 (步)              只读 */
    SCS_ADDR_PRESENT_SPEED = 0x3A,  /* [2B] 当前速度 (步/s)            只读 */
    SCS_ADDR_PRESENT_LOAD  = 0x3C,  /* [2B] 当前负载 (0.1%, 驱动电压占空比) 只读 */
    SCS_ADDR_PRESENT_VOLT  = 0x3E,  /* [1B] 当前电压 (0.1V)            只读 */
    SCS_ADDR_PRESENT_TEMP  = 0x3F,  /* [1B] 当前温度 (℃)              只读 */
    SCS_ADDR_ASYNC_FLAG    = 0x40,  /* [1B] 异步写标志                 只读 */
    SCS_ADDR_STATUS        = 0x41,  /* [1B] 舵机状态/错误位 (见 SCS_ST_*) 只读 */
    SCS_ADDR_MOVING_FLAG   = 0x42,  /* [1B] 移动标志: 1=运动中 0=停止   只读 */
    /* 0x43~0x44 [2B] ⚠️存疑, 不定义不使用。资料冲突与真机裁决法见 舵机驱动说明.md(6.1) */
    SCS_ADDR_PRESENT_CURR  = 0x45   /* [2B] 当前电流 (6.5mA/LSB, 最大 3250mA) 只读 */
} scs_mem_addr_t;

/* ==================== 15 字节反馈块的字段偏移 ====================
 * 一条 SyncRead 从 0x38 连读 15 字节, 一次拿回整套反馈(位置/速度/负载/电压/温度/状态/电流)。
 * 偏移一律用【地址差】算 —— 以后改读的起始地址, 这里自动跟着走, 不会静默错位。
 * (必须放在 scs_mem_addr_t 之后: 这些宏要用到枚举成员) */
#define SCS_FB_LEN       15u        /* 块长: 0x38(位置) ... 0x46(电流末字节) */
#define SCS_FB_OFF_POS   (SCS_ADDR_PRESENT_POS   - SCS_ADDR_PRESENT_POS)  /* = 0  位置 2B */
#define SCS_FB_OFF_SPEED (SCS_ADDR_PRESENT_SPEED - SCS_ADDR_PRESENT_POS)  /* = 2  速度 2B(有符号) */
#define SCS_FB_OFF_LOAD  (SCS_ADDR_PRESENT_LOAD  - SCS_ADDR_PRESENT_POS)  /* = 4  负载 2B(有符号) */
#define SCS_FB_OFF_VOLT  (SCS_ADDR_PRESENT_VOLT  - SCS_ADDR_PRESENT_POS)  /* = 6  电压 1B */
#define SCS_FB_OFF_TEMP  (SCS_ADDR_PRESENT_TEMP  - SCS_ADDR_PRESENT_POS)  /* = 7  温度 1B */
#define SCS_FB_OFF_STAT  (SCS_ADDR_STATUS        - SCS_ADDR_PRESENT_POS)  /* = 9  状态位 1B */
#define SCS_FB_OFF_CURR  (SCS_ADDR_PRESENT_CURR  - SCS_ADDR_PRESENT_POS)  /* = 13 电流 2B(有符号) */

/* 编译期兜底: 15 字节必须正好覆盖到"电流的末字节", 少一个都不行 */
typedef char _scs_fb_len_check[(SCS_FB_OFF_CURR + 2u == SCS_FB_LEN) ? 1 : -1];

/* ---- 0x41 舵机状态: 错误位(置 1 = 该错误出现) ---- */
#define SCS_ST_LOAD      (1u << 5)  /* 负载过载 */
#define SCS_ST_CURRENT   (1u << 3)  /* 电流过流 */
#define SCS_ST_TEMP      (1u << 2)  /* 温度过热 */
#define SCS_ST_ENCODER   (1u << 1)  /* 磁编码异常 */
#define SCS_ST_VOLT      (1u << 0)  /* 电压过压/欠压 */

/* ---- 0x13 卸载条件 / 0x14 LED报警条件: 各保护开关(置 1 = 开启) ---- */
#define SCS_UL_LOAD      (1u << 5)  /* 负载过载 */
#define SCS_UL_CURRENT   (1u << 3)  /* 过流 */
#define SCS_UL_TEMP      (1u << 2)  /* 过热 */
#define SCS_UL_ENCODER   (1u << 1)  /* 磁编码 */
#define SCS_UL_VOLT      (1u << 0)  /* 电压 */

/* ---- 0x12 相位(特殊功能字节, 无特别需求不要改) ---- */
#define SCS_PH_DIR       (1u << 7)  /* 伺服方向: 0=正方向 1=反时钟 */
#define SCS_PH_PWM       (1u << 6)  /* PWM 模式: 0=高频无死区 1=低频有死区 */
#define SCS_PH_VOLT_SAMP (1u << 5)  /* 电压采样: 0=1.5K低压 1=1K高压 */
#define SCS_PH_FB_MODE   (1u << 4)  /* 反馈模式: 0=单圈角度 1=全角度(多圈) */
#define SCS_PH_SPD_MODE  (1u << 3)  /* 速度模式: 0=速度0停止 1=速度0最高速 */
#define SCS_PH_SPD_UNIT  (1u << 2)  /* 速度单位: 0=50步/秒 1=步/秒 */
#define SCS_PH_BRIDGE    (1u << 1)  /* 驱动桥: 0=无刷马达 1=有刷马达 */
#define SCS_PH_COEF      (1u << 0)  /* 相位系数: 0=正向 1=反向 */

/* 通信结果 / 错误类型: 0x01~0x0F 是【帧层面】的失败(我们自己检测的);
 * 舵机自报的错误统一归 SCS_ERR_SERVO, 具体错误位用 servo_last_status() 取(位标志, 见 SCS_ST_*)。 */
typedef enum {
    SCS_OK           = 0x00,    /* 一切正常 */
    SCS_ERR_HEADER   = 0x01,    /* 应答字头不是 FF FF */
    SCS_ERR_ID       = 0x02,    /* 应答 ID 不是目标舵机 */
    SCS_ERR_CHECKSUM = 0x03,    /* 应答校验和不对 */
    SCS_ERR_PARAM    = 0x04,    /* 请求参数不合法(超限等), 帧没发出去 */
    SCS_ERR_SERVO    = 0x05,    /* 舵机自报错误, 具体错误位用 servo_last_status() 取 */
    SCS_ERR_TIMEOUT  = 0x06,    /* 等应答超时(舵机没答 / 断线 / 没通电) */
    SCS_ERR_LEN      = 0x07     /* 帧长不合法: 比最短应答还短, 或 LEN 字段与实际帧长对不上 */
} scs_err_t;

// ==================== 函数原型 ====================
/* ① 初始化 */
void      servo_init(void);                                        // 初始化 USART6 与 PC6/PC7（上电调用一次）
/* ② 收发 */
void      servo_data_send(const uint8_t *tx_buf, uint16_t tx_len); // 从 USART6 依次发出 tx_len 个字节
/* ③ 包组装 */
uint8_t   servo_checksum(const uint8_t *buf, uint16_t length);                   // 算 SCS 校验和(指令/应答同一公式)
scs_err_t servo_com_pack_set(uint8_t sid, uint8_t inst, const uint8_t *param, uint8_t count); // 备好指令包，自动算 LEN
void      servo_tx_frame_build(void);                                                  // com_pack → tx_frame(含校验和)
scs_err_t servo_rx_frame_check(const uint8_t *rx_buf, uint16_t rx_len, uint8_t sid);   // 校验一帧应答
uint8_t   servo_last_status(void);  //上次应答的 ERROR 字节(位标志见 SCS_ST_*); 仅对单颗操作有意义
/* ④ 指令 */
uint16_t  servo_ping_all(void);                                         // 依次查询 SERVO_COUNT 颗, 返回没应答的位图
scs_err_t servo_read(uint8_t sid, uint8_t addr, uint8_t count, uint8_t *data_buf);         // 单颗: 读控制表 count 字节
scs_err_t servo_write(uint8_t sid, uint8_t addr, const uint8_t *data_buf, uint8_t count);  // 单颗: 写控制表 count 字节
uint16_t  servo_sync_read(uint8_t addr, uint8_t count, uint8_t *data_buf);         // 多颗: 一条读指令(分组), 返回缺失位图
scs_err_t servo_sync_write(uint8_t addr, uint8_t count, const uint8_t *data_buf);  // 多颗: 一条写指令(广播), 无应答
scs_err_t servo_reset(uint8_t sid);                                                // 单颗: 状态重置(清圈数)
scs_err_t servo_recovery(uint8_t sid);                                             // 单颗: 参数恢复 ⚠️抹掉 EPROM 配置

/* ==================== 小工具 ====================
 * 把"低字节在前"的两个字节合成数值。哪些量有符号、单位怎么换, 见 舵机驱动说明.md(七)。 */
static inline uint16_t scs_u16(const uint8_t *p){
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}
static inline int16_t scs_i16(const uint8_t *p){
    return (int16_t)scs_u16(p);
}

#endif
