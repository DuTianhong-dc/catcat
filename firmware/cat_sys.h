#ifndef __CAT_SYS_H
#define __CAT_SYS_H
#include "stm32f4xx.h"

/* ============================================================
 *  观测区 —— 固定地址的一块内存, 给 ST-Link 直接读固件内部状态用
 *  (不必接串口、不必开 Keil 调试会话; 详见 .workbuddy 记忆里的读法)
 *
 *  ⚠️ 改这个结构体必须同步改三处:
 *     ① ST-Link 读法的字数  ② 下面那个长度检查  ③ 注释里的偏移量
 *  魔数读到 0xCA7DB001 才说明位置对了。
 * ============================================================ */
#define DBG_MAGIC   0xCA7DB001u

/* stage: 走到哪一步了。用宏不用裸数字, 免得读回来对不上 */
#define DBG_STAGE_INIT      1u      /* 时钟/时基配完 */
#define DBG_STAGE_BOOT      2u      /* 正在等外设就绪 */
#define DBG_STAGE_READY     3u      /* 外设就绪, 进主循环 */
#define DBG_STAGE_TIMEOUT   4u      /* 上电等待超时(外设没等到) */

typedef struct {
    /* ---- 状态量 ---- */
    uint32_t magic;        /* +0   恒为 DBG_MAGIC */
    uint32_t stage;        /* +4   DBG_STAGE_* */
    uint32_t loops;        /* +8   主循环圈数, 在涨说明没死 */
    uint32_t ms;           /* +12  运行毫秒(sys_ms()) */
    /* ---- 外设结果 ---- */
    uint32_t servo_miss;   /* +16  上电自检: 没应答的颗(位图, bit0=ID1) */
    uint32_t sync_miss;    /* +20  每拍同步读的缺失位图 */
    uint32_t servo_err;    /* +24  最近一次单颗操作的 scs_err_t */
    uint32_t imu_frames;   /* +28  IMU 累计有效帧数 */
    uint32_t imu_rate_fix; /* +32  补设输出频率的次数: 0=本来就是100Hz; 每次都是1 → 不保存 */
    /* ---- 舵机电压诊断 (2026-09-25 加的: 12 颗全报 ERROR=0x01 电压位, 要查阈值) ---- */
    uint32_t servo_vmax;   /* +36  舵机"最高输入电压"阈值(0x0E, 0.1V/LSB) */
    uint32_t servo_vmin;   /* +40  舵机"最低输入电压"阈值(0x0F, 0.1V/LSB) */
    uint32_t servo_vnow;   /* +44  舵机"当前电压"(0x3E, 0.1V/LSB) */
    uint32_t servo_lasterr;/* +48  舵机自报的错误位(应答帧 ERROR 字节, 见 SCS_ST_*) */
    /* ---- 实时数据 ---- */
    float    pos[12];      /* +52  12 颗当前位置(步) */
    float    gyro[3];      /* +100 机体系角速度 rad/s */
    float    grav[3];      /* +112 重力方向(世界 -Z 在机体系) */
    float    euler[3];     /* +124 欧拉角(度) */
    /* ---- obs 数据链探针 + 舵机反馈(2026-09-27 接 cat_obs 时加, 真机验证管道用) ---- */
    float    linvel[3];    /* +136 obs[0:3)    里程计估计的机体线速度 m/s */
    float    jpos[12];     /* +148 obs[11:23)  关节角(减默认站姿后) rad */
    float    tau[12];      /* +196 obs[35:47)  关节力矩 N·m */
    uint32_t temp[12];     /* +244 舵机温度 ℃(0x3F) */
    uint32_t status[12];   /* +292 舵机状态位(0x41) */
    /* ---- 策略输出探针(2026-09-27 接 cat_policy 时加) ---- */
    float    pol_act[12];  /* +340 本拍策略动作(tanh 后, 范围 [-1,1]) */
    uint32_t pol_clip;     /* +388 本拍被关节限位钳掉的颗数(0 = 全在限内) */
    /* ---- 分段耗时探针(2026-09-27: 定位"主循环为什么 32ms(应 20ms)") ----
     * 单位 = DWT 周期数(CYCCNT); 168MHz 下 168 周期 = 1us。记录【最近一拍】的值。
     * 用途: 一眼看出时间花在哪段, 不用反复改代码烧录试。 */
    uint32_t t_imu;        /* +392 imu_poll() */
    uint32_t t_sync;       /* +396 servo_sync_read()(含等 12 颗应答) */
    uint32_t t_obs;        /* +400 obs_update() + obs_get()(含里程计 FK) */
    uint32_t t_pol;        /* +404 policy_run() */
    uint32_t t_up;         /* +408 report_state()(含串口阻塞发送) */
    uint32_t t_send;       /* +412 servo_sync_write()(发目标位置) */
    /* ---- 版本 + 整拍(2026-09-27 下午加) ----
     * fw_ver: 观测区里"改了代码"和"没烧进去"的字段值可能一模一样, 靠它区分。
     * t_all : 整拍【忙碌】耗时(不含等下一拍); 与 ms/loops(整拍周期)相减 = 空等时间。 */
    uint32_t fw_ver;       /* +416 固件版本标识, 见 DBG_FW_VER */
    uint32_t t_all;        /* +420 整拍忙碌耗时 */
    /* ---- 安全闸探针(2026-09-28 接线收口时加) ----
     * safety_trip 非 0 = 保护锁存 → 主循环每拍广播卸力、不再发新目标。 */
    uint32_t safety_clip;  /* +424 本拍被机械行程表钳掉的颗数(0 = 全在表内) */
    uint32_t safety_trip;  /* +428 保护锁存: 0=正常 1=低压 2=过载 3=过热 */
} dbg_t;                   /* 共 432 字节 = 108 个 32 位字 */

/* 固件版本标识: 改了代码就改它。
 * ⚠️ 用途: 观测区新旧版本的字段值可能完全一样(比如 servo_err 新旧都是 0), 没有它
 *    就没法区分"烧录没成功"和"改了没效果" —— 2026-09-27 上午就为此白折腾了一轮。 */
#define DBG_FW_VER   0x20260928u

extern dbg_t g_dbg;        /* 定义在 cat_sys.c, 放在 CCM RAM 0x10000000 */

/* 长度兜底: 改了结构体忘了同步读法, 这里直接编译报错 */
typedef char _dbg_size_check[(sizeof(dbg_t) == 432u) ? 1 : -1];

/* ==================== 函数原型 ==================== */
/* ① 初始化 (顺序不能反: 先配时钟, 再配时基 —— 时基的计数值按 168MHz 算) */
void     sys_clock_init(void);      //系统时钟 → 168MHz
void     sys_tick_init(void);       //毫秒时基 → 每 1ms g_ms++

/* ② 时基 */
uint32_t sys_ms(void);              //当前毫秒数(从开机累加)
uint8_t  sys_elapsed(uint32_t from_ms, uint32_t span_ms);   //从 from_ms 起已过 span_ms 毫秒? (溢出安全)
void     sys_tick_isr(void);        //SysTick 中断入口, 由 cat_it.c 转发过来

#endif
