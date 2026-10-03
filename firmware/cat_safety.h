#ifndef __CAT_SAFETY_H
#define __CAT_SAFETY_H
#include "stm32f4xx.h"
#include "cat_servo.h"      /* SERVO_COUNT */

/* ============================================================
 *  cat_safety — 安全闸（服务层, 不依赖别的模块）
 *
 *  分层（新函数插对应区块, 别追加到末尾）:
 *    ① 目标钳位   safety_clamp_counts()
 *    ② 反馈保护   safety_protect_poll()   (电压/电流/温度 → 锁存报故障; 卸力由 main 做)
 *    ③ IWDG       safety_iwdg_*()         (看门狗: 卡死 1s 自动复位; 唯一直接碰的硬件)
 *    （后续: 配置存取 —— 还没写）
 *
 *  钳位表来源与算法 → 待写 安全说明.md; 表由 tools/gen_safety_limits.py 生成。
 * ============================================================ */

/* ① 目标钳位 */
uint8_t safety_clamp_counts(uint16_t goal_counts[SERVO_COUNT]);

/* ② 反馈保护 */
/* 阈值（煜辰 2026-09-28 定；依据 → 待写 安全说明.md）:
 *   电压 低压保护: 单颗 < 6.3V                  （0x3E, 0.1V/LSB  → 63）
 *   电流 持续过载: 单颗 |电流| ≥ 2.0A（0x45, 6.5mA/LSB → 308; ≈堵转 80%）
 *                 且在【滚动 2 秒窗口】内过载占比 ≥80%（=100 拍里 ≥80 拍）→ 触发
 *   温度 过热保护: 单颗 ≥ 70℃                   （0x3F, ℃）
 * ⚠️ 电压/温度 = 「连续 N 拍确认」才锁存（2026-09-30 去抖; 依据见 cat_safety.c 区头） */
#define SAFETY_VOLT_MIN_LSB     63u    /* 6.3V = 63 × 0.1V              */
#define SAFETY_CURR_LIMIT_LSB   308    /* 2.0A = 308 × 6.5mA            */
#define SAFETY_CURR_WIN        100u    /* 滚动窗口长度 = 100 拍 = 2s @50Hz */
#define SAFETY_CURR_SUM_LIMIT   80u    /* 窗口内过载 ≥80 拍(占比 80%) → 触发 */
#define SAFETY_TEMP_MAX_C       70u    /* 70℃                          */
#define SAFETY_DEBOUNCE_N        3u    /* 电压/温度: 连续命中 N 拍才锁存  */

/* 触发原因（锁存, 0 = 正常）。任一命中后不自动恢复 —— 重上电 / 人工处理 */
typedef enum {
    SAFETY_TRIP_NONE = 0,
    SAFETY_TRIP_VOLT = 1,   /* 单颗电压 < 6.3V(连续 3 拍)  */
    SAFETY_TRIP_CURR = 2,   /* 单颗过载: 2s 窗口占比 ≥80% */
    SAFETY_TRIP_TEMP = 3    /* 单颗温度 ≥ 70℃(连续 3 拍)   */
} safety_trip_t;

safety_trip_t safety_protect_poll(const uint8_t volt_lsb[SERVO_COUNT],
                                  const uint8_t temp_c[SERVO_COUNT],
                                  const int16_t cur_lsb[SERVO_COUNT]);

/* 取锁存的故障原因(0 = 正常; 已触发就一直报, 不自动清) —— 主循环每拍查用 */
safety_trip_t safety_get_trip(void);

/* ③ IWDG 看门狗（2026-09-29 加, 煜辰定"一起上"）
 * 1s 超时(LSI≈32kHz ÷256 = 125Hz, RLR=125); 主循环每拍喂一次(20ms, 余量 50 倍)。
 * 卡死(死循环/跑飞)超过 1s → 硬件自动复位。
 * 调试冻结: 挂调试器/断点停住时 IWDG 暂停计数, 不会被误复位(DBGMCU)。
 * OTA 擦 Flash 停摆 >8s(本板实测) → 更新流程调 relax() 临时放宽到 32s, 结束 restore()。
 * 编译开关: cat_config.h 的 CAT_IWDG_ENABLE(0 = 根本不启动)。 */
#define SAFETY_IWDG_RLR_DEFAULT  125u    /* 1.0s @ 125Hz(LSI/256) */
#define SAFETY_IWDG_RLR_RELAX   4000u    /* 32.0s @ 125Hz: OTA 擦除窗口。
                                          * ⚠️ RLR 是 12 位(≤4095)! 拉长只能靠预分频 ——
                                          *    写 15000 会被截断成 2712(=5.4s) 反而更短, 踩过。
                                          *    125Hz(PR=6) 下 4000 = 32s, 覆盖实测 >8s 的擦除。 */

void    safety_iwdg_init(void);          /* 启动看门狗+读复位源(开机一次; LSI 由启动动作自动开) */
void    safety_iwdg_feed(void);          /* 喂狗(主循环每拍调) */
void    safety_iwdg_relax(void);         /* 临时放宽(OTA 擦除前) */
void    safety_iwdg_restore(void);       /* 恢复 1s(OTA 结束) */
uint8_t safety_reset_cause(void);        /* 复位源: 0=上电/复位键 1=IWDG(卡死被救) 2=软复位(OTA后重启) */

#endif
