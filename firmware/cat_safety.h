#ifndef __CAT_SAFETY_H
#define __CAT_SAFETY_H
#include "stm32f4xx.h"
#include "cat_servo.h"      /* SERVO_COUNT */

/* ============================================================
 *  cat_safety — 安全闸（独立模块, 纯逻辑, 不碰外设/别的模块）
 *
 *  分层（新函数插对应区块, 别追加到末尾）:
 *    ① 目标钳位   safety_clamp_counts()
 *    ② 反馈保护   safety_protect_poll()   (电压/电流/温度 → 锁存报故障; 卸力由 main 做)
 *    （后续: IWDG 看门狗 / 配置存取 —— 还没写）
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
 *   温度 过热保护: 单颗 ≥ 70℃                   （0x3F, ℃） */
#define SAFETY_VOLT_MIN_LSB     63u    /* 6.3V = 63 × 0.1V              */
#define SAFETY_CURR_LIMIT_LSB   308    /* 2.0A = 308 × 6.5mA            */
#define SAFETY_CURR_WIN        100u    /* 滚动窗口长度 = 100 拍 = 2s @50Hz */
#define SAFETY_CURR_SUM_LIMIT   80u    /* 窗口内过载 ≥80 拍(占比 80%) → 触发 */
#define SAFETY_TEMP_MAX_C       70u    /* 70℃                          */

/* 触发原因（锁存, 0 = 正常）。任一命中后不自动恢复 —— 重上电 / 人工处理 */
typedef enum {
    SAFETY_TRIP_NONE = 0,
    SAFETY_TRIP_VOLT = 1,   /* 单颗电压 < 6.3V            */
    SAFETY_TRIP_CURR = 2,   /* 单颗过载: 2s 窗口占比 ≥80% */
    SAFETY_TRIP_TEMP = 3    /* 单颗温度 ≥ 70℃             */
} safety_trip_t;

safety_trip_t safety_protect_poll(const uint8_t volt_lsb[SERVO_COUNT],
                                  const uint8_t temp_c[SERVO_COUNT],
                                  const int16_t cur_lsb[SERVO_COUNT]);

/* 取锁存的故障原因(0 = 正常; 已触发就一直报, 不自动清) —— 主循环每拍查用 */
safety_trip_t safety_get_trip(void);

#endif
