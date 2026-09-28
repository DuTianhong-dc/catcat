#ifndef __CAT_MG90_H
#define __CAT_MG90_H
#include "stm32f4xx.h"

/* ============================================================
 *  MG90S 云台舵机驱动 (TIM4 / PB6-PB7, PWM)
 *  协议与参数依据 通信协议.txt 第六章。
 *
 *  ⚠️ 它跟腿上那 12 颗【不是一类东西】:
 *     腿上 = STS3215 总线舵机(USART6, SCS 协议, 有 ID / 内存表 / 会应答)
 *     云台 = MG90S 模拟舵机(一根线收 PWM, 无 ID / 无反馈 / 不应答)
 *     → 不能用 cat_servo 那套去读写它, 角度只能靠"上次发的什么"回显。
 * ============================================================ */

/* 脉宽(μs): 500 = 0 度, 2500 = 180 度, 1500 = 90 度(回中 / 装配参考位) */
#define MG90_PWM_MIN_US     500u
#define MG90_PWM_MAX_US    2500u
#define MG90_PWM_MID_US    1500u

/* 周期(μs): 50Hz 标准舵机 PWM */
#define MG90_PWM_PERIOD_US 20000u

/* 角度上限 */
#define MG90_DEG_MAX        180.0f

/* ==================== 函数原型 ==================== */
/* ① 初始化 */
void mg90_init(void);                       //配 TIM4 + PB6/PB7(AF2), 上电回中 90 度

/* ② 输出 */
void mg90_set(int8_t pan, int8_t tilt);     //±127 → 0~180 度(0 = 1500μs 中位), 带脉宽硬钳位

/* ③ 取数 */
void mg90_get(int8_t *pan, int8_t *tilt);   //回读最近一次设定的值(模拟舵机没反馈, 只能这样回显)

#endif
