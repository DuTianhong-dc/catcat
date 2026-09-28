/* ============================================================
 *  cat_safety.c — 安全闸（纯逻辑, 不碰外设/别的模块）
 *  分层(按此顺序排列, 新函数插对应区块):
 *    ① 目标钳位   safety_clamp_counts()
 *    ② 反馈保护   safety_protect_poll() / safety_get_trip()
 *  依据/来源 → 待写 安全说明.md
 * ============================================================ */
#include "cat_safety.h"
#include <stdlib.h>         /* abs() */

/* ==================== ① 目标钳位 ==================== */
/* 12 关节"机械行程 → 舵机计数"的安全范围（顺序 = 观测序: 后右|后左|前右|前左）。
 * 生成: tools/gen_safety_limits.py（robot.xml 的 joint range + cat_obs.c 的 CAL_SERVO_ZERO,
 *       与 obs_act_to_counts 同一换算公式; 边界向内取整）。
 * ⚠️ 重标定(CAL_SERVO_ZERO)或改模型行程后 → 必须重跑脚本更新下表! */
static const uint16_t SAFETY_GOAL_MIN[SERVO_COUNT] = {
    1005, 1858, 1342,   /* 后右: 外展 / 髋 / 膝 */
    1721, 1213,  637,   /* 后左 */
    1720, 1582, 1323,   /* 前右 */
    1752, 1075,  934    /* 前左 */
};
static const uint16_t SAFETY_GOAL_MAX[SERVO_COUNT] = {
    2308, 2978, 3388,   /* 后右 */
    3024, 2333, 2683,   /* 后左 */
    2436, 2689, 3369,   /* 前右 */
    2468, 2182, 2980    /* 前左 */
};

/* 目标钳位: 把 12 颗目标计数逐颗夹进 [SAFETY_GOAL_MIN, SAFETY_GOAL_MAX]（就地修改）。
 * 返回被钳掉的颗数（0 = 全在安全范围内）—— 供上层当"策略在顶机械限位"的指标看。
 * 用法(编排层, main.c): obs_act_to_counts() → safety_clamp_counts() → servo_sync_write()。
 * ⚠️ 只钳"位置": 不拒发、不限速; "连续大量越界 → 故障"留给以后的 FSM, 别塞进来。 */
uint8_t safety_clamp_counts(uint16_t goal_counts[SERVO_COUNT]){
    uint8_t clipped = 0u;

    for (uint8_t i = 0; i < SERVO_COUNT; i++){
        if (goal_counts[i] < SAFETY_GOAL_MIN[i]){
            goal_counts[i] = SAFETY_GOAL_MIN[i];
            clipped++;
        } else if (goal_counts[i] > SAFETY_GOAL_MAX[i]){
            goal_counts[i] = SAFETY_GOAL_MAX[i];
            clipped++;
        }
    }
    return clipped;
}

/* ==================== ② 反馈保护 ==================== */
/* 判定规则(阈值宏 → cat_safety.h; 依据/换算 → 待写 安全说明.md):
 * 任一命中 → 锁存(返回值 = 首次触发原因), "卸力"由编排层(main.c)做 —— 本模块不碰总线。
 * ⚠️ 只在"同步读齐全"的拍里调; 读数中断的拍不喂数据, 统计保持不动(不误清零/不误判)。
 * ⚠️ 保护不自动恢复: 谁触发谁记录, 重上电或人工处理后再议。
 * ⚠️ 窗口必须"滚动": 若按"每 100 拍清零重数", 恰跨窗口边界的 2 秒满过载(如 50 拍+50 拍)
 *    两段都到不了 80 → 一次都不触发(实测推演约六成起点会漏)。环形滚动没有这个问题。 */

static safety_trip_t s_trip = SAFETY_TRIP_NONE;             /* 锁存: 触发后不自动清 */
static uint8_t s_curr_ring[SERVO_COUNT][SAFETY_CURR_WIN];   /* 最近 100 拍: 每拍记 0/1 */
static uint8_t s_curr_sum [SERVO_COUNT];                    /* 窗口内过载拍数(滚动和)  */
static uint8_t s_ring_pos;                                  /* 环形写指针(12 颗共用)   */

safety_trip_t safety_protect_poll(const uint8_t volt_lsb[SERVO_COUNT],
                                  const uint8_t temp_c[SERVO_COUNT],
                                  const int16_t cur_lsb[SERVO_COUNT]){
    if (s_trip != SAFETY_TRIP_NONE) return s_trip;    /* 已锁存, 直接报 */

    /* 环形先推一格: 本拍要"换出"的就是这一格里最老的那一拍 */
    s_ring_pos = (uint8_t)((s_ring_pos + 1u) % SAFETY_CURR_WIN);

    for (uint8_t i = 0; i < SERVO_COUNT; i++){
        uint8_t over;

        if (temp_c[i] >= SAFETY_TEMP_MAX_C){
            s_trip = SAFETY_TRIP_TEMP;
            break;
        }
        if (volt_lsb[i] < SAFETY_VOLT_MIN_LSB){
            s_trip = SAFETY_TRIP_VOLT;
            break;
        }

        /* 电流: 换出最老一拍、记入本拍, 滚动和跟着更新 */
        over = (abs(cur_lsb[i]) >= SAFETY_CURR_LIMIT_LSB) ? 1u : 0u;
        s_curr_sum[i] = (uint8_t)(s_curr_sum[i] - s_curr_ring[i][s_ring_pos] + over);
        s_curr_ring[i][s_ring_pos] = over;

        if (s_curr_sum[i] >= SAFETY_CURR_SUM_LIMIT){
            s_trip = SAFETY_TRIP_CURR;
            break;
        }
    }
    return s_trip;
}

/* 取锁存的故障原因(0 = 正常; 已触发就一直报, 不自动清) —— 供编排层"每拍查" */
safety_trip_t safety_get_trip(void){
    return s_trip;
}
