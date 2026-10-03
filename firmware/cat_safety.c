/* ============================================================
 *  cat_safety.c — 安全闸（服务层, 不依赖别的模块）
 *  分层(按此顺序排列, 新函数插对应区块):
 *    ① 目标钳位   safety_clamp_counts()
 *    ② 反馈保护   safety_protect_poll() / safety_get_trip()
 *    ③ IWDG       safety_iwdg_*()
 *  依据/来源 → 待写 安全说明.md
 * ============================================================ */
#include "cat_safety.h"
#include "cat_config.h"     /* CAT_IWDG_ENABLE */
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
 *    两段都到不了 80 → 一次都不触发(实测推演约六成起点会漏)。环形滚动没有这个问题。
 * ⚠️ 电压/温度必须"连续 N 拍"去抖(2026-09-30 加): 实测 ID2 温度寄存器偶发单拍假读数
 *    70℃(其余 11 颗 27~29℃, 下一拍就恢复) → 单拍即锁存 = 误趴。连续 3 拍才采信,
 *    任一干净拍清零重数; 真过热的持续读数不受影响(仅确认延迟 3 拍 = 60ms)。
 * ⚠️ 计数必须【逐颗】独立(2026-10-03 审查 M3): 原先 12 颗共用一个计数器, 实际语义是
 *    "任意颗连续 3 拍" —— 多颗轮流单拍假读(总线干扰 burst)也能凑满 3 拍误触发;
 *    设计意图是"同一颗连续 3 拍", 逐颗计数才配得上这句话。 */

static safety_trip_t s_trip = SAFETY_TRIP_NONE;             /* 锁存: 触发后不自动清 */
static uint8_t s_curr_ring[SERVO_COUNT][SAFETY_CURR_WIN];   /* 最近 100 拍: 每拍记 0/1 */
static uint8_t s_curr_sum [SERVO_COUNT];                    /* 窗口内过载拍数(滚动和)  */
static uint8_t s_ring_pos;                                  /* 环形写指针(12 颗共用)   */
static uint8_t s_temp_hits[SERVO_COUNT];                    /* 温/压去抖: 每颗各自的连续命中拍数 */
static uint8_t s_volt_hits[SERVO_COUNT];

safety_trip_t safety_protect_poll(const uint8_t volt_lsb[SERVO_COUNT],
                                  const uint8_t temp_c[SERVO_COUNT],
                                  const int16_t cur_lsb[SERVO_COUNT]){
    if (s_trip != SAFETY_TRIP_NONE) return s_trip;    /* 已锁存, 直接报 */

    /* 环形先推一格: 本拍要"换出"的就是这一格里最老的那一拍 */
    s_ring_pos = (uint8_t)((s_ring_pos + 1u) % SAFETY_CURR_WIN);

    for (uint8_t i = 0; i < SERVO_COUNT; i++){
        uint8_t over;

        /* 温/压: 逐颗记"连续命中拍数", 干净一拍即清零(见循环后的采信判定) */
        s_temp_hits[i] = (temp_c[i] >= SAFETY_TEMP_MAX_C)
                       ? (uint8_t)(s_temp_hits[i] + 1u) : 0u;
        s_volt_hits[i] = (volt_lsb[i] < SAFETY_VOLT_MIN_LSB)
                       ? (uint8_t)(s_volt_hits[i] + 1u) : 0u;

        /* 电流: 换出最老一拍、记入本拍, 滚动和跟着更新 */
        over = (abs(cur_lsb[i]) >= SAFETY_CURR_LIMIT_LSB) ? 1u : 0u;
        s_curr_sum[i] = (uint8_t)(s_curr_sum[i] - s_curr_ring[i][s_ring_pos] + over);
        s_curr_ring[i][s_ring_pos] = over;

        if (s_curr_sum[i] >= SAFETY_CURR_SUM_LIMIT){
            s_trip = SAFETY_TRIP_CURR;
            break;
        }
    }

    /* 温/压去抖采信(2026-10-03: 逐颗判定)。⚠️ 只在尚未锁存时写 —— 首因优先:
     * 同拍电流已触发就别被后到的温/压覆写, 故障码要指"最先出事的那类"(审查 L3)。 */
    if (s_trip == SAFETY_TRIP_NONE) {
        for (uint8_t i = 0; i < SERVO_COUNT; i++){
            if (s_temp_hits[i] >= SAFETY_DEBOUNCE_N){ s_trip = SAFETY_TRIP_TEMP; break; }
            if (s_volt_hits[i] >= SAFETY_DEBOUNCE_N){ s_trip = SAFETY_TRIP_VOLT; break; }
        }
    }

    return s_trip;
}

/* 取锁存的故障原因(0 = 正常; 已触发就一直报, 不自动清) —— 供编排层"每拍查" */
safety_trip_t safety_get_trip(void){
    return s_trip;
}

/* ==================== ③ IWDG 看门狗 ==================== */
/* 依据 RM0090(IWDG 章) + 第一版已验证实现的写法:
 *   KEYR 三条口令 —— 0x5555 解锁 PR/RLR / 0xAAAA 喂狗 / 0xCCCC 启动;
 *   启动后 LSI 被强制打开(运行期间关不掉) —— 所以不必手动管 LSI;
 *   写 RLR 后要等 SR 的 RVU 同步(几个 LSI 周期)才生效。
 * 约束: 只在【已启动】时才动 RLR/等 RVU —— 没启动时 LSI 没跑, RVU 永不清, 会死等卡死。 */
#define SAFETY_IWDG_KEY_UNLOCK  0x5555u
#define SAFETY_IWDG_KEY_START   0xCCCCu
#define SAFETY_IWDG_KEY_FEED    0xAAAAu
/* 预分频 6 = ÷256 → LSI 32kHz/256 = 125Hz。
 * ⚠️ 为什么不用 ÷64(500Hz): RLR 只有 12 位(≤4095) —— 500Hz 下放宽上限只有 8.19s,
 *    而实测本板擦 2×128KB 扇区 >8s 擦不完; 125Hz 下 RLR=4000 就有 32s, 余量足。 */
#define SAFETY_IWDG_PR_DIV256      6u

static uint8_t s_started;        /* 启动成功后置 1(动 RLR 的前提, 见上面约束) */
static uint8_t s_reset_cause;    /* 本次复位源: 0=上电 1=IWDG */

/* 改重载值(relax/restore 共用): 解锁 → 写 RLR → 等同步 → 喂一次让新值从头计时 */
static void iwdg_set_rlr(uint32_t rlr){
    IWDG->KR  = SAFETY_IWDG_KEY_UNLOCK;
    IWDG->RLR = rlr;
    while (IWDG->SR & IWDG_SR_RVU);         /* 已在跑(LSI 有着落): 几个周期必清, 不会死等 */
    IWDG->KR  = SAFETY_IWDG_KEY_FEED;
}

/* 开机一次: 读复位源 → (按开关)启动看门狗。调用点: main 外设初始化之后、进主循环之前。
 * 之后所有循环(含 boot 等待)都必须喂狗 —— safety_iwdg_feed()。 */
void safety_iwdg_init(void){
    /* 复位源: IWDG(卡死被救) > 软复位(OTA 后自动重启) > 上电/复位键 */
    if      (RCC->CSR & RCC_CSR_IWDGRSTF) s_reset_cause = 1u;
    else if (RCC->CSR & RCC_CSR_SFTRSTF)  s_reset_cause = 2u;
    else                                  s_reset_cause = 0u;
    RCC->CSR |= RCC_CSR_RMVF;               /* 清全部复位标志, 下次复位读的才是"新账" */

    if (!CAT_IWDG_ENABLE) return;           /* 编译开关关 = 开发模式: 不启动 */

    IWDG->KR  = SAFETY_IWDG_KEY_UNLOCK;
    IWDG->PR  = SAFETY_IWDG_PR_DIV256;
    IWDG->RLR = SAFETY_IWDG_RLR_DEFAULT;
    /* 调试冻结: 挂调试器停住(halt)时看门狗暂停计数, 单步/断点不会被误复位(DBGMCU) */
    DBGMCU->APB1FZ |= DBGMCU_APB1_FZ_DBG_IWDG_STOP;
    IWDG->KR  = SAFETY_IWDG_KEY_START;      /* 启动(最后一步); 之后只能靠喂狗活着 */
    s_started = 1u;
}

/* 喂狗: 主循环每拍调一次(20ms << 1s 超时)。未启动时是空操作。 */
void safety_iwdg_feed(void){
    if (s_started) IWDG->KR = SAFETY_IWDG_KEY_FEED;
}

/* 临时放宽(OTA 擦/验 Flash 期间 CPU 停在 Flash 总线上, 本板实测 >8s): 125 → 4000(32s) */
void safety_iwdg_relax(void){
    if (s_started) iwdg_set_rlr(SAFETY_IWDG_RLR_RELAX);
}

/* 恢复 1s 超时(OTA 流程结束/中止时)。 */
void safety_iwdg_restore(void){
    if (s_started) iwdg_set_rlr(SAFETY_IWDG_RLR_DEFAULT);
}

/* 本次复位源: 0=上电(含烧录/复位键) 1=IWDG(卡死被救回) —— 开机读一次, 之后不变 */
uint8_t safety_reset_cause(void){
    return s_reset_cause;
}
