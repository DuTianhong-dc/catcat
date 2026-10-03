/* ============================================================
 *  cat_action.c — 动作模块: 加载通道(关键帧表 RAM 化)
 *
 *  分层(按此顺序排列, 新函数插对应区块, 别追加到末尾):
 *    ① 内部工具   clampf_ / le_f32
 *    ② RAM 表状态 表本身 + 会话状态
 *    ③ 加载        action_load_begin / load_kf(内部) / action_load_count / action_ready
 *    ④ 帧处理     action_frame(子类型分派: BEGIN/FRAME/END)
 *
 *  为什么"帧壳复用 0xCC、只加子类型" 与 CSV↔帧的对应关系 → 动作模块设计.md(一)(三)。
 *  ⚠️ 动作表只写 RAM, 不碰 Flash —— 所以【不需要】按住 KEY0 进更新模式(设计文档 4.1)。
 * ============================================================ */
#include "cat_action.h"
#include "cat_pose.h"       /* CAT_POSE_LIM_LO / LIM_HI: 加载时逐关节钳位 */
#include <string.h>         /* memcpy: 4 字节按位拼 float, 避开对齐/别名问题 */

/* ==================== ① 内部工具 ==================== */
static float clampf_(float v, float lo, float hi){
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

/* 小端 4 字节 → float32(与 cat_imu 的帧解析同一套做法: 先拼整数再按位拷) */
static float le_f32(const uint8_t *p){
    uint32_t raw = (uint32_t)p[0] | ((uint32_t)p[1] << 8)
                 | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    float v;
    memcpy(&v, &raw, sizeof(v));
    return v;
}

/* ==================== ② RAM 表状态 ==================== */
/* 一帧关键帧 = 一行 CSV(重排后的帧内顺序: j0..j11, seg, hold, torque, speed, acc) */
typedef struct {
    float    pose[ACT_DIM];   /* 绝对关节角(rad) —— 加载时已钳进限位, 播放器可直接用 */
    float    seg_s;           /* 插值到本帧的时长 */
    float    hold_s;          /* 到位后保持时长 */
    uint16_t torque;          /* 0x30 转矩限制: 0~1000(=100%堵转, 即不限) */
    uint16_t speed;           /* 0x2E 运行速度: 0=最大 */
    uint8_t  acc;             /* 0x29 加速度:  0=最大 */
} act_kf_t;

static act_kf_t s_kf[ACT_RAM_MAX];
static uint8_t  s_n;        /* 已收帧数 */
static uint8_t  s_open;     /* 1 = 已 BEGIN, 会话开着(没收 END) */
static uint8_t  s_locked;   /* 1 = 已 END, 表可用 */
static uint8_t  s_nf;       /* 本次会话每帧 float 个数(14/15/17): 首帧定死, 后续必须一致 */
static uint8_t  s_play_req; /* 1 = END 成功, 请求播放(编排层取走; 定义放这里: BEGIN/END 都要碰它) */

/* ==================== ③ 加载 ==================== */
void action_load_begin(void){
    s_n      = 0u;
    s_open   = 1u;
    s_locked = 0u;      /* 旧表立即失效(与 v1 一致: 加载失败就把旧表丢了, 重新发即可) */
    s_nf     = 0u;
    s_play_req = 0u;    /* 顺手清掉上一轮没被受理的播放请求(防"旧请求 + 新表"错配) */
}

uint8_t action_load_count(void){ return s_n; }
uint8_t action_ready(void){ return (uint8_t)(s_locked && s_n != 0u); }

uint8_t action_kf_pose(uint8_t idx, float out[ACT_DIM]){
    if (idx >= s_n) return 1u;
    for (uint8_t i = 0; i < ACT_DIM; i++) out[i] = s_kf[idx].pose[i];
    return 0u;
}

/* 收一帧(纯逻辑, 便于 SIL/PC 侧对拍): v = 各量, nf = 个数(14/15/17)。
 * 顺序 = j0..j11, seg, hold, [torque], [speed], [acc](与 0xCC FRAME 载荷一致)。
 * 校验不过【整帧拒收】(不算进表); 关节角越限则钳位不拒帧(外部数据不可信, 钳住更安全)。 */
static uint8_t action_load_kf(const float *v, uint8_t nf){
    float seg, hold, torque = 1000.0f, speed = 0.0f, acc = 0.0f;

    if (s_n >= ACT_RAM_MAX) return ACT_RC_FULL;
    if (s_nf == 0u) s_nf = nf;                  /* 首帧定版本 */
    else if (s_nf != nf) return ACT_RC_LEN;     /* 会话内列数必须一致(同一份 CSV) */

    for (uint8_t i = 0; i < ACT_DIM; i++){
        float q = v[i];
        if (q != q) return ACT_RC_NAN;          /* NaN 拒帧: 别让它一路传进舵机目标 */
        s_kf[s_n].pose[i] = clampf_(q, CAT_POSE_LIM_LO[i], CAT_POSE_LIM_HI[i]);
    }
    seg  = v[12];
    hold = v[13];
    /* ⚠️ 用【取反包含】的写法: NaN 也让条件成立 → 天然拒收(与 v1 一致) */
    if (!(seg  >= ACT_SEG_MIN  && seg  <= ACT_SEG_MAX))  return ACT_RC_SEG;
    if (!(hold >= 0.0f         && hold <= ACT_HOLD_MAX)) return ACT_RC_HOLD;

    /* 可选运动参数: 缺列 → 默认; NaN → 默认; 超范围 → 钳位不拒帧(v1 v3 约定) */
    if (nf >= 15u) { torque = v[14]; if (torque != torque) torque = 1000.0f; }
    if (nf >= 16u) { speed  = v[15]; if (speed  != speed)  speed  = 0.0f;    }
    if (nf >= 17u) { acc    = v[16]; if (acc    != acc)    acc    = 0.0f;    }

    s_kf[s_n].seg_s  = seg;
    s_kf[s_n].hold_s = hold;
    s_kf[s_n].torque = (uint16_t)clampf_(torque, 0.0f, 1000.0f);
    s_kf[s_n].speed  = (uint16_t)clampf_(speed,  0.0f, 3400.0f);
    s_kf[s_n].acc    = (uint8_t) clampf_(acc,    0.0f,  254.0f);
    s_n++;
    return ACT_RC_OK;
}

/* ==================== ④ 帧处理 ==================== */
/* 子类型分派 + 应答组包。与 cat_polupd_frame() 同构: main 一处分派, 两个模块各自管自己的号段。
 * ⚠️ 应答载荷 = rc + 帧数(2B), 帧数让 PC 逐帧核对"CSV 行 ↔ 实收帧"(设计文档 §3)。 */
uint8_t action_frame(const uint8_t *d, uint16_t n, uint8_t gate_ok,
                     uint8_t resp[48], uint8_t *resp_len){
    uint8_t rc = ACT_RC_LEN;

    if (n == 0u){
        rc = ACT_RC_LEN;
    } else {
        switch (d[0]) {
        case ACT_SUB_BEGIN:
            /* 门卫在 BEGIN 卡(2026-10-03 定): 未使能 + 无故障才允许换表 —— 站着不动才能换,
             * 免得表换到一半/换完立刻从行走姿态切到首帧(危险跳变)。门卫由编排层算好传进来。 */
            if (!gate_ok)      rc = ACT_RC_GATE;
            else if (n != 1u)  rc = ACT_RC_LEN;
            else { action_load_begin(); rc = ACT_RC_OK; }
            break;

        case ACT_SUB_FRAME: {
            uint16_t bn = (uint16_t)(n - 1u);           /* 载荷字节数 */
            float    v[ACT_FLOAT_MAX];
            uint8_t  nf, i;

            if (!s_open)                                     { rc = ACT_RC_ORDER; break; }
            if (bn == 0u || (bn & 3u) || bn > ACT_FLOAT_MAX * 4u){ rc = ACT_RC_LEN; break; }
            nf = (uint8_t)(bn / 4u);
            /* 只认 v1/v2/v3 三种帧长(14/15/17 个 float) —— 对应 CSV 的列数形态 */
            if (nf != 14u && nf != 15u && nf != 17u)         { rc = ACT_RC_LEN; break; }
            for (i = 0; i < nf; i++) v[i] = le_f32(d + 1u + (uint16_t)i * 4u);
            rc = action_load_kf(v, nf);
            break;
        }

        case ACT_SUB_END:
            if (n != 1u)       { rc = ACT_RC_LEN;   break; }
            if (!s_open)       { rc = ACT_RC_ORDER; break; }
            /* 中途使能/出故障 → 弃表(本次加载作废), 不给"半路换表后立刻播"的机会 */
            if (!gate_ok)      { action_load_begin(); s_open = 0u; s_locked = 0u;
                                 rc = ACT_RC_GATE; break; }
            if (s_n == 0u)     { s_open = 0u; rc = ACT_RC_ORDER; break; }   /* 空表不锁定 */
            s_locked   = 1u;   /* 锁定 = 可播 */
            s_open     = 0u;
            s_play_req = 1u;   /* END 即请求播放(2026-10-03 定, 方案 A); 是否真播由编排层门控 */
            rc = ACT_RC_OK;
            break;

        default:               rc = ACT_RC_LEN; break;      /* 未知子类型: 不理 */
        }
    }

    resp[0]     = rc;
    resp[1]     = s_n;                      /* 当前表内帧数(PC 用它逐帧核对) */
    *resp_len   = 2u;
    return ACT_UP_ACK;
}

/* ==================== ⑤ 播放器 ====================
 * v1 CAT_APP/cat_action.c 的移植: 进场 → 逐帧 smoothstep + 保持 → 出场;
 * 与 v1 的三点差异(适配 v2 无 FSM):
 *   ① 输出 = 绝对关节角(rad) —— 换算成舵机计数/写总线归编排层(cat_action 不碰总线, 守分层)
 *   ② 受理改为"门卫 + 取走即丢弃"(v1 是 FSM 在 IDLE 态受理), 见 action_play_start_if_requested
 *   ③ abort 改为"跳到出场段"(v1 是硬停 + 让调用方用 last_pose 自己平滑) —— v2 少一个接口少一处踩坑 */
#define ACT_ENTER_S   0.8f      /* 进场时长: 当前角 → 首帧 */
#define ACT_EXIT_S    0.8f      /* 出场时长: 末帧(或中断姿态) → 默认站姿 */
#define ACT_MAX_RATE  5.0f      /* 限速(rad/s): v1 v3 从 3 提到 5 —— 3(=172°/s) 掐掉舵机一半能力 */
#define ACT_DT_S      0.02f     /* 播放器节拍 = 主循环 50Hz */
#define ACT_TQ_DEF    1000u     /* 默认转矩档 (=100%, 不限流) */
#define ACT_SPD_DEF   0u        /* 默认速度档 (0 = 最大) */
#define ACT_ACC_DEF   0u        /* 默认加速度档 (0 = 最大) */

static uint8_t  s_playing;
static uint8_t  s_stage, s_idx;
static float    s_t, s_hold_t;
static float    s_from[ACT_DIM], s_to[ACT_DIM], s_cur[ACT_DIM];
static uint16_t s_wr_torque, s_wr_speed;    /* 已上报给编排层(给舵机)的参数, 去重用 */
static uint8_t  s_wr_acc;
static uint8_t  s_param_dirty;              /* 1 = 参数变了, 编排层该写一次舵机参数寄存器 */

static float min1f_(float x){ return (x > 1.0f) ? 1.0f : x; }

/* 平滑插值一拍: tgt = a + (b−a)·smoothstep(u), 再按 ACT_MAX_RATE 限速逼近 + 关节限位截断。
 * ⚠️ 进场段调用方每拍把 a 设成【当前姿态】(见 tick 里的 memcpy(s_from, s_cur)) ——
 *    这是 v1 的有意语义(指数逼近, 起步柔和), 别"优化"成固定起点。 */
static void lerp_tick(float out[ACT_DIM], const float *a, const float *b, float u){
    const float us   = u * u * (3.0f - 2.0f * u);   /* smoothstep */
    const float step = ACT_MAX_RATE * ACT_DT_S;     /* 本拍允许的最大角变化 */

    for (uint8_t i = 0; i < ACT_DIM; i++){
        float tgt = a[i] + (b[i] - a[i]) * us;
        float dq  = tgt - s_cur[i];
        if (dq >  step) dq =  step;
        if (dq < -step) dq = -step;
        s_cur[i] += dq;
        out[i] = clampf_(s_cur[i], CAT_POSE_LIM_LO[i], CAT_POSE_LIM_HI[i]);
    }
}

uint8_t action_playing(void){ return s_playing; }

uint8_t action_play_start_if_requested(const float cur[ACT_DIM], uint8_t gate_ok){
    if (!s_play_req) return 0u;
    s_play_req = 0u;            /* 无论成败都取走 —— 门卫不过就丢弃, 不挂起(免得久后一使能就意外动) */
    if (!gate_ok || !s_locked || s_n == 0u) return 0u;

    for (uint8_t i = 0; i < ACT_DIM; i++) s_cur[i] = cur[i];
    s_stage = ACT_STAGE_ENTER; s_idx = 0u; s_t = 0.f; s_hold_t = 0.f;
    /* 进场用默认档起步, 进首帧后由帧参数覆盖(与 v1 的"哨兵强制写首帧"等价) */
    s_wr_torque = ACT_TQ_DEF; s_wr_speed = ACT_SPD_DEF; s_wr_acc = ACT_ACC_DEF;
    s_param_dirty = 1u;
    s_playing = 1u;
    return 1u;
}

void action_abort(void){
    if (!s_playing) return;
    memcpy(s_from, s_cur, sizeof(s_from));      /* 从"现在在哪"平滑回站姿(不是硬停, 也不是从末帧跳) */
    s_stage = ACT_STAGE_EXIT;
    s_t = 0.f; s_hold_t = 0.f;
    s_wr_torque = ACT_TQ_DEF; s_wr_speed = ACT_SPD_DEF; s_wr_acc = ACT_ACC_DEF;
    s_param_dirty = 1u;
}

uint8_t action_cur_params(uint16_t *torque, uint16_t *speed, uint8_t *acc){
    uint8_t ch = s_param_dirty;
    s_param_dirty = 0u;                         /* 取走即清: 同一参数只让编排层写一次 */
    if (torque) *torque = s_wr_torque;
    if (speed)  *speed  = s_wr_speed;           /* speed 搭位置车, 每拍都要用, 不参与去重 */
    if (acc)    *acc    = s_wr_acc;
    return ch;
}

void action_progress(uint8_t *stage, uint8_t *idx){
    if (stage) *stage = s_playing ? s_stage : ACT_STAGE_IDLE;
    if (idx)   *idx   = s_idx;
}

uint8_t action_tick(float out[ACT_DIM]){
    if (!s_playing) return 1u;
    s_t += ACT_DT_S;

    if (s_stage == ACT_STAGE_ENTER){
        memcpy(s_from, s_cur, sizeof(s_from));          /* 见 lerp_tick 注释: 指数逼近语义 */
        memcpy(s_to,   s_kf[0].pose, sizeof(s_to));
        lerp_tick(out, s_from, s_to, min1f_(s_t / ACT_ENTER_S));
        if (s_t >= ACT_ENTER_S){ s_stage = ACT_STAGE_KF; s_idx = 0u; s_t = 0.f; }
        return 0u;
    }

    if (s_stage == ACT_STAGE_KF){
        memcpy(s_to, s_kf[s_idx].pose, sizeof(s_to));
        /* 帧参数: 与已上报值不同才标 dirty(相邻帧参数一样 → 零总线开销) */
        if (s_kf[s_idx].torque != s_wr_torque || s_kf[s_idx].acc != s_wr_acc){
            s_wr_torque = s_kf[s_idx].torque;
            s_wr_acc    = s_kf[s_idx].acc;
            s_param_dirty = 1u;
        }
        s_wr_speed = s_kf[s_idx].speed;                 /* speed 随位置写, 每帧更新 */
        lerp_tick(out, s_from, s_to, min1f_(s_t / s_kf[s_idx].seg_s));
        if (s_t >= s_kf[s_idx].seg_s){
            s_hold_t += ACT_DT_S;
            if (s_hold_t >= s_kf[s_idx].hold_s){        /* 本帧"插值 + 保持"都走完 → 下一帧 */
                s_hold_t = 0.f;
                memcpy(s_from, s_kf[s_idx].pose, sizeof(s_from));
                s_idx++;
                s_t = 0.f;
                if (s_idx >= s_n) s_stage = ACT_STAGE_EXIT;
            }
        }
        return 0u;
    }

    /* ACT_STAGE_EXIT: 出场 → 默认站姿 */
    memcpy(s_to, CAT_POSE_DEF, sizeof(s_to));
    lerp_tick(out, s_from, s_to, min1f_(s_t / ACT_EXIT_S));
    if (s_t >= ACT_EXIT_S){
        s_playing = 0u; s_stage = ACT_STAGE_IDLE;
        /* 参数回默认档: 不然后续策略会被最后一个动作帧的限流档卡住(v1 踩过) */
        s_wr_torque = ACT_TQ_DEF; s_wr_speed = ACT_SPD_DEF; s_wr_acc = ACT_ACC_DEF;
        s_param_dirty = 1u;
        for (uint8_t i = 0; i < ACT_DIM; i++) out[i] = CAT_POSE_DEF[i];
        return 1u;
    }
    return 0u;
}
