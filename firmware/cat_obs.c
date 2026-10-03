/* ============================================================
 *  cat_obs.c — 59 维观测组装器 (顺序与训练/策略严格一致)
 *
 *  分层(本文件按此顺序排列, 新函数插到对应区块, 别追加到末尾):
 *    ① 初始化     obs_init()
 *    ② 灌数据     obs_set_cmd / obs_set_lastact / obs_set_tau_from_cur
 *    ③ 更新+取数  obs_update / obs_get
 *
 *  ⚠️ 维度/分段/系数/标定表的全部依据与来源 → 观测组装说明.md, 这里不重复。
 * ============================================================ */
#include "cat_obs.h"
#include "cat_config.h"
/* 铁律⑦(模块间不许横向 include)的例外: cat_obs 本身就是聚合层 → 观测组装说明.md(八) */
#include "cat_imu.h"        /* imu_get_gyro / imu_get_quat / imu_get_grav */
#include "cat_vel_est.h"    /* vel_est_tick */
#include "cat_pose.h"       /* CAT_POSE_DEF / CAT_POSE_LIM_* (系统常量, 2026-09-29 迁自本文件) */

/* 默认站姿 / 关节限位(翻身版动作映射用) → 已迁至 cat_pose.c 统一管理,
 * 依据与数值来源 → 观测组装说明.md(七)(十)。 */

/* ---------- 标定表(来历与坑 → 观测组装说明.md(七)) ---------- */
static const uint16_t CAL_SERVO_ZERO[ACT_DIM] = {   /* 关节角 0 对应的舵机计数 */
    1852u, 2640u, 1433u, 2177u, 1551u, 2592u,
    2078u, 2364u, 1414u, 2110u, 1400u, 2889u
};
static const int8_t CAL_SERVO_DIR[ACT_DIM] = {      /* 编码方向 ↔ 模型关节方向, 全 +1 */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

#define OBS_SID_OF(idx)   ((uint8_t)((idx) + 1u))   /* 观测序号 → 总线 ID(恒等加一) */

/* ---------- 组装用的缓存 ---------- */
static float   jpos[ACT_DIM];       //关节角(绝对, rad)
static float   jvel[ACT_DIM];       //关节角速度(rad/s)
static float   tau[ACT_DIM];        //关节力矩(N·m)
static float   lastact[ACT_DIM];    //上一拍策略输出
static float   act_scale[ACT_DIM];  //动作缩放 = max(|lo|,|hi|), obs_init 由限位表算
static float   cmd_vw[OBS_LEN_CMD]; //命令 [vx, wz]
static float   linvel[OBS_LEN_VEC]; //机体系线速度估计(m/s)
static uint8_t have_prev;           //有没有上一拍关节角(差分速度要用)

/* ==================== ① 初始化 ==================== */
void obs_init(void){
    for (uint8_t i = 0; i < ACT_DIM; i++){
        float abs_lo = (CAT_POSE_LIM_LO[i] < 0.0f) ? -CAT_POSE_LIM_LO[i] : CAT_POSE_LIM_LO[i];
        float abs_hi = (CAT_POSE_LIM_HI[i] < 0.0f) ? -CAT_POSE_LIM_HI[i] : CAT_POSE_LIM_HI[i];
        act_scale[i] = (abs_lo > abs_hi) ? abs_lo : abs_hi;

        jpos[i]    = CAT_POSE_DEF[i];   /* 初值 = 站姿 → jpos 段上电输出 0 */
        jvel[i]    = 0.0f;
        tau[i]     = 0.0f;
        lastact[i] = 0.0f;
    }
    for (uint8_t i = 0; i < OBS_LEN_CMD; i++) cmd_vw[i] = 0.0f;
    for (uint8_t i = 0; i < OBS_LEN_VEC; i++) linvel[i] = 0.0f;

    have_prev = 0u;
}

/* ==================== ② 灌数据 ==================== */
/* 命令范围与训练采样一致(±0.5) → 观测组装说明.md(三) */
#define OBS_CMD_VX_MAX   0.5f
#define OBS_CMD_WZ_MAX   0.5f
#define CLAMPF(v, lo, hi)  ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))

/* 灌上位机命令 [vx, wz](无 vy)。钳位不可省 —— 训练只见过 ±0.5。 */
void obs_set_cmd(float vx, float wz){
    cmd_vw[0] = CLAMPF(vx, -OBS_CMD_VX_MAX, OBS_CMD_VX_MAX);
    cmd_vw[1] = CLAMPF(wz, -OBS_CMD_WZ_MAX, OBS_CMD_WZ_MAX);
}

/* 灌【上一拍】策略输出(12 维, 已 tanh)。
 * ⚠️ 必须在策略跑完、动作发出去【之后】调 —— 早了就是未来信息泄漏。 */
void obs_set_lastact(const float act[ACT_DIM]){
    for (uint8_t i = 0; i < ACT_DIM; i++) lastact[i] = act[i];
}

/* 灌 12 颗舵机的电流读数(0x45, 有符号 int16, 单位 LSB)。
 * 换算系数、以及"为什么不加训练侧的 tau_bias" → 观测组装说明.md(四) */
#define OBS_CURR_A_PER_LSB    6.5e-3f   /* A/LSB */
#define OBS_TORQUE_NM_PER_A   0.784f    /* N·m/A (Kt) */

void obs_set_tau_from_cur(const int16_t cur_lsbs[SERVO_COUNT]){
    for (uint8_t i = 0; i < ACT_DIM; i++)
        tau[i] = (float)cur_lsbs[i] * OBS_CURR_A_PER_LSB * OBS_TORQUE_NM_PER_A;
}

/* ==================== ③ 更新 + 取数 ==================== */
/* 计数→弧度、以及速度低通是"固件自研近似"这件事 → 观测组装说明.md(五)(六) */
#define OBS_RAD_PER_COUNT   (6.283185307f / 4096.0f)   /* 精确 2π/4096(2026-10-03 审查 T7 修正;
                                                        * 旧值 6.2832 有 ~7.5ppm 偏差, 与训练侧 2π 口径对齐) */
#define OBS_JVEL_TAU_S      0.006366f

/* 每拍更新观测的运动学部分。
 * bus_counts[12]: 舵机总线原始 12bit 计数; dt_s: 距上拍的秒数(50Hz → 0.02)。
 * ⚠️ bus_counts 必须是【同一时刻】读到的那一组, 别混拍。 */
void obs_update(const uint16_t bus_counts[SERVO_COUNT], float dt_s){
    float gyro[3], quat[4];

    for (uint8_t i = 0; i < ACT_DIM; i++){
        float rad = (float)((int16_t)(bus_counts[i] - CAL_SERVO_ZERO[i]))
                  * OBS_RAD_PER_COUNT * (float)CAL_SERVO_DIR[i];

        if (have_prev && dt_s > 1e-4f){
            float raw_v = (rad - jpos[i]) / dt_s;          /* 位置差分 */
            float a = dt_s / (dt_s + OBS_JVEL_TAU_S);
            jvel[i] += a * (raw_v - jvel[i]);              /* 一阶低通 */
        }
        jpos[i] = rad;
    }
    have_prev = 1u;

    imu_get_gyro(gyro);
    imu_get_quat(quat);
    vel_est_tick(jpos, gyro, quat, dt_s, linvel);
}

/* 取 59 维观测。八段正好铺满, 所以不用先清零。
 * ⚠️ jpos 段输出的是【相对默认站姿的偏移】, 不是绝对角。 */
void obs_get(float out[OBS_DIM]){
    float grav[3];

    for (uint8_t i = 0; i < OBS_LEN_VEC; i++) out[OBS_OFF_LINVEL + i] = linvel[i];
    imu_get_gyro(&out[OBS_OFF_GYRO]);
    imu_get_grav(grav);
    for (uint8_t i = 0; i < OBS_LEN_VEC; i++) out[OBS_OFF_GRAVITY + i] = grav[i];
    for (uint8_t i = 0; i < OBS_LEN_CMD; i++) out[OBS_OFF_CMD + i] = cmd_vw[i];

    for (uint8_t i = 0; i < ACT_DIM; i++){
        out[OBS_OFF_JPOS    + i] = jpos[i] - CAT_POSE_DEF[i];
        out[OBS_OFF_JVEL    + i] = jvel[i];
        out[OBS_OFF_TAU     + i] = tau[i];
        out[OBS_OFF_LASTACT + i] = lastact[i];
    }
}

/* ==================== ④ 反向: 动作 → 舵机目标计数 ==================== */
/* 与 obs_update() 的换算互为逆:
 *   正向(观测)  rad    = (counts − ZERO) × r2r × DIR
 *   反向(发指令) counts = ZERO + q × DIR / r2r      (DIR 只有 ±1, 除它等于乘它)
 * ⚠️ 钳到 [0,4095] —— 12bit 编码的有效范围。新口径下 q 已被限位包住, 正常不触发,
 *    保留作防御(act 越界/NaN); 触发用返回值报出去(观测区 pol_clip), 别静默夹掉。 */
uint8_t obs_act_to_counts(const float act[ACT_DIM], uint16_t out[SERVO_COUNT]){
    float q[ACT_DIM];

    for (uint8_t i = 0; i < ACT_DIM; i++){
        /* 动作 → 目标关节角: 翻身版 = 绝对角, 逐关节按物理限位裁剪(与训练 clip(a×scale) 同序) */
        float v = act[i] * act_scale[i];
        /* ⚠️ NaN 防御(2026-10-03 审查 M1): 下面的比较对 NaN 恒为 false → 钳位全部失效,
         *    q 保持 NaN → (uint16_t)NaN 是未定义行为(ARM 实际给 0), 目标位置直接落 0/4095
         *    极端值 = 12 颗舵机猛冲机械限位。来源: IMU float 帧坏位模式(校验和只管传输错,
         *    不管数值合法性) → 观测 NaN → 网络传播 → 动作 NaN。一行换掉一类机械风险。 */
        if (v != v) v = 0.0f;
        if (v < CAT_POSE_LIM_LO[i]) v = CAT_POSE_LIM_LO[i];
        if (v > CAT_POSE_LIM_HI[i]) v = CAT_POSE_LIM_HI[i];
        q[i] = v;
    }
    return obs_rad_to_counts(q, out);       /* 换算+编码边界钳位只有一份, 见下 */
}

/* 绝对关节角 → 舵机计数。⚠️ 调用方保证 q 已合法(限位内/无 NaN) —— 本函数只做换算,
 * 只对【12bit 编码边界 0/4095】做最后一道防御性钳位(返回钳掉的颗数)。 */
uint8_t obs_rad_to_counts(const float q[ACT_DIM], uint16_t out[SERVO_COUNT]){
    uint8_t clipped = 0u;

    for (uint8_t i = 0; i < ACT_DIM; i++){
        /* 关节角 → 舵机计数(与 obs_update 的正向换算互为逆; 公式见本文件 ③ 段头) */
        float c = (float)CAL_SERVO_ZERO[i]
                + q[i] * (float)CAL_SERVO_DIR[i] / OBS_RAD_PER_COUNT;

        if (c < 0.0f)    { c = 0.0f;    clipped++; }
        if (c > 4095.0f) { c = 4095.0f; clipped++; }
        out[i] = (uint16_t)c;
    }
    return clipped;
}
