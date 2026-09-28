/* ============================================================
 *  cat_vel_est.c — 机体线速度估计器(支撑腿运动学里程计)
 *
 *  分层(本文件按此顺序排列, 新函数插到对应区块, 别追加到末尾):
 *    ① 几何常量   CH / FOOT_OFF / POLY5
 *    ② 判据参数   VE_*
 *    ③ 内部状态
 *    ④ 四元数工具 quat_to_R / qmul / qrot / qrot_z   (内部)
 *    ⑤ 单腿 FK    poly5 / fk_foot                     (内部)
 *    ⑥ 对外接口   vel_est_init / vel_est_tick
 *
 *  原理与依据 → cat_vel_est.h 的文件头, 这里不重复。
 *  几何数据来自 cat3.4 模型 + SolidWorks 实测, 与 Python 验证件
 *  (verify_fk_4legs.py / odometry_geo.py)逐字节一致 —— 勿单方面修改。
 *  ⚠️ 第一版另有 CAT_SIL_DIAG 宏下的 stderr 诊断输出(供 SIL 对拍), 本版没搬;
 *     将来做 SIL 时回第一版 cat_vel_est.c 取。
 * ============================================================ */
#include "cat_vel_est.h"
#include "cat_servo.h"
#include <string.h>
#include <math.h>

/* ==================== ① 几何常量 ==================== */
typedef struct { float pos[3]; float quat[4]; } chain_step_t;

/* ---- 让表能读的宏 ---- */
/* 三维向量(位置/偏移), 单位 m */
#define VEC3(x, y, z)                      { (x), (y), (z) }

/* 足端链的【一级】= 本级相对上一级的位置(m) + 固定旋转四元数(w,x,y,z)
 *   位置: 上一级坐标系原点 → 本级坐标系原点的位移, 在【上一级】坐标系里表示
 *   旋转: 本级相对上一级的【固定】朝向 —— 不含关节角(关节角在 FK 里单独绕本级 Z 轴施加) */
#define STEP(px, py, pz, qw, qx, qy, qz)   { { (px), (py), (pz) }, \
                                             { (qw), (qx), (qy), (qz) } }

/* 四连杆四阶多项式的 5 个系数 c0..c4 */
#define POLY5F(c0, c1, c2, c3, c4)         { (c0), (c1), (c2), (c3), (c4) }

/* ---- 腿索引(与观测序号 / 舵机索引一致, 每条腿占 3 个) ---- */
#define LEG_RR  0u      /* 右后 */
#define LEG_RL  1u      /* 左后 */
#define LEG_FR  2u      /* 右前 */
#define LEG_FL  3u      /* 左前 */

/* 足端链 3 级: 髋侧摆 → 髋俯仰 → 小腿(生理膝)
 * 【不经过 Link3(曲柄)/Link4(连杆)】—— 小腿挂在髋俯仰下 */
static const chain_step_t CH[4][3] = {
    [LEG_RR] = {                                   /* 右后: Link1 -> Link2 -> Link5 */
        STEP(-0.16061f, -0.062655f,  -0.0018124f,   0.499998f, -0.5f,      0.500002f, -0.5f),
        STEP( 0.029007f, -0.0024191f, -0.02249f,    0.499323f, -0.499325f, -0.500676f, 0.500675f),
        STEP( 0.075f,     0.0f,      -0.015f,       1.0f,       0.0f,      0.0f,       0.0f)
    },
    [LEG_RL] = {                                   /* 左后: Link6 -> Link7 -> Link10 (y 镜像) */
        STEP(-0.16061f,  0.062302f,  -0.0015924f,   0.499998f,  0.5f,      0.500002f,  0.5f),
        STEP( 0.029193f,  0.0026998f, -0.02249f,    0.500673f,  0.500675f, -0.499326f, -0.499325f),
        STEP( 0.075f,     0.0f,      -0.0148f,      1.0f,       0.0f,      0.0f,       0.0f)
    },
    [LEG_FR] = {                                   /* 右前: Link11 -> Link12 -> Link15 */
        STEP( 0.16939f, -0.057655f,   0.023188f,    0.499998f, -0.5f,      0.500002f, -0.5f),
        STEP( 0.029003f, -0.0024677f, -0.02249f,    0.499723f, -0.499725f, -0.500277f, 0.500275f),
        STEP( 0.07f,      0.0f,      -0.015f,       1.0f,       0.0f,      0.0f,       0.0f)
    },
    [LEG_FL] = {                                   /* 左前: Link16 -> Link17 -> Link20 (y 镜像) */
        STEP( 0.16939f,  0.057302f,   0.023188f,    0.499998f,  0.5f,      0.500002f,  0.5f),
        STEP( 0.029197f,  0.0026509f, -0.02249f,    0.500273f,  0.500275f, -0.499727f, -0.499725f),
        STEP( 0.07f,      0.0f,      -0.0148f,      1.0f,       0.0f,      0.0f,       0.0f)
    },
};

/* 足端相对【生理膝】的偏移(m), 在生理膝局部系里表示。SolidWorks 实测 + 三重验证。
 * y 负 = 右侧, y 正 = 左侧(与 CH[] 里 pos.y 的符号规律一致)。
 * 2026-09-13 cat3.4: 后腿改用前小腿零件 → 后足偏移改为与【前腿一致】(腿长 194→181mm),
 * 所以下面 4 行数值相同; 原后腿值是 0.12917 / ∓0.04996 / 0.016。 */
static const float FOOT_OFF[4][3] = {
    [LEG_RR] = VEC3(0.11578f, -0.04504f, 0.0162f),
    [LEG_RL] = VEC3(0.11578f,  0.04504f, 0.0162f),
    [LEG_FR] = VEC3(0.11578f, -0.04504f, 0.0162f),
    [LEG_FL] = VEC3(0.11578f,  0.04504f, 0.0162f),
};

/* 四连杆: 曲柄角 q3 → 生理膝角 q5, 四阶多项式 q5 = c0 + c1*q3 + c2*q3^2 + c3*q3^3 + c4*q3^4
 * 来自 linkage_joint.xml 的 <equality> polycoef(官方生成, 勿手改)。 */
static const float POLY5[4][5] = {
    [LEG_RR] = POLY5F(-0.003411754794f, -0.2084629358f, -0.287239521f,  0.1602706708f, -0.06433772677f),
    [LEG_RL] = POLY5F( 0.003413645749f, -0.2084495883f,  0.2872821059f, 0.1603062374f,  0.06434638416f),
    [LEG_FR] = POLY5F(-0.0006214237093f, -0.2464093902f, -0.2048239656f, 0.1066213511f, -0.06579301798f),
    [LEG_FL] = POLY5F( 0.0006178764537f, -0.2464173728f,  0.2047899822f, 0.1065940917f,  0.06578540507f),
};

/* ==================== ② 判据参数 ====================
 * 2026-09-05 重写判据。原实现拿 |pdot|(机体系差分速度, 实测 0.6~0.9 m/s)
 * 当"足端世界速度"用 → σ=0.3 下权重全灭(等效支撑腿只剩 0.02/4)。两者并不相等:
 *   |pdot| = |牵连速度 + 足端世界速度|, 而牵连速度本身就有 0.5 m/s 量级。
 * 新判据(全在机体系算, 不需要全局定位):
 *   1) 4 条腿各报 v_i = -(pdot_i + w×p_i), 取【分量中位数】当初值(天然鲁棒)
 *   2) 残差 r_i = |v_i − 上一拍估计| → 高斯软权重; 支撑腿残差小, 摆动腿残差大自动降权 */
#define VE_RES_SIGMA  0.30f    /* 残差软权重尺度(m/s): 摆动腿与支撑的 v 差≈0.5, σ 需要偏小 */
#define VE_MAX_RISE   0.02f    /* 抬高粗筛窗口(m, 重力对齐): 抬过这个高度就不算支撑腿 */
#define VE_MIN_W_SUM  0.30f    /* 等效支撑腿权重和低于此 → 退化为沿用上一拍 */

/* ==================== ③ 内部状态 ==================== */
static float   prev_p[4][3];    /* 上一拍足端位置(机体系) */
static float   stance_z[4];     /* 首拍建立的足端世界高度基准(抬高粗筛用) */
static float   v_prev[3];       /* 上一拍速度估计(残差判据的参照) */
static uint8_t have_prev;       /* 有没有上一拍: 第一拍只建基准, 输出 0 */

/* ==================== ④ 四元数工具(q = w,x,y,z) ==================== */
static void quat_to_R(const float q[4], float R[3][3]){
    float w = q[0], x = q[1], y = q[2], z = q[3];
    R[0][0] = 1.f - 2.f*(y*y + z*z);  R[0][1] = 2.f*(x*y - w*z);      R[0][2] = 2.f*(x*z + w*y);
    R[1][0] = 2.f*(x*y + w*z);        R[1][1] = 1.f - 2.f*(x*x + z*z); R[1][2] = 2.f*(y*z - w*x);
    R[2][0] = 2.f*(x*z - w*y);        R[2][1] = 2.f*(y*z + w*x);      R[2][2] = 1.f - 2.f*(x*x + y*y);
}

static void qmul(const float a[4], const float b[4], float o[4]){
    float w1 = a[0], x1 = a[1], y1 = a[2], z1 = a[3];
    float w2 = b[0], x2 = b[1], y2 = b[2], z2 = b[3];
    o[0] = w1*w2 - x1*x2 - y1*y2 - z1*z2;
    o[1] = w1*x2 + x1*w2 + y1*z2 - z1*y2;
    o[2] = w1*y2 - x1*z2 + y1*w2 + z1*x2;
    o[3] = w1*z2 + x1*y2 - y1*x2 + z1*w2;
}

/* 用四元数旋转向量: v' = v + 2*(w*(qv×v) + qv×(qv×v))
 * ⚠️ 2026-09-05 修过一个公式错(原写成 qv×t1 + w*v, 把 w 乘到了 v 上):
 *    关节角=0 时四元数是单位四元数, 错公式恰好无影响 —— 之前的初始态验证是【假阴性】。 */
static void qrot(const float q[4], const float v[3], float o[3]){
    float qv[3] = {q[1], q[2], q[3]};
    float t1[3], t2[3];
    t1[0] = qv[1]*v[2] - qv[2]*v[1];                   /* t1 = qv × v */
    t1[1] = qv[2]*v[0] - qv[0]*v[2];
    t1[2] = qv[0]*v[1] - qv[1]*v[0];
    t2[0] = qv[1]*t1[2] - qv[2]*t1[1] + q[0]*t1[0];   /* t2 = qv × t1 + w*t1 */
    t2[1] = qv[2]*t1[0] - qv[0]*t1[2] + q[0]*t1[1];
    t2[2] = qv[0]*t1[1] - qv[1]*t1[0] + q[0]*t1[2];
    o[0] = v[0] + 2.f*t2[0];
    o[1] = v[1] + 2.f*t2[1];
    o[2] = v[2] + 2.f*t2[2];
}

/* 绕 Z 轴转 a 弧度 → 四元数 */
static void qrot_z(float a, float o[4]){
    o[0] = cosf(a * 0.5f); o[1] = 0.f; o[2] = 0.f; o[3] = sinf(a * 0.5f);
}

/* ==================== ⑤ 单腿 FK ==================== */
/* 四连杆四阶多项式: 曲柄角 → 生理膝角 */
static float poly5(uint8_t leg, float q3){
    const float *c = POLY5[leg];
    float q32 = q3 * q3;
    return c[0] + c[1]*q3 + c[2]*q32 + c[3]*q32*q3 + c[4]*q32*q32;
}

/* 输出足端在【机体系(base_link frame)】的位置。
 * ang[0]=髋侧摆 ang[1]=髋俯仰 ang[2]=生理膝(已由 poly5 从曲柄角换算) */
static void fk_foot(const chain_step_t ch[3], const float ang[3], const float off[3], float p[3]){
    float P[3] = {0.f, 0.f, 0.f};
    float Q[4] = {1.f, 0.f, 0.f, 0.f};
    float Qn[4], R[3][3], t[4];    /* ⚠️ t 必须 4 元素: qrot_z 会写 o[0..3], 用 [3] 是栈越界 */

    for (int i = 0; i < 3; i++){
        quat_to_R(Q, R);                                   /* p += R(Q) * pos (用父级朝向) */
        for (int r = 0; r < 3; r++)
            P[r] += R[r][0]*ch[i].pos[0] + R[r][1]*ch[i].pos[1] + R[r][2]*ch[i].pos[2];
        qmul(Q, ch[i].quat, Qn);                           /* 再累乘本级的固定旋转 */
        Q[0] = Qn[0]; Q[1] = Qn[1]; Q[2] = Qn[2]; Q[3] = Qn[3];
        qrot_z(ang[i], t);                                 /* 本级关节绕自身 Z 轴转 */
        qmul(Q, t, Qn);
        Q[0] = Qn[0]; Q[1] = Qn[1]; Q[2] = Qn[2]; Q[3] = Qn[3];
    }
    qrot(Q, off, t);                                       /* 加上 foot_off(末端局部系) */
    p[0] = P[0] + t[0]; p[1] = P[1] + t[1]; p[2] = P[2] + t[2];
}

/* ==================== ⑥ 对外接口 ==================== */
void vel_est_init(void){
    have_prev = 0u;
    memset(v_prev, 0, sizeof(v_prev));
    memset(prev_p, 0, sizeof(prev_p));
    memset(stance_z, 0, sizeof(stance_z));
}

void vel_est_tick(const float jpos[SERVO_COUNT], const float gyro_rps[3],
                  const float quat_wxyz[4], float dt_s, float v_out[3]){
    float p[4][3];

    if (dt_s < 1e-4f) dt_s = 0.02f;

    /* --- 1) 每腿 FK: 观测序号 3*leg +0/+1/+2 = 髋侧摆/髋俯仰/膝曲柄 --- */
    for (uint8_t leg = 0; leg < 4u; leg++){
        float q3 = jpos[3u*leg + 2u];                      /* 曲柄角 */
        float ang[3];
        ang[0] = jpos[3u*leg + 0u];
        ang[1] = jpos[3u*leg + 1u];
        ang[2] = poly5(leg, q3);                           /* 生理膝角由多项式换算 */
        fk_foot(CH[leg], ang, FOOT_OFF[leg], p[leg]);
    }

    /* --- 首拍: 建立站立基准, 输出零 --- */
    if (!have_prev){
        for (uint8_t leg = 0; leg < 4u; leg++){
            float pw[3];
            qrot(quat_wxyz, p[leg], pw);                   /* 足端在世界系的方向 */
            memcpy(prev_p[leg], p[leg], sizeof(prev_p[leg]));
            stance_z[leg] = pw[2];
        }
        have_prev = 1u;
        v_out[0] = v_out[1] = v_out[2] = 0.f;
        return;
    }

    /* --- 2) 每腿速度方程: v_i = -(pdot_i + ω×p_i) --- */
    float v_each[4][3];
    for (uint8_t leg = 0; leg < 4u; leg++){
        float pdot[3];
        for (int r = 0; r < 3; r++)
            pdot[r] = (p[leg][r] - prev_p[leg][r]) / dt_s;
        float cx = gyro_rps[1]*p[leg][2] - gyro_rps[2]*p[leg][1];   /* ω × p */
        float cy = gyro_rps[2]*p[leg][0] - gyro_rps[0]*p[leg][2];
        float cz = gyro_rps[0]*p[leg][1] - gyro_rps[1]*p[leg][0];
        v_each[leg][0] = -(pdot[0] + cx);
        v_each[leg][1] = -(pdot[1] + cy);
        v_each[leg][2] = -(pdot[2] + cz);
    }

    /* --- 3) 分量中位数(4 个数排序后取中间两数均值): 抑制单腿异常 --- */
    float med[3];
    for (int a = 0; a < 3; a++){
        float s[4];
        for (uint8_t leg = 0; leg < 4u; leg++) s[leg] = v_each[leg][a];
        for (int i = 1; i < 4; i++){                       /* 插入排序(只有 4 个元素) */
            float key = s[i];
            int j = i - 1;
            while (j >= 0 && s[j] > key){ s[j + 1] = s[j]; j--; }
            s[j + 1] = key;
        }
        med[a] = 0.5f * (s[1] + s[2]);
    }

    /* --- 4) 残差软权重 + 抬高粗筛 --- */
    float v[3];
    float w[4];
    float sum_w = 0.f;
    if (v_prev[0] == 0.f && v_prev[1] == 0.f){
        memcpy(v, med, sizeof(v));                         /* 冷启动(尚无历史估计): 直接用中位数 */
        memcpy(v_prev, med, sizeof(med));
    } else {
        for (uint8_t leg = 0; leg < 4u; leg++){
            float dx = v_each[leg][0] - v_prev[0];
            float dy = v_each[leg][1] - v_prev[1];
            float dz = v_each[leg][2] - v_prev[2];
            float rr = sqrtf(dx*dx + dy*dy + dz*dz);
            float wr = expf(-(rr / VE_RES_SIGMA) * (rr / VE_RES_SIGMA));
            float pw[3];
            qrot(quat_wxyz, p[leg], pw);                   /* 足端世界方向 → 判抬高 */
            float rise = pw[2] - stance_z[leg];
            w[leg] = wr * ((rise < VE_MAX_RISE) ? 1.f : 0.f);
            sum_w += w[leg];
        }
        v[0] = v[1] = v[2] = 0.f;
        if (sum_w > VE_MIN_W_SUM){
            for (uint8_t leg = 0; leg < 4u; leg++)
                for (int a = 0; a < 3; a++)
                    v[a] += w[leg] * v_each[leg][a];
            for (int a = 0; a < 3; a++) v[a] /= sum_w;
        } else {
            memcpy(v, v_prev, sizeof(v));                  /* 权重掉光(疑在腾空): 沿用上一拍 */
        }
    }
    v[2] = 0.f;                                            /* 垂直分量单靠腿估不出, 置零 */
    memcpy(v_prev, v, sizeof(v));

    /* --- 5) 状态更新 --- */
    for (uint8_t leg = 0; leg < 4u; leg++)
        memcpy(prev_p[leg], p[leg], sizeof(prev_p[leg]));

    v_out[0] = v[0]; v_out[1] = v[1]; v_out[2] = v[2];
}
