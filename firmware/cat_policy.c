/* ============================================================
 *  cat_policy.c — int8 策略网络前向 (纯 C, 不依赖 CMSIS-NN)
 *
 *  分层(本文件按此顺序排列, 新函数插到对应区块, 别追加到末尾):
 *    ① 内部算子   dense_s8 / swishf
 *    ② 前向       policy_run()
 *    ③ 溯源       policy_tag / policy_build_id
 *
 *  预算: ~20 万 MAC, F407@168MHz 手写循环预期 3~6ms(20ms 预算内, 充裕)。
 *  升级路径: 换 CMSIS-NN 的 arm_fully_connected_s8 还能更快, 但必须先与训练侧
 *            的 int8 参考实现对拍通过才允许切 —— 见文件末"对拍要求"。
 * ============================================================ */
#include "cat_policy.h"
#include <math.h>

/* ==================== ① 内部算子 ==================== */
/* 按输出通道量化的 int8 全连接层(输入仍是 float)。
 *   公式: out[o] = (Σ_i in[i] × W[i][o]) × ws[o] + b[o]
 * ⚠️ 三个坑全在第一版踩过, 谁改这段都要先看:
 *   ① 权重布局: npz 里 W 是 (in_dim, out_dim)【行主序】——
 *      按 (out_dim, in_dim) 读就是转置乘法, 输出全错(且不报错, 只表现为策略像抽风)。
 *   ② 输入不能用 (int32_t) 截断: 进来的是归一化后的 float(±1 上下),
 *      取整会把小数全变 0 → 策略彻底失效。
 *   ③ bias 只加一次, 且【不乘】ws。第一版曾加了两次、第二次还被 ws 乘。 */
static void dense_s8(const int8_t *w, const float *ws, const float *b,
                     const float *in, uint16_t in_dim, uint16_t out_dim, float *out){
    /* o 按 8 个一组分块: 组内 8 个累加器是局部变量(住 FPU 寄存器), 一组只写一次 out[],
     * 消掉原写法对 out[o] 的"读-改-写"(实测 12 条指令/MAC → 4.4 条/MAC, 省 ~4ms)。
     * ⚠️ 循环变量用 uint32_t: 用 uint16_t 会让编译器每次迭代多插一条 UXTH(截断)。
     * ⚠️ 收益依赖 -O2: -O0 下局部变量落栈, 分块基本白做 —— 别把工程的优化等级调回去。 */
    const uint32_t id = in_dim, od = out_dim;
    uint32_t o = 0;
    for (; o + 8u <= od; o += 8u){
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
        for (uint32_t i = 0; i < id; i++){
            const float  xv  = in[i];
            const int8_t *row = w + i * od + o;                /* W[i][o..o+8) */
            a0 += xv * (float)row[0];
            a1 += xv * (float)row[1];
            a2 += xv * (float)row[2];
            a3 += xv * (float)row[3];
            a4 += xv * (float)row[4];
            a5 += xv * (float)row[5];
            a6 += xv * (float)row[6];
            a7 += xv * (float)row[7];
        }
        out[o    ] = a0 * ws[o    ] + b[o    ];
        out[o + 1] = a1 * ws[o + 1] + b[o + 1];
        out[o + 2] = a2 * ws[o + 2] + b[o + 2];
        out[o + 3] = a3 * ws[o + 3] + b[o + 3];
        out[o + 4] = a4 * ws[o + 4] + b[o + 4];
        out[o + 5] = a5 * ws[o + 5] + b[o + 5];
        out[o + 6] = a6 * ws[o + 6] + b[o + 6];
        out[o + 7] = a7 * ws[o + 7] + b[o + 7];
    }
    /* 尾巴: out_dim 不是 8 的倍数时剩余输出逐个算(现网四层 512/256/128/24 都是 8 的倍数) */
    for (; o < od; o++){
        float a = 0.0f;
        for (uint32_t i = 0; i < id; i++)
            a += in[i] * (float)w[i * od + o];
        out[o] = a * ws[o] + b[o];
    }
}

/* 激活: swish(x) = x·sigmoid(x) = x/(1+e^-x) */
static inline float swishf(float x){ return x / (1.0f + expf(-x)); }

/* 中间缓冲放文件级: 主循环栈上已经有 buf(180) 等, 这里再压 900+ 字节会吃紧 */
static float h1[POL_DIM_H1], h2[POL_DIM_H2], h3[POL_DIM_H3], xn[OBS_DIM];

/* ==================== ② 前向 ==================== */
/* ⚠️ obs 必须是【原始未归一化】的观测(直接来自 cat_obs_get), 归一化在本函数里做。 */
void policy_run(const float obs[OBS_DIM], float act_out[ACT_DIM]){
    float out[POL_DIM_OUT];

    /* 归一化(与训练侧 ppo.normalize_observations=True 对应) */
    for (uint16_t i = 0; i < OBS_DIM; i++)
        xn[i] = (obs[i] - POL_OBS_MEAN[i]) / (POL_OBS_STD[i] + 1e-8f);

    dense_s8(POL_W1, POL_S1, POL_B1, xn, OBS_DIM,    POL_DIM_H1, h1);
    for (uint16_t i = 0; i < POL_DIM_H1; i++) h1[i] = swishf(h1[i]);

    dense_s8(POL_W2, POL_S2, POL_B2, h1, POL_DIM_H1, POL_DIM_H2, h2);
    for (uint16_t i = 0; i < POL_DIM_H2; i++) h2[i] = swishf(h2[i]);

    dense_s8(POL_W3, POL_S3, POL_B3, h2, POL_DIM_H2, POL_DIM_H3, h3);
    for (uint16_t i = 0; i < POL_DIM_H3; i++) h3[i] = swishf(h3[i]);

    dense_s8(POL_WH, POL_SH, POL_BH, h3, POL_DIM_H3, POL_DIM_OUT, out);

    /* 头层 24 维里只取【前 12】当动作, 过 tanh 压到 [-1,1] */
    for (uint16_t i = 0; i < ACT_DIM; i++)
        act_out[i] = tanhf(out[i]);
}

/* ==================== ③ 溯源 ==================== */
const char *policy_tag(void)      { return POL_TAG; }
const char *policy_build_id(void) { return POL_BUILD_ID; }

/* ------------------------------------------------------------
 *  对拍要求(换算子/换量化方案前必读)
 *    任何改动 dense_s8 / swishf / 归一化 的行为, 都必须先与训练侧的 int8 参考
 *    实现在【同一组观测】上逐层对拍, 通过后才允许合入。理由: 这类错不会崩、
 *    不会报, 只表现为"策略性能变差", 事后极难定位。
 * ------------------------------------------------------------ */
