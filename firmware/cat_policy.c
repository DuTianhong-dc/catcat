/* ============================================================
 *  cat_policy.c — int8 策略网络前向 + 权重来源选择(出厂常量 / 槽 A / 槽 B)
 *
 *  分层(按此顺序排列, 新函数插对应区块, 别追加到末尾):
 *    ① 内部算子   dense_s8 / swishf
 *    ② 来源选择   policy_select 及取数
 *    ③ 前向       policy_run
 *
 *  ⚠️ dense_s8 的优化(-O2 + 8 分块)是实测调出来的, 别顺手改回去 —— 依据见
 *     策略前向说明.md。任何改动都必须先与训练侧 int8 参考对拍再合入。
 * ============================================================ */
#include "cat_policy.h"
#include "cat_pose.h"
#include "cat_polupd.h"     /* 槽扫描/布局(单向: 本文件只读槽, 不写) */
#include <math.h>
#include <stdint.h>         /* uintptr_t: 槽地址→指针(PC 64 位快检也不报警) */
#include <string.h>

/* ==================== ① 内部算子 ==================== */
/* 按输出通道量化的 int8 全连接层(输入仍是 float)。
 *   公式: out[o] = (Σ_i in[i] × W[i][o]) × ws[o] + b[o]
 * ⚠️ 三个坑都在第一版踩过:
 *   ① 权重布局: W 是 (in_dim, out_dim)【行主序】, 按转置读会输出全错且不报错;
 *   ② 输入不能用 (int32_t) 截断 —— 归一化后是 ±1 上下的 float;
 *   ③ bias 只加一次, 且【不乘】ws。
 * 实现: o 按 8 个一组分块(累加器住 FPU 寄存器), 消掉对 out[o] 的读-改-写
 *       (12 条/MAC → 4.4 条/MAC, 省 ~4ms); 循环变量用 uint32_t 免 UXTH 截断。 */
static void dense_s8(const int8_t *w, const float *ws, const float *b,
                     const float *in, uint16_t in_dim, uint16_t out_dim, float *out){
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
    /* 尾巴: out_dim 不是 8 的倍数时逐个算(现网四层 512/256/128/24 都是 8 的倍数) */
    for (; o < od; o++){
        float a = 0.0f;
        for (uint32_t i = 0; i < id; i++)
            a += in[i] * (float)w[i * od + o];
        out[o] = a * ws[o] + b[o];
    }
}

/* 激活: swish(x) = x·sigmoid(x) = x/(1+e^-x) */
static inline float swishf(float x){ return x / (1.0f + expf(-x)); }

/* ==================== ② 来源选择 ==================== */
/* 参数集: 一次前向所需的全部指针(指向出厂常量或 Flash 槽数据区) */
typedef struct {
    const int8_t *w1; const float *s1; const float *b1;
    const int8_t *w2; const float *s2; const float *b2;
    const int8_t *w3; const float *s3; const float *b3;
    const int8_t *wh; const float *sh; const float *bh;
    const float  *mean; const float *std;
} pol_set_t;

static pol_set_t s_pol =
#if CAT_POL_FACTORY
    /* 静态初值 = 出厂常量: 即使 policy_select() 尚未被调用(SIL/异常路径)也绝不为空指针
     * —— 第一版的教训, 别删。 */
    { POL_W1, POL_S1, POL_B1,
      POL_W2, POL_S2, POL_B2,
      POL_W3, POL_S3, POL_B3,
      POL_WH, POL_SH, POL_BH,
      POL_OBS_MEAN, POL_OBS_STD };
#else
    { 0 };
#endif
static uint8_t   s_src;                                 /* 0=出厂常量(与静态初值一致) */
static uint8_t   s_ready = (uint8_t)CAT_POL_FACTORY;    /* 有出厂常量即"有策略" */
static uint32_t  s_seq;
static char      s_tag[CAT_POL_TAG_LEN];
static char      s_build[CAT_POL_BUILD_LEN];

/* 拷贝字符串(带上限, 总留结尾 0) */
static void pol_set_str(char *dst, uint32_t cap, const char *src){
    uint32_t i = 0;
    for (; i + 1u < cap && src[i]; i++) dst[i] = src[i];
    dst[i] = '\0';
}

/* 把参数集指向定序 blob(顺序: W1 S1 B1 … MEAN STD, 见 通信协议.txt 7.4) */
static void pol_point_blob(const uint8_t *base){
    const uint8_t *p = base;
    s_pol.w1 = (const int8_t *)p; p += (uint32_t)OBS_DIM * POL_DIM_H1;
    s_pol.s1 = (const float  *)p; p += POL_DIM_H1 * 4u;
    s_pol.b1 = (const float  *)p; p += POL_DIM_H1 * 4u;
    s_pol.w2 = (const int8_t *)p; p += (uint32_t)POL_DIM_H1 * POL_DIM_H2;
    s_pol.s2 = (const float  *)p; p += POL_DIM_H2 * 4u;
    s_pol.b2 = (const float  *)p; p += POL_DIM_H2 * 4u;
    s_pol.w3 = (const int8_t *)p; p += (uint32_t)POL_DIM_H2 * POL_DIM_H3;
    s_pol.s3 = (const float  *)p; p += POL_DIM_H3 * 4u;
    s_pol.b3 = (const float  *)p; p += POL_DIM_H3 * 4u;
    s_pol.wh = (const int8_t *)p; p += (uint32_t)POL_DIM_H3 * POL_DIM_OUT;
    s_pol.sh = (const float  *)p; p += POL_DIM_OUT * 4u;
    s_pol.bh = (const float  *)p; p += POL_DIM_OUT * 4u;
    s_pol.mean = (const float *)p; p += (uint32_t)OBS_DIM * 4u;
    s_pol.std  = (const float *)p;
}

#if CAT_POL_FACTORY
/* 参数集指到出厂常量(policy_weights.c) */
static void pol_use_const(void){
    s_pol.w1 = POL_W1; s_pol.s1 = POL_S1; s_pol.b1 = POL_B1;
    s_pol.w2 = POL_W2; s_pol.s2 = POL_S2; s_pol.b2 = POL_B2;
    s_pol.w3 = POL_W3; s_pol.s3 = POL_S3; s_pol.b3 = POL_B3;
    s_pol.wh = POL_WH; s_pol.sh = POL_SH; s_pol.bh = POL_BH;
    s_pol.mean = POL_OBS_MEAN; s_pol.std = POL_OBS_STD;
    s_src   = 0;
    s_seq   = 0;
    s_tag[0]   = '\0';      /* 清槽信息: 免"槽坏退常量"后 tag 还说上一个槽的谎 */
    s_build[0] = '\0';
}
#endif /* CAT_POL_FACTORY */

/* 开机 / 更新成功后调用: 扫槽 A/B, 取"有效且 seq 最大"者为当前策略。
 * 槽验证(cat_polupd_scan)含全量 CRC32, 单槽 ~20ms —— 只在开机与更新完成时跑。 */
void policy_select(void){
#ifdef CAT_SIL_DIAG
    /* SIL(PC 仿真): 无真 Flash, 不假装有槽 —— 只用出厂常量(与真机"槽无效"路径一致) */
  #if CAT_POL_FACTORY
    pol_use_const();
    s_ready = 1u;
  #else
    s_ready = 0u;
  #endif
#else
    uint8_t  best_slot  = 0u;           /* 0=出厂常量(若有) */
    uint32_t best_seq   = 0u;
    uint8_t  best_valid = 0u;

  #if CAT_POL_FACTORY
    pol_use_const();                    /* 默认=常量, 视为 seq=0 的兜底来源 */
    best_valid = 1u;
  #endif

    for (uint8_t slot = 1u; slot <= 2u; slot++){        /* 1=槽A 2=槽B */
        uint32_t seq;
        if (!cat_polupd_scan(slot, &seq)) continue;     /* 验不过 = 无效 */
        if (!best_valid || seq > best_seq){
            best_valid = 1u; best_slot = slot; best_seq = seq;
        }
    }

    if (best_slot){
        /* 切到槽: 指针 + 头部信息(seq 已知, tag/build 抄出来给展示) */
        const cat_pol_hdr_t *h =
            (const cat_pol_hdr_t *)(uintptr_t)cat_polupd_slot_addr(best_slot);
        pol_point_blob((const uint8_t *)(uintptr_t)cat_polupd_slot_addr(best_slot)
                       + CAT_POL_DATA_OFF);
        s_src = best_slot;
        s_seq = best_seq;
        pol_set_str(s_tag,   CAT_POL_TAG_LEN,   h->tag);
        pol_set_str(s_build, CAT_POL_BUILD_LEN, h->build);
        s_ready = 1u;
    } else {
        s_src   = 0u;
        s_seq   = 0u;
        s_ready = (uint8_t)CAT_POL_FACTORY;             /* 无出厂且无槽 → 无策略待机 */
    }
#endif /* CAT_SIL_DIAG */
}

uint8_t  policy_source(void){ return s_src; }
uint8_t  policy_ready(void) { return s_ready; }
uint32_t policy_seq(void)   { return s_seq; }

/* ==================== ③ 前向 ==================== */
/* 中间缓冲放文件级: 主循环栈上已经有 buf(180) 等, 这里再压 900+ 字节会吃紧 */
static float h1[POL_DIM_H1], h2[POL_DIM_H2], h3[POL_DIM_H3], xn[OBS_DIM];

/* ⚠️ obs 必须是【原始未归一化】的观测(直接来自 cat_obs_get), 归一化在本函数里做。 */
void policy_run(const float obs[OBS_DIM], float act_out[ACT_DIM]){
    float out[POL_DIM_OUT];

    /* 归一化(与训练侧 ppo.normalize_observations=True 对应) */
    for (uint16_t i = 0; i < OBS_DIM; i++)
        xn[i] = (obs[i] - s_pol.mean[i]) / (s_pol.std[i] + 1e-8f);

    dense_s8(s_pol.w1, s_pol.s1, s_pol.b1, xn, OBS_DIM,    POL_DIM_H1, h1);
    for (uint16_t i = 0; i < POL_DIM_H1; i++) h1[i] = swishf(h1[i]);

    dense_s8(s_pol.w2, s_pol.s2, s_pol.b2, h1, POL_DIM_H1, POL_DIM_H2, h2);
    for (uint16_t i = 0; i < POL_DIM_H2; i++) h2[i] = swishf(h2[i]);

    dense_s8(s_pol.w3, s_pol.s3, s_pol.b3, h2, POL_DIM_H2, POL_DIM_H3, h3);
    for (uint16_t i = 0; i < POL_DIM_H3; i++) h3[i] = swishf(h3[i]);

    dense_s8(s_pol.wh, s_pol.sh, s_pol.bh, h3, POL_DIM_H3, POL_DIM_OUT, out);

    /* 头层 24 维里只取【前 12】当动作, 过 tanh 压到 [-1,1] */
    for (uint16_t i = 0; i < ACT_DIM; i++)
        act_out[i] = tanhf(out[i]);
}

/* ==================== ④ 溯源 ==================== */
/* 槽生效时用槽头的 tag/build; 出厂常量时用编译期字符串(CAT_POL_FACTORY=0 时空串=无策略) */
const char *policy_tag(void){
#if CAT_POL_FACTORY
    return s_tag[0] ? s_tag : POL_TAG;
#else
    return s_tag;
#endif
}
const char *policy_build_id(void){
#if CAT_POL_FACTORY
    return s_build[0] ? s_build : POL_BUILD_ID;
#else
    return s_build;
#endif
}
