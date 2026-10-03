/* ============================================================
 *  cat_policy.h — int8 策略网络前向 + 权重来源选择(出厂常量 / 槽 A / 槽 B)
 *
 *  网络: OBS_DIM → 512 → 256 → 128 → 24, 前三层 swish, 头层取【前 12】过 tanh。
 *  量化: int8 权重 + 按【输出通道】的 float scale(比全张量 scale 误差低数倍)。
 *
 *  ⚠️ POL_DIM_OUT(24) ≠ ACT_DIM(12): 头层 24 维里【前 12 维】才是动作均值。
 *  ⚠️ 权重来源运行时决定(在线更新): 开机 / 每次更新成功后调 policy_select();
 *     槽无效 → 回退出厂常量; 出厂常量可编译期关闭(cat_config.h 的 CAT_POL_FACTORY)。
 *     槽格式与校验 → 通信协议.txt 第七章。
 * ============================================================ */
#ifndef __CAT_POLICY_H
#define __CAT_POLICY_H
#include "stm32f4xx.h"
#include "cat_config.h"     /* OBS_DIM / ACT_DIM / POL_DIM_* / CAT_POL_FACTORY */

/* ==================== 出厂常量(生成物, 见 policy_weights.c) ====================
 * 每层: int8 权重矩阵 + 按输出通道的 scale + bias(float)。
 * CAT_POL_FACTORY=0 时这些符号不被引用(文件可整个移出工程)。 */
extern const int8_t POL_W1[OBS_DIM * POL_DIM_H1];
extern const float  POL_S1[POL_DIM_H1];
extern const float  POL_B1[POL_DIM_H1];

extern const int8_t POL_W2[POL_DIM_H1 * POL_DIM_H2];
extern const float  POL_S2[POL_DIM_H2];
extern const float  POL_B2[POL_DIM_H2];

extern const int8_t POL_W3[POL_DIM_H2 * POL_DIM_H3];
extern const float  POL_S3[POL_DIM_H3];
extern const float  POL_B3[POL_DIM_H3];

extern const int8_t POL_WH[POL_DIM_H3 * POL_DIM_OUT];
extern const float  POL_SH[POL_DIM_OUT];
extern const float  POL_BH[POL_DIM_OUT];

extern const float  POL_OBS_MEAN[OBS_DIM];      /* 观测归一化(训练侧 normalize_observations) */
extern const float  POL_OBS_STD[OBS_DIM];

extern const char   POL_BUILD_ID[];             /* 来源 npz 的 md5 前 12 位(溯源) */
extern const char   POL_TAG[];                  /* 训练侧标签(如 v31/overconstraint) */

/* ==================== 来源选择(在线更新) ==================== */
/* 开机 / 更新成功后调用: 扫槽 A/B, 取"有效且 seq 最大"者为当前策略;
 * 无有效槽 → 回退出厂常量(CAT_POL_FACTORY=1)或进"无策略"状态(=0)。 */
void policy_select(void);

uint8_t  policy_source(void);       /* 0=出厂常量 1=槽A 2=槽B */
uint8_t  policy_ready(void);        /* 1=有可用策略; 0=无(未灌过且无出厂 → 待机) */
uint32_t policy_seq(void);          /* 当前策略 seq(出厂常量=0) */
const char *policy_tag(void);       /* 当前策略标签(展示/溯源) */
const char *policy_build_id(void);  /* 当前权重指纹 */

/* ==================== 前向 ==================== */
/* obs: OBS_DIM 维【原始】观测(未归一化, 直接来自 cat_obs_get);
 * act_out: ACT_DIM 维动作, 已过 tanh, 范围 [-1,1]。
 * 动作是"翻身版"语义: 目标关节角 = clip(act × max(|lo|,|hi|), lo, hi) —— 见 cat_obs.c。 */
void policy_run(const float obs[OBS_DIM], float act_out[ACT_DIM]);

#endif
