/* ============================================================
 *  cat_policy.h — int8 策略网络前向 (权重由工具从 .npz 生成)
 *
 *  网络: OBS_DIM(59) → 512 → 256 → 128 → 24, 前三层 swish, 头层取【前 12】过 tanh
 *  量化: int8 权重 + 按【输出通道】的 float scale(比全张量 scale 误差低数倍)
 *
 *  ⚠️ POL_DIM_OUT(24) ≠ ACT_DIM(12): 头层 24 维里【前 12 维】才是动作均值, 其余弃用。
 *  ⚠️ 权重版本必须与固件同步: POL_BUILD_ID 是来源 npz 的 md5 前 12 位, 启动时打印一次,
 *     防止"固件里悄悄躺着旧权重"。
 * ============================================================ */
#ifndef __CAT_POLICY_H
#define __CAT_POLICY_H
#include "stm32f4xx.h"
#include "cat_config.h"     /* OBS_DIM / ACT_DIM */

/* ==================== 网络层尺寸 ====================
 * 换策略版本可能改这里; 改完要同步: ① 权重数组维度 ② 生成工具的参数 */
#define POL_DIM_H1   512u
#define POL_DIM_H2   256u
#define POL_DIM_H3   128u
#define POL_DIM_OUT   24u

/* ==================== 权重(生成物, 见 policy_weights.c) ====================
 * 每层: int8 权重矩阵 + 按输出通道的 scale + bias(float) */
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

/* 观测归一化参数(训练侧 normalize_observations=True) */
extern const float  POL_OBS_MEAN[OBS_DIM];
extern const float  POL_OBS_STD[OBS_DIM];

/* 溯源: 启动时打印一次, 确认固件里的权重和预期版本一致 */
extern const char   POL_BUILD_ID[];     /* 来源 npz 的 md5 前 12 位 */
extern const char   POL_TAG[];          /* 训练侧标签(如 v31/overconstraint) */

/* ==================== 函数原型 ==================== */
/* 前向推理。obs: OBS_DIM 维【原始】观测(未归一化, 直接来自 cat_obs_get);
 *                    act_out: ACT_DIM 维动作, 已过 tanh, 范围 [-1,1]。
 * 输出是"相对默认站姿的归一化动作", 要乘 action_scale 再加站姿才是目标关节角。 */
void policy_run(const float obs[OBS_DIM], float act_out[ACT_DIM]);

const char *policy_tag(void);           //当前策略标签(溯源)
const char *policy_build_id(void);      //当前权重指纹

#endif
