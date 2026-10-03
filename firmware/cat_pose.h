/* ============================================================
 *  cat_pose.h — 姿态/限位系统常量 (默认站姿 + 关节限位)
 *
 *  这些是"系统常量", 不是策略权重: 不随在线更新变化; 改动 = 换模型,
 *  需重编译并同步训练侧。策略槽头部的 pose_hash 用来校验"策略与它们同源"。
 * ============================================================ */
#ifndef __CAT_POSE_H
#define __CAT_POSE_H
#include "stm32f4xx.h"
#include "cat_config.h"     /* ACT_DIM */

extern const float CAT_POSE_DEF[ACT_DIM];       /* 默认站姿(rad): jpos 基准 + 动作映射参考 */
extern const float CAT_POSE_LIM_LO[ACT_DIM];    /* 关节限位下界(rad) = 模型 jnt_range[:,0] */
extern const float CAT_POSE_LIM_HI[ACT_DIM];    /* 关节限位上界(rad) = 模型 jnt_range[:,1] */

/* 三张表的 CRC32 (顺序: DEF → LO → HI)。策略槽头部的 pose_hash 必须与它一致才被受理。 */
uint32_t cat_pose_hash(void);

#endif
