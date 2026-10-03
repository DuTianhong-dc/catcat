/* ============================================================
 *  cat_pose.c — 姿态/限位系统常量 (值来源: 模型 cat3.4 实测)
 *
 *  ⚠️ 与模型严格一致(换模型/改限位必须同步三处: 本文件、训练侧、文档),
 *     依据 → 观测组装说明.md(七)(十)。
 * ============================================================ */
#include "cat_pose.h"
#include "cat_crc32.h"

/* 默认站姿(rad): stand keyframe, jpos 段基准 */
const float CAT_POSE_DEF[ACT_DIM] = {
     0.0f,    0.25f,    0.3f,
     0.0f,   -0.25f,   -0.3f,
     0.0f,    0.0f,     0.0f,
     0.0f,    0.0f,     0.0f
};

/* 关节限位下界(rad): jnt_range[:,0] */
const float CAT_POSE_LIM_LO[ACT_DIM] = {
    -1.30f, -1.20f, -0.14f,
    -0.70f, -0.52f, -3.00f,
    -0.55f, -1.20f, -0.14f,
    -0.55f, -0.50f, -3.00f
};

/* 关节限位上界(rad): jnt_range[:,1] */
const float CAT_POSE_LIM_HI[ACT_DIM] = {
     0.70f,  0.52f,  3.00f,
     1.30f,  1.20f,  0.14f,
     0.55f,  0.50f,  3.00f,
     0.55f,  1.20f,  0.14f
};

/* 三表指纹: 策略槽头部的 pose_hash 必须与它相同(顺序 DEF→LO→HI, 小端字节序) */
uint32_t cat_pose_hash(void){
    uint32_t c = cat_crc32(0, (const uint8_t *)CAT_POSE_DEF,    ACT_DIM * 4u);
    c = cat_crc32(c, (const uint8_t *)CAT_POSE_LIM_LO, ACT_DIM * 4u);
    c = cat_crc32(c, (const uint8_t *)CAT_POSE_LIM_HI, ACT_DIM * 4u);
    return c;
}
