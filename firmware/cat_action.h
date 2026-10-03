#ifndef __CAT_ACTION_H
#define __CAT_ACTION_H
#include "stm32f4xx.h"
#include "cat_servo.h"      /* SERVO_COUNT */
#include "cat_config.h"     /* ACT_DIM / POL_* 无关, 只要动作维度 */

/* ============================================================
 *  cat_action — 动作模块(关键帧表: 加载通道 + 播放器)
 *
 *  分层(新函数插对应区块, 别追加到末尾):
 *    ① 加载通道   action_frame() / action_load_*
 *    ② 播放器     (下一轮: play/tick/abort —— v1 cat_action.c 移植)
 *
 *  数据来源与下发方案 → 动作模块设计.md(二)(三)(四); v1 参考实现 → CAT_APP/cat_action.c
 *
 *  通道复用(不新建帧类型, 2026-10-03 定):
 *    与策略 OTA 同走 0xCC 变长帧, 靠【子类型】分派 —— 0x10~0x3F 动作 / 0x70~ 策略。
 *    帧内载荷顺序 = j0..j11, seg, hold, [torque], [speed], [acc](全 float32 小端);
 *    ⚠️ 与 CSV 列序(seg,hold,j0..j11,…)【不同】—— 重排在 PC 侧做, 见设计文档。
 * ============================================================ */

/* ---- 帧子类型(0xCC 载荷首字节) ---- */
#define ACT_SUB_BEGIN   0x10u   /* 清空表, 开始一次加载会话 */
#define ACT_SUB_FRAME   0x20u   /* 追加一帧: 全 float32 (14/15/17 个) */
#define ACT_SUB_END     0x30u   /* 锁定表 (空表=拒, 不锁定) */

/* ---- 上行应答 type(v2 上行帧 "5A A5 | len | type | 载荷 | ck" 的 type 字段)
 * 沿用 v1 的 ACK 语义(rc + 帧数), 但编号用 v2 风格: 避开 OTA 的 0x76(ACK)/0x77(STAT) ---- */
#define ACT_UP_ACK      0x78u

/* ---- 返回码(与 v1 cat_action_load_frame 对齐; 0x0A 为 v2 新增) ---- */
#define ACT_RC_OK       0x00u
#define ACT_RC_FULL     0x01u   /* 表满(> ACT_RAM_MAX 帧) */
#define ACT_RC_SEG      0x02u   /* seg_s 超范围 */
#define ACT_RC_HOLD     0x03u   /* hold_s 超范围 */
#define ACT_RC_NAN      0x04u   /* 关节角含 NaN */
#define ACT_RC_LEN      0x05u   /* 载荷长度非法 / 本次会话列数不一致 */
#define ACT_RC_ORDER    0x08u   /* 顺序错(未 BEGIN 就 FRAME / 空表 END) */
#define ACT_RC_GATE     0x0Au   /* 门卫不过(已使能 或 有故障锁存) —— 站着不动才能换表 */

/* ---- 表规格 ---- */
#define ACT_RAM_MAX      64u    /* 关键帧上限(64 × 64B ≈ 4KB RAM, 与 v1 一致) */
#define ACT_FLOAT_MAX    17u    /* 单帧 float 个数上限(17 列形态) */
#define ACT_SEG_MIN    0.05f    /* seg_s 下限 */
#define ACT_SEG_MAX   25.0f     /* seg_s 上限 */
#define ACT_HOLD_MAX  25.0f     /* hold_s 上限 */

/* ==================== ① 加载通道 ==================== */
/* 处理一帧动作子协议(已去 0xCC 帧壳, d[0] = 子类型)。填应答, 返回上行 type(0 = 不理)。
 * gate_ok: 门卫结果(1 = 未使能且无故障), 由编排层算 —— 见 .c 里 BEGIN/END 的用法。
 * 应答载荷 = rc(1) + 当前帧数(1), 共 2 字节 —— 与 cat_polupd_frame() 同构, main 一处分派。 */
uint8_t action_frame(const uint8_t *d, uint16_t n, uint8_t gate_ok,
                     uint8_t resp[48], uint8_t *resp_len);

/* 清空表并开会话(BEGIN 用; 也会让旧表立即失效 —— 与 v1 行为一致) */
void    action_load_begin(void);

/* 表已锁定(END 成功)且非空 = 1 —— 播放器与编排层用它判"有没有动作可播" */
uint8_t action_ready(void);

/* 表内帧数(0 = 空; 加载中给已收帧数) */
uint8_t action_load_count(void);

/* 取第 idx 帧的 12 个绝对关节角(rad, 加载时已钳进限位)。返回 0=成功, 1=越界。
 * 播放器逐帧插值取目标用; 也让"表被读"这件事在加载通道阶段即成立(便于切片实现)。 */
uint8_t action_kf_pose(uint8_t idx, float out[ACT_DIM]);

/* ==================== ② 播放器 ====================
 * v1 CAT_APP/cat_action.c 的移植 + 适配 v2(无 FSM → 由编排层门控受理)。
 * 播放 = 进场(当前角→首帧, 0.8s) → 逐帧 smoothstep 插值 + 保持 → 出场(→默认站姿, 0.8s);
 * 全程 5 rad/s 限速 + 关节限位截断; 每帧的 torque/acc/speed 由编排层写进舵机。
 * ⚠️ 与安全链的关系: 播放【不参与】卸力决策 —— 故障了由编排层调 action_abort()(平滑回站姿),
 *    卸力仍归安全闸(0x28=0 广播)。"停止/失联"不打断播放(动作是本地序列, 停在半路更糟)。 */

/* 播放阶段(编排层/观测区探针用) */
#define ACT_STAGE_IDLE   0u
#define ACT_STAGE_ENTER  1u     /* 进场 */
#define ACT_STAGE_KF     2u     /* 关键帧段 */
#define ACT_STAGE_EXIT   3u     /* 出场(含 abort 后的平滑回站姿) */

/* 有播放请求(END 成功时置)且门卫过 → 从 cur(当前绝对关节角)开始播。
 * 返回 1 = 本拍已起播; 0 = 没请求 / 门卫不过(请求被丢弃, 不挂起 —— 免得久后一使能就意外动)。
 * gate_ok 由编排层算(= 未使能 + 无故障), 内部不反向依赖别的模块。 */
uint8_t action_play_start_if_requested(const float cur[ACT_DIM], uint8_t gate_ok);

/* 推进一拍(调用方按 50Hz 调)。out = 本拍目标【绝对关节角】。
 * 返回 0 = 进行中; 1 = 已播完(本拍 out 已回默认站姿, 播放态自动清)。 */
uint8_t action_tick(float out[ACT_DIM]);

/* 是否在播(编排层据此选"动作输出"还是"策略输出") */
uint8_t action_playing(void);

/* 中断播放: 不硬停 —— 把"当前姿态"当出场起点, 走正常出场插值平滑回站姿。
 * 调用方之后继续 tick 即可(它会返回 1 结束); 故障/急停路径用。 */
void    action_abort(void);

/* 取本帧运动参数。返回 1 = 与上次取的不同(调用方据此决定要不要写舵机参数寄存器)。 */
uint8_t action_cur_params(uint16_t *torque, uint16_t *speed, uint8_t *acc);

/* 进度探针: 阶段 + 当前关键帧号(进观测区用) */
void    action_progress(uint8_t *stage, uint8_t *idx);

#endif
