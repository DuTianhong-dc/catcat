/* ============================================================
 *  cat_obs.h — 59 维观测组装器 (顺序与训练/策略严格一致)
 *
 *  分层: ① 初始化 obs_init()   ② 灌数据 obs_set_*()   ③ 更新+取数 obs_update()/obs_get()
 *
 *  分段表(左闭右开) —— 全部依据、系数、标定表 → 观测组装说明.md:
 *    [ 0: 3) linvel   机体系线速度估计 m/s       ← 腿里程计(是估计, 不是真值)
 *    [ 3: 6) gyro     机体系角速度 rad/s         ← IMU
 *    [ 6: 9) gravity  世界 -Z 在机体系(单位向量)  ← IMU 四元数
 *    [ 9:11) cmd      [vx, wz]                   ← 上位机(无 vy)
 *    [11:23) jpos     关节角 − 默认站姿 (rad)     ← 舵机, ⚠️ 是【相对量】
 *    [23:35) jvel     关节角速度 rad/s           ← 位置差分 + 一阶低通
 *    [35:47) tau      关节力矩 N·m               ← 0x45 电流 × Kt
 *    [47:59) last_act 上一拍动作                 ← 策略输出
 *
 *  ⚠️ 维度要与【要部署的那个策略文件】对齐, 不是与训练代码对齐 ——
 *     实测 cat33_policy_v31.npz 的 W1=(59,512)、obs_mean/obs_std 均 (59,)。
 * ============================================================ */
#ifndef __CAT_OBS_H
#define __CAT_OBS_H
#include "stm32f4xx.h"
#include "cat_config.h"     /* OBS_DIM / ACT_DIM */
#include "cat_servo.h"      /* SERVO_COUNT */

/* 各段起点。⚠️ 改布局必须同步改这里和上面的分段表 —— 下面的编译期断言会兜底。 */
#define OBS_OFF_LINVEL       0u
#define OBS_OFF_GYRO         3u
#define OBS_OFF_GRAVITY      6u
#define OBS_OFF_CMD          9u
#define OBS_OFF_JPOS        11u
#define OBS_OFF_JVEL        23u
#define OBS_OFF_TAU         35u
#define OBS_OFF_LASTACT     47u

/* 各段长度 */
#define OBS_LEN_VEC          3u     /* linvel / gyro / gravity 都是 3 维 */
#define OBS_LEN_CMD          2u     /* vx, wz —— 没有 vy */

/* 编译期兜底: 分段必须正好铺满 OBS_DIM, 少一维多一维都直接编译报错 */
typedef char _obs_layout_check[
    (OBS_OFF_LASTACT + ACT_DIM == OBS_DIM) ? 1 : -1];

/* ==================== 函数原型 ==================== */
/* ① 初始化 */
void obs_init(void);                        //各缓存置初值(关节角 = 默认站姿)

/* ② 灌数据 */
void obs_set_cmd(float vx, float wz);                             //命令 [vx, wz], 带上限钳位
void obs_set_lastact(const float act[ACT_DIM]);                   //⚠️ 上一拍, 不是这一拍
void obs_set_tau_from_cur(const int16_t cur_lsbs[SERVO_COUNT]);    //0x45 电流 LSB(有符号)

/* ③ 更新 + 取数 */
void obs_update(const uint16_t bus_counts[SERVO_COUNT], float dt_s);  //计数→关节角/速度/linvel
void obs_get(float out[OBS_DIM]);                                     //填 59 维

/* ④ 反向: 策略动作 → 舵机目标计数
 * act: ACT_DIM 维动作(已过 tanh, [-1,1]); out: 各颗 12bit 目标位置(直接喂 servo_sync_write)。
 * 链路 = 默认站姿 + act×ACTION_SCALE → 标定表逆变换 → 钳到 [0,4095]。
 * 返回被钳掉的颗数(0 = 全在范围内) —— 用来发现"策略想去的位置超出机械行程"。 */
uint8_t obs_act_to_counts(const float act[ACT_DIM], uint16_t out[SERVO_COUNT]);

#endif
