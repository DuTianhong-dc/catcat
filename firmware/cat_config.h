/**
 ******************************************************************************
 * @file    cat_config.h — 系统级配置（硬件 / 时钟常量, 各模块共用）
 * @brief   只放"换块板子、改个主频就要动"的常量。
 *          某个模块自己的参数(比如串口波特率、缓冲区大小)不放这儿, 放它自己的头文件里。
 *
 * 时钟依据 通信协议.txt 的（2.3）: SYSCLK = 168MHz, APB2 = HCLK/2 = 84MHz, APB1 = HCLK/4 = 42MHz。
 ******************************************************************************
 */
#ifndef CAT_CONFIG_H
#define CAT_CONFIG_H

/* ==================== 时钟 ==================== */
#define CPU_MHZ     168u    //内核主频(MHz), 用于 DWT 周期数 ↔ 微秒 换算
#define APB2_MHZ     84u    //APB2 总线频率(MHz), USART6(舵机) 挂在这条总线上
#define APB1_MHZ     42u    //APB1 总线频率(MHz), USART2(IMU) 挂在这条总线上

/* ==================== 策略维度 ====================
 * 放这里(而不是某个模块头)是为了避开横向 include: cat_obs 和 cat_policy 都要用,
 * 各自 include cat_config.h 即可, 谁也不依赖谁。
 * ⚠️ OBS_DIM 必须与【要部署的那个策略文件】一致 —— 不是与训练代码一致:
 *    实测 play/cat33_policy_v31.npz 的 W1=(59,512)、obs_mean/obs_std 均为 (59,)。 */
#define OBS_DIM      59u    //观测维数(2026-09-10 删侧移 vy, 命令 3->2 维)
#define ACT_DIM      12u    //动作维数 = 12 颗舵机

/* 动作缩放: q_target = 默认站姿 + act × 它(act 已过 tanh, 范围 [-1,1])。
 * ⚠️ 必须与训练侧 joystick.py 的 action_scale 一致 —— 改它等于换了动作幅度, 别随意动。
 * 训练侧现值 0.75(walkbias2 2026-08-28 "动作幅度+50%"), 注释留了"0.75->1.0 可再放"的余地。 */
#define ACTION_SCALE 0.75f

#endif /* CAT_CONFIG_H */
