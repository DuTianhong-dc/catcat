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

/* ==================== 安全 ==================== */
/* IWDG 看门狗总开关(2026-09-29 深夜定, 煜辰拍板"一起上"):
 *   1 = 上电启动(1s 超时, 主循环每拍喂); 固件卡死 → 硬件自动复位, 不用拔电。
 *   0 = 不启动(纯开发模式, 比如要靠断点单步查问题时的备用闸)。
 * ⚠️ 平时别关: 关掉后卡死只能手动断电; 调试冻结(DBGMCU)已经在, 挂着调试器不会被误复位。 */
#define CAT_IWDG_ENABLE   1u

/* ==================== 策略维度 ====================
 * 放这里(而不是某个模块头)是为了避开横向 include: cat_obs 和 cat_policy 都要用,
 * 各自 include cat_config.h 即可, 谁也不依赖谁。
 * ⚠️ OBS_DIM 必须与【要部署的那个策略文件】一致 —— 不是与训练代码一致:
 *    实测 play/cat33_policy_v31.npz 的 W1=(59,512)、obs_mean/obs_std 均为 (59,)。 */
#define OBS_DIM      59u    //观测维数(2026-09-10 删侧移 vy, 命令 3->2 维)
#define ACT_DIM      12u    //动作维数 = 12 颗舵机

/* 网络分层尺寸(2026-09-29 从 cat_policy.h 挪来: 策略前向与在线更新两侧都要用) */
#define POL_DIM_H1  512u
#define POL_DIM_H2  256u
#define POL_DIM_H3  128u
#define POL_DIM_OUT  24u    /* ⚠️ 不是 ACT_DIM: 头层 24 维里【前 12】才是动作均值 */

/* 出厂策略常量开关(2026-09-29 建; 2026-09-30 切 0 = "无出厂"终态):
 * 0 = 无出厂(纯双槽 —— 两槽都空则"无策略待机": 卸力 + 拒使能 + 等 OTA);
 * 1 = policy_weights.c 的编译期权重当兜底(seq=0 来源)。
 * ⚠️ SIL 构建(build_host2.py)显式 -DCAT_POL_FACTORY=1 用它的自带权重 —— 别误删那个旗标。
 * (policy_weights.c 已移出 Keil 工程, 省 ~205KB Flash; 文件留在磁盘上备查。) */
#ifndef CAT_POL_FACTORY   /* ⚠️ 必须可被构建命令覆盖: SIL 用 -DCAT_POL_FACTORY=1(自带权重);
                           * 裸 #define 会把命令行 -D 顶掉(2026-09-30 踩过: SIL 空策略崩) */
#define CAT_POL_FACTORY   0u
#endif

/* 动作→关节角映射(2026-09-28 起): 「翻身版」q = clip(act × max(|lo|,|hi|), lo, hi),
 * 每关节按物理限位"不多不少" —— 常量(ACT_LIM_LO/HI)与实现都在 cat_obs.c;
 * 旧口径 `默认站姿 + act×0.75`(旧 ACTION_SCALE) 已废弃。依据 → 观测组装说明.md(十)。 */

#endif /* CAT_CONFIG_H */
