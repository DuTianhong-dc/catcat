# -*- coding: utf-8 -*-
"""【配置】副本专用 default_config()。

前半段从主线 default_config() 逐字搬来（数值一个没改），
末尾追加 obs_config / action_config / reward_config.terms / task_config 四类结构键。
改结构 = 改 JSON 的这几个键；改默认值才动本文件。
"""
from ml_collections import config_dict

from mujoco_playground._src.locomotion.cat40 import cat40_constants as _consts

from . import obs, rewards


def default_config() -> config_dict.ConfigDict:
    c = config_dict.ConfigDict()
    # 2026-11-04: self-righting 用按策略放宽的 scene_selfright.xml；
    # 默认仍是主线 scene.xml，保证 c40w1~18 与副本等价性回归不变。
    c.model_xml = _consts.MODEL_XML
    c.episode_length = 1000
    c.sim_dt = 0.002
    # timescalefix 2026-08-28: 0.008 -> 0.02, 与官方 Go1 同频。
    # 一次解决三件事(三项都是按步数定义的, 步数不变、物理时间自动 x2.5):
    #   critic 视野   0.27s -> 0.66s   (1/(1-0.97)=33 步; 0.008 时装不下 0.3~0.5s 的步态周期)
    #   unroll 视野   0.16s -> 0.40s   (20 步)
    #   episode 长度    8s  ->   20s   (1000 步)
    # sim_dt 不变, n_substeps 自动从 4 变 10, 仿真精度没有损失。
    c.ctrl_dt = 0.02
    c.action_scale = 0.75  # 旧口径遗留, 仅存档参考(见 _action_scale); 新的用 pos_action_scale
    # ===== 动作映射（2026-10-28）=====
    # 三档可选，默认 per_side：
    #   "absolute" 旧式：clip(a × max(|lo|,|hi|), lo, hi)
    #        ✓ a=0 是站姿   ✗ 不对称限位的窄侧被裁 → 该区间梯度恒 0（实测 joint2 44.6% 时间
    #          落在死区、覆盖 64% 动作行程），策略在里面自由漂移
    #   "offset"  Open Duck 式：站姿 + a × 0.25，再按 max_motor_velocity 限速
    #        ✓ a=0 是站姿  ✓ 无死区   ✗ 行程只用 25%~87%（他们膝行程 180°、用 0.25 只占 16%；
    #          我们膝只有 42°，±0.25rad 已吃掉 87%，想迈大步就没余量）
    #   "per_side" 按侧归一化（默认）：站姿 + a×(hi−站姿) / 站姿 + a×(站姿−lo)
    #        ✓ a=0 是站姿  ✓ 无死区  ✓ a=±1 打满两侧真实行程
    #        这就是"上下限对应 ±1"的正确版本：中点取站姿而不是限位几何中点
    #        （纯归一化 lo+(a+1)/2×(hi−lo) 的中点是 +0.269/+0.148，a=0 时腿已偏 15.4°/8.5°）
    c.act_mapping = "per_side"
    c.pos_action_scale = 0.25    # 仅 "offset" 档使用（Open Duck 原值）
    # 速度上限（rad/s），**按关节分开**，2026-11-01 台架实测（脚本在 bench/）：
    #   舵机轴实测最大 4.9 rad/s（3050 step/s，225g 电池负载；手册标称 5.45）
    #   髋侧摆 / 髋俯仰是舵机直驱  i = 1    → 关节 4.9
    #   膝是五连杆               i ≈ 1.9  → 关节 4.9/1.9 = 2.58
    # ⚠️ 原来用一个标量 5.24（抄 Open Duck，那是直驱 i=1），对膝**快约 2 倍**，
    #    策略会输出真机执行不了的摆腿指令。顺序与 ACTIVE_JOINTS 一致 (abd,pitch,knee)*4。
    c.max_motor_velocity = (4.9, 4.9, 2.58) * 4
    c.dof_vel_scale = 0.05       # Open Duck 原值：obs 里关节速度 ×0.05
    # obs 消融(2026-09-29): True = state_obs 去掉 tau(12) 段。
    # ★ 2026-10-28 改为默认 True：Open Duck 的 state 里没有 tau，只把它放进 privileged
    #   （asymmetric actor-critic）。我们膝/外展长期顶限幅，tau 那几维近似常数，信息量本就低。
    c.ablate_tau = True
    # obs 消融(2026-09-30 第二刀): True = state_obs 去掉 jvel(12) 段。
    # 同样只消策略侧; privileged 里 jvel_c 副本不动。与 ablate_tau 可独立/叠加。
    c.ablate_qvel = False
    # Open Duck 的 actor 也看不到 linvel 和 gravity（只有 acc/gyro）。
    # 但他们是双足 + 有 imitation 参考运动提供步态相位；我们四足没有那两样，
    # 去掉速度反馈风险大，先保留，留开关做单变量对照。
    c.ablate_linvel = False
    c.ablate_gravity = False
    # obs 消融(2026-10-31): True = state_obs 去掉 contact(4) 段。
    #   真机没有足端接触开关 —— Open Duck 是两足、真机焊了微动开关(GPIO D22/D27),
    #   我们抄 obs 时把硬件前提漏了。主流四足 RL 一致不放: legged_gym(ANYmal) 与
    #   MuJoCo Playground 的 go1/g1/t1/h1/op3/barkour/berkeley_humanoid/apollo 的 actor
    #   里都没有 contact; 只有 spot 用了【足端位置 feet_pos】, 那是 FK 能算的量。
    #   奖励里的 feet_air_time / slip / step_bonus 仍用仿真真值 contact, 不受影响。
    c.ablate_contact = True
    c.impl = "jax"                       # 用 JAX(不需 warp)
    c.naconmax = 4 * 8192
    c.njmax = 40
    c.noise_config = config_dict.create(
        level=1.0,
        scales=config_dict.create(joint_pos=0.03,
                                  gyro=0.008,    # imu实测 2026-09-06: 0.4°/s=0.007rad/s(原0.2高估25倍)
                                  gravity=0.01,  # imu实测: quat反算gravity std≈0.01(原0.05高估5倍)
                                  linvel=0.03,   # obsdr: 0.1->0.03, 大头挪给下面的慢变误差
                                  joint_vel=0.86))  # 舵机实测 2026-09-06: 差分残差std(各关节0.31~1.57, 均值0.86)
    # obsdr 2026-08-28: 线速度按"腿部里程计"建模(linvel 方案 A), 不再是完美传感器。
    # 真机上 linvel 只能靠 12 个编码器做正运动学推算, 误差特征是"慢变零偏 + 比例误差"——
    # 关键是它在几秒内持续存在, 不像每步独立抖动那种噪声能被网络几帧平均掉。
    # 这两个都是"每回合采样一次、回合内保持不变", 见 reset() 与 step()。
    c.noise_config.linvel_est = config_dict.create(
        bias=0.05,   # v28 2026-09-15: 0.15->0.05 先教会走路, v29 再加回练真机鲁棒
        # 慢变零偏 (m/s): 标定误差 + 打滑累积
        scale=0.03,  # v28: 0.10->0.03, 同上。慢变零偏/比例误差是蹲着刷赏帮凶:
        # 速度观测 RMS 误差~0.3 m/s, 对 0.3 信号 SNR~1, 闭眼走路十五版走不出
    )
    # tau_est 2026-09-06: 真机 tau = 电流(0x45) x Kt, 电流含各关节不同的保持基线
    # (悬空保持实测 0~45mA ≈ 0~0.035 N·m), 训练建模为每关节独立的慢变零偏。
    c.noise_config.tau_est = config_dict.create(
        bias=0.035,   # v11: 恢复 tau_bias(逐个加回验证第3项)  # 慢变零偏 (N·m): 摩擦补偿差异 + 电流基线(45mA×0.784≈0.035)
    )
    # v30 2026-09-15: 外力推搡训练(移植 Go1 _maybe_apply_perturbation)。
    # 力幅 = 体重 x velocity_kick / 时长 => velocity_kick 即"速度改变量"标尺:
    # cat 行走 0.45 m/s -> kick 0.8 ≈ 1.8x 行走速度(Go1 用 3.0 ≈ 2x 它的速度)。
    # v30b 2026-09-16 煜辰拍板提到 [1.2, 2.4]: 实测 v24 裸策略在设计档 0.8 下
    # 16ep 零摔, 梯度信号近零; 1.2~2.4 档可触发 10~30% 失稳, 逼出恢复步。
    # 评测验收仍用 0.8/1.6 档(v24 舒适区)对比。
    # 默认 enable=False: 评测链零影响; 训练用 --playground_config_overrides 打开。
    c.pert_config = config_dict.create(
        enable=False,
        velocity_kick=[1.2, 2.4],
        kick_durations=[0.05, 0.2],
        kick_wait_times=[2.0, 5.0],
    )
    c.reward_config = config_dict.create(
        scales=config_dict.create(
            # yawfix 2026-08-29: tracking_ang_vel 0.5 -> 1.0。
            # 传感器改成机体系后, 侧滑(螃蟹步)已经从系统性 -15° 降到 ±8° 零均值,
            # 但机身 yaw 仍在缓慢自转(30 s 约 10°): orientation 惩罚的是重力向量
            # 偏离, 绕重力轴旋转不改变重力方向, 所以它对 yaw 完全不敏感;
            # 真正能压住 yaw 的只有这一项, 而它原来只有前进(2.0)的 1/4, 压不住。
            tracking_lin_vel=8.0, tracking_ang_vel=1.0,   # v29 2026-09-15: lin 2->8(v28账本实测走路对弹跳净利润~0, 溢价x4逼步态); walkbias 2026-08-28: 前进x2
            lin_vel_z=-0.5, ang_vel_xy=-0.05, orientation=-1.5,  # orientationfix: 过渡期减税(补上 gaitfix 漏改的这一刀)
            # liftfix2 2026-08-29: pose 0.5 -> 0.2。
            # 原值是无条件把关节往默认站姿拉, 走路时也一直拉 —— 抬脚和后退都要求
            # 关节偏离默认姿态, pose 越大越不敢偏。且前腿默认膝弯 60°、后腿 0°,
            # 前腿要抬脚偏离更多、被罚更狠, 这正是前腿只抬 2.1cm(后腿 4.6cm)的嫌疑。
            # 站桩时仍由 stand_still=-1.0 强约束回站姿, 放松走路时的约束不影响站姿。
            pose=0.2, stand_still=-1.0,
            torques=-0.0002, action_rate=-0.05, energy=-0.001,
            termination=-1.0,
            # liftfix 2026-08-29: feet_air_time 0.1 -> 1.5。
            # 原权重 0.1 配上 0.1s 阈值, 实测滞空 0.125s 时该项贡献仅 0.0025 —— 形同虚设,
            # 策略没有任何动力把脚抬起来。官方 legged_gym(ANYmal C) 这项给 2.0。
            # liftfix3 2026-08-30: feet_clearance -2.0 -> -0.5。
            # 它也拿 max_foot_height 当目标(|foot_z - target_z| x 脚速度),
            # 是瞬时量且按速度加权, 权重又是 feet_height 的 10 倍 ——
            # 才是"每步必须抬够 6cm"的真正执行者。
            # 降到 -0.5: 保留"别抬太高"的约束, 去掉"抬不够就抽"的逼迫。
            # 2026-09-06 改回 -2.0: -0.5 太弱, 策略不认真抬脚(贴地走), 恢复 Go1 同款。
            # v4 2026-09-07: air_time 1.5->2.0(ANYmal C 官方值), clearance -2.0->-0.5 —— 与 v3 反向组合, 验证两参数对步态的主导性
            # v5 2026-09-07: air_time 2.0->1.5(单变量对比 v4), 其余同 v4, 57万实际步短训
            feet_air_time=1.5, feet_clearance=-0.5, feet_height=-0.2,
            # heightfix 2026-08-30: 基座高度惩罚, 只罚低于目标(见下方 target 注释)。
            # -50 的量级: 后退蹲低 0.056m -> 罚 0.156, 约为 tracking_lin_vel(2.0) 的 8%,
            # 是温和引导而非主约束。蹲到离地 0.05m 时罚 0.66, 才明显变重。
            base_height=-100.0,   # v28: -50->-100, 坐姿平均 -243/回合照蹲, 罚款加码抬姿态
            feet_slip=-0.1, dof_pos_limits=-1.0,
            # v26 2026-09-15: step_bonus +8.0 —— 给"抬脚"直接发钱。v25 实证
            # feet_air_time 在猫尺度(滞空0.115s)仅 +0.0007/步, 迈步在账本上
            # 从无正收益, 罚不出抬脚。权重取 pose(0.2) 的 40 倍: 单脚一次
            # 落步最高 0.16, 整周期四脚 ≈0.64(≈14x pose 时薪)。落步瞬间
            # 按弧峰值结算(first_contact, 非逐步) -> 无法悬停刷单; 原地踏步
            # 可赚, 但前进叠加 tracking EMA 赚更多 -> 站 < 原地踏步 < 走的阶梯。
            step_bonus=8.0,
            # overconstraint 2026-09-06: 避免"三腿长时支撑"(过约束最重+真机3舵机长时间大电流)。
            # 软约束: 三腿支撑越久罚越多(平方, 归一化到1s), 无硬阈值, 短时三腿几乎不罚, 给硬件留余地
            # (不逼死"必须2腿", 2腿交替平衡要求高, 舵机精度/响应未必扛得住)。
            # v20 2026-09-10: 死区惩罚(公式自含负号, 故权重取正 1.0)。
            # T_dead=0.2s 死区给步态切换留余量, 超过部分才罚 -> 减少长时三腿
            # (防真机3舵机大电流)。v19 的倒U型因给正奖励而失败(策略去追奖励)。
            # 沿革: v13 设 0.0 关闭(旧纯平方惩罚会带偏早期探索);
            #       v19 倒U型(失败, 位移-33%); v20 死区惩罚(本版)
            over_support=1.0,
            # v42 2026-11-01 (c40w10): 累计偏航偏离, 见 _reward_yaw_drift。
            # 权重 2.0 = 与 tracking_lin_vel 同级: 漂移的收益约 0.45×2.0=0.9,
            # 必须让损失(最大 2.0)明显大于收益才堵得住。
            yaw_drift=2.0,
            # v45 (c40w13): 世界系位置偏离起点。只对 |cmd|<0.1 生效(见 _reward_base_pos)。
            # 量级: 漂 0.5m -> 1.5, 乘 dt 后约等于 tracking(8.0) 满分的 19%, 温和但持续。
            base_pos=-3.0,
            # v46 2026-11-02 (c40w18): 单边超速惩罚。c40w17 实测 cmd 0.2~0.4 段
            # 系统性过冲 0.06~0.11 m/s（跟踪率 1.27~1.31），对称核在 a 小时只损失
            # 约 30%，罚不动。这一项只罚 "v > cmd" 那一半，不与"跑不到"的惩罚冲突。
            # 默认 0 关闭，由训练 override 打开。
            overspeed=0.0),
        tracking_sigma=0.25,
        # σ 单位 rad²。参照: 6°(0.105rad) -> 0.93(仅罚 7%, c40w8 的水平不外罚);
        # 17°(0.30) -> 0.55; 74°(1.29) -> exp(-11) ≈ 0。
        yaw_drift_sigma=0.15,
        # v46 2026-11-02 (c40w18): 偏航死区(rad)。步态本身左右摆头有 ±5~8°，
        # 直接拿瞬时偏航开会把正常摆动一起罚，σ 一小就把步态压死。
        # 死区内的偏离不计，超出部分才进高斯核: dy_eff = max(0, |dy| - dead)。
        yaw_drift_dead=0.0,
        # 超速惩罚的归一化下限: excess = max(0, v-cmd) / max(cmd, overspeed_floor)
        overspeed_floor=0.25,
        # v45 (c40w13): 站立/原地踏步工况(|cmd|<=0.1)的高斯核带宽, 与行走分支分开。
        # 原来是共用的 tracking_sigma=0.25: v=0.3 m/s 时 exp(-0.36)=0.70 ——
        # "原地踏步"要求机身速度≈0, 0.70 的宽容度等于没约束。
        # 收紧后: v=0.1 -> 0.82, v=0.2 -> 0.45, v=0.3 -> 0.17。
        stand_sigma_lin=0.05,
        stand_sigma_ang=0.05,
        # ---- v44 2026-11-01 (c40w12) 照抄 TU Darmstadt 单策略论文 ----
        # §4.1: tracking 带宽 = max(floor, slope*|cmd|)，单位 m/s。
        #   分母 = max(track_floor, track_slope*|a|)，err = (v-a)/分母。
        #   floor=0.0/slope=1.0 等价旧行为(denom=|a|)。
        #   论文: floor 0.5->0.05、slope 0->0.2，在 333~834 迭代内完成。
        #   我们能力 0.58(Go2 是 2.5)，floor 按半能力取 0.3 -> 0.05。
        #   ⚠️ 开局必须宽 —— 论文原话是"fixed width keeps it there(钉在局部最优)"。
        track_tol=0.25,      # 速度追踪容差(相对误差, 25%% 内满分; 11-17 煜辰)
        track_sharp=1.0,     # 容差外衰减斜率倍数(1.0=旧式等价; >1 更陡, 容差不变)
        track_floor=0.0,
        track_slope=1.0,
        # §3.6: 采样偏置 10% 站立 / 31.5% 低速 / 13.5% 原地转 / 45% 常规
        cmd_bias=False,
        cmd_low_max=0.30)
    # v31 2026-09-16: 世界系跟踪开关。True 时 tracking_lin_vel 对
    # "世界系速度在初始航向 d0 上的投影"付费(转弯=世界系掉速, v29/v30
    # 左漂套利关闭)。默认 False = v24/v30 机体系行为, 评测链零影响。
    c.reward_config.world_frame_tracking = False
    # liftfix 2026-08-29: 0.035 -> 0.06 m。
    # 原目标 3.5cm 时实测四脚平均只抬 3.6cm(前腿 2.3/2.5、后腿 4.6/4.8),
    # 且 feet_height=-0.2 会惩罚超过目标的部分, 等于把抬脚高度按死在 3.5cm。
    # 猫腿长约 0.2m, 抬 6cm 约合 30%, 对标 ANYmal 的 10~15cm/腿长 40cm(约 30%)。
    # v18 2026-09-10: 0.06 -> 0.03。实测猫抬脚仅 12~28mm(60mm 目标达成率 21~46%),
# 目标不可达时抬脚奖励形同空转(够不着=拿不到分=没动力)。降到 3cm 进入可达范围。
# 注意: 该项同时是 feet_clearance 的目标与 feet_height 的天花板(同一语义, 自洽)。
    # v19 2026-09-10: 0.03 -> 0.06 改回。方案C已证伪(降目标=压天花板, 抬脚反降
    # 11.7~19.2mm < v17 的 12.7~25.8mm); 本轮配合方案A(三腿倒U型奖励)一起放开。
    c.reward_config.max_foot_height = 0.06  # 摆动抬脚目标(相对脚底基准, m)
    # overconstraint 起步宽限: 前 warmup_steps 步豁免 over_support 惩罚。
    # 从 4 腿站立切到交替步态有个过渡, 不豁免则策略可能学不会起步。
    c.reward_config.warmup_steps = 50        # 50 步 @50Hz = 1.0s
    # liftfix2 2026-08-29: 滞空阈值 0.1 -> 0.05 s。
    # 上一轮把权重从 0.1 提到 1.5(15 倍) 抬脚却纹丝不动(3.6->3.4cm), 原因就在
    # 这个 0.1: 猫实测滞空只有 0.115 s, 减去阈值只剩 0.015 —— 被减掉的才是大头,
    # 权重再大也只是放大一个残差值。0.1 是 Go1 的数, 猫的步态周期只有 0.23 s
    # (Go1 约 0.6 s), 阈值必须按自己的尺度取: 取滞空的 40~50% = 0.05 s。
    # (同类坑: legged_gym 的 0.5 s 阈值是给 ANYmal 的, 也不能照抄。)
    c.reward_config.air_time_threshold = 0.05  # 滞空超过多少秒才开始给分(s)
    # heightfix 2026-08-30: 基座离地目标高度(m)。
    # 0.165 = 站姿实测(站姿 qpos[2]=0.095 + 地面 geom 偏移 0.07)。
    # 前进时离地 0.166~0.181 -> 不受罚;
    # 后退蹲低时离地 0.109 -> 罚 0.156。
    # cat4.1 H154/+16 零位：base_z=0.14878，站立 3s 实测约 0.1469
    # 目标取 0.147，只罚低于新站高的姿态，避免把策略拉回旧 0.140 站高
    c.reward_config.base_height_target = 0.147
    c.soft_joint_pos_limit_factor = 0.95  # 软限位系数(Go1 同款)
    # terminate 2026-08-28: 摔倒终止阈值(照官方 Go1 机制, 按 cat3.3 尺寸定)
    c.terminate_config = config_dict.create(
        min_base_height=0.10,  # cat4.1: 侧躺约 0.082、新站高约 0.147；0.08 判不出摔倒
                               # 取 0.10 卡在两者之间
        up_vec_z=0.4,          # 机身竖直度低于此值判摔倒(cos66°, 侧翻/仰翻)
    )
    # zero_prob 2026-08-28: 指令采样每个分量置零概率(见 _sample_command)
    # novy 2026-09-10: 彻底删除侧移(vy), 命令仅保留 [前进vx, 转向wz]
    # 依据: 猫腿矢状面摆动为主, 侧移=螃蟹步效率低; 实测 vy 仅给 ±0.2
    # 影响: 观测 60 -> 59 维, 策略权重不兼容, 需从头训练
    # min_vx: 前进指令下限。c40w12 起可配（原来硬编码 0.3）。
    #   取值要点: 上限 a[0] 必须贴近【实测速度能力】，否则对称奖励在不可达段
    #   一直给"加速"梯度，与低段需要的"减速"冲突，策略会退化成单速。
    # max_wz 2026-11-02 (c40w16): 转向幅度上限(rad/s)。设 0 = 完全禁转, 只做直线速度跟随。
    #   原来 -0.3~0.3 是写死的三处(非偏置分支 / cmd_bias 的 wn / reset 初始指令),
    #   只改 b 关不掉 reset 那一处 -> 每回合开场仍会带一个随机 wz。
    c.command_config = config_dict.create(a=[0.5, 0.5], b=[0.9, 0.5], zero_prob=0.1,
                                          fixed=None, min_vx=0.3, max_wz=0.3,
                                          cmd_mode='')  # 11-16: 'three_way_mix'=三类命令各1/3(vx对称含0); 其余stage留空等价
    # ---- 2026-11-04 复现副本：四类结构配置驱动（主线没有这些键）----
    c.obs_config = config_dict.create(
        blocks=list(obs.DEFAULT_BLOCKS),
        # 2026-11-16 walk: True 时 critic 特权观测去掉 linvel_c (默认关, 历史等价)
        drop_privileged_linvel=False)
    c.action_config = config_dict.create(
        mode=c.act_mapping,             # absolute / per_side / offset / delta_current / delta_anchored
        apply_velocity_limit=True,      # 论文式动作可设 False
        delta_scale=0.25,               # delta_current / delta_anchored 档的增量尺度
        # 2026-11-22 v16: delta_anchored —— 部分锚定站姿，lambda=0 时与 delta_current 一致。
        # lambda 按机身高度渐入（bz<anchor_bz_lo 关断），避免把趴姿的腿推向站姿而把人推倒。
        anchor_lambda=0.0,
        anchor_bz_lo=0.09,
        anchor_bz_span=0.03)
    # scales 由奖励注册表生成 —— 不再需要「dict 与 scales 两处手工对齐」
    c.reward_config.scales = config_dict.create(**rewards.default_scales())
    # 默认只启用主线 22 项：Table II 的 9 项成本由配方显式列出（保证等价性回归）
    c.reward_config.terms = list(rewards.MAINLINE_TERMS)

    # ---- 2026-11-04 Lee 2019 self-righting：全部默认关闭，主线等价性不变 ----
    # 论文 r = -Σ k·c。cost_mode=True 时骨架对加权和取负；scales 里存的就是论文的 k>0。
    c.reward_config.cost_mode = False
    # 骨架 jp.clip(sum*dt, 0, 10000) 会吞掉论文的有界负回报；配方设 -10000 允许负值。
    c.reward_config.clip_min = 0.0
    # logistic kernel K(e|α) = -1/(exp(αe)+2+exp(-αe)) ∈ [-0.25,0)
    # ⚠️ α 与全部 k 论文未公开，属自定参数（对照文档 §6，实验卡登记）
    c.reward_config.alpha_o = 4.0
    c.reward_config.alpha_jp = 4.0
    # 论文 p4 要求 bounded cost：0.5 m 落地瞬间 Σqacc² 可达 1e6~1e8，
    # 无界会让单步 reward 撞到 -10000（实测），梯度被冲击帧主导。
    # 对每项成本取 min(c, cost_cap)；K 核项本身 ∈[-0.25,0) 不受影响。
    c.reward_config.cost_cap = 100.0
    # EXP-04A：冲击帧单独上界；0 = 沿用 cost_cap（默认，等价性保持）。
    c.reward_config.impulse_cap = 0.0
    # 论文 Table II 的 c_h = 1 if h < 0.35 else 0，用于 standing-up 任务。
    # 0 = 关闭（默认，历史配方逐位不变）；本机标定值 0.115（零位站高 0.1473 的 78%）。
    c.reward_config.c_height_thr = 0.0
    # ---- 2026-11-14 Stage-2 v2 (默认全关, 历史配方逐位不变) ----
    # c_h 铰链: 参考高度=零位站高 (keyframe stand 沉降 600 步实测)
    c.reward_config.c_height_hinge_enable = False
    c.reward_config.c_height_ref = 0.1473
    # 2026-11-15 s2 pose-gate: 站正(up_z)->站高(bz)->目标姿态 逐级解锁, 默认关
    # w1=clip((up_z-u_lo)/(u_hi-u_lo)) 作用于 c_height; w2=clip((bz-h_lo)/(h_hi-h_lo)) 作用于 c_joint_pos
    c.reward_config.pose_gate_enable = False
    c.reward_config.pose_gate_u_lo = 0.75
    c.reward_config.pose_gate_u_hi = 0.95
    c.reward_config.pose_gate_h_lo = 0.105
    c.reward_config.pose_gate_h_hi = 0.125
    # 到目标高度后 c_height 失效(交给 c_joint_pos 主导); 目标 = ratio * c_height_ref
    c.reward_config.pose_gate_h_done_ratio = 0.9
    # 2026-11-15 walk: 论文 locomotion 的跟踪/功率/足端项 (默认关)
    c.reward_config.c_walk_enable = False
    c.reward_config.c_walk_alpha = 2.0
    c.reward_config.c_foot_clearance_h = 0.04
    # 2026-11-16 累积量跟踪（默认关，历史配方逐位不变）
    # 为什么不用瞬时量: 四足的转向与前进都是步进式，瞬时速度与指令必然对不上
    #   （实测瞬时角速度误差 0.41~0.52，正确转向段反而比直走段还大）。
    #   只有「指令积分出的期望转角/位移」与「实际累积量」的差才是真误差。
    # 容差固定且左右对称: 若容差随行程增长，误差上限就等于最终容差，约束失效。
    c.reward_config.c_track_accum_enable = False
    # 2026-11-16 连续姿态惩罚: 相对起点 z/up_z 的恶化量, 带容差、无饱和
    c.reward_config.c_posture_enable = False
    # 2026-11-17 walk: 净位移门控 —— 原地踏步/站立不再领足端奖
    c.reward_config.c_moved_gate_enable = False
    c.reward_config.c_moved_gate_dist = 0.005
    # 净位移棘轮的归一化步长：每前进 2cm 记满 1.0，只认历史最远点的新增量
    c.reward_config.c_progress_step = 0.02
    c.reward_config.post_tol_ratio = 0.05   # 容差 = 初始值的 5%
    c.reward_config.post_scale_z = 0.030
    c.reward_config.post_scale_u = 0.10
    c.reward_config.c_yaw_track_tol = 0.0873    # rad = 5 deg（煜辰 11-16 定；注意会误伤步进式转向的启动滞后）
    c.reward_config.c_dist_track_tol = 0.50     # m（20s 行程的 ~5~12%）
    # 高度渐入门控: gate0=(h0+H)/2, w=clip((h-gate0)/(H-gate0),0,1)
    # h0>=H 的回合直接 w=1 (本来已站够高)
    c.reward_config.height_gate_enable = False
    c.reward_config.height_gate_ref = 0.1473
    c.reward_config.height_gate_terms = []
    # 四足端投影面积上限: A/A_stand > 1.2 才罚, 归一化分母 0.2
    c.reward_config.c_area_enable = False
    c.reward_config.c_area_ref = 0.040915     # m^2, 零位姿态四足端投影面积
    c.reward_config.c_area_thr_ratio = 1.2
    c.reward_config.c_area_norm = 0.2
    # 平方项在质心/面积远离约束区时会爆炸, 必须限幅 (smoke 实测 c_stab 峰值 26.7)
    c.reward_config.c_area_cap = 1.0
    c.reward_config.c_stab_cap = 1.0
    # 质心投影稳定裕度: 到最近边内距 < margin 才罚
    c.reward_config.c_stab_enable = False
    c.reward_config.c_stab_margin = 0.020     # m, 约形心内切距 0.064 的 30%
    # 左右交叉门控: 机体系左右两组平均 y 间距低于此值逐渐关闭几何项
    c.reward_config.c_cross_dy_thr = 0.0257  # m, 约零位间距 0.1286 的 20%
    # 2026-11-14 P1 r3: 结构门控 + 起身棘轮增量势, 默认全关 (历史配方逐位不变)。
    # s0 = -j6 + j3 + 0.5*j0; 门控与状态门控边界取自 r2 轨迹离线回放标定。
    c.reward_config.c_boot_enable = False
    c.reward_config.c_boot_s_lo = -0.62
    c.reward_config.c_boot_s_hi = 0.08
    c.reward_config.c_boot_u_lo = -0.999
    c.reward_config.c_boot_u_hi = 0.30
    # ---- 2026-11-14 Stage-2 v3: c_height 抬升增量信号 (默认全关, 历史配方逐位不变) ----
    # 只认回合内 body_z 棘轮新高增量, 与绝对高度无关; 每拍上限防落地冲击尖峰。
    c.reward_config.c_height_delta_enable = False
    c.reward_config.c_height_delta_cap = 0.005   # m/拍 (dt=0.02 -> 0.25 m/s)
    # ---- 2026-11-15 Stage-2 v5: 高度增量项的两道门 + 防摔重罚 (默认全关) ----
    # 入场资格: up_z 低于 u_lo 不给增量分(还在躺/侧), u_hi 以上满额。
    c.reward_config.c_height_delta_u_lo = 0.60
    c.reward_config.c_height_delta_u_hi = 0.75
    # 天花板: 到 站高 x lo_ratio 开始衰减, 到 x hi_ratio 完全失效(再高不给分)。
    c.reward_config.c_height_delta_top_lo_ratio = 0.95
    c.reward_config.c_height_delta_top_hi_ratio = 1.00
    # 记账门控: 只有 up_z 达标时才推进 body_z 棘轮(否则翻倒尖峰会吃掉额度)。
    c.reward_config.c_height_delta_ratchet_gate = True
    # 防摔: 曾经正到 u_gate 后, up_z 从峰值回撤超过 tol 开始罚, 到 dmax 罚满。
    c.reward_config.c_fall_enable = False
    c.reward_config.c_fall_u_gate = 0.75
    c.reward_config.c_fall_drop_tol = 0.20
    c.reward_config.c_fall_drop_max = 0.50
    # ---- 2026-11-15 Stage-2 v6: 稳→正→高→姿 的四级顺序 (默认全关) ----
    # c_stab 改线性形式（平方+cap 在质心出多边形时恒为 1.0，没有梯度）
    c.reward_config.c_stab_linear_enable = False
    c.reward_config.c_stab_scale = 0.10      # m: (margin_ref - margin)/scale，0.10 覆盖 -80~+20mm
    # c_area 改两侧偏离（旧式只罚"面积超标"，实测站正时只有 0.866 倍，永不触发）
    c.reward_config.c_area_two_sided_enable = False
    c.reward_config.c_area_dev_scale = 0.30  # 偏离站姿面积 30% 即罚满
    # 以稳门控"正"（c_boot）: w = max(floor, 1 - c_stab)
    c.reward_config.c_boot_stab_gate_enable = False
    c.reward_config.c_boot_stab_floor = 0.20
    # 2026-11-15 v6b: 稳由"罚"改"奖"（raw-1 → reward = w·(1-raw)·dt，越稳拿越多）
    c.reward_config.c_stab_reward_mode = False
    c.reward_config.c_area_reward_mode = False
    # 2026-11-18 Stage-2 v9：站姿奖按高度资格缩放（默认关，历史配方逐位不变）。
    # 打开后 c_stab/c_area 的奖励乘以 clip((bz-h_floor)/(h_ref-h_floor))，
    # 消掉"低蹲静止也拿满站姿奖"的套利（诊断见 audit_s2_bank_terms.py）。
    # 2026-11-20 v11 新增（默认值 = 历史行为逐位不变）
    c.reward_config.c_foot_drag_enable = False
    c.reward_config.c_foot_drag_v0 = 0.02       # m/s 以内免罚
    c.reward_config.c_foot_drag_scale = 0.20    # 达到 v0+scale 记满 1
    c.reward_config.c_upz_jerk_enable = False
    c.reward_config.c_upz_jerk_tol = 0.02
    c.reward_config.c_upz_jerk_scale = 0.10
    c.reward_config.c_area_hi_cap = 1.0         # >1 允许“面积越大罚越重”
    c.reward_config.c_jp_gate_mode = "height"   # height=旧 / stab=按站稳门控
    # 2026-11-21 v14: 支撑几何只统计着地足（默认关，关时逐位不变）。
    # 打开后 c_area/c_stab 只在接触足 >= c_geo_min_contacts 时计算，
    # 且多边形顶点只取着地足 —— 堵掉"抬一条腿把投影外推、面积与内距同时变大"这条通路。
    c.reward_config.c_geo_contact_only = False
    c.reward_config.c_geo_min_contacts = 3
    c.reward_config.c_area_ref3_ratio = 0.5     # 三足支撑的参考面积 = 站立面积 × 该比例

    # ---- 2026-11-21 v15: 阶段式「全奖励」（默认关；关时历史配方逐位不变）----
    # 每一项都是有界奖励 r∈[0,1]，无负项；阶段 1..4 累积判定，权重随阶段变化。
    c.reward_config.c_stage_enable = False
    c.reward_config.c_stage_base_pose = 2.0
    c.reward_config.c_stage_base_stab = 3.0
    c.reward_config.c_stage_base_area = 2.0
    c.reward_config.c_stage_base_hdelta = 2.0
    c.reward_config.c_stage_base_upz = 2.0
    c.reward_config.c_stage_base_contact = 1.0
    c.reward_config.c_stage_base_slip = 1.0
    c.reward_config.c_stage_w_pose = [0.00, 0.25, 0.25, 0.50, 1.00]
    c.reward_config.c_stage_w_stab = [0.00, 1.00, 0.50, 0.25, 0.25]
    c.reward_config.c_stage_w_area = [0.00, 1.00, 0.50, 0.25, 0.25]
    c.reward_config.c_stage_w_hdelta = [0.25, 0.25, 0.50, 1.00, 0.25]
    c.reward_config.c_stage_w_upz = [0.25, 0.25, 1.00, 0.50, 0.25]
    c.reward_config.c_stage_w_contact = [1.0, 1.0, 1.0, 1.0, 1.0]
    c.reward_config.c_stage_w_slip = [1.0, 1.0, 1.0, 1.0, 1.0]
    c.reward_config.c_stage_bonus = [0.0, 0.0, 0.6, 2.0, 3.0]
    c.reward_config.c_stage_m_cap = 0.0754   # 站姿内距 62.8mm × 1.2
    c.reward_config.c_stage_pose_tol = 1.0
    c.reward_config.c_stage_area_dev = 0.6
    c.reward_config.c_stage_hdelta_v = 0.01
    c.reward_config.c_stage_upz_lo = 0.6
    c.reward_config.c_stage_slip_v = 0.02
    c.reward_config.c_stage_hold = 50
    c.reward_config.c_stance_h_gate_enable = False
    c.reward_config.c_stance_h_floor = 0.075
    c.reward_config.c_stance_h_ref = 0.1473
    # EXP-04A：c_orientation 改相对起点势；False = 原论文 K(e)（默认）。
    c.reward_config.orientation_baseline = False
    # EXP-04B：'start'=减本回合起点 K(e0)（默认，等价）；
    #          'inverted'=减固定倒立基线 K(e=2)，消白拿但保留保持正立的持续奖励。
    c.reward_config.orientation_baseline_mode = "start"
    # 目标坐姿关节角（相对零位，rad），配方从 sit_pose_touch.json 覆盖
    c.reward_config.q_sit = [0.0] * 12
    # ---- EXP-14: c_joint_pos 四髋侧摆 + up_z 门控（默认全关，旧配方等价）----
    c.reward_config.c_jp_hip_only = False
    # 2026-11-13：关 hip_only 时是否也减 K(c_jp_ref) 基线。默认 False（历史逐位等价）；
    # True 时 c_jp 变成 [0,0.25] 的纯惩罚，与论文 dφ>=0 同向（Stage-2 必须开）。
    c.reward_config.c_jp_kernel_baseline = False
    c.reward_config.c_jp_hip_target = [0.0, 0.0, 0.0, 0.0]  # 中位 = 0
    c.reward_config.c_jp_hip_mode = "min2_abs"  # rms | mean_abs | mean_signed | min2_abs
    c.reward_config.c_jp_ref = 0.6        # 减基线用的参考误差 (rad)
    c.reward_config.c_jp_gate_lo = -0.8   # up_z 低于此值门控为 0
    c.reward_config.c_jp_gate_hi = 0.0    # up_z 到此处门控满
    c.reward_config.c_jp_gate_decay = 0.5 # 超过 hi 后线性衰减到 0 的跨度
    c.reward_config.c_jp_slope_fix = False # 用 slope_theta 修正 up_z 基准
    # ---- EXP-14 阶段 A：倒立起始姿态（一侧中位 + 一侧限位）----
    c.reward_config.c_jp_stageA_enable = False   # 默认关，上一轮分支逐位等价
    c.reward_config.c_jp_hip_lo = [-1.003, -1.003, -1.134, -1.134]
    c.reward_config.c_jp_hip_hi = [1.003, 1.003, 0.524, 0.524]
    c.reward_config.c_jp_stageA_frac = 0.85      # prop=行程比例, abs=绝对值(rad)
    c.reward_config.c_jp_stageA_mode = "prop"    # prop | abs
    c.reward_config.c_jp_up_z1 = -0.99577        # 阶段 A 退场起点
    c.reward_config.c_jp_up_z3 = -0.89471        # 阶段 B 进场起点
    c.reward_config.c_jp_side_thresh = 0.0       # -sin(slope_phi) > thr 选 A 侧；
                                                 # 0.0 = A/B 各 50%（0.836 只覆盖 18.5%，2026-11-13 修）
    c.reward_config.c_jp_ref_a = 0.5             # 阶段 A 线性核满量程
    c.reward_config.c_jp_ref_b = 0.7             # 阶段 B 线性核满量程
    c.reward_config.c_jp_stageA_gain = 0.15      # 与上一轮 K 核量级对齐
    # ---- EXP-03 阶段 1 课程：势能塑形 + 侧躺成功奖励（默认全 0 = 关闭）----
    # Ng 1999 potential-based shaping: F = w*(up_z_t - up_z_{t-1})；
    # gamma=1 时首尾相消，不改最优策略，只把「跨槛才得分」变成
    # 「每一点进展都得分」。全程 -1->0 累计恰好 = w。
    c.reward_config.stage1_shaping_w = 0.0
    c.reward_config.stage1_success_bonus = 0.0
    # 2026-11-09 迁移 cat3.4 Cat33Flip v2 口径：成功前把 9 项成本整体置 0，
    # 成功后（stage1_hold>=need）才恢复惩罚。默认 False = 旧路径逐位不变。
    c.reward_config.gate_penalty_on_success = False
    # 只门控列出的项（空 = 全部）。保留 c_orientation 这类密集项，
    # 避免「成功前全免罚 → advantage≈0 → 只剩熵」把策略练成随机（§7.47.2）。
    c.reward_config.gate_terms = []
    # 论文 p6：四脚全离地时放大 base linvel/位置噪声以模拟 TSIF 漂移（#27）
    # 默认 1.0 = 不加噪声，rng 流不变 → 等价性保持
    c.noise_config.airborne_gain = 1.0
    c.task_config = config_dict.create(
        init_mode="stand", terminate_mode="fall",
        # 2026-11-20: 关节符号转换层（策略侧“旧约定”↔ 当前 MJCF“新约定”），默认关。
        # 11-18 模型重生成把左右镜像从 joint axis(±1) 搬进了 range（右侧镜像、axis 全 +1），
        # 同一物理姿态 q 差符号：q_new = sign(axis_old)*sign(axis_new)*q_old。
        # 打开后：起点库(旧约定)在 reset 转换、观测转回旧约定、动作转成新约定。
        joint_sign_mask_enable=False,
        joint_sign_mask=[-1.0, -1.0, 1.0,
                         1.0, 1.0, -1.0,
                         -1.0, -1.0, 1.0,
                         1.0, 1.0, -1.0],
        pose_bank_npz="",  # P1: offline settled-pose bank for init_mode=pose_bank
        drop_height=0.5,          # legacy：固定绝对高度（仅 drop_clearance<0 时生效）
        # 2026-11-05 方案 A：离地净空(m)。>=0 时按该随机姿态最低 geom 反推 qpos[2]，
        # 默认 -1 关闭 → 默认行为与主线逐位等价。取值 0.08 = 约 0.5 倍站高净空。
        drop_clearance=-1.0,
        joint_range_scale=0.8,    # 随机关节角占 jnt_range 的比例，留 20% 余量防顶限位
        # EXP-02 倒立初始分布：默认 0 = 关闭（等价性保持）。
        # frac>0 时按比例采样「绕随机水平轴 θ∈[arccos(uzmax),π]」的四元数，
        # 使 up_z=cosθ ≤ uzmax（默认 -0.2 = 明显倒立/侧倒）。
        init_inverted_frac=0.0,
        init_up_z_max=-0.2,
        # EXP-03 阶段 1：初始全倒立 -> 只要「随意侧躺」就成功。
        # 默认全关；开启时 info 才多一个 stage1_hold 计数器。
        stage1_enable=False,
        stage1_terminate_on_success=False,
        stage1_up_z_min=-0.5,   # 脱离四脚朝天区即算跨过槛（连续 10 帧）
        stage1_hold_steps=10,
        # 2026-11-11 EXP-10：势能塑形改「棘轮」(回合内 up_z 单调上包络)。
        # 电平势 duz 在倒立附近来回摆、净增≈0，梯度消失（EXP-09 §3.3）。
        # 默认 False → 旧路径逐位不变。
        stage1_ratchet=False,
        # 2026-11-11 EXP-11：参考轨迹偏差奖励（默认关闭）。
        # 奖励 = w·exp(−k·d)，d 为 qpos 相对参考帧的归一化均方偏差（按 ep_steps 取帧）。
        # 参考只进奖励、不进动作、不进 obs → 部署时不需要相位对齐。
        ref_track_enable=False,
        ref_track_npz="",
        ref_track_w=1.0,
        ref_track_k=1.0,
        ref_track_sigma_base=0.10,   # m
        ref_track_sigma_quat=0.20,   # 四元数分量
        ref_track_sigma_joint=0.25,  # rad
        # 默认只比「12 主动关节 + up_z」：base 平移/roll/yaw 受 DR（质量+0.1kg、
        # 质心、坡度 xfrc）影响太大，参考轨迹又是零 DR 录的，比了只会互相打架。
        ref_track_use_base=False,
        ref_track_sigma_upz=0.15,
        ref_track_w_ori=0.5,
        ref_track_window=5,          # 时间容差：允许比参考慢/快几拍
        # 2026-11-11 EXP-12：起点侧倾随机化（rad）。机身朝哪边倾就选哪一侧的参考，
        # 真机上这个线索等价于坡度/支撑面方向。默认 0 → 不倾斜，旧行为不变。
        ref_tilt_max=0.0,
        ref_track_abort_d=8.0,       # 偏差超过它连续 N 拍就终止
        ref_track_abort_steps=3,
        ref_track_abort_pen=0.0,     # 因发散被终止时的额外惩罚（负值）
        ref_track_abort_keep=0.0,    # 终止时保留已得分的比例（>0 时优先于 abort_pen）
        ref_track_shift=0.0,         # 每拍平移量：r = exp(-k·d) − shift
        ref_track_action_w=0.0,      # 动作匹配奖权重（需要库里带 act）
        ref_track_action_sigma=0.35, # 动作匹配的容差（越大越宽松）
        ref_track_prog_w=0.0,        # 进度奖权重：d 每下降 1 给多少分
        ref_track_d_min=0.0,         # d 的下限：d 低于它按满分算，避免追求 d→0
        # EXP-04A：里程碑状态课程，默认全关（等价性保持）。
        demo_states=[],          # 参考轨迹里程碑 qpos 列表，空 = 关闭
        demo_frac=0.0,          # reset 时从里程碑起步的比例
        demo_noise=0.0,         # 四元数/12 个主动关节角的均匀扰动幅度(rad)
        stage1_success_rising_edge=False,  # True = bonus 只在首次达成那一拍发
        # ---- EXP-08：默认全关，等价性回归保证旧配方逐位不变 ----
        exp08_enable=False,
        exp08_m0=1.9590,              # 整机标称质量 (kg)
        exp08_x_com0=0.0052,          # 标称质心 x (m)
        exp08_x_add=0.1967,           # 挂载点 x (m)
        exp08_mass_max=0.100,         # 挂载质量上限 (kg)
        exp08_dx_bias=0.005,          # 前后各加 5 mm
        exp08_dy_ratio=0.300,         # 左右幅度 = 该比例 * dx_ref
        exp08_slope_max_deg=7.0,      # 坡度上限，方位全向
        exp08_init_q0=[],             # 数据集固定倒立起点 (31 维)，配方填入
        exp08_init_quat_noise=0.05,   # 水平轴四元数扰动 (rad)
        exp08_init_joint_noise=0.02,  # 主动关节角扰动 (rad)
        exp08_init_z_noise=0.003,     # 初始高度扰动 (m)
        # ---- 专家动作前缀课程（默认全关，旧路径逐位不变） ----
        warm_start_enable=False,
        warm_start_max=50,            # 每回合最多前 W 拍下发专家动作
        warm_start_zero_prob=0.25,    # 强制 W=0（纯策略）的比例
        warm_start_action_path="",    # robust_best_robust.npz，含 action (T,12)
        # ---- Stage 0：随机倒立 -> 专家 q0 邻域（默认全关） ----
        stage0_enable=False,
        stage0_target_qpos=[],        # 专家 q0 (31 维)
        stage0_tol_ori=0.15,          # 成功容差：机身 z 轴夹角 (rad)
        stage0_tol_joint=0.10,        # 成功容差：主动关节角 (rad)
        stage0_tol_xy=0.04,           # 成功容差：基座 xy (m)
        stage0_tol_z=0.012,           # 成功容差：基座 z (m)
        stage0_rew_ori=0.30,          # 奖励尺度：2x 成功容差
        stage0_rew_joint=0.20,
        stage0_rew_xy=0.08,
        stage0_rew_z=0.024,
        stage0_w_ori=0.35,
        stage0_w_joint=0.35,
        stage0_w_xy=0.15,
        stage0_w_z=0.15,
        stage0_hold_steps=10,         # 连续满足容差 10 拍 = 成功
        stage0_success_bonus=20.0,
        stage0_action_rate_w=0.005,
        stage0_torque_w=1e-5,
        stage0_terminate_on_success=True,
        stage0_out_xy=0.5,            # 出界保护 (m)
        stage0_out_z=0.02,            # 出界保护 (m)
        # 2026-11-10: 奖励减基线。d 的绝对水平 ~3 会让整回合 return 到 -300 量级，
        # value 输出层 bias 在 256 次 SGD 内学不到这个量级（实测 -0.032 vs -200），
        # baseline≈0 使 advantage 反映"初始状态多难"而非"动作好坏"。减掉典型 d 后
        # return ≈ 0，value 从 0 初始化即接近正确。默认 0.0 = 旧路径逐位不变。
        stage0_rew_baseline=0.0,
        # 2026-11-10: reset 后的自由沉降步数（sim step, 非控制拍）。
        # Go1/Spot Getup 用 settle_time=0.5s 让机器人先落地稳定再计奖励；
        # 我们起点贴地，只需消除初始穿透/接触冲击 → 0.1s = 50 sim step。
        # 默认 0 时 reset 逐位不变。
        settle_steps=0,
        # 2026-11-10: reset 时按最低碰撞点贴地（消掉 initial_state 留下的悬空 ~5cm）。
        # 与 settle_steps 独立：默认 False → 旧路径逐位不变。
        reset_ground=False,
        # EXP-09 A 段：关节角为主项（默认关闭 → 旧 stage0 逐位不变）
        stage0_v2=False,
        stage0_sigma_q=0.25,          # e_q 的关节角尺度 (rad)
        stage0_v2_sigma_ori=0.30,     # e_ori 的姿态尺度 (rad)
        stage0_tol_eq=0.4,            # 成功容差（e_q≤0.4 ≈ RMS 0.1 rad）
        stage0_tol_ev=0.5,            # 成功容差（e_v≤0.5 ≈ 1 rad/s）
        stage0_v2_w_eq=1.0,
        stage0_v2_w_ori=0.3,
        stage0_v2_w_ev=0.1)
    return c


