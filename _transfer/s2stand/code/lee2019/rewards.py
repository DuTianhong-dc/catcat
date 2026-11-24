# -*- coding: utf-8 -*-
"""【奖励】奖励项注册表 + 全部公式（从 _get_reward 逐字搬来，未改任何一个数）。

主线痛点：项名在 _get_reward() 的 dict 里，权重在 default_config().reward_config.scales，
两边必须严格一致，加一项要同时改两处。
本模块用 TERM_SPECS 一处声明「项名 + 默认权重」，default_config() 由它自动生成 scales；
compute_rewards() 按 reward_config.terms 过滤，于是 JSON 能直接启停奖励项：
    "reward_config.terms": ["orientation", "pose", "..."]
新增一个奖励项 = all_rewards() 的 dict 加一行 + TERM_SPECS 加一行（同一文件）。
"""
import jax.numpy as jp

# (项名, 默认权重) —— 数值与主线 default_config().reward_config.scales 完全一致，
# 由 play/_check_lee2019_equiv.py 的逐位等价回归保证。
TERM_SPECS = [
    ("tracking_lin_vel", 8.0),
    ("tracking_ang_vel", 1.0),
    ("yaw_drift", 2.0),
    ("overspeed", 0.0),
    ("base_pos", -3.0),
    ("lin_vel_z", -0.5),
    ("ang_vel_xy", -0.05),
    ("orientation", -1.5),
    ("pose", 0.2),
    ("base_height", -100.0),
    ("stand_still", -1.0),
    ("torques", -0.0002),
    ("action_rate", -0.05),
    ("energy", -0.001),
    ("termination", -1.0),
    ("feet_air_time", 1.5),
    ("feet_clearance", -0.5),
    ("step_bonus", 8.0),
    ("feet_height", -0.2),
    ("feet_slip", -0.1),
    ("over_support", 1.0),
    ("dof_pos_limits", -1.0),
    # ---- Lee 2019 Table II 的 9 项成本（默认不进 terms，由配方启用）----
    # 权重 k 是自定起点值，论文只说「k_o 最高」，未给任何数值（对照文档 §6）
    ("c_orientation", 8.0),
    ("c_joint_pos", 2.0),
    ("c_joint_accel", 0.05),
    ("c_body_impulse", 0.5),
    ("c_body_slip", 0.5),
    ("c_self_collision", 0.0),   # 精简模型仅 10 碰撞 geom，该项几乎恒 0
    ("c_action_diff", 0.05),
    ("c_joint_speed", 0.5),
    ("c_torque", 0.0002),
    # 论文 Table II 的 c_h，只在 standing-up 配方里进 terms（默认权重 0 = 关闭）
    ("c_height", 0.0),
    # 2026-11-14 Stage-2 v2: 四足端投影几何项, 默认权重 0 (历史配方逐位不变)
    ("c_area", 0.0),
    ("c_stab", 0.0),
    # 2026-11-14 P1 r3: 结构门控 + 起身棘轮增量势, 默认权重 0 (历史配方逐位不变)
    ("c_boot", 0.0),
    # 2026-11-14 Stage-2 v3: c_height 的抬升增量信号, 默认权重 0 (历史配方逐位不变)
    ("c_height_delta", 0.0),
    # 2026-11-15 Stage-2 v5: 正过之后再跌回去的重罚, 默认权重 0
    ("c_fall", 0.0),
    # 2026-11-20 v11: 足端拖滑（全程）与 up_z 瞬变（失稳）惩罚，默认 0 = 关闭
    ("c_foot_drag", 0.0),
    ("c_upz_jerk", 0.0),
    # 2026-11-15 walk: 论文 locomotion 的跟踪/功率/足端项, 默认权重 0
    ("c_omega", 0.0),
    ("c_lin_vel", 0.0),
    ("c_power", 0.0),
    ("c_foot_clearance", 0.0),
    ("c_foot_slip", 0.0),
    # 2026-11-16 累积量跟踪: 期望转角/位移(指令积分) vs 实际累积, 固定对称容差
    ("c_yaw_track", 0.0),
    ("c_dist_track", 0.0),
    # 11-16 v7: 梯度型跟踪奖励（煜辰要的「越接近越奖、偏差越小罚越少」）。
    # 与上面 c_* 的「死区+饱和」不同：这两项是连续奖励 1->0，误差 0 给满分。
    ("dist_track", 0.0),
    # 2026-11-16 连续姿态惩罚: 相对起点的高度/朝向恶化，带容差、无上限
    ("posture", 0.0),
    ("yaw_track", 0.0),
]

# 主线 22 项：默认 terms，保证副本与主线逐位等价
MAINLINE_TERMS = [
    "tracking_lin_vel", "tracking_ang_vel", "yaw_drift", "overspeed",
    "base_pos", "lin_vel_z", "ang_vel_xy", "orientation", "pose",
    "base_height", "stand_still", "torques", "action_rate", "energy",
    "termination", "feet_air_time", "feet_clearance", "step_bonus",
    "feet_height", "feet_slip", "over_support", "dof_pos_limits",
]
COST_TERMS = ["c_orientation", "c_joint_pos", "c_joint_accel",
              "c_body_impulse", "c_body_slip", "c_self_collision",
              "c_action_diff", "c_joint_speed", "c_torque", "c_height",
              "c_area", "c_stab", "c_boot", "c_height_delta", "c_fall",
              "c_omega", "c_lin_vel", "c_power", "c_foot_clearance",
              "c_foot_slip", "c_yaw_track", "c_dist_track"]


def term_names():
    return [n for n, _ in TERM_SPECS]


def default_scales():
    return {n: w for n, w in TERM_SPECS}


def compute_rewards(env, data, action, info, done, first_contact, contact,
                    foot_pos, foot_vel):
    r = all_rewards(env, data, action, info, done, first_contact, contact,
                    foot_pos, foot_vel)
    names = list(env._config.reward_config.get("terms", term_names()))
    missing = [n for n in names if n not in r]
    if missing:
        raise ValueError("reward_config.terms 里有未注册的项: %s" % missing)
    return {k: r[k] for k in names}


def _quat_yaw(q):
    """与骨架 joystick_lee2019.py 同式（v31，世界系偏航）。"""
    w, x, y, zz = q[0], q[1], q[2], q[3]
    return jp.arctan2(2.0 * (w * zz + x * y), 1.0 - 2.0 * (y * y + zz * zz))


def _term_tracking_lin_vel(env, cmd, linvel):
    # novy: 只跟踪前进 vx(侧移已删除)
    # v24 2026-09-15: 高斯误差核 -> 比例型(单变量: 只改形状, 权重不动)。
    # 动机: 高斯核下"站着"白拿 exp(-cmd²/σ)(cmd=0.3, σ=0.25 时≈0.70),
    # 而真走路要付 slip/torque/clearance 等罚 —— v10~v23 十五个版本全部
    # 收敛到"蹲着不动"(实测摆腿 0.000m), return/0摔 指标对此全盲。
    # 比例型: 站 -> 0 奖; 同方向走满 cmd -> 1; 反向/不动 -> 0。
    # |cmd|<=0.1 维持旧高斯核(站立工况与 stand_still 语义一致)。
    # v41 2026-11-01 (c40w9): 单侧比例型 -> **对称**比例型（单变量：只改形状，权重不动）。
    # 动机: 旧式 clip(v/a, 0, 1) 对超速完全免疫 —— 跑 2a 也拿满分。
    #   于是 cmd ∈ [0.3, 1.0] 时 PPO 收敛到"一套万能步态": 实测四档指令
    #   (0.3/0.5/0.8/1.0) 全是 v≈0.6, 拟合 v = -0.044·cmd + 0.652 斜率≈0。
    # 新式: 1 - |v - a| / |a|   —— 跑快跑慢同罚。
    #   站着(v=0) -> 0，保留 v24 防"蹲着白拿分"的成果；
    #   走满(v=a) -> 1；2a -> 0；1.5a -> 0.5。
    # ⚠️ 别退回高斯核: 高斯在 v=0, a=0.3 时白给 exp(-0.36)=0.70，
    #   v10~v23 十五个版本全部因此收敛到蹲着不动。
    a = cmd[0]
    # v44 (c40w12): 分母 = max(floor, slope*|a|)，照抄论文 §4.1。
    # 开局 floor=0.3/slope=0 是宽带宽(先学出能用的步态)，后期 floor=0.05/slope=0.2 收紧。
    # floor=0/slope=1 等价旧行为。
    _den = jp.maximum(env._config.reward_config.track_floor,
                      env._config.reward_config.track_slope * jp.abs(a))
    _den = jp.maximum(_den, 0.02)          # 防除零
    err = (linvel[0] - a) / _den
    # 11-16 v4 煜辰拍板「没追踪好就罚」: 原 clip(1-|err|, 0, 1) 只奖不罚 ——
    # 站着 v=0,a=0.5 -> err=-1 -> clip(0,0,1)=0, 放弃奖励但从不倒扣, 于是站住是最优。
    # 改为对称罚 1-2|err|: err=0(完美)->+1; 0.5(跟一半)->0; 1(完全没动)->-1。
    # 站着 = -1*8.0*dt = -0.16/步, 走路 = +0.16/步, 净差 0.32/步, 梯度指向动起来。
    # 11-16 v6: 对称罚会让每步为负 -> 策略学「早摔止损」(v4 六 ckpt 摔倒步 13->11 递减,
    # 连 stand 命令也摔)。改回只奖不罚: 站着 0, 走满 +1, 摔 -0.04 最差。
    # 11-17 煜辰: 与 dist_track 同构 —— 相对误差 + 容差，追踪越好分越高、不好也不罚。
    #   |err| <= tol -> 1.0；|err| = 1 -> 0（完全没动）；>1 -> 0（走反也不罚）。
    # 容差 0.25: EMA 速度本身有抖动，25% 以内不扣分，免得把正常步态压死。
    _tol_v = float(env._config.reward_config.get("track_tol", 0.25))
    # 11-17 v15 煜辰: 满分带(tol)与衰减斜率(sharp)解耦 —— 只改斜率不改容差。
    #   ratio = 1 - sharp*max(|err|-tol,0)/(1-tol)，sharp=1 与旧式逐值等价。
    #   |err|<=tol 恒 1.0（容差不动）；超出后按 sharp 倍速掉到 0。
    _sharp_v = float(env._config.reward_config.get("track_sharp", 1.0))
    ratio = jp.clip(1.0 - _sharp_v * jp.maximum(jp.abs(err) - _tol_v, 0.0)
                    / max(1.0 - _tol_v, 1e-6), 0.0, 1.0)
    # v45: 站立工况用独立带宽(见 config.stand_sigma_lin), 行走工况不受影响
    kernel = jp.exp(-jp.square(a - linvel[0])
                    / env._config.reward_config.stand_sigma_lin)
    # 11-16 walk 白分修复: |vx_cmd|<=0.1(纯转向/站立)时线速度命令本应为 0,
    # 站着不动不该白拿 kernel(exp(0)=1)*8.0*dt。给 0 = 该项不参与, 不给满分也不罚。
    # 仅 all_rewards(walk) 调用本函数, p1/s2stand 走 cost_mode 不受影响。
    return jp.where(jp.abs(a) > 0.1, ratio, 0.0)


def _term_tracking_ang_vel(env, cmd, gyro):
    # novy: wz 现在是 cmd[1]。v24: 同 lin, 比例型。
    a = cmd[1]
    # 11-16 v4: 同 lin 改对称罚。err=(w-a)/|a|; err=0->+1, 0.5->0, 1(不转)->-1。
    # 原式 gyro/a 是单侧比例型(不转给 0 不倒扣), 且对负 a 符号易错, 一并统一成对称式。
    _aden = jp.maximum(jp.abs(a), 0.1)
    _aerr = (gyro[2] - a) / _aden
    # 11-16 v6: 同 lin, 恢复只奖不罚(站着 0), 避免自杀陷阱。
    # 11-17 煜辰: 同 lin，相对误差 + 容差，不好也不罚。
    _tol_a = float(env._config.reward_config.get("track_tol", 0.25))
    # 11-17 v15: 与 lin 同构，同一 track_sharp 控制斜率，容差带宽度仍由 track_tol 决定。
    _sharp_a = float(env._config.reward_config.get("track_sharp", 1.0))
    ratio = jp.clip(1.0 - _sharp_a * jp.maximum(jp.abs(_aerr) - _tol_a, 0.0)
                    / max(1.0 - _tol_a, 1e-6), 0.0, 1.0)
    # v45: 同上, 站立用独立带宽
    kernel = jp.exp(-jp.square(a - gyro[2])
                    / env._config.reward_config.stand_sigma_ang)
    # 11-16 walk 白分修复: |wz_cmd|<=0.1(直行/站立)时角速度命令为 0, 站着不转
    # 不该白拿 kernel*2.0*dt。给 0 = 该项不参与。仅 walk 调用。
    return jp.where(jp.abs(a) > 0.1, ratio, 0.0)


def _term_base_pos(env, cmd, qpos, pos0):
    """v45 2026-11-01 (c40w13): 世界系位置偏离起点 —— "原地"的硬约束。

    为什么它不能由 tracking_lin_vel 代劳: 那一项吃的是【EMA 速度】,
    踏步时机身前后来回晃, EMA 会把它平均成 ≈0, 于是"慢慢走开 0.5m"
    在速度上完全看不出来。而"原地踏步"要的恰恰是位置不漂。

    只对 |cmd|<0.1 生效: 行走任务那里 cmd>0.1, gate=0, 本项恒 0 ——
    不会影响 c40w10 ~ c40w12 那套走路奖励的语义。
    返回【距离】(m), 由 scales 的负权重变成惩罚。
    """
    gate = (jp.linalg.norm(cmd[:2]) < 0.1).astype(jp.float32)
    d = jp.array(qpos[0:2]) - pos0
    return jp.sqrt(jp.sum(jp.square(d)) + 1e-6) * gate


def _term_yaw_drift(env, cmd, qpos, yaw0):
    """v42 2026-11-01 (c40w10): 不转弯时惩罚【累计偏航偏离】。

    为什么需要: c40w8 开 world_frame_tracking 后, c40w9 又把它改成对称比例型
    (超速也罚), 于是策略找到新套利 —— **不降速**, 改用每 18s 漂 74° 的慢速偏航
    把世界系前向投影压低, 假装跟上了低 cmd (实测机体系 v 恒定 0.575, v_fwd 从
    0.66 掉到 0.17)。慢漂移的平均 wz 只有 0.073 rad/s, 高斯角速度核
    exp(-0.073²/0.25)=0.979 几乎不罚 -> 0.02 的成本换 0.45 的收益。

    为什么是"累计偏航"而不是"角速度": 步态摇摆使瞬时 wz 的 std 达 0.59,
    EMA 后约 0.09, 与漂移量 0.073 同量级, 用 wz 根本分不开; 而累计偏航
    对正负抵消的摇摆不敏感、只对单向漂移敏感。

    gate: |wz_cmd| < 0.1 时才罚 —— 转弯工况本来就要改朝向。
    """
    yaw = _quat_yaw(jp.array(qpos[3:7]))
    dy = jp.arctan2(jp.sin(yaw - yaw0), jp.cos(yaw - yaw0))   # wrap 到 [-pi, pi]
    gate = 1.0 - jp.clip(jp.abs(cmd[1]) / 0.1, 0.0, 1.0)
    # v46: 死区内的摆头不算漂移（见 yaw_drift_dead 注释）
    _dead = float(env._config.reward_config.get("yaw_drift_dead", 0.0))
    dy_eff = jp.maximum(0.0, jp.abs(dy) - _dead)
    val = jp.exp(-jp.square(dy_eff) / env._config.reward_config.yaw_drift_sigma)
    # 11-16 v4 遗留修复(煜辰拍板): 原式 gate*val+(1-gate)*0.25 是【只奖不罚】——
    # 直行时零偏航 val=1 拿满 0.16/步, 站着不动天然零偏航, 于是变成"站着奖励",
    # 这正是 v2 策略站住不走的第二大来源(白分审计: 直行零动作 0.16 全来自本项)。
    # 改为【只罚不奖】pen = val - 1 ∈ [-1, 0]:
    #   零偏航 -> 0(基线, 不奖励"什么也没做"); 偏航越大 -> 越接近 -1(罚)。
    # 与 tracking 的对称罚互补: tracking 罚"没走", 本项罚"走歪了"。
    # 转弯工况 gate=0 -> 返回 0(该项不参与), 不再保留 0.25 兜底(转向梯度由
    # tracking_ang_vel 的对称罚负责)。
    return gate * (val - 1.0)


def _term_dist_track(env, cmd, info):
    """相对位移跟踪奖励（世界系位置真值，无 EMA 延迟，无饱和）。

    r = 1 - 误差/期望行程（相对量）：
      跟到期望位置 -> +1.0；站着没走 -> 0.0；走反/偏离 -> 0.0（不罚）。
    用相对量而非绝对量（煜辰 2026-11-16）：不同幅度指令的绝对误差差 3.5 倍，
    策略会只顾大指令；除以期望行程后两者对等。
    ★ 下界截 0（11-17）：追踪越好分越高，不好也不罚。上界不必截（天然 <=1）。
      注意这里的 clip 不制造"死区"——它在 0 处仍然连续，梯度只在 e>=span 后归零，
      而那时本来也无分可得，不存在"卡住不动就停"的收益。
    gate: |vx_cmd| <= 0.1（纯转向回合）不参与 —— 此时期望位移本就为 0。
    span 下限 0.2m：回合最初 0.4s 内期望行程很小，比值噪声大。
    """
    gate = (jp.abs(cmd[0]) > 0.1).astype(jp.float32)
    p0 = jp.asarray(info["trk_pos0"])
    p_now = jp.asarray(info["trk_pos_now"])
    # 2026-11-17 walk: 净位移棘轮。
    # 只对「本拍超过历史最远点的新增量」付费：
    #   站着/原地踏步 -> progress=0；抽动 1cm 后再停 -> 不再继续得分；
    #   只有持续向前开拓新最远点才有分，避免早期 ratio 被一次抽动收割。
    _d_now = jp.linalg.norm(p_now - p0)
    _d_prev = jp.asarray(info["trk_dmax_prev"])
    _step = max(float(env._config.reward_config.get("c_progress_step", 0.02)),
                1e-6)
    _progress = jp.maximum(_d_now - _d_prev, 0.0)
    return jp.clip(gate * _progress / _step, 0.0, 1.0)


def _term_yaw_track(env, cmd, info):
    """梯度型转角跟踪奖励：同上，读 yaw 真值，wrap 到 [-pi,pi)。"""
    gate = (jp.abs(cmd[1]) > 0.1).astype(jp.float32)
    y0 = jp.asarray(info["trk_yaw0"])
    y_cmd = y0 + jp.asarray(info["trk_ycmd"])
    y_now = jp.asarray(info["trk_yaw_now"])
    d = y_now - y_cmd
    d = jp.mod(d + jp.pi, 2.0 * jp.pi) - jp.pi
    e = jp.abs(d)
    # 相对量（同 dist_track）：转对 +1 / 不转 0 / 转反 0（不罚）。
    # 下限 5deg：回合最初转向指令很小时比值噪声大。
    # ★上界 pi：角度误差 wrap 到 [-pi,pi)，e 天然 ≤ pi；但期望转角可累积到
    #   几十 rad。若 span 不封顶，「完全不转」的相对误差会趋近 0、白拿满分。
    #   span 封顶 pi 后：不转 -> e=pi, span=pi -> ratio=0（正确）。
    _span_raw = jp.abs(jp.asarray(info["trk_ycmd"]))
    _ok = (_span_raw > 0.0873).astype(jp.float32)
    span = jp.clip(_span_raw, 0.0873, jp.pi)
    # 11-17 煜辰: 同 dist, 下界截 0。这条尤其重要 —— span 被钳到 >=0.0873
    # 而 e<=pi, 转反时比值可到 -34 (x 6 x dt = -4.07/步), 比 termination 还重。
    return jp.clip(gate * _ok * (1.0 - e / span), 0.0, 1.0)


def _term_posture(env, z, up_z, info):
    """连续姿态恶化惩罚（相对本回合起点，带容差，无饱和）。

    posture = max(0, (z0-tol_z)-z)/s_z + max(0, (uz0-tol_u)-uz)/s_u
    站着不动 = 0；蹲 3cm 或歪 0.1 各贡献 1.0；越差线性增长，永不饱和。
    配负权重使用（这是"越差越大"的量）。
    """
    cfg = env._config.reward_config
    z0 = jp.asarray(info["body_z0"])
    uz0 = jp.asarray(info["up_z0"])
    # 容差取初始值的 5%（煜辰 23:57）：z0=0.147 -> 7.4mm；up_z0=1 -> 0.05
    _tol_r = float(cfg.get("post_tol_ratio", 0.05))
    s_z = max(float(cfg.get("post_scale_z", 0.030)), 1e-6)
    s_u = max(float(cfg.get("post_scale_u", 0.10)), 1e-6)
    e_z = jp.maximum(z0 * (1.0 - _tol_r) - z, 0.0) / s_z
    e_u = jp.maximum(uz0 * (1.0 - _tol_r) - up_z, 0.0) / s_u
    return e_z + e_u


def _term_overspeed(env, cmd, v_fwd):
    """v46 2026-11-02 (c40w18): 只罚"跑快了"，不罚"跑慢了"。

    背景: c40w17 的 cmd∈[0.20,0.55] 实测 0.26/0.39/0.51/0.55/0.56 ——
    全程单调但低档过冲 +0.06~0.11(跟踪率 1.27~1.31)。对称核
    ratio = 1-|v-a|/|a| 在 a=0.3 时 v 要偏 0.3 才掉 30%，梯度太软；
    而这条步态自身有 ≈0.14 的速度下限（线性拟合截距），不加罚就永远过冲。

    v_fwd 用 tracking 同一个 EMA 前向投影（世界系投影到初始航向），
    不用瞬时速度 —— 否则起步/落地冲击会被罚成"不敢迈步"。
    """
    a = cmd[0]
    floor = float(env._config.reward_config.get("overspeed_floor", 0.25))
    excess = jp.maximum(0.0, v_fwd - a) / jp.maximum(jp.abs(a), floor)
    gate = (a > 0.1).astype(jp.float32)          # 站立/后退指令不适用
    return gate * jp.minimum(excess, 1.5)

# v30 2026-09-15: 外力推搡(逐行移植 Go1 joystick.py:553, 变量改名对齐 cat33:
# _torso_body_id->_base_body_id, 已存在; env.dt=ctrl_dt 两边同名)。
# 随机方向水平推力, 半正弦包络, 随机间隔; 推力写 data.xfrc_applied 由下一步
# 积分消费。critic 特权观测早就在读 xfrc_applied[base](预留接口), 一开就见。


def _term_over_support(env, info):
    """三腿支撑时长: **死区惩罚** (v20 2026-09-10, 修正 v19 的设计错误)。
    诉求: 在【保留必要平衡余量供抬腿】的前提下, **尽量减少**三腿时长。
    形状: t <= T_dead -> 0                        (死区: 不奖不罚)
          t >  T_dead -> -fall*(t - T_dead)       (只罚超出的那部分)
    ★ v19 的教训: 曾用"倒U型"(前段给 +0.05 正奖励), 结果策略**去追那个奖励**
      ⇒ **主动制造三腿支撑** ⇒ 步态变慢 33%(位移 7.15 -> 4.81m)。
      结论: **想"减少"某行为, 绝不能给正奖励**(优化器只会朝奖励大的方向走)。
      正确形式是"死区 + 纯惩罚" —— 给足必要余量, 只砍多出来的。
    与旧版(纯平方惩罚, 已被 v13 关停)区别: 旧版从 t=0 就开始罚且随平方放大
      (t=1s -> 1.0, 权重-0.1 仍达 -0.1), 训练早期站桩被罚而带偏探索(v10 位移 0.284m);
      新版有 0.2s 死区(步态切换天然需要三腿), t<=0.2s 完全不罚, 无早期带偏风险。"""
    t = info["tripod_time"].astype(jp.float32) / 50.0    # 秒
    T_dead, fall = 0.20, 0.20                            # 死区 0.2s / 超时斜率
    return -fall * jp.maximum(0.0, t - T_dead)


def _logistic_kernel(e, alpha):
    """论文 II-C: K(e|α) = -1/(e^{αe}+2+e^{-αe}) ∈ [-0.25,0)，误差越小越负。

    论文在 minimize 框架下让主项趋于 -0.25；映射到 reward 时取负号，
    于是「接近目标」给正分、上限 0.25·k。α 控制对误差的敏感度。
    """
    a = jp.maximum(float(alpha), 1e-3)
    return -1.0 / (jp.exp(a * e) + 2.0 + jp.exp(-a * e))


def paper_costs(env, data, action, info, contact, foot_vel, gravity, foot_pos):
    """Table II 的 9 项成本，返回**未加权**的 c_i（与主线奖励项同一口径）。

    标注近似的 3 项（论文用接触冲量/接触点速度/自碰撞点数，MJX 无直接量）：
      c_body_impulse ← 机身 6 DoF 加速度平方和（冲量代理）
      c_body_slip    ← 触地足端水平速度平方和
      c_self_collision ← 非地面/非足端的穿透接触对数（精简模型恒 0，权重设 0）
    """
    cfg = env._config.reward_config
    q = data.qpos[env._jnt_qpos_adr]
    if bool(env._config.task_config.get("joint_sign_mask_enable", False)):
        q = q * jp.asarray(env._config.task_config.joint_sign_mask, dtype=q.dtype)
    v = data.qvel[env._jnt_qvel_adr]
    acc = data.qacc[env._jnt_qvel_adr]
    tau = data.actuator_force[1:]
    q_sit = jp.array(cfg.q_sit)

    # c_o: 机身重力向量误差（正立时 get_gravity ≈ [0,0,-1]）
    e_o = jp.linalg.norm(jp.array(gravity) - jp.array([0.0, 0.0, -1.0]))
    c_o = _logistic_kernel(e_o, cfg.alpha_o)
    # EXP-04A：原 K(e) 在倒立时恒 -0.105 -> 不动也白拿 +4.2/回合。
    # 减去起点处 K(e0) 后，静止时 c_o=0，只奖励由自身动作带来的姿态改善。
    if bool(cfg.get("orientation_baseline", False)):
        _mode = str(cfg.get("orientation_baseline_mode", "start"))
        if _mode == "inverted":
            # 倒立时 g_body=[0,0,1] -> e=||[0,0,2]||=2；
            # 固定基线让「保持正立」每步都有持续收益，
            # 同时倒立静止仍恰好为 0（不白拿）。
            _e0 = jp.array(2.0)
        else:
            _e0 = jp.sqrt(jp.maximum(
                2.0 * (1.0 - jp.array(info["up_z0"])), 0.0))
        c_o = c_o - _logistic_kernel(_e0, cfg.alpha_o)
    # c_jp: 到目标姿态的关节角误差（论文 dφ）
    # EXP-14 (2026-11-12): 可选「只约束四髋侧摆 + 分数随 up_z 门控」。
    # 默认全关 -> 与历史配方逐位等价（仍是 12 维到 q_sit 的距离）。
    if bool(cfg.get("c_jp_hip_only", False)):
        _hip = jp.array([0, 3, 6, 9])  # ACTIVE_JOINTS 每腿第 1 个 = 髋侧摆
        _tgt = jp.array(cfg.get("c_jp_hip_target", [0.0, 0.0, 0.0, 0.0]))
        # 门控用 up_z（由 gravity 推得，与四元数口径恒等）。
        # 坡度不再修正 up_z：坡度已用于选边，再修会二次计入。
        if bool(cfg.get("c_jp_stageA_enable", False)):
            # ---- 阶段 A：倒立起始姿态，一侧髋中位 + 另一侧髋限位附近 ----
            # 侧别由 reset 时的坡度方位决定（slope_phi 每回合固定，不随姿态抖）。
            # 判据 -sin(phi) > thr 中的 thr 是 sin 值，阈值 0.0 才等于「按坡向二选一」：
            #   _tilt = 下坡方向 y 分量 ∝ sin(phi)（dr.py: axis=(-sin,cos,0)，
            #   n=(cos*sin_theta, sin*sin_theta, cos_theta)，符号一致率 100%）。
            # 旧默认 0.836 = sin(56.8°) 只覆盖 phi∈(236.8°,303.2°)（A 侧 18.5%），
            # 导致 81.5% 回合都要求 B 目标 → 策略被塑形成只会向 +y 翻（EXP-14 实测
            # 成功 283/1019 全走 +y）。0.836 疑为「偏差容限」误当 sin 阈值，2026-11-13 修正。
            _sa_u = 1.0 - 0.5 * e_o * e_o
            _sa_dr = info.get("exp08_dr", {})
            _sa_phi = jp.array(_sa_dr.get("slope_phi", 0.0))
            _sa_a = (-jp.sin(_sa_phi)) > float(cfg.get("c_jp_side_thresh", 0.0))
            _sa_lo = jp.array(cfg.c_jp_hip_lo)
            _sa_hi = jp.array(cfg.c_jp_hip_hi)
            _sa_fr = abs(float(cfg.get("c_jp_stageA_frac", 0.85)))
            if str(cfg.get("c_jp_stageA_mode", "prop")) == "abs":
                _sa_lm = -_sa_fr * jp.ones(4)
                _sa_lp = _sa_fr * jp.ones(4)
            else:  # prop: 各髋按自身行程同比例，前后腿都不顶限位
                _sa_lm = _sa_lo * _sa_fr
                _sa_lp = _sa_hi * _sa_fr
            # q[_hip] 真实顺序 = [后-y(joint1), 后+y(joint6), 前-y(joint11), 前+y(joint16)]
            # （2026-11-13 核对 active_joints[0,3,6,9] 的 body_pos y 符号；旧注释写反了，
            #   数组本身一直是对的：A 让 +y 侧前后髋到负限位，B 让 -y 侧到正限位。）
            # 实测对照：EXP-14 中 81.5% 回合选 B -> 成功 283 条 100% 走 +y 翻，与该映射自洽。
            _sa_ta = jp.array([0.0, _sa_lm[1], 0.0, _sa_lm[3]])  # A: +y 侧下限位
            _sa_tb = jp.array([_sa_lp[0], 0.0, _sa_lp[2], 0.0])  # B: -y 侧上限位
            _sa_tgt = jp.where(_sa_a, _sa_ta, _sa_tb)
            _sa_qh = q[_hip]
            # 阶段 A 误差：四髋到目标姿态的平均绝对偏差
            _sa_ea = jp.mean(jp.abs(_sa_qh - _sa_tgt))
            # 阶段 B 误差：|q| 最大的两髋均值。中位侧那两髋天然接近 0，
            # 不参与评分，避免「一侧已在中位 -> 白拿满分」。
            _sa_eb = jp.mean(jp.sort(jp.abs(_sa_qh))[2:])
            _sa_ra = max(float(cfg.get("c_jp_ref_a", 0.5)), 1e-6)
            _sa_rb = max(float(cfg.get("c_jp_ref_b", 0.7)), 1e-6)
            # 线性有界核：K 核在 e>0.6 近乎饱和，而本阶段误差尺度是 0~1。
            _sa_ca = jp.clip((_sa_ea - _sa_ra) / _sa_ra, -1.0, 1.0)
            _sa_cb = jp.clip((_sa_eb - _sa_rb) / _sa_rb, -1.0, 1.0)
            _sa_z1 = float(cfg.get("c_jp_up_z1", -0.99577))
            _sa_z3 = float(cfg.get("c_jp_up_z3", -0.89471))
            _sa_dec = max(float(cfg.get("c_jp_gate_decay", 0.5)), 1e-6)
            # A 从 up_z1 到 up_z3 连续退场（避免硬切换自激）；
            # B 从 up_z3 进场，过 0 后按 decay 衰减，不干预保持正立。
            _sa_wa = 1.0 - jp.clip(
                (_sa_u - _sa_z1) / max(_sa_z3 - _sa_z1, 1e-6), 0.0, 1.0)
            _sa_wb = jp.clip((_sa_u - _sa_z3) / max(-_sa_z3, 1e-6), 0.0, 1.0)
            _sa_wb = _sa_wb * (1.0 - jp.clip(_sa_u / _sa_dec, 0.0, 1.0))
            c_jp = float(cfg.get("c_jp_stageA_gain", 0.15)) * (
                _sa_ca * _sa_wa + _sa_cb * _sa_wb)
        else:
            # 度量选择：实测 rms 在过槛段反而比倒立段大（方向反），
            # min2_abs = |q_hip| 最小的两髋均值，对应「触地侧两髋在中位」。
            _mode = str(cfg.get("c_jp_hip_mode", "min2_abs"))
            _dq = jp.abs(q[_hip] - _tgt)
            if _mode == "min2_abs":
                e_jp = jp.mean(jp.sort(_dq)[:2])
            elif _mode == "mean_signed":
                e_jp = jp.abs(jp.mean(q[_hip] - _tgt))
            elif _mode == "mean_abs":
                e_jp = jp.mean(_dq)
            else:  # rms
                e_jp = jp.sqrt(jp.mean(jp.square(q[_hip] - _tgt)) + 1e-8)
            c_jp = _logistic_kernel(e_jp, cfg.alpha_jp)
            # 减基线：e >= c_jp_ref 时不再白拿分，给出有符号的收髋引导。
            c_jp = c_jp - _logistic_kernel(float(cfg.get("c_jp_ref", 0.6)),
                                           cfg.alpha_jp)
            # 门控输入由 gravity 推出（e_o^2 = 2(1-up_z)，与四元数口径恒等）。
            _u = 1.0 - 0.5 * e_o * e_o
            if bool(cfg.get("c_jp_slope_fix", False)):
                _dr = info.get("exp08_dr", {})
                _th = jp.array(_dr.get("slope_theta", 0.0))
                _u = _u - (1.0 - jp.cos(_th))  # 倒立基准 -1 -> -cos(theta)
            _lo = float(cfg.get("c_jp_gate_lo", -0.8))
            _hi = float(cfg.get("c_jp_gate_hi", 0.0))
            _dec = max(float(cfg.get("c_jp_gate_decay", 0.5)), 1e-6)
            _gate = jp.clip((_u - _lo) / max(_hi - _lo, 1e-6), 0.0, 1.0)
            _gate = _gate * (1.0 - jp.clip((_u - _hi) / _dec, 0.0, 1.0))
            c_jp = c_jp * _gate
    else:
        e_jp = jp.sqrt(jp.sum(jp.square(q - q_sit)) + 1e-8)
        c_jp = _logistic_kernel(e_jp, cfg.alpha_jp)
        # 2026-11-13（Stage-2 改写）：关 hip_only 时也必须减基线。
        # K(e|α)∈[-0.25,0) 随误差单调**上升**，cost_mode 取负后等于
        # 「离目标越远奖励越多」——实测零动作白拿 +0.018/拍（500 拍 +9），
        # 与 EXP-08 s1 的吸收态同型。扣掉 K(c_jp_ref) 后 c_jp∈[0,0.25] 纯惩罚，
        # 与论文 dφ(φ_j,φ̂_j)≥0 同向。开关默认关（历史配方逐位不变）。
        if bool(cfg.get("c_jp_kernel_baseline", False)):
            c_jp = c_jp - _logistic_kernel(float(cfg.get("c_jp_ref", 0.0)),
                                           cfg.alpha_jp)
    # c_a: 关节加速度
    c_a = jp.sum(jp.square(acc))
    # c_bi: 非足端接触冲量（代理：机身加速度平方和）
    c_bi = jp.sum(jp.square(data.qacc[0:6]))
    # c_bs: 接触点滑移速度
    c_bs = jp.sum(jp.sum(jp.square(foot_vel[:, :2]), axis=-1) * contact.astype(jp.float32))
    # c_c,in: 自碰撞接触点数（穿透且两 geom 都不是地面/足端）
    g1, g2 = data.contact.geom1, data.contact.geom2
    near = data.contact.dist < -1e-5
    nf = (g1 != env._floor_geom_id) & (g2 != env._floor_geom_id)
    not_foot = jp.all(g1[:, None] != env._foot_geom_ids[None, :], axis=1) & \
               jp.all(g2[:, None] != env._foot_geom_ids[None, :], axis=1)
    c_cin = jp.sum((near & nf & not_foot).astype(jp.float32))
    # c_ad: 相邻拍动作差（step 里 last_last_act 更新后正好是上一拍动作）
    c_ad = jp.sum(jp.square(action - jp.array(info["last_last_act"])))
    # c_jslim: 超过实测关节限速的部分（台架标定 (4.9,4.9,2.58)×4）
    vlim = jp.array(env._config.max_motor_velocity)
    c_jslim = jp.sum(jp.square(jp.maximum(jp.abs(v) - vlim, 0.0)))
    # c_τ: 力矩平方
    c_tau = jp.sum(jp.square(tau))
    # c_h: 机身高度罚（论文 Table II: c_h = 1 if h < 0.35 else 0；0.35 是 ANYmal ~0.5m 站高的 70%）。
    # 本机零位站姿 base z = 0.1473（keyframe stand 沉降 600 步实测），取 78% = 0.115：
    # 姿态库起点 body_z 中位 0.079 / 95% 分位 0.109 全部计入惩罚，站起后 0.147 免罚。
    # c_height_thr <= 0 时该项恒 0（默认关闭，历史配方逐位不变）。
    _h_thr = float(cfg.get("c_height_thr", 0.0))
    if bool(cfg.get("c_height_hinge_enable", False)):
        # 铰链: 越接近零位站高罚越轻, 抬身全程有梯度 (二值只在阈值处跳变)
        _h_ref = float(cfg.get("c_height_ref", 0.1473))
        c_h = jp.clip((_h_ref - data.qpos[2]) / _h_ref, 0.0, 1.0)
    elif _h_thr > 0.0:
        c_h = (data.qpos[2] < _h_thr).astype(jp.float32)
    else:
        c_h = jp.zeros((), dtype=jp.float32)
    # ---- 2026-11-14 Stage-2 v2: 四足端地面投影几何约束 (默认全关) ----
    # 去掉 contact mask: 足端不触地也计入 (煜辰口径)。
    # 极角排序: 倒地时腿会左右交叉, 固定腿序会自相交使 shoelace 面积互相抵消。
    c_area = jp.zeros((), dtype=jp.float32)
    c_stab = jp.zeros((), dtype=jp.float32)
    _w_cross = jp.zeros((), dtype=jp.float32)
    c_boot = jp.zeros((), dtype=jp.float32)
    # 2026-11-21 v14: 支撑几何只统计"着地足"的开关（默认关，关时逐位不变）
    _geo_con = False
    _w_n = jp.ones((), dtype=jp.float32)
    _aref_scale = jp.ones((), dtype=jp.float32)
    _geo_on = bool(cfg.get("c_area_enable", False)) or bool(cfg.get("c_stab_enable", False))
    if _geo_on and foot_pos is not None:
        _fp = jp.asarray(foot_pos)
        _p = _fp[:, :2]                                  # (4,2) 地面投影
        # ---- 2026-11-21 v14: 只让着地足参与几何（默认关）----
        # 动机（v13 实测）：四足 site 不论是否着地都计入时，抬一条腿把它的投影往外推
        # → 四边形面积与质心内距同时变大 → c_area/c_stab（回合占比 67%）都加钱。
        # v13 策略因此稳定收敛到"三足支撑、一条腿翘起"。
        # 做法：非着地足的投影搬到第一个着地足上（重复顶点，shoelace 贡献为 0），
        # 形状仍是 (4,2)，不引入动态形状；接触数 < min_contacts 时整项门控到 0。
        _geo_con = bool(cfg.get("c_geo_contact_only", False)) and contact is not None
        _ncon = jp.asarray(4.0, dtype=jp.float32)
        if _geo_con:
            _cf = jp.asarray(contact) > 0.5
            _ncon = jp.sum(_cf.astype(jp.float32))
            _pick = jp.argmax(_cf.astype(jp.int32))   # 第一个着地足（全 False 时=0）
            _p = jp.where(_cf[:, None], _p, _p[_pick][None, :])
            _w_n = jp.where(_ncon >= float(cfg.get("c_geo_min_contacts", 3)), 1.0, 0.0)
            # 三足支撑的参考面积：站立四边形沿对角线剖出的三角形 = 站立面积的一半
            _aref_scale = jp.where(_ncon >= 4.0, 1.0,
                                   float(cfg.get("c_area_ref3_ratio", 0.5)))
        _pc = jp.mean(_p, axis=0)
        _ang = jp.arctan2(_p[:, 1] - _pc[1], _p[:, 0] - _pc[0])
        _ps = _p[jp.argsort(_ang)]                      # 环形序, 无自相交
        _pn = jp.roll(_ps, -1, axis=0)
        _A = 0.5 * jp.abs(jp.sum(_ps[:, 0] * _pn[:, 1] - _pn[:, 0] * _ps[:, 1]))

        # 左右交叉门控: 在机体系量左右两组的 y 间距。
        # foot_pos 行序 = consts.FEET_SITES = [foot_fl(右前), foot_fr(左前), foot_rl(右后), foot_rr(左后)]
        # -> {1,3} 是左侧前后, {0,2} 是右侧前后; 机体系 +y 朝左, 故 dy = 左-右 > 0。
        # ⚠️ 勿与 joystick_lee2019.py 的 _odom_foot_site_id(里程计腿序 [rl,rr,fl,fr]) 混淆;
        #    site 名与物理左右错位(fl=右前), 依据见 cat40_constants.py 头注。
        # 2026-11-14 实测(stand keyframe): dy=+0.1285 -> w_cross=1, 左侧 y=+0.065 右侧 y=-0.064。
        # 交叉/贴合时四边形退化, 质心投影与面积都失去意义 -> 平滑门控到 0。
        _R = data.site_xmat[env._imu_site_id]          # (3,3) body->world
        _pb = (_fp - data.site_xpos[env._imu_site_id]) @ _R
        _dy = 0.5 * (_pb[1, 1] + _pb[3, 1]) - 0.5 * (_pb[0, 1] + _pb[2, 1])
        _dy_thr = float(cfg.get("c_cross_dy_thr", 0.0257))
        _w_cross = jp.clip(_dy / _dy_thr, 0.0, 1.0)

        if bool(cfg.get("c_area_enable", False)):
            _Aref = float(cfg.get("c_area_ref", 0.040915)) * _aref_scale
            if bool(cfg.get("c_area_two_sided_enable", False)):
                # 2026-11-15 v6: 两侧都罚 —— 实测站正时面积/基准只有 0.866（腿收在身下），
                # 旧的"只罚超标"永远不会触发。改为罚"偏离站姿面积"。
                _sca = float(cfg.get("c_area_dev_scale", 0.30))
                # 2026-11-20 v11：上侧不再固定饱和在 1。旧公式 clip(|r-1|/s, 0, 1)
                # 在 r>1+s(默认 1.6) 后梯度为 0，实测 v10 摊到 2.59 倍后仍然继续摊。
                # 这里把上侧上限做成可配 c_area_hi_cap（默认 1.0 = 与旧公式逐位一致）。
                _r = _A / _Aref
                _cap_hi = float(cfg.get("c_area_hi_cap", 1.0))
                _dev = jp.where(_r < 1.0,
                                jp.clip((1.0 - _r) / _sca, 0.0, 1.0),
                                jp.clip((_r - 1.0) / _sca, 0.0, _cap_hi))
                c_area = _dev * _w_cross
            else:
                _thr = float(cfg.get("c_area_thr_ratio", 1.2))
                _norm = float(cfg.get("c_area_norm", 0.2))
                c_area = jp.minimum(
                    jp.square(jp.maximum(0.0, _A / _Aref - _thr) / _norm),
                    float(cfg.get("c_area_cap", 1.0))) * _w_cross
        if bool(cfg.get("c_stab_enable", False)):
            # 质心沿重力方向投影到地面; 水平地面时即 xy。
            _com = data.subtree_com[1][:2]
            _e = _pn - _ps
            _n = jp.stack([_e[:, 1], -_e[:, 0]], axis=-1)
            _n = _n / (jp.linalg.norm(_n, axis=-1, keepdims=True) + 1e-8)
            _inner = jp.sum(_n * (_pc[None, :] - _ps), axis=-1)
            _sgn = jp.where(_inner > 0.0, -1.0, 1.0)   # 翻成外法向
            _d = jp.sum(_n * (_com[None, :] - _ps), axis=-1) * _sgn  # >0 在外
            # 2026-11-21 v14: 重复顶点（非着地足被搬到着地足上）会形成零长度边，
            # 其法向为 0 -> _d 恒为 0 -> 把 max(_d) 抬到 0, 内距被误算成 0（相当于
            # "刚好在边界上"，三足支撑会被无端罚 0.2 raw）。这里把退化边排除。
            # 无重复顶点时 _valid 全 True，结果与旧实现逐位一致。
            _valid = jp.linalg.norm(_e, axis=-1) > 1e-6
            _d = jp.where(_valid, _d, jp.asarray(-1e9, dtype=_d.dtype))
            _m = -jp.max(_d)                            # 到最近边的内距
            _mt = float(cfg.get("c_stab_margin", 0.020))
            if bool(cfg.get("c_stab_linear_enable", False)):
                # 2026-11-15 v6: 平方+cap 在质心出多边形时必然饱和成常数 1.0（无梯度）。
                # 改线性: 实测 margin -46.5mm、scale=0.10 时给 0.67，-80~+20mm 全程可导。
                _scs = float(cfg.get("c_stab_scale", 0.10))
                c_stab = jp.clip((_mt - _m) / _scs, 0.0, 1.0) * _w_cross
            else:
                c_stab = jp.minimum(
                    jp.square(jp.maximum(0.0, _mt - _m) / _mt),
                    float(cfg.get("c_stab_cap", 1.0))) * _w_cross

    # ---- 2026-11-15 v6b: 稳改"奖励"形式 (默认关, 历史逐位不变) ----
    # 原始 raw ∈ [0,1]，0=最稳、1=最不稳。取 raw-1 ∈ [-1,0]，
    # cost_mode 取负后 reward = w·(1-raw)·dt > 0：越稳拿得越多，不稳拿 0（不扣分）。
    # 门控仍要用未平移的 raw，否则 max(floor, 1-raw) 全被抬成 1。
    # Invalid/crossed geometry has zero support quality.
    _stab_raw = (c_stab + (1.0 - _w_cross)
                 if bool(cfg.get("c_stab_reward_mode", False)) else c_stab)
    if bool(cfg.get("c_stab_reward_mode", False)):
        c_stab = c_stab - _w_cross
    if bool(cfg.get("c_area_reward_mode", False)):
        c_area = c_area - _w_cross

    # 2026-11-21 v14：接触门控必须在"平移成奖励"之后乘，否则 n<3 时
    # c_stab*0 - _w_cross 会变成 +_w_cross 的正奖励（反向激励）。
    if _geo_con:
        c_area = c_area * _w_n
        c_stab = c_stab * _w_n
        # 支撑不足时不算“稳”，姿态项不罚（与 v11 的 gating 语义一致）
        _stab_raw = jp.where(_w_n > 0.5, _stab_raw, 1.0)

    # 2026-11-20 v11：姿态追踪与高度“同级”，但只在该稳的时候要求。
    # _stab_raw∈[0,1]（0=最稳），乘 (1-_stab_raw) 表示“站得稳才罚姿态偏差”，
    # 站不稳时先允许姿态乱（否则会把‘还没站稳就先收敛姿态’变成反向激励）。
    if str(cfg.get("c_jp_gate_mode", "height")) == "stab":
        c_jp = c_jp * (1.0 - _stab_raw)

    # ---- 2026-11-18 Stage-2 v9: 站姿奖按"高度资格"缩放 (默认关) ----
    # 诊断 (audit_s2_bank_terms.py)：reward_mode 下 c_stab/c_area 是
    # w_cross*(1-raw) in [0,1]，与身高无关 —— 任意四足摊开的低蹲都拿满
    # 0.16/拍 (约 40/回合)，而"把腿伸直"只有 c_height_delta 的一次性 ~6 分。
    # 反向起点库里 60% 是 0.10~0.13 m 的过渡蹲姿，静止即满额，于是策略学会
    # "蹲稳不动"（v8 实测 bz_max 0.075、stand 0）。修法：两项乘高度资格
    #   w_h = clip((bz - h0) / (H - h0), 0, 1)
    # 贴地 0.078 -> 0.04；过渡蹲 0.115 -> 0.55；站满 0.1473 -> 1.0。
    # 稳仍是必要条件 (raw 高则 0)，只是"稳"必须与"高"同时成立才付钱。
    if bool(cfg.get("c_stance_h_gate_enable", False)):
        _Hg = float(cfg.get("c_stance_h_ref", 0.1473))
        _h0 = float(cfg.get("c_stance_h_floor", 0.075))
        _wh = jp.clip((data.qpos[2] - _h0) / max(_Hg - _h0, 1e-6), 0.0, 1.0)
        c_stab = c_stab * _wh
        c_area = c_area * _wh

    # ---- 2026-11-14 P1 r3: 结构门控 + 起身棘轮增量势 (默认全关) ----
    # 只在起点髋侧摆 s0 偏低(不动/少动域)的回合生效; w_s 由 reset 冻结, 不随姿态抖。
    # 棘轮只认 up_z 历史新高: 停在半程与反复抬-退-抬都拿不到额外分(回放 AUROC 0.998)。
    # c_boot 为负 -> cost_mode 取负后给正 reward; 除以 dt 使 reward 增量 = k*w_s*w_u*dPhi。
    if bool(cfg.get("c_boot_enable", False)):
        _bs_lo = float(cfg.get("c_boot_s_lo", -0.62))
        _bs_hi = float(cfg.get("c_boot_s_hi", 0.08))
        _bu_lo = float(cfg.get("c_boot_u_lo", -0.999))
        _bu_hi = float(cfg.get("c_boot_u_hi", 0.30))
        _bws = jp.clip((_bs_hi - jp.asarray(info["boot_s0"]))
                       / max(_bs_hi - _bs_lo, 1e-6), 0.0, 1.0)
        _bphi = jp.asarray(info["boot_peak"])
        _bwu = jp.clip((_bu_hi - _bphi) / max(_bu_hi - _bu_lo, 1e-6),
                       0.0, 1.0)
        _bdphi = jp.maximum(_bphi - jp.asarray(info["boot_prev"]), 0.0)
        # 2026-11-15 v6: 以"稳"门控"正" —— 不稳时只给地板比例(默认 0.2)的起身奖励，
        # 避免"还没站稳就猛翻"白拿钱；地板不为 0 是为了防止倒立起点死锁。
        _bgate = 1.0
        if bool(cfg.get("c_boot_stab_gate_enable", False)):
            _bfl = float(cfg.get("c_boot_stab_floor", 0.20))
            _bgate = jp.maximum(_bfl, 1.0 - _stab_raw)
        c_boot = -_bws * _bwu * _bgate * _bdphi / max(float(env.dt), 1e-6)

    # ---- 2026-11-14 Stage-2 v3: c_height 的抬升增量信号 (默认全关) ----
    # 只认回合内 body_z 棘轮的新高增量: 与绝对高度无关, 低起点抬 1mm 也有即时梯度。
    # 棘轮使「抬到半程躺平」「抬-退-抬反复刷」都拿不到额外分 (P1 r3 同构形式已验证)。
    # c 为负 -> cost_mode 取负后给正 reward, reward 每拍增量 = k * dPhi。
    c_height_delta = jp.zeros((), dtype=jp.float32)
    c_fall = jp.zeros((), dtype=jp.float32)
    # up_z = 1 - 2(x^2+y^2)，与 task.up_z 同式（此处不引入 task 依赖）
    _uz_now = 1.0 - 2.0 * (data.qpos[4] ** 2 + data.qpos[5] ** 2)
    if bool(cfg.get("c_height_delta_enable", False)):
        _dh = jp.maximum(jp.asarray(info["hzd_peak"])
                         - jp.asarray(info["hzd_prev"]), 0.0)
        _dh = jp.minimum(_dh, float(cfg.get("c_height_delta_cap", 0.005)))
        # 2026-11-15 v5 第 1 道门: 入场资格(摆正)。没正起来时抬身不给分，
        # 避免"侧躺着把身体蹭高"也拿钱。阈值来自 c_height 原来的 up_z 门控。
        _du_lo = float(cfg.get("c_height_delta_u_lo", 0.60))
        _du_hi = float(cfg.get("c_height_delta_u_hi", 0.75))
        _w_up = jp.clip((_uz_now - _du_lo) / max(_du_hi - _du_lo, 1e-6), 0.0, 1.0)
        # 2026-11-15 v5 第 2 道门: 天花板(站高比例)。到 0.95H 开始衰减、到 H 完全失效，
        # 再高不给分 —— 防止用"顶过头/踮脚"刷增量。
        _Hd = float(cfg.get("c_height_ref", 0.1473))
        _t_lo = float(cfg.get("c_height_delta_top_lo_ratio", 0.95)) * _Hd
        _t_hi = float(cfg.get("c_height_delta_top_hi_ratio", 1.00)) * _Hd
        _w_top = 1.0 - jp.clip((data.qpos[2] - _t_lo) / max(_t_hi - _t_lo, 1e-6),
                               0.0, 1.0)
        c_height_delta = -_w_up * _w_top * _dh / max(float(env.dt), 1e-6)
    # ---- 2026-11-20 v11: 足端拖滑（全程）与 up_z 瞬变（失稳） ----
    c_foot_drag = jp.zeros((), dtype=jp.float32)
    c_upz_jerk = jp.zeros((), dtype=jp.float32)
    if bool(cfg.get("c_foot_drag_enable", False)) and foot_vel is not None:
        _v0 = float(cfg.get("c_foot_drag_v0", 0.02))
        _vsc = max(float(cfg.get("c_foot_drag_scale", 0.20)), 1e-6)
        _cf = jp.asarray(contact) > 0.5
        _sp = jp.linalg.norm(jp.asarray(foot_vel)[:, :2], axis=-1)
        # 只罚“接触中还在地上蹭”的足端；抬起换脚时水平速度不计
        c_foot_drag = jp.mean(jp.where(
            _cf, jp.clip((_sp - _v0) / _vsc, 0.0, 1.0), 0.0))
    if bool(cfg.get("c_upz_jerk_enable", False)):
        _du = jp.abs(_uz_now - jp.asarray(info["upz_prev"]))
        _j0 = float(cfg.get("c_upz_jerk_tol", 0.02))
        _jsc = max(float(cfg.get("c_upz_jerk_scale", 0.10)), 1e-6)
        # 赦免小抖动（<tol），只罚单拍内的大幅姿态翻转（失稳）
        c_upz_jerk = jp.clip((_du - _j0) / _jsc, 0.0, 1.0)

    # 2026-11-15 v5: 防摔重罚。曾经正到过 u_gate(锁定) 之后, up_z 相对峰值的
    # 回撤超过 tol 开始罚, dmax 处罚满。只罚"正过又跌回去", 不罚起始的倒立阶段。
    if bool(cfg.get("c_fall_enable", False)):
        _pk = jp.asarray(info["fall_uz_peak"])
        _latched = (_pk >= float(cfg.get("c_fall_u_gate", 0.75))).astype(jp.float32)
        _drop = jp.clip(_pk - _uz_now, 0.0, 1.0)
        _f_tol = float(cfg.get("c_fall_drop_tol", 0.20))
        _f_max = float(cfg.get("c_fall_drop_max", 0.50))
        c_fall = _latched * jp.clip((_drop - _f_tol) / max(_f_max - _f_tol, 1e-6),
                                    0.0, 1.0)

    # ---- 2026-11-15 walk: 论文 locomotion 的 tracking/功率/足端项 (默认全关) ----
    # c_omega/c_lin_vel 用 logistic 核跟踪指令；c_power 只在正功率时计；
    # 足端项按接触状态分开：触地罚滑移、摆动罚离地高度偏差。
    c_omega = jp.zeros((), dtype=jp.float32)
    c_lin_vel = jp.zeros((), dtype=jp.float32)
    c_power = jp.zeros((), dtype=jp.float32)
    c_foot_clearance = jp.zeros((), dtype=jp.float32)
    c_foot_slip = jp.zeros((), dtype=jp.float32)
    if bool(cfg.get("c_walk_enable", False)):
        _cmd = jp.asarray(info["command"])
        _alpha = float(cfg.get("c_walk_alpha", 2.0))
        _gyro = env.get_gyro(data)
        c_omega = _logistic_kernel(jp.abs(_gyro[2] - _cmd[1]), _alpha)
        _lin = jp.asarray(info["odom_linvel"])
        c_lin_vel = _logistic_kernel(jp.abs(_lin[0] - _cmd[0]), _alpha)
        c_power = jp.sum(jp.maximum(v * tau, 0.0))
        _cf = jp.asarray(contact) > 0.5
        _vf = jp.asarray(foot_vel)
        _pf = jp.asarray(foot_pos)
        _sp = jp.linalg.norm(_vf, axis=-1)
        c_foot_slip = jp.sum(jp.where(_cf, _sp, 0.0))
        _h_tgt = float(cfg.get("c_foot_clearance_h", 0.04))
        # 2026-11-16: 改成相对站姿脚心的高度（不依赖地面绝对高度，坡/台阶也成立）。
        # 原来直接用世界系 foot_z，地面抬高时目标就偏了。cat4.0 的 feet_clearance
        # 本来就是 target_z = stance_z + max_foot_height 这个形式。
        c_foot_clearance = jp.sum(
            jp.where(_cf, 0.0,
                     jp.square(_pf[:, 2] - jp.asarray(env._foot_stance_z) - _h_tgt) * _sp))

    # ---- 2026-11-16 walk 追踪: 世界系状态量 vs 指令期望状态 (默认全关) ----
    # 为什么不用瞬时量: 四足转向/前进是步进式，瞬时量与指令必然对不上。
    #   实测（w18）正确转向段瞬时角速度误差 0.52 比直走 0.46 还大（完全反了）；
    #   状态量口径下正确转向 4.0deg、走歪 127.5deg，差 32 倍。
    # 实际侧直接读 qpos 真值（trk_yaw_now / trk_pos_now），不对速度积分：
    #   状态量是状态的确定性函数，绕圈净收益为 0；增量型奖励会重复给分。
    # 指令侧才积分，生成期望航向 θ_cmd 与期望世界系轨迹 p_cmd。
    # 容差固定、左右对称、不随行程增长（增长则误差上限=最终容差，约束失效）。
    c_yaw_track = jp.zeros((), dtype=jp.float32)
    c_dist_track = jp.zeros((), dtype=jp.float32)
    if bool(cfg.get("c_track_accum_enable", False)):
        _tol_y = float(cfg.get("c_yaw_track_tol", 0.0873))   # 5 deg
        # 转角误差: 差值 wrap 到 [-pi,pi) 再取绝对值，天然左右对称，
        # 转 340deg 不会被误判成 -20deg，无需累加器记录圈数。
        _y_cmd = jp.asarray(info["trk_yaw0"]) + jp.asarray(info["trk_ycmd"])
        _dy = jp.asarray(info["trk_yaw_now"]) - _y_cmd
        _dy = jp.mod(_dy + jp.pi, 2.0 * jp.pi) - jp.pi
        _e_y = jp.abs(_dy)
        c_yaw_track = jp.clip((_e_y - _tol_y) / _tol_y, 0.0, 1.0)
        # 位移误差: 世界系 2 维欧氏距离 (A1)。走歪会同时被转角项与位移项罚，
        # 因为期望轨迹本身就是世界系的。
        _tol_d = float(cfg.get("c_dist_track_tol", 0.50))
        _p_cmd = jp.asarray(info["trk_pos0"]) + jp.asarray(info["trk_pcmd"])
        _e_d = jp.linalg.norm(jp.asarray(info["trk_pos_now"]) - _p_cmd)
        c_dist_track = jp.clip((_e_d - _tol_d) / _tol_d, 0.0, 1.0)

    out = {
        "c_orientation": c_o, "c_joint_pos": c_jp, "c_joint_accel": c_a,
        "c_body_impulse": c_bi, "c_body_slip": c_bs,
        "c_self_collision": c_cin, "c_action_diff": c_ad,
        "c_joint_speed": c_jslim, "c_torque": c_tau, "c_height": c_h,
        "c_area": c_area, "c_stab": c_stab, "c_boot": c_boot,
        "c_height_delta": c_height_delta,
        "c_foot_drag": c_foot_drag, "c_upz_jerk": c_upz_jerk,
        "c_fall": c_fall,
        "c_omega": c_omega, "c_lin_vel": c_lin_vel, "c_power": c_power,
        "c_foot_clearance": c_foot_clearance, "c_foot_slip": c_foot_slip,
        "c_yaw_track": c_yaw_track, "c_dist_track": c_dist_track,
    }
    # 2026-11-04: 论文 p4 的 bounded cost。0.5m 投落落地帧 Σqacc² 会爆到 1e7，
    # 无界时单步 reward 直接撞 clip 下界 -10000，训练被冲击帧主导（实测）。
    # K 核项恒负，min(_, cap) 对它们无影响；只截断正的惩罚项。
    _cap = float(cfg.get("cost_cap", 100.0))
    # EXP-04A：冲击单独放开截断（默认 0 = 与原来完全一致）。
    # 完全无界会让 Σqacc²~1e7 的落地帧单步惩罚撞 clip 下界，
    # 故用更高但仍有限的 cap，让「靠冲击弹一下」不再划算。
    _bicap = float(cfg.get("impulse_cap", 0.0))
    _capped = {}
    for k, v in out.items():
        _ck = _bicap if (k == "c_body_impulse" and _bicap > 0.0) else _cap
        _capped[k] = jp.minimum(v, _ck)
    return _capped


def all_rewards(env, data, action, info, done, first_contact, contact, foot_pos, foot_vel):
    # gaitfix 2026-08-28: 返回"未加权"的 16 项(Go1 风格, 加权在 step 里做)。
    # 16/16 与 Go1 对齐; orientation 已按计划降到 -1.5。
    linvel = env.get_local_linvel(data); gyro = env.get_gyro(data); gravity = env.get_gravity(data)
    cmd = jp.array(info["command"])
    qpos_abs = data.qpos[env._jnt_qpos_adr]
    qpos_diff = qpos_abs - env._default_ctrl[1:]
    pose_err = jp.sum(jp.square(qpos_diff))
    tau = data.actuator_force[1:]
    moving = (jp.linalg.norm(cmd) > 0.01).astype(jp.float32)
    # overconstraint 起步宽限掩码: ep_steps 前 warmup_steps 步 = 0(豁免), 之后 = 1
    warm = (info["ep_steps"] >= env._config.reward_config.warmup_steps).astype(jp.float32)
    foot_z = foot_pos[:, -1]
    foot_vxy = foot_vel[:, :2]
    target_z = env._foot_stance_z + env._config.reward_config.max_foot_height
    # 2026-11-17 walk: 足端奖必须建立在「真的移动了」之上。
    # 否则原地踏步能靠 step_bonus/feet_air_time 刷分，而 dist_track 在
    # 零净位移时是 0（不罚），站着/原地踏步变成安全局部最优。
    _moved_gate = jp.array(1.0, dtype=jp.float32)
    if (bool(env._config.reward_config.get("c_moved_gate_enable", False))
            and "trk_pos0" in info):
        _eps_m = float(env._config.reward_config.get("c_moved_gate_dist", 0.005))
        _d_m = jp.linalg.norm(
            jp.asarray(info["trk_pos_now"]) - jp.asarray(info["trk_pos0"]))
        _moved_gate = (_d_m > _eps_m).astype(jp.float32)
    # heightfix 2026-08-30: 基座离地高度(与 _get_termination 同一套算法)
    height = data.qpos[2] - env._floor_z
    out_of_limits = -jp.clip(qpos_abs - env._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos_abs - env._soft_uppers, 0.0, None)
    r = {
        # v25: 跟踪输入换 EMA(track_ema, 震荡≈0 不给分)。linvel[1:]/gyro[:2]
        # 只是形状占位, 两个函数分别只读 [0]/[2]。
        "tracking_lin_vel": _term_tracking_lin_vel(env, 
            cmd, jp.array([info["track_ema"][0], linvel[1], linvel[2]])),
        "tracking_ang_vel": _term_tracking_ang_vel(env, 
            cmd, jp.array([gyro[0], gyro[1], info["track_ema"][1]])),
        # v42: 堵慢漂移套利(见 _reward_yaw_drift)
        "yaw_drift": _term_yaw_drift(env, cmd, data.qpos, info["yaw0"]),
        # v46: 超速单边罚(见 _reward_overspeed)
        "overspeed": _term_overspeed(env, cmd, info["track_ema"][0]),
        # v45: 原地踏步的位置硬约束(只对站立指令生效)
        "base_pos": _term_base_pos(env, cmd, data.qpos, info["pos0"]),
        "lin_vel_z": jp.square(linvel[2]),
        "ang_vel_xy": jp.sum(jp.square(gyro[:2])),
        "orientation": jp.sum(jp.square(gravity[:2])),
        "pose": jp.exp(-jp.sum(jp.square(qpos_diff) * jp.array([1.0, 1.0, 0.1] * 4))),
        # heightfix 2026-08-30: 只罚【低于】目标的部分(平方项 = 软约束)。
        # 起因: 后退时策略学会了"蹲着走"(离地 0.109 < 站姿 0.165)。
        # 原因: 原先 16 项里没有任何一项约束基座高度, 只有摔倒阈值 0.08 兜底,
        #       而 0.109 离它还远 -> 蹲着走零代价。
        #       且 pose 的膝权重只有 0.1(髋是 1.0), 膝盖基本放任。
        # 用平方项: 稍微低一点几乎不罚, 蹲很低才重罚, 不会一刀切。
        # 只罚低不罚高: 避免压制任何需要降低身体的动作。
        "base_height": jp.square(jp.maximum(
            0.0, env._config.reward_config.base_height_target - height)),
        "stand_still": pose_err * (jp.linalg.norm(cmd[:2]) < 0.1).astype(jp.float32),
        "torques": jp.sqrt(jp.sum(jp.square(tau))) + jp.sum(jp.abs(tau)),
        "action_rate": jp.sum(jp.square(action - info["last_last_act"])),
        "energy": jp.sum(jp.abs(data.qvel[env._jnt_qvel_adr]) * jp.abs(tau)),
        "termination": done,
        # liftfix2 2026-08-29: 阈值从写死的 0.1 改成配置项(见 air_time_threshold)。
        "feet_air_time": jp.sum(
            (info["feet_air_time"] - env._config.reward_config.air_time_threshold)
            * first_contact) * moving * _moved_gate,
        "feet_clearance": jp.sum(jp.abs(foot_z - target_z) * jp.linalg.norm(foot_vxy, axis=-1)),
        # v26 2026-09-15: 抬脚发钱。落步瞬间(first_contact)按该脚本次腾空弧
        # 峰值结算: 离地 1cm 起付、线性升到 4cm 封顶。触地脚的 swing_peak 在
        # step 尾部清零之前仍持有弧峰值(清零发生在 reward 之后), 正好一次性结算。
        # 站立/微弹跳(<1cm)不付。_foot_stance_z 为脚底基准(同 feet_height 语义)。
        # 11-17 煜辰: 去掉起付，改成「越高越好」—— 抬脚 0cm->0、1cm->0.25、2cm->0.5、4cm->1.0。
        # 原式起付 2.5cm 是 v28 为「掐断蹲坐微抬刷赏」设的；现在蹲低由 posture(20% 容差) 管，
        # 而 2.5cm 门槛太高 —— 前两个 ckpt 的 step_bonus 恒为 0，随机探索根本触发不到。
        # 仍乘 first_contact（必须有起飞-落地循环，站立无此事件）和 moving（cmd>0.01）。
        "step_bonus": jp.sum(jp.clip(
            (info["swing_peak"] - env._foot_stance_z) / 0.04, 0.0, 1.0)
            * first_contact) * moving * _moved_gate,
        # liftfix3 2026-08-30: 从"必须抬到目标"改回"天花板"语义。
        # 原式 ((peak/target) - 1)^2 在峰值【低于】目标时也给出惩罚
        # (只抬到一半 -> 0.25 的平方项), 等于把 max_foot_height 变成了
        # 强制达标线, 而不是设计意图里的"上限"。
        # 实测后果: 后退时策略抬不到 6cm, 于是干脆不迈步 —— 脚不抬就
        # 不产生 first_contact, feet_height 与 feet_air_time 全部不结算,
        # 一次性规避掉所有步态惩罚(观测到的表现就是"完全不动")。
        # 现在只罚【超过】目标的部分: 抬不够不罚, 抬过头才罚。
        # 抬脚的激励交给 feet_air_time(正奖励 +1.5), 那才是驱动源。
        "feet_height": jp.sum(jp.square(
            jp.maximum(0.0, (info["swing_peak"] - env._foot_stance_z)
                       / env._config.reward_config.max_foot_height - 1.0)
        ) * first_contact) * moving,
        "feet_slip": jp.sum(jp.sum(jp.square(foot_vxy), axis=-1) * contact) * moving,
        # overconstraint 2026-09-06: 三腿支撑时长惩罚(软约束), * moving(行走中) * warm(起步宽限)
        "over_support":  _term_over_support(env, info) * moving * warm,
        "dof_pos_limits": jp.sum(out_of_limits),
        # 11-16 v7: 梯度型位移/转角跟踪（读位置真值，无 EMA 延迟）
        # 11-17 回归修复: 字典字面量会无条件求值 -> cost_mode 的 p1 没有 trk_*
        # 就 KeyError 崩训练。缺失时给 0（p1 的 terms 也不含本项，不影响结果）。
        "dist_track": (_term_dist_track(env, cmd, info)
                       if "trk_pos0" in info else jp.zeros((), dtype=jp.float32)),
        # 连续姿态恶化（依赖起点 z/up_z，带容差）
        "posture": (_term_posture(env, data.qpos[2], -gravity[2], info)
                    if bool(env._config.reward_config.get("c_posture_enable", False))
                    else jp.zeros((), dtype=jp.float32)),
        "yaw_track": (_term_yaw_track(env, cmd, info)
                      if "trk_yaw0" in info else jp.zeros((), dtype=jp.float32)),
    }
    # 论文成本项：只在配方把它们列进 terms 时才计算（默认不进，等价性不变）
    if any(t.startswith("c_") for t in env._config.reward_config.get("terms", [])):
        r.update(paper_costs(env, data, action, info, contact, foot_vel, gravity, foot_pos))
    return r




# ============================================================================
# 2026-11-21 v15: 阶段式「全奖励」（c_stage_enable，默认关）
# ----------------------------------------------------------------------------
# 与前面 13 项成本的最大区别：这里每一项都是有界奖励 r∈[0,1]，没有负项；
# "惩罚"只表现为奖励变小。四个阶段累积判定（前一级是后一级的门控）：
#   1 站稳 = >=3 足着地 且 质心内距 m>0
#   2 站正 = 阶段1 + up_z>=0.95
#   3 站高 = 阶段2 + bz>=0.115
#   4 追踪 = 阶段3 连续 >= c_stage_hold 拍
# 权重随阶段变化，由 c_stage_w_<term>（4 个值）与 c_stage_base_<term> 给出。
# 面积沿用"以站姿面积为基准、双侧、有界"的原逻辑；质心内距上界取
# 站姿关键帧实测 62.8 mm x 1.2 = 75.4 mm（c_stage_m_cap），超过不再加分。
# 阶段基础分 c_stage_bonus=[0,0.6,1.9,3.0] 保证"后一阶段总量不小于前一阶段"。
# ============================================================================
def stage_rewards(env, data, info, contact, foot_pos, foot_vel):
    """返回 (各项加权奖励 dict, 阶段索引 -1..3, 阶段基础分, 新 hold 计数)。"""
    cfg = env._config.reward_config
    cf = jp.asarray(contact) > 0.5
    nf = jp.sum(cf.astype(jp.float32))
    p = jp.asarray(foot_pos)[:, :2]
    pc = jp.mean(p, axis=0)
    ang = jp.arctan2(p[:, 1] - pc[1], p[:, 0] - pc[0])
    ps = p[jp.argsort(ang)]
    pn = jp.roll(ps, -1, axis=0)
    A = 0.5 * jp.abs(jp.sum(ps[:, 0] * pn[:, 1] - pn[:, 0] * ps[:, 1]))
    com = data.subtree_com[1][:2]
    e = pn - ps
    n = jp.stack([e[:, 1], -e[:, 0]], axis=-1)
    n = n / (jp.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    inner = jp.sum(n * (pc[None, :] - ps), axis=-1)
    sgn = jp.where(inner > 0.0, -1.0, 1.0)
    d = jp.sum(n * (com[None, :] - ps), axis=-1) * sgn
    m = -jp.max(d)

    q = data.qpos[env._jnt_qpos_adr]
    if bool(env._config.task_config.get("joint_sign_mask_enable", False)):
        q = q * jp.asarray(env._config.task_config.joint_sign_mask, dtype=q.dtype)
    qerr = jp.max(jp.abs(q))
    bz = data.qpos[2]
    # 本模块没有 import up_z，直接按定义算（与 task.up_z 等价）
    uz = 1.0 - 2.0 * (data.qpos[4] ** 2 + data.qpos[5] ** 2)

    a_ref = float(cfg.get("c_area_ref", 0.040915))
    m_cap = float(cfg.get("c_stage_m_cap", 0.0754))
    r_pose = 1.0 - jp.clip(qerr / float(cfg.get("c_stage_pose_tol", 1.0)), 0.0, 1.0)
    r_stab = jp.clip(m / m_cap, 0.0, 1.0)
    r_area = 1.0 - jp.clip(jp.abs(A / a_ref - 1.0) /
                           float(cfg.get("c_stage_area_dev", 0.6)), 0.0, 1.0)
    dbz = bz - jp.asarray(info["stg_bz_prev"])
    r_hd = jp.clip(dbz / float(cfg.get("c_stage_hdelta_v", 0.01)), 0.0, 1.0)
    uz_lo = float(cfg.get("c_stage_upz_lo", 0.6))
    r_upz = jp.clip((uz - uz_lo) / max(1.0 - uz_lo, 1e-6), 0.0, 1.0)
    r_ct = (nf >= 3.0).astype(jp.float32)
    sp = jp.linalg.norm(jp.asarray(foot_vel)[:, :2], axis=-1)
    r_slip = 1.0 - jp.mean(jp.clip(sp / float(cfg.get("c_stage_slip_v", 0.02)), 0.0, 1.0))

    ok1 = (nf >= 3.0) & (m > 0.0)
    ok2 = ok1 & (uz >= 0.95)
    ok3 = ok2 & (bz >= 0.115)
    hold_new = jp.where(ok3, jp.asarray(info["stg_hold"]) + 1,
                        jp.zeros((), dtype=jp.int32))
    ok4 = hold_new >= int(cfg.get("c_stage_hold", 50))
    idx = jp.where(ok4, 3, jp.where(ok3, 2, jp.where(ok2, 1, jp.where(ok1, 0, -1))))

    raw = {"pose": r_pose, "stab": r_stab, "area": r_area, "hdelta": r_hd,
           "upz": r_upz, "contact": r_ct, "slip": r_slip}
    terms = {}
    # 权重表 5 项：索引 0 = 未站（仍付触地/滑移 + 小额 upz/hdelta，避免 0 分平台），
    # 1..4 = 四阶段。idx=-1 → 表索引 0。
    for k, r in raw.items():
        base = float(cfg.get("c_stage_base_" + k, 1.0))
        w = jp.asarray(cfg.get("c_stage_w_" + k, [0.0, 0.25, 0.5, 0.75, 1.0]))
        wk = w[jp.clip(idx + 1, 0, 4)]
        terms[k] = base * wk * r
    bonus_tab = jp.asarray(cfg.get("c_stage_bonus", [0.0, 0.0, 0.6, 1.9, 3.0]))
    bonus = bonus_tab[jp.clip(idx + 1, 0, 4)]
    return terms, idx, bonus, hold_new
