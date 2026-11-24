# -*- coding: utf-8 -*-
"""【输入】actor / critic 观测段注册表。

主线把 12 段硬编码在 _get_obs() 里，只有 ablate_* 布尔能删段。
本模块把每段命名成 parts 的键，由 obs_config.blocks 决定拼接顺序与取舍：
    "obs_config.blocks": ["gravity", "gyro", "jpos", ...]   ← JSON 直接改
privileged(critic) 暂保持主线逐字公式，后续需要再拆。
"""
import jax
import jax.numpy as jp

# 与主线默认 ablate 配置等价的 10 段（tau / contact 默认关）。
# ⚠️ 顺序 = 权重语义，改顺序必须重新训练。
DEFAULT_BLOCKS = [
    "linvel", "gyro", "gravity", "cmd", "jpos", "jvel",
    "last_act", "last2_act", "last3_act", "motor_targets",
]
BLOCK_DIMS = {
    "linvel": 3, "gyro": 3, "gravity": 3, "cmd": 2,
    "jpos": 12, "jvel": 12, "tau": 12,
    "last_act": 12, "last2_act": 12, "last3_act": 12,
    "motor_targets": 12, "contact": 4,
    # 论文 Table I：关节误差史（当前角 − 上拍目标），我们 50Hz 用 3 拍动作史近似
    "jpos_err": 12,
    # 2026-11-15 walk: 估计机身高度（论文 he）。训练用仿真真值，部署换估计器。
    "base_height": 1,
}


def obs_size(blocks):
    return sum(BLOCK_DIMS[b] for b in blocks)


def build_obs(env, data, info):
    # obsfix 保留(传感器噪声/真力矩)。gaitfix 2026-08-28: 改为返回
    # {"state": 59维带噪 actor 观测(novy: 删除侧移后 60->59),
    #  "privileged_state": critic 特权观测(Go1 结构): 干净真值 + 加速度计 +
    #  全局角速度 + 接触/摆动时长 + 外力; 脚速传感器缺失该项暂缺}。
    nc = env._config.noise_config

    def _noisy(x, scale, key):
        return x + (2 * jax.random.uniform(key, shape=x.shape) - 1) * nc.level * scale

    info["rng"], n1 = jax.random.split(info["rng"])
    linvel_c = env.get_local_linvel(data)   # 保留: critic 特权观测仍看干净真值
    # v6 2026-09-07: obs[0:3] 恢复真值+噪声(摘除 odom), 单变量验证学习质量假设
    # leg_odometry 2026-09-06: actor 观测改用【真机固件同款里程计】的输出(在 step 里
    # 用 leg_odometry.estimate_velocity 算好, 存 info["odom_linvel"]), 替代 obsdr 的
    # "真值 x 比例误差 + 慢变零偏 + 高频噪声"理想化模型 —— 让策略直接适应真实估计器的
    # 非线性/瞬态/边界行为, 而非理想高斯误差。
    linvel = info["odom_linvel"]   # v12: 恢复 odom 观测(回到 V1)
    gyro_c = env.get_gyro(data)
    info["rng"], n2 = jax.random.split(info["rng"])
    gyro = _noisy(gyro_c, nc.scales.gyro, n2)
    gravity_c = env.get_gravity(data)
    info["rng"], n3 = jax.random.split(info["rng"])
    gravity = _noisy(gravity_c, nc.scales.gravity, n3)
    # 2026-11-04 论文 p6 (#27)：四脚全离地时放大 base linvel/位置噪声以模拟 TSIF 漂移。
    # gain=1.0（默认）时整段是 Python 常量分支：不分裂 rng、不加噪声 → 主线等价性保持。
    _air_gain = float(nc.get("airborne_gain", 1.0))
    if _air_gain != 1.0:
        _airborne = jp.all(jp.array(info["contact_obs"]) < 0.5)
        info["rng"], _na = jax.random.split(info["rng"])
        _extra = ((2 * jax.random.uniform(_na, (3,)) - 1)
                  * _air_gain * float(nc.scales.linvel))
        _extra = jp.where(_airborne, _extra, jp.zeros(3))
        linvel_c = linvel_c + _extra
        linvel = linvel + _extra
    cmd = jp.array(info["command"])
    # ★ 2026-10-28 回差角补偿（照抄 Open Duck）：
    #   真机磁编码器在齿轮箱【输出侧】，读数 = 主关节角 + 回差角。
    #   他们：joint_angles = get_actuator_joints_qpos() + get_actuator_backlash_qpos()
    _jpos_raw = data.qpos[env._jnt_qpos_adr]
    if env._bl_qpos_adr is not None:
        _jpos_raw = _jpos_raw + data.qpos[env._bl_qpos_adr]
    # 2026-11-20：模型是“新约定”，策略按“旧约定”训练/部署 -> 这里转回去。
    if bool(env._config.task_config.get("joint_sign_mask_enable", False)):
        _jpos_raw = _jpos_raw * jp.asarray(
            env._config.task_config.joint_sign_mask, dtype=_jpos_raw.dtype)
    jpos_c = _jpos_raw - env._default_ctrl[1:]
    info["rng"], n4 = jax.random.split(info["rng"])
    jpos = _noisy(jpos_c, nc.scales.joint_pos, n4)
    # ★ 2026-10-28 关节速度照抄 Open Duck：噪声在 rad/s 域加，最后整体 ×dof_vel_scale(0.05)
    #   注意顺序不能反 —— 先缩放再加 0.86 的噪声会把噪声放大 20 倍。
    jvel_c = data.qvel[env._jnt_qvel_adr]
    if bool(env._config.task_config.get("joint_sign_mask_enable", False)):
        jvel_c = jvel_c * jp.asarray(
            env._config.task_config.joint_sign_mask, dtype=jvel_c.dtype)
    info["rng"], n5 = jax.random.split(info["rng"])
    jvel = _noisy(jvel_c, nc.scales.joint_vel, n5) * env._config.dof_vel_scale
    # tau_est 2026-09-06: tau 加各关节独立慢变零偏(对齐真机电流基线 0~45mA 差异)
    tau = data.actuator_force[1:] + info["tau_bias"]  # 真实执行器力, 12 个舵机
    # obs 拼装。★ 2026-10-28 照抄 Open Duck 的结构：
    #   gyro / cmd / jpos(减站姿) / jvel(×0.05) / 3 步动作史 / motor_targets
    #   contact 4 维默认关(ablate_contact=True, 真机无传感器) → state 87 -> 83
    #   tau 默认不进 actor（ablate_tau=True），与他们的 asymmetric actor-critic 一致。
    # ★ 注册表驱动：段名 -> 张量。想加/删/换序只改 obs_config.blocks（JSON 可覆盖）。
    # 2026-11-20：motor_targets 存的是【模型侧新约定】的目标角，策略要的是旧约定，
    # 必须与 jpos 一起转换，否则 jpos_err = jpos_old - mt_new 两个坐标系相减。
    _mt_obs = jp.array(info["motor_targets"])
    if bool(env._config.task_config.get("joint_sign_mask_enable", False)):
        _mt_obs = _mt_obs * jp.asarray(
            env._config.task_config.joint_sign_mask, dtype=_mt_obs.dtype)
    parts = {
        "linvel": linvel,
        "gyro": gyro,
        "gravity": gravity,
        "cmd": cmd,
        "jpos": jpos,
        "jvel": jvel,
        "tau": tau,
        "last_act": jp.array(info["last_act"]),
        "last2_act": jp.array(info["last_last_act"]),
        "last3_act": jp.array(info["last_last_last_act"]),
        "motor_targets": _mt_obs,
        "contact": jp.array(info["contact_obs"]),
        "jpos_err": _jpos_raw - _mt_obs,
        # 2026-11-15 walk: 机身高度 1 维（默认 blocks 不含，历史配方不变）
        "base_height": jp.reshape(data.qpos[2], (1,)),
    }
    blocks = list(env._config.obs_config.blocks)
    unknown = [b for b in blocks if b not in parts]
    if unknown:
        raise ValueError("未知观测段: %s" % unknown)
    state_obs = jp.concatenate([parts[b] for b in blocks])

    angvel_world = data.site_xmat[env._imu_site_id] @ gyro_c
    acc = data.sensordata[env._acc_adr:env._acc_adr + 3]
    # 2026-11-16 walk: critic 特权观测也可去掉线速度（部署链路彻底不依赖速度）。
    # 默认 False -> 历史逐位等价；walk 配方显式设 True。
    _drop_lv = bool(env._config.obs_config.get("drop_privileged_linvel", False))
    _priv = [
        state_obs,                                  # 83(ablate_tau+contact) 等组合
        gyro_c,                                     # 3
        acc,                                        # 3
        gravity_c,                                  # 3
    ]
    if not _drop_lv:
        _priv.append(linvel_c)                      # 3
    _priv += [
        angvel_world,                               # 3
        jpos_c,                                     # 12
        jvel_c,                                     # 12
        tau,                                        # 12
        info["last_contact"].astype(jp.float32),    # 4
        info["feet_air_time"],                      # 4
        data.xfrc_applied[env._base_body_id, :3],  # 3
    ]
    privileged = jp.hstack(_priv)
    return {"state": state_obs, "privileged_state": privileged}

