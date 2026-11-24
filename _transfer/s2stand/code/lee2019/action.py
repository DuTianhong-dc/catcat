# -*- coding: utf-8 -*-
"""【输出】动作 -> 12 个关节目标角的映射注册表。

主线三档：absolute / per_side / offset（原来内联在 step() 里）。
本模块新增论文 Lee 2019 用的 delta_current：φd = φt + k·o，
即在【当前关节角】上叠加增量，便于自翻正时大范围探索。
    "action_config.mode": "delta_current"          ← JSON 直接改
    "action_config.delta_scale": 0.25
    "action_config.apply_velocity_limit": false    ← 论文无逐拍限速
公式与主线逐字一致，等价性由 play/_check_lee2019_equiv.py 保证。
"""
import jax.numpy as jp

ACTION_MODES = ("absolute", "per_side", "offset", "delta_current", "delta_anchored")


def map_action(env, state, action):
    """返回 (12,) 关节目标角；调用方负责写回 info["motor_targets"]。"""
    mode = str(env._config.action_config.mode)
    if mode not in ACTION_MODES:
        raise ValueError("未知动作映射: %s" % mode)

    # action_delay 0/1/2/3 拍（台架实测舵机 60~80 ms 一阶惯性）
    _delay = state.info["action_delay"]
    # 专家前缀期间绕过执行延迟，保证 teacher forcing 的拍点不被打乱。
    if getattr(env, "_warm_actions", None) is not None:
        _delay = jp.where(
            state.info["ep_steps"] < state.info["warm_steps"], 0, _delay)
    a_apply = jp.where(_delay == 0, action,
                       jp.where(_delay == 1, state.info["last_act"],
                                jp.where(_delay == 2, state.info["last_last_act"],
                                         state.info["last_last_last_act"])))
    _d = env._default_ctrl[1:]           # 站姿关节角（keyframe stand 的 ctrl）

    if mode == "absolute":
        _mt = jp.clip(a_apply * env._act_scale, env._act_lo, env._act_hi)
    elif mode == "per_side":
        _sc = jp.where(a_apply > 0, env._act_hi - _d, _d - env._act_lo)
        _mt = _d + a_apply * _sc
    elif mode == "offset":
        _mt = _d + a_apply * env._config.pos_action_scale
    else:                                  # delta_current / delta_anchored（论文式）
        _q = state.data.qpos[env._jnt_qpos_adr]
        _a = a_apply
        if bool(env._config.task_config.get("joint_sign_mask_enable", False)):
            # 2026-11-20：a 是策略侧旧约定，目标角要换到当前模型的新约定。
            _s = jp.asarray(env._config.task_config.joint_sign_mask, dtype=_a.dtype)
            _a = _a * _s
        if mode == "delta_anchored":
            # 2026-11-22 v16：部分锚定站姿。lambda=0 时与 delta_current 逐位一致。
            # 零动作时目标角 = (1-lam)*q + lam*q_stand，腿被压弯后目标不会跟着无限下移
            # （实测 lam=0.15、零动作，近站起点末 2 s 有效支撑从 0% 升到 90%）。
            # lam 按机身高度渐入：bz<lo 时关断，避免把趴姿的腿推向站姿而把人推倒
            # （实测未门控的 lam=0.15 会让趴姿 up_z 从 0.75 掉到 -0.05）。
            _lam = float(env._config.action_config.anchor_lambda)
            if _lam > 0.0:
                _lo = float(env._config.action_config.anchor_bz_lo)
                _sp = max(float(env._config.action_config.anchor_bz_span), 1e-6)
                _w = jp.clip((state.data.qpos[2] - _lo) / _sp, 0.0, 1.0)
                _lam = _lam * _w
                _q = _q + _lam * (_d - _q)
        _mt = _q + _a * float(env._config.action_config.delta_scale)

    if mode != "absolute":
        if bool(env._config.action_config.apply_velocity_limit):
            # 逐关节限速：防 target 一步跳太远让 kp×(target−q) 顶到 forcerange 饱和
            _prev = state.info["motor_targets"]
            _dv = env._vel_lim * env.dt
            _mt = jp.clip(_mt, _prev - _dv, _prev + _dv)
        else:
            _mt = jp.clip(_mt, env._act_lo, env._act_hi)
    return _mt
