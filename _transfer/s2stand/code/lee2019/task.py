# -*- coding: utf-8 -*-
"""【复位 / 终止】初始状态分布与回合终止判据。

复现 Lee 2019 最关键的两处就在这里：
  · 论文的 self-righting 任务**只有时间限制**作为终止条件（倒地不许结束回合）；
  · 论文从 0.5 m 高、随机四元数、随机关节角投落作为初始分布。
本模块把这两处从 reset() / _get_termination() 抽出来，复现时只改这里。

当前 stand / fall 模式与主线**逐字等价**（等价性回归保证）；
fall 模式已实现（含可选几何贴地）；默认 drop_clearance=-1 保持固定绝对高度。
"""
import jax
import jax.numpy as jp
from mujoco import mjx

INIT_MODES = ("stand", "fall", "demo", "exp08", "pose_bank")  # pose_bank = 离线沉降稳定的倒立姿态库
TERMINATE_MODES = ("fall", "timeout", "stand_fall")   # timeout = 论文式只按时间终止（由 EpisodeWrapper 判）


_POSE_BANK_CACHE = {}


def _pose_bank(path):
    """惰性加载离线姿态库，只取 train split；缓存成 jp array 供 JIT 索引。"""
    if path not in _POSE_BANK_CACHE:
        import numpy as _np
        with _np.load(path, allow_pickle=True) as _z:
            _q = _np.asarray(_z["qpos"], dtype=_np.float32)
            if "split" in _z.files:
                _q = _q[_np.asarray(_z["split"]) == 0]
        if _q.ndim != 2 or _q.shape[0] == 0:
            raise ValueError("姿态库形状非法: %s" % (_q.shape,))
        _POSE_BANK_CACHE[path] = _q  # cache numpy only; never cache a tracer
    return jp.asarray(_POSE_BANK_CACHE[path])


def pose_bank_initial_state(env, rng):
    """从沉降稳定倒立姿态库采样起点；qvel 恒 0，避免投落带来的免费能量。"""
    _p = str(env._config.task_config.get("pose_bank_npz", ""))
    if not _p:
        raise ValueError("pose_bank 模式需要 task_config.pose_bank_npz")
    _qb = _pose_bank(_p)
    if _qb.shape[1] != env._mjx_model.nq:
        raise ValueError("姿态库 qpos 维度 %d != nq %d"
                         % (_qb.shape[1], env._mjx_model.nq))
    rng, k = jax.random.split(rng)
    _idx = jax.random.randint(k, (), 0, _qb.shape[0])
    _q = _qb[_idx]
    # 2026-11-20：姿态库是“旧约定”生成的（左右同限位），当前模型把镜像写进了
    # range/axis，同一物理姿态 q 差符号。开关打开时在这里做一次转换。
    if bool(env._config.task_config.get("joint_sign_mask_enable", False)):
        _s = jp.asarray(env._config.task_config.joint_sign_mask, dtype=_q.dtype)
        _q = _q.at[env._jnt_qpos_adr].multiply(_s)
    return _q, jp.zeros(env._mjx_model.nv), rng


def initial_state(env, rng):
    """返回 (qpos, qvel, rng)。rng 分裂次数与顺序必须与主线一致。"""
    mode = str(env._config.task_config.init_mode)
    if mode == "pose_bank":
        return pose_bank_initial_state(env, rng)
    if mode == "exp08":
        return exp08_initial_state(env, rng)
    if mode in ("fall", "demo"):
        # 论文 II-D：从 0.5 m 高、任意姿态投落。随机单位四元数 + 随机关节角，
        # 回差铰链保持 0（继承 keyframe stand），qvel 置零。
        cfg = env._config.task_config
        qpos = env._init_q
        rng, k = jax.random.split(rng)
        quat = jax.random.normal(k, (4,))
        quat = quat / jp.linalg.norm(quat)
        # EXP-02: 偏向倒立的初始分布。默认 frac=0 走原路径，rng 流不变，
        # 由 _check_lee2019_equiv.py 的 30/30 回归保证等价。
        _frac = float(cfg.get("init_inverted_frac", 0.0))
        if _frac > 0.0:
            _uzmax = float(cfg.get("init_up_z_max", -0.2))
            _thmin = jp.arccos(jp.asarray(_uzmax))
            rng, k = jax.random.split(rng)
            _inv = jax.random.uniform(k) < _frac
            rng, k = jax.random.split(rng)
            _th = jax.random.uniform(k, minval=_thmin, maxval=jp.pi)
            rng, k = jax.random.split(rng)
            _phi = jax.random.uniform(k, minval=0.0, maxval=2.0 * jp.pi)
            _n = jp.array([jp.cos(_phi), jp.sin(_phi), 0.0])
            _qinv = jp.concatenate(
                [jp.cos(_th / 2.0)[None], jp.sin(_th / 2.0) * _n])
            quat = jp.where(_inv, _qinv, quat)
        qpos = qpos.at[3:7].set(quat)
        qpos = qpos.at[2].set(float(cfg.get("drop_height", 0.5)))
        rng, k = jax.random.split(rng)
        qpos = qpos.at[0:2].set(
            jax.random.uniform(k, (2,), minval=-0.1, maxval=0.1))
        rng, k = jax.random.split(rng)
        s = float(cfg.get("joint_range_scale", 0.8))
        _u = jax.random.uniform(k, (12,))
        _lo, _hi = jp.array(env._act_lo), jp.array(env._act_hi)
        qj = _lo * s + (_hi - _lo) * s * _u
        qpos = qpos.at[env._active_qpos_adr].set(qj)
        qvel = jp.zeros(env._mjx_model.nv)
        # ---- EXP-04A 里程碑状态课程（demo_states 为空时整段是 Python 常量
        #      分支：不分裂 rng、不改 qpos；fall 模式逐字等价）----
        if mode == "demo":
            _states = list(cfg.get("demo_states", []))
            if _states:
                _S = jp.asarray(_states, dtype=jp.float32)
                rng, kd1 = jax.random.split(rng)
                _use = jax.random.uniform(kd1) < float(cfg.get("demo_frac", 0.5))
                rng, kd2 = jax.random.split(rng)
                _idx = jax.random.randint(kd2, (), 0, _S.shape[0])
                _qd = _S[_idx]
                _n = float(cfg.get("demo_noise", 0.0))
                rng, kd3 = jax.random.split(rng)
                _quat = _qd[3:7] + jax.random.uniform(
                    kd3, (4,), minval=-_n, maxval=_n)
                _quat = _quat / jp.linalg.norm(_quat)
                rng, kd4 = jax.random.split(rng)
                _qj = _qd[env._active_qpos_adr] + jax.random.uniform(
                    kd4, (12,), minval=-_n, maxval=_n)
                # 8 个被动关节沿用参考帧（保持五连杆几何一致）；
                # qvel 保持 0，高度交给下面的几何贴地逻辑重算，
                # 让 MJX 自己重建接触，避免 CPU MuJoCo 接触状态迁移。
                _qdemo = _qd.at[3:7].set(_quat)
                _qdemo = _qdemo.at[env._active_qpos_adr].set(_qj)
                qpos = jp.where(_use, _qdemo, qpos)
        # ---- 几何贴地投落（方案 A，2026-11-05） ----
        # drop_clearance < 0 = 关闭（保持论文原样的固定绝对高度，等价性不变）。
        # 开启时：以【本回合这个随机姿态】的最低 geom 为基准反推 qpos[2]，
        # 使初始离地恰好 clearance；避免固定 z 在低高度下把机身插进地面。
        _clear = float(cfg.get("drop_clearance", -1.0))
        if _clear >= 0.0:
            _d0 = mjx.make_data(env._mjx_model)
            _d0 = _d0.replace(qpos=qpos, qvel=qvel, ctrl=env._default_ctrl)
            _d0 = mjx.forward(env._mjx_model, _d0)
            _lo = _d0.geom_xpos[:, 2] - env._mjx_model.geom_rbound
            _fid = int(env._mj_model.geom("floor").id)
            _lo = _lo.at[_fid].set(1.0e6)      # 地面自己不计
            # ⚠️ _lo 是在【当前 qpos[2]】（legacy drop_height）下算的，
            #    它已含这段高度；必须加回来，否则净空会凭空少掉 drop_height
            #    （第一版实测变成 -0.42 m，初始 60 个穿透接触把猫弹到 1.3 m）。
            qpos = qpos.at[2].set(qpos[2] - jp.min(_lo) + _clear)
        return qpos, qvel, rng
    if mode != "stand":
        raise ValueError("未知 init_mode: %s" % mode)
    qpos = env._init_q
    qvel = jp.zeros(env._mjx_model.nv)
    rng, k = jax.random.split(rng)
    qpos = qpos.at[0:2].set(qpos[0:2] + jax.random.uniform(k, (2,), minval=-0.1, maxval=0.1))
    rng, k = jax.random.split(rng)
    qvel = qvel.at[0:6].set(jax.random.uniform(k, (6,), minval=-0.1, maxval=0.1))
    # drfix 2026-08-28: 初始关节角扰动(只抖 12 个主动关节)。
    # randomize.py 里的 qpos0 项对 cat3.3 不生效 —— reset 用的是 keyframe
    # "stand" 的 qpos, 而 _init_q 早在 __init__ 就固定成常量了, DR 改不到它
    # (官方 Go1 有同样的洞: 它抖 qpos0, 可它的 reset 用的是 keyframe "home")。
    # 所以真正生效的初始位姿扰动只能在这里做。
    # 只抖主动关节: 8 个从动关节受 equality 四连杆约束, 抖了会在 reset
    # 瞬间被约束求解器猛拉, 起手就炸。
    rng, k = jax.random.split(rng)
    qpos = qpos.at[env._active_qpos_adr].set(
        qpos[env._active_qpos_adr]
        + jax.random.uniform(k, (12,), minval=-0.03, maxval=0.03))
    return qpos, qvel, rng


def termination(env, data, info=None):
    mode = str(env._config.task_config.terminate_mode)
    if mode == "timeout":
        # 论文 II-D：self-righting 的终止条件**只有时间限制**。
        # 环境内永不提前终止；超时由 brax EpisodeWrapper 按 episode_length 判定。
        return jp.zeros((), dtype=bool)
    if mode == "stand_fall":
        # 2026-11-21 v13：锁存式跌倒终止。
        # 只有“本回合曾经站正且抬到 0.10 以上”（info["stand_latch"]）之后，
        # 再掉到 up_z<0.65 或 bz<0.06 才终止。低趴起点在还没站起来之前不受影响。
        # 目的：把“摔倒后陪跑 200 拍收罚款”换成“提前结束、只损失未来正分”，
        # 从博弈结构上消掉“趴平苟活”的套利（walk v13 同款结论）。
        _uz = 1.0 - 2.0 * (data.qpos[4] ** 2 + data.qpos[5] ** 2)
        _fall = (_uz < 0.65) | (data.qpos[2] < 0.06)
        if info is None:
            return _fall
        return _fall & (jp.asarray(info["stand_latch"]) > 0.5)
    if mode != "fall":
        raise ValueError("未知 terminate_mode: %s" % mode)
    # terminate 2026-08-28: 摔倒检测(照官方 Go1 机制, 阈值按 cat3.3 尺寸):
    # 基座离地过低 或 机身过度倾斜(侧翻/仰翻) -> 回合提前结束。
    height = data.qpos[2] - env._floor_z
    up_z = -env.get_gravity(data)[2]  # 1=正立, 0=侧躺, -1=四脚朝天
    fall = (height < env._config.terminate_config.min_base_height) | (
        up_z < env._config.terminate_config.up_vec_z
    )
    return fall  # gaitfix: 返回 bool, step 内再转 float


def up_z(qpos):
    """机身朝上方向与世界 +z 的点积：1 正立 / 0 侧躺 / -1 四脚朝天。

    qpos[3:7] = [w, x, y, z]；R[2,2] = 1 - 2(x^2 + y^2)。
    与 e_o = ||gravity_body - [0,0,-1]|| 的关系：e_o = sqrt(2(1-up_z))。
    """
    q = qpos[3:7]
    return 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)


def stage1_signals(cfg, prev_qpos, cur_qpos, hold):
    """阶段 1：返回 (hold_new, success, duz)；关闭时全为中性值。

    success = up_z > up_z_min 连续 hold_steps 帧；duz = 本帧 up_z 增量，
    由调用方乘 shaping_w 后加进 reward。
    """
    if not bool(cfg.get("stage1_enable", False)):
        return jp.array(0, dtype=jp.int32), jp.zeros((), dtype=bool), 0.0
    thr = float(cfg.get("stage1_up_z_min", -0.5))
    need = int(cfg.get("stage1_hold_steps", 10))
    u = up_z(cur_qpos)
    # up_z > -0.5 表示已脱离四脚朝天；侧躺(0)与正立(+1)都算成功。
    # 不用 |up_z|<0.5：那样会把快速翻正（穿过侧躺带 <10 帧）误判失败。
    h = jp.where(u > thr, hold + 1, jp.array(0, dtype=jp.int32))
    # EXP-04A：电平 -> 上升沿。默认 False 保持原语义（等价性保证）。
    _succ = (h == need) if bool(cfg.get(
        "stage1_success_rising_edge", False)) else (h >= need)
    return h, _succ, u - up_z(prev_qpos)


def body_up(qpos):
    """机体系 z 轴在世界系的投影（yaw 无关）。"""
    w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
    return jp.stack([
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y)])


def stage0_signals(env, cur_qpos, cur_qvel, hold):
    """Stage 0：返回 (hold_new, success, d, terms)。

    d 是 yaw 无关的密集距离：姿态 + 12 主动关节 + 基座 xy/z。
    success 用严格容差连续 hold_steps 拍；d 用 2x 容差的奖励尺度。
    stage0_v2=True 时改用 EXP-09 A 段口径：关节角为主，去掉 xy/z。
    """
    cfg = env._config.task_config
    if not bool(cfg.get("stage0_enable", False)):
        return (jp.array(0, dtype=jp.int32), jp.zeros((), dtype=bool),
                jp.zeros(()), {})
    q0 = jp.asarray(cfg.get("stage0_target_qpos"), dtype=jp.float32)
    # ---- EXP-09 A 段：e_q 为主项，姿态/速度只做辅项（默认关闭）----
    if bool(cfg.get("stage0_v2", False)):
        qa = cur_qpos[env._jnt_qpos_adr]
        q0a = q0[env._jnt_qpos_adr]
        sig = float(cfg.get("stage0_sigma_q", 0.25))
        e_q = jp.sqrt(jp.mean(jp.square((qa - q0a) / sig)) + 1e-8)
        e_ori = jp.linalg.norm(
            body_up(cur_qpos) - body_up(q0)) / float(
                cfg.get("stage0_v2_sigma_ori", 0.30))
        e_v = jp.linalg.norm(cur_qvel[env._jnt_qvel_adr]) / 2.0
        ok = ((e_q <= float(cfg.get("stage0_tol_eq", 0.4)))
              & (e_ori <= 1.0)
              & (e_v <= float(cfg.get("stage0_tol_ev", 0.5))))
        need = int(cfg.get("stage0_hold_steps", 10))
        h = jp.where(ok, hold + 1, jp.array(0, dtype=jp.int32))
        succ = h >= need
        d = (float(cfg.get("stage0_v2_w_eq", 1.0)) * e_q ** 2
             + float(cfg.get("stage0_v2_w_ori", 0.1)) * e_ori ** 2
             + float(cfg.get("stage0_v2_w_ev", 0.1)) * e_v ** 2)
        return h, succ, d, {"e_q": e_q, "e_ori": e_ori, "e_v": e_v}
    qa = cur_qpos[env._jnt_qpos_adr]
    q0a = q0[env._jnt_qpos_adr]
    _up = body_up(cur_qpos)
    _up0 = body_up(q0)
    e_ori = jp.linalg.norm(_up - _up0) / float(cfg.get("stage0_tol_ori", 0.15))
    e_q = jp.mean(jp.square(
        (qa - q0a) / float(cfg.get("stage0_tol_joint", 0.10))))
    e_xy = jp.linalg.norm(cur_qpos[0:2] - q0[0:2]) / float(
        cfg.get("stage0_tol_xy", 0.04))
    e_z = jp.abs(cur_qpos[2] - q0[2]) / float(cfg.get("stage0_tol_z", 0.012))
    ok = (e_ori <= 1.0) & (e_q <= 1.0) & (e_xy <= 1.0) & (e_z <= 1.0)
    need = int(cfg.get("stage0_hold_steps", 10))
    h = jp.where(ok, hold + 1, jp.array(0, dtype=jp.int32))
    succ = h >= need
    d_ori = jp.linalg.norm(_up - _up0) / float(
        cfg.get("stage0_rew_ori", 0.30))
    d_q = jp.mean(jp.square(
        (qa - q0a) / float(cfg.get("stage0_rew_joint", 0.20))))
    d_xy = jp.linalg.norm(cur_qpos[0:2] - q0[0:2]) / float(
        cfg.get("stage0_rew_xy", 0.08))
    d_z = jp.abs(cur_qpos[2] - q0[2]) / float(
        cfg.get("stage0_rew_z", 0.024))
    d = (float(cfg.get("stage0_w_ori", 0.35)) * d_ori ** 2
         + float(cfg.get("stage0_w_joint", 0.35)) * d_q
         + float(cfg.get("stage0_w_xy", 0.15)) * d_xy ** 2
         + float(cfg.get("stage0_w_z", 0.15)) * d_z ** 2)
    terms = {"e_ori": e_ori, "e_joint": e_q, "e_xy": e_xy, "e_z": e_z}
    return h, succ, d, terms



def exp08_initial_state(env, rng):
    """EXP-08 窄起点：与 _expert_bc.py::_sim_one 同口径的倒立 Q0 + 小扰动。

    坡度/质心 DR **不体现在 qpos**：数据集侧把地面与机身同步旋转、保持
    相对地面姿态不变，在地面水平系里等价于只改重力方向，由 dr.py 的
    xfrc_applied 实现。qvel 全零、贴地，消掉投落落地冲击。
    """
    cfg = env._config.task_config
    q0 = jp.asarray(cfg.get("exp08_init_q0"), dtype=jp.float32)
    rng, k = jax.random.split(rng)
    axis = jax.random.normal(k, (3,)).at[2].set(0.0)
    axis = axis / (jp.linalg.norm(axis) + 1e-9)
    rng, k = jax.random.split(rng)
    amp = float(cfg.get("exp08_init_quat_noise", 0.05))
    ang = jax.random.uniform(k, minval=-amp, maxval=amp)
    dq = jp.concatenate([jp.cos(ang / 2.0)[None], jp.sin(ang / 2.0) * axis])
    w, x, y, z = q0[3], q0[4], q0[5], q0[6]
    w2, x2, y2, z2 = dq[0], dq[1], dq[2], dq[3]
    quat = jp.array([
        w * w2 - x * x2 - y * y2 - z * z2,
        w * x2 + x * w2 + y * z2 - z * y2,
        w * y2 - x * z2 + y * w2 + z * x2,
        w * z2 + x * y2 - y * x2 + z * w2])
    qpos = q0.at[3:7].set(quat / jp.linalg.norm(quat))
    rng, k = jax.random.split(rng)
    jn = float(cfg.get("exp08_init_joint_noise", 0.02))
    qpos = qpos.at[env._active_qpos_adr].add(
        jax.random.uniform(k, (12,), minval=-jn, maxval=jn))
    rng, k = jax.random.split(rng)
    zn = float(cfg.get("exp08_init_z_noise", 0.003))
    qpos = qpos.at[2].add(jax.random.uniform(k, minval=-zn, maxval=zn))
    qvel = jp.zeros(env._mjx_model.nv)
    return qpos, qvel, rng
