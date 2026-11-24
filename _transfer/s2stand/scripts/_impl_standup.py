# -*- coding: utf-8 -*-
r"""Lee 2019 第二个行为「standing-up（趴姿 → 站起来）」的环境补丁。

⚠️ 2026-11-05 状态：**只做静态校验，尚未执行**（EXP-01 占着 GPU，执行后必须跑
   等价性回归 30/30，而回归会抢显存）。默认 `--check` 只读；`--apply` 才写文件。

论文依据（p5 II-D / p5 Table I / p5 II-D 2) Action Representation）：
  · 成本项 7 项：kjpcjp + koco + khch + kaca + kadcad + kjslimcjslim + kτcτ
    = self-righting 9 项去掉 c_bi / c_bs / c_c,in，新增 height cost c_h；
  · 目标关节配置 = **standing configuration**（站姿，相对零位全 0）；
  · 初始分布 = 与 self-righting 同策略但 **near-upright pose**（任意坐姿，不投落）；
  · 终止 = 只有时间限制（同 self-righting）；
  · 动作 = φd = k·o + φt（相对当前角增量），与 self-righting 相同 → 沿用 delta_current；
  · 网络 = 2 层 tanh 128/128 → 沿用 net_spec lee2019_behavior；
  · 状态空间 = self-righting 状态空间 + **base linear velocity in body frame**
    （base height 只进 Locomotion 的观测，不进 standing-up；但 c_h 作为成本项
     在仿真里可直接用真值，二者不冲突 —— 见 实验记录.md §3）。

实现（只改副本三个文件，主线 joystick.py 零改动）：
  1) task.py   : INIT_MODES 加 "sit"，initial_state 加 sit 分支（近正立四元数 +
                 坐姿关节角 + 小扰动 + 几何贴地），复用 fall 的净空逻辑；
  2) rewards.py: TERM_SPECS / COST_TERMS 加 "c_height"；paper_costs 用
                 q_target 代替 q_sit，并加 c_h = max(0, h_target - h)^2（自定公式）；
  3) defaults.py: 加 q_target / height_target / sit_* 配置键，默认值保证
                 默认行为与主线逐字等价（q_target 默认 == q_sit）。
"""
import io
import shutil
import sys
import time

BASE = "/home/dth15/mujoco_playground/mujoco_playground/_src/locomotion/cat40/"
P_TASK = BASE + "lee2019/task.py"
P_REW = BASE + "lee2019/rewards.py"
P_DEF = BASE + "lee2019/defaults.py"


def patch(check=True):
    tag = time.strftime("%H%M")
    ok = []
    files = {}

    # ---------------- 1) task.py ----------------
    s = io.open(P_TASK, encoding="utf-8").read()
    old = 'INIT_MODES = ("stand", "fall")          # fall = 论文式 0.5m/随机四元数/随机关节角 投落\n'
    new = 'INIT_MODES = ("stand", "fall", "sit")  # sit = 论文 II-D standing-up：近正立 + 任意坐姿\n'
    assert s.count(old) == 1, "task: INIT_MODES 锚点"
    s = s.replace(old, new, 1)

    anchor = '    if mode != "stand":\n        raise ValueError("未知 init_mode: %s" % mode)\n'
    assert s.count(anchor) == 1, "task: stand 分支锚点"
    sit = '''    if mode == "sit":
        # 论文 II-D standing-up：从任意坐姿站起，起始为 near-upright pose。
        # 与 self-righting 的区别：四元数接近单位（不翻转）、关节角围绕**坐姿**小扰动、
        # 不做 0.5 m 投落。几何贴地逻辑与 fall 相同，保证不穿地。
        cfg = env._config.task_config
        qpos = env._init_q
        rng, k = jax.random.split(rng)
        _tm = float(cfg.get("sit_tilt_max", 0.15))
        _t = jax.random.uniform(k, (2,), minval=-_tm, maxval=_tm)
        _q = jp.array([1.0, 0.5 * _t[0], 0.5 * _t[1], 0.0])
        qpos = qpos.at[3:7].set(_q / jp.linalg.norm(_q))
        rng, k = jax.random.split(rng)
        _n = float(cfg.get("sit_joint_noise", 0.08))
        _qs = jp.array(cfg.get("sit_qpos", [0.0] * 12)) + jax.random.uniform(
            k, (12,), minval=-_n, maxval=_n)
        qpos = qpos.at[env._active_qpos_adr].set(
            jp.clip(_qs, jp.array(env._act_lo), jp.array(env._act_hi)))
        qpos = qpos.at[2].set(float(cfg.get("sit_height", 0.0416)))
        rng, k = jax.random.split(rng)
        qpos = qpos.at[0:2].set(jax.random.uniform(k, (2,), minval=-0.05, maxval=0.05))
        qvel = jp.zeros(env._mjx_model.nv)
        _clear = float(cfg.get("sit_clearance", 0.0))
        if _clear >= 0.0:
            _d0 = mjx.make_data(env._mjx_model)
            _d0 = _d0.replace(qpos=qpos, qvel=qvel, ctrl=env._default_ctrl)
            _d0 = mjx.forward(env._mjx_model, _d0)
            _lo = _d0.geom_xpos[:, 2] - env._mjx_model.geom_rbound
            _fid = int(env._mj_model.geom("floor").id)
            _lo = _lo.at[_fid].set(1.0e6)
            qpos = qpos.at[2].set(qpos[2] - jp.min(_lo) + _clear)
        return qpos, qvel, rng
'''
    s = s.replace(anchor, sit + anchor, 1)
    files[P_TASK] = s
    ok.append("task.py: INIT_MODES + sit 分支")

    # ---------------- 2) rewards.py ----------------
    r = io.open(P_REW, encoding="utf-8").read()
    o = '    ("c_torque", 0.0002),\n]'
    n = ('    ("c_torque", 0.0002),\n'
         '    # 2026-11-05 standing-up：论文 p5 新增的高度成本 khch（默认 0 = 不启用）\n'
         '    ("c_height", 0.0),\n]')
    assert r.count(o) == 1, "rewards: TERM_SPECS 锚点"
    r = r.replace(o, n, 1)

    o = 'COST_TERMS = ["c_orientation", "c_joint_pos", "c_joint_accel",\n              "c_body_impulse", "c_body_slip", "c_self_collision",\n              "c_action_diff", "c_joint_speed", "c_torque"]'
    n = ('COST_TERMS = ["c_orientation", "c_joint_pos", "c_joint_accel",\n'
         '              "c_body_impulse", "c_body_slip", "c_self_collision",\n'
         '              "c_action_diff", "c_joint_speed", "c_torque", "c_height"]')
    assert r.count(o) == 1, "rewards: COST_TERMS 锚点"
    r = r.replace(o, n, 1)

    o = '    q_sit = jp.array(cfg.q_sit)\n'
    n = ('    # 2026-11-05: q_target 允许 standing-up 把目标换成站姿；\n'
         '    # 默认等于 q_sit，self-righting 配方行为不变（等价性回归保证）。\n'
         '    q_target = jp.array(cfg.get("q_target", cfg.q_sit))\n')
    assert r.count(o) == 1, "rewards: q_sit 锚点"
    r = r.replace(o, n, 1)

    o = '    e_jp = jp.sqrt(jp.sum(jp.square(q - q_sit)) + 1e-8)\n'
    n = '    e_jp = jp.sqrt(jp.sum(jp.square(q - q_target)) + 1e-8)\n'
    assert r.count(o) == 1, "rewards: e_jp 锚点"
    r = r.replace(o, n, 1)

    o = '    # c_τ: 力矩平方\n    c_tau = jp.sum(jp.square(tau))\n'
    n = ('    # c_τ: 力矩平方\n    c_tau = jp.sum(jp.square(tau))\n'
         '    # c_h: 高度成本（论文未给公式/权重，自定）：只罚低于站姿高度的部分。\n'
         '    # 用途是【连续成本项】，与「二值高度分类判据」不同 —— 后者已证不可用。\n'
         '    _h = data.qpos[2] - env._floor_z\n'
         '    c_h = jp.square(jp.maximum(\n'
         '        0.0, float(cfg.get("height_target", 0.147)) - _h))\n')
    assert r.count(o) == 1, "rewards: c_tau 锚点"
    r = r.replace(o, n, 1)

    o = '        "c_joint_speed": c_jslim, "c_torque": c_tau,\n    }'
    n = '        "c_joint_speed": c_jslim, "c_torque": c_tau, "c_height": c_h,\n    }'
    assert r.count(o) == 1, "rewards: out dict 锚点"
    r = r.replace(o, n, 1)
    files[P_REW] = r
    ok.append("rewards.py: c_height + q_target")

    # ---------------- 3) defaults.py ----------------
    d = io.open(P_DEF, encoding="utf-8").read()
    o = '    c.reward_config.q_sit = [0.0] * 12\n'
    n = ('    c.reward_config.q_sit = [0.0] * 12\n'
         '    # 2026-11-05 standing-up：c_jp 的目标关节配置（站姿）。默认 == q_sit\n'
         '    # 保证 self-righting 行为不变；配方覆盖为 [0]*12 即站姿目标。\n'
         '    c.reward_config.q_target = [0.0] * 12\n'
         '    c.reward_config.height_target = 0.147  # 站姿基座高度实测 0.1473\n')
    assert d.count(o) == 1, "defaults: q_sit 锚点"
    d = d.replace(o, n, 1)

    o = '        joint_range_scale=0.8)    # 随机关节角占 jnt_range 的比例，留 20% 余量防顶限位\n'
    n = ('        joint_range_scale=0.8,   # 随机关节角占 jnt_range 的比例，留 20% 余量防顶限位\n'
         '        # ---- standing-up 初始分布（init_mode="sit" 时才生效）----\n'
         '        sit_qpos=[0.0] * 12,     # 配方从 sit_pose_touch.json 覆盖\n'
         '        sit_height=0.0416,      # 定稿趴姿机身高度（底面触地）\n'
         '        sit_tilt_max=0.15,      # near-upright：四元数 x/y 分量扰动上限(rad)\n'
         '        sit_joint_noise=0.08,   # 任意坐姿：关节角扰动幅度(rad)\n'
         '        sit_clearance=0.0)      # 几何贴地净空，0 = 最低点正好贴地\n')
    assert d.count(o) == 1, "defaults: task_config 锚点"
    d = d.replace(o, n, 1)
    files[P_DEF] = d
    ok.append("defaults.py: q_target/height_target/sit_*")

    print("静态校验通过，%d 处锚点全部唯一命中：" % len(ok))
    for line in ok:
        print("  ✓", line)
    if check:
        print("\n[--check] 未写任何文件。EXP-01 训练结束后执行 `--apply`，\n"
              "然后跑 play/_check_lee2019_equiv.py --check 确认 30/30 等价。")
        return 0
    for p, text in files.items():
        shutil.copy(p, p + ".bak_standup_" + tag)
        io.open(p, "w", encoding="utf-8", newline="\n").write(text)
    print("\n[--apply] 已写入，备份后缀 .bak_standup_%s" % tag)
    return 0


if __name__ == "__main__":
    sys.exit(patch(check="--apply" not in sys.argv))
