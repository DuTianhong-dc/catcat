# -*- coding: utf-8 -*-
"""EXP-08 域随机化：MJX 侧用 data.xfrc_applied 等效数据集侧的 model 级 DR。

数据集侧 (_expert_bc.py::_dr_apply) 改 model 的五项：
    body_mass[base] += m_add
    body_ipos[base] += (dx, dy, 0)
    geom_quat[floor] = q_slope  (+ 机身/初始位置同步旋转，相对地面姿态不变)
    geom_friction[floor] *= fric
    actuator_gainprm[:, 0] *= kp

MJX 的 model 在 vmap 下是单一常量，前三项改用 per-env 可写的 xfrc_applied
（CPU 与 MJX 的 xfrc 力都作用于 xipos=质心，源码级一致）：
    g_eff = R(q_slope)^T g0
    F_i   = body_mass_i * (g_eff - g0)          # 重力方向，严格等价
    F_base += m_add * g_eff                     # 挂载质量
    tau_base = (R_base·Δr) × ((M_base+m_add)·g_eff)   # 质心偏移力偶

后两项（摩擦 / kp）是 model 常量，MJX 无法 per-env 改 -> 训练侧固定标称值，
作为已知缺口记录（实验记录 §7.31）。这是实现限制，不是遗漏。

精度实测见 evidence/probe_dr_equiv.log：
    重力方向        严格等价，max|Δqacc| = 3.6e-15
    挂载+质心 1 拍  max|Δqpos@10 物理步| = 8.5e-06
    250 拍有接触    up_z 轨迹最大偏差 0.035 / qpos 末态 4.2e-04
    惯性矩阵改不了（5% 质量差）：无接触时会被放大，故起点必须贴地。
"""
import jax
import jax.numpy as jp


def _quat_to_R(q):
    w, x, y, z = q
    return jp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def sample_dr(cfg, rng):
    """采样一回合的 DR 参数；返回 (params, rng)。

    与 _expert_bc.py::_sim_one 同口径：m_add ~ U(0, mass_max)；
    dx = dx_bias + m_add*(x_add-x_com0)/(m0+m_add)；dy 左右成对随机；
    坡度 theta ~ U(0, slope_max)、方位 phi ~ U(0, 2pi)。
    """
    rng, k = jax.random.split(rng)
    m_add = jax.random.uniform(k, minval=0.0, maxval=float(cfg.exp08_mass_max))
    rng, k = jax.random.split(rng)
    dx_bias = jax.random.uniform(
        k, minval=-float(cfg.exp08_dx_bias), maxval=float(cfg.exp08_dx_bias))
    dx = dx_bias + m_add * (float(cfg.exp08_x_add) - float(cfg.exp08_x_com0)) \
        / (float(cfg.exp08_m0) + m_add)
    dx_ref = float(cfg.exp08_mass_max) * (
        float(cfg.exp08_x_add) - float(cfg.exp08_x_com0)) \
        / (float(cfg.exp08_m0) + float(cfg.exp08_mass_max))
    dy_amp = float(cfg.exp08_dy_ratio) * dx_ref
    rng, k = jax.random.split(rng)
    dy_abs = jax.random.uniform(k, minval=0.0, maxval=dy_amp)
    rng, k = jax.random.split(rng)
    dy = jp.where(jax.random.uniform(k) < 0.5, -dy_abs, dy_abs)
    rng, k = jax.random.split(rng)
    theta = jax.random.uniform(
        k, minval=0.0, maxval=jp.deg2rad(float(cfg.exp08_slope_max_deg)))
    rng, k = jax.random.split(rng)
    phi = jax.random.uniform(k, minval=0.0, maxval=2.0 * jp.pi)
    axis = jp.array([-jp.sin(phi), jp.cos(phi), 0.0])
    qs = jp.concatenate(
        [jp.cos(theta / 2.0)[None], jp.sin(theta / 2.0) * axis])
    g0 = jp.array([0.0, 0.0, -9.81], dtype=jp.float32)
    g_eff = _quat_to_R(qs).T @ g0
    return {"m_add": m_add, "dx": dx, "dy": dy, "g_eff": g_eff,
            "slope_theta": theta, "slope_phi": phi, "q_slope": qs}, rng


def dr_xfrc(model, data, dr, base_id):
    """返回本 ctrl 拍要叠加到 data.xfrc_applied 上的 (nbody, 6) 外力。

    力作用于 xipos（质心），力偶用【当前】机身姿态算，10 个物理子步内恒定。
    与 pert_config 的 xfrc 叠加而非覆盖，调用方用 += 注入。
    """
    g0 = model.opt.gravity
    dg = dr["g_eff"] - g0
    F = model.body_mass[:, None] * dg[None, :]
    F = F.at[base_id].add(dr["m_add"] * dr["g_eff"])
    dr_w = data.xmat[base_id] @ jp.array(
        [dr["dx"], dr["dy"], 0.0], dtype=jp.float32)
    F_new = (model.body_mass[base_id] + dr["m_add"]) * dr["g_eff"]
    tau = jp.cross(dr_w, F_new)
    tor = jp.zeros((model.nbody, 3), dtype=jp.float32).at[base_id].set(tau)
    return jp.concatenate([F, tor], axis=1)
