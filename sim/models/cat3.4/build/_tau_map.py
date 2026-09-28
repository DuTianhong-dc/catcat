"""(j2,j3) 扭矩地图 + 邻域最坏扭矩 (cat3.4, 2026-09-13)
判据(煜辰定): 不看待承重, 看【后腿扭矩】——
  合理 = 该站姿附近"可达范围"内的扭矩尽可能小 (走起来任意摆腿位置都不吃满)
做法:
  ① 每个 (j2,j3): 解机身(z,俯仰,横滚) 使四脚落地
  ② 完整 6 自由度力平衡: 4 接触 × 3 分量 = 12 未知, 6 平衡方程 → 最小范数解
  ③ 后腿扭矩 τ = JᵀF (j2 髋俯仰 / j3 膝曲柄)
  ④ 邻域最坏扭矩 = max(|τ|) over (j2±W2, j3±W3) 窗口内所有可行点
"""
import numpy as np
import mujoco

P4 = [-0.001650620535, -0.4535536832, -0.0502349092, 0.01883377116, -0.03557780823]
P5 = [-0.003411754794, -0.2084629358, -0.287239521, 0.1602706708, -0.06433772677]
P9 = [0.001650508856, -0.4535548943, 0.0502290454, 0.01882795195, 0.03557624523]
P10 = [0.003413645749, -0.2084495883, 0.2872821059, 0.1603062374, 0.06434638416]
poly = lambda c, v: c[0] + c[1] * v + c[2] * v ** 2 + c[3] * v ** 3 + c[4] * v ** 4

M = mujoco.MjModel.from_xml_path('scene.xml')
D = mujoco.MjData(M)
adr = lambda n: M.jnt_qposadr[mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_JOINT, n)]
dof = lambda n: M.jnt_dofadr[mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_JOINT, n)]
sid = lambda n: mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_SITE, n)
mujoco.mj_resetDataKeyframe(M, D, mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_KEY, 'stand'))
BASE = D.qpos.copy()
GROUND = -0.0700
W = float(M.body_mass.sum()) * 9.81
FEET = ['foot_rl', 'foot_rr', 'foot_fl', 'foot_fr']


def build(j2, j3, bz, bp, br=0.0):
    q = BASE.copy()
    q[2] = bz
    hp, hr = bp / 2.0, br / 2.0
    q[3] = np.cos(hp) * np.cos(hr)
    q[4] = np.cos(hp) * np.sin(hr)
    q[5] = np.sin(hp) * np.cos(hr)
    q[6] = -np.sin(hp) * np.sin(hr)
    q[adr('joint2')], q[adr('joint3')] = j2, j3
    q[adr('joint4')], q[adr('joint5')] = poly(P4, j3), poly(P5, j3)
    q[adr('joint7')], q[adr('joint8')] = -j2, -j3
    q[adr('joint9')], q[adr('joint10')] = poly(P9, -j3), poly(P10, -j3)
    return q


def feet_z(q):
    D.qpos[:] = q
    mujoco.mj_kinematics(M, D)
    return np.array([D.site_xpos[sid(f)][2] for f in FEET])


def solve_body(j2, j3):
    v = np.array([BASE[2], 0.0, 0.0])
    for _ in range(80):
        z = feet_z(build(j2, j3, v[0], v[1], v[2]))
        e = z - GROUND
        if np.abs(e).max() < 1e-11:
            break
        eps = 1e-5
        J = np.zeros((4, 3))
        for i in range(3):
            vv = v.copy(); vv[i] += eps
            J[:, i] = (feet_z(build(j2, j3, vv[0], vv[1], vv[2])) - z) / eps
        A = np.vstack([J, 1e-4 * np.eye(3)])
        bv = np.concatenate([-e, np.zeros(3)])
        st, *_ = np.linalg.lstsq(A, bv, rcond=None)
        v[0] = float(np.clip(v[0] + np.clip(st[0], -0.01, 0.01), 0.075, 0.155))
        v[1] = float(np.clip(v[1] + np.clip(st[1], -0.02, 0.02), -0.18, 0.18))
        v[2] = float(np.clip(v[2] + np.clip(st[2], -0.02, 0.02), -0.12, 0.12))
    z = feet_z(build(j2, j3, v[0], v[1], v[2]))
    if max(abs(z[2] - GROUND), abs(z[3] - GROUND)) > 1e-3:
        return None
    return v


def torques(j2, j3, v):
    """完整 6 自由度力平衡(12 未知 6 方程) → 后腿 j2/j3 扭矩"""
    D.qpos[:] = build(j2, j3, v[0], v[1], v[2])
    mujoco.mj_forward(M, D)
    P = np.array([D.site_xpos[sid(f)] for f in FEET])
    com = np.zeros(3); mt = 0.0
    for k in range(1, M.nbody):
        com += M.body_mass[k] * D.xipos[k]; mt += M.body_mass[k]
    com /= mt
    # 每个足 i 的力 F_i ∈ R³; 平衡: ΣF = (0,0,W), Σ (p_i - com) × F_i = 0
    A = np.zeros((6, 12))
    for i in range(4):
        A[0:3, 3 * i:3 * i + 3] = np.eye(3)
        r = P[i] - com
        A[3, 3 * i + 1] = -r[2]; A[3, 3 * i + 2] = r[1]
        A[4, 3 * i + 0] = r[2];  A[4, 3 * i + 2] = -r[0]
        A[5, 3 * i + 0] = -r[1]; A[5, 3 * i + 1] = r[0]
    b = np.array([0, 0, W, 0, 0, 0], dtype=float)
    F, *_ = np.linalg.lstsq(A, b, rcond=None)       # 最小范数接触力
    F = F.reshape(4, 3)
    jacp = np.zeros((3, M.nv)); jacr = np.zeros((3, M.nv))
    mujoco.mj_jacSite(M, D, jacp, jacr, sid('foot_rl'))
    tau2 = float(jacp[:, dof('joint2')] @ F[0])
    tau3 = float(jacp[:, dof('joint3')] @ F[0])
    return tau2, tau3, F


if __name__ == '__main__':
    rows = []
    for j3 in np.arange(0.10, 2.96, 0.05):
        for j2 in np.arange(-1.15, 0.51, 0.05):
            v = solve_body(j2, j3)
            if v is None:
                continue
            t2, t3, F = torques(j2, j3, v)
            rows.append((j2, j3, v[0], v[1], t2, t3, abs(t2), abs(t3)))
    a = np.array(rows)
    print('可行(四脚落地) %d 组' % len(a))
    np.save('_tau_map.npy', a)

    # 邻域最坏扭矩: 窗口 ±0.2 rad(j2) / ±0.3 rad(j3)
    W2, W3 = 0.20, 0.30
    worst = []
    for r in a:
        m = a[(np.abs(a[:, 0] - r[0]) <= W2) & (np.abs(a[:, 1] - r[1]) <= W3)]
        worst.append(np.abs(m[:, 4]).max() + np.abs(m[:, 5]).max())
    worst = np.array(worst)
    a2 = np.column_stack([a, worst])
    np.save('_tau_map_worst.npy', a2)
    k = np.argsort(worst)[:15]
    print()
    print('按【邻域最坏扭矩(±0.20/±0.30 rad)】排序前 15:')
    print('   j2      j3    | 该点 τ_j2  τ_j3 | 邻域最坏 | 髋余量')
    for i in k:
        r = a[i]
        print('  %+.3f  %+.3f | %+.3f %+.3f | %+.3f  | %.1f°'
              % (r[0], r[1], r[4], r[5], worst[i], np.degrees(0.52 - r[0])))
    print()
    print('对比: 旧站姿(0.337,1.204)在该窗口内', end=' ')
    m = a[(np.abs(a[:, 0] - 0.337) <= W2) & (np.abs(a[:, 1] - 1.204) <= W3)]
    print('最坏 %.3f N·m (有效点 %d)' % (np.abs(m[:, 4]).max() + np.abs(m[:, 5]).max(), len(m)))
