"""后腿 (j2, j3) 双层遍历 (cat3.4) —— 按煜辰思路的最简版
每格: ① 解机身 (z, 俯仰) 让【两个前脚落地】(2 方程 2 未知)
      ② 高度判据: 后脚 z 误差(够不着地 = 站不住) + 机身高度(太低 = 摔/塌)
      ③ 承重: 解析载荷分配, Σ 必须 = 总重力 W; 看四条腿各分多少
      ④ 扭矩: 后腿 j2/j3 的静力矩 τ = JᵀF  (J = 足端对关节的雅可比)
右后腿 = 左后镜像 (j7 = -j2, j8 = -j3)
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


FEET = ['foot_rl', 'foot_rr', 'foot_fl', 'foot_fr']


def fk(q):
    """注意: 不能用 D 做快照(同一对象会被后续调用覆盖) —— 只返回值副本"""
    D.qpos[:] = q
    mujoco.mj_kinematics(M, D)
    return np.array([D.site_xpos[sid(f)][2] for f in FEET]), np.array([D.site_xpos[sid(f)] for f in FEET])


def solve_body(j2, j3):
    """解 (z, 俯仰, 横滚) 让 4 只脚尽量都落地 —— 4 方程 3 未知, 用最小二乘
    返回 (z, bp, br, 最大足端 z 误差[前4只中的最大])"""
    v = np.array([BASE[2], 0.0, 0.0])
    for _ in range(80):
        z, _ = fk(build(j2, j3, v[0], v[1], v[2]))
        e = z - GROUND
        if np.abs(e).max() < 1e-11:
            break
        eps = 1e-5
        J = np.zeros((4, 3))
        for i in range(3):
            vv = v.copy(); vv[i] += eps
            z2, _ = fk(build(j2, j3, vv[0], vv[1], vv[2]))
            J[:, i] = (z2 - z) / eps
        # 阻尼最小二乘(应对并排前脚的秩亏)
        A = np.vstack([J, 1e-4 * np.eye(3)])
        bvec = np.concatenate([-e, np.zeros(3)])
        st, *_ = np.linalg.lstsq(A, bvec, rcond=None)
        v[0] = float(np.clip(v[0] + np.clip(st[0], -0.01, 0.01), 0.075, 0.155))
        v[1] = float(np.clip(v[1] + np.clip(st[1], -0.02, 0.02), -0.18, 0.18))
        v[2] = float(np.clip(v[2] + np.clip(st[2], -0.02, 0.02), -0.12, 0.12))
    z, _ = fk(build(j2, j3, v[0], v[1], v[2]))
    e = np.abs(z - GROUND)
    e_front = max(e[2], e[3])
    e_rear = max(e[0], e[1])
    if e_front > 1e-3:
        return None
    return v[0], v[1], e_rear


def loads_and_tau(j2, j3, bz, bp):
    D.qpos[:] = build(j2, j3, bz, bp)
    mujoco.mj_kinematics(M, D)
    P = np.array([D.site_xpos[sid(f)] for f in FEET])
    com = np.zeros(3); mt = 0.0
    for k in range(1, M.nbody):
        com += M.body_mass[k] * D.xipos[k]; mt += M.body_mass[k]
    com /= mt
    A = np.array([[1., 1., 1., 1.],
                  [P[0][1] - com[1], P[1][1] - com[1], P[2][1] - com[1], P[3][1] - com[1]],
                  [-(P[0][0] - com[0]), -(P[1][0] - com[0]), -(P[2][0] - com[0]), -(P[3][0] - com[0])]])
    f, *_ = np.linalg.lstsq(A, np.array([W, 0., 0.]), rcond=None)   # A f = b, 4 未知 3 方程 → 最小范数解
    mujoco.mj_forward(M, D)          # jacSite 需要 comPos/雅可比缓冲
    tau = {}
    for jn, fz in (('joint2', f[0]), ('joint3', f[0])):
        jacp = np.zeros((3, M.nv)); jacr = np.zeros((3, M.nv))
        mujoco.mj_jacSite(M, D, jacp, jacr, sid('foot_rl'))
        tau[jn] = float(jacp[2, dof(jn)] * fz)
    return f, tau, P


if __name__ == '__main__':
    rows = []
    for j3 in np.arange(0.10, 2.96, 0.05):
        for j2 in np.arange(-1.15, 0.51, 0.05):
            r = solve_body(j2, j3)
            if r is None:
                continue
            bz, bp, e_rear = r
            f, tau, P = loads_and_tau(j2, j3, bz, bp)
            rows.append((j2, j3, bz, bp, e_rear, *f, tau['joint2'], tau['joint3']))
    a = np.array(rows)
    print('遍历 %d 组, 前脚可落地的 %d 组' % (len(np.arange(0.10, 2.96, 0.05)) * len(np.arange(-1.15, 0.51, 0.05)), len(a)))
    if len(a) == 0:
        raise SystemExit('全部不可行 — 前脚都落不了地')
    stand = a[np.abs(a[:, 4]) < 1e-3]                       # 后脚也要落地
    print('四脚同时落地(后脚误差<1mm): %d 组' % len(stand))
    if len(stand):
        pos = stand[(stand[:, 5] > 0) & (stand[:, 6] > 0) & (stand[:, 7] > 0) & (stand[:, 8] > 0)]
        print('四腿承重全为正: %d 组' % len(pos))
        if len(pos):
            good = pos[(pos[:, 2] > 0.085) & (pos[:, 2] < 0.135)]
            print('机身高度 85-135mm: %d 组' % len(good))
            if len(good):
                bal = np.abs(good[:, 5] + good[:, 6] - (good[:, 7] + good[:, 8]))
                trk = np.abs(good[:, 9]) + np.abs(good[:, 10])
                k = np.lexsort((trk, bal))[:14]
                print()
                print('   j2      j3     机身z   俯仰°  |  f后左  f后右  f前左  f前右 |  τ_j2   τ_j3')
                for i in k:
                    r = good[i]
                    print('  %+.3f  %+.3f  %.4f  %+.2f | %5.2f %5.2f %5.2f %5.2f | %+.3f %+.3f'
                          % (r[0], r[1], r[2], np.degrees(r[3]), r[5], r[6], r[7], r[8], r[9], r[10]))
                np.save('_scan_stand.npy', stand)
                print()
                print('(全部四脚落地解已存 _scan_stand.npy, %d 行)' % len(stand))
    np.save('_scan_all.npy', a)
