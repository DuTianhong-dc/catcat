#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
足端标定: 用 POL_DEFAULT_POSE(默认站姿) 标定"世界Z最低点"在 Link5 的局部偏移
=====================================================================
用户定夺: 足端 = 世界 Z 方向的最低点(真正触地的点)

做法:
  1) 设 POL_DEFAULT_POSE + 四连杆一致(被动角查表)
  2) 把 Link5_hull 碰撞网格所有顶点转到世界系
  3) 取世界 z 最小的顶点 -> 反算它在 Link5 局部的坐标 -> 作为固定足端偏移
  4) 验证: 四条腿的足端世界 z 是否接近(站立时都应贴地 z≈0)

运行: cd C:\\Users\\dth15\\Desktop\\mujoco\\models\\cat3.3 && python calibrate_foot.py
"""
import json
import numpy as np
import mujoco

SCENE = 'scene.xml'
FOURBAR = 'fourbar_map_4legs_pure.json'

# POL_DEFAULT_POSE(来自 CAT_APP\policy_weights.c): 右后/左后/前右/前左, 各 髋侧摆/髋俯仰/膝
POSE = [0, -0.0476000011, 1.03999996,
        0, 0.0476000011, -1.03999996,
        0, 0, 0,
        0, 0, 0]

# 四条腿对应的足端 body 与四连杆表名
LEGS = [('Link5', 'leg1'), ('Link10', 'leg2'), ('Link15', 'leg3'), ('Link20', 'leg4')]
HULLS = ['Link5_hull', 'Link10_hull', 'Link15_hull', 'Link20_hull']


def load_fourbar(leg):
    d = json.load(open(FOURBAR, encoding='utf-8'))
    rows = d[leg]['rows']
    cr = np.array([r['q_crank'] for r in rows])
    a1 = np.array([r['q1'] for r in rows])
    a2 = np.array([r['q2'] for r in rows])
    o = np.argsort(cr)
    return np.deg2rad(cr[o]), np.deg2rad(a1[o]), np.deg2rad(a2[o])


def main():
    m = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(m)

    adr = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]

    # 1) 设驱动角 + 四连杆被动角(base 保持 qpos0)
    data.qpos[:] = m.qpos0.copy()
    for k in range(4):
        cr, A1, A2 = load_fourbar('leg%d' % (k + 1))
        for j in range(3):
            data.qpos[adr('joint%d' % (1 + k * 5 + j))] = POSE[k * 3 + j]
        qc = POSE[k * 3 + 2]                    # 膝 = 曲柄角
        data.qpos[adr('joint%d' % (4 + k * 5))] = np.interp(qc, cr, A1)
        data.qpos[adr('joint%d' % (5 + k * 5))] = np.interp(qc, cr, A2)
    mujoco.mj_forward(m, data)

    print('=== 用 POL_DEFAULT_POSE 站姿标定足端(世界Z最低点) ===\n')
    results = {}
    for (bname, fname), hname in zip(LEGS, HULLS):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, hname)
        if gid < 0:
            print('  %s: 找不到碰撞体, 跳过' % bname)
            continue
        mid = m.geom_dataid[gid]
        va, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = m.mesh_vert[va:va + nv].reshape(-1, 3)
        gp = m.geom_pos[gid]
        R = data.xmat[bid].reshape(3, 3)
        t = data.xpos[bid]
        world = (R @ (v + gp).T).T + t          # 顶点世界坐标
        i = world[:, 2].argmin()                 # 世界 z 最小
        local = v[i] + gp                        # 反算: 该顶点在 body 局部的坐标
        results[bname] = local
        print('  %-7s 世界最低点=[%7.4f %7.4f %7.4f]  z=%+.4f' %
              (bname, world[i][0], world[i][1], world[i][2], world[i][2]))
        print('           足端局部偏移 = [%7.4f %7.4f %7.4f]' % tuple(local))

    print('\n=== 一致性检查(四条腿的局部偏移应接近, 镜像腿符号可能相反) ===')
    for b, v in results.items():
        print('  %-7s %s' % (b, np.round(v, 4)))

    print('\n=== 验证: 用 leg1 的足端偏移算四条腿的足端世界 z(应都接近) ===')
    if 'Link5' in results:
        off = results['Link5']
        for (bname, fname) in LEGS:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
            R = data.xmat[bid].reshape(3, 3)
            p = data.xpos[bid] + R @ off
            print('  %-7s 足端世界 z = %+.4f' % (bname, p[2]))


if __name__ == '__main__':
    main()
