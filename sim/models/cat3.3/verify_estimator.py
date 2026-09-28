#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linvel 估计器 —— 闭环验证(不需要硬件)
=====================================
思路(运动学闭环):
  1) 给机器人一个已知的机体速度 v_true(机体系)
  2) 支撑腿条件: 足端相对世界静止 -> 反解关节速度 qdot = -J^-1 * v_true
  3) 用【估计器】(数值雅可比 + 支撑腿假设) 从 qdot 反算机体速度 v_est
  4) 对比 v_est 与 v_true -> 得到估计误差

验证 STM32 上要跑的完整链路: 关节角/速度 -> 足端速度 -> 机体线速度

运行: cd models\\cat3.3 && python verify_estimator.py
"""
import json
import numpy as np
import mujoco
from verify_fk_4legs import LEGS, fk, load_fourbar

SCENE = 'scene.xml'
DRIVE = {'leg1': ('joint1', 'joint2', 'joint3'),
         'leg2': ('joint6', 'joint7', 'joint8'),
         'leg3': ('joint11', 'joint12', 'joint13'),
         'leg4': ('joint16', 'joint17', 'joint18')}


def foot_pos_body(k, q1, q2, q3, qp):
    """自写 FK -> 足端在机体系的位置"""
    leg = 'leg%d' % (k + 1)
    cfg = LEGS[leg]
    ang = {cfg['chain'][0][2]: q1, cfg['chain'][1][2]: q2, cfg['chain'][-1][2]: qp}
    return fk(cfg['chain'], ang, cfg['foot_off'])


def num_jacobian(k, q1, q2, q3, passive_of):
    """数值雅可比 d足端/d(3驱动角); 扰动时同步更新被动角(四连杆)"""
    J = np.zeros((3, 3))
    eps = 1e-4
    for j, qv in enumerate((q1, q2, q3)):
        qq = [q1, q2, q3]
        qq[j] = qv + eps
        pp = foot_pos_body(k, qq[0], qq[1], qq[2], passive_of(qq[2]))
        qq[j] = qv - eps
        pm = foot_pos_body(k, qq[0], qq[1], qq[2], passive_of(qq[2]))
        J[:, j] = (pp - pm) / (2 * eps)
    return J


def main():
    m = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(m)
    adr = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    bid_base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')

    pose = {0: (np.deg2rad(5), np.deg2rad(-10), np.deg2rad(45)),
            1: (np.deg2rad(-5), np.deg2rad(-10), np.deg2rad(-45)),
            2: (np.deg2rad(3), np.deg2rad(8), np.deg2rad(60)),
            3: (np.deg2rad(-3), np.deg2rad(8), np.deg2rad(-60))}

    data.qpos[:] = m.qpos0.copy()
    for k in range(4):
        cr, A1, A2 = load_fourbar('leg%d' % (k + 1))
        d = DRIVE['leg%d' % (k + 1)]
        for j in range(3):
            data.qpos[adr(d[j])] = pose[k][j]
        data.qpos[adr('joint%d' % (4 + k * 5))] = np.interp(pose[k][2], cr, A1)
        data.qpos[adr('joint%d' % (5 + k * 5))] = np.interp(pose[k][2], cr, A2)
    mujoco.mj_forward(m, data)

    print('=== linvel 估计器闭环验证(4 腿平均) ===')
    print('真值v(机体系)            估计v                    误差')
    print('-' * 66)

    for v_in in [(0.30, 0, 0), (0, 0.15, 0), (0.20, -0.10, 0), (0.50, 0.20, 0)]:
        v_body = np.array(v_in, dtype=float)
        ests = []
        for k in range(4):
            cr, A1, A2 = load_fourbar('leg%d' % (k + 1))
            d = DRIVE['leg%d' % (k + 1)]
            pn = LEGS['leg%d' % (k + 1)]['chain'][-1][2]
            q1 = data.qpos[adr(d[0])]; q2 = data.qpos[adr(d[1])]; q3 = data.qpos[adr(d[2])]
            passive_of = lambda qc: np.interp(qc, cr, A2)
            J = num_jacobian(k, q1, q2, q3, passive_of)
            try:
                qdot = -np.linalg.solve(J, v_body)      # 支撑: 足端世界静止
            except np.linalg.LinAlgError:
                continue
            ests.append(-J @ qdot)                      # 估计器: 机体速度 = -J*qdot
        ests = np.array(ests)
        mean_est = ests.mean(0)
        err = np.linalg.norm(mean_est - v_body)
        print('[%5.2f %5.2f %5.2f]   [%5.2f %5.2f %5.2f]   %.5f m/s' % (
            v_body[0], v_body[1], v_body[2],
            mean_est[0], mean_est[1], mean_est[2], err))

    print()
    print('说明: 误差≈0 属数学闭环(反解与正算同一雅可比)。')
    print('真实精度需带: 真实步态 + 触地判定 + 关节噪声 -> 下一步做。')


if __name__ == '__main__':
    main()
