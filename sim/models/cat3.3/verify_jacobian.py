#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
足端雅可比验证: 数值雅可比(STM32 要跑的算法) vs MuJoCo 解析雅可比(真值)
=====================================================================
足端速度 v_foot = J * qdot
  数值 J: 扰动某个驱动角(并同步查四连杆表更新被动角) -> 足端位置差分
  解析 J: mj_jacBody 取 Link5 的 jacp/jacr, 再用 v = v_p + w x r 修正到足端点

运行: cd models\\cat3.3 && python verify_jacobian.py
"""
import json
import numpy as np
import mujoco
from verify_fk_4legs import LEGS, fk, qrot, qmul, rz, load_fourbar, qconj

SCENE = 'scene.xml'

# 四条腿的驱动关节名(髋侧摆/髋俯仰/膝)与被动关节名(曲柄-连杆/生理膝)
DRIVE = {
    'leg1': ('joint1', 'joint2', 'joint3'),
    'leg2': ('joint6', 'joint7', 'joint8'),
    'leg3': ('joint11', 'joint12', 'joint13'),
    'leg4': ('joint16', 'joint17', 'joint18'),
}
PASSIVE = {
    'leg1': ('joint4', 'joint5'),
    'leg2': ('joint9', 'joint10'),
    'leg3': ('joint14', 'joint15'),
    'leg4': ('joint19', 'joint20'),
}


def set_leg(m, data, k, q1, q2, q3):
    """设第 k 条腿的 3 个驱动角, 被动角由四连杆表给出"""
    leg = 'leg%d' % (k + 1)
    cr, A1, A2 = load_fourbar(leg)
    adr = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    d = DRIVE[leg]
    data.qpos[adr(d[0])] = q1
    data.qpos[adr(d[1])] = q2
    data.qpos[adr(d[2])] = q3
    p = PASSIVE[leg]
    data.qpos[adr(p[0])] = np.interp(q3, cr, A1)
    data.qpos[adr(p[1])] = np.interp(q3, cr, A2)


def foot_body_pos(m, data, k, q1, q2, q3, base_pos, base_quat_inv):
    """自写 FK -> 足端在【机体系】的位置(给定该腿的 3 个驱动角)"""
    leg = 'leg%d' % (k + 1)
    cr, A1, A2 = load_fourbar(leg)
    adr = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    cfg = LEGS[leg]
    # 被动角(随曲柄角 q3 变化)
    pn = cfg['chain'][-1][2]                    # 链最后一级的关节(生理膝)
    ang = {DRIVE[leg][0]: q1, DRIVE[leg][1]: q2,
           pn: np.interp(q3, cr, A2)}
    return fk(cfg['chain'], ang, cfg['foot_off'])


def main():
    m = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(m)
    adr = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    bid_base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')

    # 测试姿态
    pose = {0: (np.deg2rad(5), np.deg2rad(-10), np.deg2rad(45)),
            1: (np.deg2rad(-5), np.deg2rad(-10), np.deg2rad(-45)),
            2: (np.deg2rad(3), np.deg2rad(8), np.deg2rad(60)),
            3: (np.deg2rad(-3), np.deg2rad(8), np.deg2rad(-60))}

    data.qpos[:] = m.qpos0.copy()
    for k, qs in pose.items():
        set_leg(m, data, k, *qs)
    mujoco.mj_forward(m, data)

    bp = data.xpos[bid_base]
    bqinv = qconj(tuple(data.xquat[bid_base]))

    print('足端雅可比验证 (d足端/d各驱动角, 机体系, 单位 m/rad)')
    print('=' * 76)
    eps = 1e-4
    for k in range(4):
        leg = 'leg%d' % (k + 1)
        cfg = LEGS[leg]
        q1, q2, q3 = pose[k]
        print('\n[%s]  足端 body=%s' % (leg, cfg['foot_body']))

        # ---- 数值雅可比(自写 FK + 四连杆) ----
        num = np.zeros((3, 3))
        for j, qv in enumerate((q1, q2, q3)):
            qq = [q1, q2, q3]
            qq[j] = qv + eps
            pp = foot_body_pos(m, data, k, *qq, bp, bqinv)
            qq[j] = qv - eps
            pm = foot_body_pos(m, data, k, *qq, bp, bqinv)
            num[:, j] = (pp - pm) / (2 * eps)

        # ---- 解析雅可比(MuJoCo, 真值) ----
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cfg['foot_body'])
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, data, jacp, jacr, bid)
        R = data.xmat[bid].reshape(3, 3)
        r_world = R @ np.array(cfg['foot_off'])         # 足端相对 body 原点的世界矢量
        rx = np.array([[0, -r_world[2], r_world[1]],
                       [r_world[2], 0, -r_world[0]],
                       [-r_world[1], r_world[0], 0]])
        J_world = jacp - rx @ jacr                      # 足端点的世界雅可比
        # 转到机体系
        Rb = data.xmat[bid_base].reshape(3, 3)       # base_link 的世界旋转矩阵
        J_body = Rb.T @ J_world
        cols = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in DRIVE[leg]]
        ana = J_body[:, cols]

        for j, nm in enumerate(DRIVE[leg]):
            diff = np.max(np.abs(num[:, j] - ana[:, j]))
            print('  %-8s 数值=[%7.4f %7.4f %7.4f]  解析=[%7.4f %7.4f %7.4f]  差=%.2e %s' % (
                nm, num[0, j], num[1, j], num[2, j],
                ana[0, j], ana[1, j], ana[2, j], diff,
                'OK' if diff < 1e-3 else 'CHECK'))


if __name__ == '__main__':
    main()
