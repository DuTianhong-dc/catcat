#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四条腿正向运动学(FK)验证
========================
对比"自写 FK"与"MuJoCo 计算"的足端位置(都在机体系/base_link frame 下)

几何来源: robot.xml(训练用的本体), 逐行抄取 body pos/quat
足端偏移: calibrate_foot.py 标定结果(POL_DEFAULT_POSE 站姿 + 世界Z最低点)
关节语义: 每 5 个一组; 组内 1/2/3 = 髋侧摆/髋俯仰/膝舵机(驱动), 4/5 被动

运行: cd models\\cat3.3 && python verify_fk_4legs.py
"""
import json
import numpy as np
import mujoco

SCENE = 'scene.xml'
FOURBAR = 'fourbar_map_4legs_pure.json'

# ============ 四条腿的几何链(从 robot.xml 抄取, 单位 m, quat=[w,x,y,z]) ============
# 每级: (相对父 body 的平移, 相对父的固定四元数, 该级可动关节名)
# 注意: 链从 Link1/Link6/Link11/Link16 开始(即相对 base_link 的第一级)
LEGS = {
    # ---- leg1 右后: Link1 -> Link2 -> Link5(小腿, 挂在 Link2 下=生理膝) ----
    # ⚠️ 结构: Link2 有两个子 body: Link3(曲柄, 带 joint3 驱动) 和 Link5(小腿, 带 joint5 被动)
    #    足端在 Link5 上, 所以 FK 链是 Link1->Link2->Link5, 不经过 Link3
    #    joint3(驱动膝舵机) 通过四连杆表决定 joint5 -> 带动小腿
    'leg1': {
        'chain': [
            ((-0.16061, -0.062655, -0.0018124), (0.499998, -0.5, 0.500002, -0.5), 'joint1'),
            ((0.029007, -0.0024191, -0.02249), (0.499323, -0.499325, -0.500676, 0.500675), 'joint2'),
            ((0.075, 0.0, -0.015), (1.0, 0.0, 0.0, 0.0), 'joint5'),
        ],
        'foot_off': (0.0469, 0.0079, -0.0190),      # Link5 局部
        'foot_body': 'Link5',
    },
    # ---- leg2 左后: Link6 -> Link7 -> Link10 ----
    'leg2': {
        'chain': [
            ((-0.16061, 0.062302, -0.0015924), (0.499998, 0.5, 0.500002, 0.5), 'joint6'),
            ((0.029193, 0.0026998, -0.02249), (0.500673, 0.500675, -0.499326, -0.499325), 'joint7'),
            ((0.075, 0.0, -0.0148), (1.0, 0.0, 0.0, 0.0), 'joint10'),
        ],
        'foot_off': (0.0469, -0.0079, -0.0190),     # Link10 局部
        'foot_body': 'Link10',
    },
    # ---- leg3 前右: Link11 -> Link12 -> Link15 ----
    'leg3': {
        'chain': [
            ((0.16939, -0.057655, 0.023188), (0.499998, -0.5, 0.500002, -0.5), 'joint11'),
            ((0.029003, -0.0024677, -0.02249), (0.499723, -0.499725, -0.500277, 0.500275), 'joint12'),
            ((0.07, 0.0, -0.015), (1.0, 0.0, 0.0, 0.0), 'joint15'),
        ],
        'foot_off': (0.0391, -0.0067, -0.0249),     # Link15 局部
        'foot_body': 'Link15',
    },
    # ---- leg4 前左: Link16 -> Link17 -> Link20 ----
    'leg4': {
        'chain': [
            ((0.16939, 0.057302, 0.023188), (0.499998, 0.5, 0.500002, 0.5), 'joint16'),
            ((0.029197, 0.0026509, -0.02249), (0.500273, 0.500275, -0.499727, -0.499725), 'joint17'),
            ((0.07, 0.0, -0.0148), (1.0, 0.0, 0.0, 0.0), 'joint20'),
        ],
        'foot_off': (0.0391, 0.0067, -0.0251),      # Link20 局部
        'foot_body': 'Link20',
    },
}


# ---------------- 四元数 / 旋转 ----------------
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2)


def qrot(q, v):
    w, x, y, z = q
    qv = np.array([x, y, z])
    return v + 2 * np.cross(qv, np.cross(qv, v) + w * v)


def qconj(q):
    return (q[0], -q[1], -q[2], -q[3])


def rz(a):
    return (np.cos(a / 2), 0.0, 0.0, np.sin(a / 2))


# ---------------- 自写 FK ----------------
def fk(chain, angles, foot_off):
    """chain: 几何链; angles: {关节名: 弧度}; foot_off: 足端相对末端 body 的偏移
       返回: 足端在机体系(base_link frame)的位置"""
    p = np.zeros(3)
    q = (1.0, 0.0, 0.0, 0.0)
    for dp, dq, jn in chain:
        # ⚠️ 顺序关键: body 的 pos 定义在"父 body frame"下,
        # 所以先用【父的朝向】旋转这个平移量, 之后再累乘本级的固定旋转
        p = p + qrot(q, np.array(dp))
        q = qmul(q, dq)
        # 关节旋转(绕自身原点, 只影响朝向与子 body)
        if jn is not None:
            q = qmul(q, rz(angles[jn]))
    return p + qrot(q, np.array(foot_off))


# ---------------- MuJoCo 侧 ----------------
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
    bid_base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')

    # 测试姿态: 每条腿给一组不同的驱动角
    test = {
        'leg1': (np.deg2rad(5), np.deg2rad(-10), np.deg2rad(45)),
        'leg2': (np.deg2rad(-5), np.deg2rad(-10), np.deg2rad(-45)),
        'leg3': (np.deg2rad(3), np.deg2rad(8), np.deg2rad(60)),
        'leg4': (np.deg2rad(-3), np.deg2rad(8), np.deg2rad(-60)),
    }

    data.qpos[:] = m.qpos0.copy()
    for k, (leg, (q1, q2, q3)) in enumerate(test.items()):
        cr, A1, A2 = load_fourbar(leg)
        n1, n2, n3 = 1 + k * 5, 2 + k * 5, 3 + k * 5
        data.qpos[adr('joint%d' % n1)] = q1
        data.qpos[adr('joint%d' % n2)] = q2
        data.qpos[adr('joint%d' % n3)] = q3
        data.qpos[adr('joint%d' % (n3 + 1))] = np.interp(q3, cr, A1)   # 被动角4
        data.qpos[adr('joint%d' % (n3 + 2))] = np.interp(q3, cr, A2)   # 被动角5
    mujoco.mj_forward(m, data)

    # base 的世界位姿(用于把世界坐标转回机体系)
    bp = data.xpos[bid_base]
    bq = data.xquat[bid_base]        # MuJoCo 的 xquat 是 [w,x,y,z]
    print('基座世界位置:', np.round(bp, 4), ' 朝向四元数:', np.round(bq, 4))
    print('\n腿    自写FK足端(机体系)              MuJoCo足端(机体系)             误差mm')
    print('-' * 84)

    for k, (leg, (q1, q2, q3)) in enumerate(test.items()):
        cfg = LEGS[leg]
        n1, n2, n3, n5 = 1 + k * 5, 2 + k * 5, 3 + k * 5, 5 + k * 5
        angles = {'joint%d' % n1: q1, 'joint%d' % n2: q2, 'joint%d' % n3: q3,
                  'joint%d' % n5: data.qpos[adr('joint%d' % n5)]}
        mine = fk(cfg['chain'], angles, cfg['foot_off'])

        # MuJoCo: 足端世界位置 -> 转到机体系
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cfg['foot_body'])
        R = data.xmat[bid].reshape(3, 3)
        foot_world = data.xpos[bid] + R @ np.array(cfg['foot_off'])
        rel = foot_world - bp
        ref = qrot(qconj(bq), rel)        # 世界 -> 机体系

        err = np.linalg.norm(mine - ref) * 1000
        flag = 'OK' if err < 1.0 else ('小' if err < 10 else 'BAD')
        print('%s  [%7.4f %7.4f %7.4f]   [%7.4f %7.4f %7.4f]   %7.2f  %s' % (
            leg, mine[0], mine[1], mine[2], ref[0], ref[1], ref[2], err, flag))


if __name__ == '__main__':
    main()
