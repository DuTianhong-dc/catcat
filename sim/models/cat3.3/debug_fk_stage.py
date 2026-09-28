#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐级定位 FK 偏差: 对比 leg1 每一级 body 在机体系的位置(自写 vs MuJoCo)"""
import json
import numpy as np
import mujoco
from verify_fk_4legs import LEGS, qmul, qrot, qconj, rz, load_fourbar

m = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(m)
adr = lambda n: m.jnt_qposabr if False else m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
bid_base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')

cfg = LEGS['leg1']
q1, q2, q3 = np.deg2rad(5), np.deg2rad(-10), np.deg2rad(45)
cr, A1, A2 = load_fourbar('leg1')
data.qpos[:] = m.qpos0.copy()
data.qpos[adr('joint1')] = q1
data.qpos[adr('joint2')] = q2
data.qpos[adr('joint3')] = q3
data.qpos[adr('joint4')] = np.interp(q3, cr, A1)
data.qpos[adr('joint5')] = np.interp(q3, cr, A2)
mujoco.mj_forward(m, data)

bp = data.xpos[bid_base]
bq = tuple(data.xquat[bid_base])


def to_body(world_p):
    """世界坐标 -> 机体系"""
    return qrot(qconj(bq), world_p - bp)


angles = {'joint1': q1, 'joint2': q2, 'joint3': q3, 'joint5': data.qpos[adr('joint5')]}

print('级    body    自写(机体系)                  MuJoCo(机体系)               误差mm')
print('-' * 78)
p = np.zeros(3)
q = (1.0, 0.0, 0.0, 0.0)
bodies = ['Link1', 'Link2', 'Link3', 'Link5']
for (dp, dq, jn), bname in zip(cfg['chain'], bodies):
    p = p + qrot(q, np.array(dp))          # 先用父朝向平移
    q = qmul(q, dq)
    if jn is not None:
        q = qmul(q, rz(angles[jn]))
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
    ref = to_body(data.xpos[bid])
    print('%-5s %-7s [%7.4f %7.4f %7.4f]   [%7.4f %7.4f %7.4f]   %7.2f' % (
        bodies.index(bname) + 1, bname, p[0], p[1], p[2], ref[0], ref[1], ref[2],
        np.linalg.norm(p - ref) * 1000))

# 额外: 打印 MuJoCo 认为的 base_link 与 Link1 的相对关系(验证"从 base 开始"的假设)
print()
print('MuJoCo: base_link body_pos  =', np.round(m.body_pos[bid_base], 4))
print('MuJoCo: base_link body_quat =', np.round(m.body_quat[bid_base], 4))
print('MuJoCo: Link1    body_pos   =', np.round(m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'Link1')], 4))
print('MuJoCo: Link1 的父 body id  =', m.body_parentid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'Link1')], '(base_link id=%d)' % bid_base)
print()
print('机体系下 Link1 的位置(MuJoCo) =', np.round(to_body(data.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'Link1')]), 4))
print('我用的 Link1 平移量(原始 pos)  =', np.round(cfg['chain'][0][0], 4))
print('-> 若两者不同, 说明 base_link 还有额外变换没算进去')
