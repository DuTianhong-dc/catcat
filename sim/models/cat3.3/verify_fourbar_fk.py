#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cat3.3 支撑腿运动学估计器 —— 验证脚本(PC 端, 不需要硬件)
=======================================================
目的: 为 STM32 的 linvel(机体线速度) 估计器做算法验证
      观测 obs[0:3]=linvel 训练时来自 MuJoCo 传感器(真值), 真机必须估计。

原理(业界标准): 支撑腿足端相对世界静止 -> 足端相对机体速度 = -机体速度
    1) 舵机角(3驱动/腿) -> 查四连杆表得被动角 -> FK 算足端位置
    2) 数值雅可比 J: 扰动关节角 -> 足端位置变化
    3) 足端速度 v = J * qdot
    4) 支撑腿判定(舵机 0x3C 负载 / 0x45 电流)
    5) 机体线速度 = -mean(v_支撑腿), 投到机体系(用 IMU 重力方向)

运行: cd models/cat3.3 && python verify_fourbar_fk.py
环境: mujoco_test

============================================================
=== 关节语义(模型设计者 2026-09-04 确认) ===
每 5 个关节一组, 共 4 组(4 条腿):
  组内 1/2/3 = 髋侧摆 / 髋俯仰 / 膝舵机      <- 3 个驱动(真机舵机)
  组内 4     = 曲柄与连杆连接关节(被动)
  组内 5     = 小腿与大腿连接关节(生理膝, 被动)
四连杆: 通过四阶拟合函数约束小腿与连杆; 小腿与摇臂固连
  Link4 = 连杆 ;  Link5 = 小腿 + 摇臂
观测 12 关节顺序: 右后 -> 左后 -> 前右 -> 前左, 每腿 髋侧摆/髋俯仰/膝

=== 已验证结论 ===
1. 环境入口: scene.xml = robot.xml(本体) + linkage_joint.xml(闭环四连杆模块)
   -> 训练用的就是这套, 几何以它为准
2. 四连杆表 fourbar_map_4legs_pure.json: 度, 每腿 341 行, ±170°, 步长 ~28°
   闭链误差: 节点 0.001~0.946mm, 节点间线性插值 0.052~0.952mm -> 插值够用, 不需加密
3. joint3(曲柄) 确实通过四连杆带动足端(用正确足端点后雅可比非零):
     joint3 10->45->90->135 度, 足端位移 13.5 / 28.6 / 58.7 mm
     d足端/djoint3 = [0.021, -0.004, 0.046]
4. ⚠️ MuJoCo 坑: 第 1 个关节是 FREE(6DOF), qpos[0:3]=位置 qpos[3:7]=四元数
   必须 data.qpos[:] = m.qpos0.copy(), 全零会导致变换异常(数值不变)

=== ⏳ 待定: 足端位置(两种候选, 差 59.8mm) ===
 Link5 局部坐标下(geom_pos + 顶点):
   A) 局部 z 最小(=小腿最远端): offset = (0.0180, -0.0054, -0.0848)
      -> 世界(-0.1673,-0.1910, 0.0452)
   B) 世界 z 最小(=最接近地面): 局部顶点 (-0.0081, 0.0020,-0.0958) 对应的另一点
      -> 世界(-0.1651,-0.1384, 0.0169)
 两者不同是因为 Link5 有旋转: 局部 -z 不指向世界 -z
 注: qpos0 是"零关节角"姿态, 猫脚悬空约 4.5cm, 不是站立姿态
 判定方法: 用 POL_DEFAULT_POSE(固件默认站姿)设关节角, 看哪个点贴地(z≈0)
"""

import json
import numpy as np
import mujoco

SCENE = 'scene.xml'          # 环境入口(robot.xml + linkage_joint.xml)
FOURBAR = 'fourbar_map_4legs_pure.json'

# 足端候选(Link5 局部): (名称, 偏移)
FOOT_CANDIDATES = {
    'A_局部z最小(小腿最远端)': np.array([0.0180, -0.0054, -0.0848]),
}


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def qrot(q, v):
    return v + 2 * np.cross(q[1:], np.cross(q[1:], v) + q[0] * v)


def rz(a):
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


# leg1(右后)几何链: 从 robot.xml 抄(单位 m, quat=[w,x,y,z])
# 注: 第一级 base_link 的 pos 是"世界->base"的变换, 真机只要机体系 -> 从 Link1 开始
CHAIN_LEG1 = [
    ((-0.16061, -0.062655, -0.0018124), [0.499998, -0.5, 0.500002, -0.5], 'joint1'),   # Link1 髋侧摆
    ((0.029007, -0.0024191, -0.02249), [0.499323, -0.499325, -0.500676, 0.500675], 'joint2'),  # Link2 髋俯仰
    ((-0.04467, 0.0, -0.015), [-3.67321e-06, 0.0, 0.0, 1.0], 'joint3'),               # Link3 膝(曲柄)
    ((0.075, 0.0, -0.015), [1.0, 0.0, 0.0, 0.0], 'joint5'),                           # Link5 小腿+摇臂
]
# TODO: leg2/3/4 的链按同样方式补(镜像腿注意 quat 符号与 y 取反)


def fk_leg1(q1, q2, q3, q5, foot_off):
    """正向运动学 -> 足端在机体系(base_link frame)的位置"""
    p = np.zeros(3)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    qs = {'joint1': q1, 'joint2': q2, 'joint3': q3, 'joint5': q5}
    for dp, dq, jn in CHAIN_LEG1:
        q = qmul(q, np.array(dq))
        p = p + qrot(q, np.array(dp))
        if jn:
            q = qmul(q, rz(qs[jn]))
    return p + qrot(q, foot_off)     # 最后加上足端相对 Link5 的偏移


def load_fourbar(leg='leg1'):
    d = json.load(open(FOURBAR, encoding='utf-8'))
    rows = d[leg]['rows']
    cr = np.array([r['q_crank'] for r in rows])
    a1 = np.array([r['q1'] for r in rows])
    a2 = np.array([r['q2'] for r in rows])
    o = np.argsort(cr)
    return np.deg2rad(cr[o]), np.deg2rad(a1[o]), np.deg2rad(a2[o])


def set_leg_consistent(m, data, k, q1, q2, q3):
    """设第 k 条腿(0起)的 3 个驱动角, 并用四连杆表补上 2 个被动角"""
    cr, A1, A2 = load_fourbar('leg%d' % (k + 1))
    adr = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    data.qpos[adr('joint%d' % (1 + k * 5))] = q1
    data.qpos[adr('joint%d' % (2 + k * 5))] = q2
    data.qpos[adr('joint%d' % (3 + k * 5))] = q3
    data.qpos[adr('joint%d' % (4 + k * 5))] = np.interp(q3, cr, A1)
    data.qpos[adr('joint%d' % (5 + k * 5))] = np.interp(q3, cr, A2)


def main():
    m = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(m)
    data.qpos[:] = m.qpos0.copy()          # ⚠️ 必须用 qpos0, 不能全零
    set_leg_consistent(m, data, 0, np.deg2rad(5), np.deg2rad(-10), np.deg2rad(45))
    mujoco.mj_forward(m, data)

    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'Link5')
    R = data.xmat[bid].reshape(3, 3)
    t = data.xpos[bid]
    print('=== 足端候选(MuJoCo 世界系) ===')
    for name, off in FOOT_CANDIDATES.items():
        p = t + R @ off
        print('  %-24s -> [%7.4f %7.4f %7.4f]' % (name, p[0], p[1], p[2]))

    print()
    print('=== joint3 扫角: 足端位移(验证四连杆传动) ===')
    prev = None
    for qd in [10, 45, 90, 135]:
        set_leg_consistent(m, data, 0, np.deg2rad(5), np.deg2rad(-10), np.deg2rad(qd))
        mujoco.mj_forward(m, data)
        R = data.xmat[bid].reshape(3, 3)
        p = data.xpos[bid] + R @ list(FOOT_CANDIDATES.values())[0]
        d = '-' if prev is None else '%.1f mm' % (np.linalg.norm(p - prev) * 1000)
        print('  joint3=%5.1f度  足端[%7.4f %7.4f %7.4f]  位移 %s' % (qd, p[0], p[1], p[2], d))
        prev = p


if __name__ == '__main__':
    main()
