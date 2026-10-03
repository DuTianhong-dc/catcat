#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""check_act_lim.py — 一次核对三张表的同源性（铁律⑩: 抄表必写脚本逐条比对）

  ① cat_obs.c  ACT_LIM_LO/HI          vs 模型 jnt_range          （动作口径表）
  ② cat_obs.c  CAL_SERVO_ZERO + 换算   → counts 落点
  ③ cat_safety.c SAFETY_GOAL_MIN/MAX   vs ② 的换算结果           （机械行程防线的 counts 版）
     （③ 允许 ≤1 count 取整差: safety 是"向内取整", obs 输出是截断）

运行环境: **WSL**（要 mujoco 包; 用 ~/mjx_gpu/bin/python）。
    wsl -d Ubuntu -- bash -lc "cp /mnt/c/Users/dth15/Desktop/STM32/CAT_APP2.0/tools/check_act_lim.py ~/ && \
        cd ~/mujoco_playground && ~/mjx_gpu/bin/python ~/check_act_lim.py"
退出码: 0 = 全一致; 1 = 有 MISMATCH。
"""
import math
import re
import sys
import numpy as np
import mujoco

APP = "/mnt/c/Users/dth15/Desktop/STM32/CAT_APP2.0"
XML = "/home/dth15/mujoco/models/cat3.4/scene_full.xml"
NAMES = ["joint1", "joint2", "joint3", "joint6", "joint7", "joint8",
         "joint11", "joint12", "joint13", "joint16", "joint17", "joint18"]
R2R = 6.2832 / 4096.0   # OBS_RAD_PER_COUNT（与固件同值）


def parse_c_array(src, name, expect=12):
    m = re.search(name + r"\[[^\]]*\]\s*=\s*\{(.*?)\};", src, re.S)
    if not m:
        raise SystemExit("找不到数组: " + name)
    body = re.sub(r"/\*.*?\*/", " ", m.group(1), flags=re.S)   # 先剥注释(里面有数字陷阱)
    body = re.sub(r"//[^\n]*", " ", body)
    vals = re.findall(r"-?\d+(?:\.\d+)?", body)
    if len(vals) < expect:
        raise SystemExit("%s 解析出 %d 个值(应>=%d)" % (name, len(vals), expect))
    return np.array([float(v) for v in vals[:expect]], dtype=np.float64)


src_obs = open(APP + "/cat_obs.c", "rb").read().decode("utf-8")
src_saf = open(APP + "/cat_safety.c", "rb").read().decode("utf-8")

c_lo = parse_c_array(src_obs, "ACT_LIM_LO")
c_hi = parse_c_array(src_obs, "ACT_LIM_HI")
c_zero = parse_c_array(src_obs, "CAL_SERVO_ZERO")
s_min = parse_c_array(src_saf, "SAFETY_GOAL_MIN")
s_max = parse_c_array(src_saf, "SAFETY_GOAL_MAX")

m = mujoco.MjModel.from_xml_path(XML)
jids = [int(m.joint(n).id) for n in NAMES]
jr = m.jnt_range[jids]
m_lo, m_hi = jr[:, 0], jr[:, 1]

# ① ACT_LIM vs 模型 jnt_range
ok1 = all(abs(c_lo[i] - m_lo[i]) < 1e-9 and abs(c_hi[i] - m_hi[i]) < 1e-9 for i in range(12))

# ②③ 换算 counts 落点 vs SAFETY 表
counts_lo = c_zero + c_lo / R2R      # q=lo 时的 counts（未取整）
counts_hi = c_zero + c_hi / R2R
ok23 = True
for i in range(12):
    # safety 表是"向内取整": min=ceil(lo_counts), max=floor(hi_counts)
    e_min = math.ceil(counts_lo[i])
    e_max = math.floor(counts_hi[i])
    d_min = abs(s_min[i] - e_min)
    d_max = abs(s_max[i] - e_max)
    f1 = "" if d_min <= 1 else " <<<MIN"
    f2 = "" if d_max <= 1 else " <<<MAX"
    ok23 &= (d_min <= 1 and d_max <= 1)
    print("%2d | %-7s | lo表 %7.4f→cnt %8.2f | safety_min %5d(期望%5d, d=%d)%s | hi表 %7.4f→cnt %8.2f | safety_max %5d(期望%5d, d=%d)%s" % (
        i, NAMES[i], c_lo[i], counts_lo[i], int(s_min[i]), e_min, d_min, f1,
        c_hi[i], counts_hi[i], int(s_max[i]), e_max, d_max, f2))

ok = ok1 and ok23
print()
print("① ACT_LIM vs jnt_range:", "ALL MATCH" if ok1 else "HAS MISMATCH")
print("②③ ACT_LIM 换算 vs SAFETY 表(容差1count):", "ALL MATCH" if ok23 else "HAS MISMATCH")
print("RESULT:", "ALL MATCH" if ok else "HAS MISMATCH")
sys.exit(0 if ok else 1)
