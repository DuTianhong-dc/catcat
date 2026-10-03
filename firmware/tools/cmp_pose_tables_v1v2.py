#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cmp_pose_tables_v1v2.py — 两版固件「站姿/零位/限位」三组表逐条比对（铁律⑩，2026-09-30）

对照对象:
  v2 (CAT_APP2.0) : cat_pose.c  CAT_POSE_DEF / CAT_POSE_LIM_LO / CAT_POSE_LIM_HI
                    cat_obs.c   CAL_SERVO_ZERO / CAL_SERVO_DIR
  v1 (CAT_APP)    : policy_weights.c  POL_DEFAULT_POSE / CAL_JOINT_LO / CAL_JOINT_HI (生成表在文件尾)
                    cat_obs.c        CAL_SERVO_ZERO / CAL_SERVO_DIR
说明: 浮点按【float32 语义】比对 —— 两版字面量精度不同(0.3 vs 0.300000012), 固件里是同一 float32。
运行: 任意 python3（仅需 numpy）:  python cmp_pose_tables_v1v2.py
退出码: 0 = 全部一致; 1 = 有差异。
"""
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

V1 = r"C:\Users\dth15\Desktop\STM32\CAT_APP"
V2 = r"C:\Users\dth15\Desktop\STM32\CAT_APP2.0"


def parse_c_array(path, name, expect=12):
    src = open(path, "rb").read().decode("utf-8")
    m = re.search(re.escape(name) + r"\[[^\]]*\]\s*=\s*\{(.*?)\};", src, re.S)
    if not m:
        raise SystemExit("找不到数组: %s @ %s" % (name, path))
    body = re.sub(r"/\*.*?\*/", " ", m.group(1), flags=re.S)   # 剥块注释(防注释里的数字陷阱)
    body = re.sub(r"//[^\n]*", " ", body)                      # 剥行注释
    vals = re.findall(r"-?\d+(?:\.\d+)?", body)
    if len(vals) < expect:
        raise SystemExit("%s 解析出 %d 值(应 >=%d)" % (name, len(vals), expect))
    return vals[:expect], np.array([float(v) for v in vals[:expect]], dtype=np.float64)


groups = [
    # (组名, v2文件, v2名, v1文件, v1名, 整数?)
    ("站姿 DEF ", V2 + r"\cat_pose.c",        "CAT_POSE_DEF",    V1 + r"\policy_weights.c", "POL_DEFAULT_POSE", False),
    ("限位 LO  ", V2 + r"\cat_pose.c",        "CAT_POSE_LIM_LO", V1 + r"\policy_weights.c", "CAL_JOINT_LO",     False),
    ("限位 HI  ", V2 + r"\cat_pose.c",        "CAT_POSE_LIM_HI", V1 + r"\policy_weights.c", "CAL_JOINT_HI",     False),
    ("零位 ZERO", V2 + r"\cat_obs.c",         "CAL_SERVO_ZERO",  V1 + r"\cat_obs.c",        "CAL_SERVO_ZERO",   True),
    ("方向 DIR ", V2 + r"\cat_obs.c",         "CAL_SERVO_DIR",   V1 + r"\cat_obs.c",        "CAL_SERVO_DIR",    True),
]

bad_total = 0
for tag, f2, n2, f1, n1, is_int in groups:
    sv2, a2 = parse_c_array(f2, n2)
    sv1, a1 = parse_c_array(f1, n1)
    if is_int:
        same = (a1.astype(np.int64) == a2.astype(np.int64))
    else:
        same = (a1.astype(np.float32) == a2.astype(np.float32))
    n_bad = int((~same).sum())
    bad_total += n_bad
    print("== %s ==  %s（12 项，差异 %d）" % (tag, "✅ 一致" if n_bad == 0 else "❌ 有差异", n_bad))
    print("   #        v1(字面量)                 v2(字面量)")
    for i in range(12):
        mark = "" if same[i] else "   <== 差异!"
        print("  %2d   %-20s   %-20s%s" % (i + 1, sv1[i], sv2[i], mark))
    print()

print("总判定:", "✅ 三组表逐条全部一致" if bad_total == 0 else "❌ 共 %d 项不一致" % bad_total)
sys.exit(1 if bad_total else 0)
