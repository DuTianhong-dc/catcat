"""gen_safety_limits.py — 生成 cat_safety.c 的"目标钳位"计数范围表。

数据源(只读, 不跑仿真):
  ① robot.xml 的 <joint range="lo hi">      —— cat3.4 机械行程 (rad)
  ② ../cat_obs.c 的 CAL_SERVO_ZERO          —— 关节角 0 对应的舵机计数(单一数据源)
换算: counts = ZERO + q / R2R   (与 cat_obs.c 的 obs_act_to_counts 同公式; DIR 全 +1)
取整方向: 向内取(下限向上取整/上限向下取整) —— 保证钳位结果不越过模型行程。

用法:
  python tools/gen_safety_limits.py
输出: 逐关节核对表 + 可直接粘贴进 cat_safety.c 的 C 数组。

⚠️ 重标定(CAL_SERVO_ZERO)或改模型行程后, 必须重跑本脚本并更新 cat_safety.c 的表!
⚠️ 用 XML 解析器读 range 属性(不能用正则: 行内还有 actuatorfrcrange, 正则会吃错)。
"""
import math
import os
import re
import xml.etree.ElementTree as ET

ROBOT_XML = r"C:\Users\dth15\Desktop\mujoco\models\cat3.4\build\robot.xml"
OBS_C = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "cat_obs.c"))
R2R = 6.2832 / 4096.0
# 观测序号 ↔ 模型关节名(后右 | 后左 | 前右 | 前左, 每腿 abd/hip/knee)
JOINTS = [1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18]


def parse_zero():
    s = open(OBS_C, encoding="utf-8").read()
    m = re.search(r"CAL_SERVO_ZERO\s*\[[^\]]*\]\s*=\s*\{([^}]*)\}", s, re.S)
    if not m:
        raise SystemExit("cat_obs.c 里找不到 CAL_SERVO_ZERO")
    body = re.sub(r"/\*.*?\*/", " ", m.group(1), flags=re.S)  # 去块注释
    return [int(t.strip().rstrip("uU")) for t in body.replace("\n", " ").split(",")
            if t.strip()]


def parse_ranges():
    root = ET.parse(ROBOT_XML).getroot()
    out = {}
    for j in root.iter("joint"):
        name = j.get("name", "")
        rng = j.get("range")            # 只要真·关节 range, 不含 actuatorfrcrange
        m = re.fullmatch(r"joint(\d+)", name)
        if m and rng:
            out[int(m.group(1))] = [float(x) for x in rng.split()]
    return out


def main():
    zero = parse_zero()
    rng = parse_ranges()
    assert len(zero) == 12, "CAL_SERVO_ZERO 不是 12 个"

    lo, hi = [], []
    print("关节      行程(rad)       计数下限 计数上限   (ZERO+Δ/%.6f)" % R2R)
    for i, j in enumerate(JOINTS):
        q0, q1 = rng[j]
        c0 = math.ceil(zero[i] + q0 / R2R)   # 下限向上取整(向内)
        c1 = math.floor(zero[i] + q1 / R2R)  # 上限向下取整(向内)
        c0 = max(0, min(4095, c0))
        c1 = max(0, min(4095, c1))
        assert c0 < c1, "joint%d 钳位区间反了" % j
        lo.append(c0)
        hi.append(c1)
        print("joint%-2d  [%+.2f, %+.2f]   %5d    %5d" % (j, q0, q1, c0, c1))

    def fmt(v):
        # 3+3+3+3 按腿分组便于核对
        rows = ["%5d,%5d,%5d" % tuple(v[k:k + 3]) for k in range(0, 12, 3)]
        return ",\n    ".join(rows)

    print("\n/* 可直接粘贴 (顺序 = 观测序: 后右 | 后左 | 前右 | 前左) */")
    print("static const uint16_t SAFETY_GOAL_MIN[SERVO_COUNT] = {\n    %s\n};" % fmt(lo))
    print("static const uint16_t SAFETY_GOAL_MAX[SERVO_COUNT] = {\n    %s\n};" % fmt(hi))


if __name__ == "__main__":
    main()
