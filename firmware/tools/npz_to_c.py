"""npz_to_c.py — 把训练侧导出的策略 .npz 生成 Keil 可编译的 policy_weights.c

用法(需要 numpy):
  python tools/npz_to_c.py <npz路径> [输出C路径] [--tag v31] [--reward "..."]

量化: 按输出通道对称 int8 —— s[o] = max_i|W[i][o]|/127, wq = clip(round(W/s), -127, 127)。
      ⚠️ 与训练侧 sil_deploy.py 的 quantize_int8() 是同一套, 两端必须同步改。

⚠️ 本脚本【只】生成权重与归一化参数, 不含 POL_DEFAULT_POSE / 关节限位:
     第二版里站姿常量在 cat_obs.c 的 OBS_DEFAULT_POSE, 关节限位暂时没人用。
     所以本脚本不需要 mujoco 库, 也不读模型文件。

⚠️ 生成前会校验 npz 各数组形状与固件维度是否匹配; 不匹配直接退出, 不产出半成品。
     维度来源: cat_config.h 的 OBS_DIM / cat_policy.h 的 POL_DIM_*。
"""
import argparse
import datetime
import hashlib
import os
import sys

import numpy as np

# ---- 期望维度(改这里之前先改固件头文件, 两边必须一致) ----
OBS_DIM = 59
DIM_H1  = 512
DIM_H2  = 256
DIM_H3  = 128
DIM_OUT = 24        # 头层宽度, ⚠️ 不是 ACT_DIM(12)

EXPECT = {
    "W1": (OBS_DIM, DIM_H1), "b1": (DIM_H1,),
    "W2": (DIM_H1, DIM_H2),  "b2": (DIM_H2,),
    "W3": (DIM_H2, DIM_H3),  "b3": (DIM_H3,),
    "Wh": (DIM_H3, DIM_OUT), "bh": (DIM_OUT,),
    "obs_mean": (OBS_DIM,),  "obs_std": (OBS_DIM,),
}


def q8(w):
    """按输出通道对称 int8 量化。w: (in, out) -> 每列一个 scale。"""
    s = np.abs(w).max(axis=0) / 127.0
    s = np.where(s == 0, np.float32(1e-8), s).astype(np.float32)
    wq = np.clip(np.round(w / s), -127, 127).astype(np.int8)
    return wq, s


def emit(f, name, arr, ctype):
    f.write("const %s %s[%d] = {\n" % (ctype, name, arr.size))
    flat = arr.ravel()
    for i in range(0, flat.size, 20):
        if ctype == "float":
            row = ",".join("%.9g" % v for v in flat[i:i + 20])   # 浮点不能按整数写
        else:
            row = ",".join(str(int(v)) for v in flat[i:i + 20])
        f.write("  " + row + ",\n")
    f.write("};\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--reward", default="")
    a = ap.parse_args()

    if not os.path.exists(a.npz):
        print("[错误] 找不到 npz:", a.npz)
        return 1
    out = a.out or os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "policy_weights.c"))

    z = {k: v.astype(np.float32) for k, v in np.load(a.npz).items()}

    # ---- 形状校验: 不对就退出, 绝不产出半成品 ----
    bad = []
    for k, shape in EXPECT.items():
        if k not in z:
            bad.append("%s 缺失" % k)
        elif tuple(z[k].shape) != shape:
            bad.append("%s 形状 %s != 期望 %s" % (k, tuple(z[k].shape), shape))
    if bad:
        print("[错误] npz 与固件维度不匹配, 未生成任何文件:")
        for m in bad:
            print("   -", m)
        print("   (固件侧: OBS_DIM=%d, POL_DIM_H1/H2/H3/OUT=%d/%d/%d/%d)"
              % (OBS_DIM, DIM_H1, DIM_H2, DIM_H3, DIM_OUT))
        return 1

    md5full = hashlib.md5(open(a.npz, "rb").read()).hexdigest()
    build_id = md5full[:12]
    tag = a.tag or ("unknown/" + os.path.basename(a.npz))

    with open(out, "w", encoding="utf-8") as f:
        f.write("/* 自动生成: tools/npz_to_c.py <- %s\n" % os.path.basename(a.npz))
        f.write("   导出时间 : %s\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        f.write("   npz md5  : %s\n" % md5full)
        f.write("   最终奖励 : %s\n" % (a.reward or "(未提供)"))
        f.write("   手改无效! 换策略请重新跑本脚本 */\n")
        f.write('#include "cat_policy.h"\n\n')
        f.write('const char POL_BUILD_ID[] = "%s";\n' % build_id)
        f.write('const char POL_TAG[] = "%s";\n\n' % tag)

        for n, wi, bi in (("1", "W1", "b1"), ("2", "W2", "b2"), ("3", "W3", "b3")):
            wq, s = q8(z[wi])
            emit(f, "POL_W%s" % n, wq, "int8_t")
            emit(f, "POL_S%s" % n, s, "float")
            emit(f, "POL_B%s" % n, z[bi], "float")

        wq, s = q8(z["Wh"])
        emit(f, "POL_WH", wq, "int8_t")
        emit(f, "POL_SH", s, "float")
        emit(f, "POL_BH", z["bh"], "float")

        emit(f, "POL_OBS_MEAN", z["obs_mean"], "float")
        emit(f, "POL_OBS_STD",  z["obs_std"],  "float")

    print("[ok] 已生成:", out, "(%d KB)" % (os.path.getsize(out) // 1024))
    print("     POL_BUILD_ID =", build_id)
    print("     POL_TAG      =", tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
