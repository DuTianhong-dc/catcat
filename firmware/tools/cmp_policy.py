"""cmp_policy.py — 用训练侧的 int8 参考实现复算策略输出, 与固件实测对比

目的: 验证固件里的前向(cat_policy.c)与训练侧是不是一致 —— 也就是"量化 + 移植"到底引了多大误差。

用法:
    python tools/cmp_policy.py <npz路径> --obs "v1,v2,...,v59"
    python tools/cmp_policy.py <npz路径> --from-dbg     # 从 read_dbg 读到的探针值拼 obs(近似)

⚠️ 用 --from-dbg 时 obs 是【拼】出来的, 有两处近似:
     · jvel 段填 0（猫静止时确实≈0, 但策略在动就不准）
     · last_act 段用"读到的 pol_act"（差一拍 —— obs 要的是上一拍的动作）
   所以要精确对拍, 得给固件加完整 obs 探针。
   这里的作用是: 看量级对不对（如果差 0.1 以上, 那就是真有 bug, 不是近似造成的）。

量化口径与 sil_deploy.py / npz_to_c.py 完全一致: per-channel 对称 int8,
    s[o] = max_i|W[i][o]|/127,  wq = clip(round(W/s), -127, 127)
"""
import argparse
import os
import sys

import numpy as np

OBS_DIM = 59
H1, H2, H3, OUT = 512, 256, 128, 24


def q8(w):
    s = np.abs(w).max(axis=0) / 127.0
    s = np.where(s == 0, np.float32(1e-8), s).astype(np.float32)
    return np.clip(np.round(w / s), -127, 127).astype(np.int8), s


def run_ref(z, obs):
    """训练侧 int8 参考前向 —— 与 sil_deploy.py 的 Int8Policy.__call__ 同序"""
    h = (obs - z["obs_mean"]) / (z["obs_std"] + 1e-8)
    for wi, bi in (("W1", "b1"), ("W2", "b2"), ("W3", "b3")):
        wq, s = q8(z[wi])
        h = (h @ wq.astype(np.float32)) * s + z[bi]
        h = h * (1.0 / (1.0 + np.exp(-h)))            # swish
    wq, s = q8(z["Wh"])
    out = (h @ wq.astype(np.float32)) * s + z["bh"]
    return np.tanh(out[:12]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--obs", default="", help="59 个数, 逗号/空格分隔")
    ap.add_argument("--firmware", default="", help="固件实测的 12 个动作, 逗号分隔 (算误差用)")
    a = ap.parse_args()

    z = {k: v.astype(np.float32) for k, v in np.load(a.npz).items()}
    if not a.obs:
        print("需要 --obs")
        return 1
    obs = np.array([float(x) for x in a.obs.replace(",", " ").split()], dtype=np.float32)
    if obs.size != OBS_DIM:
        print("⚠️ obs 有 %d 个数, 应该是 %d" % (obs.size, OBS_DIM))
        return 1

    act = run_ref(z, obs)
    print("训练侧 int8 参考输出(12 维动作):")
    print("  " + " ".join("%+.4f" % v for v in act))
    print("  max|act| = %.4f" % np.abs(act).max())

    if a.firmware:
        fw = np.array([float(x) for x in a.firmware.replace(",", " ").split()], dtype=np.float32)
        if fw.size != 12:
            print("⚠️ 固件动作有 %d 个数, 应该是 12" % fw.size)
            return 1
        d = np.abs(fw - act)
        print("\n固件实测:")
        print("  " + " ".join("%+.4f" % v for v in fw))
        print("\n逐维绝对差:")
        print("  " + " ".join("%.4f" % v for v in d))
        print("  最大 %.4f   平均 %.4f" % (d.max(), d.mean()))
        print("  %s" % ("✅ 一致(误差在噪声量级)" if d.max() < 0.05
                        else "⚠️ 偏差偏大 —— 若 obs 是拼的, 先排除近似; 否则是真 bug"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
