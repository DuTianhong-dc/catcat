"""verify_weights.py — 校验 npz_to_c.py 生成的 policy_weights.c

做两件事:
  A. 生成无损: C 数组 == q8(npz) 逐元素一致(浮点的 %.9g 往返舍入在容差内)
  B. 量化可信: 用量化权重跑网络 ≈ 用原始 float 权重跑网络 —— 输出动作差异应该很小。

  B 才是关键: A 只证明"抄写没错", B 才能证明"这份 int8 权重还是原来的策略"。
  两者都过才敢烧进固件。

⚠️ C 里的权重是【一维展平】的(行优先), 与固件 dense_s8 的 w + i*out_dim 寻址一致。
   比对/推理前要 reshape 回矩阵形状, 否则形状对不上。

用法:
  python tools/verify_weights.py <npz路径> [policy_weights.c路径]
"""
import os
import re
import sys

import numpy as np

OBS_DIM, DIM_H1, DIM_H2, DIM_H3, DIM_OUT = 59, 512, 256, 128, 24


def parse_c(path):
    """把 policy_weights.c 里的数组抠出来: name -> (ctype, N, 一维 np.array)"""
    s = open(path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"const\s+(int8_t|float)\s+(\w+)\[(\d+)\]\s*=\s*\{(.*?)\};",
                         s, re.S):
        ctype, name, n, body = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        toks = [t for t in body.replace("\n", " ").split(",") if t.strip()]
        if len(toks) != n:
            raise RuntimeError("%s: 声明 %d 个, 实际写了 %d 个" % (name, n, len(toks)))
        dt = np.int8 if ctype == "int8_t" else np.float32
        out[name] = (ctype, n, np.array([float(t) for t in toks], dtype=np.float64).astype(dt))
    return out


def q8(w):
    s = np.abs(w).max(axis=0) / 127.0
    s = np.where(s == 0, np.float32(1e-8), s).astype(np.float32)
    return np.clip(np.round(w / s), -127, 127).astype(np.int8), s


def run_net(layers, head, mean, std, x):
    """layers: [(W, scale_or_None, b), ...]; head 同。
    与固件 policy_run 同序: 归一化 → 各层(矩阵乘 → 反量化+bias → swish) → 头层 → tanh 前 12"""
    h = (x - mean) / (std + 1e-8)
    for w, s, b in layers:
        acc = h @ w.astype(np.float32)
        h = (acc * s + b) if s is not None else (acc + b)
        h = h * (1.0 / (1.0 + np.exp(-h)))          # swish
    w, s, b = head
    out = h @ w.astype(np.float32)
    out = (out * s + b) if s is not None else (out + b)
    return np.tanh(out[:12]).astype(np.float32)


def main():
    npz = sys.argv[1]
    cpath = sys.argv[2] if len(sys.argv) > 2 else os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "policy_weights.c"))

    z = {k: v.astype(np.float32) for k, v in np.load(npz).items()}
    C = parse_c(cpath)
    print("npz : %s" % npz)
    print("C   : %s  (%d 个数组)" % (cpath, len(C)))
    print()

    # 期望形状表(同时拿到 q8 的结果, 后面 A/B 都要用)
    Q = {}
    for n, wi, bi in (("1", "W1", "b1"), ("2", "W2", "b2"), ("3", "W3", "b3")):
        wq, s = q8(z[wi])
        Q["POL_W" + n] = wq
        Q["POL_S" + n] = s
        Q["POL_B" + n] = z[bi]
    wq, s = q8(z["Wh"])
    Q["POL_WH"] = wq
    Q["POL_SH"] = s
    Q["POL_BH"] = z["bh"]
    Q["POL_OBS_MEAN"] = z["obs_mean"]
    Q["POL_OBS_STD"] = z["obs_std"]

    def Cv(name):
        """取 C 里的数组并 reshape 回矩阵形状"""
        return C[name][2].reshape(Q[name].shape)

    # ---------- A. 生成无损 ----------
    print("== A. 生成无损(逐元素比对) ==")
    okA = True
    for name, ref in Q.items():
        if name not in C:
            print("   ✗ %s 不存在" % name)
            okA = False
            continue
        got = Cv(name)
        if ref.dtype == np.int8:
            same = np.array_equal(got, ref)
            print("   %-12s %-7d int8    %s" % (name, got.size, "一致" if same else "✗ 不一致"))
            okA &= same
        else:
            d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
            rel = (d / (np.abs(ref.astype(np.float64)) + 1e-12)).max()
            good = rel < 1e-6        # %.9g 往返只该有位级误差
            print("   %-12s %-7d float   最大相对差 %.1e  %s"
                  % (name, got.size, rel, "一致" if good else "✗ 超差"))
            okA &= good
    print("   -> A %s" % ("通过" if okA else "失败"))
    print()

    # ---------- B. 量化可信 ----------
    print("== B. 量化可信(量化网络 vs 原始 float 网络) ==")
    rng = np.random.default_rng(20260927)
    N = 200
    # 按训练分布采样观测(不是纯随机), 才有代表性
    X = (z["obs_mean"] + rng.normal(0, 1, size=(N, OBS_DIM)) * z["obs_std"]).astype(np.float32)

    ref_act = run_net(
        [(z["W1"], None, z["b1"]), (z["W2"], None, z["b2"]), (z["W3"], None, z["b3"])],
        (z["Wh"], None, z["bh"]), z["obs_mean"], z["obs_std"], X)

    q_act = run_net(
        [(Cv("POL_W1"), Cv("POL_S1"), Cv("POL_B1")),
         (Cv("POL_W2"), Cv("POL_S2"), Cv("POL_B2")),
         (Cv("POL_W3"), Cv("POL_S3"), Cv("POL_B3"))],
        (Cv("POL_WH"), Cv("POL_SH"), Cv("POL_BH")),
        Cv("POL_OBS_MEAN"), Cv("POL_OBS_STD"), X)

    d = np.abs(q_act - ref_act)
    print("   样本 %d × 动作 %d" % (N, ref_act.shape[1]))
    print("   参考动作范围   : [%.3f, %.3f]" % (ref_act.min(), ref_act.max()))
    print("   最大绝对差     : %.6f" % d.max())
    print("   平均绝对差     : %.6f" % d.mean())
    print("   差 >0.02 的比例: %.3f%%" % (100.0 * (d > 0.02).mean()))
    okB = d.max() < 0.05          # 单点动作差 < 5% 满量程
    print("   -> B %s (阈值 0.05)" % ("通过" if okB else "失败"))
    print()

    good = okA and okB
    print("总结: A=%s  B=%s  ->  %s"
          % ("通过" if okA else "失败", "通过" if okB else "失败",
             "可以烧进固件" if good else "先别烧, 查上面"))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
