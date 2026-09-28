"""check_policy_err.py — 端到端误差: 把固件【当时的观测】喂给训练侧网络, 与固件自己的输出对比

用途: 验证固件的 int8 前向 + obs 组装 与训练侧是否一致(能同时暴露 obs 组装错、
      量化实现差异、系数抄错等问题)。

流程: ST-Link 读观测区 → 重建 59 维 obs → 本地跑 npz 网络 → 与固件的 pol_act 比。

⚠️ 重建的近似之处(看结果时要心里有数):
   · jvel(obs[23:35))    观测区没有 → 用 0。猫静止时真的接近 0, 影响小。
   · last_act(obs[47:59)) 观测区没有 → 用【当前 pol_act】。
     这要求策略已在【稳定态】(指令持续不变、输出不再变化), 否则会偏。
   → 所以本工具要在"持续发同一指令 + 猫静止不动"的情况下用。

用法:
    python tools/check_policy_err.py --vx 0.3 --wz 0.2
"""
import argparse
import re
import struct
import subprocess
import sys

import numpy as np

CLI = (r"C:\Program Files\STMicroelectronics\STM32Cube"
       r"\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe")

BASE = 0x10000000
OFF = {                      # dbg_t 里各探针的偏移(改了 cat_sys.h 要同步)
    #        偏移   个数  类型
    "gyro":   (100, 3, "f"),
    "grav":   (112, 3, "f"),
    "linvel": (136, 3, "f"),
    "jpos":   (148, 12, "f"),
    "tau":    (196, 12, "f"),
    "pol_act": (340, 12, "f"),
    "loops":  (8, 1, "u"),      # ⚠️ u32, 不能按 float 解
    "ms":     (12, 1, "u"),     # ⚠️ 同上
}


def read_words(addr, nbytes):
    out = subprocess.run([CLI, "-c", "port=SWD", "mode=hotplug",
                          "-r32", hex(addr), str(nbytes)],
                         capture_output=True, text=True, timeout=120)
    blob = out.stdout + out.stderr
    d = {}
    for line in blob.splitlines():
        m = re.match(r"^0x([0-9A-Fa-f]{8})\s*:\s*(.*)$", line.strip())
        if not m:
            continue
        a = int(m.group(1), 16)
        for i, wd in enumerate(re.findall(r"\b[0-9A-Fa-f]{8}\b", m.group(2))):
            d[a + 4 * i] = int(wd, 16)
    if not d:
        print("读内存失败:")
        print(blob[-1200:])
        sys.exit(1)
    return d


def f32(w):
    return struct.unpack("<f", struct.pack("<I", w & 0xFFFFFFFF))[0]


def grab(d, name):
    off, n, ty = OFF[name]
    w = [d.get(BASE + off + 4 * i, 0) for i in range(n)]
    return [f32(x) for x in w] if ty == "f" else w


def q8(w):
    s = np.abs(w).max(axis=0) / 127.0
    s = np.where(s == 0, np.float32(1e-8), s).astype(np.float32)
    return np.clip(np.round(w / s), -127, 127).astype(np.int8), s


def run_net(z, obs):
    """与固件 policy_run 同序: 归一化 → 3层(int8×scale + swish) → 头层 → tanh 前12"""
    h = (obs - z["obs_mean"]) / (z["obs_std"] + 1e-8)
    for wi, bi in (("W1", "b1"), ("W2", "b2"), ("W3", "b3")):
        wq, s = q8(z[wi])
        h = (h @ wq.astype(np.float32)) * s + z[bi]
        h = h * (1.0 / (1.0 + np.exp(-h)))          # swish
    wq, s = q8(z["Wh"])
    out = (h @ wq.astype(np.float32)) * s + z["bh"]
    return np.tanh(out[:12]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=r"C:\Users\dth15\Desktop\mujoco\play\cat33_policy_v31.npz")
    ap.add_argument("--vx", type=float, required=True, help="当前下行命令 vx (m/s)")
    ap.add_argument("--wz", type=float, required=True, help="当前下行命令 wz (rad/s)")
    ap.add_argument("--gyro", type=float, nargs=2, default=None,
                    metavar=("VX", "WZ"), help="(备用) 直接指定, 不用")
    a = ap.parse_args()

    d = read_words(BASE, 392)
    loops, ms = grab(d, "loops")[0], grab(d, "ms")[0]
    linvel, gyro, grav = grab(d, "linvel"), grab(d, "gyro"), grab(d, "grav")
    jpos, tau, pact = grab(d, "jpos"), grab(d, "tau"), grab(d, "pol_act")

    print("固件状态: loops=%d  ms=%d  主循环周期=%.1fms"
          % (loops, ms, ms / loops if loops else 0))
    print("  linvel = %s" % np.round(linvel, 4))
    print("  gyro   = %s" % np.round(gyro, 4))
    print("  grav   = %s" % np.round(grav, 4))
    print("  固件 pol_act = %s" % np.round(pact, 4))

    # ---- 重建 59 维 obs ----
    obs = np.zeros(59, dtype=np.float32)
    obs[0:3] = linvel               # [0:3)  linvel
    obs[3:6] = gyro                 # [3:6)  gyro
    obs[6:9] = grav                 # [6:9)  gravity
    obs[9:11] = [a.vx, a.wz]        # [9:11) cmd  ← 我知道我发的是什么
    obs[11:23] = jpos               # [11:23) jpos
    obs[23:35] = 0.0                # [23:35) jvel —— 观测区没有, 静止≈0
    obs[35:47] = tau                # [35:47) tau
    obs[47:59] = pact               # [47:59) last_act —— 稳定态下 = 当前输出

    print("\n=== 重建的 obs(59) ===")
    print("  linvel   %s" % np.round(obs[0:3], 4))
    print("  gyro     %s" % np.round(obs[3:6], 4))
    print("  gravity  %s" % np.round(obs[6:9], 4))
    print("  cmd      %s   ← 本次发的" % np.round(obs[9:11], 4))
    print("  jpos     %s" % np.round(obs[11:23], 4))
    print("  jvel     %s   ← ⚠️ 用 0 代替(观测区没有)" % np.round(obs[23:35], 4))
    print("  tau      %s" % np.round(obs[35:47], 4))
    print("  last_act %s   ← ⚠️ 用当前 pol_act 代替" % np.round(obs[47:59], 4))

    z = dict(np.load(a.npz))
    ref = run_net(z, obs)

    print("\n=== 输出对比 ===")
    print("  训练侧: %s" % np.round(ref, 4))
    print("  固件  : %s" % np.round(pact, 4))
    diff = ref - np.array(pact, dtype=np.float32)
    print("  差值  : %s" % np.round(diff, 4))
    print()
    print("  最大绝对差 %.5f   平均绝对差 %.5f"
          % (np.abs(diff).max(), np.abs(diff).mean()))
    amax = np.abs(np.array(pact)).max()
    if amax > 0:
        print("  相对满量程: %.3f%%" % (100.0 * np.abs(diff).max() / max(amax, 1e-6)))
    print()
    if np.abs(diff).max() < 0.02:
        print("  ✅ 误差很小 —— 固件与训练侧一致(差异应只来自量化 + jvel/last_act 的近似)")
    elif np.abs(diff).max() < 0.05:
        print("  ⚠️ 有可见误差 —— 先看是不是 jvel=0 / last_act 近似带来的(换个命令重测)")
    else:
        print("  ❌ 误差大 —— 大概率真有问题: 查 obs 分段偏移 / 权重版本 / 归一化")
    return 0


if __name__ == "__main__":
    sys.exit(main())
