# -*- coding: utf-8 -*-
r"""standing-up 验收脚本草案（⚠️ 待执行，等训练跑完 + 检查点落盘）。

用法（WSL ~/mjx_gpu，把 EXP 改成实际实验目录）：
  python tasks/lee2019_standup/scripts/_eval_standup.py <实验目录>

判据（实验记录.md §5）：
  · 末 50 步（1 s）up_z 均值 > 0.9            —— 正立
  · 末 50 步基座高度 > 0.13 m                 —— 站起来了（站姿实测 0.1473）
  · 末态 ‖q − q_stand‖ 小（q_stand = 0，相对零位）
  · 站起时间：首次同时满足上两条的时刻
  · 与零动作基线对照（delta_current 下 a=0 = 保持当前角）
"""
import glob
import json
import os
import sys

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import checkpoint as brax_ckpt
from brax.training.acme import running_statistics as RS
from mujoco_playground import registry

ROOT = "/mnt/c/Users/dth15/Desktop/mujoco"
CFG = os.path.join(ROOT, "tasks/lee2019_standup/configs/standup.json")
EXP = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/dth15/mujoco_playground/logs/Cat40Lee2019-YYYYMMDD-HHMMSS-_lee2019_standup_exp01")
NEP, NSTEP = 16, 250
H_OK, UP_OK = 0.13, 0.9


def main():
    if not os.path.isdir(EXP):
        print("⚠️ 实验目录不存在，本脚本只是草案：", EXP)
        print("   训练跑完后把实际目录作为第一个参数传入。")
        return 0
    ov = json.load(open(CFG, encoding="utf-8"))["overrides"]
    env = registry.load("Cat40Lee2019", config_overrides=ov)
    ck = sorted(glob.glob(os.path.join(EXP, "checkpoints", "0*")))[-1]
    params = tuple(brax_ckpt.load(ck))
    pol = params[1]["params"]
    names = sorted([k for k in pol if k.startswith("hidden_")],
                   key=lambda s: int(s.split("_")[1]))
    W = [jp.asarray(pol[n]["kernel"]) for n in names]
    B = [jp.asarray(pol[n]["bias"]) for n in names]

    def act(obs, mode):
        if mode == "zero":
            return jp.zeros(12)
        h = RS.normalize(obs, params[0])["state"]
        for w, b in zip(W[:-1], B[:-1]):
            h = jax.nn.silu(h @ w + b)
        return jp.tanh((h @ W[-1] + B[-1])[:12])

    def episode(rng, mode):
        st = env.reset(rng)

        def one(c, _):
            s, acc = c
            s = env.step(s, act(s.obs, mode))
            quat = s.data.qpos[3:7]
            uz = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
            h = s.data.qpos[2] - env._floor_z
            q = s.data.qpos[env._jnt_qpos_adr]
            acc = jax.tree.map(lambda a, b: a + b, acc, s.metrics)
            return (s, acc), (s.reward, uz, h, jp.linalg.norm(q))

        acc0 = jax.tree.map(jp.zeros_like, st.metrics)
        (st, acc), (rew, uz, h, qn) = jax.lax.scan(one, (st, acc0), None, length=NSTEP)
        ok = (uz[-50:] > UP_OK) & (h[-50:] > H_OK)
        first = jp.where(jp.any(ok), jp.argmax(ok) * 0.02, float("nan"))
        return (jp.sum(rew), uz[-50:].mean(), h[-50:].mean(), qn[-1], first,
                acc["reward/c_height"] / NSTEP)

    rngs = jax.random.split(jax.random.PRNGKey(7), NEP)
    print("%-8s %10s %10s %10s %10s %10s %12s" %
          ("组", "回合回报", "末up_z", "末高度m", "‖q‖", "起立s", "c_height均值"))
    print("-" * 78)
    for mode in ("policy", "zero"):
        # mode 用默认参数捕获成 Python 常量，不参与 vmap（字符串不是 JAX 数组）
        batch = jax.jit(jax.vmap(lambda rng, m=mode: episode(rng, m)))
        (rew, uz, h, qn, t, ch) = batch(rngs)
        print("%-8s %10.3f %10.3f %10.4f %10.3f %10.2f %12.4f" %
              (mode, np.mean(rew), np.mean(uz), np.mean(h), np.mean(qn),
               float(np.nanmean(np.asarray(t))), np.mean(ch)))
    print("\n合格判据：末 up_z>0.9 且 末高度>0.13 m，同时显著优于零动作基线。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
