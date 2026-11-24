# -*- coding: utf-8 -*-
r"""pd_baseline.py —— 第 0 步物理基线：不跑 RL，只下发恒定站姿目标，看纯 PD 能不能撑住。

背景（外部诊断 2026-11-24 第 0 条）：s2stand 用 delta_current，a=0 时目标角 = 当前角，
稳态抗重力力矩为 0；且模型 kv=0（无阻尼）。若纯 PD 连 near_stand 都撑不住，
那"站不住"就不是策略问题，而是执行器/参数问题，先修物理再谈奖励。

做法：对起点库每一条 qpos，ctrl 恒为 0（= 站姿，CAT_POSE_DEF 全 0），
      跑 T 秒，统计机身高度、up_z、关节力矩占额、角度抖动。

用法:
  python pd_baseline.py                 # 全 300 条
  python pd_baseline.py --n 30 --sec 3
"""
import argparse
import io
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
MODEL = os.path.join(PKG, "model", "scene_selfright_full_v2.xml")
BANK = os.path.join(PKG, "data", "curriculum_mix.npz")
J = ["joint1", "joint2", "joint3", "joint6", "joint7", "joint8",
     "joint11", "joint12", "joint13", "joint16", "joint17", "joint18"]
FEET = ["foot_rr", "foot_rl", "foot_fr", "foot_fl"]
CTRL_DT, NSUB = 0.02, 10


def feet_down(m, d, thr=0.005):
    z = [float(d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)][2]) for n in FEET]
    return sum(1 for v in z if v < thr), z


def run_one(m, q0, nstep, vadr, jadr_now, ctrl_q, foot_ids, delta=False):
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    d.qvel[:] = 0.0
    d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)
    bz, upz, tau, qv, nfeet = [], [], [], [], []
    for k in range(nstep):
        d.ctrl[0] = 0.0
        if delta:                       # a≡0 的 delta_current：目标角 = 当前角（无重力补偿）
            d.ctrl[1:13] = np.asarray(d.qpos)[jadr_now]
        else:                           # 绝对锚点：目标角恒为站姿
            d.ctrl[1:13] = ctrl_q
        for _ in range(NSUB):
            mujoco.mj_step(m, d)
        qp = np.asarray(d.qpos)
        bz.append(float(qp[2]))
        upz.append(float(1.0 - 2.0 * (qp[4] ** 2 + qp[5] ** 2)))
        tau.append(np.abs(np.asarray(d.actuator_force)[1:13]).copy())
        qv.append(np.abs(np.asarray(d.qvel)[vadr]).copy())
        nfeet.append(sum(1 for s in foot_ids if d.site_xpos[s][2] < 0.005))
    return (np.array(bz), np.array(upz), np.array(tau), np.array(qv), np.array(nfeet))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="只跑前 N 条（0=全部）")
    ap.add_argument("--sec", type=float, default=5.0)
    ap.add_argument("--md", default=None)
    ap.add_argument("--png", default=None)
    ap.add_argument("--mode", default="anchor", choices=("anchor", "delta"),
                    help="anchor=目标恒为站姿(绝对锚点) / delta=目标=当前角(a≡0 的 delta_current)")
    a = ap.parse_args()
    delta = (a.mode == "delta")

    m = mujoco.MjModel.from_xml_path(MODEL)
    g = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    bad = [n for n in J if g(n) < 0]
    if bad:
        raise SystemExit("模型里找不到关节 %s" % bad)
    jadr = np.array([m.jnt_qposadr[g(n)] for n in J], dtype=int)
    vadr = np.array([m.jnt_dofadr[g(n)] for n in J], dtype=int)
    foot_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in FEET]
    trange = np.abs(np.asarray(m.actuator_forcerange)[1:13, 0])   # 每关节力矩上限 (N·m)
    kv = np.asarray(m.dof_damping)[vadr]

    z = np.load(BANK, allow_pickle=True)
    Q, GRP = np.asarray(z["qpos"], float), np.asarray(z["group"])
    if a.n:
        Q, GRP = Q[:a.n], GRP[:a.n]
    nstep = int(a.sec / CTRL_DT)
    ctrl_q = np.zeros(12)                      # = 站姿（CAT_POSE_DEF 全 0）

    print("模型 %s" % os.path.basename(MODEL))
    print("  kp=%s  force=%s  dof_damping(kv)=%s" %
          (np.round(np.asarray(m.actuator_gainprm)[1:4, 0], 2),
           np.round(trange[:3], 2), np.round(kv[:3], 4)))
    print("起点 %d 条，恒定目标 q_stand=0，跑 %.0f s\n" % (len(Q), a.sec))

    rows = []
    curves = {}
    for i, q0 in enumerate(Q):
        bz, upz, tau, qv, nf = run_one(m, q0, nstep, vadr, jadr, ctrl_q, foot_ids, delta)
        sat = float((tau > trange * 0.999).mean())
        rows.append(dict(
            grp=str(GRP[i]), bz0=bz[0], bz_end=bz[-1], bz_min=bz.min(),
            upz_end=upz[-1], upz_min=upz.min(),
            tau_p95=float(np.percentile(tau, 95)), tau_max=float(tau.max()),
            sat=sat, qv_end=float(qv[-50:].mean()), nf_end=int(nf[-1])))
        if str(GRP[i]) not in curves:
            curves[str(GRP[i])] = []
        if len(curves[str(GRP[i])]) < 12:
            curves[str(GRP[i])].append(bz)
        if (i + 1) % 60 == 0:
            print("  %d/%d" % (i + 1, len(Q)))

    out = io.StringIO()
    _mode_txt = "delta（目标=当前角，a≡0 的 delta_current）" if delta else "anchor（目标恒为站姿 q_stand=0）"
    out.write("# 第 0 步 物理基线 · %s（%d 条起点，%.0f s）\n\n" % (_mode_txt, len(rows), a.sec))
    out.write("模型 `model/scene_selfright_full_v2.xml`，%s；`kp=%s`、`kv=%s`、`τ_max=%s N·m`。\n\n"
              % (_mode_txt,
                 np.round(np.asarray(m.actuator_gainprm)[1:4, 0], 2).tolist(),
                 np.round(kv[:3], 4).tolist(), np.round(trange[:3], 2).tolist()))
    out.write("| 分组 | n | 起始 bz | 末态 bz | 末态 up_z | 力矩 p95 | 顶满比例 | 末段角速度 |\n")
    out.write("|---|---|---|---|---|---|---|---|\n")
    for grp in ("near_stand", "transition", "original"):
        r = [x for x in rows if x["grp"] == grp]
        if not r:
            continue
        f = lambda k: np.array([x[k] for x in r])
        out.write("| %s | %d | %.3f | %.3f | %+.2f | %.2f | %.1f%% | %.2f rad/s |\n" % (
            grp, len(r), f("bz0").mean(), f("bz_end").mean(), f("upz_end").mean(),
            np.median(f("tau_p95")), 100 * f("sat").mean(), f("qv_end").mean()))
    ok = [x for x in rows if x["bz_end"] > 0.12 and x["upz_end"] > 0.8]
    out.write("\n**站住（bz>0.12 且 up_z>0.8）**: %d/%d = %.0f%%\n" % (len(ok), len(rows),
                                                                 100 * len(ok) / len(rows)))
    out.write("\n判读：near_stand 组若大面积站不住，且 `顶满比例` 高 → 力矩不够；"
              "若 `顶满比例` 低但高度掉 → 目标角不产生抗重力偏置（delta 参数化 + kv=0 的锅）；"
              "若 `末段角速度` 大 → 无阻尼震荡。\n")
    txt = out.getvalue()
    print("\n" + txt)
    if a.md:
        io.open(a.md, "w", encoding="utf-8").write(txt)
        print("md →", a.md)

    if a.png:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return 0
        W, H = 1200, 700
        img = Image.new("RGB", (W, H), (250, 250, 252))
        dr = ImageDraw.Draw(img)
        try:
            fnt = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 18)
            fnt2 = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 14)
        except Exception:
            fnt = fnt2 = ImageFont.load_default()
        dr.text((24, 14), "第 0 步 物理基线 · %s" % ("delta（目标=当前角, a≡0）" if delta else "anchor（目标恒为站姿）"), font=fnt, fill=(20, 20, 20))
        x0, y0, w, h = 70, 70, W - 110, H - 150
        dr.rectangle([x0, y0, x0 + w, y0 + h], fill=(255, 255, 255), outline=(200, 200, 200))
        ys = [b for v in curves.values() for b in v]
        lo = min(b.min() for b in ys) - 0.01
        hi = max(b.max() for b in ys) + 0.01
        px = lambda k, v: (x0 + 8 + (w - 16) * k / (len(ys[0]) - 1), y0 + h - 26 - (h - 55) * (v - lo) / (hi - lo))
        for k in range(5):
            vv = lo + (hi - lo) * k / 4
            dr.line([x0 + 8, px(0, vv)[1], x0 + w - 8, px(0, vv)[1]], fill=(232, 232, 232))
            dr.text((x0 - 58, px(0, vv)[1] - 8), "%.3f" % vv, font=fnt2, fill=(110, 110, 110))
        cols = {"near_stand": (220, 70, 60), "transition": (60, 130, 220), "original": (60, 160, 100)}
        for grp, cs in curves.items():
            for b in cs:
                dr.line([px(k, float(v)) for k, v in enumerate(b)], fill=cols.get(grp, (90, 90, 90)), width=2)
        yy = y0 + h + 8
        for i, (grp, c) in enumerate(cols.items()):
            dr.line([x0 + 8 + i * 190, yy + 10, x0 + 38 + i * 190, yy + 10], fill=c, width=3)
            dr.text((x0 + 44 + i * 190, yy), grp, font=fnt2, fill=(70, 70, 70))
        dr.text((x0, yy + 26), "横轴 %.0f s；纵轴机身高度 bz(m)；每条起点一条线（每组合最多画 12 条）"
                % a.sec, font=fnt2, fill=(110, 110, 110))
        img.save(a.png)
        print("png →", a.png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
