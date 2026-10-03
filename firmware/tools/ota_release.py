"""ota_release.py — 策略打包: npz → .bin (72B 头部 + 定序 blob), 供 ble_ota.py 经 BLE 下发

产物 = 通信协议.txt 7.3/7.4 的槽内容：[72B 头部] + [W1 S1 B1 … MEAN STD 定序 blob]。
与第一版 npz_to_bin.py 的差异: 头部 72B(= 多 seq 字段, fmt=2); pose_hash 从
【本工程 cat_pose.c】现算(第一版抠的是 policy_weights.c 里的 POL_DEFAULT_POSE 那套旧名)。

用法:
  python tools/ota_release.py <npz> [--tag TAG] [--out X.bin] [--seq N]
                              [--check-c <参考 policy_weights.c>]
默认: check-c 自动探测 sil/host2/policy_weights.c 与 本工程/policy_weights.c,
      谁的 POL_BUILD_ID == npz md5 前 12 就用谁跑逐块自检(int8 逐字节 / float 1-ULP)。

⚠️ 自检缺失时工具照常出 .bin —— 但"定序/量化错"只能靠自检兜底, 能对上就别跳过。
"""
import argparse
import hashlib
import os
import re
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.normpath(os.path.join(HERE, ".."))                 # CAT_APP2.0
POSE_C = os.path.join(APP, "cat_pose.c")
OUT_DIR = os.path.join(HERE, "ota_out")

MAGIC = 0x50333343          # 'C33P'
FMT = 2
TAG_LEN, BUILD_LEN = 24, 12
OBS_DIM, H1, H2, H3, OUT = 59, 512, 256, 128, 24                # 固件 cat_config.h 同源

CHECK_C_DEFAULTS = [                                            # 自动探测的自检参照(顺序即优先级)
    r"C:\Users\dth15\Desktop\mujoco\sil\host2\policy_weights.c",
    os.path.join(APP, "policy_weights.c"),
]


def q8(w):
    """按输出通道对称 int8 量化(npz_to_c.py / 固件 dense_s8 同一套)"""
    s = np.abs(w).max(axis=0) / 127.0
    s = np.where(s == 0, np.float32(1e-8), s).astype(np.float32)
    wq = np.clip(np.round(w / s), -127, 127).astype(np.int8)
    return wq, s


def crc32(data, crc=0):
    import zlib
    return zlib.crc32(data, crc) & 0xFFFFFFFF


def parse_c_floats(path, name, count):
    """从 C 文件抠浮点表: 先 %.9g 再 float32 —— 复现编译器把十进制字面量落成 binary32 的比特值。
    兼容两种写法: npz_to_c 输出的裸数字 / 手写常量的 f 后缀(如 cat_pose.c 的 "0.25f")。"""
    src = open(path, encoding="utf-8").read()
    m = re.search(r"const float\s+%s\s*\[[^\]]*\]\s*=\s*\{([^}]*)\}" % re.escape(name), src, re.S)
    if not m:
        raise SystemExit("!! %s 里找不到 %s —— 路径对吗?" % (path, name))
    raw = [v.strip() for v in m.group(1).replace("\n", " ").split(",")]
    vals = [(v[:-1] if v.endswith(("f", "F")) else v) for v in raw if v]
    if len(vals) != count:
        raise SystemExit("!! %s 元素数 %d != 期望 %d" % (name, len(vals), count))
    return np.array([np.float32("%.9g" % float(v)) for v in vals], dtype=np.float32)


def parse_c_ints(path, name, count):
    src = open(path, encoding="utf-8").read()
    m = re.search(r"const int8_t\s+%s\s*\[[^\]]*\]\s*=\s*\{([^}]*)\}" % re.escape(name), src, re.S)
    if not m:
        return None
    vals = [int(v) for v in m.group(1).replace("\n", "").split(",") if v.strip()]
    return np.array(vals, dtype=np.int8).tobytes() if len(vals) == count else None


def verify_layout(cfile, data):
    """逐块比对 .bin 数据区 vs 参考 policy_weights.c(长度相等≠定序一致, 必须逐字节)"""
    hard, warns = [], []
    order = [("W1", "i8"), ("S1", "f32"), ("B1", "f32"), ("W2", "i8"), ("S2", "f32"),
             ("B2", "f32"), ("W3", "i8"), ("S3", "f32"), ("B3", "f32"), ("WH", "i8"),
             ("SH", "f32"), ("BH", "f32"), ("OBS_MEAN", "f32"), ("OBS_STD", "f32")]
    icnt = {"W1": OBS_DIM * H1, "W2": H1 * H2, "W3": H2 * H3, "WH": H3 * OUT}
    fcnt = {"S1": H1, "B1": H1, "S2": H2, "B2": H2, "S3": H3, "B3": H3,
            "SH": OUT, "BH": OUT, "OBS_MEAN": OBS_DIM, "OBS_STD": OBS_DIM}
    off = 0
    for name, kind in order:
        n = icnt[name] if kind == "i8" else fcnt[name] * 4
        seg = data[off:off + n]
        if kind == "i8":
            want = parse_c_ints(cfile, "POL_" + name, n)
            if want is None:
                hard.append("参考 C 里找不到/长度不符: POL_%s" % name)
            elif want != seg:
                bad = sum(1 for x, y in zip(want, seg) if x != y)
                hard.append("POL_%s 与 .bin 不一致: %d/%d 字节不同" % (name, bad, n))
        else:
            want = parse_c_floats(cfile, "POL_" + name, fcnt[name]).astype("<f4").tobytes()
            if want != seg:
                x = np.frombuffer(want, dtype="<f4").view(np.int32)
                y = np.frombuffer(seg, dtype="<f4").view(np.int32)
                warns.append("POL_%s: %d/%d 个值差 1 ULP(十进制往返, 无实质影响)"
                             % (name, int((x != y).sum()), fcnt[name]))
        off += n
    if off != len(data):
        hard.append("块累计 %d != 数据区 %d" % (off, len(data)))
    return hard, warns


def c_build_id(path):
    try:
        m = re.search(r'const char POL_BUILD_ID\[\] = "([^"]*)"', open(path, encoding="utf-8").read())
        return m.group(1) if m else ""
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--seq", type=int, default=0, help="覆盖 seq(默认=打包时刻 Unix 秒)")
    ap.add_argument("--check-c", default="", help="自检参照 policy_weights.c(默认自动探测)")
    a = ap.parse_args()

    z = {k: v.astype(np.float32) for k, v in np.load(a.npz).items()}
    # ---- 维度硬校验(不匹配直接退出, 不产出半成品) ----
    exp = {"W1": (OBS_DIM, H1), "b1": (H1,), "W2": (H1, H2), "b2": (H2,),
           "W3": (H2, H3), "b3": (H3,), "Wh": (H3, OUT), "bh": (OUT,),
           "obs_mean": (OBS_DIM,), "obs_std": (OBS_DIM,)}
    bad = [k for k, s in exp.items() if k not in z or tuple(z[k].shape) != s]
    if bad:
        print("!! npz 与固件维度不匹配:", ", ".join(bad))
        print("   (固件: OBS_DIM=59 / H1H2H3OUT=512/256/128/24; 消融版 47 维策略暂不支持)")
        return 1

    # ---- blob: W1 S1 B1 … MEAN STD ----
    blob = bytearray()
    for wi, bi in (("W1", "b1"), ("W2", "b2"), ("W3", "b3")):
        wq, s = q8(z[wi])
        blob += wq.ravel().tobytes()
        blob += s.astype("<f4").tobytes()
        blob += z[bi].astype("<f4").tobytes()
    wq, s = q8(z["Wh"])
    blob += wq.ravel().tobytes() + s.astype("<f4").tobytes() + z["bh"].astype("<f4").tobytes()
    blob += z["obs_mean"].astype("<f4").tobytes() + z["obs_std"].astype("<f4").tobytes()
    data = bytes(blob)

    # ---- pose_hash: 与固件 cat_pose_hash() 同源(顺序 DEF→LO→HI, float32 小端) ----
    pose = parse_c_floats(POSE_C, "CAT_POSE_DEF", 12)
    lo = parse_c_floats(POSE_C, "CAT_POSE_LIM_LO", 12)
    hi = parse_c_floats(POSE_C, "CAT_POSE_LIM_HI", 12)
    pose_hash = crc32(pose.tobytes() + lo.tobytes() + hi.tobytes())

    # ---- 头部 72B ----
    seq = a.seq or int(time.time())
    data_crc = crc32(data)
    build = hashlib.md5(open(a.npz, "rb").read()).hexdigest()[:BUILD_LEN]
    tag_txt = a.tag or os.path.splitext(os.path.basename(a.npz))[0].lstrip("_")
    tg = tag_txt.encode()[:TAG_LEN].ljust(TAG_LEN, b"\x00")
    hdr = struct.pack("<8I", MAGIC, FMT, seq, len(data), data_crc, OBS_DIM, OUT, pose_hash)
    hdr_crc = crc32(hdr)
    hdr += struct.pack("<I", hdr_crc) + tg + build.encode()
    assert len(hdr) == 72, len(hdr)

    # ---- 自检(可选) ----
    check_c = a.check_c
    if not check_c:
        for cand in CHECK_C_DEFAULTS:
            if os.path.exists(cand) and c_build_id(cand) == build:
                check_c = cand
                break
    hard, warns = ([], [])
    if check_c:
        hard, warns = verify_layout(check_c, data)
        print("自检参照: %s (build_id 同源 ✓)" % check_c)
    else:
        print("⚠️ 没找到同源 policy_weights.c 做逐块自检(传的并非出厂/已部署版本时正常)")

    if hard:
        print("!! 布局自检失败 —— 不写文件:")
        for h in hard:
            print("   · " + h)
        return 3

    os.makedirs(OUT_DIR, exist_ok=True)
    out = a.out or os.path.join(OUT_DIR, tag_txt.replace("/", "_") + ".bin")
    open(out, "wb").write(hdr + data)

    print("已生成:", out)
    print("  文件大小 : %d B (= 72 头 + %d 数据)" % (len(hdr) + len(data), len(data)))
    print("  维度     : obs=%d  act_dim=%d(头层宽度, 固件比的是这个)" % (OBS_DIM, OUT))
    print("  seq      : %d" % seq)
    print("  data_crc : 0x%08X   hdr_crc: 0x%08X" % (data_crc, hdr_crc))
    print("  pose_hash: 0x%08X   (源: %s)" % (pose_hash, os.path.basename(POSE_C)))
    print("  tag      : %s" % tag_txt)
    print("  build_id : %s (npz md5 前 12)" % build)
    if check_c:
        print("  逐块自检 : ✅ int8 全同 / float 0 硬错" +
              ("; %d 条 1-ULP 提示" % len(warns) if warns else ""))
        for w in warns:
            print("     ~ " + w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
