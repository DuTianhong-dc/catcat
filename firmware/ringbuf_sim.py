#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环形缓冲实验：读写指针频率不一致时，固定序列还能不能活下来

模型（最朴素版本，不加任何额外防御）：
    数组 buf[N]，两个指针 rx_r / rx_w
    写指针每 iw 个 tick 写一次，内容 = 固定序列 0,1,2,...,L-1 循环
    读指针每 ir 个 tick 读一次（非空才读）
    满时：读指针让一格（丢最旧）—— 跟 cat_esp32.c:59 的规则完全一致

统计：
    把读出来的字节拼成长序列，在里面找"完整的固定序列"出现了多少次
    帧存活率 = 实际匹配次数 / 理论上写出的完整序列数

变量：
    N/L  —— 数组长度与固定序列长度的比例
    ir/iw —— 读周期与写周期的比值（>1 表示读比写慢，会积压）

用法：python ringbuf_sim.py
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def sim(N, L, iw, ir, n_ticks, ph=0):
    """跑一次仿真，返回各项计数。

    ph —— 读事件相对写事件的固定相位偏移(tick)。0 = 同相位。
          读事件发生在 t = ph, ph+ir, ph+2*ir, ...
    """
    seq = list(range(L))          # 固定序列：0,1,...,L-1（无自重叠）
    buf = [0] * N
    r = w = 0
    k = 0                         # 已经写出去多少个字节(用于取 seq 的下一位)

    n_written = 0                 # 写指针实际写入的字节数
    n_dropped = 0                 # 因"满"而被丢掉的字节数
    out = []                      # 读指针真正读出来的字节序列

    for t in range(n_ticks):
        if t % iw == 0:                       # ---- 写事件 ----
            nxt = (w + 1) % N
            if nxt == r:                      # 满了 → 读指针让一格，丢最旧
                r = (r + 1) % N
                n_dropped += 1
            buf[w] = seq[k % L]
            k += 1
            w = nxt
            n_written += 1

        if (t - ph) % ir == 0:                # ---- 读事件(相对写事件偏 ph 个 tick) ----
            if r != w:                        # 非空才读
                out.append(buf[r])
                r = (r + 1) % N

    n_matched = 0                             # 在 out 里找完整的 seq
    for i in range(len(out) - L + 1):
        if out[i:i + L] == seq:
            n_matched += 1

    return {
        "n_written": n_written,
        "n_frames": n_written // L,           # 理论上写出了多少个完整序列
        "n_read": len(out),
        "n_dropped": n_dropped,
        "n_matched": n_matched,
    }


L_DEF = 10          # 固定序列长度（对应我们 10 字节的下行帧）
IW_DEF = 10         # 写周期固定：每 10 tick 写一次
TICKS_DEF = 200000  # 总时长


def freq_sweep():
    """实验一：扫频率比（相位固定为 0）。"""
    print("=" * 96)
    print("实验一：读写频率不一致时，固定序列的存活率（相位固定 0）")
    print(f"固定序列长 L = {L_DEF}    写周期 iw = {IW_DEF} tick/次    总时长 = {TICKS_DEF} tick")
    print("=" * 96)
    print()

    for N in [8, 32, 128]:
        print(f"--- 数组长度 N = {N}  (N/L = {N / L_DEF:.1f}) ---")
        print(f"{'ir':>4} {'读/写':>7} {'写字节':>8} {'写帧数':>7} "
              f"{'读字节':>8} {'丢弃':>8} {'命中':>7} {'帧存活率':>9}")
        for ir in [3, 5, 8, 10, 11, 12, 15, 20, 30]:
            res = sim(N, L_DEF, IW_DEF, ir, TICKS_DEF)
            rate = res["n_matched"] / res["n_frames"] if res["n_frames"] else 0.0
            print(f"{ir:>4} {ir / IW_DEF:>7.2f} {res['n_written']:>8} {res['n_frames']:>7} "
                  f"{res['n_read']:>8} {res['n_dropped']:>8} {res['n_matched']:>7} "
                  f"{rate * 100:>8.1f}%")
        print()


def phase_sweep(N=32):
    """实验二：扫读事件相对写事件的相位偏移（频率比固定）。"""
    ph_list = list(range(IW_DEF))       # 0..iw-1，正好覆盖一个完整写周期
    print("=" * 96)
    print("实验二：相位偏移的影响（读事件相对写事件偏 ph 个 tick）")
    print(f"L = {L_DEF}   iw = {IW_DEF}   N = {N}   总时长 = {TICKS_DEF} tick")
    print("=" * 96)
    print()

    for ir in [10, 11, 12, 15, 20]:
        print(f"--- ir = {ir}  (读/写 = {ir / IW_DEF:.2f}) ---")
        print(f"{'相位ph':>7} {'写帧':>7} {'命中':>7} {'帧存活率':>9}")
        for ph in ph_list:
            res = sim(N, L_DEF, IW_DEF, ir, TICKS_DEF, ph)
            rate = res["n_matched"] / res["n_frames"] if res["n_frames"] else 0.0
            print(f"{ph:>7} {res['n_frames']:>7} {res['n_matched']:>7} {rate * 100:>8.1f}%")
        print()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "phase":
        phase_sweep()
    else:
        freq_sweep()


if __name__ == "__main__":
    main()
