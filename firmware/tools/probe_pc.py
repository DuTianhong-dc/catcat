"""probe_pc.py — 用 pyocd halt 采样 CPU 现场(PC/SP/LR), 排查"固件卡在哪"。

用法:
    python tools/probe_pc.py [--rounds 8] [--interval 1.5]

⚠️ 需要 ST-Link 独占(SWD); 采样期间不要跑 read_dbg / 烧录。
⚠️ halt 时 IWDG 被 DBGMCU 冻结(固件配了 DBG_IWDG_STOP) —— 采样不会扰乱固件。
"""
import argparse
import time

from pyocd.core.helpers import ConnectHelper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--interval", type=float, default=1.5)
    a = ap.parse_args()

    sess = ConnectHelper.session_with_chosen_probe(options={
        "target_override": "stm32f407zgtx",
        "connect_mode": "attach",
        "reset_type": "hw",
    })
    sess.open()
    try:
        t = sess.target
        print("connected:", t.part_number)
        for i in range(a.rounds):
            t.halt()
            time.sleep(0.1)
            pc = t.read_core_register("pc")
            sp = t.read_core_register("sp")
            lr = t.read_core_register("lr")
            print("sample %2d: PC=0x%08X SP=0x%08X LR=0x%08X" % (i, pc, sp, lr))
            t.resume()
            if i != a.rounds - 1:
                time.sleep(a.interval)
    finally:
        sess.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
