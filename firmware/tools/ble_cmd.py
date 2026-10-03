"""ble_cmd.py — 用电脑蓝牙扮演小程序, 给第二版固件发下行指令并收上行

用法:
    python tools/ble_cmd.py --addr XX:XX:XX:XX:XX:XX [--vx 0.3] [--wz 0.0]
                            [--enable 1] [--seconds 5] [--hz 50]
                            [--pan 90] [--tilt 90]     # 云台角 0~180°(90=中位; 缺省=中位)

⚠️ 帧格式(第二版 10 字节)必须与固件 cat_esp32.c 严格一致:
       AA 55 | vx[2] | wz[2] | flag | pan | tilt | ck
        0  1    2-3     4-5      6      7     8      9
    ck = (f[2]+...+f[8]) & 0xFF  —— 覆盖 7 字节, 对应固件 ESP32_DL_CK_FIRST=2 / CK_OFF=9
    vx/wz 是定点: 实际值 = 原始/1000 (mm/s, mrad/s)
    flag: bit0 使能, bit1 急停

⚠️ 固件 DEADMAN=1000ms: 1 秒收不到下行就判失联、把使能归零 → 本工具按 50Hz 持续发。

⚠️ 本脚本只用于【第二版】(10 字节)。第一版固件是 12 字节(带 vy), 别拿这个去喂它。
"""
import argparse
import asyncio
import struct
import sys
import time

from bleak import BleakClient

SVC_KEY = "6e400001"


def frame_aa(vx_mms, wz_mrad, enable, estop=0, pan=0x00, tilt=0x00):
    """按第二版协议组一帧 10 字节的下行运动帧。
    pan/tilt: int8, 0 = 中位 90°(⚠️ 别用 0x7F — 那是 180° 顶死限位); 缺省 0x00 = 中位。"""
    flag = (enable & 1) | ((estop & 1) << 1)
    body = list(struct.pack("<hh", vx_mms, wz_mrad)) + [flag, pan & 0xFF, tilt & 0xFF]
    ck = sum(body) & 0xFF                      # 覆盖 f[2..8], 共 7 字节
    return bytes([0xAA, 0x55] + body + [ck])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", required=True)
    ap.add_argument("--vx", type=float, default=0.0, help="m/s, 正=前进")
    ap.add_argument("--wz", type=float, default=0.0, help="rad/s, 正=左转")
    ap.add_argument("--enable", type=int, default=0)
    ap.add_argument("--estop", type=int, default=0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--pan",  type=float, default=90.0, help="云台水平角 0~180°(90=中位)")
    ap.add_argument("--tilt", type=float, default=90.0, help="云台俯仰角 0~180°(90=中位)")
    a = ap.parse_args()

    vx_mms = int(round(a.vx * 1000))
    wz_mrad = int(round(a.wz * 1000))
    # deg 0~180 → int8(与小程序 proto.js 同一公式); 越界先钳
    pan_i8  = max(-127, min(127, round(max(0.0, min(180.0, a.pan))  / 180 * 254 - 127)))
    tilt_i8 = max(-127, min(127, round(max(0.0, min(180.0, a.tilt)) / 180 * 254 - 127)))
    pkt = frame_aa(vx_mms, wz_mrad, a.enable, a.estop, pan_i8 & 0xFF, tilt_i8 & 0xFF)
    print("下行帧: %s   (vx=%+.3f m/s, wz=%+.3f rad/s, enable=%d, 云台 %.0f°/%.0f° → int8 %d/%d)"
          % (pkt.hex().upper(), a.vx, a.wz, a.enable, a.pan, a.tilt, pan_i8, tilt_i8))

    rx_chunks = []
    parse_buf = bytearray()
    latest_st = {}
    uart_rx = None
    uart_tx = None

    async with BleakClient(a.addr, timeout=30.0) as cli:
        # 按服务找 Nordic UART 的 rx(写) / tx(通知)
        for svc in cli.services:
            if SVC_KEY in str(svc.uuid).lower():
                for ch in svc.characteristics:
                    u = str(ch.uuid).lower()
                    if "6e400002" in u:
                        uart_rx = ch
                    elif "6e400003" in u:
                        uart_tx = ch
        if uart_rx is None:
            print("⚠️ 没找到 Nordic UART 的写特征")
            return 1
        print("已连 %s   rx=%s" % (a.addr, uart_rx.uuid))

        def on_notify(_, data: bytearray):
            d = bytes(data)
            rx_chunks.append((time.time(), d))
            # ESP32 桥每片带 1B 通道前缀: 0x02=桥数据(要) / 0x01=遥测 / 其它=文本(都丢)
            if not d or d[0] != 0x02:
                return
            # 流解析(5A A5 帧): 找状态包(type=0x01)抽"指令回显"——下行是否真的送达的直接证据
            parse_buf.extend(d[1:])
            while True:
                i = parse_buf.find(b"\x5a\xa5")
                if i < 0:
                    del parse_buf[:-1]
                    break
                if i:
                    del parse_buf[:i]
                if len(parse_buf) < 5:
                    break
                ln = parse_buf[2]
                tot = 2 + 1 + 1 + ln + 1
                if len(parse_buf) < tot:
                    break
                if (sum(parse_buf[:tot - 1]) & 0xFF) != parse_buf[tot - 1]:
                    del parse_buf[:2]
                    continue
                typ = parse_buf[3]
                pl = bytes(parse_buf[4:4 + ln])
                del parse_buf[:tot]
                if typ == 0x01 and len(pl) >= 79:
                    latest_st["ms"]    = struct.unpack("<I", pl[0:4])[0]
                    latest_st["loops"] = struct.unpack("<I", pl[4:8])[0]
                    latest_st["vx"]    = struct.unpack("<h", pl[10:12])[0]
                    latest_st["wz"]    = struct.unpack("<h", pl[12:14])[0]
                    latest_st["pan"]   = struct.unpack("<b", pl[77:78])[0]   # 云台回显(0=中位90°)
                    latest_st["tilt"]  = struct.unpack("<b", pl[78:79])[0]

        if uart_tx is not None:
            await cli.start_notify(uart_tx, on_notify)
            print("   已订阅上行通知")
        else:
            print("   ⚠️ 没找到通知特征, 收不到上行")

        # 50Hz 持续发, 同时收
        n = int(a.seconds * a.hz)
        t0 = time.perf_counter()
        for i in range(n):
            await cli.write_gatt_char(uart_rx, pkt, response=False)
            target = t0 + (i + 1) / a.hz
            w = target - time.perf_counter()
            if w > 0:
                await asyncio.sleep(w)
        print("已发 %d 帧 (%.1f 秒 @ %.0fHz)" % (n, a.seconds, a.hz))
        mid_st = dict(latest_st)    # 发送期间的回显快照(之后静默 1.5s 会被 DEADMAN 归零)

        # 再静默收 1.5s 抓后续上行
        await asyncio.sleep(1.5)

        # 发送后清零(防呆: 别让猫一直走)
        if a.enable:
            stop = frame_aa(0, 0, 1)            # 仍使能, 速度归零
            for _ in range(20):
                await cli.write_gatt_char(uart_rx, stop, response=False)
                await asyncio.sleep(0.02)
            print("已补发 20 帧 vx=wz=0 (防呆)")

    print("\n=== 上行统计 ===")
    print("共收到 %d 片" % len(rx_chunks))
    if not rx_chunks:
        print("⚠️ 完全没收到上行 —— ESP32 没在转发 STM32 的串口, 或上报没开")
        return 0
    sizes = {}
    for _, d in rx_chunks:
        sizes[len(d)] = sizes.get(len(d), 0) + 1
    for ln, cnt in sorted(sizes.items()):
        print("  %d 字节 × %d 片" % (ln, cnt))
    span = rx_chunks[-1][0] - rx_chunks[0][0]
    if span > 0:
        print("  跨度 %.2fs → 约 %.1f 片/秒" % (span, len(rx_chunks) / span))
    print("\n=== 状态包解析(验证下行是否送达固件) ===")
    if mid_st:
        print("  最新: ms=%d loops=%d | 指令回显 vx=%d wz=%d  (本工具发的是 vx=%d wz=%d)"
              % (mid_st["ms"], mid_st["loops"], mid_st["vx"], mid_st["wz"], vx_mms, wz_mrad))
        if "pan" in mid_st:
            print("  云台回显: pan=%d tilt=%d  (0 = 中位 90°)" % (mid_st["pan"], mid_st["tilt"]))
        if mid_st["vx"] == vx_mms and mid_st["wz"] == wz_mrad:
            print("  ✅ 回显一致 —— 下行真的送达固件, 链路 OK")
        else:
            print("  ⚠️ 回显不一致 —— 下行可能没送达(或发的是其它值)")
    else:
        print("  (没解析到状态包)")
    print("\n前 3 片(hex):")
    for _, d in rx_chunks[:3]:
        print("  " + d.hex().upper())
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
