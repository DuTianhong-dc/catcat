"""ble_ota.py — 把 .bin 经 BLE(ESP32 桥) 写进 STM32 策略槽(通信协议.txt 第七章)

用法:
    python tools/ble_ota.py <bin> [--addr MAC] [--stat-only] [--dry-run]
    --addr 缺省自动扫描设备名"猫猫"。

流程: 连接 → 0x74 查状态 → 0x70 BEGIN(固件校验+擦目标槽, 等 1~4s) →
      0x71 DATA ×3203(逐帧等 ACK) → 0x72 END(全量 CRC+read-back 自检) → 0x74 复查。

重发策略(每帧):
    首发超时 → 清空待读 ACK 后重发一次; 重发收到 0x15 说明"首发其实已收"→ 继续;
    重发再超时 / 其它非 0 rc → 终止并发 ABORT(激活槽毫发无损, 重传即可)。
⚠️ 固件在更新期间卸力(协议 7.7), 结束后【保持卸力】—— 恢复运动要另发使能帧。
"""
import argparse
import asyncio
import os
import struct
import sys
import time

from bleak import BleakClient, BleakScanner

HERE = os.path.dirname(os.path.abspath(__file__))
SVC_KEY = "6e400001"
HDR = 72
CHUNK = 64
# T_BEGIN 放宽到 120s: 诊断"擦除到底多久"(2026-09-30); 若正常应在 10~20s 内见 ACK
T_STAT, T_BEGIN, T_FRAME, T_END = 5.0, 120.0, 3.0, 20.0

RC_NAME = {0x00: "OK", 0x10: "未在更新流程中", 0x11: "头部坏", 0x12: "维度不符",
           0x13: "pose_hash 不符", 0x14: "长度非法", 0x15: "顺序错", 0x16: "flash 擦写失败",
           0x17: "整体 CRC 不符", 0x18: "写后自检失败", 0x19: "seq 不大于激活槽(拒收)",
           0x1A: "正常模式不受理 BEGIN —— 请按住 KEY0 重上电进更新模式"}
SRC_NAME = {0: "出厂常量", 1: "槽A", 2: "槽B"}


class Fatal(Exception):
    pass


class OtaSession:
    """BLE 连接上的 CC 发送 + 5A A5 上行流解析。"""

    def __init__(self, cli, uart_rx):
        self.cli = cli
        self.rx = uart_rx
        self.buf = bytearray()
        self.raw = bytearray()            # 全部桥数据(剥前缀后), 诊断用(限 16KB)
        self.q = asyncio.Queue()          # (type, payload)

    # ---- 上行字节流 → 帧 ----
    def feed(self, data: bytes):
        # ESP32 桥每片带 1B 通道前缀: 0x02=桥数据(要) / 0x01=遥测(丢) / 其它=文本(丢)
        if len(data) < 1 or data[0] != 0x02:
            return
        self.raw += data[1:]
        if len(self.raw) > 16384:
            del self.raw[:len(self.raw) - 16384]
        self.buf += data[1:]
        while True:
            i = self.buf.find(b"\x5a\xa5")
            if i < 0:
                del self.buf[:-1]                      # 留最后 1 字节(可能是半个帧头)
                return
            if i:
                del self.buf[:i]
            if len(self.buf) < 5:
                return
            ln = self.buf[2]
            total = 2 + 1 + 1 + ln + 1
            if len(self.buf) < total:
                return
            if sum(self.buf[:total - 1]) & 0xFF != self.buf[total - 1]:
                del self.buf[:2]                       # 校验错: 跳过这个"5A A5"重扫
                continue
            self.q.put_nowait((self.buf[3], bytes(self.buf[4:4 + ln])))
            del self.buf[:total]

    # ---- 下行 CC 帧 ----
    async def send_cc(self, sub, payload=b""):
        body = bytes([sub]) + payload
        f = bytes([0xCC, len(body)]) + body
        f += bytes([sum(f) & 0xFF])
        await self.cli.write_gatt_char(self.rx, f, response=False)
        return len(f)

    async def wait(self, typ, timeout):
        """等一个指定 type 的上行帧; 期间状态包(0x01)等自动丢弃。超时返回 None。"""
        t0 = time.perf_counter()
        while True:
            remain = timeout - (time.perf_counter() - t0)
            if remain <= 0:
                return None
            try:
                t, p = await asyncio.wait_for(self.q.get(), remain)
            except asyncio.TimeoutError:
                return None
            if t == typ:
                return p

    def flush(self):
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except asyncio.QueueEmpty:
                break


def show_stat(p):
    if p is None:
        print("  (没等到 STAT 应答)")
        return
    rc, active = p[0], p[1]
    seq = struct.unpack("<I", p[2:6])[0]
    slotA, slotB = p[6], p[7]
    tag = p[8:32].rstrip(b"\x00").decode("utf-8", "replace")
    build = p[32:44].rstrip(b"\x00").decode("utf-8", "replace")
    print("  状态: 激活=%s  seq=%d | 槽A=%s 槽B=%s | tag=%s build=%s"
          % (SRC_NAME.get(active, active), seq,
             "有效" if slotA else "空", "有效" if slotB else "空", tag or "-", build or "-"))


async def run(a):
    # ---- 本地先验 ----
    if not a.stat_only and not a.probe:
        raw = open(a.bin, "rb").read()
        if len(raw) < HDR:
            raise Fatal("文件太短")
        hdr, data = raw[:HDR], raw[HDR:]
        magic, fmt, seq, dlen, dcrc = struct.unpack("<5I", hdr[:20])
        obs, act, phash, hcrc = struct.unpack("<4I", hdr[20:36])
        import zlib
        if magic != 0x50333343 or fmt != 2:
            raise Fatal("头部魔数/格式不对(要 fmt=2 的 v2 包)")
        if len(data) != dlen or phash == 0:
            raise Fatal("数据长度不符")
        if zlib.crc32(hdr[:32]) & 0xFFFFFFFF != hcrc:
            raise Fatal("hdr_crc 本地校验失败")
        if zlib.crc32(data) & 0xFFFFFFFF != dcrc:
            raise Fatal("data_crc 本地校验失败")
        tag = hdr[36:60].rstrip(b"\x00").decode("utf-8", "replace")
        build = hdr[60:72].rstrip(b"\x00").decode("utf-8", "replace")
        nframes = (len(data) + CHUNK - 1) // CHUNK
        print("=== 包 %s ===" % a.bin)
        print("  tag=%s  build=%s  seq=%d" % (tag, build, seq))
        print("  数据 %d B → %d 帧(每帧 %dB); obs=%d act_dim=%d" % (len(data), nframes, CHUNK, obs, act))
        if a.dry_run:
            print("  [dry-run] 到此为止(本地校验通过)")
            return 0

    # ---- 扫描/连接 ----
    addr = a.addr
    if not addr:
        print("扫描 BLE …")
        ds = await BleakScanner.discover(timeout=8.0)
        for d in ds:
            if d.name and "猫猫" in d.name:
                addr = d.address
                break
        if not addr:
            raise Fatal("没扫到名为'猫猫'的设备 —— ESP32 上电了吗?")
    print("连接 %s …" % addr)

    async with BleakClient(addr, timeout=30.0) as cli:
        uart_rx = uart_tx = None
        for svc in cli.services:
            if SVC_KEY in str(svc.uuid).lower():
                for ch in svc.characteristics:
                    u = str(ch.uuid).lower()
                    if "6e400002" in u:
                        uart_rx = ch
                    elif "6e400003" in u:
                        uart_tx = ch
        if uart_rx is None:
            raise Fatal("没找到 Nordic UART 写特征")

        sess = OtaSession(cli, uart_rx)
        await cli.start_notify(uart_tx, lambda _, d: sess.feed(bytes(d)))
        print("已连, 上行已订阅\n")

        # 连接后先发一帧 ABORT 清场: 万一上次传输中断、固件还挂在更新态(卸力),
        # 这帧让它退出、回正常态(不在更新态时回 rc=0, 无害)。
        await sess.send_cc(0x73)
        await sess.wait(0x76, 3.0)

        # ---- 0x74 查状态 ----
        await sess.send_cc(0x74)
        print("【当前策略】")
        p = await sess.wait(0x77, T_STAT)
        show_stat(p)
        if p is None:
            # 诊断: STAT 帧头在不在原始流里? (5A A5 2C 77 = STAT 应答的开头)
            hunt = sess.raw.find(bytes([0x5A, 0xA5, 0x2C, 0x77]))
            if hunt >= 0:
                print("  [诊断] 桥数据流里【有】STAT 帧头 @%d: %s"
                      % (hunt, bytes(sess.raw[hunt:hunt + 49]).hex().upper()))
            else:
                print("  [诊断] 桥数据流(剥前缀后 %d B)里【没有】 5A A5 2C 77"
                      " —— 固件没发出 STAT 应答" % len(sess.raw))
        if a.stat_only:
            return 0

        # ---- probe: 排查用 ----
        # ① 长度扫描: N 字节的 STAT 型帧(CC|len|74|pad|ck, STAT 不看 pad) —— 全应回 STAT。
        #    找"从多长开始断" → 区分 BLE 分片 / ESP32 转发 / 固件状态机。
        # ② 坏头 BEGIN(76B, magic 破坏 → 必被拒回 ACK rc=0x11, 绝不擦除)——
        #    验证"76B + BEGIN 路径"本身; 若它通而真头不通, 问题就锁定在"擦除那一环"。
        if a.probe:
            def mk_stat_frame(total_len):
                pad = total_len - 4
                body = bytes([0x74]) + b"\x00" * pad
                f = bytes([0xCC, len(body)]) + body
                return f + bytes([sum(f) & 0xFF])

            print("\n[probe] --- 长度扫描: N 字节 STAT 型帧(全应回 STAT) ---")
            for tot in (10, 20, 40, 60, 72):
                sess.raw.clear()
                frame = mk_stat_frame(tot)
                await cli.write_gatt_char(uart_rx, frame, response=False)
                await asyncio.sleep(1.5)
                i = sess.raw.find(b"\x5a\xa5\x2c\x77")
                print("  整帧 %2dB (len=%2d) → %s" % (tot, frame[1],
                      "STAT ✓" if i >= 0 else "无应答 ✗"))
            binp = os.path.join(HERE, "ota_out", "b1lr-peak_1.15M.bin")
            hdr = bytearray(open(binp, "rb").read()[:HDR])
            hdr[0] ^= 0xFF                      # 破坏 magic → 必被拒(不擦除), 回 ACK rc=0x11
            print("\n[probe] --- 坏头 BEGIN(76B, 不会擦除) ---")
            sess.raw.clear()
            await sess.send_cc(0x70, bytes(hdr))
            await asyncio.sleep(2.5)
            i = sess.raw.find(b"\x5a\xa5\x01\x76")
            if i >= 0:
                print("  坏头 BEGIN → ACK: %s (rc=0x%02X, 期望 0x11)"
                      % (bytes(sess.raw[i:i + 6]).hex().upper(), sess.raw[i + 4]))
            else:
                print("  坏头 BEGIN → 无应答 ✗")
            return 0

        # ---- 0x70 BEGIN ----
        print("\nBEGIN(固件校验头部并擦目标槽)…")
        t_beg = time.perf_counter()
        await sess.send_cc(0x70, hdr)
        r = await sess.wait(0x76, T_BEGIN)
        print("  (BEGIN 应答耗时 %.1fs)" % (time.perf_counter() - t_beg))
        if r is None:
            raise Fatal("BEGIN 超时(没收到 ACK)")
        if r[0] != 0:
            raise Fatal("BEGIN 被拒: rc=0x%02X %s" % (r[0], RC_NAME.get(r[0], "")))
        print("BEGIN OK — 目标槽已擦净, 开始传输")

        # ---- 0x71 DATA ----
        off, nfr, t0 = 0, 0, time.perf_counter()
        while off < len(data):
            n = min(CHUNK, len(data) - off)
            chunk = data[off:off + n]
            done = False
            for attempt in (1, 2):
                if attempt == 2:
                    sess.flush()                        # 清掉首发可能迟到的 ACK
                await sess.send_cc(0x71, struct.pack("<I", off) + chunk)
                r = await sess.wait(0x76, T_FRAME)
                if r is None:
                    continue                            # 超时 → 重发
                rc = r[0]
                if rc == 0:
                    done = True
                    break
                if rc == 0x15 and attempt == 2:         # 重发被拒 = 首发其实已收
                    done = True
                    break
                raise Fatal("DATA@%d 被拒: rc=0x%02X %s" % (off, rc, RC_NAME.get(rc, "")))
            if not done:
                raise Fatal("DATA@%d 两次超时" % off)
            off += n
            nfr += 1
            if nfr % 100 == 0 or off == len(data):
                el = time.perf_counter() - t0
                rate = nfr / el
                eta = (nframes - nfr) / rate if rate > 0 else 0
                print("  [%4d/%d] %5.1f%%  %.1f 帧/s  ETA %d:%02d"
                      % (nfr, nframes, 100.0 * off / len(data), rate, int(eta // 60), int(eta % 60)))
        el = time.perf_counter() - t0
        print("DATA 完成: %d 帧 %.1fs (%.1f 帧/s)" % (nfr, el, nfr / el))

        # ---- 0x72 END ----
        print("\nEND(全量 CRC + 写 magic + read-back 自检)…")
        await sess.send_cc(0x72)
        r = await sess.wait(0x76, T_END)
        if r is None:
            raise Fatal("END 超时")
        if r[0] != 0:
            raise Fatal("END 被拒: rc=0x%02X %s" % (r[0], RC_NAME.get(r[0], "")))
        print("END OK ✅")

        # ---- 0x74 复查 ----
        await sess.send_cc(0x74)
        print("\n【更新后策略】")
        show_stat(await sess.wait(0x77, T_STAT))
        print("\n新策略已生效(END 后立即切换, 无需复位)。")
        print("⚠️ 固件当前处于卸力状态 —— 要恢复运动: 用 ble_cmd.py 发 enable=1 的控制帧。")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin", nargs="?", help="ota_release.py 产出的 .bin")
    ap.add_argument("--addr", default="")
    ap.add_argument("--stat-only", action="store_true", help="只查状态(不发文件)")
    ap.add_argument("--probe", action="store_true", help="排查用: 连接后逐组发构造帧看应答")
    ap.add_argument("--dry-run", action="store_true", help="只做本地校验")
    a = ap.parse_args()
    try:
        return asyncio.run(run(a))
    except Fatal as e:
        print("\n!! 中止: %s" % e)
        print("   (激活槽未受影响; 排查后重跑本命令即可)")
        return 1
    except KeyboardInterrupt:
        print("\n!! 用户中断")
        return 130


if __name__ == "__main__":
    sys.exit(main())
