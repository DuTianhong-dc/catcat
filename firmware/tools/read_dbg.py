"""read_dbg.py — 用 ST-Link 读固件观测区(g_dbg)并解析成可读状态

用法:
    python tools/read_dbg.py [地址] [字节数]
    默认 0x10000000 / 432 (与 cat_sys.h 的 dbg_t 布局对应)

原理: 观测区是 CCM RAM 里的固定地址结构体, 不接串口也能看固件跑到哪、自检结果如何。
      ⚠️ 改了 cat_sys.h 的 dbg_t 就必须同步改下面的 LAYOUT —— 否则解析全错(而且不会报错)。

⚠️ CLI 的 -r32 参数是【字节数】不是字数, 且会按 4 字节对齐。
"""
import re
import subprocess
import struct
import sys
import time

CLI = (r"C:\Program Files\STMicroelectronics\STM32Cube"
       r"\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe")

DBG_MAGIC = 0xCA7DB001
DBG_FW_VER = 0x20260928      # ⚠️ 必须与 cat_sys.h 的 DBG_FW_VER 一致
CYCLES_PER_US = 168          # CPU_MHZ
STAGE = {1: "INIT", 2: "BOOT(等外设)", 3: "READY(进主循环)", 4: "TIMEOUT(没等到外设)"}

# (偏移, 名字, 类型, 数量, 说明)  —— 类型: u32 / f32
LAYOUT = [
    (0,   "magic",        "u32", 1,  "恒为 0xCA7DB001"),
    (4,   "stage",        "u32", 1,  "DBG_STAGE_*"),
    (8,   "loops",        "u32", 1,  "主循环圈数, 在涨说明没死"),
    (12,  "ms",           "u32", 1,  "运行毫秒"),
    (16,  "servo_miss",   "u32", 1,  "上电自检: 没应答的颗(位图, bit0=ID1), 0=12颗全在"),
    (20,  "sync_miss",    "u32", 1,  "每拍同步读缺失位图, 0=全成功"),
    (24,  "servo_err",    "u32", 1,  "最近一次单颗操作的 scs_err_t"),
    (28,  "imu_frames",   "u32", 1,  "IMU 累计有效帧数"),
    (32,  "imu_rate_fix", "u32", 1,  "补设输出频率次数(0=本来就100Hz; 每次都1→不保存)"),
    (36,  "servo_vmax",   "u32", 1,  "舵机过压阈值(0x0E, 0.1V/LSB), 期望 90=9.0V"),
    (40,  "servo_vmin",   "u32", 1,  "舵机欠压阈值(0x0F, 0.1V/LSB), 期望 40=4.0V"),
    (44,  "servo_vnow",   "u32", 1,  "舵机当前电压(0x3E, 0.1V/LSB)"),
    (48,  "servo_lasterr","u32", 1,  "舵机自报错误位"),
    (52,  "pos",          "f32", 12, "12 颗当前位置(步)"),
    (100, "gyro",         "f32", 3,  "机体系角速度 rad/s   ← obs[3:6)"),
    (112, "grav",         "f32", 3,  "重力方向(世界-Z 在机体系) ← obs[6:9)"),
    (124, "euler",        "f32", 3,  "欧拉角(度)"),
    (136, "linvel",       "f32", 3,  "里程计估计的机体线速度 m/s ← obs[0:3)"),
    (148, "jpos",         "f32", 12, "关节角(减默认站姿) rad ← obs[11:23)"),
    (196, "tau",          "f32", 12, "关节力矩 N·m ← obs[35:47)"),
    (244, "temp",         "u32", 12, "舵机温度 ℃(0x3F)"),
    (292, "status",       "u32", 12, "舵机状态位(0x41)"),
    (340, "pol_act",      "f32", 12, "本拍策略动作(tanh 后)"),
    (388, "pol_clip",     "u32", 1,  "被关节限位钳掉的颗数(0=全在限内)"),
    # ---- 分段耗时(单位: DWT 周期数, 168 = 1us) + 版本 ----
    (392, "t_imu",        "u32", 1,  "耗时: imu_poll"),
    (396, "t_sync",       "u32", 1,  "耗时: servo_sync_read(含等 12 颗应答)"),
    (400, "t_obs",        "u32", 1,  "耗时: obs_update + obs_get(里程计 FK 在这)"),
    (404, "t_pol",        "u32", 1,  "耗时: policy_run + 动作→计数换算"),
    (408, "t_up",         "u32", 1,  "耗时: report_state(含串口阻塞发送)"),
    (412, "t_send",       "u32", 1,  "耗时: servo_sync_write(不使能时≈0)"),
    (416, "fw_ver",       "u32", 1,  "固件版本标识(期望 0x20260928)"),
    (420, "t_all",        "u32", 1,  "整拍【忙碌】耗时"),
    # ---- 安全闸(2026-09-28 接线收口时加) ----
    (424, "safety_clip",  "u32", 1,  "目标钳位: 被机械行程表钳掉的颗数(0=全在表内)"),
    (428, "safety_trip",  "u32", 1,  "保护锁存: 0=正常 1=低压 2=过载 3=过热"),
]


def read_mem(addr, nbytes):
    out = subprocess.run(
        [CLI, "-c", "port=SWD", "mode=hotplug", "-r32", hex(addr), str(nbytes)],
        capture_output=True, text=True, timeout=120)
    blob = out.stdout + out.stderr
    data = {}
    for line in blob.splitlines():
        m = re.match(r"^0x([0-9A-Fa-f]{8})\s*:\s*(.*)$", line.strip())
        if not m:
            continue
        a = int(m.group(1), 16)
        words = re.findall(r"\b[0-9A-Fa-f]{8}\b", m.group(2))
        for i, w in enumerate(words):
            data[a + 4 * i] = int(w, 16)
    if not data:
        print("读内存失败, 原始输出:")
        print(blob[-1500:])
        sys.exit(1)
    return data


def get(data, off, n):
    """按偏移取 n 个 32 位字, 缺的补 0"""
    return [data.get(off + 4 * i, 0) for i in range(n)]


def f32(w):
    return struct.unpack("<f", struct.pack("<I", w & 0xFFFFFFFF))[0]


def vec(vals, dp=3):
    """vals 已经是 float 列表(main 里转好的)"""
    return "[" + ", ".join("%.*f" % (dp, v) for v in vals) + "]"


def main():
    # 只把"不是选项"的参数当位置参数; --cmd 后面的两个数值先剔掉
    argv = sys.argv[1:]
    if "--cmd" in argv:
        i = argv.index("--cmd")
        del argv[i:i + 3]
    pos = [a for a in argv if not a.startswith("--")]
    addr = int(pos[0], 0) if len(pos) > 0 else 0x10000000
    # ⚠️ 默认长度必须 = sizeof(dbg_t)(见 cat_sys.h)。读少了末尾字段会【静默显示成 0】——
    #    2026-09-27 就因为这个把"工具读少了"误判成"固件没烧成功"。当前结构体 = 432 字节。
    nbytes = int(pos[1]) if len(pos) > 1 else 432
    need = max(off + 4 * n for off, _, _, n, _ in LAYOUT)
    if nbytes < need:
        print("⚠️ 读取长度 %d < LAYOUT 需要的 %d 字节 —— 末尾字段会显示成 0(假象!)"
              % (nbytes, need))

    data = read_mem(addr, nbytes)
    base = addr

    magic = get(data, base + 0, 1)[0]
    print("=" * 62)
    if magic != DBG_MAGIC:
        print("⚠️ 魔数不对: 0x%08X (期望 0x%08X)" % (magic, DBG_MAGIC))
        print("   → 固件没跑起来 / 没写观测区 / 地址不对")
        print("=" * 62)
        return 1
    print("✅ 固件在跑 (magic = 0x%08X)" % magic)
    print("=" * 62)

    vals = {}
    for off, name, ty, n, _ in LAYOUT:
        w = get(data, base + off, n)
        vals[name] = [f32(x) for x in w] if ty == "f32" else w

    # ---- 状态 ----
    stage = vals["stage"][0]
    ms = vals["ms"][0]
    loops = vals["loops"][0]
    print("\n【状态】")
    print("  stage      : %d  %s" % (stage, STAGE.get(stage, "?")))
    print("  ms / loops : %d ms / %d 圈" % (ms, loops))
    if ms > 0 and loops > 0:
        per = ms / loops
        print("  主循环周期 : %.2f ms  (50Hz 应 ≈20ms)%s"
              % (per, "  ⚠️ 偏慢" if per > 23 else "  ✅"))

    # ---- 自检 ----
    print("\n【上电自检】")
    sm, sy = vals["servo_miss"][0], vals["sync_miss"][0]
    print("  servo_miss : 0x%03X  %s" % (sm, "✅ 12 颗全在" if sm == 0 else "⚠️ 缺: " +
          ",".join(str(i + 1) for i in range(12) if sm >> i & 1)))
    print("  sync_miss  : 0x%03X  %s" % (sy, "✅ 每拍读全成功" if sy == 0 else "⚠️"))
    print("  servo_err  : %d  %s" % (vals["servo_err"][0],
          "✅" if vals["servo_err"][0] == 0 else "⚠️"))
    print("  servo_lasterr : %d" % vals["servo_lasterr"][0])
    print("  imu_frames : %d  %s" % (vals["imu_frames"][0],
          "✅" if vals["imu_frames"][0] > 0 else "⚠️ IMU 没数据"))
    print("  imu_rate_fix : %d  (0=本来就100Hz; 每次都1→补设没保存)" % vals["imu_rate_fix"][0])
    if ms > 1000:
        # ⚠️ 两个坑都会让"频率"读错, 所以用【两次读数的差值】算(这里隔 1s 再读一次):
        #   ① g_frames 是 uint16, 跑 ~3.7 分钟(300 帧/s → 65536)就回绕 →
        #      "累计值 ÷ 总时长"过回绕点就是垃圾(曾读出 293 / 137 这种假数)
        #   ② 模块每周期自动推 3 类帧(0x04 原始 / 0x16 四元数 / 0x26 欧拉角)
        #      → 帧率 = 3 × 输出频率, 不能直接跟 100Hz 比
        FRAMES_PER_SAMPLE = 3
        time.sleep(1.0)
        d2 = read_mem(base, 32)                      # 覆盖 loops(8) / ms(12) / imu_frames(28)
        df = (d2.get(base + 28, 0) - vals["imu_frames"][0]) & 0xFFFF   # 16 位差值: 天然处理回绕
        dm = d2.get(base + 12, 0) - ms
        if dm > 0:
            fr = df * 1000.0 / dm
            out_hz = fr / FRAMES_PER_SAMPLE
            print("  IMU 频率(两次差) : %.1f 帧/s ÷ %d 帧/周期 = %.1f Hz  (期望 100)%s"
                  % (fr, FRAMES_PER_SAMPLE, out_hz,
                     "  ✅" if 80 < out_hz < 120 else "  ⚠️ 偏离"))
            print("           (imu_frames 读到 %d —— 16 位累计值会回绕, 别拿它直接换算频率)"
                  % vals["imu_frames"][0])
    print("  舵机电压: 阈值 %.1fV / 当前 %.1fV  (vmax=%d vmin=%d vnow=%d)"
          % (vals["servo_vmax"][0] / 10.0, vals["servo_vnow"][0] / 10.0,
             vals["servo_vmax"][0], vals["servo_vmin"][0], vals["servo_vnow"][0]))

    # ---- obs 数据链 ----
    print("\n【obs 数据链】")
    print("  linvel : %s   ← 站着应≈0" % vec(vals["linvel"]))
    print("  gyro   : %s   ← 静止应≈0" % vec(vals["gyro"]))
    print("  grav   : %s   ← 应≈单位向量, z≈-1" % vec(vals["grav"]))
    g = vals["grav"]
    nrm = (g[0] ** 2 + g[1] ** 2 + g[2] ** 2) ** 0.5
    print("           |grav| = %.3f  %s" % (nrm, "✅" if 0.9 < nrm < 1.1 else "⚠️ 不是单位向量"))
    print("  jpos   : %s" % vec(vals["jpos"]))
    print("  tau    : %s" % vec(vals["tau"], 4))

    # ---- 舵机反馈 ----
    print("\n【舵机反馈】")
    print("  pos    : %s" % " ".join("%.0f" % v for v in vals["pos"]))
    print("  temp   : %s" % " ".join(str(v) for v in vals["temp"]))
    print("  status : %s" % " ".join("0x%02X" % v for v in vals["status"]))

    # ---- 策略 ----
    print("\n【策略输出】")
    print("  pol_act : %s" % vec(vals["pol_act"]))
    amax = max(abs(v) for v in vals["pol_act"])
    print("           max|act| = %.4f  %s" % (amax,
          "✅" if amax > 0.001 else "⚠️ 全 0 → 策略没跑(或命令没使能)"))
    print("  pol_clip: %d  %s" % (vals["pol_clip"][0],
          "✅ 没被钳" if vals["pol_clip"][0] == 0 else "⚠️ 有被关节限位钳掉的"))

    # ---- 安全闸 ----
    print("\n【安全闸】")
    TRIP = {0: "正常", 1: "低压(<6.3V)", 2: "过载(2s 占比≥80%)", 3: "过热(≥70℃)"}
    print("  safety_clip: %d  %s" % (vals["safety_clip"][0],
          "✅ 没被钳" if vals["safety_clip"][0] == 0 else "⚠️ 有被机械行程表钳掉的"))
    st = vals["safety_trip"][0]
    print("  safety_trip: %d  %s  %s" % (st, TRIP.get(st, "?"),
          "✅" if st == 0 else "⛔ 已触发 → 每拍卸力中"))
    print("=" * 62)

    # ---- 可选: 导出"可直接喂 cmp_policy.py"的 obs 串 ----
    #  用法: python read_dbg.py --obs --cmd <vx> <wz>
    if "--obs" in sys.argv:
        cmd = [0.0, 0.0]
        if "--cmd" in sys.argv:
            i = sys.argv.index("--cmd")
            cmd = [float(sys.argv[i + 1]), float(sys.argv[i + 2])]
        print("\n⚠️ 下面是【拼】出来的 obs(非固件原样): jvel 填 0, last_act 用本拍 pol_act(差一拍)")
        print("⚠️ 只适合量级对拍, 精确对拍要给固件加完整 obs 探针")
        o = []
        o += vals["linvel"]          # [0:3)
        o += vals["gyro"]            # [3:6)
        o += vals["grav"]            # [6:9)
        o += cmd                     # [9:11)
        o += vals["jpos"]            # [11:23)
        o += [0.0] * 12              # [23:35) jvel —— 近似
        o += vals["tau"]             # [35:47)
        o += vals["pol_act"]         # [47:59) last_act —— 近似(差一拍)
        print("\n【OBS】" + ",".join("%.6f" % v for v in o))
        print("【FW 】" + ",".join("%.6f" % v for v in vals["pol_act"]))

    # ---- 固件版本(一眼确认烧的是哪版) ----
    fwv = vals["fw_ver"][0]
    print("\n【固件版本】")
    if fwv == DBG_FW_VER:
        print("  fw_ver = 0x%08X  ✅ 与当前源码一致" % fwv)
    else:
        print("  fw_ver = 0x%08X  ⚠️⚠️ 期望 0x%08X —— 烧进去的不是这份源码(或没烧成功)!"
              % (fwv, DBG_FW_VER))

    # ---- 耗时分解 ----
    print("\n【耗时分解】(1us = %d 周期; 记的是【最近一拍】)" % CYCLES_PER_US)
    tot_cyc = vals["t_all"][0]
    for key, name in (("t_imu",  "imu_poll"),
                      ("t_sync", "sync_read 12颗"),
                      ("t_obs",  "obs_update+get"),
                      ("t_pol",  "policy_run"),
                      ("t_send", "sync_write"),
                      ("t_up",   "report_state")):
        v = vals[key][0]
        pct = (100.0 * v / tot_cyc) if tot_cyc else 0.0
        print("  %-16s %9.2f us  %5.1f%%  %s"
              % (name, v / CYCLES_PER_US, pct, "#" * int(pct / 4)))
    print("  %-16s %9.2f us" % ("[忙碌合计]", tot_cyc / CYCLES_PER_US))
    if ms > 0 and loops > 0:
        per_us = ms * 1000.0 / loops
        print("  %-16s %9.2f us  ← ms/loops" % ("[整拍周期]", per_us))
        print("  %-16s %9.2f us  (空等 + 未打点的段: 第0步/云台/诊断量)"
              % ("[没计到的]", per_us - tot_cyc / CYCLES_PER_US))
    return 0


if __name__ == "__main__":
    sys.exit(main())
