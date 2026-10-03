# firmware — STM32 固件

catcat 的主控固件（cat3.4 机型；CAT_APP2.0，C99 / Keil AC5）。已迁入（2026-09-28），**2026-10-03 大更新**。

## 内容

- 模块（自下而上分层，见各文件头的分层表）：
  `cat_sys`（时钟/时基/观测区）· `cat_servo`（STS3215 总线舵机）· `cat_imu` · `cat_esp32`（BLE 桥链路）
  · `cat_mg90`（云台）· `cat_obs`（59 维观测组装）· `cat_vel_est`（腿里程计线速度估计）
  · `cat_pose`（姿态/限位系统常量 + pose_hash）· `cat_crc32`
  · `cat_policy`（int8 前向，单独 -O3）· `cat_polupd`（策略在线更新：双槽 Flash + CRC32，永不砖）
  · `cat_safety`（目标钳位 + 电压/电流/温度保护 + IWDG）· `cat_action`（动作加载 + 播放器，RAM 关键帧表）
- `main.c` — 主循环编排（50Hz 本地闭环；**PPO 在 STM32 跑**，ESP32 只做透传桥、上位机不进控制环）
- `policy_weights.c` — 策略权重参考（由 `tools/npz_to_c.py` 生成，手改无效）。
  注：当前固件默认 `CAT_POL_FACTORY=0`（不编译它，策略走双槽 OTA 下发），文件留档参考。
- 说明文档：[`固件说明.md`](固件说明.md)（总览 / 各模块设计与实测依据 / 安全模型 / 调试工具，
  含"事实与存疑总表"——标注了每项结论的可靠性）；对外协议契约见
  [`通信协议.txt`](通信协议.txt)（第七章 策略 OTA · 第八章 动作加载）
- 工具：`check_syntax.sh`（语法快检）· `run_*.sh`（PC 仿真，不需要硬件）·
  `tools/`（调试 / 权重转换 / OTA 链路：`ble_cmd.py` 遥控 · `ota_release.py`+`ble_ota.py` 策略下发 ·
  `read_dbg.py`/`probe_pc.py` 真机观测）

## 快速上手

1. **Keil AC5** 新建工程：芯片 STM32F407ZGTx；加入全部 `.c`（`policy_weights.c` 视 `CAT_POL_FACTORY` 而定、
   `test_*.c` 是测试件不进工程）+ `startup_stm32f407xx.s`；**`cat_policy.c` 单独挂 -O3**（其余 -O0）。
2. 编译烧录（ST-Link）。烧录/调试连接后**按一次复位** —— DWT 周期计数器会被调试工具清掉，
   不复位的话耗时探针全 0（详见 `系统层说明.md`）。
3. 整机闭环：`tools/ble_cmd.py` 发 10B 控制帧（使能 + 速度指令）。
4. 更新策略：按住 KEY0（PE4）上电进更新模式 → `tools/ota_release.py` 把 .npz 打包成 .bin →
   `tools/ble_ota.py` 上传；END 成功后自动软复位、新策略生效（协议第七章）。
5. 动作加载（不需要 KEY0）：PC 侧准备关键帧 CSV → 按协议第八章的 BEGIN/FRAME×N/END 节奏逐帧下发。

## 安全模型（一句话）

「停止」≠「卸力」：失联/急停 = 保持出力站住；唯一卸力 = 安全故障（过压/过流/过热，去抖后锁存）
或策略更新中 → 每拍广播 `0x28=0`。IWDG 1s 看门狗兜底卡死。详见 `安全说明.md`。
