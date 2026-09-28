# firmware — STM32 固件

catcat 的主控固件(cat3.3 机型;CAT_APP2.0,C99 / Keil AC5)。已迁入(2026-09-28)。

## 内容

- 模块:`cat_sys`(时钟 / 时基 / 观测区)、`cat_servo`(STS3215 总线舵机)、`cat_imu`、`cat_esp32`(BLE 桥链路)、
  `cat_mg90`(云台)、`cat_obs`、`cat_policy`(int8 前向)、`cat_safety`、`cat_vel_est`
- `main.c` — 主循环编排(50Hz 本地闭环)
- `policy_weights.c` — 策略权重(由 `tools/npz_to_c.py` 生成,手改无效)
- 说明文档:`舵机驱动说明.md` / `IMU驱动说明.md` / `ESP32通信说明.md` / `策略前向说明.md` /
  `观测组装说明.md` / `系统层说明.md`
- 协议 → [`通信协议.txt`](通信协议.txt)
- 工具:`check_syntax.sh`(语法快检)、`run_*.sh`(PC 仿真,不需要硬件)、`tools/`(调试 / 权重转换脚本)
