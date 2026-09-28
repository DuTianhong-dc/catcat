#ifndef __CAT_IMU_H
#define __CAT_IMU_H
#include "stm32f4xx.h"

/* ==================== 接收缓冲 ====================
 * 环形缓冲大小。一帧最长 21 字节(气压计), 256 能存约 12 帧, 够主循环慢慢消费。 */
#define IMU_RX_BUF  256u

/* ==================== 帧常量 ====================
 * 帧: 7E 23 | 长度 | 功能字 | 数据… | 校验位
 * 长度 = 整帧总字节数; 校验位 = 从包头1 累加到校验位前(含包头), 取最低字节。 */
#define IMU_HEAD1        0x7E
#define IMU_HEAD2        0x23
#define IMU_FRAME_FIXED  4u     //包头1+包头2+长度+功能字; 数据区(含校验)长度 = 长度字段 - 它
#define IMU_FRAME_MAX    40u    //单帧最大字节数(官方库的接收上限, 最长帧 0x04 是 23B)

/* 下行帧最大长度: 7E 23 长度 功能字 + 参数(最多 3 个) + 校验 = 8 字节 */
#define IMU_CMD_PARAM_MAX 3u
#define IMU_CMD_MAX      (IMU_FRAME_FIXED + IMU_CMD_PARAM_MAX + 1u)

/* 功能字 (来源 通信协议.txt 第四章) */
typedef enum {
    IMU_FUNC_VERSION   = 0x01,  /* 上行: 固件版本(3 字节) */
    IMU_FUNC_RAW       = 0x04,  /* 上行: 原始数据(加速度/陀螺/磁力 各 3×int16) */
    IMU_FUNC_QUAT      = 0x16,  /* 上行: 四元数(4×float32) */
    IMU_FUNC_EULER     = 0x26,  /* 上行: 欧拉角(3×float32, **弧度**) */
    IMU_FUNC_BARO      = 0x32,  /* 上行: 气压计(4×float32, 仅十轴) */
    IMU_FUNC_SET_RATE  = 0x60,  /* 下行: 设输出频率(10~100Hz), 无返回 */
    IMU_FUNC_SET_ALGO  = 0x61,  /* 下行: 设算法类型(6=六轴 / 9=九轴), 无返回 */
    IMU_FUNC_CALIB_IMU = 0x70,  /* 下行: 校准陀螺+加速度 */
    IMU_FUNC_CALIB_MAG = 0x71,  /* 下行: 校准磁力计 */
    IMU_FUNC_CALIB_TMP = 0x73,  /* 下行: 校准温度 */
    IMU_FUNC_REQUEST   = 0x80,  /* 下行: 请求数据(参数1 = 要请求的功能字) */
    IMU_FUNC_STATE     = 0x81,  /* 上行: 返回状态(功能字 + 状态 0失败/1成功) */
    IMU_FUNC_RESET     = 0xA0   /* 下行: 重置用户数据(清所有校准值), 慎用 */
} imu_func_t;

/* ==================== 数值换算 ====================
 * 原始数据(0x04)是 int16, 要乘比例; 欧拉角/四元数本身就是 float32, 不用乘。
 * (比例来源: 官方驱动 + 官方 Python 库, 两处一致) */
#define IMU_ACCEL_SCALE  (16.0f / 32767.0f)                        //→ g      (量程 ±16g)
#define IMU_GYRO_SCALE   ((2000.0f / 32767.0f) * 0.01745329252f)   //→ rad/s  (量程 ±2000°/s)
#define IMU_MAG_SCALE    (800.0f / 32767.0f)                       //→ uT
#define IMU_RAD2DEG      57.2957795f

/* ==================== 错误类型 ==================== */
typedef enum {
    IMU_OK        = 0,      /* 成功 */
    IMU_ERR_PARAM = 1       /* 请求参数不合法(超长), 帧没发出去 */
} imu_err_t;

/* ==================== 函数原型 ==================== */
/* ① 初始化 */
void imu_init(void);                    //初始化 USART2 与 PA2/PA3(AF7), 开接收中断, 并设输出频率 100Hz

/* ② 收发 */
void imu_data_recv(uint8_t byte);       //中断入口用: 把 1 个字节推进环形缓冲
imu_err_t imu_send_cmd(uint8_t func, const uint8_t *params, uint8_t param_len);   //发一条下行指令

/* ③ 解析 */
void imu_poll(void);                 //主循环里调: 把环形缓冲排空, 逐帧解析(保留最新)

/* ③.5 频率确认 (等模块醒了自己出帧之后再调, 见 .c 里的说明)
 * 返回 1 = 发现频率不对, 补设了一次; 0 = 本来就是对的不必设。
 * (返回值有诊断价值: 每次都返回 1 → 说明输出频率【不】掉电保存, 每次上电都得设) */
uint8_t imu_ensure_rate(void);

/* ④ 取数 (读的是"最后一个完整帧", 每次收到新帧就整体刷新)
 * ⚠️ 还没收到任何帧时, 这些函数返回的是全 0 —— 其中 gravity 会算成 [0,0,-1],
 *    "看起来像正立"但其实是无数据。用 imu_get_frames() 判断有没有数据再采信。 */
void     imu_get_gyro(float out[3]);      //机体系角速度 rad/s —— 直接进观测 obs[3:6)
void     imu_get_quat(float out[4]);      //四元数 w x y z
void     imu_get_grav(float out[3]);   //重力方向: 世界 -Z 在机体系 —— 观测 obs[6:9)
void     imu_get_euler(float out[3]);     //欧拉角 roll/pitch/yaw, **弧度**(模块原样)
void     imu_get_euler_deg(float out[3]); //同上, 换成**角度**(调试/日志看得直观)
uint16_t imu_get_frames(void);                //累计有效帧数; 两次调用间变了 = 数据在来

#endif
