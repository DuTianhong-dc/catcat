/* ============================================================
 *  cat_imu.c — 亚博 YbImu 姿态传感器驱动 (USART2 / PA2-PA3)
 *
 *  分层(本文件按此顺序排列, 新函数插到对应区块, 别追加到末尾):
 *    ① 初始化   imu_init()
 *    ② 收发     imu_data_recv() / imu_send_cmd()   [+ 环形接收缓冲]
 *    ③ 解析     imu_parse_frame()(内部) / imu_poll()
 *    ④ 取数     imu_get_gyro() / imu_get_quat() / imu_get_grav() / imu_get_euler[_deg]() / imu_get_frames()
 *
 *  命名: 发送侧 tx_ 前缀, 接收侧 rx_ 前缀。
 *  模块是【主动上报】的: 上电后自己按频率推数据, 我们只收不发 → 详见 IMU驱动说明.md(一)
 * ============================================================ */
#include "cat_imu.h"
#include "cat_config.h"
#include "cat_sys.h"        /* imu_ensure_rate 要用时基 */
#include <string.h>         /* memcpy: 4 字节按位拼成 float, 避开对齐/别名问题 */

/* ---------- 环形接收缓冲 ----------
 * 中断只"入队", 解析放主循环 → 中断停留最短。
 * 满时丢【最旧】(丢新会导致"永远解析不出新帧") → 详见 IMU驱动说明.md(二)
 * ⚠️ 已知竞态(2026-10-03 审查 T2, 同 cat_esp32): rx_r 主循环与中断都写(满时丢最旧),
 *    极端时序丢/重 1 字节 → 单帧校验不过被丢, 靠下一帧(100Hz, 每拍 2 帧)自然恢复。 */
static volatile uint8_t  rx_buf[IMU_RX_BUF];
static volatile uint16_t rx_w;      //写入下标
static volatile uint16_t rx_r;      //读取下标

/* ---------- 解析结果缓存 ----------
 * 每解出一帧就覆盖一份 → 主循环读到的永远是最新那帧。欧拉角保持模块原样的弧度。 */
static volatile float    g_gyro[3];     //0x04 帧的陀螺部分
static volatile float    g_quat[4];     //0x16 帧
static volatile float    g_euler[3];    //0x26 帧 (roll/pitch/yaw, 弧度)
static volatile uint16_t g_frames;      //有效帧计数(在涨 = 数据在来)

/* ==================== ① 初始化 ==================== */
void imu_init(void){
    //1、开时钟: GPIOA(AHB1) + USART2(APB1 —— 跟舵机的 APB2 不是同一条)
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
    RCC->APB1ENR |= RCC_APB1ENR_USART2EN;

    //2、PA2(TX)/PA3(RX) 配 USART2 复用(AF7): 复用 + 推挽 + 高速 + 上拉(RX 上拉防断线时浮空乱收)
    GPIOA->MODER = (GPIOA->MODER & ~((3u << 4) | (3u << 6))) | (2u << 4) | (2u << 6);
    GPIOA->OTYPER  &= ~((1u << 2) | (1u << 3));
    GPIOA->OSPEEDR |=  (3u << 4) | (3u << 6);
    GPIOA->PUPDR = (GPIOA->PUPDR & ~((3u << 4) | (3u << 6))) | (1u << 4) | (1u << 6);
    GPIOA->AFR[0] = (GPIOA->AFR[0] & ~((0xFu << 8) | (0xFu << 12))) | (7u << 8) | (7u << 12);

    //3、USART2: 115200 8N1。BRR = 42MHz/115200 = 364.6 → 365(误差 0.114%)
    USART2->BRR = (APB1_MHZ * 1000000u + 57600u) / 115200u;
    USART2->CR2 = 0;
    USART2->CR3 = 0;
    USART2->CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE | USART_CR1_RXNEIE;

    //4、开接收中断(ISR 在 stm32f4xx_it.c, 只做"读 DR + 调 imu_data_recv")。优先级 6(低)
    NVIC_SetPriority(USART2_IRQn, 6u);
    NVIC_EnableIRQ(USART2_IRQn);

    //5、设输出频率 100Hz(出厂 25Hz 跟不上 50Hz 主循环)。这条可能丢, 靠 imu_ensure_rate() 兜底
    {   uint8_t p[2] = {100u, 0x5F};
        imu_send_cmd(IMU_FUNC_SET_RATE, p, 2);
    }
}

/* ==================== ② 收发 ==================== */
/* 中断入口用: 把收到的 1 个字节推进环形缓冲, 只做这一件事就返回。
 * (官方实现在这里死等 RXNE、且攒够 127 字节才推 → IMU驱动说明.md(八) */
void imu_data_recv(uint8_t byte){
    uint16_t next = (uint16_t)((rx_w + 1u) % IMU_RX_BUF);
    if (next == rx_r) rx_r = (uint16_t)((rx_r + 1u) % IMU_RX_BUF);   //满了丢最旧
    rx_buf[rx_w] = byte;
    rx_w = next;
}

/* 发一条下行指令: 7E 23 | 长度 | 功能字 | 参数… | 校验
 * 长度 = 整帧字节数; 校验 = 累加到校验位前取低字节(含包头)。最多 3 个参数。 */
imu_err_t imu_send_cmd(uint8_t func, const uint8_t *params, uint8_t param_len){
    uint8_t f[IMU_CMD_MAX];
    uint8_t n = 0, i;
    uint8_t ck = 0;

    if (param_len > IMU_CMD_PARAM_MAX || (param_len > 0u && params == 0)) return IMU_ERR_PARAM;

    f[n++] = IMU_HEAD1;
    f[n++] = IMU_HEAD2;
    f[n++] = 0u;                    //长度占位, 最后填
    f[n++] = func;
    for (i = 0; i < param_len; i++) f[n++] = params[i];

    f[2] = (uint8_t)(n + 1u);       //长度 = 整帧字节数(含校验位)
    for (i = 0; i < n; i++) ck = (uint8_t)(ck + f[i]);
    f[n++] = ck;

    for (i = 0; i < n; i++){
        while (!(USART2->SR & USART_SR_TXE));
        USART2->DR = f[i];
    }
    while (!(USART2->SR & USART_SR_TC));
    return IMU_OK;
}

/* 确认输出频率是不是 100Hz: 实测帧率, 不够就补发一条。返回 1 = 补设了, 0 = 本来就对。
 * 必须放在"模块确实在出帧"之后调(如 boot_wait_ready 之后) → 详见 IMU驱动说明.md(五) */
uint8_t imu_ensure_rate(void){
    const uint8_t  want_hz  = 100u;                                          /* 要的频率 */
    const uint32_t win_ms   = 200u;                                          /* 观察窗 */
    const uint32_t min_fr   = (uint32_t)want_hz * 3u * win_ms / 1000u / 2u;  /* 窗内帧数下限, 打对折容错 */

    uint32_t f0 = imu_get_frames();
    uint32_t t0 = sys_ms();
    uint32_t df;

    while (!sys_elapsed(t0, win_ms));           //数这个窗口里进了多少帧
    df = (uint32_t)(imu_get_frames() - f0);

    if (df < min_fr) {                          //明显不够快 → 没设上, 补一条
        uint8_t p[2];
        p[0] = want_hz;
        p[1] = 0x5F;                            //该指令的固定第二参数
        imu_send_cmd(IMU_FUNC_SET_RATE, p, 2);
        return 1u;
    }
    return 0u;
}

/* ==================== ③ 解析 ==================== */
/* 把一帧的数据区(含校验)按功能字解析, 结果覆盖缓存。
 * 数据区长度 dlen = 长度字段 - 4, 末字节是校验位(调用方已验过)。 */
static void imu_parse_frame(uint8_t func, const uint8_t *d, uint8_t dlen){
    switch (func) {
    case IMU_FUNC_RAW:                                   //0x04: 加速度3 + 陀螺3 + 磁力3, 各 int16
        if (dlen >= 12u) {
            for (uint8_t k = 0; k < 3u; k++){            //只用陀螺部分(偏移 6) = 观测 obs[3:6)
                int16_t raw = (int16_t)(d[6u + 2u*k] | ((uint16_t)d[7u + 2u*k] << 8));
                g_gyro[k] = (float)raw * IMU_GYRO_SCALE;
            }
            g_frames++;
        }
        break;

    case IMU_FUNC_QUAT:                                  //0x16: q0~q3, 各 float32(小端)
        if (dlen >= 16u) {
            for (uint8_t k = 0; k < 4u; k++){
                uint32_t raw = (uint32_t)d[4u*k] | ((uint32_t)d[4u*k+1u] << 8) |
                               ((uint32_t)d[4u*k+2u] << 16) | ((uint32_t)d[4u*k+3u] << 24);
                float v; memcpy(&v, &raw, sizeof(float));   //按位拷, 避免对齐/别名问题
                g_quat[k] = v;
            }
            g_frames++;
        }
        break;

    case IMU_FUNC_EULER:                                 //0x26: roll/pitch/yaw, 各 float32, 单位弧度
        if (dlen >= 12u) {
            for (uint8_t k = 0; k < 3u; k++){
                uint32_t raw = (uint32_t)d[4u*k] | ((uint32_t)d[4u*k+1u] << 8) |
                               ((uint32_t)d[4u*k+2u] << 16) | ((uint32_t)d[4u*k+3u] << 24);
                float v; memcpy(&v, &raw, sizeof(float));
                g_euler[k] = v;
            }
            g_frames++;
        }
        break;

    default: break;                                      //气压/版本/校准回执: 本层不用
    }
}

/* 主循环里调: 把环形缓冲【排空】(不能只吃一帧, 否则越积越多), 逐字节跑状态机,
 * 收齐一帧就校验+解析。同一拍同类数据后帧覆盖前帧 = 天然保留最新 → 详见 IMU驱动说明.md(三) */
void imu_poll(void){
    static uint8_t  buf[IMU_FRAME_MAX];     //攒一帧
    static uint16_t idx;                    //buf 里已收字节数(跨调用保持的状态)
    static uint8_t  flen, ffunc;
    uint8_t byte;

    while (rx_r != rx_w) {                  //排空
        byte = rx_buf[rx_r];
        rx_r = (uint16_t)((rx_r + 1u) % IMU_RX_BUF);

        switch (idx) {
        case 0u:                            //① 等包头1
            if (byte == IMU_HEAD1) {
                buf[idx] = byte;
                idx++;
            }
            break;

        case 1u:                            //② 等包头2
            /* 来的还是 0x7E 就保持 idx=1(把它当新包头1), 不直接复位 ——
             * 复位会把 "7E(噪声) 7E 23(真帧头)" 里的真包头吃掉 → 详见 IMU驱动说明.md(四) */
            if (byte == IMU_HEAD2) {
                buf[idx] = byte;
                idx++;
            } else if (byte != IMU_HEAD1) {
                idx = 0u;                   //既不是 7E 也不是 23 → 重新找包头
            }
            break;

        case 2u:                            //③ 长度字段
            flen = byte;
            buf[idx] = byte;
            idx++;
            if (flen < 5u || flen > IMU_FRAME_MAX) idx = 0u;   //最短帧 7E 23 LEN FUNC CK = 5 字节
            break;

        case 3u:                            //④ 功能字
            ffunc = byte;
            buf[idx] = byte;
            idx++;
            break;

        default:                            //⑤ 数据区 + 校验, 逐字节攒
            buf[idx] = byte;
            idx++;
            if (idx >= flen) {              //一帧收齐
                uint8_t ck = (uint8_t)(IMU_HEAD1 + IMU_HEAD2 + flen + ffunc);
                for (uint8_t k = 4u; k < (uint8_t)(flen - 1u); k++) ck = (uint8_t)(ck + buf[k]);
                if (ck == buf[flen - 1u]) { //校验过了才认
                    imu_parse_frame(ffunc, &buf[4], (uint8_t)(flen - IMU_FRAME_FIXED));
                }
                idx = 0u;
            }
            break;
        }
    }
}

/* ==================== ④ 取数 ==================== */
/* 机体系角速度 rad/s —— 观测 obs[3:6) 直接用, 别转成角度 */
void imu_get_gyro(float out[3]){
    for (uint8_t k = 0; k < 3u; k++) out[k] = g_gyro[k];
}

/* 四元数 w x y z (模块原样) */
void imu_get_quat(float out[4]){
    for (uint8_t k = 0; k < 4u; k++) out[k] = g_quat[k];
}

/* 重力方向: "世界 -Z"在机体坐标系下的单位向量 —— 观测 obs[6:9)。
 * 静止正立 ≈ [0,0,-1]; 与训练的 get_gravity() 同义。公式与验证 → IMU驱动说明.md(六) */
void imu_get_grav(float out[3]){
    float w = g_quat[0], x = g_quat[1], y = g_quat[2], z = g_quat[3];
    out[0] =   2.0f * (w * y - x * z);
    out[1] =  -2.0f * (y * z + w * x);
    out[2] = -(1.0f - 2.0f * (x * x + y * y));
}

/* 欧拉角 roll/pitch/yaw, 弧度(模块原样)。策略观测里没有欧拉角, 这个给调试/日志用 */
void imu_get_euler(float out[3]){
    for (uint8_t k = 0; k < 3u; k++) out[k] = g_euler[k];
}

/* 同一份数据的角度版, 打印出来好读 */
void imu_get_euler_deg(float out[3]){
    for (uint8_t k = 0; k < 3u; k++) out[k] = g_euler[k] * IMU_RAD2DEG;
}

/* 累计有效帧数: 两次读到的不一样 = 这段时间收到过数据。
 * ⚠️ 判"数据在不在来"必须先看它 —— 没收到帧时 gravity 会算出 [0,0,-1] 假装正立 → IMU驱动说明.md(七) */
uint16_t imu_get_frames(void){
    return g_frames;
}
