/* ============================================================
 *  test_ble_fw.c — 用【第一版真实固件 cat_ble.c】在 PC 上跑时序仿真
 *
 *  跟 test_ringbuf_fw.c 是同一套方法(见该文件头的说明), 区别只在被测对象:
 *    第二版 cat_esp32.c : 10 字节帧, 满时【丢最旧】
 *    第一版 cat_ble.c   : 12 字节帧, 满时【丢新】(if (next != s_tail) 才存)
 *
 *  真代码路径:
 *    USART3->DR 放字节 → 调 USART3_IRQHandler() → cat_ble_on_rx()   (真)
 *                      → 真环形缓冲 / 满时丢新
 *    t % LOOP_TICKS==0 → cat_ble_poll()                            (真)
 *                      → 真状态机 + 真校验和 → g_ble
 *
 *  第一版自带的诊断计数器(可直接读, 第二版没有):
 *    s_rx_isr   中断收到的字节总数
 *    s_aa_frame 成功解出的完整 AA 帧数
 *
 *  ⚠️ 桩掉了什么(这些不是被测对象, 但真机上是有逻辑的):
 *    cat_cfg_get(在线参数) / cat_gimbal_set(云台) / cat_fsm_get_state(状态机)
 *    cat_ble_load_parse_byte + cat_ble_poll_frame(0xCC 二进制加载协议, 整块不用)
 * ============================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "stm32f4xx_hal.h"      /* 桩: 寄存器 + HAL_GetTick 声明 */
#include "cat_ble.h"            /* 真: 第一版 */
#include "cat_ble_proto.h"      /* 桩: 只留类型与函数声明 */
#include "cat_gimbal.h"         /* 桩 */
#include "cat_fsm.h"            /* 桩 */

/* ==================== 桩: 外设寄存器落到真内存 ==================== */
USART_TypeDef usart3_mem;

/* 第一版 cat_ble.c 里的诊断计数器(故意非 static, 供 SWD 直读; 头文件里没声明) */
extern volatile uint32_t s_rx_isr;      /* 中断收到的字节总数 */
extern volatile uint32_t s_aa_frame;    /* 成功解出的完整 AA 帧数 */

/* ==================== 桩: 时基 ==================== */
static volatile uint32_t g_ms;
uint32_t HAL_GetTick(void){ return g_ms; }

/* ==================== 桩: 在线参数区 ==================== */
float g_cat_cfg[CAT_CFG_ID_MAX];
float cat_cfg_get(cat_cfg_id_t id){
    return (id > 0 && id < CAT_CFG_ID_MAX) ? g_cat_cfg[id] : 0.0f;
}

/* ==================== 桩: 其它模块 ==================== */
void cat_gimbal_set(int8_t pan, int8_t tilt){ (void)pan; (void)tilt; }
cat_fsm_state_t cat_fsm_get_state(void){ return FS_IDLE; }
/* 注: cat_ble_load_parse_byte / cat_ble_poll_frame 在真头文件里是 static(实现写在头里),
 *     所以桩头 cat_ble_proto.h 给了 static 空实现, 这里不能再定义同名外部函数。 */

/* ==================== 中断入口 ====================
 * 第一版的 USART3 ISR 就是转发给 cat_ble_on_rx() (见 cat_ble.h 的注释) */
void USART3_IRQHandler(void){
    cat_ble_on_rx();
}

/* ==================== 模拟参数 ==================== */
#define TICK_US        100u          /* 1 tick = 100μs */
#define LOOP_TICKS     200u          /* 主循环 20ms */
#define BYTE_TICKS     1u            /* 115200 下 1 字节 ≈ 870μs */
#define T_TICKS        600000u       /* 60 秒 */

/* ==================== 组一帧(第一版 12 字节) ====================
 * AA 55 | vx[2] | vy[2] | wz[2] | flag | cksum | pan | tilt
 *   vx/vy/wz 全 0(不用它们编码, 因为会被限幅+死区抹平)
 *   pan/tilt 拿来编码递增 seq —— 它们不进校验、不限幅, 直接赋值
 *   ⚠️ 避开 0xCC: 第一版 cat_ble_on_rx 会把【任意位置】的 0xCC 当加载帧吞掉 */
static uint8_t tx_frame[BLE_FRAME_LEN];

static void build_frame(uint16_t seq){
    tx_frame[0] = BLE_HEADER0;
    tx_frame[1] = BLE_HEADER1;
    tx_frame[2] = 0u;  tx_frame[3] = 0u;      /* vx = 0 */
    tx_frame[4] = 0u;  tx_frame[5] = 0u;      /* vy = 0 */
    tx_frame[6] = 0u;  tx_frame[7] = 0u;      /* wz = 0 */
    tx_frame[8] = 0x01u;                      /* flag: 使能 */
    {   uint8_t sum = 0u;
        for (int i = 2; i <= 8; i++) sum = (uint8_t)(sum + tx_frame[i]);
        tx_frame[9] = sum;                    /* cksum: 字节 2..8 累加 */
    }
    tx_frame[10] = (uint8_t)(seq & 0xFFu);            /* pan  = seq 低字节 */
    tx_frame[11] = (uint8_t)((seq >> 8) & 0xFFu);     /* tilt = seq 高字节 */
}

static uint16_t pan_tilt_to_seq(int8_t pan, int8_t tilt){
    return (uint16_t)((uint8_t)pan | ((uint16_t)(uint8_t)tilt << 8));
}

/* ==================== 主函数 ==================== */
int main(int argc, char **argv){
    double freq_hz   = (argc > 1) ? atof(argv[1]) : 10.0;
    double jitter_ms = (argc > 2) ? atof(argv[2]) : 0.0;

    const double period_ticks = 1000000.0 / freq_hz / (double)TICK_US;
    const double jitter_ticks = jitter_ms * 1000.0 / (double)TICK_US;

    /* 在线参数按第一版默认值填(对齐训练) */
    g_cat_cfg[CAT_CFG_FALL_UPZ]    = 0.4f;
    g_cat_cfg[CAT_CFG_CMD_VX_MAX]  = 0.5f;
    g_cat_cfg[CAT_CFG_CMD_VY_MAX]  = 0.2f;
    g_cat_cfg[CAT_CFG_CMD_WZ_MAX]  = 0.5f;
    g_cat_cfg[CAT_CFG_BLE_DEADMAN] = 500.0f;

    USART3->SR = USART_SR_TXE | USART_SR_TC;

    srand(12345u);

    uint32_t t;
    double   next_send_t    = 0.0;
    uint32_t next_byte_tick = 0u;
    uint8_t  cur_frame[BLE_FRAME_LEN];
    int      cur_idx        = -1;
    uint16_t seq            = 0u;
    uint16_t last_done_seq  = 0u;
    uint32_t frame_k        = 0u;
    uint32_t n_sent         = 0u;
    uint32_t n_parsed       = 0u;      /* 主循环拿到"新鲜指令"的拍数 */
    uint32_t n_polls        = 0u;
    uint32_t n_check        = 0u;
    uint32_t n_exact        = 0u;
    uint16_t last_got       = 0xFFFFu;

    for (t = 0u; t < T_TICKS; t++) {
        g_ms = (uint32_t)((uint64_t)t * TICK_US / 1000u);

        /* ---- 1、上游: 该起一帧了吗 ---- */
        if ((double)t >= next_send_t) {
            seq++;
            while ((seq & 0xFFu) == 0xCCu || ((seq >> 8) & 0xFFu) == 0xCCu) seq++;
            build_frame(seq);
            memcpy(cur_frame, tx_frame, BLE_FRAME_LEN);
            cur_idx        = 0;
            next_byte_tick = t;
            n_sent++;

            frame_k++;
            double jit = 0.0;
            if (jitter_ticks > 0.0)
                jit = ((double)rand() / (double)RAND_MAX * 2.0 - 1.0) * jitter_ticks;
            next_send_t = (double)frame_k * period_ticks + jit;
            if (next_send_t <= (double)t) next_send_t = (double)t + 1.0;
        }

        /* ---- 2、串口: 送一个字节 → 触发中断 ---- */
        if (cur_idx >= 0 && t >= next_byte_tick) {
            USART3->DR = cur_frame[cur_idx];
            USART3->SR |= USART_SR_RXNE;
            USART3_IRQHandler();                 /* ← 真实中断入口 */
            cur_idx++;
            if (cur_idx >= (int)BLE_FRAME_LEN) { cur_idx = -1; last_done_seq = seq; }
            next_byte_tick = t + BYTE_TICKS;
        }

        /* ---- 3、下游: 主循环排空 ---- */
        if ((t % LOOP_TICKS) == 0u) {
            uint16_t expect = last_done_seq;

            n_polls++;
            cat_ble_poll();                      /* ← 真实的消费函数 */

            {   uint16_t got = pan_tilt_to_seq(g_ble.pan, g_ble.tilt);
                if (got != last_got) { last_got = got; n_parsed++; }
                if (expect != 0u) {
                    n_check++;
                    if (got == expect) n_exact++;
                }
            }
        }
    }

    printf("%6.0f  %8.1f  %7.2f  %9.1f%%  %9.1f%%  %10.1f%%\n",
           freq_hz, jitter_ms,
           n_polls ? (double)n_sent / (double)n_polls : 0.0,
           n_polls ? (double)n_parsed / (double)n_polls * 100.0 : 0.0,
           n_check ? (double)n_exact / (double)n_check * 100.0 : 0.0,
           n_sent  ? (double)s_aa_frame / (double)n_sent * 100.0 : 0.0);
    return 0;
}
