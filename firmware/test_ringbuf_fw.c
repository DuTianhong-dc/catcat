/* ============================================================
 *  test_ringbuf_fw.c — 用【真实的 cat_esp32.c】在 PC 上跑环形缓冲实验
 *
 *  跟 ringbuf_sim.py 的根本区别:
 *    那份是抽象的"事件序列"模型(if t%period==0 就当事件发生), 里面没有中断、没有 ISR、
 *    没有一行固件代码 —— 它算得出"丢多少字节", 但测不出任何跟中断时机有关的东西。
 *    这份走的是**固件真代码**:
 *
 *      USART3->DR 放字节  →  调 USART3_IRQHandler()      (真实中断入口, 同 cat_it.c:38)
 *                          →  esp32_data_recv()          (cat_esp32.c 的真函数)
 *                          →  真实环形缓冲 / 满时丢最旧
 *
 *      t % LOOP_TICKS==0  →  esp32_poll()               (cat_esp32.c 的真函数)
 *                          →  esp32_parse_byte() → cmd  (真状态机 + 真校验和)
 *
 *  编译: 见 run_ringbuf_fw.sh
 *  用法: ./test_ringbuf_fw.exe <上位机频率Hz> <抖动ms>
 *
 *  ⚠️ 能模拟的 / 不能模拟的(诚实边界):
 *    能: 字节到达时序、真实缓冲与解析逻辑、满时丢最旧、帧内字节跨主循环周期
 *    不能: 中断的**精确抢占时机**(这里是按时间顺序"插入"ISR, 不是真抢占)、
 *          μs 级中断响应延迟、USART 单字节接收缓冲的 ORE 溢出、主循环自身耗时
 * ============================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stm32f4xx.h"
#include "cat_esp32.h"
#include "cat_sys.h"

/* 桩寄存器落到真内存(见桩 stm32f4xx.h) */
USART_TypeDef usart3_mem;

/* ==================== cat_sys 的桩: 时基 ==================== */
static volatile uint32_t g_ms;
uint32_t sys_ms(void){ return g_ms; }
uint8_t  sys_elapsed(uint32_t from_ms, uint32_t span_ms){
    return (uint8_t)((uint32_t)(g_ms - from_ms) >= span_ms);
}

/* ==================== 中断入口 ==================== */
/* 与 cat_it.c:38 一字不差 —— 只做"读 DR 推进缓冲"这一件事 */
void USART3_IRQHandler(void){
    esp32_data_recv((uint8_t)(USART3->DR & 0xFF));
}

/* ==================== 模拟参数 ==================== */
#define TICK_US        100u          /* 1 个仿真 tick = 100μs */
#define LOOP_TICKS     200u          /* 主循环周期 20ms = 200 tick */
#define BYTE_TICKS     1u            /* 115200 下 1 字节 ≈ 870μs, 折合 ≈ 0.9 tick */
#define T_TICKS        600000u       /* 总时长 60 秒 */
#define SEQ_WRAP       30000u        /* seq 上限, 别让 int16 溢出 */

/* ==================== 组一帧下行包 ====================
 * AA 55 | vx[2] | wz[2] | flags | pan | tilt | ck
 * vx 拿 seq 填(递增), 用来认出"这一帧是不是被正确解析了" */
static uint8_t tx_frame[ESP32_DL_LEN];

static void build_frame(uint16_t seq){
    tx_frame[0] = ESP32_DL_HEAD0;
    tx_frame[1] = ESP32_DL_HEAD1;
    tx_frame[2] = (uint8_t)(seq & 0xFFu);
    tx_frame[3] = (uint8_t)((seq >> 8) & 0xFFu);
    tx_frame[4] = 0u;                       /* wz */
    tx_frame[5] = 0u;
    tx_frame[6] = 0x01u;                    /* flags: 使能 */
    tx_frame[7] = 0u;                       /* pan */
    tx_frame[8] = 0u;                       /* tilt */
    {   uint8_t ck = 0;
        for (uint8_t i = ESP32_DL_CK_FIRST; i < ESP32_DL_CK_OFF; i++)
            ck = (uint8_t)(ck + tx_frame[i]);
        tx_frame[ESP32_DL_CK_OFF] = ck;
    }
}

/* ==================== 主函数 ==================== */
int main(int argc, char **argv){
    double freq_hz   = (argc > 1) ? atof(argv[1]) : 10.0;
    double jitter_ms = (argc > 2) ? atof(argv[2]) : 0.0;

    /* 发送周期(tick): 1/freq 秒 → tick */
    const double period_ticks = 1000000.0 / freq_hz / (double)TICK_US;
    const double jitter_ticks = jitter_ms * 1000.0 / (double)TICK_US;

    USART3->SR = USART_SR_TXE | USART_SR_TC;    /* 假装发送空闲, 免得 send 卡住 */

    srand(12345u);                              /* 固定种子: 结果可复现 */

    uint32_t t;
    double   next_send_t    = 0.0;
    uint32_t next_byte_tick = 0u;
    uint8_t  cur_frame[ESP32_DL_LEN];
    int      cur_idx        = -1;               /* -1 = 当前没有正在发的帧 */
    uint16_t seq            = 0u;
    uint16_t last_done_seq  = 0u;               /* 最近一帧"10 个字节全发完"时的 seq */
    uint32_t frame_k        = 0u;               /* 发出的第几帧(算绝对发送时刻用) */
    uint32_t n_sent         = 0u;               /* 上位机发了多少帧 */
    uint32_t n_parsed       = 0u;               /* 主循环拿到"新鲜指令"的拍数 */
    uint32_t n_polls        = 0u;               /* 主循环一共排空了几次 */
    uint32_t n_check        = 0u;               /* 能判"末帧"的拍数 */
    uint32_t n_exact        = 0u;               /* 其中: 末帧被原样解出来的拍数 */
    int32_t  last_vx        = 0;                /* 上次看到的 vx(原始定点值) */

    for (t = 0u; t < T_TICKS; t++) {
        g_ms = (uint32_t)((uint64_t)t * TICK_US / 1000u);   /* 100μs tick → ms */

        /* ---- 1、上游: 该起一帧了吗(周期 + 抖动) ---- */
        if ((double)t >= next_send_t) {
            seq = (uint16_t)(seq % SEQ_WRAP + 1u);
            build_frame(seq);
            memcpy(cur_frame, tx_frame, ESP32_DL_LEN);
            cur_idx        = 0;
            next_byte_tick = t;
            n_sent++;

            /* 抖动加在"绝对发送时刻"上, 而不是累积进间隔 ——
             * 否则 clamp 会改变平均发送率, 就分不清"抖动的影响"和"发少了的影響" */
            frame_k++;
            double jit = 0.0;
            if (jitter_ticks > 0.0)
                jit = ((double)rand() / (double)RAND_MAX * 2.0 - 1.0) * jitter_ticks;
            next_send_t = (double)frame_k * period_ticks + jit;
            if (next_send_t <= (double)t) next_send_t = (double)t + 1.0;
        }

        /* ---- 2、串口: 该送一个字节了吗 → 触发中断 ---- */
        if (cur_idx >= 0 && t >= next_byte_tick) {
            USART3->DR = cur_frame[cur_idx];
            USART3->SR |= USART_SR_RXNE;        /* 硬件置 RXNE */
            USART3_IRQHandler();                /* ← 走真实中断入口 */
            cur_idx++;
            if (cur_idx >= (int)ESP32_DL_LEN) { cur_idx = -1; last_done_seq = seq; }
            next_byte_tick = t + BYTE_TICKS;
        }

        /* ---- 3、下游: 主循环该排空了吗 ---- */
        if ((t % LOOP_TICKS) == 0u) {
            /* 最近"整帧发完"的那条 seq —— 它才是应该被解出来的最新那条。
             * 若 poll 之后 cmd.vx 不等于它, 说明这一帧没能原样解出(丢字节/对齐失败) */
            uint16_t expect = last_done_seq;

            n_polls++;
            esp32_poll();                       /* ← 真实的消费函数 */

            {   const esp32_cmd_t *c = esp32_get_cmd();
                int32_t v = (int32_t)(c->vx * ESP32_DL_SCALE);
                /* vx==0 是 DEADMAN 归零, 不算"解出一帧" */
                if (v != last_vx && v != 0) { last_vx = v; n_parsed++; }
                else if (v == 0) last_vx = 0;

                if (expect != 0u) {
                    n_check++;
                    if ((uint16_t)v == expect) n_exact++;
                }
            }
        }
    }

    printf("%6.0f  %8.1f  %7.2f  %9.1f%%  %9.1f%%\n",
           freq_hz, jitter_ms,
           n_polls ? (double)n_sent / (double)n_polls : 0.0,
           n_polls ? (double)n_parsed / (double)n_polls * 100.0 : 0.0,
           n_check ? (double)n_exact / (double)n_check * 100.0 : 0.0);
    return 0;
}
