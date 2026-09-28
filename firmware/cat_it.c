/* ============================================================
 *  cat_it.c — 中断服务函数（本工程自己的, 不用根目录那份 stm32f4xx_it.c）
 *
 *  里面放: Cortex-M4 标准异常(空实现) + SysTick(转发给 cat_sys) + 各外设 ISR。
 *  ⚠️ 每个 Handler 全工程只能定义一处 —— 所以工程里【不能再引用】根目录的 stm32f4xx_it.c,
 *     否则 NMI_Handler 这些会重复定义。那份 it.c 依赖 HAL, 我们这份不依赖。
 *  本文件只做"响应", 具体实现归各模块: 时基在 cat_sys, IMU 收字节在 cat_imu。
 * ============================================================ */
#include "stm32f4xx.h"
#include "cat_config.h"
#include "cat_sys.h"
#include "cat_imu.h"
#include "cat_esp32.h"

/* ==================== Cortex-M4 异常 ==================== */
void NMI_Handler(void){}
void HardFault_Handler(void){ while (1){} }     /* 卡死在这: 读 PC/LR 可定位出错点 */
void MemManage_Handler(void){ while (1){} }
void BusFault_Handler(void){ while (1){} }
void UsageFault_Handler(void){ while (1){} }
void SVC_Handler(void){}
void DebugMon_Handler(void){}
void PendSV_Handler(void){}

void SysTick_Handler(void){
    sys_tick_isr();                             /* 实现在 cat_sys.c: g_ms++ */
}

/* ==================== 外设中断 ==================== */
/* IMU(USART2): 收到一个字节就推进环形缓冲。中断里只做这一件事, 尽快返回。
 * 读 DR 本身就会清 RXNE 标志, 不用手动清。 */
void USART2_IRQHandler(void){
    imu_data_recv((uint8_t)(USART2->DR & 0xFF));
}

/* ESP32(USART3): 收到一个字节就推进环形缓冲。跟 IMU 那一套一样: 中断里只做这一件事。
 * ⚠️ 必须读 DR: 读一下才清 RXNE, 不读会反复进中断把 CPU 占满。 */
void USART3_IRQHandler(void){
    esp32_data_recv((uint8_t)(USART3->DR & 0xFF));
}
