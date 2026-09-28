/* ============================================================
 *  cat_sys.c — 系统级配置 (时钟 / 时基 / 观测区)
 *  分层: ① 初始化 sys_clock_init/sys_tick_init  ② 时基 sys_ms/sys_elapsed/sys_tick_isr
 *  ⚠️ 不依赖任何外设模块(纯系统层); 跨模块的启动编排属于"编排", 放 main.c
 *  设计依据 / 踩坑记录 → 系统层说明.md
 * ============================================================ */
#include "cat_sys.h"
#include "cat_config.h"

/* 观测区: 固定地址方便 ST-Link 直读; 放 CCM 避开 at() 的 RAM 填充 → 系统层说明.md(四) */
dbg_t g_dbg __attribute__((at(0x10000000)));

/* 毫秒时基。⚠️ volatile 不能省(中断里改、主循环读) → 系统层说明.md(七) */
volatile uint32_t g_ms;

/* ==================== ① 初始化 ==================== */
/* 系统时钟 → 168MHz。必须最先调, 且早于 servo_init/imu_init(它们算 BRR 假设 APB 已配好);
 * 不配就跑 HSI 16MHz → 波特率差 10 倍 → 系统层说明.md(二) */
void sys_clock_init(void){
    uint8_t hse_ok;

    /* 1、开 HSE 并等就绪; 带有限次等待, 免得晶振坏了死在这儿 */
    RCC->CR |= RCC_CR_HSEON;
    for (volatile uint32_t i = 0; i < 0x8000u; i++) {
        if (RCC->CR & RCC_CR_HSERDY) break;
    }
    hse_ok = (RCC->CR & RCC_CR_HSERDY) ? 1u : 0u;      /* 起不来走 HSI 退路, 同样到 168MHz */

    /* 2、Flash: 168MHz 要 5 个等待周期, 另开预取 + 指令/数据缓存 */
    FLASH->ACR = FLASH_ACR_ICEN | FLASH_ACR_DCEN | FLASH_ACR_PRFTEN | FLASH_ACR_LATENCY_5WS;

    /* 3、PLL: 1MHz 进 ×336 ÷2 = 168MHz (两种源都先降到 1MHz) */
    RCC->PLLCFGR = (hse_ok ? 8u : 16u)              /* PLLM */
                 | (336u << 6)                       /* PLLN */
                 | (0u << 16)                        /* PLLP: 00 → ÷2 */
                 | ((hse_ok ? 4u : 7u) << 24)        /* PLLQ: USB/SDIO 用 */
                 | (hse_ok ? RCC_PLLCFGR_PLLSRC_HSE : 0u);
    RCC->CR |= RCC_CR_PLLON;
    while (!(RCC->CR & RCC_CR_PLLRDY));

    /* 4、切到 PLL + 定三条总线分频 */
    RCC->CFGR = RCC_CFGR_HPRE_DIV1                      /* AHB  ÷1 = 168MHz (内核/SRAM) */
              | RCC_CFGR_PPRE1_DIV4                     /* APB1 ÷4 =  42MHz (USART2=IMU) */
              | RCC_CFGR_PPRE2_DIV2                     /* APB2 ÷2 =  84MHz (USART6=舵机) */
              | RCC_CFGR_SW_PLL;
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL);
}

/* 毫秒时基: 每 1ms 中断一次, ISR 里只 g_ms++(见 cat_it.c)。必须在 sys_clock_init 之后 */
void sys_tick_init(void){
    SysTick->LOAD = CPU_MHZ * 1000u - 1u;    /* 1ms = 168000 周期(减 1 → 系统层说明.md(五)) */
    SysTick->VAL  = 0;                        /* 只在开机清这一次 */
    SysTick->CTRL = (1u << 0)                 /* ENABLE   : 启动计数 */
                  | (1u << 1)                 /* TICKINT  : 数到 0 产生中断 */
                  | (1u << 2);                /* CLKSOURCE: 用内核时钟 */

    /* ⚠️ 优先级压到【最低】(复位默认是 0, 即最高)。
     * 理由: SysTick 跑 1kHz, 若优先级最高就会【插队所有通信中断】(IMU / ESP32 都是 6),
     *   把它们的响应拖长 → 串口丢字节。2026-09-27 实测: 改前主循环 32ms、
     *   IMU 帧率在 34~267Hz 之间乱跳(应 100Hz)。
     * ⚠️ 压到最低【不影响时基准确性】: SysTick 是【硬件计数器】, 中断被延迟时
     *   溢出标志会挂起(pending), 返回后立刻补执行 —— g_ms 一拍都不会少, 最多晚几微秒。
     *   (只有 ISR 执行超过 1ms 才会丢拍, 而本 ISR 只做 g_ms++。) */
    NVIC_SetPriority(SysTick_IRQn, 15u);
}

/* ==================== ② 时基 ==================== */
/* 当前毫秒数(从开机累加, 49.7 天溢出一次) */
uint32_t sys_ms(void){
    return g_ms;
}

/* 从 from_ms 起是否过了 span_ms 毫秒。⚠️ 必须用差值(无符号回绕才对) → 系统层说明.md(六) */
uint8_t sys_elapsed(uint32_t from_ms, uint32_t span_ms){
    return (uint32_t)(g_ms - from_ms) >= span_ms;
}

/* SysTick 中断入口(由 cat_it.c 的 SysTick_Handler 转发) */
void sys_tick_isr(void){
    g_ms++;
}
