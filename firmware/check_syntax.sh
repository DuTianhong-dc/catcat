#!/bin/bash
# ============================================================
# cat_servo.c 等源文件的「快速语法检查」
#
# 用途：不开 Keil，用 gcc + 桩头文件检查 语法/类型/警告
#       （绕开 ARMCC 依赖：CMSIS 是用 armcc 内联汇编写的，gcc 读不了）
#
# 用法：在 Git Bash 里  cd 到本目录，执行  bash check_syntax.sh
# ============================================================

SRC="$(cd "$(dirname "$0")" && pwd)"
GCC="C:/Users/dth15/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin/gcc.exe"
STUB=$(mktemp -d)

# 桩头文件：补 stdint 类型 + 本项目用到的外设寄存器（够检查逻辑与寄存器写法）
cat > "$STUB/stm32f4xx.h" <<'EOF'
#ifndef STUB_STM32F4XX_H
#define STUB_STM32F4XX_H

/* ---------- 整数类型 ---------- */
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef signed char        int8_t;
typedef signed short       int16_t;
typedef signed int         int32_t;
typedef unsigned long long uint64_t;
typedef unsigned long long uintptr_t;   /* 2026-09-29 加: cat_policy 的地址转换用 */

/* ---------- 外设寄存器 ----------
 * 只保证【成员名】正确，地址与字段偏移不代表真芯片(真机以 Keil 的 stm32f407xx.h 为准)。
 * 目的：让 gcc 能解析 USART6->BRR / GPIOC->AFR[0] 这类写法，查出拼写和类型错误。 */
typedef struct { volatile uint32_t MODER, OTYPER, OSPEEDR, PUPDR,
                                  IDR, ODR, BSRR, LCKR, AFR[2]; } GPIO_TypeDef;
typedef struct { volatile uint32_t SR, DR, BRR, CR1, CR2, CR3, GTPR; } USART_TypeDef;
/* RCC: 前 4 个是时钟配置(clock_init)要用的, 后面 AHB1/APB1/APB2ENR 是各模块开时钟要用的。
 * 中间用占位数组把顺序撑开, 偏移不要求与实际一致 —— 桩只保证"成员名"能编过。 */
typedef struct { volatile uint32_t CR, PLLCFGR, CFGR, CIR;
                 volatile uint32_t PAD1[8], AHB1ENR, AHB2ENR, AHB3ENR;
                 volatile uint32_t PAD2[3], APB1ENR, APB2ENR, CSR; } RCC_TypeDef;   /* CSR: cat_safety 读复位源用 */
/* TIM(定时器): 顺序按 RM0090, 桩只保证成员名能编过 */
typedef struct { volatile uint32_t CR1, CR2, SMCR, DIER, SR, EGR, CCMR1, CCMR2, CCER,
                                  CNT, PSC, ARR, CCR1, CCR2, CCR3, CCR4; } TIM_TypeDef;
typedef struct { volatile uint32_t CTRL, CYCCNT; } DWT_TypeDef;
typedef struct { volatile uint32_t DEMCR; } CoreDebug_TypeDef;
typedef struct { volatile uint32_t CTRL, LOAD, VAL, CALIB; } SysTick_TypeDef;
typedef struct { volatile uint32_t ACR, KEYR, OPTKEYR, SR, CR, OPTCR; } FLASH_TypeDef;
typedef struct { volatile uint32_t KR, PR, RLR, SR; } IWDG_TypeDef;                 /* cat_safety 看门狗 */
typedef struct { volatile uint32_t IDCODE, CR, APB1FZ, APB2FZ; } DBGMCU_TypeDef;    /* 调试冻结用 */

#define GPIOC     ((GPIO_TypeDef *)  0x40020800UL)
#define GPIOB     ((GPIO_TypeDef *)  0x40020400UL)
#define GPIOA     ((GPIO_TypeDef *)  0x40020000UL)
#define GPIOE     ((GPIO_TypeDef *)  0x40021000UL)   /* 按键 KEY0(PE4) 用 */
#define USART6    ((USART_TypeDef *) 0x40011400UL)
#define USART2    ((USART_TypeDef *) 0x40004400UL)
#define USART3    ((USART_TypeDef *) 0x40004800UL)
#define TIM4      ((TIM_TypeDef *)   0x40000800UL)
#define RCC       ((RCC_TypeDef *)   0x40023800UL)
#define DWT       ((DWT_TypeDef *)   0xE0001000UL)
#define CoreDebug ((CoreDebug_TypeDef *) 0xE000EDF0UL)
#define SysTick   ((SysTick_TypeDef *)    0xE000E010UL)
#define FLASH     ((FLASH_TypeDef *)      0x40023C00UL)
#define IWDG      ((IWDG_TypeDef *)       0x40003000UL)
#define DBGMCU    ((DBGMCU_TypeDef *)     0xE0042000UL)

/* ---------- 中断号 + NVIC（真值分别是 CMSIS 的 IRQn_Type / core_cm4.h 内联函数） ---------- */
/* 中断号取真值: stm32f407xx.h 的 IRQn_Type (USART2=38, USART3=39)
 * SysTick_IRQn 取真值 -1((CMSIS 里就是负号的那一个) —— cat_sys.c 设时基优先级要用 */
typedef enum { SysTick_IRQn = -1, USART2_IRQn = 38, USART3_IRQn = 39 } IRQn_Type;
#define NVIC_EnableIRQ(x)        ((void)(x))
#define NVIC_SetPriority(x, p)   ((void)(x), (void)(p))
#define __WFI()                  ((void)0)   /* core_cm4.h 的内联汇编 → 桩里变空操作 */

/* ---------- 位宏（用真实值） ---------- */
#define USART_SR_RXNE            (1u << 5)
#define USART_SR_TC              (1u << 6)
#define USART_SR_TXE             (1u << 7)
#define USART_CR1_RE             (1u << 2)
#define USART_CR1_TE             (1u << 3)
#define USART_CR1_RXNEIE         (1u << 5)
#define USART_CR1_UE             (1u << 13)
#define RCC_AHB1ENR_GPIOAEN      (1u << 0)
#define RCC_AHB1ENR_GPIOBEN      (1u << 1)
#define RCC_AHB1ENR_GPIOCEN      (1u << 2)
#define RCC_AHB1ENR_GPIOEEN      (1u << 4)
#define RCC_APB1ENR_USART2EN     (1u << 17)
#define RCC_APB1ENR_USART3EN     (1u << 18)
#define RCC_APB1ENR_TIM4EN       (1u << 2)
#define RCC_APB2ENR_USART6EN     (1u << 5)
#define CoreDebug_DEMCR_TRCENA_Msk   (1u << 24)
#define DWT_CTRL_CYCCNTENA_Msk       (1u << 0)

/* ---------- FLASH 位宏(2026-09-29 加: cat_polupd 用; 值按 RM0090) ---------- */
#define FLASH_SR_EOP             (1u << 0)
#define FLASH_SR_WRPERR          (1u << 4)
#define FLASH_SR_PGAERR          (1u << 5)
#define FLASH_SR_PGPERR          (1u << 6)
#define FLASH_SR_PGSERR          (1u << 7)
#define FLASH_SR_BSY             (1u << 16)
#define FLASH_CR_PG              (1u << 0)
#define FLASH_CR_SER             (1u << 1)
#define FLASH_CR_SNB             (0xFu << 3)
#define FLASH_CR_SNB_Pos         3u
#define FLASH_CR_PSIZE           (3u << 8)      /* 大小档掩码(补齐/清位用) */
#define FLASH_CR_PSIZE_1         (2u << 8)      /* x32 (VDD 2.7~3.6V) */
#define FLASH_CR_STRT            (1u << 16)
#define FLASH_CR_LOCK            (1u << 31)

/* ---------- IWDG / 复位源位宏(2026-09-29 深夜加: cat_safety 看门狗; 值按 RM0090) ---------- */
#define IWDG_SR_RVU              (1u << 1)
#define RCC_CSR_RMVF             (1u << 24)
#define RCC_CSR_SFTRSTF          (1u << 28)
#define RCC_CSR_IWDGRSTF         (1u << 29)
#define DBGMCU_APB1_FZ_DBG_IWDG_STOP   (1u << 12)
#define NVIC_SystemReset()       ((void)0)   /* core_cm4.h 内联 → 桩里变空操作 */

/* ---------- 时钟配置用到的位宏(main.c 的 clock_init) ---------- */
#define RCC_CR_HSEON             (1u << 16)
#define RCC_CR_HSERDY            (1u << 17)
#define RCC_CR_PLLON             (1u << 24)
#define RCC_CR_PLLRDY            (1u << 25)
#define RCC_PLLCFGR_PLLSRC_HSE   (1u << 22)
#define RCC_CFGR_SW_PLL          (2u)
#define RCC_CFGR_SWS             (3u << 2)
#define RCC_CFGR_SWS_PLL         (2u << 2)
#define RCC_CFGR_HPRE_DIV1       (0u)
#define RCC_CFGR_PPRE1_DIV4      (5u << 10)
#define RCC_CFGR_PPRE2_DIV2      (4u << 13)
#define FLASH_ACR_ICEN           (1u << 9)
#define FLASH_ACR_DCEN           (1u << 10)
#define FLASH_ACR_PRFTEN         (1u << 8)
#define FLASH_ACR_LATENCY_5WS    (5u)

#endif
EOF

echo "=============================================="
echo "  语法检查（-Wall -Wextra）"
echo "  目录: $SRC"
echo "=============================================="
echo ""

n_ok=0; n_bad=0
for f in "$SRC"/*.c; do
    [ -f "$f" ] || continue
    echo "--- $(basename "$f") ---"
    # -Wno-attributes: Keil 的 __attribute__((at(地址))) 是 ARMCC 专有, gcc 只会警告"不认识"
    out=$("$GCC" -I"$STUB" -fsyntax-only -Wall -Wextra -Wno-attributes "$f" 2>&1)
    if [ -z "$out" ]; then
        echo "  OK  零警告零错误"
        n_ok=$((n_ok+1))
    else
        echo "$out" | head -25 | sed 's/^/  /'
        n_bad=$((n_bad+1))
    fi
    echo ""
done

rm -rf "$STUB"

echo "=============================================="
echo "  通过 $n_ok 个，有问题 $n_bad 个"
echo "=============================================="
echo ""
echo "提示：这只查【语法/类型/警告】，查不出逻辑错误。"
echo "      逻辑错误要靠：小步测试 + 对协议手册核对。"
