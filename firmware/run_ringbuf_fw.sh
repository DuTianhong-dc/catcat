#!/bin/bash
# ============================================================
# run_ringbuf_fw.sh — 编译并运行【真实固件代码】的环形缓冲实验
#
# 跟 check_syntax.sh 的区别: 那份只 -fsyntax-only(不产出可执行文件),
# 这份要真编译 + 真运行, 跑的是 cat_esp32.c 的原样代码。
#
# 做法: 用桩头文件顶掉 stm32f4xx.h / cat_sys.h,
#       其中 USART3 指向**真内存**(usart3_mem), 所以读写寄存器是真实有效的。
#
# 用法: bash run_ringbuf_fw.sh
# ============================================================

SRC="$(cd "$(dirname "$0")" && pwd)"
GCC="C:/Users/dth15/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin/gcc.exe"
STUB=$(mktemp -d)

# ---------- 桩: 外设寄存器 ----------
# 与 check_syntax.sh 的桩同源, 只保留本实验用得到的部分。
# 关键差异: USART3 不指向固定地址, 而是 extern 到真内存 —— 必须能读写。
cat > "$STUB/stm32f4xx.h" <<'EOF'
#ifndef STUB_STM32F4XX_H
#define STUB_STM32F4XX_H

typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef signed char        int8_t;
typedef signed short       int16_t;
typedef signed int         int32_t;
typedef unsigned long long uint64_t;

typedef struct { volatile uint32_t MODER, OTYPER, OSPEEDR, PUPDR,
                                  IDR, ODR, BSRR, LCKR, AFR[2]; } GPIO_TypeDef;
typedef struct { volatile uint32_t SR, DR, BRR, CR1, CR2, CR3, GTPR; } USART_TypeDef;
typedef struct { volatile uint32_t CR, PLLCFGR, CFGR, CIR;
                 volatile uint32_t PAD1[8], AHB1ENR, AHB2ENR, AHB3ENR;
                 volatile uint32_t PAD2[3], APB1ENR, APB2ENR; } RCC_TypeDef;

typedef enum { USART2_IRQn = 38, USART3_IRQn = 39 } IRQn_Type;

extern USART_TypeDef usart3_mem;
#define USART3    (&usart3_mem)
#define GPIOB     ((GPIO_TypeDef *)  0x40020400UL)
#define RCC       ((RCC_TypeDef *)   0x40023800UL)

#define NVIC_EnableIRQ(x)        ((void)(x))
#define NVIC_SetPriority(x, p)   ((void)(x), (void)(p))

#define USART_SR_RXNE            (1u << 5)
#define USART_SR_TC              (1u << 6)
#define USART_SR_TXE             (1u << 7)
#define USART_CR1_RE             (1u << 2)
#define USART_CR1_TE             (1u << 3)
#define USART_CR1_RXNEIE         (1u << 5)
#define USART_CR1_UE             (1u << 13)
#define RCC_AHB1ENR_GPIOBEN      (1u << 1)
#define RCC_APB1ENR_USART3EN     (1u << 18)

#endif
EOF

# ---------- 桩: 系统层(只留时基两个函数) ----------
cat > "$STUB/cat_sys.h" <<'EOF'
#ifndef CAT_SYS_STUB_H
#define CAT_SYS_STUB_H
#include "stm32f4xx.h"
uint32_t sys_ms(void);
uint8_t  sys_elapsed(uint32_t from_ms, uint32_t ms);
#endif
EOF

echo "=============================================="
echo "  编译(真实 cat_esp32.c + 桩寄存器)"
echo "=============================================="
"$GCC" -O2 -Wall -Wextra -I"$STUB" -I"$SRC" \
       "$SRC/test_ringbuf_fw.c" "$SRC/cat_esp32.c" \
       -o "$STUB/rb_test.exe" 2>&1 | head -30

if [ ! -f "$STUB/rb_test.exe" ]; then
    echo "编译失败。"
    rm -rf "$STUB"
    exit 1
fi
echo "  编译通过 → 开始跑"
echo ""
echo "  仿真: 1 tick = 100μs, 主循环 20ms, 总时长 60s"
echo "  走的是 cat_esp32.c / cat_it.c 的真实代码路径"
echo ""
echo "  频率(Hz)  抖动(ms)    帧/拍      交付率    末帧命中率"
echo "  --------  --------  -------  ----------  ----------"

for f in 10 20 50 100 200 500; do
    for j in 0 2 10 30; do
        "$STUB/rb_test.exe" "$f" "$j"
    done
done

echo ""
rm -rf "$STUB"
