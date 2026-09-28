#!/bin/bash
# ============================================================
# run_ble_fw.sh — 编译并运行【第一版固件 cat_ble.c】的时序仿真
#
# 跟 run_ringbuf_fw.sh 同一套方法, 换被测对象:
#   第二版 CAT_APP2.0/cat_esp32.c : 10B 帧, 满时丢最旧
#   第一版 CAT_APP/cat_ble.c      : 12B 帧, 满时丢新
#
# 桩覆盖 4 个真头文件(它们各自拉了别的模块, 整块不编):
#   stm32f4xx_hal.h  → 寄存器 + 位宏 + HAL_GetTick
#   cat_ble_proto.h  → 0xCC 加载协议整块换成 static 空实现(真头里也是 static)
#   cat_gimbal.h     → cat_gimbal_set 空实现
#   cat_fsm.h        → cat_fsm_get_state 桩
# 真头照用: cat_config.h / cat_ble.h (两者只依赖 stdint.h)
#
# 用法: bash run_ble_fw.sh
# ============================================================

SRC="$(cd "$(dirname "$0")" && pwd)"
FIRST="$(cd "$SRC/../CAT_APP" && pwd)"
GCC="C:/Users/dth15/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin/gcc.exe"
STUB=$(mktemp -d)

echo "  被测第一版目录: $FIRST"
echo ""

# ---------- 桩: 外设寄存器 ----------
cat > "$STUB/stm32f4xx_hal.h" <<'EOF'
#ifndef STUB_STM32F4XX_HAL_H
#define STUB_STM32F4XX_HAL_H
#include <stdint.h>

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

uint32_t HAL_GetTick(void);
#endif
EOF

# ---------- 桩: 0xCC 加载协议(真头里是 static 实现, 这里换成空实现) ----------
cat > "$STUB/cat_ble_proto.h" <<'EOF'
#ifndef __CAT_BLE_PROTO_H
#define __CAT_BLE_PROTO_H
#include <stdint.h>
#include <string.h>     /* 真头也 include 了它 —— cat_ble.c 的 memset 靠这里带入 */
#include "cat_config.h"

/* 常量按真头文件取值(cat_ble_proto.h:73/150) */
#define BLE_FRAME_LOAD    0xCCu
#define BLE_LS_IDLE         0u

/* 类型只保留 cat_ble.c 访问得到的成员(state); 大小不重要, 因为只用 sizeof 自己 */
typedef struct { uint8_t state; uint8_t pad[131]; } cat_ble_load_parser_t;
typedef struct { uint8_t state; uint8_t on;  uint8_t pad[66]; } cat_ble_maint_t;

/* 真头里这两个就是 static(实现写在头文件内) —— 桩必须也保持 static,
 * 否则 cat_ble.c 编译时看到的是"外部函数", 链接就找不到定义了 */
static void cat_ble_load_parse_byte(cat_ble_load_parser_t *p, uint8_t b){
    (void)p; (void)b;
}
static uint16_t cat_ble_poll_frame(cat_ble_load_parser_t *p, cat_ble_maint_t *m,
                                   uint8_t *ack, uint16_t ack_max){
    (void)p; (void)m; (void)ack; (void)ack_max;
    return 0u;
}
#endif
EOF

# ---------- 桩: 云台 ----------
cat > "$STUB/cat_gimbal.h" <<'EOF'
#ifndef STUB_CAT_GIMBAL_H
#define STUB_CAT_GIMBAL_H
#include <stdint.h>
void cat_gimbal_set(int8_t pan, int8_t tilt);
#endif
EOF

# ---------- 桩: 状态机 ----------
cat > "$STUB/cat_fsm.h" <<'EOF'
#ifndef STUB_CAT_FSM_H
#define STUB_CAT_FSM_H
#include "cat_config.h"     /* cat_fsm_state_t 定义在这里 */
cat_fsm_state_t cat_fsm_get_state(void);
#endif
EOF

echo "=============================================="
echo "  编译(第一版真实 cat_ble.c + 桩)"
echo "=============================================="
# ⚠️ 必须把 cat_ble.c 复制到桩目录再编译:
#    #include "cat_ble_proto.h" 会【优先找源文件所在目录】, 直接编 CAT_APP 里那份
#    就会拿到真的 proto 头(它又拉 cat_action.h/cat_polupd.h/servo.h... 编不动)。
#    放进 $STUB 后, 它才认我们写的桩头。
cp "$FIRST/cat_ble.c" "$STUB/cat_ble.c"
"$GCC" -O2 -Wall -Wno-unused-function -I"$STUB" -I"$FIRST" \
       "$SRC/test_ble_fw.c" "$STUB/cat_ble.c" \
       -o "$STUB/ble_test.exe" 2>&1 | head -30

if [ ! -f "$STUB/ble_test.exe" ]; then
    echo "编译失败。"
    rm -rf "$STUB"
    exit 1
fi
echo "  编译通过 → 开始跑"
echo ""
echo "  仿真: 1 tick = 100μs, 主循环 20ms, 总时长 60s"
echo "  走的是第一版 cat_ble.c 的真实代码路径"
echo ""
echo "  频率(Hz)  抖动(ms)    帧/拍      交付率    末帧命中率    帧解码率"
echo "  --------  --------  -------  ----------  ----------  ----------"

for f in 10 20 50 100 200 500; do
    for j in 0 2 10 30; do
        "$STUB/ble_test.exe" "$f" "$j"
    done
done

echo ""
rm -rf "$STUB"
