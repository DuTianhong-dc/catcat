#!/bin/bash
# run_bench.sh — 在 PC 上量 policy_run 的耗时(编译真实 cat_policy.c + policy_weights.c)
#
# 用法: bash tools/run_bench.sh
# 说明: 只桩掉 stm32f4xx.h(纯类型), 策略代码是原样的 —— 所以测的是真算法。
# ⚠️ PC 的绝对时间不能直接当 F407 用, 要看输出里"周期/MAC → ms"那张折算表。
# 见 tools/bench_policy.c 的文件头。

SRC="$(cd "$(dirname "$0")/.." && pwd)"
GCC="C:/Users/dth15/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin/gcc.exe"
STUB=$(mktemp -d)

# 桩: 只要类型定义(cat_policy.c 不碰任何寄存器)
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
#endif
EOF

echo "=============================================="
echo "  编译(真实 cat_policy.c + policy_weights.c)"
echo "=============================================="
"$GCC" -O2 -Wall -Wextra -I"$STUB" -I"$SRC" -I"$SRC/tools" \
       "$SRC/tools/bench_policy.c" "$SRC/cat_policy.c" "$SRC/policy_weights.c" \
       -o "$STUB/bench.exe" 2>&1 | head -20

if [ ! -f "$STUB/bench.exe" ]; then
    echo "编译失败。"
    rm -rf "$STUB"
    exit 1
fi

"$STUB/bench.exe"
rm -rf "$STUB"
