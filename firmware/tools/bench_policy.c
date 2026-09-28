/* bench_policy.c — PC 上量 policy_run 的耗时(评"20ms 预算够不够")
 *
 * ⚠️ 读结果前先明白: PC(x86-64 @几GHz) 和 F407(Cortex-M4F @168MHz) 完全不是一个量级,
 *    **绝对时间不可比**。真正能带走的信息是【每 MAC 需要几个周期】:
 *      PC   : 内层循环 out[o] += xv * row[o], 实测周期数
 *      M4F  : LDRB(取 int8) + VCVT(转 float) + VMLA(乘加) + 循环开销
 *             ≈ 4~6 周期/MAC(有 FPU; 且我们没开循环展开, 编译器不会自动向量化 int8)
 *    把 PC 的 ns/MAC 换成"周期/MAC", 再按 168MHz 折算, 才是 F407 的量级。
 *
 * 用法: 见 tools/run_bench.sh
 */
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include "cat_policy.h"

#define N_ITER 20000

int main(void)
{
    float obs[OBS_DIM], act[ACT_DIM];

    /* 造一组"在训练分布内"的观测: 以 obs_mean 为中心、按 obs_std 缩放扰动。
     * (别用随便造的数值 —— 落在分布外时网络输出会全饱和到 ±1, 虽然不影响耗时, 
     *  但那样测出来的是"极端输入"而不是"典型输入"。) */
    for (uint16_t i = 0; i < OBS_DIM; i++)
        obs[i] = POL_OBS_MEAN[i]
               + 0.5f * POL_OBS_STD[i] * (float)((int)((i * 7u) % 11u) - 5);

    /* 预热: 让权重数组进 cache、把分支预测器喂熟 */
    for (int k = 0; k < 2000; k++)
        policy_run(obs, act);

    clock_t t0 = clock();
    for (int k = 0; k < N_ITER; k++)
        policy_run(obs, act);
    clock_t t1 = clock();

    double us = (double)(t1 - t0) / CLOCKS_PER_SEC * 1e6 / (double)N_ITER;

    /* 总 MAC 数(与 POL_DIM_* 一致): L1 + L2 + L3 + head */
    const long mac =
        (long)OBS_DIM * 512 + 512L * 256 + 256L * 128 + 128L * 24;

    printf("========== policy_run 基准 ==========\n");
    printf("迭代次数      : %d\n", N_ITER);
    printf("总 MAC/次     : %ld\n", mac);
    printf("PC 耗时       : %.2f us/次  (%.4f ns/MAC)\n", us, us * 1000.0 / (double)mac);
    printf("\n");
    printf("输出动作[0..5]: %.4f %.4f %.4f %.4f %.4f %.4f\n",
           act[0], act[1], act[2], act[3], act[4], act[5]);
    printf("(只用来确认网络真的跑了, 不是全 0)\n");

    /* ---- 按 F407 折算: 周期/MAC 用实测的 PC 值反推不了 M4, 所以给区间估计 ---- */
    printf("\n========== F407 @168MHz 估计 ==========\n");
    for (int cyc = 2; cyc <= 8; cyc += 2) {
        double ms = (double)mac * cyc / 168.0e6 * 1000.0;
        printf("  %d 周期/MAC  ->  %.2f ms/次   (占 20ms 的 %.0f%%)\n",
               cyc, ms, ms / 20.0 * 100.0);
    }
    printf("  参考: 4~6 周期/MAC 是「没开循环展开 + int8→float 转换」的常见区间;\n");
    printf("        sil_deploy.py 里按 2 MAC/cyc 估过(那是理论峰值, 偏乐观)。\n");
    return 0;
}
