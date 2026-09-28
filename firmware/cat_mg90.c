/* ============================================================
 *  cat_mg90.c — MG90S 云台舵机驱动 (TIM4 / PB6-PB7, PWM)
 *
 *  分层(本文件按此顺序排列, 新函数插到对应区块, 别追加到末尾):
 *    ① 初始化   mg90_init()
 *    ② 输出     mg90_set()
 *    ③ 取数     mg90_get()
 *
 *  参数依据 通信协议.txt(六), 不是猜的 —— 几个关键坑在注释里指了路。
 * ============================================================ */
#include "cat_mg90.h"
#include "cat_config.h"

/* ==================== ① 初始化 ==================== */
/* 配 TIM4 CH1(PB6=pan) / CH2(PB7=tilt), 上电回中 90 度。 */
void mg90_init(void){
    //1、开时钟: GPIOB(AHB1) + TIM4(APB1)
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOBEN;
    RCC->APB1ENR |= RCC_APB1ENR_TIM4EN;

    //2、PB6/PB7 配 TIM4 复用(AF2): 复用 + 推挽 + 高速 + 无上下拉
    //   ⚠️ PB6 同时是 I2C1_SCL 默认复用脚 —— OTYPER/PUPDR/OSPEEDR 必须全部显式设,
    //      只改 MODER/AFR 会残留开漏 → PWM 变毛刺、舵机嘎嘎抖 → 协议(6.5)
    GPIOB->MODER = (GPIOB->MODER & ~((3u << 12) | (3u << 14)))
                 | (2u << 12) | (2u << 14);
    GPIOB->OTYPER  &= ~((1u << 6) | (1u << 7));
    GPIOB->OSPEEDR |=  (3u << 12) | (3u << 14);
    GPIOB->PUPDR   &= ~((3u << 12) | (3u << 14));
    GPIOB->AFR[0] = (GPIOB->AFR[0] & ~((0xFu << 24) | (0xFu << 28)))
                  | (2u << 24) | (2u << 28);

    //3、TIM4: ⚠️ F4 定时器时钟 = APB1×2 = 84MHz(不是 42MHz)
    //   PSC = 84-1 = 83 → 1MHz(1μs/tick); ARR = 19999 → 20ms(50Hz)
    //   若按 42MHz 取 41 会得 2MHz → 脉宽减半(1500μs 发成 750μs) → 协议(6.3)
    TIM4->PSC   = (APB1_MHZ * 2u) - 1u;
    TIM4->ARR   = MG90_PWM_PERIOD_US - 1u;
    TIM4->CCMR1 = 0x6868u;          //CH1/CH2: PWM 模式1 + 预装载
    /* ⚠️ bit3/bit11 (OCxPE) 是预装载, 必须开: 不开则写 CCR 【立即生效】, 若正好落在
     *    PWM 周期中间就会吐出畸变脉宽(例: 1500→500 在 CNT 已到 1000 时改, 本周期只剩
     *    1000μs ≈ 45 度) → 表现成云台每改一次角度就抖一下。 */
    TIM4->CCER |= 0x11u;            //CH1/CH2 输出使能
    TIM4->EGR   = 1u;               //生成更新事件, 把 PSC/ARR 装进去
    TIM4->CR1  |= 1u;               //启动

    //4、回中(90 度 = 1500μs): 先上电停在这, 再对齐齿轮花
    TIM4->CCR1 = MG90_PWM_MID_US;
    TIM4->CCR2 = MG90_PWM_MID_US;
}

/* ==================== ② 输出 ==================== */
/* 最近一次设定的角度 —— 云台是模拟舵机: 没反馈、不应答(协议 6.1), 想知道"当前角度"
 * 只能靠"上次发出去的是什么"回显。协议(6.6)要求上行载荷 +77/+78 回显的就是这两个。 */
static int8_t last_pan, last_tilt;

/* 脉宽硬钳位(μs)。⚠️ 协议(6.4): 不管哪条路径都不许把 CCR 送出这个区间 ——
 * MG90S 是金属齿, 超范围会顶死机械限位持续堵转 → 发热烧线圈。 */
#define MG90_CCR_MIN   MG90_PWM_MIN_US
#define MG90_CCR_MAX   MG90_PWM_MAX_US

#define MG90_CCR_CLAMP(c)  ((c) < (int)MG90_CCR_MIN ? (uint32_t)MG90_CCR_MIN : \
                            ((c) > (int)MG90_CCR_MAX ? (uint32_t)MG90_CCR_MAX : (uint32_t)(c)))

/* 设云台角度。入参 -127~127(跟协议 5.2 的 int8 一致), 映射 0~180 度。
 * 换算: 0 → 1500μs(中位), ±127 → 2500/500μs。直接算 CCR, 不经过角度。 */
void mg90_set(int8_t pan, int8_t tilt){
    int c1 = (int)MG90_PWM_MID_US + (int)pan  * (int)(MG90_PWM_MAX_US - MG90_PWM_MID_US) / 127;
    int c2 = (int)MG90_PWM_MID_US + (int)tilt * (int)(MG90_PWM_MAX_US - MG90_PWM_MID_US) / 127;

    TIM4->CCR1 = MG90_CCR_CLAMP(c1);        /* pan  水平 */
    TIM4->CCR2 = MG90_CCR_CLAMP(c2);        /* tilt 俯仰 */
    /* 不用手动触发更新: CCMR1 里开了预装载, 新值会在下一个更新事件(20ms 内)生效 */

    last_pan  = pan;                        /* 记下来, 供 mg90_get 回显 */
    last_tilt = tilt;
}

/* ==================== ③ 取数 ==================== */
/* 回读最近一次设定的值(不是"实际角度" —— 模拟舵机压根没有反馈)。
 * 上电后没设过时是 0 = 中位, 与 mg90_init 的回中一致。 */
void mg90_get(int8_t *pan, int8_t *tilt){
    *pan  = last_pan;
    *tilt = last_tilt;
}
