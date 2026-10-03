/* ============================================================
 *  main.c — 主程序(只负责"编排": 按顺序调各模块, 不装实现)
 *
 *  启动顺序(2026-09-30 收口: 按键分模式):
 *    ① sys_clock_init() / sys_tick_init()   时钟 168MHz + 毫秒时基   ← cat_sys
 *    ② 观测区清零 + magic/版本               真机调试窗口            ← 结构体在 cat_sys.h
 *    ③ safety_iwdg_init()                   看门狗 + 读复位源(两种模式都要) ← cat_safety
 *    ④ 上电按住 KEY0(PE4)? → 【更新模式】 ota_mode_run():
 *       只开 ESP32 链路 + OTA —— IMU/舵机/云台全不初始化
 *    ⑤ 正常模式: 外设初始化 → boot_wait_ready() → 自检/选策略 → 主循环 50Hz
 *
 *  ⚠️ 为什么有"更新模式"(2026-09-30 真机踩坑后定): 正常模式擦 Flash 时, IMU 的
 *     100Hz 数据流会在擦除结束后形成中断风暴把主循环饿死(pyocd 抓 CPU 现场实锤) ——
 *     更新模式不初始化 IMU, 从根上避开; 正常模式收到 BEGIN 会直接拒绝(rc=0x1A)。
 *  ⚠️ 上电【纯不动】(煜辰 2026-09-29 定): 开机只"读"(ping/读阈值/读 IMU), 不主动发
 *     任何位置/扭矩指令 —— 舵机保持上电原状; 第一条目标指令由主循环"在连+使能开"闸门放行。
 *  ⚠️ 睡眠节拍: 忙等到期就 __WFI() 睡(睡眠模式下外设全在跑, SysTick 每 1ms 中断唤醒;
 *     依据 RM0090 低功耗章: "睡眠=只停内核时钟, 外设含 SysTick 照常")。
 *
 *  约定: 配置/实现都归各自模块(cat_servo / cat_imu / cat_sys), 这个文件只排顺序。
 *        跨模块的启动流程(如 ⑤)也算"编排", 放这里 —— 别塞进某个模块, 否则它会反过来依赖别人。
 * ============================================================ */
#include "stm32f4xx.h"
#include <string.h>             /* memset: 开机清观测区 */
#include "cat_config.h"
#include "cat_sys.h"
#include "cat_servo.h"
#include "cat_imu.h"
#include "cat_esp32.h"
#include "cat_mg90.h"
#include "cat_obs.h"
#include "cat_policy.h"
#include "cat_polupd.h"
#include "cat_action.h"
#include "cat_pose.h"       /* CAT_POSE_DEF: 动作播放的"当前绝对角"与出场目标 */
#include "cat_safety.h"
#include "cat_vel_est.h"

/* ==================== ④ 上电等待 ====================
 * 这里放(不放 cat_sys): 它【跨模块】(既要 ping 舵机又要读 IMU 帧), 属于"编排"而不是某个模块的实现。
 * 目的: 不去规定"开发板和舵机/IMU 的电源谁先开", 而是开机后给个窗口反复等。
 * 每轮看两件事: 12 颗舵机全应答 + IMU 这段时间收到了新帧。
 * 连续 BOOT_OK_ROUNDS 轮都过 → 过关, 之后不再检测; 窗口用完没过 → 返回 0, 但不卡死。
 * 窗口说明: 资料写 IMU"启动时间 5000ms", 实测约 1 秒内就能收到帧(它一直有电, 复位的是 STM32)。
 *           10 秒对"等别人上电"来说够宽, 又不至于让"真没接"的情况等太久。 */
#define BOOT_WAIT_MS     10000u    /* 窗口长度 */
#define BOOT_ROUND_MS      100u    /* 每轮间隔(给 IMU 留出吐至少一帧的时间) */
#define BOOT_OK_ROUNDS       2u    /* 连续通过几轮才算过关(1 轮可能侥幸) */

/* 主循环周期。50Hz = 20ms。跟策略训练时的控制频率对齐, 改这里要同步改训练侧 */
#define LOOP_PERIOD_MS      20u

/* IMU 静默阈值(2026-10-03 审查 M2): 连续这么多拍没有新帧 = 判定 IMU 掉了。
 * 100Hz 出帧 → 每拍本应有 2 帧, 500ms(25 拍) 连一帧都没有, 只可能是断线/模块死。
 * ⚠️ 触发后【不发新目标】(舵机保持出力) —— 与 DEADMAN 同语义: 停, 但不卸力;
 *    卸力只留给真故障(安全闸)。恢复出新帧即自动解除(计数清零)。 */
#define IMU_SILENT_LOOPS   (500u / LOOP_PERIOD_MS)

/* ==================== 更新模式(上电按键进入) ====================
 * 进入: 上电时按住 KEY0(PE4, 按下为低)。只初始化 ESP32 链路 + OTA 状态机,
 *       IMU/舵机/云台一概不初始化 —— 没有它们的定时器/中断, 更新过程绝对干净。
 * 退出: END 成功 → 延时 1s(让 ACK 发出) → 软复位 → 正常启动即用新策略。
 * 协议不变: PC 工具(ble_ota.py)照旧; 连接后 0x74 查询有应答 = 更新模式在跑。 */

/* 上电按键: 探索者 KEY0/1/2 = PE4/PE3/PE2, 任意一个按住 → 进更新模式(按下为低)。
 * ⚠️ 官方资料明确"按键外部没有上下拉" —— 必须设【内部上拉】, 否则引脚悬空、
 *    上电电平随机(首测"不按键也误进更新模式"的根因; 2026-09-30 资料+实测双确认)。
 *    另加"上电后等 150ms 再检测"双保险(等上电瞬态稳定)。
 * ⚠️ 板子上没丝印也别慌: 按住【任意一个】用户键上电即可, 三个键共用此入口。 */
#define KEY_MASK  ((1u << 4) | (1u << 3) | (1u << 2))   /* PE4/PE3/PE2 */
static uint8_t boot_key_held(void){
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOEEN;
    GPIOE->MODER &= ~((3u << 8) | (3u << 6) | (3u << 4));                   /* PE4/3/2 输入 */
    GPIOE->PUPDR &= ~((3u << 8) | (3u << 6) | (3u << 4));
    GPIOE->PUPDR |=  ((1u << 8) | (1u << 6) | (1u << 4));                   /* 三脚内部上拉 */
    {   uint32_t r0 = sys_ms();
        while (!sys_elapsed(r0, 150u));             /* 等上电瞬态稳定 */
        if ((GPIOE->IDR & KEY_MASK) == KEY_MASK) return 0u;                 /* 全高 = 没按 */
        r0 = sys_ms();
        while (!sys_elapsed(r0, 20u));
        return ((GPIOE->IDR & KEY_MASK) == KEY_MASK) ? 0u : 1u;             /* 仍有一位低 = 真按住 */
    }
}

/* 更新模式主循环(不返回; END 成功后 1s 软复位退出)。调用前: 时钟/时基/观测区/IWDG 已就绪。 */
static void ota_mode_run(void){
    uint32_t t_now  = sys_ms();
    uint32_t die_at = 0u;                           /* 非 0 = END 已成功, 软复位倒计时中 */
    esp32_init();                                   /* 唯一的通信外设(USART3) */
    cat_polupd_set_ota_mode(1u);                    /* 放行 BEGIN(正常模式会拒) */
    /* 扫槽定"激活"信息 —— 少了这步 polupd 以为无激活: 目标槽总选 A、seq 防误刷检查失效 */
    policy_select();
    cat_polupd_set_active(policy_source(), policy_seq(), policy_tag(), policy_build_id());
    g_dbg.stage = DBG_STAGE_OTA;

    while (1) {
        g_dbg.ms = sys_ms();                        /* 观测区心跳: 涨 = 更新模式活着 */
        g_dbg.loops++;
        esp32_poll();                               /* 排空缓冲(收 CC 帧) */

        {   uint16_t ota_n = 0u;
            const uint8_t *ota = esp32_ota_fetch(&ota_n);
            if (ota) {
                uint8_t resp[48], rtype = 0u, rlen = 0u;
                uint8_t sub = ota[0];
                cat_polupd_frame(ota, ota_n, &rtype, resp, &rlen);
                if (rtype) esp32_send_up(rtype, resp, rlen);
                if (sub == POL_SUB_END && resp[0] == POL_RC_OK)
                    die_at = sys_ms();              /* 不立即复位: 留 1s 应答 PC 的 STAT 复查 */
            }
        }

        if (die_at != 0u && sys_elapsed(die_at, 1000u))
            NVIC_SystemReset();                     /* 复位后正常启动 = 新策略生效 */
        safety_iwdg_feed();
        while (!sys_elapsed(t_now, LOOP_PERIOD_MS)) __WFI();   /* 20ms 节拍(同主循环) */
        t_now += LOOP_PERIOD_MS;
    }
}

static uint8_t boot_wait_ready(void){
    uint32_t t0 = sys_ms();
    uint16_t last_frames = imu_get_frames();        /* 记上轮帧数, 用来判"有没有新数据" */
    uint8_t  ok_rounds = 0;

    while (!sys_elapsed(t0, BOOT_WAIT_MS)) {
        uint8_t round_ok = 0;

        safety_iwdg_feed();     /* 等待窗口最长 10s, 也要喂狗(每轮 100ms, 余量 10 倍) */

        /* ⚠️ 这里必须自己解析一遍 IMU: imu_get_frames() 只在 imu_poll() 里增长,
         *    而本函数跑在主循环之前 —— 不调它的话 imu_get_frames() 恒为 0,
         *    下面那个判断永远不成立, 会白等满整个窗口。
         *    (真踩过: 误以为是"IMU 启动要 19 秒", 其实是没人去解析缓冲) */
        imu_poll();

        if (servo_ping_all() == 0u) {           /* 舵机: 12 颗全应答 */
            if (imu_get_frames() != last_frames)    /* IMU: 帧数动了 = 有新帧进来 */
                round_ok = 1u;
        }
        last_frames = imu_get_frames();

        ok_rounds = round_ok ? (uint8_t)(ok_rounds + 1u) : 0u;   /* 断了就重新数 */
        if (ok_rounds >= BOOT_OK_ROUNDS)
            return 1u;                                           /* 过关 */

        {   /* 这一轮歇会儿。用 sys_ms() 忙等(1ms 精度够用), 不依赖 DWT/别的模块 */
            uint32_t r0 = sys_ms();
            while (!sys_elapsed(r0, BOOT_ROUND_MS));
        }
    }
    return 0u;                                  /* 窗口用完还没就绪 */
}

/* ==================== ⑥ 状态上报（编排） ====================
 * 为什么放这里而不是 cat_esp32: 组包要同时取 IMU(姿态) 和舵机(位置) 的数据, 属于【跨模块编排】。
 *   放进模块会逼着 cat_esp32 反过来 include cat_imu —— 模块之间横向依赖, 分层就乱了。
 *   cat_esp32 只管"收发 + 解析", 组包归编排层, 跟 ④ boot_wait_ready 是同一个道理。 */
#define PUT16(p, v) do { (p)[0] = (uint8_t)((v) & 0xFFu); \
                         (p)[1] = (uint8_t)(((v) >> 8) & 0xFFu); } while (0)
#define PUT32(p, v) do { (p)[0] = (uint8_t)((v) & 0xFFu);         (p)[1] = (uint8_t)(((v) >> 8)  & 0xFFu); \
                         (p)[2] = (uint8_t)(((v) >> 16) & 0xFFu); (p)[3] = (uint8_t)(((v) >> 24) & 0xFFu); } while (0)

/* ==================== ⑥-1 上报分片 ====================
 * 整帧 91 字节 @115200 要发 7.9ms, 而一拍只有 20ms —— 一次发完等于每拍被砍掉四成时间。
 * 摊到 UP_CHUNKS 拍里发, 每拍只发一片(≈1.6ms)。
 * ⭐ UP_CHUNKS 由【上报周期 ÷ 主循环周期】自动算出: 以后改任何一边的频率, 片数自己跟着变。
 *    写死成 5 的话, 上层改了频率、下层还按老片数发, 节拍就悄悄错位了。 */
#define UP_CHUNKS      (ESP32_UP_PERIOD_MS / LOOP_PERIOD_MS)   /* 100/20 = 5 片 */
#define UP_CHUNK_BYTES (ESP32_UP_LEN / UP_CHUNKS)              /* 91/5   = 18 字节 */

/* 不整除就得改频率, 别让它静默错位; 片数比帧长还多会发出 0 字节的空片 */
#if (ESP32_UP_PERIOD_MS % LOOP_PERIOD_MS) != 0
#error "上报周期必须能被主循环周期整除, 否则'几拍发一帧'不是整数"
#endif
#if UP_CHUNKS > ESP32_UP_LEN
#error "分片数超过帧长, 会出现 0 字节的片"
#endif

/* ⚠️ 这两个必须跨拍活着: 组好的整帧不能随函数返回丢掉(所以放文件级, 不是局部变量) */
static uint8_t up_frame[ESP32_UP_LEN];   //整帧: 第 0 片那拍填满, 之后几拍接着发
static uint8_t up_chunk;                 //本拍发第几片(0 .. UP_CHUNKS-1)
static uint8_t s_torque_on;              //是否已发过"扭矩使能"(0x28=1): enable 跳变沿发一次

/* 本拍观测(59 维)。放文件级不占栈, 且 report_state 要回显其中的 cmd 段(审查 T3)。
 * 由主循环每拍 obs_get() 刷新 —— 即策略当拍看到的那一份, 避免"回显与实发不一致"。 */
static float s_obs[OBS_DIM];

/* IMU 静默检测状态(审查 M2): 连续无新帧的拍数 —— 详见 main 主循环 1b 段 */
static uint16_t s_imu_last_frames;
static uint8_t  s_imu_silent;

/* 观测差分速度的实际 dt(审查 L4): 上拍 obs_update 的 DWT 时刻 */
static uint32_t s_obs_last_cyc;

/* 组一次全量状态包, 摊在 UP_CHUNKS 拍里发完。主循环每拍调一次, 内部自己数拍到哪了。
 * ⚠️ 现在只有一部分字段是真数据, 标"待接"的还是 0(对应模块没做)。 */
static void report_state(void){
    float    v[3];
    uint32_t i;

    /* ---- 1、该组新包了吗: 每 UP_CHUNKS 拍组一次(up_chunk 归 0 时) ---- */
    if (up_chunk == 0u) {
        for (i = 4u; i < (uint32_t)ESP32_UP_LEN - 1u; i++) up_frame[i] = 0u;  //先清 0: 未接字段就是 0
        up_frame[0] = ESP32_UP_HEAD0;
        up_frame[1] = ESP32_UP_HEAD1;
        up_frame[2] = (uint8_t)ESP32_UP_PAYLOAD;
        up_frame[3] = ESP32_UP_TYPE_FULL;

        /* 注释里的 "载荷+x" 对应协议 (5.4) 的偏移; 帧内偏移 = 载荷偏移 + 4 */
        PUT32(&up_frame[4], sys_ms());                               //载荷+0  运行毫秒
        PUT32(&up_frame[8], g_dbg.loops);                            //载荷+4  主循环计数(临时取观测区)
        /*                                                           载荷+8/+9 FSM 状态/故障码 待接 */
        /* ⚠️ 回显用【策略当拍实际看到的 cmd 段】(s_obs), 不是原始指令(审查 T3) ——
         *    原始值未经 obs 的 ±0.5 钳位, 直接回显会与实发不一致, 排查时误导。 */
        PUT16(&up_frame[14], (uint16_t)(int16_t)(s_obs[OBS_OFF_CMD + 0u] * ESP32_DL_SCALE));  //载荷+10 指令回显 vx
        PUT16(&up_frame[16], (uint16_t)(int16_t)(s_obs[OBS_OFF_CMD + 1u] * ESP32_DL_SCALE));  //载荷+12 指令回显 wz
        imu_get_euler_deg(v);                                        //载荷+14 姿态(0.01 度)
        PUT16(&up_frame[18], (uint16_t)(int16_t)(v[0] * 100.0f));
        PUT16(&up_frame[20], (uint16_t)(int16_t)(v[1] * 100.0f));
        PUT16(&up_frame[22], (uint16_t)(int16_t)(v[2] * 100.0f));
        imu_get_gyro(v);                                             //载荷+20 角速度(0.01 rad/s)
        PUT16(&up_frame[24], (uint16_t)(int16_t)(v[0] * 100.0f));
        PUT16(&up_frame[26], (uint16_t)(int16_t)(v[1] * 100.0f));
        PUT16(&up_frame[28], (uint16_t)(int16_t)(v[2] * 100.0f));
        imu_get_grav(v);                                          //载荷+26 重力方向(×127 归一化)
        up_frame[30] = (uint8_t)(int8_t)(v[0] * 127.0f);
        up_frame[31] = (uint8_t)(int8_t)(v[1] * 127.0f);
        up_frame[32] = (uint8_t)(int8_t)(v[2] * 127.0f);
    for (i = 0; i < 12u; i++)                                    //载荷+29 舵机位置 ×12(临时取观测区)
        PUT16(&up_frame[33 + i * 2u], (uint16_t)(int16_t)g_dbg.pos[i]);
    {   int8_t pan, tilt;                                        //载荷+77 云台回显(帧内偏移 = 载荷偏移 + 4)
        mg90_get(&pan, &tilt);                                   //模拟舵机没反馈, 只能靠模块记着的设定值
        up_frame[81] = (uint8_t)pan;
        up_frame[82] = (uint8_t)tilt;
    }
    for (i = 0; i < 12u; i++)                                    //载荷+53 舵机温度 ×12(℃)
        up_frame[57 + i] = (uint8_t)g_dbg.temp[i];
    for (i = 0; i < 12u; i++)                                    //载荷+65 舵机状态位 ×12(位标志)
        up_frame[69 + i] = (uint8_t)g_dbg.status[i];
    /* 载荷+8/+9 FSM 状态与故障码、+79 电池电压 —— 待接, 现在都是 0 */

        {   uint8_t ck = 0;                                          //校验和: 从字节 0 累加(含帧头)
            for (i = 0; i < (uint32_t)ESP32_UP_LEN - 1u; i++) ck = (uint8_t)(ck + up_frame[i]);
            up_frame[ESP32_UP_LEN - 1u] = ck;
        }
    }

    /* ---- 2、把本拍那一片推出去 ----
     * 末片发到帧尾(吃掉除不尽的余数, 91 = 18×4 + 19), 前几片都是标准片长。 */
    {
        uint16_t off = (uint16_t)((uint16_t)up_chunk * UP_CHUNK_BYTES);
        uint16_t n   = (up_chunk == (uint8_t)(UP_CHUNKS - 1u))
                     ? (uint16_t)(ESP32_UP_LEN - off)      //末片: 一直发到帧尾
                     : (uint16_t)UP_CHUNK_BYTES;           //其余: 标准片长
        esp32_data_send(&up_frame[off], n);
    }

    /* ---- 3、记住下一拍轮到哪一片 ---- */
    up_chunk = (uint8_t)((up_chunk + 1u) % UP_CHUNKS);
}

/* ==================== 主函数 ==================== */
int main(void){
    sys_clock_init();       /* 时钟必须最先: 后面所有波特率都按它算 */
    sys_tick_init();        /* 时基: 从这一刻起 sys_ms() 开始走 */

    /* 先把观测区清干净: CCM 掉电不清, 不清的话读到的是上次运行的残留(比如 loops 显示几亿) */
    /* ⚠️ 顺序: 上轮的异常留痕要先"抢救"出来, 再清 —— 见下面 reset_cause 细化(审查 L5) */
    {   uint32_t fault_prev = g_dbg.fault_mark;

        memset(&g_dbg, 0, sizeof(g_dbg));
        g_dbg.magic  = DBG_MAGIC;
        g_dbg.fw_ver = DBG_FW_VER;   /* 一眼确认烧的是哪版 —— 见 cat_sys.h 的注释 */
        g_dbg.stage  = DBG_STAGE_INIT;

        safety_iwdg_init();     /* 看门狗最先起: 两种模式都要它兜底(1s 超时, 卡死自动复位) */
        g_dbg.reset_cause = safety_reset_cause();   /* 复位源: 上电 / IWDG 被救 / 软复位 */

        /* 异常留痕(审查 L5): 若本轮是 IWDG 复位 且 上轮留了异常 magic → 细化成 3
         * (=硬件异常跑飞被看门狗救回, 不是普通死循环)。只在 IWDG 复位时采信 ——
         * 冷上电 CCM 残留的旧 magic 不会被误当异常(那时 reset_cause=0)。 */
        g_dbg.fault_mark = fault_prev;      /* 先显示(供观测区查), 之后被下轮覆盖 */
        if (g_dbg.reset_cause == 1u && fault_prev == DBG_FAULT_MAGIC)
            g_dbg.reset_cause = 3u;
    }

    /* ---- 上电按键: KEY0(PE4) 按住 → 更新模式(不返回; END 后软复位回来) ---- */
    if (boot_key_held()) ota_mode_run();

    /* ---- 正常模式 ---- */
    servo_init();           /* 舵机  : USART6 @ PC6/PC7, 1Mbps */
    imu_init();             /* IMU   : USART2 @ PA2/PA3, 115200, 并设 100Hz */
    esp32_init();           /* 上位机: USART3 @ PB10/PB11, 115200(ESP32 只做透传桥) */
    mg90_init();            /* 云台  : TIM4 @ PB6/PB7 PWM, 上电回中 90 度(原行为, 云台不算"动猫") */
    obs_init();             /* 观测  : 各缓存置初值(关节角 = 默认站姿, 速度/力矩/动作 = 0) */
    vel_est_init();         /* 里程计: 清上拍足端/速度历史(首拍输出 0, 见 cat_vel_est.h) */
    /* 注: 以上 init 只"配置", 不发任何舵机位置/扭矩指令 —— 上电纯不动, 见文件头 ⚠️ */

    /* (看门狗与复位源已提前到开机最前 —— 见上; 这里不再重复) */

    /* ---- 上电等待: 等两边都就绪(最多 10 秒, 不卡死) ---- */
    g_dbg.stage = DBG_STAGE_BOOT;
    g_dbg.ms    = sys_ms();

    if (boot_wait_ready())
        g_dbg.stage = DBG_STAGE_READY;
    else
        g_dbg.stage = DBG_STAGE_TIMEOUT;    /* 没等到也往下走 */

    g_dbg.imu_rate_fix = imu_ensure_rate(); /* 模块确实醒了 → 实测帧率, 不是 100Hz 就补设 */

    g_dbg.servo_miss = servo_ping_all();    /* 记自检结果: 0 = 12 颗全在 */

    /* 舵机电压【阈值】(0x0E/0x0F) 是配置量、不是实时值 —— 开机读一次就够。
     * (2026-09-27: 原来这两项在主循环里每拍读, 是查"12 颗报 ERROR=0x01"的临时手段,
     *  问题已解决, 已从主循环删掉, 只留这一次开机读。) */
    {
        uint8_t one[2];
        if (servo_read(3u, SCS_ADDR_VOLT_MAX, 1u, one) != SCS_ERR_TIMEOUT)
            g_dbg.servo_vmax = one[0];
        if (servo_read(3u, SCS_ADDR_VOLT_MIN, 1u, one) != SCS_ERR_TIMEOUT)
            g_dbg.servo_vmin = one[0];
    }

    g_dbg.ms         = sys_ms();

    /* ---- 策略来源: 扫双槽选激活(开机一次; 更新成功后由 OTA 流程再调) ---- */
    policy_select();
    cat_polupd_set_active(policy_source(), policy_seq(), policy_tag(), policy_build_id());

    /* ==================== ⑦ 主循环 50Hz ====================
     * 节拍用"固定时刻"而不是"每拍末尾延时 20ms":
     *   末尾延时 → 周期 = 20ms + 本拍执行时间, 会越来越慢;
     *   固定时刻 → 下一拍的时刻 = 本拍时刻 + 20ms, 执行超了就直接进下一拍(不累积)。 */
    {
        uint32_t t_now = sys_ms();                  /* 本拍的起点时刻 */

        while (1) {
            uint8_t  buf[12u * 15u];                /* 12 颗 × 15 字节反馈块(0x38~0x46) */
            uint16_t counts[12];                    /* 各颗原始 12bit 位置计数 */
            uint16_t goal_counts[12];               /* 策略/动作算出的目标位置计数 */
            uint8_t  goal_buf[12u * 6u];            /* 目标字节流: 策略 2B/颗; 动作 6B/颗(0x2A位置+0x2C时间+0x2E速度) */
            float    act[ACT_DIM];                  /* 本拍策略动作(tanh 后, [-1,1]) */
            float    cur_abs[ACT_DIM];              /* 动作播放用: 本拍当前绝对关节角(播放起点/参考) */
            float    mot[ACT_DIM];                  /* 动作播放用: 播放器输出的目标绝对关节角 */
            uint8_t  act_go;                        /* 本拍是否走动作路径 */
            uint8_t  a_dirty, a_acc;                /* 动作帧参数: 是否变化 / 加速度档 */
            uint16_t a_tq, a_sp;                    /* 动作帧参数: 转矩档 / 速度档 */
            uint16_t miss;
            uint8_t  i;
            /* 分段计时: 单位 = DWT 周期数(CYCCNT), 168MHz 下 168 周期 = 1us。
             * DWT 已在 servo_init 里使能过(超时计时用的), 这里直接用, 不用额外初始化。 */
            uint32_t tc;
            const uint32_t t_loop0 = DWT->CYCCNT;       /* 整拍起点 */

            /* ---- 0、上位机: 排空缓冲解析指令 + 判失联(超时把指令归零) ---- */
            esp32_poll();
            {   /* 命令进观测(内部带 ±0.5 钳位; 失联时 esp32_poll 已把 vx/wz 归零) */
                const esp32_cmd_t *c = esp32_get_cmd();
                obs_set_cmd(c->vx, c->wz);
            }

            /* ---- 0b、0xCC 变长帧: 按【子类型】分派(策略 OTA / 动作加载), 处理 + 应答 ----
             * 号段约定: 0x10~0x3F = 动作加载(cat_action) / 0x70~0x74 = 策略在线更新(cat_polupd)。
             * ⚠️ 分派必须放编排层: v2 原先把所有 0xCC 直接喂 cat_polupd, 动作帧会撞它的 default。
             * 动作门卫 = "未使能 + 无故障" —— 站着不动才能换动作表(动作模块设计.md 4.1);
             * 由这里算好传进去(cat_action 不反向依赖 cat_esp32/cat_safety, 守分层铁律⑦)。
             * 策略 END 成功后立即重选(指针切换, 无需复位) —— 协议 7.8。 */
            {
                uint16_t var_n = 0u;
                const uint8_t *var = esp32_ota_fetch(&var_n);   /* 变长帧通道(接口名沿用旧称) */
                if (var) {
                    uint8_t resp[48], rtype = 0u, rlen = 0u;
                    uint8_t sub = var[0];
                    if (sub == ACT_SUB_BEGIN || sub == ACT_SUB_FRAME || sub == ACT_SUB_END) {
                        uint8_t gate_ok = (uint8_t)((safety_get_trip() == SAFETY_TRIP_NONE) &&
                                                    !esp32_get_cmd()->enable);
                        rtype = action_frame(var, var_n, gate_ok, resp, &rlen);
                    } else {
                        cat_polupd_frame(var, var_n, &rtype, resp, &rlen);
                        if (sub == POL_SUB_END && resp[0] == POL_RC_OK) {
                            policy_select();
                            cat_polupd_set_active(policy_source(), policy_seq(),
                                                  policy_tag(), policy_build_id());
                        }
                    }
                    if (rtype) esp32_send_up(rtype, resp, rlen);
                }
            }

            /* ---- 1、IMU: 排空缓冲解析, 取到的永远是最新一帧 ---- */
            tc = DWT->CYCCNT;
            imu_poll();
            imu_get_gyro(g_dbg.gyro);
            imu_get_grav(g_dbg.grav);
            imu_get_euler_deg(g_dbg.euler);
            g_dbg.imu_frames = imu_get_frames();
            g_dbg.t_imu = DWT->CYCCNT - tc;

            /* ---- 1b、IMU 静默检测(2026-10-03 审查 M2) ----
             * 三条数据链里只有 IMU 原先没有"死活检测": 断线时 gyro/gravity 会一直停在
             * 最后一帧, 策略拿半真半假的观测继续走。这里按"连续多少拍没新帧"判掉线。
             * ⚠️ 响应 = 停发新目标(舵机保持出力), 【不】卸力 —— 与 DEADMAN 同语义:
             *    卸力只留给安全故障。恢复出新帧自动清零, 无需人工。 */
            {   uint16_t f_now = imu_get_frames();
                if (f_now != s_imu_last_frames) s_imu_silent = 0u;
                else if (s_imu_silent < 250u)   s_imu_silent++;
                s_imu_last_frames = f_now;
                g_dbg.imu_silent  = s_imu_silent;
            }

            /* ---- 2、舵机: 一条指令读回 12 颗的 15 字节反馈块(0x38~0x46 一整套反馈) ---- */
            tc = DWT->CYCCNT;
            miss = servo_sync_read(SCS_ADDR_PRESENT_POS, SCS_FB_LEN, buf);
            g_dbg.t_sync = DWT->CYCCNT - tc;        /* 含等 12 颗应答(分 3 组) */
            g_dbg.sync_miss = miss;
            if (miss == 0u) {                       /* 12 颗齐了才更新, 免得半截数据 */
                int16_t cur_lsbs[12];
                uint8_t volt_lsbs[12];                  /* 电压 1B/颗(0x3E) */
                uint8_t temp_cs[12];                    /* 温度 1B/颗(0x3F) */
                for (i = 0; i < 12u; i++) {
                    const uint8_t *p = &buf[(uint16_t)i * SCS_FB_LEN];
                    counts[i]    = scs_u16(&p[SCS_FB_OFF_POS]);    /* 位置 */
                    cur_lsbs[i]  = scs_i16(&p[SCS_FB_OFF_CURR]);   /* ⚠️ 电流是有符号量 */
                    volt_lsbs[i] = p[SCS_FB_OFF_VOLT];
                    temp_cs[i]   = p[SCS_FB_OFF_TEMP];
                    g_dbg.pos[i]    = (float)counts[i];
                    g_dbg.temp[i]   = temp_cs[i];
                    g_dbg.status[i] = p[SCS_FB_OFF_STAT];
                }

                /* 2.1 安全保护: 电压/电流/温度 → 锁存故障(命中后由 2.5 每拍广播卸力) */
                g_dbg.safety_trip = safety_protect_poll(volt_lsbs, temp_cs, cur_lsbs);

                /* 2.2 喂观测管道: 位置→关节角/速度/linvel, 电流→tau
                 * ⚠️ dt 用【实际拍长】(DWT 差值), 不再写死 20ms(审查 L4):
                 *    拍被拉长时(同步读超时/长任务), 差分速度按名义 dt 算会出尖峰失真。
                 *    越界值(首拍/DWT 被清/异常长卡)退回名义周期, 别让单拍毁掉估计。 */
                tc = DWT->CYCCNT;
                {   uint32_t dcyc = tc - s_obs_last_cyc;
                    s_obs_last_cyc = tc;
                    obs_update(counts, (dcyc > (CPU_MHZ * 5000u) && dcyc < (CPU_MHZ * 100000u))
                                     ? (float)dcyc / (float)(CPU_MHZ * 1000000u)
                                     : (LOOP_PERIOD_MS / 1000.0f));
                }
                obs_set_tau_from_cur(cur_lsbs);

                /* 2.3 取本拍观测(s_obs), 挑几个关键段进观测区 —— 真机验证数据链的路子 */
                obs_get(s_obs);
                for (i = 0; i < 3u; i++)  g_dbg.linvel[i] = s_obs[OBS_OFF_LINVEL + i];
                for (i = 0; i < 12u; i++) {
                    g_dbg.jpos[i] = s_obs[OBS_OFF_JPOS + i];
                    g_dbg.tau[i]  = s_obs[OBS_OFF_TAU  + i];
                }
                g_dbg.t_obs = DWT->CYCCNT - tc;     /* obs_update + obs_get(里程计 FK 在这) */

                /* 2.4 目标位置生成: 【动作播放优先】→ 否则策略推理 → 舵机指令
                 * ⚠️ 这段必须在 if(miss==0) 里面: 读数不齐时 obs 还是旧的, 拿旧观测推理等于让猫瞎走。
                 * ⚠️ obs_set_lastact 必须在 policy_run【之后】调 —— 观测段要的是"上一拍"的动作。
                 * ⚠️ 更新中(cat_polupd_busy) / 无策略(policy_ready=0) → 跳过本段:
                 *    不推理、不发目标; 卸力由 2.5 路径统一做。 */
                act_go = action_playing() ? 1u : 0u;        /* 播放中? */
                if (!act_go) {
                    /* 没在播 → 看有没有"起播请求 + 门卫过"。当前绝对角 = obs 的 jpos 段 + 默认站姿。
                     * 门卫 = 未使能 + 无故障(与 0b 段 BEGIN/END 同一条件): 站着不动才能起播。 */
                    for (i = 0; i < ACT_DIM; i++)
                        cur_abs[i] = s_obs[OBS_OFF_JPOS + i] + CAT_POSE_DEF[i];
                    if (action_play_start_if_requested(cur_abs,
                            (uint8_t)((safety_get_trip() == SAFETY_TRIP_NONE) &&
                                      !esp32_get_cmd()->enable)))
                        act_go = 1u;
                }

                if (act_go) {
                    /* ---- 2.4a 动作播放(2026-10-03): 动作输出替代策略输出 ----
                     * 门卫只在【起播】那一刻要求"未使能"(站着不动才能换表/起播);
                     * 播放【中】只要求无故障 —— 动作是本地序列, 中途按"停止"不该把猫停在半路,
                     * 失联也不打断(播完自然回站姿)。故障 → abort(跳到出场段平滑回站姿),
                     * 卸力仍归 2.5 段 —— 播放器不参与卸力决策。 */
                    if (safety_get_trip() != SAFETY_TRIP_NONE) action_abort();  /* 故障: 平滑收尾 */

                    tc = DWT->CYCCNT;
                    (void)action_tick(mot);            /* 0=进行中 1=播完(下一拍 action_playing()=0 → 回策略) */
                    obs_rad_to_counts(mot, goal_counts);          /* 绝对角 → 计数(与 obs 同一份公式) */
                    g_dbg.pol_clip    = 0u;                       /* 播放中无"策略动作"概念 */
                    g_dbg.safety_clip = safety_clamp_counts(goal_counts);
                    for (i = 0; i < 12u; i++) g_dbg.pol_act[i] = 0.0f;
                    g_dbg.t_pol = DWT->CYCCNT - tc;

                    tc = DWT->CYCCNT;
                    if (safety_get_trip() == SAFETY_TRIP_NONE) {
                        /* 扭矩使能(与策略路径同一原则: 只开不关; 停止/故障都不主动卸) */
                        if (!s_torque_on) {
                            static const uint8_t torque_on[12] = { 1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u };
                            servo_sync_write(SCS_ADDR_TORQUE_EN, 1u, torque_on);
                            s_torque_on = 1u;
                        }
                        /* 帧运动参数: torque/acc 变了才写(相邻帧一样 → 零总线开销); speed 每拍搭车 */
                        a_dirty = action_cur_params(&a_tq, &a_sp, &a_acc);
                        if (a_dirty) {
                            for (i = 0; i < 12u; i++){        /* 0x30 转矩限制 2B/颗(12 颗同值) */
                                goal_buf[i * 2u]      = (uint8_t)(a_tq & 0xFFu);
                                goal_buf[i * 2u + 1u] = (uint8_t)(a_tq >> 8);
                            }
                            servo_sync_write(SCS_ADDR_TORQUE_LIMIT, 2u, goal_buf);
                            for (i = 0; i < 12u; i++) goal_buf[i] = a_acc;   /* 0x29 加速度 1B/颗 */
                            servo_sync_write(SCS_ADDR_ACCEL, 1u, goal_buf);
                        }
                        /* 目标位置 6B 连续写(与 v1 同规矩): 0x2A 位置 + 0x2C 时间(=0, 位置模式忽略)
                         * + 0x2E 速度 —— 速度档搭这趟车, 不额外占一次总线。 */
                        for (i = 0; i < 12u; i++) {
                            uint8_t *g = &goal_buf[(uint16_t)i * 6u];
                            g[0] = (uint8_t)(goal_counts[i] & 0xFFu);
                            g[1] = (uint8_t)(goal_counts[i] >> 8);
                            g[2] = 0u; g[3] = 0u;
                            g[4] = (uint8_t)(a_sp & 0xFFu);
                            g[5] = (uint8_t)(a_sp >> 8);
                        }
                        servo_sync_write(SCS_ADDR_GOAL_POS, 6u, goal_buf);
                    } else {
                        s_torque_on = 0u;   /* 故障: 下次重开(卸力由 2.5 段广播) */
                    }
                    g_dbg.t_send = DWT->CYCCNT - tc;
                } else if (cat_polupd_busy() || !policy_ready()) {
                    g_dbg.t_pol  = 0u;
                    g_dbg.t_send = 0u;
                } else {
                    tc = DWT->CYCCNT;
                    policy_run(s_obs, act);
                    obs_set_lastact(act);

                    {
                        uint8_t clipped = obs_act_to_counts(act, goal_counts);

                        g_dbg.pol_clip    = clipped;
                        /* 安全钳位: 第二道钳位, 逐颗夹进【机械行程表】(obs 那道只夹 0~4095 编码边界) */
                        g_dbg.safety_clip = safety_clamp_counts(goal_counts);
                        for (i = 0; i < 12u; i++) g_dbg.pol_act[i] = act[i];
                        g_dbg.t_pol = DWT->CYCCNT - tc;     /* policy_run + 动作→计数 + 安全钳位 */

                        tc = DWT->CYCCNT;
                        /* 安全闸: 只有"上位机在连 + 使能开 + 无故障 + IMU 有数据"才真发指令。
                         * 失联/急停时【不发新指令】= 舵机保持原位继续出力(协议 6.6)。
                         * ⚠️ "停止" ≠ "卸力": 失联/急停/IMU 静默绝不能去关舵机扭矩, 那是让猫
                         *    直接趴下; 唯一该卸力的情形是安全故障 —— 由 2.5 的故障广播统一做。
                         * ⚠️ IMU 静默(审查 M2)也挡在这: 姿态反馈没了还按策略走 = 睁眼瞎走。 */
                        if ((safety_get_trip() == SAFETY_TRIP_NONE) && esp32_alive() &&
                            esp32_get_cmd()->enable && (s_imu_silent < IMU_SILENT_LOOPS)) {
                            /* 使能跳变沿(首次 enable): 给 12 颗写一次 0x28=1 开扭矩(2026-09-30 定)。
                             * "上电纯不动"不变: 只有上位机主动 enable 才开; 停止(enable=0)不卸、保持扭矩。 */
                            if (!s_torque_on) {
                                static const uint8_t torque_on[12] = { 1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u };
                                servo_sync_write(SCS_ADDR_TORQUE_EN, 1u, torque_on);
                                s_torque_on = 1u;
                            }
                            for (i = 0; i < 12u; i++) {          /* 目标位置是小端 2 字节 */
                                goal_buf[i * 2u]      = (uint8_t)(goal_counts[i] & 0xFFu);
                                goal_buf[i * 2u + 1u] = (uint8_t)(goal_counts[i] >> 8);
                            }
                            servo_sync_write(SCS_ADDR_GOAL_POS, 2u, goal_buf);
                        } else {
                            s_torque_on = 0u;   /* 失联/失能/故障 → 下次 enable 重新开(故障卸力由 2.5 段广播) */
                        }
                        g_dbg.t_send = DWT->CYCCNT - tc;    /* 含安全闸判断; 不使能时≈0 */
                    }
                }
            }

            /* ---- 2.5 安全故障 / 更新态: 卸力广播(每拍重发) ----
             * 任一保护触发(锁存)后 → 每拍广播"扭矩使能=0", 不自动恢复、不等确认。
             * 2026-09-29 扩: 策略在线更新中 / 无策略 也走这条(卸力等 OTA 或重新使能)。
             * ⚠️ 放在读数判断【外面】: 触发后即使某拍读数残缺, 也继续重发, 不依赖后续反馈。
             * ⚠️ 优先于失联闸: 故障时连"保持出力"都不允许 —— 卸力是保护命中后的直接动作。 */
            if (safety_get_trip() != SAFETY_TRIP_NONE ||
                cat_polupd_busy() || !policy_ready()) {
                static const uint8_t torque_off[12] = { 0 };
                servo_sync_write(SCS_ADDR_TORQUE_EN, 1u, torque_off);
            }

            /* ---- 2.6 云台: 独立通道, 不受任何状态限制(2026-09-30 煜辰定) ----
             * 只认帧[7]/[8](pan/tilt): 与"在连/使能/维护/故障"全解耦 —— 收到什么跟什么。
             * 失联时自然"保持最后角度": pan/tilt 不参与 DEADMAN 归零(cat_esp32.c),
             *   这里每拍重写同样的值 = 保持; 固件【永不主动回中】(协议 6.6)。 */
            {
                const esp32_cmd_t *c = esp32_get_cmd();
                mg90_set(c->pan, c->tilt);
            }

            /* ---- 3、单颗诊断量: 全部从【已经读到的 15 字节块】里取 ----
             * 原来这里是 4 次单颗 servo_read(查"12 颗都报 ERROR=0x01 电压位"的临时手段),
             * 那个问题已解决(现在 lasterr=0、电压 8.0V 在阈值 9.0V 内), 而每拍 4 次单颗读
             * 要花 ~2-4ms —— 主循环正是被它拖到 32ms(应 20ms)。
             * 改从同一块缓冲里取: 零额外开销, 诊断量一个不少。
             * ⚠️ 只在 miss==0 时更新(2026-10-03 审查 L1): 同步读失败时 buf 是【未初始化】的
             *    栈内存, 照读等于把栈垃圾当电压/状态显示 —— 会把人往错方向带(09-25 就吃过假值的亏)。
             *    宁可显示上一次的有效值, 也不显示垃圾。 */
            if (miss == 0u) {
                g_dbg.servo_vnow    = buf[2u * SCS_FB_LEN + SCS_FB_OFF_VOLT];  /* ID3 当前电压 */
                g_dbg.servo_lasterr = buf[2u * SCS_FB_LEN + SCS_FB_OFF_STAT];  /* ID3 状态位 */
            }
            g_dbg.servo_err     = 0u;   /* 单颗路径已不再跑, 留 0 占位(块里没有单颗 err) */

            /* ---- 4、状态 ---- */
            g_dbg.ms      = sys_ms();
            g_dbg.loops++;
            g_dbg.pol_src = policy_source();        /* 策略来源(在线更新探针) */
            g_dbg.pol_seq = policy_seq();
            g_dbg.act_frames = action_load_count(); /* 动作表帧数(0 = 无表) */
            {   uint8_t st, ix;                     /* 动作播放进度(空闲/进场/帧/出场 + 帧号) */
                action_progress(&st, &ix);
                g_dbg.act_stage = st;
                g_dbg.act_idx   = ix;
            }

            /* ---- 5、上报(10Hz): 每拍只发一片, 内部自己数拍到第几片了 ---- */
            tc = DWT->CYCCNT;
            report_state();
            g_dbg.t_up = DWT->CYCCNT - tc;          /* 含串口阻塞发送 */

            g_dbg.t_all = DWT->CYCCNT - t_loop0;    /* 整拍【忙碌】耗时(不含下面睡眠) */

            /* ---- 6、等下一拍: 忙等到期 → 睡眠(WFI) ----
             * 睡下后: 外设全在跑(串口收发中断照常进), SysTick 每 1ms 中断把内核唤醒,
             * 醒了回到这里再查是否到点 —— 没到就继续睡(睡眠模式=只停内核, 依据见文件头)。
             * ⚠️ 喂狗放在睡之前: 本拍活干完了才续命; 哪段卡死超 1s → IWDG 复位兜底。 */
            safety_iwdg_feed();
            while (!sys_elapsed(t_now, LOOP_PERIOD_MS)) __WFI();
            t_now += LOOP_PERIOD_MS;
        }
    }
}
