/* ============================================================
 *  main.c — 主程序(只负责"编排": 按顺序调各模块, 不装实现)
 *
 *  当前进度: ①②③④⑤⑥⑦ 都已完成; ⑧ 已接线(待真机)
 *
 *    ① sys_clock_init()            系统时钟 168MHz        ← cat_sys
 *    ② sys_tick_init()             毫秒时基              ← cat_sys
 *    ③ servo_init() / imu_init()   外设初始化            ← 各自模块
 *    ④ boot_wait_ready()           上电等待: 等两边就绪    ← 本文件(跨模块, 属编排)
 *    ⑤ 观测区 g_dbg                真机调试窗口           ← 结构体在 cat_sys.h
 *    ⑥ report_state()              状态上报 10Hz(分片发)   ← 本文件(跨模块, 属编排)
 *    ⑦ 主循环                      50Hz                  ← 已完成
 *    ⑧ 安全闸: 目标钳位 + 反馈保护(故障卸力)    ← cat_safety
 *
 *  约定: 配置/实现都归各自模块(cat_servo / cat_imu / cat_sys), 这个文件只排顺序。
 *        跨模块的启动流程(如 ④)也算"编排", 放这里 —— 别塞进某个模块, 否则它会反过来依赖别人。
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
#include "cat_safety.h"

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

static uint8_t boot_wait_ready(void){
    uint32_t t0 = sys_ms();
    uint16_t last_frames = imu_get_frames();        /* 记上轮帧数, 用来判"有没有新数据" */
    uint8_t  ok_rounds = 0;

    while (!sys_elapsed(t0, BOOT_WAIT_MS)) {
        uint8_t round_ok = 0;

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

/* 组一次全量状态包, 摊在 UP_CHUNKS 拍里发完。主循环每拍调一次, 内部自己数拍到哪了。
 * ⚠️ 现在只有一部分字段是真数据, 标"待接"的还是 0(对应模块没做)。 */
static void report_state(void){
    const esp32_cmd_t *c;
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
        c = esp32_get_cmd();
        PUT16(&up_frame[14], (uint16_t)(int16_t)(c->vx * ESP32_DL_SCALE));  //载荷+10 指令回显 vx
        PUT16(&up_frame[16], (uint16_t)(int16_t)(c->wz * ESP32_DL_SCALE));  //载荷+12 指令回显 wz
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
    memset(&g_dbg, 0, sizeof(g_dbg));
    g_dbg.magic  = DBG_MAGIC;
    g_dbg.fw_ver = DBG_FW_VER;   /* 一眼确认烧的是哪版 —— 见 cat_sys.h 的注释 */
    g_dbg.stage  = DBG_STAGE_INIT;

    servo_init();           /* 舵机  : USART6 @ PC6/PC7, 1Mbps */
    imu_init();             /* IMU   : USART2 @ PA2/PA3, 115200, 并设 100Hz */
    esp32_init();           /* 上位机: USART3 @ PB10/PB11, 115200(ESP32 只做透传桥) */
    mg90_init();            /* 云台  : TIM4 @ PB6/PB7 PWM, 上电回中 90 度 */
    obs_init();             /* 观测  : 各缓存置初值(关节角 = 默认站姿, 速度/力矩/动作 = 0) */

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

    /* ==================== ⑦ 主循环 50Hz ====================
     * 节拍用"固定时刻"而不是"每拍末尾延时 20ms":
     *   末尾延时 → 周期 = 20ms + 本拍执行时间, 会越来越慢;
     *   固定时刻 → 下一拍的时刻 = 本拍时刻 + 20ms, 执行超了就直接进下一拍(不累积)。 */
    {
        uint32_t t_now = sys_ms();                  /* 本拍的起点时刻 */

        while (1) {
            uint8_t  buf[12u * 15u];                /* 12 颗 × 15 字节反馈块(0x38~0x46) */
            uint16_t counts[12];                    /* 各颗原始 12bit 位置计数 */
            uint16_t goal_counts[12];               /* 策略算出的目标位置计数 */
            uint8_t  goal_buf[12u * 2u];            /* 目标位置的小端字节流(喂 sync_write) */
            float    act[ACT_DIM];                  /* 本拍策略动作(tanh 后, [-1,1]) */
            uint16_t miss;
            uint8_t  i;
            /* 本拍观测(59 维)。放 static 省栈: 主循环栈上已经有 buf(180)+counts(24) 了 */
            static float obs[OBS_DIM];
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

            /* ---- 1、IMU: 排空缓冲解析, 取到的永远是最新一帧 ---- */
            tc = DWT->CYCCNT;
            imu_poll();
            imu_get_gyro(g_dbg.gyro);
            imu_get_grav(g_dbg.grav);
            imu_get_euler_deg(g_dbg.euler);
            g_dbg.imu_frames = imu_get_frames();
            g_dbg.t_imu = DWT->CYCCNT - tc;

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

                /* 2.2 喂观测管道: 位置→关节角/速度/linvel, 电流→tau */
                tc = DWT->CYCCNT;
                obs_update(counts, LOOP_PERIOD_MS / 1000.0f);
                obs_set_tau_from_cur(cur_lsbs);

                /* 2.3 取本拍观测, 挑几个关键段进观测区 —— 真机验证数据链的路子 */
                obs_get(obs);
                for (i = 0; i < 3u; i++)  g_dbg.linvel[i] = obs[OBS_OFF_LINVEL + i];
                for (i = 0; i < 12u; i++) {
                    g_dbg.jpos[i] = obs[OBS_OFF_JPOS + i];
                    g_dbg.tau[i]  = obs[OBS_OFF_TAU  + i];
                }
                g_dbg.t_obs = DWT->CYCCNT - tc;     /* obs_update + obs_get(里程计 FK 在这) */

                /* 2.4 策略推理 → 目标位置 → 舵机指令
                 * ⚠️ 这段必须在 if(miss==0) 里面: 读数不齐时 obs 还是旧的, 拿旧观测推理等于让猫瞎走。
                 * ⚠️ obs_set_lastact 必须在 policy_run【之后】调 —— 观测段要的是"上一拍"的动作。 */
                tc = DWT->CYCCNT;
                policy_run(obs, act);
                obs_set_lastact(act);

                {
                    uint8_t clipped = obs_act_to_counts(act, goal_counts);

                    g_dbg.pol_clip    = clipped;
                    /* 安全钳位: 第二道钳位, 逐颗夹进【机械行程表】(obs 那道只夹 0~4095 编码边界) */
                    g_dbg.safety_clip = safety_clamp_counts(goal_counts);
                    for (i = 0; i < 12u; i++) g_dbg.pol_act[i] = act[i];
                    g_dbg.t_pol = DWT->CYCCNT - tc;     /* policy_run + 动作→计数 + 安全钳位 */

                    tc = DWT->CYCCNT;
                    /* 安全闸: 只有"上位机在连 + 使能开 + 无故障"才真发指令。
                     * 失联/急停时【不发新指令】= 舵机保持原位继续出力(协议 6.6)。
                     * ⚠️ "停止" ≠ "卸力": 失联/急停绝不能去关舵机扭矩, 那是让猫直接趴下;
                     *    唯一该卸力的情形是安全故障 —— 由 2.5 的故障广播统一做。 */
                    if ((safety_get_trip() == SAFETY_TRIP_NONE) && esp32_alive() && esp32_get_cmd()->enable) {
                        for (i = 0; i < 12u; i++) {          /* 目标位置是小端 2 字节 */
                            goal_buf[i * 2u]      = (uint8_t)(goal_counts[i] & 0xFFu);
                            goal_buf[i * 2u + 1u] = (uint8_t)(goal_counts[i] >> 8);
                        }
                        servo_sync_write(SCS_ADDR_GOAL_POS, 2u, goal_buf);
                    }
                    g_dbg.t_send = DWT->CYCCNT - tc;    /* 含安全闸判断; 不使能时≈0 */
                }
            }

            /* ---- 2.5 安全故障: 卸力广播(每拍重发) ----
             * 任一保护触发(锁存)后 → 每拍广播"扭矩使能=0", 不自动恢复、不等确认。
             * ⚠️ 放在读数判断【外面】: 触发后即使某拍读数残缺, 也继续重发, 不依赖后续反馈。
             * ⚠️ 优先于失联闸: 故障时连"保持出力"都不允许 —— 卸力是保护命中后的直接动作。 */
            if (safety_get_trip() != SAFETY_TRIP_NONE) {
                static const uint8_t torque_off[12] = { 0 };
                servo_sync_write(SCS_ADDR_TORQUE_EN, 1u, torque_off);
            }

            /* ---- 2.6 云台: 只在"在连 + 使能开"时才跟指令动 ----
             * 协议(6.6): 失联/急停/维护时【保持不动, 不回中】。
             * ⚠️ 故障【不】保持 —— 故障后要靠云台保留观察能力, 照常跟指令动。
             *   这里不调用就是"保持" —— 别写成 else 里回中(猫躺地时相机被甩很危险)。 */
            {
                const esp32_cmd_t *c = esp32_get_cmd();
                if (esp32_alive() && c->enable) mg90_set(c->pan, c->tilt);
            }

            /* ---- 3、单颗诊断量: 全部从【已经读到的 15 字节块】里取 ----
             * 原来这里是 4 次单颗 servo_read(查"12 颗都报 ERROR=0x01 电压位"的临时手段),
             * 那个问题已解决(现在 lasterr=0、电压 8.0V 在阈值 9.0V 内), 而每拍 4 次单颗读
             * 要花 ~2-4ms —— 主循环正是被它拖到 32ms(应 20ms)。
             * 改从同一块缓冲里取: 零额外开销, 诊断量一个不少。 */
            g_dbg.servo_vnow    = buf[2u * SCS_FB_LEN + SCS_FB_OFF_VOLT];  /* ID3 当前电压 */
            g_dbg.servo_lasterr = buf[2u * SCS_FB_LEN + SCS_FB_OFF_STAT];  /* ID3 状态位 */
            g_dbg.servo_err     = 0u;   /* 单颗路径已不再跑, 留 0 占位(块里没有单颗 err) */

            /* ---- 4、状态 ---- */
            g_dbg.ms = sys_ms();
            g_dbg.loops++;

            /* ---- 5、上报(10Hz): 每拍只发一片, 内部自己数拍到第几片了 ---- */
            tc = DWT->CYCCNT;
            report_state();
            g_dbg.t_up = DWT->CYCCNT - tc;          /* 含串口阻塞发送 */

            g_dbg.t_all = DWT->CYCCNT - t_loop0;    /* 整拍【忙碌】耗时(不含下面等待) */

            /* ---- 6、等下一拍 ---- */
            while (!sys_elapsed(t_now, LOOP_PERIOD_MS));
            t_now += LOOP_PERIOD_MS;
        }
    }
}
