/* ============================================================
 *  cat_servo.c — STS3215 总线舵机驱动
 *
 *  分层(本文件按此顺序排列, 新函数插到对应区块, 别追加到末尾):
 *    ① 初始化   servo_init()
 *    ② 收发     servo_data_send() / servo_data_recv()
 *    ③ 包组装   servo_checksum() / servo_com_pack_set() / servo_tx_frame_build() / servo_rx_frame_check() / servo_last_status()
 *    ④ 指令     servo_ping_all() / servo_read() / servo_write() / servo_sync_read() / servo_sync_write()
 *               servo_reset() / servo_recovery()
 *
 *  命名: 发送侧 tx_ 前缀, 接收侧 rx_ 前缀, 不混用。
 *  设计依据 / 实测数据 / 踩坑历史 → 舵机驱动说明.md
 * ============================================================ */
#include "cat_servo.h"      /* 自己的头文件用双引号, 编译器才会找同目录 */
#include "cat_config.h"     /* 系统级配置(时钟等硬件常量) */

/* ---------- 全局变量 ---------- */
uint8_t id;     //本次指令包的目标 ID(0~253, 254=广播)。servo_com_pack_set() 写入, servo_tx_frame_build() 取用
static command_packet_t com_pack;
static uint8_t tx_frame[BUFFER];    //待发送的整帧(含字头/ID/LEN/指令/参数/校验和)
static uint16_t tx_frame_len;       //tx_frame 里的实际字节数(用 16 位: 免得 BUFFER 再调时被顶溢出)
static uint8_t rx_frame[BUFFER];    //收到的应答帧
static uint8_t rx_last_status;      //上次应答里的 ERROR 字节(servo_last_status() 的出口)

/* ==================== ① 初始化 ==================== */
void servo_init(void){
    // 1、打开时钟: GPIOC(挂 AHB1) + USART6(挂 APB2)
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOCEN;
    RCC->APB2ENR |= RCC_APB2ENR_USART6EN;

    // 2、PC6/PC7 配成 USART6 的复用脚(AF8): 复用 + 推挽 + 高速 + 无上下拉
    GPIOC->MODER = (GPIOC->MODER & ~((3u << 12) | (3u << 14)))
                 | (2u << 12) | (2u << 14);
    GPIOC->OTYPER  &= ~((1u << 6) | (1u << 7));
    GPIOC->OSPEEDR |=  (3u << 12) | (3u << 14);
    GPIOC->PUPDR   &= ~((3u << 12) | (3u << 14));
    GPIOC->AFR[0] = (GPIOC->AFR[0] & ~((0xFu << 24) | (0xFu << 28)))
                  | (8u << 24) | (8u << 28);

    // 3、USART6: 1Mbps 8N1。BRR = 84MHz / 1Mbps = 84(整数, 零误差)
    USART6->BRR = (APB2_MHZ * 1000000u + 500000u) / 1000000u;
    USART6->CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;
    USART6->CR2 = 0;
    USART6->CR3 = 0;

    // 4、DWT 周期计数器: 收字节时用它做微秒级超时(168MHz → 1us = 168 个周期)
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;     // 打开跟踪总闸
    DWT->CYCCNT = 0;                                    // 计数清零
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;                // 启动计数
}

/* ==================== ② 收发 ==================== */
/* 把 tx_len 个字节依次送进 USART6。调用前必须先 servo_init()。
 * 两个 while 为何不加超时 → 说明.md(8.1) */
void servo_data_send(const uint8_t *tx_buf, uint16_t tx_len){
    while (USART6->SR & USART_SR_RXNE) (void)(USART6->DR & 0xFF);   //发前清残留, 否则会让本次应答错位
    for (uint16_t i = 0; i < tx_len; i++){
        while (!(USART6->SR & USART_SR_TXE));   // 等发送寄存器空, 才能放下一个字节
        USART6->DR = tx_buf[i];
    }
    while (!(USART6->SR & USART_SR_TC));        // 等最后一字节真正发完, 再返回
}

/* 收 rx_len 个字节到 rx_buf; 整串共用 SCS_TMO_US 一份预算, 返回 SCS_OK / SCS_ERR_TIMEOUT。
 * 回绕安全的计时写法 → 说明.md(8.2) */
static scs_err_t servo_data_recv(uint8_t *rx_buf, uint16_t rx_len){
    const uint32_t limit = SCS_TMO_US * CPU_MHZ;    //微秒 → 周期数(1us = 168 周期)
    const uint32_t t0 = DWT->CYCCNT;                //整串只记一次起点
    for (uint16_t i = 0; i < rx_len; i++){
        while (!(USART6->SR & USART_SR_RXNE)){      // 等收到一个字节
            if ((DWT->CYCCNT - t0) >= limit) return SCS_ERR_TIMEOUT;
        }
        rx_buf[i] = (uint8_t)(USART6->DR & 0xFF);   // 读 DR 会自动清 RXNE
    }
    return SCS_OK;
}

/* ==================== ③ 包组装 ==================== */
/* SCS 校验和 = ~(buf 起 length 个字节之和), 取最低字节。指令包与应答包同一公式。
 * 参与范围: 从 ID 起到最后一个参数/数据, 不含两个 0xFF 字头, 也不含校验和自己。 */
uint8_t servo_checksum(const uint8_t *buf, uint16_t length){
    uint16_t sum = 0;
    for (uint16_t i = 0; i < length; i++) sum = (uint16_t)(sum + buf[i]);
    return (uint8_t)(~sum);
}

/* 备好一条指令包: 设 ID / 指令码 / 参数, 并自动算 length = count + 2。
 * 无参数指令(如 PING)传 count = 0, param 传 0。
 * 返回 SCS_ERR_PARAM = 参数超容量; 此时包【没被改动】, 调用方别拿去发。 */
scs_err_t servo_com_pack_set(uint8_t sid, uint8_t inst, const uint8_t *param, uint8_t count){
    if (count > BUFFER - SCS_FRAME_FIXED) return SCS_ERR_PARAM;   //放不下就拒绝, 否则会越界写
    id = sid;
    com_pack.instruction = inst;
    for (uint8_t i = 0; i < count; i++) com_pack.parameter[i] = param[i];
    com_pack.length = (uint8_t)(count + 2);     //LEN = 参数个数 + 2
    return SCS_OK;
}

void servo_tx_frame_build(void){      //指令包构建函数
    rx_last_status = 0;         //要发新指令了 → 上次的舵机状态作废(免得这次超时时残留、误导调用方)
    uint16_t t = 0;
    tx_frame[t++] = SCS_HEADER1;
    tx_frame[t++] = SCS_HEADER2;
    tx_frame[t++] = id;
    tx_frame[t++] = com_pack.length;
    tx_frame[t++] = com_pack.instruction;
    for(int i = 0;i < com_pack.length-2;i++)tx_frame[t++] = com_pack.parameter[i];
    //校验和就地算: 此刻 tx_frame[2..t-1] 正好是 ID/LEN/指令/参数 这一段连续字节
    tx_frame[t] = servo_checksum(&tx_frame[2], (uint8_t)(t - 2));
    t++;
    tx_frame_len = t;       //记下帧长, 发送时直接用它(帧长 = com_pack.length + 4)
}

/* ---------- 应答帧校验 ---------- */
/* 校验一帧应答(rx_len = 整帧总字节数), 依次查: 长度 / LEN 字段 / 字头 / ID / 校验和 / ERROR。
 * 舵机自报错误的返回 SCS_ERR_SERVO, 具体错误码读 rx_buf[4]。 */
scs_err_t servo_rx_frame_check(const uint8_t *rx_buf, uint16_t rx_len, uint8_t sid){
    rx_last_status = 0;                 //默认清零: 帧没走到下面时就别让上次的值残留、误导调用方
    if (rx_len < SCS_RESP_FIXED) return SCS_ERR_LEN;   //比最短应答还短, 不是合法帧(也挡住下面 rx_len-3 回绕)
    if (rx_buf[0] != SCS_HEADER1 || rx_buf[1] != SCS_HEADER2) return SCS_ERR_HEADER;   //字头不对
    if (rx_buf[2] != sid) return SCS_ERR_ID;                                           //不是这颗在答应
    //LEN 字段应等于"数据个数 + 2"; 本帧数据个数 = rx_len - SCS_RESP_FIXED
    if (rx_buf[3] != (uint8_t)(rx_len - SCS_RESP_FIXED + 2)) return SCS_ERR_LEN;
    //校验和: 参与计算的是 ID 起、到数据末尾(rx_len-3 个), 不含末尾的 CK 自己
    if (servo_checksum(&rx_buf[2], (uint16_t)(rx_len - 3)) != rx_buf[rx_len - 1]) return SCS_ERR_CHECKSUM;
    rx_last_status = rx_buf[4];         //校验和过了, ERROR 字节才可信 → 记下来
    if (rx_last_status != 0) return SCS_ERR_SERVO;
    return SCS_OK;
}

/* 上次应答里的 ERROR 字节(位标志, 见 SCS_ST_*)。rx_frame 是 static, 上层拿不到那帧,
 * 这里是补的口子: SCS_ERR_SERVO 时用它知道到底报的什么错。
 * ⚠️ 只对【单颗】操作有意义 —— sync_read 会连验 12 帧, 这儿只留最后一帧的;
 *    想要按颗的状态, 去读观测块偏移 9(0x41 状态位)。 */
uint8_t servo_last_status(void){
    return rx_last_status;
}

/* ==================== ④ 指令 ==================== */
/* 依次查询全部 SERVO_COUNT 颗(上电自检)。
 * 返回"没应答或报错"的位图: bit0=ID1 … bit11=ID12; 0 = 全部在线且正常。
 * 逐颗查而非广播: 广播时多颗会同时答应, 在总线上互相撞车。 */
uint16_t servo_ping_all(void){
    uint16_t miss = 0;
    for (uint8_t sid = 1; sid <= SERVO_COUNT; sid++){
        if (servo_com_pack_set(sid, SCS_PING, 0, 0) != SCS_OK){   //PING 无参数(count=0, 理论不失败)
            miss |= (uint16_t)(1u << (sid - 1));
            continue;
        }
        servo_tx_frame_build();
        servo_data_send(tx_frame, tx_frame_len);
        if (servo_data_recv(rx_frame, SCS_RESP_FIXED) != SCS_OK){                    //PING 应答固定 6 字节
            miss |= (uint16_t)(1u << (sid - 1));                        //超时: 这颗没答
            continue;
        }
        if (servo_rx_frame_check(rx_frame, SCS_RESP_FIXED, sid) != SCS_OK) miss |= (uint16_t)(1u << (sid - 1));
    }
    return miss;
}

/* 从 addr 起读 count 个字节到 data_buf(低字节在前)。 */
scs_err_t servo_read(uint8_t sid, uint8_t addr, uint8_t count, uint8_t *data_buf){
    uint8_t   param[2] = {addr, count};
    uint16_t  rx_len = (uint16_t)(SCS_RESP_FIXED + count);  //应答总长 = 固定开销 + 数据
    scs_err_t err;
    if (count == 0 || count > BUFFER - SCS_RESP_FIXED) return SCS_ERR_PARAM;   //0 无意义; 上限由接收缓冲定(应答整帧 6+count 要装得下)
    err = servo_com_pack_set(sid, SCS_READ, param, 2);
    if (err != SCS_OK) return err;
    servo_tx_frame_build();
    servo_data_send(tx_frame, tx_frame_len);
    err = servo_data_recv(rx_frame, rx_len);
    if (err != SCS_OK) return err;
    err = servo_rx_frame_check(rx_frame, rx_len, sid);
    /* ⚠️ 即使返回 SCS_ERR_SERVO(舵机自报错误位), 数据本身仍是有效的(校验和已过), 所以照样拷出去,
     *    再由返回值告诉调用方"这颗报错了"。不这样写的话: 一旦某颗带错误位应答(如电压/过热),
     *    调用方读到的永远是旧值 —— 我们真踩过这个: 12 颗全报电压位, 电压值一个都读不出来。 */
    if (err == SCS_OK || err == SCS_ERR_SERVO) {
        for (uint8_t i = 0; i < count; i++) data_buf[i] = rx_frame[SCS_RESP_DATA + i];  //数据段
    }
    return err;
}

/* 从 addr 起写 count 个字节(data_buf)。
 * sid 传 SCS_BROADCAST 即广播: 都执行但【都不应答】, 直接返回 SCS_OK(不会去等应答)。 */
scs_err_t servo_write(uint8_t sid, uint8_t addr, const uint8_t *data_buf, uint8_t count){
    uint8_t param[BUFFER - 6];      //参数 = 1 个地址 + count 个数据
    if (count == 0 || count > BUFFER - SCS_FRAME_FIXED - 1) return SCS_ERR_PARAM;   //0 无意义; 参数区里再留 1 个给首地址
    param[0] = addr;
    for (uint8_t i = 0; i < count; i++) param[1 + i] = data_buf[i];
    {   //参数个数 = count + 1; 若超限, 包没被改动, 不能拿去发
        scs_err_t err = servo_com_pack_set(sid, SCS_WRITE, param, (uint8_t)(count + 1));
        if (err != SCS_OK) return err;
    }
    servo_tx_frame_build();
    servo_data_send(tx_frame, tx_frame_len);
    if (sid == SCS_BROADCAST) return SCS_OK;        //广播无应答, 不能去等
    {   //非广播: 应答固定 6 字节
        scs_err_t err = servo_data_recv(rx_frame, SCS_RESP_FIXED);
        if (err != SCS_OK) return err;
    }
    return servo_rx_frame_check(rx_frame, SCS_RESP_FIXED, sid);
}

/* 一条指令读回全部 SERVO_COUNT 颗(内部分组), 按【观测序 = ID-1】填入 data_buf, 每颗占 count 字节。
 * data_buf 需备 SERVO_COUNT × count 字节; 没应答的颗其数据段保持 0(不残留上次的值)。
 * 返回缺失位图: bit0=ID1 … bit11=ID12; 0 = 全读到(没答 / 帧坏 / 舵机自报错误 都算缺失)。
 * 为何分组、为何按 ID 认领而非"第 k 帧=第 k 颗" → 说明.md(四)(8.4) */
uint16_t servo_sync_read(uint8_t addr, uint8_t count, uint8_t *data_buf){
    uint8_t  param[2 + SCS_SYNC_GROUP];     //首地址 + 读长度 + 本组各 ID
    uint16_t rx_len = (uint16_t)(SCS_RESP_FIXED + count);
    uint16_t got = 0;                       //收到应答的颗(位图)
    uint16_t idx;
    uint8_t  g, i, k, n;

    if (count == 0 || count > BUFFER - SCS_RESP_FIXED) return 0x0FFFu;   //长度不合法 → 全缺
    for (idx = 0; idx < (uint16_t)SERVO_COUNT * count; idx++) data_buf[idx] = 0;

    for (g = 0; g < SERVO_COUNT; g = (uint8_t)(g + SCS_SYNC_GROUP)){
        n = (uint8_t)((SERVO_COUNT - g > SCS_SYNC_GROUP) ? SCS_SYNC_GROUP : (SERVO_COUNT - g));
        param[0] = addr;
        param[1] = count;
        for (i = 0; i < n; i++) param[2 + i] = (uint8_t)(g + 1 + i);    //ID 依次 1..SERVO_COUNT
        if (servo_com_pack_set(SCS_BROADCAST, SCS_SYNC_READ, param, (uint8_t)(n + 2)) != SCS_OK) continue;

        servo_tx_frame_build();
        servo_data_send(tx_frame, tx_frame_len);

        /* 收本组应答: 最多 n 帧。超时 → 跳出; 坏帧 → 丢它继续读下一帧, 不整组放弃 */
        for (k = 0; k < n; k++){
            uint8_t  sid;
            scs_err_t cerr;
            if (servo_data_recv(rx_frame, rx_len) != SCS_OK) break;   //超时
            sid = rx_frame[2];
            /* 传 rx_frame[2] 当 sid ⇒ ID 那项恒过, 等于白拿 LEN + 校验和 + ERROR 三项检查;
             * 任一不过就丢帧 —— 翻转的位若不挡, 会静默污染填进 data_buf 的观测数据 */
            cerr = servo_rx_frame_check(rx_frame, rx_len, sid);
            if (cerr == SCS_ERR_SERVO) {
                /* 舵机自报错误位: 帧本身是好的(校验和过了), 数据也有效 → 照样填进 data_buf,
                 * 但【不】置 got 位: servo_miss 的语义保持"这颗没能正常回应", 不把错误掩盖掉。 */
                for (i = 0; i < n; i++){
                    if (param[2 + i] == sid){
                        for (uint8_t b = 0; b < count; b++)
                            data_buf[(uint16_t)(sid - 1) * count + b] = rx_frame[SCS_RESP_DATA + b];
                        break;
                    }
                }
                continue;
            }
            if (cerr != SCS_OK) continue;
            for (i = 0; i < n; i++){                                             //按 ID 认领
                if (param[2 + i] == sid){
                    for (uint8_t b = 0; b < count; b++)
                        data_buf[(uint16_t)(sid - 1) * count + b] = rx_frame[SCS_RESP_DATA + b];
                    got |= (uint16_t)(1u << (sid - 1));
                    break;
                }
            }
        }
    }
    return (uint16_t)(~got & 0x0FFFu);      //取反后置 1 的位 = 没读到的颗
}

/* 一条指令给全部 SERVO_COUNT 颗写同一段地址(广播, 无应答), 按【观测序 = ID-1】取数:
 * 第 id 颗写 data_buf[(id-1)*count .. +count-1]。data_buf 需备 SERVO_COUNT × count 字节。
 * 返回 SCS_OK 只表示"帧发出去了"(广播收不到确认, 舵机有没有真收到看不出来);
 * count = 0 或放不下返回 SCS_ERR_PARAM。
 * 为何必须一条帧写完 12 颗、上限为何是 9 字节/颗 → 说明.md(8.5)(8.6) */
scs_err_t servo_sync_write(uint8_t addr, uint8_t count, const uint8_t *data_buf){
    uint8_t  param[BUFFER - 6];
    uint16_t need;                          //本帧参数总个数
    uint16_t p = 0;

    //参数 = 首地址(1) + 数据长(1) + 每颗(ID 1 + 数据 count)
    need = (uint16_t)(2 + (uint16_t)SERVO_COUNT * (1u + count));
    if (count == 0 || need > BUFFER - SCS_FRAME_FIXED) return SCS_ERR_PARAM;   //必须先算再写, 否则下面的循环会越界

    param[p++] = addr;                      //参数1: 首地址
    param[p++] = count;                     //参数2: 每颗写多少字节
    for (uint8_t i = 0; i < SERVO_COUNT; i++){
        param[p++] = (uint8_t)(i + 1);      //参数3+: 第 i 颗的 ID, 依次 1~SERVO_COUNT
        for (uint8_t b = 0; b < count; b++)
            param[p++] = data_buf[(uint16_t)i * count + b];
    }
    if (servo_com_pack_set(SCS_BROADCAST, SCS_SYNC_WRITE, param, (uint8_t)p) != SCS_OK)
        return SCS_ERR_PARAM;               //兜底: 算漏了也在这儿拦住(包没被改动, 不能发)

    servo_tx_frame_build();
    servo_data_send(tx_frame, tx_frame_len);
    return SCS_OK;
}

/* 状态重置: 清圈数(把多圈计数值归零)。无参数指令, 舵机应答 6 字节。
 * 例9: 请求 FF FF 01 02 0A F2 → 应答 FF FF 01 02 00 FC */
scs_err_t servo_reset(uint8_t sid){
    if (servo_com_pack_set(sid, SCS_RESET, 0, 0) != SCS_OK) return SCS_ERR_PARAM;
    servo_tx_frame_build();
    servo_data_send(tx_frame, tx_frame_len);
    if (sid == SCS_BROADCAST) return SCS_OK;        //广播无应答, 不能去等
    {
        scs_err_t err = servo_data_recv(rx_frame, SCS_RESP_FIXED);
        if (err != SCS_OK) return err;
    }
    return servo_rx_frame_check(rx_frame, SCS_RESP_FIXED, sid);
}

/* 参数恢复: 把除 ID 号以外的全部参数恢复为出厂值。无参数指令, 应答 6 字节。
 * ⚠️ 危险: 会抹掉 EPROM 里的全部配置(角度限位 / PID / 保护阈值…), 且【不可撤销】。
 *    手册要求先用 0x37 写 0 解锁 EPROM, 否则可能不生效或掉电不恢复;
 *    想留退路就先用 0x09(参数备份) 存一份。ID 号不受影响。
 * 例11: 请求 FF FF 01 02 06 F6 → 应答 FF FF 01 02 00 FC */
scs_err_t servo_recovery(uint8_t sid){
    if (servo_com_pack_set(sid, SCS_RECOVERY, 0, 0) != SCS_OK) return SCS_ERR_PARAM;
    servo_tx_frame_build();
    servo_data_send(tx_frame, tx_frame_len);
    if (sid == SCS_BROADCAST) return SCS_OK;        //广播无应答, 不能去等
    {
        scs_err_t err = servo_data_recv(rx_frame, SCS_RESP_FIXED);
        if (err != SCS_OK) return err;
    }
    return servo_rx_frame_check(rx_frame, SCS_RESP_FIXED, sid);
}
