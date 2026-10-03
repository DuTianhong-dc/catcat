/* ============================================================
 *  cat_esp32.c — ESP32-S3 通信链路驱动 (USART3 / PB10-PB11)
 *
 *  分层(本文件按此顺序排列, 新函数插到对应区块, 别追加到末尾):
 *    ① 初始化   esp32_init()
 *    ② 收发     esp32_data_recv() / esp32_data_send() / esp32_send_up()  [+ 环形接收缓冲]
 *    ③ 解析     esp32_parse_byte()(内部) / esp32_poll()
 *    ④ 取数     esp32_get_cmd() / esp32_alive() / esp32_ota_fetch()
 *
 *  命名: 发送侧 tx_ 前缀, 接收侧 rx_ 前缀。
 *  协议 → 通信协议.txt(五); 实现依据 → ESP32通信说明.md
 * ============================================================ */
#include "cat_esp32.h"
#include "cat_config.h"
#include "cat_sys.h"        /* esp32_poll 的失联判定要用时基 */

/* ---------- 环形接收缓冲 ----------
 * 中断只"入队", 解析放主循环 → 中断停留最短。满时丢【最旧】→ ESP32通信说明.md(五)
 * ⚠️ 已知竞态(2026-10-03 审查 T2, 保留不修): rx_r 被两边写 —— 主循环消费时推进,
 *    中断在"满"时也推进(丢最旧)。极端时序下可能丢/重收 1 字节, 后果限于单帧错乱,
 *    由帧校验和(ck)整帧丢弃兜底。要根治得让中断只写 rx_w(满时丢新), 但那会牺牲
 *    "保最新"的语义(OTA/控制流里新数据更重要) —— 取舍后维持现状。 */
static volatile uint8_t  rx_buf[ESP32_RX_BUF];
static volatile uint16_t rx_w;      //写入下标
static volatile uint16_t rx_r;      //读取下标

/* ---------- 解析结果 ----------
 * 每解出一帧就覆盖一份 → 上层取到的永远是最新的那条指令。 */
static esp32_cmd_t cmd;         //当前指令(供 esp32_get_cmd 取)
static uint32_t        last_ms;     //最后一次收到有效帧的时刻(失联判定用)

/* ---------- 下行控制帧(0xAA)状态机变量 (2026-09-29 提出函数, 供 OTA 分流判断) ---------- */
static uint8_t s_dl_buf[ESP32_DL_LEN];
static uint8_t s_dl_idx;

/* ---------- OTA 帧(0xCC)收集状态机 ----------
 * CC | len | 数据(len, 首字节=子类型) | ck。收齐且校验过 → s_ota_ready, 等主循环取走。 */
static uint8_t          s_ota_stage;        /* 0=空闲 1=收len 2=收数据 3=收ck */
static uint8_t          s_ota_buf[ESP32_OTA_MAX];
static uint8_t          s_ota_len;          /* 期望数据长度 */
static uint8_t          s_ota_idx;
static uint8_t          s_ota_ck;           /* 累加校验(含 CC/len/数据) */
static volatile uint8_t s_ota_ready;        /* 1=有一帧待取 */

/* ==================== ① 初始化 ==================== */
/* 配 USART3 + PB10(TX)/PB11(RX), 开接收中断。调用前 sys_clock_init() 必须先跑
 * (BRR 按 APB1=42MHz 算)。引脚/AF依据 与 ⚠️P2 跳线帽必须拔掉 → ESP32通信说明.md(二)(三) */
void esp32_init(void){
    //1、开时钟: GPIOB(AHB1) + USART3(APB1 —— 跟 IMU 同一条总线, 不是舵机的 APB2)
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOBEN;
    RCC->APB1ENR |= RCC_APB1ENR_USART3EN;

    //2、PB10(TX)/PB11(RX) 配 USART3 复用(AF7): 复用 + 推挽 + 高速 + 上拉(RX 上拉防浮空乱收)
    GPIOB->MODER = (GPIOB->MODER & ~((3u << 20) | (3u << 22)))
                 | (2u << 20) | (2u << 22);
    GPIOB->OTYPER  &= ~((1u << 10) | (1u << 11));
    GPIOB->OSPEEDR |=  (3u << 20) | (3u << 22);
    GPIOB->PUPDR = (GPIOB->PUPDR & ~((3u << 20) | (3u << 22))) | (1u << 20) | (1u << 22);
    GPIOB->AFR[1] = (GPIOB->AFR[1] & ~((0xFu << 8) | (0xFu << 12))) | (7u << 8) | (7u << 12);

    //3、USART3: 115200 8N1。BRR = 42MHz / 115200 = 364.6 → 365(误差 0.114%)
    USART3->BRR = (APB1_MHZ * 1000000u + 57600u) / 115200u;
    USART3->CR2 = 0;
    USART3->CR3 = 0;
    USART3->CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE | USART_CR1_RXNEIE;

    //4、开接收中断(ISR 在 cat_it.c)。优先级 6: 跟 IMU 同级, 都是"只收字节"类
    NVIC_SetPriority(USART3_IRQn, 6u);
    NVIC_EnableIRQ(USART3_IRQn);
}

/* ==================== ② 收发 ==================== */
/* 中断入口用: 把收到的 1 个字节推进环形缓冲, 只做这一件事就返回。 */
void esp32_data_recv(uint8_t byte){
    uint16_t next = (uint16_t)((rx_w + 1u) % ESP32_RX_BUF);
    if (next == rx_r) rx_r = (uint16_t)((rx_r + 1u) % ESP32_RX_BUF);   //满了丢最旧
    rx_buf[rx_w] = byte;
    rx_w = next;
}

/* 把 tx_len 个字节依次送进 USART3。调用前必须先 esp32_init()。 */
void esp32_data_send(const uint8_t *tx_buf, uint16_t tx_len){
    for (uint16_t i = 0; i < tx_len; i++) {
        while (!(USART3->SR & USART_SR_TXE));
        USART3->DR = tx_buf[i];
    }
    while (!(USART3->SR & USART_SR_TC));
}

/* 组一条上行小帧并发送(OTA 的 ACK/STAT 用):
 *   5A A5 | len | type | 载荷(len) | ck,  ck = 从字节 0 累加 & 0xFF(含帧头, 协议 5.3)。
 * 缓冲够 ACK(1B)/STAT(44B); 更大的载荷先扩缓冲。 */
void esp32_send_up(uint8_t type, const uint8_t *payload, uint8_t len){
    uint8_t f[64u];
    uint8_t n = 0u, ck = 0u;
    f[n++] = ESP32_UP_HEAD0;
    f[n++] = ESP32_UP_HEAD1;
    f[n++] = len;
    f[n++] = type;
    for (uint8_t i = 0; i < len; i++) f[n++] = payload[i];
    for (uint8_t i = 0; i < n; i++) ck = (uint8_t)(ck + f[i]);
    f[n++] = ck;
    esp32_data_send(f, n);
}

/* ==================== ③ 解析 ==================== */
/* 下行有两类帧, 用首字节分流: 0xCC = OTA 变长帧(策略在线更新, 第七章);
 * 0xAA = 10B 控制帧。一旦进入 OTA 收集就只喂 OTA 状态机。 */
static void esp32_ota_byte(uint8_t b){
    if (s_ota_ready) return;                    /* 上帧还没被取走, 忽略(防覆盖) */
    switch (s_ota_stage){
    case 0u:                                    /* 帧头 CC(入口已判) */
        s_ota_ck    = b;
        s_ota_stage = 1u;
        break;
    case 1u:                                    /* len */
        if (b == 0u || b > ESP32_OTA_MAX) { s_ota_stage = 0u; return; }
        s_ota_len   = b;
        s_ota_idx   = 0u;
        s_ota_ck    = (uint8_t)(s_ota_ck + b);
        s_ota_stage = 2u;
        break;
    case 2u:                                    /* 数据 */
        s_ota_buf[s_ota_idx++] = b;
        s_ota_ck = (uint8_t)(s_ota_ck + b);
        if (s_ota_idx >= s_ota_len) s_ota_stage = 3u;
        break;
    default:                                    /* ck: 从 CC 起累加(含帧头) */
        if (b == s_ota_ck) s_ota_ready = 1u;
        s_ota_stage = 0u;
        break;
    }
}

/* 喂 1 个字节进下行控制帧状态机(定长 10 字节), 收齐一帧就校验+解析。 */
static void esp32_dl_byte(uint8_t byte){
    switch (s_dl_idx) {
    case 0u:                                            //① 等帧头0
        if (byte == ESP32_DL_HEAD0) {
            s_dl_buf[s_dl_idx] = byte;
            s_dl_idx++;
        }
        break;

    case 1u:                                            //② 等帧头1
        /* 来的还是 AA 就把它当新包头1(保持 idx=1), 不直接复位 —— 否则
         * "AA(噪声) AA 55" 里的真包头会被吃掉。跟 IMU 那个坑同理: IMU驱动说明.md(四) */
        if (byte == ESP32_DL_HEAD1) {
            s_dl_buf[s_dl_idx] = byte;
            s_dl_idx++;
        } else if (byte != ESP32_DL_HEAD0) {
            s_dl_idx = 0u;                              //既不是 AA 也不是 55 → 重新找包头
        }
        break;

    default:                                            //③ 收数据(idx = 2~9)
        s_dl_buf[s_dl_idx] = byte;
        s_dl_idx++;
        if (s_dl_idx < ESP32_DL_LEN) break;             //还没收齐, 等下一个字节

        s_dl_idx = 0u;                                  //一帧收齐(校验过不过都先复位)
        {   uint8_t ck = 0;                             //校验和: 从 vx 累加到 tilt(不含帧头)
            for (uint8_t i = ESP32_DL_CK_FIRST; i < ESP32_DL_CK_OFF; i++)
                ck = (uint8_t)(ck + s_dl_buf[i]);
            if (ck == s_dl_buf[ESP32_DL_CK_OFF]) {      //校验过了才解析, 不过就丢整帧
                cmd.vx     = (float)(int16_t)(s_dl_buf[2] | ((uint16_t)s_dl_buf[3] << 8)) / ESP32_DL_SCALE;
                cmd.wz     = (float)(int16_t)(s_dl_buf[4] | ((uint16_t)s_dl_buf[5] << 8)) / ESP32_DL_SCALE;
                cmd.enable = (uint8_t)(s_dl_buf[6] & 0x01u);    //只有 bit0 = 使能
                cmd.pan    = (int8_t)s_dl_buf[7];
                cmd.tilt   = (int8_t)s_dl_buf[8];
                last_ms    = sys_ms();                  //记时刻, 失联判定要用
            }
        }
        break;
    }
}

/* 喂 1 个字节(主循环 esp32_poll 里从环形缓冲取出后调)。 */
static void esp32_parse_byte(uint8_t byte){
    if (s_ota_stage != 0u || (s_dl_idx == 0u && byte == ESP32_OTA_HEAD)){
        esp32_ota_byte(byte);
        return;
    }
    esp32_dl_byte(byte);
}

/* 主循环里调: 排空缓冲喂状态机, 再做失联判定。
 * ⚠️ 失联只能自己判: 上位机断连时不会有人通知你, 只看"多久没收到过有效帧"。 */
void esp32_poll(void){
    while (rx_r != rx_w) {                               //排空
        uint8_t b = rx_buf[rx_r];
        rx_r = (uint16_t)((rx_r + 1u) % ESP32_RX_BUF);
        esp32_parse_byte(b);
    }

    /* DEADMAN: 超时没收到有效帧 → 停。不能让猫抱着最后一条指令一直走下去。
     * ⚠️ 只归零运动量(enable/vx/wz): 云台 pan/tilt【故意不归零】—— 云台是独立通道,
     *    失联保持最后角度(2026-09-30 定, 见 通信协议.txt 6.6); 顺手把它归零才是 bug。 */
    if (sys_elapsed(last_ms, ESP32_DEADMAN_MS)) {
        cmd.enable = 0u;
        cmd.vx     = 0.0f;
        cmd.wz     = 0.0f;
    }
}

/* ==================== ④ 取数 ==================== */
/* 取最新一条指令。返回指针不拷贝 —— 上层只读, 别改里面的值。 */
const esp32_cmd_t *esp32_get_cmd(void){
    return &cmd;
}

/* 上位机是否还在连(没超过 DEADMAN)。0 = 已失联, 此时 get_cmd 的使能必然是 0。 */
uint8_t esp32_alive(void){
    return (uint8_t)!sys_elapsed(last_ms, ESP32_DEADMAN_MS);
}

/* 取一帧已收好的 OTA 数据(首字节=子类型); 无则 NULL。取走即释放(处理完再收下一帧)。 */
const uint8_t *esp32_ota_fetch(uint16_t *len){
    if (!s_ota_ready) return 0;
    if (len) *len = s_ota_len;
    s_ota_ready = 0u;
    return s_ota_buf;
}
