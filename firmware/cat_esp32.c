/* ============================================================
 *  cat_esp32.c — ESP32-S3 通信链路驱动 (USART3 / PB10-PB11)
 *
 *  分层(本文件按此顺序排列, 新函数插到对应区块, 别追加到末尾):
 *    ① 初始化   esp32_init()
 *    ② 收发     esp32_data_recv() / esp32_data_send()   [+ 环形接收缓冲]
 *    ③ 解析     esp32_parse_byte()(内部) / esp32_poll()
 *    ④ 取数     esp32_get_cmd() / esp32_alive()
 *
 *  命名: 发送侧 tx_ 前缀, 接收侧 rx_ 前缀。
 *  协议 → 通信协议.txt(五); 实现依据 → ESP32通信说明.md
 * ============================================================ */
#include "cat_esp32.h"
#include "cat_config.h"
#include "cat_sys.h"        /* esp32_poll 的失联判定要用时基 */

/* ---------- 环形接收缓冲 ----------
 * 中断只"入队", 解析放主循环 → 中断停留最短。满时丢【最旧】→ ESP32通信说明.md(五) */
static volatile uint8_t  rx_buf[ESP32_RX_BUF];
static volatile uint16_t rx_w;      //写入下标
static volatile uint16_t rx_r;      //读取下标

/* ---------- 解析结果 ----------
 * 每解出一帧就覆盖一份 → 上层取到的永远是最新的那条指令。 */
static esp32_cmd_t cmd;         //当前指令(供 esp32_get_cmd 取)
static uint32_t        last_ms;     //最后一次收到有效帧的时刻(失联判定用)

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

/* ==================== ③ 解析 ==================== */
/* 喂 1 个字节进下行帧状态机(定长 10 字节, 比 IMU 的变长帧简单), 收齐一帧就校验+解析。 */
static void esp32_parse_byte(uint8_t byte){
    static uint8_t buf[ESP32_DL_LEN];
    static uint8_t idx;

    switch (idx) {
    case 0u:                                            //① 等帧头0
        if (byte == ESP32_DL_HEAD0) {
            buf[idx] = byte;
            idx++;
        }
        break;

    case 1u:                                            //② 等帧头1
        /* 来的还是 AA 就把它当新包头1(保持 idx=1), 不直接复位 —— 否则
         * "AA(噪声) AA 55" 里的真包头会被吃掉。跟 IMU 那个坑同理: IMU驱动说明.md(四) */
        if (byte == ESP32_DL_HEAD1) {
            buf[idx] = byte;
            idx++;
        } else if (byte != ESP32_DL_HEAD0) {
            idx = 0u;                                   //既不是 AA 也不是 55 → 重新找包头
        }
        break;

    default:                                            //③ 收数据(idx = 2~9)
        buf[idx] = byte;
        idx++;
        if (idx < ESP32_DL_LEN) break;                  //还没收齐, 等下一个字节

        idx = 0u;                                       //一帧收齐(校验过不过都先复位)
        {   uint8_t ck = 0;                             //校验和: 从 vx 累加到 tilt(不含帧头)
            for (uint8_t i = ESP32_DL_CK_FIRST; i < ESP32_DL_CK_OFF; i++)
                ck = (uint8_t)(ck + buf[i]);
            if (ck == buf[ESP32_DL_CK_OFF]) {           //校验过了才解析, 不过就丢整帧
                cmd.vx     = (float)(int16_t)(buf[2] | ((uint16_t)buf[3] << 8)) / ESP32_DL_SCALE;
                cmd.wz     = (float)(int16_t)(buf[4] | ((uint16_t)buf[5] << 8)) / ESP32_DL_SCALE;
                cmd.enable = (uint8_t)(buf[6] & 0x01u); //只有 bit0 = 使能
                cmd.pan    = (int8_t)buf[7];
                cmd.tilt   = (int8_t)buf[8];
                last_ms    = sys_ms();                  //记时刻, 失联判定要用
            }
        }
        break;
    }
}

/* 主循环里调: 排空缓冲喂状态机, 再做失联判定。
 * ⚠️ 失联只能自己判: 上位机断连时不会有人通知你, 只看"多久没收到过有效帧"。 */
void esp32_poll(void){
    while (rx_r != rx_w) {                               //排空
        uint8_t b = rx_buf[rx_r];
        rx_r = (uint16_t)((rx_r + 1u) % ESP32_RX_BUF);
        esp32_parse_byte(b);
    }

    /* DEADMAN: 超时没收到有效帧 → 停。不能让猫抱着最后一条指令一直走下去 */
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
