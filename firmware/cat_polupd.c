/* ============================================================
 *  cat_polupd.c — 策略在线更新(双槽)实现
 *
 *  分层(按此顺序排列, 新函数插对应区块):
 *    ① Flash 裸寄存器操作(解锁/擦扇区/按字编程; F407)
 *    ② 槽检查       hdr 校验 / scan 全验
 *    ③ 帧处理       begin / data / end / abort / stat
 *
 *  ⚠️ 擦一个 128KB 扇区要 1~4 秒, 期间 CPU 停在 Flash 总线上 —— 更新态已卸力,
 *     属预期行为; 期间由 pol_iwdg_relax/restore 放宽 IWDG 超时(已接 cat_safety)。
 * ============================================================ */
#include "cat_polupd.h"
#include "cat_safety.h"     /* IWDG 放宽(safety_iwdg_relax/restore) */
#include "cat_pose.h"
#include "cat_crc32.h"
#include <stdint.h>         /* uintptr_t: 地址→指针(PC 64 位快检也不报警) */
#include <string.h>

/* ==================== ① Flash 裸寄存器操作 ==================== */
/* ⚠️ F4 的 CMSIS 头没有 FLASH_SR_OPERR 宏 —— 位 1 用字面量(避免"宏不存在"编译坑)。
 *    数据正确性最终由 END 的全量 CRC32 把关(不信任何单一标志位)。 */
#define FLASH_SR_OPERR_BIT 0x2u
#define FLASH_ERR_MASK (FLASH_SR_OPERR_BIT | FLASH_SR_WRPERR | FLASH_SR_PGAERR | \
                        FLASH_SR_PGPERR | FLASH_SR_PGSERR)

static uint8_t flash_unlock(void){
    FLASH->KEYR = 0x45670123u;
    FLASH->KEYR = 0xCDEF89ABu;
    return (FLASH->CR & FLASH_CR_LOCK) ? 1u : 0u;
}

static void flash_lock(void){ FLASH->CR |= FLASH_CR_LOCK; }

static void flash_clear_sr(void){ FLASH->SR = FLASH_SR_EOP | FLASH_ERR_MASK; }

/* 擦一个扇区(snb: F407 的 S0~S11)。擦除期间 flash 总线停摆、CPU 停住 —— 见文件头。 */
static uint8_t flash_erase_sector(uint32_t snb){
    while (FLASH->SR & FLASH_SR_BSY) { }
    flash_clear_sr();
    FLASH->CR &= ~FLASH_CR_PSIZE;
    FLASH->CR |= FLASH_CR_PSIZE_1;                          /* x32 (VDD 2.7~3.6V) */
    FLASH->CR |= FLASH_CR_SER;
    FLASH->CR &= ~FLASH_CR_SNB;
    FLASH->CR |= (snb << FLASH_CR_SNB_Pos);
    FLASH->CR |= FLASH_CR_STRT;
    while (FLASH->SR & FLASH_SR_BSY) { }
    uint8_t err = (FLASH->SR & FLASH_ERR_MASK) ? 1u : 0u;
    FLASH->CR &= ~FLASH_CR_SER;
    flash_clear_sr();
    return err;
}

/* 32 位字编程; addr 与 nwords 必须 4 字节对齐(调用方保证) */
static uint8_t flash_program_words(uint32_t addr, const uint32_t *w, uint32_t nwords){
    uint8_t err = 0u;
    while (FLASH->SR & FLASH_SR_BSY) { }
    flash_clear_sr();
    FLASH->CR &= ~FLASH_CR_PSIZE;
    FLASH->CR |= FLASH_CR_PSIZE_1;
    FLASH->CR |= FLASH_CR_PG;
    for (uint32_t i = 0; i < nwords; i++){
        *(volatile uint32_t *)(uintptr_t)(addr + i * 4u) = w[i];
        while (FLASH->SR & FLASH_SR_BSY) { }
        if (FLASH->SR & FLASH_ERR_MASK) err = 1u;
    }
    FLASH->CR &= ~FLASH_CR_PG;
    flash_clear_sr();
    return err;
}

/* 看门狗放宽钩子: 擦/验 Flash 时 CPU 停在 Flash 总线上 1~4s, 不放宽会被 IWDG 误复位。
 * 直接调 cat_safety 的 IWDG 服务 —— "谁干长活谁放宽", 逻辑内聚在本模块;
 * cat_safety 是服务层(与 cat_pose/cat_crc32 同待遇), 不算模块间横向依赖。 */
static void pol_iwdg_relax(void)  { safety_iwdg_relax(); }
static void pol_iwdg_restore(void){ safety_iwdg_restore(); }

/* ==================== ② 槽检查 ==================== */
/* 槽有效性缓存(2026-10-03 审查 L2): scan 一次全量 CRC 要 ~20ms, 而 STAT 应答若现扫
 * 两槽就是 ~40ms 阻塞 —— 正常模式下工具发一次 0x74, 主循环节拍被拉长 2 倍。
 * 缓存由 cat_polupd_scan 自己维护(每次扫描顺手写), 而 scan 的调用点只有:
 *   开机/更新后 policy_select(必调) · END 的 read-back · STAT(现改读缓存不再扫)。
 * 所以缓存总在"槽内容变化后"被刷新; 唯一的失效点 = BEGIN 擦槽 → 那儿显式清 0。 */
static uint8_t s_slot_ok[3];

uint32_t cat_polupd_slot_addr(uint8_t slot){
    if (slot == 1u) return CAT_POL_SLOTA_ADDR;
    if (slot == 2u) return CAT_POL_SLOTB_ADDR;
    return 0u;
}

uint32_t cat_polupd_hdr_crc(const cat_pol_hdr_t *h){
    return cat_crc32(0, (const uint8_t *)h, 32u);           /* magic..pose_hash 共 8 字 */
}

/* 头部检查(不含数据)。返回 POL_RC_*; begin 用它分报错误码。 */
static uint8_t pol_hdr_check(const cat_pol_hdr_t *h){
    if (h->magic != CAT_POL_MAGIC || h->fmt != CAT_POL_FMT)      return POL_RC_BAD_HDR;
    if (h->hdr_crc != cat_polupd_hdr_crc(h))                     return POL_RC_BAD_HDR;
    if (h->obs_dim != OBS_DIM || h->act_dim != POL_DIM_OUT)      return POL_RC_DIM;
    if (h->pose_hash != cat_pose_hash())                         return POL_RC_POSE;
    if (h->data_len != CAT_POL_DATA_BYTES || (h->data_len & 3u)) return POL_RC_LEN;
    return POL_RC_OK;
}

uint8_t cat_polupd_scan(uint8_t slot, uint32_t *seq_out){
    uint32_t addr = cat_polupd_slot_addr(slot);
    uint8_t  ok   = 0u;
    if (addr) {
        const cat_pol_hdr_t *h = (const cat_pol_hdr_t *)(uintptr_t)addr;
        if (pol_hdr_check(h) == POL_RC_OK) {
            const uint8_t *d = (const uint8_t *)(uintptr_t)(addr + CAT_POL_DATA_OFF);
            if (cat_crc32(0, d, h->data_len) == h->data_crc) {   /* 全量 ~20ms */
                if (seq_out) *seq_out = h->seq;
                ok = 1u;
            }
        }
    }
    s_slot_ok[slot & 3u] = ok;          /* 顺手更新缓存(见上方说明; slot 只会是 1/2) */
    return ok;
}

/* ==================== ③ 帧处理 ==================== */
static uint8_t  s_state;                                    /* 0=空闲 1=接收中 */
static uint8_t  s_target;                                   /* 目标槽 1/2 */
static uint32_t s_expect_len, s_expect_crc;
static uint32_t s_off, s_crc;                               /* 已收偏移 / 增量 CRC32 */
static cat_pol_hdr_t s_hdr;                                 /* BEGIN 收到, END 时落盘 */

/* 当前生效信息(main 每次 policy_select 后同步; 供 STAT 应答与"选目标槽") */
static uint8_t  s_act_src;
static uint8_t  s_ota_mode;             /* 1 = 更新模式(main 进 OTA 循环时置): BEGIN 只受理于该模式 */
static uint32_t s_act_seq;
static char     s_act_tag[CAT_POL_TAG_LEN];
static char     s_act_build[CAT_POL_BUILD_LEN];

uint8_t cat_polupd_busy(void){ return s_state; }

/* 把字符串按"到 '\0' 为止(最多填满) + 其余补零"拷进定长字段。
 * ⚠️ 不能直接 memcpy 定长: 源是字符串字面量(如 "v31/v31world" 只有 12B),
 *    定长 24B 会越界读相邻内存 → STAT 里显示乱码尾巴(2026-09-30 真机见)。
 *    (填满不留 '\0' 是故意的: tag[24]/build[12] 本是定长字段, 显示端按定长取) */
static void pol_copy_field(char *dst, uint32_t cap, const char *src){
    uint32_t i = 0;
    for (; i < cap && src[i]; i++) dst[i] = src[i];
    for (; i < cap; i++)           dst[i] = '\0';
}

void cat_polupd_set_active(uint8_t src, uint32_t seq, const char *tag, const char *build){
    s_act_src = src;
    s_act_seq = seq;
    if (tag)   pol_copy_field(s_act_tag,   CAT_POL_TAG_LEN,   tag);
    if (build) pol_copy_field(s_act_build, CAT_POL_BUILD_LEN, build);
}

/* 擦目标槽(两个连续 128KB 扇区): 槽A = S10+S11, 槽B = S8+S9 */
static uint8_t pol_erase_slot(uint8_t slot){
    uint32_t s0  = (slot == 1u) ? 10u : 8u;
    uint8_t  err = 0u;
    pol_iwdg_relax();
    if (flash_unlock()) err = 1u;
    else {
        if (flash_erase_sector(s0))              err = 1u;
        safety_iwdg_feed();     /* 扇区间喂一口: 每个扇区各有独立预算(慢擦除板实测 >4s/扇区) */
        if (!err && flash_erase_sector(s0 + 1u)) err = 1u;
        flash_lock();
    }
    pol_iwdg_restore();
    return err;
}

/* 模式开关(由 main 编排层调用): 进更新模式置 1, 正常模式 = 0(默认)。
 * ⚠️ 复位后归零 —— 正常模式永远收不了 BEGIN, 必须按住 KEY0 重上电走更新模式。
 * 理由(2026-09-30 真机): 正常模式下擦除会被 IMU 中断风暴打死(CPU 现场抓包实锤), 
 * 更新模式只开 ESP32 链路, 从根上避开干扰。 */
void cat_polupd_set_ota_mode(uint8_t on){ s_ota_mode = on ? 1u : 0u; }

/* 0x70 BEGIN: d = 72B 头部原文 */
static uint8_t pol_begin(const uint8_t *d, uint16_t n){
    if (!s_ota_mode)            return POL_RC_NEED_OTAMODE;
    if (s_state)                return POL_RC_ORDER;
    if (n != CAT_POL_HDR_BYTES) return POL_RC_LEN;

    cat_pol_hdr_t h;
    memcpy(&h, d, CAT_POL_HDR_BYTES);
    uint8_t rc = pol_hdr_check(&h);
    if (rc != POL_RC_OK) return rc;

    /* seq 必须比当前激活的更新(防误刷旧版); 首次(无激活)不检查 —— 协议 7.5 */
    if (s_act_src != 0u && h.seq <= s_act_seq) return POL_RC_SEQ;

    /* 目标槽 = 激活之外的那个; 无激活 → 槽 A(协议 7.6) */
    uint8_t target = (s_act_src == 1u) ? 2u : 1u;

    if (pol_erase_slot(target)) return POL_RC_FLASH;        /* 旧内容在此刻物理清除 */
    s_slot_ok[target] = 0u;                                 /* 擦完 → 有效性缓存同步失效(审查 L2) */

    s_hdr        = h;
    s_target     = target;
    s_expect_len = h.data_len;
    s_expect_crc = h.data_crc;
    s_off        = 0u;
    s_crc        = 0u;
    s_state      = 1u;
    return POL_RC_OK;
}

/* 0x71 DATA: d = 子类型 + 偏移u32 + 4~64B 数据 */
static uint8_t pol_data(const uint8_t *d, uint16_t n){
    if (s_state != 1u) return POL_RC_ORDER;
    if (n < 5u)        return POL_RC_LEN;

    uint32_t off = (uint32_t)d[1] | ((uint32_t)d[2] << 8) |
                   ((uint32_t)d[3] << 16) | ((uint32_t)d[4] << 24);
    const uint8_t *blob = d + 5;
    uint16_t bn = (uint16_t)(n - 5u);

    if (off != s_off)                                return POL_RC_ORDER;
    if (bn == 0u || bn > CAT_POL_CHUNK || (bn & 3u)) return POL_RC_LEN;
    if (s_off + bn > s_expect_len)                   return POL_RC_ORDER;

    /* 拷进对齐缓冲再按字编程: 帧内数据不保证 4 字节对齐 */
    uint32_t w[CAT_POL_CHUNK / 4u];
    memcpy(w, blob, bn);

    uint8_t err = 0u;
    if (flash_unlock()) err = 1u;
    else {
        if (flash_program_words(cat_polupd_slot_addr(s_target) + CAT_POL_DATA_OFF + s_off,
                                w, bn / 4u)) err = 1u;
        flash_lock();
    }
    if (err){
        s_state = 0u;                                       /* 写失败: 终止本轮 */
        pol_iwdg_restore();
        return POL_RC_FLASH;
    }

    s_crc  = cat_crc32(s_crc, blob, bn);                    /* 增量 CRC, 免二次读 flash */
    s_off += bn;
    return POL_RC_OK;
}

/* 0x72 END: 全量 CRC 通过 → 写头部(最后写 magic) → read-back 自检 */
static uint8_t pol_end(void){
    if (s_state != 1u)         return POL_RC_ORDER;
    if (s_off != s_expect_len) return POL_RC_ORDER;         /* 数据没收全 */

    uint8_t rc = POL_RC_OK;
    if (s_crc != s_expect_crc){
        rc = POL_RC_CRC;                                    /* magic 不写 → 槽无效 */
    } else {
        /* 落盘顺序: 先写除 magic 外的全部 → 最后单独写 magic(唯一有效开关) */
        uint32_t addr = cat_polupd_slot_addr(s_target);
        uint32_t w[CAT_POL_HDR_BYTES / 4u];
        memcpy(w, &s_hdr, CAT_POL_HDR_BYTES);
        if (flash_unlock()) rc = POL_RC_FLASH;
        else {
            if (flash_program_words(addr + 4u, &w[1], (CAT_POL_HDR_BYTES / 4u) - 1u))
                rc = POL_RC_FLASH;
            if (rc == POL_RC_OK && flash_program_words(addr, &w[0], 1u))
                rc = POL_RC_FLASH;
            flash_lock();
        }
        /* read-back 权威判据: 重读 flash 全量验证(不信 RAM 里累积的 s_crc) */
        if (rc == POL_RC_OK && !cat_polupd_scan(s_target, 0)){
            rc = POL_RC_VERIFY;
            /* 自检不过 → 擦掉头部扇区, 让"无效"状态确定(免得下次开机白验 ~20ms) */
            pol_iwdg_relax();
            if (!flash_unlock()){
                flash_erase_sector((s_target == 1u) ? 10u : 8u);
                flash_lock();
            }
            pol_iwdg_restore();
        }
    }
    s_state = 0u;
    pol_iwdg_restore();
    return rc;
}

static uint8_t pol_abort(void){
    s_state = 0u;                                           /* 目标槽保持无效 */
    pol_iwdg_restore();
    return POL_RC_OK;
}

/* 0x74 STAT: 应答载荷 44B = rc active seq_active slotA slotB tag build ——
 * ⚠️ slotA/slotB 读【缓存】(审查 L2): 原来每应答一次就现扫两槽(~40ms 阻塞主循环节拍)。
 *    缓存由 policy_select(开机/更新后必跑)与 END 的 read-back 维护, 见 s_slot_ok 说明。 */
static uint8_t pol_stat(uint8_t resp[48], uint8_t *resp_len){
    uint8_t *p = resp;
    *p++ = POL_RC_OK;
    *p++ = s_act_src;
    *p++ = (uint8_t)(s_act_seq      );
    *p++ = (uint8_t)(s_act_seq >>  8);
    *p++ = (uint8_t)(s_act_seq >> 16);
    *p++ = (uint8_t)(s_act_seq >> 24);
    *p++ = s_slot_ok[1];                                    /* slotA 有效性(缓存) */
    *p++ = s_slot_ok[2];                                    /* slotB 有效性(缓存) */
    memcpy(p, s_act_tag,   CAT_POL_TAG_LEN);   p += CAT_POL_TAG_LEN;
    memcpy(p, s_act_build, CAT_POL_BUILD_LEN); p += CAT_POL_BUILD_LEN;
    *resp_len = 44u;
    return POL_UP_STAT;
}

uint8_t cat_polupd_frame(const uint8_t *d, uint16_t n,
                         uint8_t *resp_type, uint8_t resp[48], uint8_t *resp_len){
    uint8_t rc = POL_RC_LEN;
    if (n != 0u){
        switch (d[0]){
        case POL_SUB_BEGIN: rc = pol_begin(d + 1, (uint16_t)(n - 1u)); break;
        case POL_SUB_DATA:  rc = pol_data(d, n);                       break;
        case POL_SUB_END:   rc = pol_end();                            break;
        case POL_SUB_ABORT: rc = pol_abort();                          break;
        case POL_SUB_STAT:                                             /* 应答=STAT */
            /* ⚠️ 2026-09-30 修: 本路径从这直接 return, 绕过了函数尾部的统一设置 ——
             * 必须在这里自己设 resp_type, 否则 main 看到 rtype=0 会【静默不发】!
             * (真机首发实测: CC 帧收成、ready 置位、fetch 正常, 就是不出 STAT) */
            *resp_type = POL_UP_STAT;
            return pol_stat(resp, resp_len);
        default:            rc = POL_RC_BAD_HDR;                       break;
        }
    }
    resp[0]    = rc;
    *resp_len  = 1u;
    *resp_type = POL_UP_ACK;
    return POL_UP_ACK;
}
