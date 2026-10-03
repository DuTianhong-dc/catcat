/* ============================================================
 *  cat_polupd.h — 策略在线更新(双槽): 槽定义 + 帧处理接口
 *
 *  槽 A = S10+S11 @0x080C0000 / 槽 B = S8+S9 @0x08080000, 各 256KB;
 *  每槽 = [+0x000 头部区(256B 预留)] + [+0x100 数据区(定序 blob)]。
 *  更新只写【非激活槽】; 写完全量 CRC 通过才写 magic(唯一有效开关) —— 永不砖。
 *  完整协议(帧格式/时序/返回码/PC 侧约定) → 通信协议.txt 第七章。
 * ============================================================ */
#ifndef __CAT_POLUPD_H
#define __CAT_POLUPD_H
#include "stm32f4xx.h"
#include "cat_config.h"     /* OBS_DIM / POL_DIM_* */

/* ---- 槽规格 ---- */
#define CAT_POL_SLOTA_ADDR  0x080C0000u     /* S10+S11 */
#define CAT_POL_SLOTB_ADDR  0x08080000u     /* S8+S9  */
#define CAT_POL_SLOT_SIZE   0x00040000u     /* 每槽 256KB */
#define CAT_POL_DATA_OFF    0x00000100u     /* 数据区偏移(头部区 256B 预留) */

#define CAT_POL_MAGIC       0x50333343u     /* 'C33P' */
#define CAT_POL_FMT         2u              /* 格式版本(1 = 第一版无 seq, 不兼容) */
#define CAT_POL_TAG_LEN     24u
#define CAT_POL_BUILD_LEN   12u
#define CAT_POL_HDR_BYTES   72u             /* 头部字节数(= sizeof(cat_pol_hdr_t)) */
#define CAT_POL_CHUNK       64u             /* 单帧数据块上限(4 字节对齐) */

/* 期望 blob 长度(定序: W1 S1 B1 W2 S2 B2 W3 S3 B3 WH SH BH MEAN STD, 协议 7.4)。
 * 与 cat_policy.c 的 pol_point_blob() 同源 —— 两边必须一起改。 */
#define CAT_POL_DATA_BYTES ( \
    (uint32_t)OBS_DIM * POL_DIM_H1 + POL_DIM_H1 * 4u + POL_DIM_H1 * 4u + \
    (uint32_t)POL_DIM_H1 * POL_DIM_H2 + POL_DIM_H2 * 4u + POL_DIM_H2 * 4u + \
    (uint32_t)POL_DIM_H2 * POL_DIM_H3 + POL_DIM_H3 * 4u + POL_DIM_H3 * 4u + \
    (uint32_t)POL_DIM_H3 * POL_DIM_OUT + POL_DIM_OUT * 4u + POL_DIM_OUT * 4u + \
    (uint32_t)OBS_DIM * 4u + (uint32_t)OBS_DIM * 4u )

/* ---- 槽头部(72B; 上位机打包时生成, 原样转发) ---- */
typedef struct {
    uint32_t magic;                     /* 'C33P' */
    uint32_t fmt;                       /* = 2 */
    uint32_t seq;                       /* 版本序号(打包时刻时间戳); 两槽都有效时选大的 */
    uint32_t data_len;                  /* = CAT_POL_DATA_BYTES */
    uint32_t data_crc;                  /* CRC32(blob) */
    uint32_t obs_dim;                   /* = OBS_DIM */
    uint32_t act_dim;                   /* = POL_DIM_OUT(24, 注意不是 ACT_DIM) */
    uint32_t pose_hash;                 /* = cat_pose_hash() */
    uint32_t hdr_crc;                   /* CRC32(本结构前 32 字节: magic..pose_hash) */
    char     tag[CAT_POL_TAG_LEN];      /* 策略标签(展示) */
    char     build[CAT_POL_BUILD_LEN];  /* 权重 md5 前 12(溯源) */
} cat_pol_hdr_t;

/* ---- 返回码(ACK/STAT 的 rc; 0x00 OK, 其余避开 1~9) ---- */
#define POL_RC_OK        0x00u
#define POL_RC_BAD_HDR   0x11u      /* 头部坏(magic/fmt/hdr_crc) */
#define POL_RC_DIM       0x12u      /* 维度不符 */
#define POL_RC_POSE      0x13u      /* pose_hash 不符 */
#define POL_RC_LEN       0x14u      /* 长度非法 */
#define POL_RC_ORDER     0x15u      /* 顺序错(未 BEGIN 就 DATA / 偏移不连续等) */
#define POL_RC_FLASH     0x16u      /* 擦/写失败 */
#define POL_RC_CRC       0x17u      /* 整体 CRC 不符 */
#define POL_RC_VERIFY    0x18u      /* 写后自检失败 */
#define POL_RC_SEQ       0x19u      /* 新 seq 不大于激活槽(拒收, 防误刷旧版) */
#define POL_RC_NEED_OTAMODE 0x1Au   /* 正常模式不受理 BEGIN: 请按住 KEY0 重上电进更新模式 */

/* 模式开关(main 编排层调): 进更新模式时置 1。复位后归零 —— 正常模式永远收不了 BEGIN。 */
void cat_polupd_set_ota_mode(uint8_t on);

/* ---- 帧子类型(0xCC 帧的数据首字节) 与 应答类型(上行帧 type) ---- */
#define POL_SUB_BEGIN   0x70u
#define POL_SUB_DATA    0x71u
#define POL_SUB_END     0x72u
#define POL_SUB_ABORT   0x73u
#define POL_SUB_STAT    0x74u
#define POL_UP_ACK      0x76u
#define POL_UP_STAT     0x77u

/* ==================== 接口 ==================== */
/* 槽地址(slot: 1=槽A 2=槽B); 非法返回 0 */
uint32_t cat_polupd_slot_addr(uint8_t slot);

/* 扫一个槽: 全验(头部 + 全量 data_crc)。返回 1=有效(并给出 seq), 0=无效。
 * 耗时 ~20ms —— 只在开机 / 更新完成 / STAT 查询时调。 */
uint8_t cat_polupd_scan(uint8_t slot, uint32_t *seq_out);

/* 头部自洽 CRC32(前 32 字节) —— PC 打包工具必须同算 */
uint32_t cat_polupd_hdr_crc(const cat_pol_hdr_t *h);

/* 处理一帧 OTA 数据(已去 0xCC 帧壳, d[0] = 子类型)。
 * 填充应答: *resp_type = POL_UP_ACK / POL_UP_STAT, resp 载荷, *resp_len。
 * 返回值 = resp_type。⚠️ END 成功后由编排层(main)调 policy_select() 使新策略生效。 */
uint8_t cat_polupd_frame(const uint8_t *d, uint16_t n,
                         uint8_t *resp_type, uint8_t resp[48], uint8_t *resp_len);

/* 更新中 = 1。编排层据此卸力并跳过策略推理/舵机指令 */
uint8_t cat_polupd_busy(void);

/* 编排层每次 policy_select() 后调用: 同步"当前生效信息"(供 STAT 应答/选目标槽) */
void cat_polupd_set_active(uint8_t src, uint32_t seq, const char *tag, const char *build);

#endif
