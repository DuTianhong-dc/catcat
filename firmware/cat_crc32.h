/**
 * @file    cat_crc32.h — 标准 CRC-32/ISO-HDLC (同 zlib.crc32), 头文件内联实现。
 *
 * 为什么放头文件: 固件(Keil)与 SIL(zig/PC)都要用, 放头可两侧零改动、无重复符号。
 * 约定: 初值传 0; 支持增量 —— cat_crc32(cat_crc32(0,a),b) == cat_crc32(0,ab)。
 *
 * 用途: ① 策略槽头部自洽(hdr_crc) ② 槽数据全量校验(data_crc)
 *       ③ 姿态/限位常量指纹(pose_hash, 防策略与姿态表不同源)
 */
#ifndef __CAT_CRC32_H
#define __CAT_CRC32_H

#include <stdint.h>

static uint32_t cat_crc32(uint32_t crc, const uint8_t *p, uint32_t n) {
    crc = ~crc;                                  /* 初值 0 → 内部 0xFFFFFFFF */
    while (n--) {
        crc ^= *p++;
        for (int k = 0; k < 8; k++)
            crc = (crc >> 1) ^ (0xEDB88320u & (uint32_t)(-(int32_t)(crc & 1u)));
    }
    return ~crc;
}

#endif /* __CAT_CRC32_H */
