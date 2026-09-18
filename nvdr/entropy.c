/*
 * NVDR entropy layer. See entropy.h for what the contexts are and why.
 *
 * The arithmetic coder itself is the LZMA range coder: 11-bit adaptive
 * probabilities, 32-bit range, carry handled by the cache/cache_size pair
 * in shift_low. It is used unmodified because it is well understood and
 * the interesting work here is the model, not the coder.
 */
#include "entropy.h"

#include <stdlib.h>
#include <string.h>

#define TOP_VALUE (1u << 24)

void nvdr_models_init(NvdrModels* m) {
    uint16_t* p = (uint16_t*)m;
    size_t n = sizeof(NvdrModels) / sizeof(uint16_t);
    for (size_t i = 0; i < n; i++) p[i] = NVDR_PROB_INIT;
}

int nvdr_area_context(int w, int h) {
    /* Bucket by log2 of the area. A 2x2 tile and a half-canvas block have
     * wildly different odds of subdividing, and this is the cheapest
     * summary of that both sides can compute. */
    unsigned area = (unsigned)(w * h);
    int bucket = 0;
    while (area > 1 && bucket < NVDR_AREA_CTX - 1) { area >>= 1; bucket++; }
    return bucket;
}

int nvdr_prev_context(int value) {
    int magnitude = value < 0 ? -value : value;
    if (magnitude == 0) return 0;
    if (magnitude == 1) return 1;
    if (magnitude == 2) return 2;
    if (magnitude <= 4) return 3;
    if (magnitude <= 8) return 4;
    if (magnitude <= 16) return 5;
    return 6;
}

/* ================================================================ encoder */

static void enc_put(NvdrEncoder* enc, uint8_t byte) {
    if (enc->count == enc->capacity) {
        size_t grown = enc->capacity ? enc->capacity * 2 : 4096;
        uint8_t* next = (uint8_t*)realloc(enc->bytes, grown);
        if (!next) { enc->failed = 1; return; }
        enc->bytes = next;
        enc->capacity = grown;
    }
    enc->bytes[enc->count++] = byte;
}

int nvdr_enc_init(NvdrEncoder* enc, size_t expected_bytes) {
    memset(enc, 0, sizeof(*enc));
    enc->capacity = expected_bytes ? expected_bytes : 4096;
    enc->bytes = (uint8_t*)malloc(enc->capacity);
    enc->range = 0xFFFFFFFFu;
    enc->cache_size = 1;
    return enc->bytes ? 0 : -1;
}

static void enc_shift_low(NvdrEncoder* enc) {
    if ((uint32_t)enc->low < 0xFF000000u || (int)(enc->low >> 32) != 0) {
        uint8_t carry = (uint8_t)(enc->low >> 32);
        uint8_t temp = enc->cache;
        do {
            enc_put(enc, (uint8_t)(temp + carry));
            temp = 0xFF;
        } while (--enc->cache_size != 0);
        enc->cache = (uint8_t)((uint32_t)enc->low >> 24);
    }
    enc->cache_size++;
    enc->low = (uint32_t)enc->low << 8;
}

void nvdr_enc_bit(NvdrEncoder* enc, uint16_t* prob, int bit) {
    uint32_t bound = (enc->range >> NVDR_PROB_BITS) * (*prob);
    if (!bit) {
        enc->range = bound;
        *prob = (uint16_t)(*prob + (((1 << NVDR_PROB_BITS) - *prob) >> NVDR_MOVE_BITS));
    } else {
        enc->low += bound;
        enc->range -= bound;
        *prob = (uint16_t)(*prob - (*prob >> NVDR_MOVE_BITS));
    }
    while (enc->range < TOP_VALUE) {
        enc->range <<= 8;
        enc_shift_low(enc);
    }
}

void nvdr_enc_direct(NvdrEncoder* enc, uint32_t value, int bit_count) {
    while (bit_count > 0) {
        enc->range >>= 1;
        bit_count--;
        enc->low += enc->range & (0u - ((value >> bit_count) & 1u));
        while (enc->range < TOP_VALUE) {
            enc->range <<= 8;
            enc_shift_low(enc);
        }
    }
}

void nvdr_enc_tree(NvdrEncoder* enc, uint16_t* probs, uint32_t value, int bit_count) {
    /* Walk a binary tree of contexts from the most significant bit, the
     * way a literal coder does: each node is conditioned on the prefix. */
    uint32_t node = 1;
    for (int i = bit_count - 1; i >= 0; i--) {
        int bit = (int)((value >> i) & 1);
        nvdr_enc_bit(enc, &probs[node], bit);
        node = (node << 1) | (uint32_t)bit;
    }
}

void nvdr_enc_residual(NvdrEncoder* enc, NvdrModels* m, int value,
                       int split_ctx, int channel, int prev_ctx) {
    int significant = value != 0;
    nvdr_enc_bit(enc, &m->sig[split_ctx][channel][prev_ctx], significant);
    if (!significant) return;

    nvdr_enc_bit(enc, &m->sign[split_ctx][channel], value < 0);

    int magnitude = value < 0 ? -value : value;   /* 1..127 */
    int remaining = magnitude - 1;

    /* Adaptive unary while the values are small — which is where almost
     * all of them are — then a flat escape for the tail. */
    int i = 0;
    for (; i < NVDR_MAG_CTX; i++) {
        int more = remaining > i;
        nvdr_enc_bit(enc, &m->mag[split_ctx][channel][prev_ctx][i], more);
        if (!more) return;
    }
    nvdr_enc_direct(enc, (uint32_t)(remaining - NVDR_MAG_CTX), 7);
}

int nvdr_enc_finish(NvdrEncoder* enc) {
    for (int i = 0; i < 5; i++) enc_shift_low(enc);
    return enc->failed ? -1 : 0;
}

void nvdr_enc_free(NvdrEncoder* enc) {
    free(enc->bytes);
    memset(enc, 0, sizeof(*enc));
}

/* ================================================================ decoder */

static uint8_t dec_next(NvdrDecoder* dec) {
    if (dec->pos >= dec->size) { dec->overrun = 1; return 0; }
    return dec->bytes[dec->pos++];
}

void nvdr_dec_init(NvdrDecoder* dec, const uint8_t* bytes, size_t size) {
    memset(dec, 0, sizeof(*dec));
    dec->bytes = bytes;
    dec->size = size;
    dec->range = 0xFFFFFFFFu;
    /* The encoder's first byte is always zero — an artefact of cache
     * starting empty — so five are consumed and the first discarded. */
    for (int i = 0; i < 5; i++) dec->code = (dec->code << 8) | dec_next(dec);
}

static void dec_normalize(NvdrDecoder* dec) {
    while (dec->range < TOP_VALUE) {
        dec->range <<= 8;
        dec->code = (dec->code << 8) | dec_next(dec);
    }
}

int nvdr_dec_bit(NvdrDecoder* dec, uint16_t* prob) {
    uint32_t bound = (dec->range >> NVDR_PROB_BITS) * (*prob);
    int bit;
    if (dec->code < bound) {
        dec->range = bound;
        *prob = (uint16_t)(*prob + (((1 << NVDR_PROB_BITS) - *prob) >> NVDR_MOVE_BITS));
        bit = 0;
    } else {
        dec->code -= bound;
        dec->range -= bound;
        *prob = (uint16_t)(*prob - (*prob >> NVDR_MOVE_BITS));
        bit = 1;
    }
    dec_normalize(dec);
    return bit;
}

uint32_t nvdr_dec_direct(NvdrDecoder* dec, int bit_count) {
    uint32_t result = 0;
    while (bit_count-- > 0) {
        dec->range >>= 1;
        dec->code -= dec->range;
        uint32_t mask = 0u - (dec->code >> 31);   /* all ones when it went negative */
        dec->code += dec->range & mask;
        dec_normalize(dec);
        result = (result << 1) + mask + 1;
    }
    return result;
}

uint32_t nvdr_dec_tree(NvdrDecoder* dec, uint16_t* probs, int bit_count) {
    uint32_t node = 1;
    for (int i = 0; i < bit_count; i++)
        node = (node << 1) | (uint32_t)nvdr_dec_bit(dec, &probs[node]);
    return node - ((uint32_t)1 << bit_count);
}

int nvdr_dec_residual(NvdrDecoder* dec, NvdrModels* m,
                      int split_ctx, int channel, int prev_ctx) {
    if (!nvdr_dec_bit(dec, &m->sig[split_ctx][channel][prev_ctx])) return 0;

    int negative = nvdr_dec_bit(dec, &m->sign[split_ctx][channel]);

    int remaining = 0;
    int i = 0;
    for (; i < NVDR_MAG_CTX; i++) {
        if (!nvdr_dec_bit(dec, &m->mag[split_ctx][channel][prev_ctx][i])) break;
        remaining = i + 1;
    }
    if (i == NVDR_MAG_CTX)
        remaining = NVDR_MAG_CTX + (int)nvdr_dec_direct(dec, 7);

    int magnitude = remaining + 1;
    return negative ? -magnitude : magnitude;
}
