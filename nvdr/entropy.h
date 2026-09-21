/*
 * NVDR entropy layer — adaptive binary arithmetic coding with a context
 * model derived from the geometry.
 *
 * This replaces deflate on the level streams. Deflate models repetition;
 * what the residual planes actually contain is a distribution heavily
 * concentrated near zero, and deflate cannot spend a fraction of a bit on
 * a value it has no probability for. An arithmetic coder can, which is
 * what spec v0.14.1 §7 reserves `context_model` for.
 *
 * THE CONTEXT
 * -----------
 * The useful split is already in the format. A rectangle at level k either
 * came from a parent that subdivided — in which case its colour is a
 * genuinely new measurement and the delta against the parent is large — or
 * it is the same rectangle as before, in which case the delta is a small
 * correction to a colour that was already close. Those are two very
 * different distributions, and the previous deflate pass coded both with
 * one model.
 *
 * The decoder knows which case it is in before it needs the residual: it
 * has just replayed the split bitstream for that parent, so it knows
 * whether the parent produced one rectangle or several. The context costs
 * nothing to transmit — it is derived on both sides from data that is
 * already there. That is better than the spec's arrangement, where the
 * context model is a 2-6 MB payload shipped in the container.
 *
 * The second conditioning is the magnitude of the co-located residual in
 * the previous colour plane. A rectangle that needs a large correction
 * needs it in all three channels, and that correlation survives the split
 * context above because it is about *this* rectangle rather than about
 * which regime it belongs to.
 *
 * The in-plane predecessor was tried first, on the reasoning that
 * depth-first order keeps image neighbours close in the stream. It was
 * worth 0.04%: whatever it predicts, the split context already said. The
 * cross-plane neighbour is worth two orders of magnitude more. Channel 0
 * has no previous plane, so it falls back to its in-plane predecessor,
 * which costs nothing and recovers a little.
 *
 * Like the other context this is free — the decoder has the value because
 * it just decoded it — and the buckets are deliberately fine near zero,
 * where almost every residual lives.
 */
#ifndef NVDR_ENTROPY_H
#define NVDR_ENTROPY_H

#include <stdint.h>
#include <stddef.h>

/* Probabilities are 11-bit, adapted by a shift of 5 — the LZMA tuning,
 * which is well tested and cheap enough to run per bit. */
#define NVDR_PROB_BITS  11
#define NVDR_PROB_INIT  (1 << (NVDR_PROB_BITS - 1))
#define NVDR_MOVE_BITS  5

/* Split-flag contexts are bucketed by rectangle area: a large region is
 * far more likely to subdivide than a small one, and both sides know the
 * area before the bit is read. */
#define NVDR_AREA_CTX   16

/* Residual contexts: [came from a split][channel]. */
#define NVDR_SPLIT_CTX  2
#define NVDR_CHANNELS   3

/* Adaptive unary prefix length before magnitudes escape to direct bits. */
#define NVDR_MAG_CTX    8

/* Buckets over the neighbouring residual's magnitude, fine near zero. */
#define NVDR_PREV_CTX   7

typedef struct {
    uint16_t split[NVDR_AREA_CTX];
    /* Whether a rectangle carries a ramp, and along which axis. Both are
     * conditioned on area: a big rectangle spans more of a gradient, so it
     * is far likelier to want one. */
    uint16_t grad[NVDR_AREA_CTX];
    uint16_t grad_axis[NVDR_AREA_CTX];
    uint16_t slope_sig[NVDR_CHANNELS];
    uint16_t slope_sign[NVDR_CHANNELS];
    uint16_t slope_mag[NVDR_CHANNELS][NVDR_MAG_CTX];
    uint16_t token[256];                 /* binary tree over anchor_bits */
    uint16_t sig[NVDR_SPLIT_CTX][NVDR_CHANNELS][NVDR_PREV_CTX];
    uint16_t sign[NVDR_SPLIT_CTX][NVDR_CHANNELS];
    uint16_t mag[NVDR_SPLIT_CTX][NVDR_CHANNELS][NVDR_PREV_CTX][NVDR_MAG_CTX];
} NvdrModels;

void nvdr_models_init(NvdrModels* m);

/* Area bucket shared by encoder and decoder, so both index the same slot. */
int  nvdr_area_context(int w, int h);

/* Bucket of a residual's magnitude, used as the context for the next one. */
int  nvdr_prev_context(int value);

/* ------------------------------------------------------------- encoder */

typedef struct {
    uint8_t* bytes;
    size_t   capacity;
    size_t   count;
    uint64_t low;
    uint32_t range;
    uint8_t  cache;
    uint64_t cache_size;
    int      failed;
} NvdrEncoder;

int  nvdr_enc_init(NvdrEncoder* enc, size_t expected_bytes);
void nvdr_enc_bit(NvdrEncoder* enc, uint16_t* prob, int bit);
void nvdr_enc_direct(NvdrEncoder* enc, uint32_t value, int bit_count);
void nvdr_enc_tree(NvdrEncoder* enc, uint16_t* probs, uint32_t value, int bit_count);
void nvdr_enc_residual(NvdrEncoder* enc, NvdrModels* m, int value,
                       int split_ctx, int channel, int prev_ctx);
int  nvdr_enc_finish(NvdrEncoder* enc);   /* flushes; 0 on success */
void nvdr_enc_free(NvdrEncoder* enc);

/* ------------------------------------------------------------- decoder */

typedef struct {
    const uint8_t* bytes;
    size_t         size;
    size_t         pos;
    uint32_t       range;
    uint32_t       code;
    int            overrun;
} NvdrDecoder;

void     nvdr_dec_init(NvdrDecoder* dec, const uint8_t* bytes, size_t size);
int      nvdr_dec_bit(NvdrDecoder* dec, uint16_t* prob);
uint32_t nvdr_dec_direct(NvdrDecoder* dec, int bit_count);
uint32_t nvdr_dec_tree(NvdrDecoder* dec, uint16_t* probs, int bit_count);
int      nvdr_dec_residual(NvdrDecoder* dec, NvdrModels* m,
                           int split_ctx, int channel, int prev_ctx);

/* Ramp slopes use their own models: their distribution has nothing to do
 * with the residuals', and sharing would pollute both. */
void nvdr_enc_slope(NvdrEncoder* enc, NvdrModels* m, int value, int channel);
int  nvdr_dec_slope(NvdrDecoder* dec, NvdrModels* m, int channel);

/* Bits a slope would cost, for the encoder's rate-distortion decision. */
double nvdr_slope_bits(int value);

#endif /* NVDR_ENTROPY_H */
