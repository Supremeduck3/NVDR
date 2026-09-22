#include "nvdrv.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "entropy.h"

/* ---------------------------------------------------------------- bytes */

static void put_u16v(uint8_t* p, uint16_t v) { p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); }
static void put_u32v(uint8_t* p, uint32_t v) {
    p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); p[2]=(uint8_t)(v>>16); p[3]=(uint8_t)(v>>24);
}
static uint16_t get_u16v(const uint8_t* p) { return (uint16_t)(p[0] | (p[1]<<8)); }
static uint32_t get_u32v(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}
static int clamp255v(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

NvdrvConfig nvdrv_default_config(void) {
    NvdrvConfig c;
    c.frame = nvdr_default_config();
    /* Every frame's texture in one layer: nobody watches a video frame
     * arrive, and the split costs bytes. */
    c.frame.band = 0;
    /* Two seconds at 24fps. Short enough to join a stream quickly, long
     * enough that the intra frames — four times the size of a predicted
     * one — do not dominate the bitrate. */
    c.gop = 48;
    /* Wide enough for a brisk handheld pan at this resolution. The search
     * is global, so its cost is one pass per candidate over a lattice, not
     * per block. */
    c.search = 12;
    c.intra_threshold = 24.0f;
    /* Per-block motion on, in quarter pixels, with the block size chosen
     * by resolution (-1). With whole-pixel vectors 8x8 won everywhere,
     * because the finer grid made up for the coarse precision. With
     * quarter pixels, 16x16 wins on the 960x540 clean clip: 591 bytes a
     * frame of field against 2386, for 0.16 dB of the 3 dB quarter pixels
     * bought. On the 128x128 sequence in the regression gate, where the
     * moving object is 30 px, 8x8 wins both ways: 7189 bytes against 8626
     * and 0.3 dB more by the last frame. So 16 from 0.2 Mpx up, 8 below. */
    c.block = -1;
    c.pred_q = 0;
    /* In SAD per bit of field. With quarter-pixel deltas a vector costs
     * more bits than it did, so the weight that balances them rose from 8. */
    c.mv_lambda = 16;
    c.fps = 24;
    return c;
}

/* ------------------------------------------------------- motion search */

/*
 * One translation for the whole frame, minimising mean absolute
 * difference. Coarse pass on a stride-4 lattice over the full range, then
 * a fine pass over the neighbourhood of the winner — a full search at
 * every offset costs more than the vector is worth at this stage.
 */
static double shift_cost(const NvdrImage* cur, const NvdrImage* ref,
                         int dx, int dy, int step, double give_up) {
    double acc = 0.0; long n = 0;
    int m = 8;
    for (int y = m; y < cur->height - m; y += step) {
        const unsigned char* a = cur->pixels + ((size_t)y * cur->width) * 3;
        int sy = y + dy;
        if (sy < 0) sy = 0;
        if (sy >= ref->height) sy = ref->height - 1;
        const unsigned char* b = ref->pixels + ((size_t)sy * ref->width) * 3;
        for (int x = m; x < cur->width - m; x += step) {
            int sx = x + dx;
            if (sx < 0) sx = 0;
            if (sx >= ref->width) sx = ref->width - 1;
            const unsigned char* pa = a + (size_t)x * 3;
            const unsigned char* pb = b + (size_t)sx * 3;
            for (int c = 0; c < 3; c++) { int d = pa[c]-pb[c]; acc += d < 0 ? -d : d; }
            n++;
        }
        if (n > 4096 && acc / n > give_up) return acc / n;   /* already lost */
    }
    return n ? acc / n : 1e30;
}

static void find_shift(const NvdrImage* cur, const NvdrImage* ref, int range,
                       int* bdx, int* bdy) {
    *bdx = 0; *bdy = 0;
    if (range <= 0) return;
    double best = shift_cost(cur, ref, 0, 0, 4, 1e30);
    for (int dy = -range; dy <= range; dy += 2)
        for (int dx = -range; dx <= range; dx += 2) {
            double c = shift_cost(cur, ref, dx, dy, 4, best);
            if (c < best) { best = c; *bdx = dx; *bdy = dy; }
        }
    int cx = *bdx, cy = *bdy;
    best = shift_cost(cur, ref, cx, cy, 2, 1e30);
    for (int dy = cy - 2; dy <= cy + 2; dy++)
        for (int dx = cx - 2; dx <= cx + 2; dx++) {
            double c = shift_cost(cur, ref, dx, dy, 2, best);
            if (c < best) { best = c; *bdx = dx; *bdy = dy; }
        }
}

/* The reference translated by (dx, dy), edges held rather than wrapped. */
static void shift_into(const NvdrImage* src, NvdrImage* dst, int dx, int dy) {
    for (int y = 0; y < dst->height; y++) {
        int sy = y + dy;
        if (sy < 0) sy = 0;
        if (sy >= src->height) sy = src->height - 1;
        for (int x = 0; x < dst->width; x++) {
            int sx = x + dx;
            if (sx < 0) sx = 0;
            if (sx >= src->width) sx = src->width - 1;
            memcpy(dst->pixels + ((size_t)y*dst->width + x)*3,
                   src->pixels + ((size_t)sy*src->width + sx)*3, 3);
        }
    }
}

/* -------------------------------------------------------- block motion */

/*
 * One vector per block instead of one per frame.
 *
 * A single translation describes a camera pan and nothing else. The moment
 * something in the shot moves differently from the camera — a person
 * walking across a pan, a car against a still street — the global vector
 * is right for one of them and wrong for the other, and the wrong part
 * shows up as residual the frame has to pay for.
 *
 * Each block is searched in two windows: around the global vector, which
 * is where most blocks are, and around zero, because the thing moving
 * against a pan is very often something holding still relative to the
 * camera, or the reverse. Blocks are independent, so the search threads
 * without the order of anything depending on scheduling.
 *
 * A block only leaves the global vector when doing so buys more than a
 * level per pixel of absolute difference. Below that, the gain is noise
 * the residual would have absorbed anyway, and every block that agrees
 * with its neighbours costs about a bit once the field is coded (see
 * pack_field).
 */
#define NVDRV_LOCAL_RANGE 4

static int block_sad(const NvdrImage* cur, const NvdrImage* ref,
                     int x0, int y0, int bw, int bh, int dx, int dy, int limit) {
    int acc = 0;
    for (int y = y0; y < y0 + bh; y++) {
        int sy = y + dy;
        if (sy < 0) sy = 0;
        if (sy >= ref->height) sy = ref->height - 1;
        const unsigned char* a = cur->pixels + ((size_t)y * cur->width + x0) * 3;
        const unsigned char* row = ref->pixels + (size_t)sy * ref->width * 3;
        for (int x = x0; x < x0 + bw; x++, a += 3) {
            int sx = x + dx;
            if (sx < 0) sx = 0;
            if (sx >= ref->width) sx = ref->width - 1;
            const unsigned char* b = row + (size_t)sx * 3;
            int d0 = a[0] - b[0], d1 = a[1] - b[1], d2 = a[2] - b[2];
            acc += (d0 < 0 ? -d0 : d0) + (d1 < 0 ? -d1 : d1) + (d2 < 0 ? -d2 : d2);
        }
        if (acc >= limit) return acc;    /* already lost */
    }
    return acc;
}

static void block_search(const NvdrImage* cur, const NvdrImage* ref, int block,
                         int gdx, int gdy, int8_t* vx, int8_t* vy) {
    int nbx = (cur->width + block - 1) / block;
    int nby = (cur->height + block - 1) / block;
    int nb = nbx * nby;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
    for (int b = 0; b < nb; b++) {
        int x0 = (b % nbx) * block, y0 = (b / nbx) * block;
        int bw = cur->width - x0 < block ? cur->width - x0 : block;
        int bh = cur->height - y0 < block ? cur->height - y0 : block;
        int bias = bw * bh;
        int bdx = gdx, bdy = gdy;
        int best = block_sad(cur, ref, x0, y0, bw, bh, gdx, gdy, INT_MAX);
        for (int pass = 0; pass < 2; pass++) {
            int cx = pass ? 0 : gdx, cy = pass ? 0 : gdy;
            for (int dy = cy - NVDRV_LOCAL_RANGE; dy <= cy + NVDRV_LOCAL_RANGE; dy++)
                for (int dx = cx - NVDRV_LOCAL_RANGE; dx <= cx + NVDRV_LOCAL_RANGE; dx++) {
                    if (dx == gdx && dy == gdy) continue;
                    int limit = best - bias;
                    if (limit <= 0) continue;
                    int c = block_sad(cur, ref, x0, y0, bw, bh, dx, dy, limit);
                    if (c + bias < best) { best = c + bias; bdx = dx; bdy = dy; }
                }
        }
        vx[b] = (int8_t)bdx;
        vy[b] = (int8_t)bdy;
    }
}

/*
 * QUARTER-PIXEL MOTION
 * --------------------
 * Block vectors are in quarter pixels. A whole-pixel vector cannot follow
 * a pan of 1.12 px a frame or a slow zoom, and with a transform coding the
 * residual, the misalignment it leaves is texture over the whole frame,
 * which is the most expensive thing to code. Measured on two frames of the
 * clean clip against the source (scripts/analysis/subpel), quarter pixels
 * cut the residual's bytes by 65% and raised PSNR by 3 dB.
 *
 * Between whole pixels the reference is interpolated the way H.264 does
 * luma, in integers, identically in nvdrv.js. With p the reference, edges
 * held:
 *
 *   B  half pixel across:  clip((p[-2] - 5p[-1] + 20p[0] + 20p[1] - 5p[2] + p[3] + 16) >> 5)
 *   H  half pixel down:    the same filter vertically
 *   J  the centre:         the same filter down a column of the unrounded
 *                          horizontal sums, clip((sum + 512) >> 10)
 *
 * and every quarter position the rounded mean of its two nearest whole or
 * half samples. Bilinear was what this started with; the 6-tap filter
 * keeps the detail bilinear blurs away, measured 5 to 7% less residual on
 * the source frames.
 *
 * The half-pixel planes are built once per reference, over the frame and
 * a margin wide enough for any vector the field can hold (int8 quarter
 * pixels: 32 px) plus the filter's reach. The global vector stays in
 * whole pixels; the field is predicted from it multiplied by four.
 */
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

#define SP_PAD 40

typedef struct {
    int w, h, sw, sh;            /* frame, and padded plane size */
    uint8_t* F;                  /* whole pixels */
    uint8_t* B;                  /* half pixel to the right */
    uint8_t* H;                  /* half pixel below */
    uint8_t* J;                  /* half pixel right and below */
} Subpel;

static inline size_t sp_at(const Subpel* s, int x, int y) {
    return ((size_t)(y + SP_PAD) * s->sw + (size_t)(x + SP_PAD)) * 3;
}

static inline int clip8i(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

static void subpel_free(Subpel* s) {
    free(s->F); free(s->B); free(s->H); free(s->J);
    memset(s, 0, sizeof(*s));
}

static int subpel_build(Subpel* s, const NvdrImage* ref) {
    memset(s, 0, sizeof(*s));
    s->w = ref->width; s->h = ref->height;
    s->sw = s->w + 2 * SP_PAD; s->sh = s->h + 2 * SP_PAD;
    size_t n = (size_t)s->sw * s->sh * 3;
    s->F = (uint8_t*)malloc(n); s->B = (uint8_t*)malloc(n);
    s->H = (uint8_t*)malloc(n); s->J = (uint8_t*)malloc(n);
    /* The unrounded horizontal sums, three rows beyond the plane either
     * way for the centre's vertical taps. */
    int rows = s->sh + 5;
    int* b1 = (int*)malloc(sizeof(int) * (size_t)s->sw * rows * 3);
    if (!s->F || !s->B || !s->H || !s->J || !b1) { free(b1); subpel_free(s); return -1; }
    const unsigned char* p = ref->pixels;
    int W = s->w, Hh = s->h;
#define P(X, Y, C) p[((size_t)clampi((Y), 0, Hh - 1) * W + clampi((X), 0, W - 1)) * 3 + (C)]
    for (int r = 0; r < rows; r++) {
        int y = r - SP_PAD - 2;
        for (int x = -SP_PAD; x < W + SP_PAD; x++)
            for (int c = 0; c < 3; c++)
                b1[((size_t)r * s->sw + (x + SP_PAD)) * 3 + c] =
                    P(x - 2, y, c) - 5 * P(x - 1, y, c) + 20 * P(x, y, c) +
                    20 * P(x + 1, y, c) - 5 * P(x + 2, y, c) + P(x + 3, y, c);
    }
    for (int y = -SP_PAD; y < Hh + SP_PAD; y++)
        for (int x = -SP_PAD; x < W + SP_PAD; x++)
            for (int c = 0; c < 3; c++) {
                size_t at = sp_at(s, x, y) + c;
                s->F[at] = (uint8_t)P(x, y, c);
                int r = y + SP_PAD + 2;
                const int* col = b1 + ((size_t)r * s->sw + (x + SP_PAD)) * 3 + c;
                size_t rs = (size_t)s->sw * 3;
                s->B[at] = (uint8_t)clip8i((col[0] + 16) >> 5);
                int h1 = P(x, y - 2, c) - 5 * P(x, y - 1, c) + 20 * P(x, y, c) +
                         20 * P(x, y + 1, c) - 5 * P(x, y + 2, c) + P(x, y + 3, c);
                s->H[at] = (uint8_t)clip8i((h1 + 16) >> 5);
                int j1 = col[-2 * (long)rs] - 5 * col[-(long)rs] + 20 * col[0] +
                         20 * col[rs] - 5 * col[2 * rs] + col[3 * rs];
                s->J[at] = (uint8_t)clip8i((j1 + 512) >> 10);
            }
#undef P
    free(b1);
    return 0;
}

/* One channel of the reference at (x + fx/4, y + fy/4). */
static inline int qsample(const Subpel* s, int x, int y, int fx, int fy, int c) {
    int X = x + (fx >> 2), Y = y + (fy >> 2);
    size_t at = sp_at(s, X, Y) + c;
    size_t right = at + 3, down = at + (size_t)s->sw * 3;
    switch (((fy & 3) << 2) | (fx & 3)) {
    case 0:  return s->F[at];
    case 1:  return (s->F[at] + s->B[at] + 1) >> 1;
    case 2:  return s->B[at];
    case 3:  return (s->B[at] + s->F[right] + 1) >> 1;
    case 4:  return (s->F[at] + s->H[at] + 1) >> 1;
    case 5:  return (s->B[at] + s->H[at] + 1) >> 1;
    case 6:  return (s->B[at] + s->J[at] + 1) >> 1;
    case 7:  return (s->B[at] + s->H[right] + 1) >> 1;
    case 8:  return s->H[at];
    case 9:  return (s->H[at] + s->J[at] + 1) >> 1;
    case 10: return s->J[at];
    case 11: return (s->J[at] + s->H[right] + 1) >> 1;
    case 12: return (s->H[at] + s->F[down] + 1) >> 1;
    case 13: return (s->H[at] + s->B[down] + 1) >> 1;
    case 14: return (s->J[at] + s->B[down] + 1) >> 1;
    default: return (s->H[right] + s->B[down] + 1) >> 1;
    }
}

static int block_sad_q(const NvdrImage* cur, const Subpel* ref,
                       int x0, int y0, int bw, int bh, int fx, int fy, int limit) {
    int acc = 0;
    for (int y = y0; y < y0 + bh; y++) {
        const unsigned char* a = cur->pixels + ((size_t)y * cur->width + x0) * 3;
        for (int x = x0; x < x0 + bw; x++, a += 3)
            for (int c = 0; c < 3; c++) {
                int d = a[c] - qsample(ref, x, y, fx, fy, c);
                acc += d < 0 ? -d : d;
            }
        if (acc >= limit) return acc;
    }
    return acc;
}

/* The whole-pixel vectors from block_search, refined to half and then to
 * quarter pixels around themselves. Blocks are independent. */
static void block_refine(const NvdrImage* cur, const Subpel* ref, int block,
                         int8_t* vx, int8_t* vy) {
    int nbx = (cur->width + block - 1) / block;
    int nby = (cur->height + block - 1) / block;
    int nb = nbx * nby;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
    for (int b = 0; b < nb; b++) {
        int x0 = (b % nbx) * block, y0 = (b / nbx) * block;
        int bw = cur->width - x0 < block ? cur->width - x0 : block;
        int bh = cur->height - y0 < block ? cur->height - y0 : block;
        int fx = clampi(vx[b] * 4, -127, 127), fy = clampi(vy[b] * 4, -127, 127);
        int best = block_sad_q(cur, ref, x0, y0, bw, bh, fx, fy, INT_MAX);
        for (int step = 2; step >= 1; step--) {
            int cx = fx, cy = fy;
            for (int dy = -step; dy <= step; dy += step)
                for (int dx = -step; dx <= step; dx += step) {
                    if (!dx && !dy) continue;
                    int tx = cx + dx, ty = cy + dy;
                    if (tx < -127 || tx > 127 || ty < -127 || ty > 127) continue;
                    int c = block_sad_q(cur, ref, x0, y0, bw, bh, tx, ty, best);
                    if (c < best) { best = c; fx = tx; fy = ty; }
                }
        }
        vx[b] = (int8_t)fx;
        vy[b] = (int8_t)fy;
    }
}

/* The reference assembled block by block, each block from its own
 * quarter-pixel vector. Encoder and decoder both build it here. */
static void block_predict(const Subpel* src, NvdrImage* dst, int block,
                          const int8_t* vx, const int8_t* vy) {
    int nbx = (dst->width + block - 1) / block;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < dst->height; y++) {
        const int8_t* rvx = vx + (size_t)(y / block) * nbx;
        const int8_t* rvy = vy + (size_t)(y / block) * nbx;
        unsigned char* out = dst->pixels + (size_t)y * dst->width * 3;
        for (int x = 0; x < dst->width; x++, out += 3) {
            int b = x / block;
            for (int c = 0; c < 3; c++) out[c] = (unsigned char)qsample(src, x, y, rvx[b], rvy[b], c);
        }
    }
}

/*
 * THE FIELD
 * ---------
 * Neighbouring blocks nearly always move together, so each vector is
 * coded against a prediction from the blocks already coded: the
 * component-wise median of the left, top and top-right neighbours, the
 * rule H.264 uses. A neighbour off the frame is replaced by the global
 * vector. On the top row there is only the left neighbour, which is then
 * the prediction on its own.
 *
 * A block whose vector equals its prediction costs one adaptive bit,
 * conditioned on whether its left and top neighbours were the same. On
 * a pan or a smooth zoom that bit is most of the field. Any other block
 * codes its difference from the prediction, x and then y, each as a zero
 * flag, a sign and an adaptive unary magnitude that escapes to eight direct
 * bits. When x is zero, y cannot be, since the block would have been a
 * match, so y skips its zero flag.
 *
 * This replaces two planes of deltas against the global vector, deflated.
 * Deflate sees runs; it cannot see that the vector above is the likeliest
 * value, and on the clean benchmark clip the field had become most of a
 * predicted frame.
 */
#define MV_MAG_CTX 6

typedef struct {
    uint16_t same[3];
    uint16_t zero[2];
    uint16_t sign[2];
    uint16_t mag[2][MV_MAG_CTX];
} MvModels;

static void mv_models_init(MvModels* m) {
    uint16_t* p = (uint16_t*)m;
    for (size_t i = 0; i < sizeof(*m) / sizeof(uint16_t); i++) p[i] = NVDR_PROB_INIT;
}

static int median3(int a, int b, int c) {
    if (a > b) { int t = a; a = b; b = t; }
    return c < a ? a : (c > b ? b : c);
}

static void mv_predict(const int8_t* vx, const int8_t* vy, int nbx, int b,
                       int gdx, int gdy, int* px, int* py) {
    int x = b % nbx, y = b / nbx;
    if (y == 0) {
        if (x > 0) { *px = vx[b - 1]; *py = vy[b - 1]; }
        else       { *px = gdx;       *py = gdy; }
        return;
    }
    int t = b - nbx;
    int lx = x > 0 ? vx[b - 1] : gdx, ly = x > 0 ? vy[b - 1] : gdy;
    int rx = x + 1 < nbx ? vx[t + 1] : gdx, ry = x + 1 < nbx ? vy[t + 1] : gdy;
    *px = median3(lx, vx[t], rx);
    *py = median3(ly, vy[t], ry);
}

/* The context of the match bit: how many of left and top matched. */
static int mv_same_ctx(const uint8_t* same, int nbx, int b) {
    int x = b % nbx, y = b / nbx;
    return (x > 0 && same[b - 1]) + (y > 0 && same[b - nbx]);
}

static void mv_enc_component(NvdrEncoder* enc, MvModels* m, int c, int v, int can_be_zero) {
    if (can_be_zero) {
        nvdr_enc_bit(enc, &m->zero[c], v != 0);
        if (!v) return;
    }
    nvdr_enc_bit(enc, &m->sign[c], v < 0);
    int remaining = (v < 0 ? -v : v) - 1;
    for (int i = 0; i < MV_MAG_CTX; i++) {
        int more = remaining > i;
        nvdr_enc_bit(enc, &m->mag[c][i], more);
        if (!more) return;
    }
    nvdr_enc_direct(enc, (uint32_t)(remaining - MV_MAG_CTX), 8);
}

static int mv_dec_component(NvdrDecoder* dec, MvModels* m, int c, int can_be_zero) {
    if (can_be_zero && !nvdr_dec_bit(dec, &m->zero[c])) return 0;
    int negative = nvdr_dec_bit(dec, &m->sign[c]);
    int remaining = 0, i = 0;
    for (; i < MV_MAG_CTX; i++) {
        if (!nvdr_dec_bit(dec, &m->mag[c][i])) break;
        remaining = i + 1;
    }
    if (i == MV_MAG_CTX) remaining = MV_MAG_CTX + (int)nvdr_dec_direct(dec, 8);
    return negative ? -(remaining + 1) : remaining + 1;
}

/* Roughly what a difference costs under the binarisation above, for the
 * search to weigh against the prediction error it saves. */
static int mv_component_bits(int v, int can_be_zero) {
    int bits = can_be_zero;
    if (!v) return bits;
    int a = v < 0 ? -v : v;
    return bits + 1 + (a <= MV_MAG_CTX ? a : MV_MAG_CTX + 8);
}

static int mv_bits(int dx, int dy) {
    if (!dx && !dy) return 1;
    return 1 + mv_component_bits(dx, 1) + mv_component_bits(dy, dx != 0);
}

static uint8_t* pack_field(const int8_t* vx, const int8_t* vy, int nbx, int nby,
                           int gdx, int gdy, size_t* out_len) {
    int nb = nbx * nby;
    uint8_t* same = (uint8_t*)malloc((size_t)nb);
    if (!same) return NULL;
    MvModels m;
    mv_models_init(&m);
    NvdrEncoder enc;
    if (nvdr_enc_init(&enc, (size_t)nb / 2 + 64) != 0) { free(same); return NULL; }
    for (int b = 0; b < nb; b++) {
        int px, py;
        mv_predict(vx, vy, nbx, b, gdx, gdy, &px, &py);
        int ex = vx[b] - px, ey = vy[b] - py;
        same[b] = !ex && !ey;
        nvdr_enc_bit(&enc, &m.same[mv_same_ctx(same, nbx, b)], !same[b]);
        if (same[b]) continue;
        mv_enc_component(&enc, &m, 0, ex, 1);
        mv_enc_component(&enc, &m, 1, ey, ex != 0);
    }
    free(same);
    if (nvdr_enc_finish(&enc) != 0) { nvdr_enc_free(&enc); return NULL; }
    *out_len = enc.count;
    return enc.bytes;      /* ownership passes to the caller */
}

/* A vector that does not fit the field's int8 is damage, not motion. */
static int unpack_field(const uint8_t* packed, size_t len, int nbx, int nby,
                        int gdx, int gdy, int8_t* vx, int8_t* vy) {
    int nb = nbx * nby;
    uint8_t* same = (uint8_t*)malloc((size_t)nb);
    if (!same) return -1;
    MvModels m;
    mv_models_init(&m);
    NvdrDecoder dec;
    nvdr_dec_init(&dec, packed, len);
    for (int b = 0; b < nb; b++) {
        int px, py;
        mv_predict(vx, vy, nbx, b, gdx, gdy, &px, &py);
        same[b] = !nvdr_dec_bit(&dec, &m.same[mv_same_ctx(same, nbx, b)]);
        int ex = 0, ey = 0;
        if (!same[b]) {
            ex = mv_dec_component(&dec, &m, 0, 1);
            ey = mv_dec_component(&dec, &m, 1, ex != 0);
        }
        int x = px + ex, y = py + ey;
        if (x < -128 || x > 127 || y < -128 || y > 127) { free(same); return -1; }
        vx[b] = (int8_t)x;
        vy[b] = (int8_t)y;
    }
    free(same);
    return 0;
}

/*
 * The search above picks each block's vector by prediction error alone.
 * This pass revisits the blocks in coding order, when the prediction each
 * one will be coded against is known, and trades error against the bits
 * the vector costs: a block keeps its own vector only if that beats the
 * predicted one, or a neighbour's, by more than the difference is worth.
 * It has to run in order, since every choice moves the prediction of the
 * blocks after it; it only evaluates a handful of candidates per block.
 */
static void field_rd(const NvdrImage* cur, const Subpel* ref, int block,
                     int gdx, int gdy, int lambda, int8_t* vx, int8_t* vy) {
    int nbx = (cur->width + block - 1) / block;
    int nby = (cur->height + block - 1) / block;
    for (int b = 0; b < nbx * nby; b++) {
        int x0 = (b % nbx) * block, y0 = (b / nbx) * block;
        int bw = cur->width - x0 < block ? cur->width - x0 : block;
        int bh = cur->height - y0 < block ? cur->height - y0 : block;
        int px, py;
        mv_predict(vx, vy, nbx, b, gdx, gdy, &px, &py);
        int cand[9][2] = {
            { px, py }, { vx[b], vy[b] }, { gdx, gdy },
            { b % nbx ? vx[b - 1] : gdx, b % nbx ? vy[b - 1] : gdy },
            { b >= nbx ? vx[b - nbx] : gdx, b >= nbx ? vy[b - nbx] : gdy },
            { px - 1, py }, { px + 1, py }, { px, py - 1 }, { px, py + 1 },
        };
        int best = INT_MAX, bx = px, by = py;
        for (int c = 0; c < 9; c++) {
            int cx = cand[c][0], cy = cand[c][1];
            int dup = 0;
            for (int k = 0; k < c; k++) dup |= cand[k][0] == cx && cand[k][1] == cy;
            if (dup) continue;
            int rate = lambda * mv_bits(cx - px, cy - py);
            if (rate >= best) continue;
            if (cx < -127 || cx > 127 || cy < -127 || cy > 127) continue;
            int cost = block_sad_q(cur, ref, x0, y0, bw, bh, cx, cy, best - rate) + rate;
            if (cost < best) { best = cost; bx = cx; by = cy; }
        }
        vx[b] = (int8_t)bx;
        vy[b] = (int8_t)by;
    }
}

static int field_blocks(int w, int h, int block) {
    return ((w + block - 1) / block) * ((h + block - 1) / block);
}

/* ------------------------------------------------------------ encoder */

struct NvdrvEncoder {
    FILE*       f;
    NvdrvConfig cfg;
    int         width, height;
    uint32_t    count;
    /* What the decoder holds after the last frame. The encoder predicts
     * from this and never from the source, because this is all the decoder
     * will have; predicting from the source is how a codec drifts. */
    NvdrImage   state;
    NvdrImage   scratch;   /* the shifted reference */
    NvdrImage   error;     /* the biased prediction error */
    int8_t*     vx;        /* block motion field, when enabled */
    int8_t*     vy;
};

static int alloc_image(NvdrImage* img, int w, int h) {
    img->width = w; img->height = h;
    img->pixels = (unsigned char*)calloc((size_t)w * h * 3, 1);
    return img->pixels ? 0 : -1;
}

/* Decode a frame's container at full quality, which is what the next
 * frame predicts from. */
static int reconstruct(const uint8_t* blob, size_t len, NvdrImage* out) {
    NvdrImage img;
    NvdrHeader hdr;
    if (nvdr_decode_mem(blob, len, -1, &img, &hdr, NULL) != 0) return -1;
    /* A frame's container has to be the size of the sequence it sits in,
     * or it is not a frame of this sequence. */
    if (hdr.width != out->width || hdr.height != out->height) {
        nvdr_image_free(&img);
        return -1;
    }
    memcpy(out->pixels, img.pixels, (size_t)out->width * out->height * 3);
    nvdr_image_free(&img);
    return 0;
}

int nvdrv_encode_open(NvdrvEncoder** out, const char* path,
                      int width, int height, const NvdrvConfig* cfg) {
    *out = NULL;
    NvdrvEncoder* e = (NvdrvEncoder*)calloc(1, sizeof(*e));
    if (!e) return -1;
    e->cfg = cfg ? *cfg : nvdrv_default_config();
    e->width = width; e->height = height;

    if (e->cfg.block < 0) e->cfg.block = (long)width * height >= 200000 ? 16 : 8;
    if (e->cfg.block > 128) e->cfg.block = 0;
    if (alloc_image(&e->state, width, height) != 0 ||
        alloc_image(&e->scratch, width, height) != 0 ||
        alloc_image(&e->error, width, height) != 0) {
        nvdrv_encode_close(e); return -1;
    }
    if (e->cfg.block > 0) {
        int nb = field_blocks(width, height, e->cfg.block);
        e->vx = (int8_t*)malloc((size_t)nb);
        e->vy = (int8_t*)malloc((size_t)nb);
        if (!e->vx || !e->vy) { nvdrv_encode_close(e); return -1; }
    }

    e->f = fopen(path, "wb");
    if (!e->f) { nvdrv_encode_close(e); return -1; }

    uint8_t h[NVDRV_HEADER_SIZE];
    memset(h, 0, sizeof(h));
    memcpy(h, NVDRV_MAGIC, 4);
    h[4] = NVDRV_VERSION;
    put_u16v(h + 6, (uint16_t)width);
    put_u16v(h + 8, (uint16_t)height);
    /* Frame count is patched on close; a stream that never closes still
     * decodes, since frames are self-delimiting. */
    put_u32v(h + 10, 0);
    h[14] = (uint8_t)(e->cfg.fps > 0 && e->cfg.fps < 256 ? e->cfg.fps : 24);
    h[15] = (uint8_t)(e->cfg.gop > 255 ? 255 : e->cfg.gop);
    if (fwrite(h, 1, sizeof(h), e->f) != sizeof(h)) { nvdrv_encode_close(e); return -1; }

    *out = e;
    return 0;
}

/* A predicted frame with a block field carries it ahead of the container:
 * [u32 field length][field][container]. The header's second byte is the
 * block size, zero when there is no field. */
static int write_frame(NvdrvEncoder* e, int kind, int dx, int dy, int block,
                       const uint8_t* field, size_t field_len,
                       const uint8_t* payload, size_t len) {
    uint8_t h[NVDRV_FRAME_HEADER];
    memset(h, 0, sizeof(h));
    h[0] = (uint8_t)kind;
    h[1] = (uint8_t)block;
    put_u16v(h + 2, (uint16_t)(int16_t)dx);
    put_u16v(h + 4, (uint16_t)(int16_t)dy);
    size_t body = len + (block ? 4 + field_len : 0);
    put_u32v(h + 6, (uint32_t)body);
    if (fwrite(h, 1, sizeof(h), e->f) != sizeof(h)) return -1;
    if (block) {
        uint8_t n[4];
        put_u32v(n, (uint32_t)field_len);
        if (fwrite(n, 1, 4, e->f) != 4) return -1;
        if (fwrite(field, 1, field_len, e->f) != field_len) return -1;
    }
    if (fwrite(payload, 1, len, e->f) != len) return -1;
    return 0;
}

int nvdrv_encode_frame(NvdrvEncoder* e, const NvdrImage* frame,
                       int* kind_out, size_t* bytes_out,
                       int* dx_out, int* dy_out) {
    if (frame->width != e->width || frame->height != e->height) return -1;
    size_t npx = (size_t)e->width * e->height * 3;

    int forced_intra = (e->count == 0) ||
                       (e->cfg.gop > 0 && (int)(e->count % (uint32_t)e->cfg.gop) == 0);

    int dx = 0, dy = 0;
    const NvdrImage* ref = NULL;
    double mad = 1e30;

    int block = 0;
    uint8_t* field = NULL;
    size_t field_len = 0;

    if (!forced_intra) {
        find_shift(frame, &e->state, e->cfg.search, &dx, &dy);
        if (e->cfg.block > 0) {
            block = e->cfg.block;
            block_search(frame, &e->state, block, dx, dy, e->vx, e->vy);
            Subpel sp;
            if (subpel_build(&sp, &e->state) != 0) return -1;
            block_refine(frame, &sp, block, e->vx, e->vy);
            if (e->cfg.mv_lambda > 0)
                field_rd(frame, &sp, block, dx * 4, dy * 4, e->cfg.mv_lambda, e->vx, e->vy);
            block_predict(&sp, &e->scratch, block, e->vx, e->vy);
            subpel_free(&sp);
            ref = &e->scratch;
        } else if (dx || dy) {
            shift_into(&e->state, &e->scratch, dx, dy); ref = &e->scratch;
        } else {
            ref = &e->state;
        }

        double acc = 0.0;
        for (size_t i = 0; i < npx; i++) {
            int d = (int)frame->pixels[i] - (int)ref->pixels[i];
            acc += d < 0 ? -d : d;
        }
        mad = acc / (double)npx;
    }

    /* A cut, or anything else the bias to 128 would clip, goes intra. */
    int kind = (forced_intra || mad > e->cfg.intra_threshold)
               ? NVDRV_INTRA : NVDRV_PRED;

    const NvdrImage* to_code = frame;
    if (kind == NVDRV_PRED) {
        for (size_t i = 0; i < npx; i++)
            e->error.pixels[i] =
                (unsigned char)clamp255v((int)frame->pixels[i] - (int)ref->pixels[i] + 128);
        to_code = &e->error;
    } else {
        dx = dy = 0;
        block = 0;
    }

    if (block) {
        field = pack_field(e->vx, e->vy, (e->width + block - 1) / block,
                           (e->height + block - 1) / block, dx * 4, dy * 4, &field_len);
        if (!field) return -1;
    }

    NvdrConfig fcfg = e->cfg.frame;
    if (kind == NVDRV_PRED) {
        fcfg.residual = 1;
        /* The filter smooths seams in a picture; a residual is not one,
         * and its seams are not what the viewer sees. */
        fcfg.deblock = 0;
        /* Coarser than the intra frames by 1.2 unless told otherwise: on the
         * clean clip that was 0.1 dB better at equal rate across q 16-40. */
        fcfg.q = e->cfg.pred_q > 0 ? e->cfg.pred_q : (e->cfg.frame.q * 6 + 2) / 5;
    }

    uint8_t* blob = NULL;
    size_t len = 0;
    NvdrHeader fh;
    if (nvdr_encode_mem(&blob, &len, to_code, &fcfg, &fh) != 0) { free(field); return -1; }

    if (write_frame(e, kind, dx, dy, block, field, field_len, blob, len) != 0) {
        free(blob); free(field); return -1;
    }
    free(field);

    /* Carry the decoder's state forward by decoding what was just written,
     * so the two sides hold the same bytes from here on. */
    int rc;
    if (kind == NVDRV_INTRA) {
        rc = reconstruct(blob, len, &e->state);
    } else {
        rc = reconstruct(blob, len, &e->error);
        if (rc == 0) {
            /* `ref` may alias e->state, so the sum is written into scratch
             * and swapped in rather than updated in place. */
            for (size_t i = 0; i < npx; i++)
                e->scratch.pixels[i] = (unsigned char)clamp255v(
                    (int)e->error.pixels[i] - 128 + (int)ref->pixels[i]);
            unsigned char* tmp = e->state.pixels;
            e->state.pixels = e->scratch.pixels;
            e->scratch.pixels = tmp;
        }
    }
    free(blob);
    if (rc != 0) return -1;

    e->count++;
    if (kind_out) *kind_out = kind;
    if (bytes_out) *bytes_out = len + NVDRV_FRAME_HEADER + (block ? 4 + field_len : 0);
    if (dx_out) *dx_out = dx;
    if (dy_out) *dy_out = dy;
    return 0;
}

int nvdrv_encode_close(NvdrvEncoder* e) {
    int rc = 0;
    if (!e) return 0;
    if (e->f) {
        if (fseek(e->f, 10, SEEK_SET) == 0) {
            uint8_t n[4];
            put_u32v(n, e->count);
            if (fwrite(n, 1, 4, e->f) != 4) rc = -1;
        }
        if (fclose(e->f) != 0) rc = -1;
    }
    free(e->state.pixels);
    free(e->scratch.pixels);
    free(e->error.pixels);
    free(e->vx);
    free(e->vy);
    free(e);
    return rc;
}

/* ------------------------------------------------------------ decoder */

struct NvdrvDecoder {
    uint8_t*  data;
    size_t    size, pos;
    int       width, height;
    NvdrImage state;
    NvdrImage scratch;
};

int nvdrv_decode_open(NvdrvDecoder** out, const char* path, NvdrvInfo* info) {
    *out = NULL;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size < NVDRV_HEADER_SIZE) { fclose(f); return -1; }

    uint8_t* data = (uint8_t*)malloc((size_t)size);
    if (!data) { fclose(f); return -1; }
    size_t got = fread(data, 1, (size_t)size, f);
    fclose(f);
    if (got < NVDRV_HEADER_SIZE ||
        memcmp(data, NVDRV_MAGIC, 4) != 0 || data[4] != NVDRV_VERSION) {
        free(data); return -1;
    }

    NvdrvDecoder* d = (NvdrvDecoder*)calloc(1, sizeof(*d));
    if (!d) { free(data); return -1; }
    d->data = data; d->size = got; d->pos = NVDRV_HEADER_SIZE;
    d->width = get_u16v(data + 6);
    d->height = get_u16v(data + 8);
    /* Every frame buffer is allocated at this size before a single frame
     * is read, so a damaged header gets refused here rather than trusted. */
    if (d->width <= 0 || d->height <= 0 ||
        (size_t)d->width * d->height > NVDR_MAX_PIXELS ||
        alloc_image(&d->state, d->width, d->height) != 0 ||
        alloc_image(&d->scratch, d->width, d->height) != 0) {
        nvdrv_decode_close(d); return -1;
    }
    if (info) {
        info->width = d->width;
        info->height = d->height;
        info->frame_count = (int)get_u32v(data + 10);
        info->fps = data[14];
        info->gop = data[15];
    }
    *out = d;
    return 0;
}

int nvdrv_decode_next(NvdrvDecoder* d, NvdrImage* out,
                      int* kind_out, int* partial_out) {
    if (partial_out) *partial_out = 0;
    if (d->pos + NVDRV_FRAME_HEADER > d->size) return 0;

    const uint8_t* h = d->data + d->pos;
    int kind = h[0];
    int block = h[1];
    int dx = (int16_t)get_u16v(h + 2);
    int dy = (int16_t)get_u16v(h + 4);
    size_t len = get_u32v(h + 6);
    if (kind != NVDRV_INTRA && kind != NVDRV_PRED) return -1;
    if (kind == NVDRV_INTRA && block) return -1;
    d->pos += NVDRV_FRAME_HEADER;

    size_t npx = (size_t)d->width * d->height * 3;
    const NvdrImage* ref = &d->state;

    if (kind == NVDRV_PRED && block) {
        /* The field has to arrive whole: without it there is no reference
         * to add the residual to, so a cut inside it ends the stream. */
        if (d->pos + 4 > d->size || len < 4) return 0;
        size_t field_len = get_u32v(d->data + d->pos);
        if (field_len > len - 4 || d->pos + 4 + field_len > d->size) return 0;
        int nb = field_blocks(d->width, d->height, block);
        int8_t* vx = (int8_t*)malloc((size_t)nb);
        int8_t* vy = (int8_t*)malloc((size_t)nb);
        if (!vx || !vy ||
            unpack_field(d->data + d->pos + 4, field_len, (d->width + block - 1) / block,
                         (d->height + block - 1) / block, dx * 4, dy * 4, vx, vy) != 0) {
            free(vx); free(vy); return -1;
        }
        Subpel sp;
        if (subpel_build(&sp, &d->state) != 0) { free(vx); free(vy); return -1; }
        block_predict(&sp, &d->scratch, block, vx, vy);
        subpel_free(&sp);
        free(vx); free(vy);
        ref = &d->scratch;
        d->pos += 4 + field_len;
        len -= 4 + field_len;
    } else if (kind == NVDRV_PRED && (dx || dy)) {
        shift_into(&d->state, &d->scratch, dx, dy);
        ref = &d->scratch;
    }

    /* A cut file ends mid-frame. The still decoder reads as far as the
     * bytes reach, so the last frame is shown at whatever quality arrived
     * instead of being dropped. */
    size_t have = d->size - d->pos;
    int partial = 0;
    if (len > have) { len = have; partial = 1; }
    if (len == 0) return 0;

    if (kind == NVDRV_INTRA) {
        /* A frame cut before its colour layer has no picture yet: the
         * stream ends there, it is not damaged. */
        if (reconstruct(d->data + d->pos, len, &d->state) != 0) return partial ? 0 : -1;
    } else {
        NvdrImage err;
        err.width = d->width; err.height = d->height;
        err.pixels = (unsigned char*)malloc(npx);
        if (!err.pixels) return -1;
        if (reconstruct(d->data + d->pos, len, &err) != 0) { free(err.pixels); return partial ? 0 : -1; }
        for (size_t i = 0; i < npx; i++)
            err.pixels[i] = (unsigned char)clamp255v(
                (int)err.pixels[i] - 128 + (int)ref->pixels[i]);
        memcpy(d->state.pixels, err.pixels, npx);
        free(err.pixels);
    }

    d->pos += len;
    memcpy(out->pixels, d->state.pixels, npx);
    if (kind_out) *kind_out = kind;
    if (partial_out) *partial_out = partial;
    return 1;
}

void nvdrv_decode_close(NvdrvDecoder* d) {
    if (!d) return;
    free(d->data);
    free(d->state.pixels);
    free(d->scratch.pixels);
    free(d);
}
