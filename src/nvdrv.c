#include "nvdrv.h"

#include <limits.h>
#include <math.h>
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
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
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
    c.bframes = 7;
    c.b_q_step = 0.5f;
    c.lookahead = 16;
    c.tpl_strength = 1.0f;
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

/* Around (cx, cy) instead of zero: a reference several frames away is
 * searched around the motion the frames in between add up to. */
static void find_shift_at(const NvdrImage* cur, const NvdrImage* ref, int cx0, int cy0, int range,
                          int* bdx, int* bdy) {
    *bdx = cx0; *bdy = cy0;
    if (range <= 0) return;
    double best = shift_cost(cur, ref, cx0, cy0, 4, 1e30);
    for (int dy = cy0 - range; dy <= cy0 + range; dy += 2)
        for (int dx = cx0 - range; dx <= cx0 + range; dx += 2) {
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

static void find_shift(const NvdrImage* cur, const NvdrImage* ref, int range,
                       int* bdx, int* bdy) {
    find_shift_at(cur, ref, 0, 0, range, bdx, bdy);
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

/*
 * A reference several frames away puts what moves against the camera
 * well outside those windows: an object crossing at 5 px a frame is 40 px
 * off by an anchor eight frames on. So when the distance is more than one
 * frame, every block also searches a coarse copy of both frames, averaged
 * 4x4 (2x2 for small blocks) and summed over the three channels, over a
 * range that grows with the distance, and the full-resolution search then
 * looks around the coarse winner too. Averaging before searching is what
 * keeps a textured block from matching an alias of itself.
 */
typedef struct { int w, h, f; int* p; } Coarse;

static int coarse_build(Coarse* c, const NvdrImage* img, int f) {
    c->f = f; c->w = img->width / f; c->h = img->height / f;
    c->p = (int*)malloc(sizeof(int) * (size_t)(c->w > 0 ? c->w : 1) * (c->h > 0 ? c->h : 1));
    if (!c->p) return -1;
    for (int y = 0; y < c->h; y++)
        for (int x = 0; x < c->w; x++) {
            int acc = 0;
            for (int j = 0; j < f; j++) {
                const unsigned char* r = img->pixels + ((size_t)(y * f + j) * img->width + (size_t)x * f) * 3;
                for (int i = 0; i < 3 * f; i++) acc += r[i];
            }
            c->p[y * c->w + x] = acc;
        }
    return 0;
}

static int coarse_sad(const Coarse* a, const Coarse* b, int x0, int y0, int n, int dx, int dy, int limit) {
    int acc = 0;
    for (int y = y0; y < y0 + n && y < a->h; y++) {
        int sy = clampi(y + dy, 0, b->h - 1);
        for (int x = x0; x < x0 + n && x < a->w; x++) {
            int d = a->p[y * a->w + x] - b->p[sy * b->w + clampi(x + dx, 0, b->w - 1)];
            acc += d < 0 ? -d : d;
        }
        if (acc >= limit) return acc;
    }
    return acc;
}

static void block_search(const NvdrImage* cur, const NvdrImage* ref, int block, int dist,
                         int gdx, int gdy, int16_t* vx, int16_t* vy) {
    int nbx = (cur->width + block - 1) / block;
    int nby = (cur->height + block - 1) / block;
    int nb = nbx * nby;
    int f = block >= 16 ? 4 : 2;
    Coarse ca, cb;
    ca.p = cb.p = NULL;
    int coarse = dist > 1 && cur->width >= 4 * f && cur->height >= 4 * f &&
                 coarse_build(&ca, cur, f) == 0 && coarse_build(&cb, ref, f) == 0;
    /* 8 px a frame, and no further than a vector can reach. */
    int range = (8 * dist < NVDRV_MV_MAX / 4 - 16 ? 8 * dist : NVDRV_MV_MAX / 4 - 16) / f;
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
        int centres = 2, cxs[3] = { gdx, 0, 0 }, cys[3] = { gdy, 0, 0 };
        if (coarse) {
            int n = block / f, gx = gdx / f, gy = gdy / f;
            int cbest = INT_MAX, kx = gx, ky = gy;
            for (int dy = gy - range; dy <= gy + range; dy++)
                for (int dx = gx - range; dx <= gx + range; dx++) {
                    int c = coarse_sad(&ca, &cb, x0 / f, y0 / f, n, dx, dy, cbest);
                    if (c < cbest) { cbest = c; kx = dx; ky = dy; }
                }
            cxs[2] = kx * f; cys[2] = ky * f; centres = 3;
        }
        for (int pass = 0; pass < centres; pass++) {
            int cx = cxs[pass], cy = cys[pass];
            for (int dy = cy - NVDRV_LOCAL_RANGE; dy <= cy + NVDRV_LOCAL_RANGE; dy++)
                for (int dx = cx - NVDRV_LOCAL_RANGE; dx <= cx + NVDRV_LOCAL_RANGE; dx++) {
                    if (dx == gdx && dy == gdy) continue;
                    int limit = best - bias;
                    if (limit <= 0) continue;
                    int c = block_sad(cur, ref, x0, y0, bw, bh, dx, dy, limit);
                    if (c + bias < best) { best = c + bias; bdx = dx; bdy = dy; }
                }
        }
        vx[b] = (int16_t)bdx;
        vy[b] = (int16_t)bdy;
    }
    free(ca.p); free(cb.p);
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
 * a margin wide enough for any vector the field can hold (NVDRV_MV_MAX
 * quarter pixels) plus the filter's reach. The global vector stays in
 * whole pixels; the field is predicted from it multiplied by four.
 */

#define SP_PAD (NVDRV_MV_MAX / 4 + 8)

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
static void block_refine(const NvdrImage* cur, const Subpel* ref, int block, int lim,
                         int16_t* vx, int16_t* vy) {
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
        int fx = clampi(vx[b] * 4, -lim, lim), fy = clampi(vy[b] * 4, -lim, lim);
        int best = block_sad_q(cur, ref, x0, y0, bw, bh, fx, fy, INT_MAX);
        for (int step = 2; step >= 1; step--) {
            int cx = fx, cy = fy;
            for (int dy = -step; dy <= step; dy += step)
                for (int dx = -step; dx <= step; dx += step) {
                    if (!dx && !dy) continue;
                    int tx = cx + dx, ty = cy + dy;
                    if (tx < -lim || tx > lim || ty < -lim || ty > lim) continue;
                    int c = block_sad_q(cur, ref, x0, y0, bw, bh, tx, ty, best);
                    if (c < best) { best = c; fx = tx; fy = ty; }
                }
        }
        vx[b] = (int16_t)fx;
        vy[b] = (int16_t)fy;
    }
}

/* The reference assembled block by block, each block from its own
 * quarter-pixel vector. Encoder and decoder both build it here. */
static void block_predict(const Subpel* src, NvdrImage* dst, int block,
                          const int16_t* vx, const int16_t* vy) {
    int nbx = (dst->width + block - 1) / block;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < dst->height; y++) {
        const int16_t* rvx = vx + (size_t)(y / block) * nbx;
        const int16_t* rvy = vy + (size_t)(y / block) * nbx;
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

/*
 * The prediction is made relative to the frame's motion model (see "The
 * motion model" below): `mx`, `my` hold the model's vector at each block,
 * and the neighbours contribute how far they stray from the model there.
 * A block that follows the model under neighbours that do costs one
 * cheap bit, whether the model is a pan, a zoom or a rotation. With a
 * model that is one translation this is exactly the median of the
 * neighbours' vectors, a neighbour off the frame counting as the global
 * vector, which is what albums use.
 */
static void mv_predict(const int16_t* vx, const int16_t* vy, const int16_t* mx, const int16_t* my,
                       int nbx, int b, int* px, int* py) {
    int x = b % nbx, y = b / nbx;
    if (y == 0) {
        if (x > 0) { *px = mx[b] + vx[b - 1] - mx[b - 1]; *py = my[b] + vy[b - 1] - my[b - 1]; }
        else       { *px = mx[b];                         *py = my[b]; }
        return;
    }
    int t = b - nbx;
    int lx = x > 0 ? vx[b - 1] - mx[b - 1] : 0, ly = x > 0 ? vy[b - 1] - my[b - 1] : 0;
    int rx = x + 1 < nbx ? vx[t + 1] - mx[t + 1] : 0, ry = x + 1 < nbx ? vy[t + 1] - my[t + 1] : 0;
    *px = mx[b] + median3(lx, vx[t] - mx[t], rx);
    *py = my[b] + median3(ly, vy[t] - my[t], ry);
}

/* The context of the match bit: how many of left and top matched. */
static int mv_same_ctx(const uint8_t* same, int nbx, int b) {
    int x = b % nbx, y = b / nbx;
    return (x > 0 && same[b - 1]) + (y > 0 && same[b - nbx]);
}

/* How many direct bits a magnitude past the adaptive ones escapes to: 8
 * in an album, whose vectors fit int8, and 10 in a sequence, whose
 * anchors sit several frames apart (see NVDRV_MV_MAX). */
#define MV_ESC_ALBUM 8
#define MV_ESC_SEQ   10

static void mv_enc_component(NvdrEncoder* enc, MvModels* m, int c, int v, int can_be_zero, int esc) {
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
    nvdr_enc_direct(enc, (uint32_t)(remaining - MV_MAG_CTX), esc);
}

static int mv_dec_component(NvdrDecoder* dec, MvModels* m, int c, int can_be_zero, int esc) {
    if (can_be_zero && !nvdr_dec_bit(dec, &m->zero[c])) return 0;
    int negative = nvdr_dec_bit(dec, &m->sign[c]);
    int remaining = 0, i = 0;
    for (; i < MV_MAG_CTX; i++) {
        if (!nvdr_dec_bit(dec, &m->mag[c][i])) break;
        remaining = i + 1;
    }
    if (i == MV_MAG_CTX) remaining = MV_MAG_CTX + (int)nvdr_dec_direct(dec, esc);
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

/* One block's vector against its prediction; `same` holds, per block,
 * whether it matched. */
static void enc_vector(NvdrEncoder* enc, MvModels* m, uint8_t* same, const int16_t* vx,
                       const int16_t* vy, const int16_t* mx, const int16_t* my, int nbx, int b, int esc) {
    int px, py;
    mv_predict(vx, vy, mx, my, nbx, b, &px, &py);
    int ex = vx[b] - px, ey = vy[b] - py;
    same[b] = !ex && !ey;
    nvdr_enc_bit(enc, &m->same[mv_same_ctx(same, nbx, b)], !same[b]);
    if (same[b]) return;
    mv_enc_component(enc, m, 0, ex, 1, esc);
    mv_enc_component(enc, m, 1, ey, ex != 0, esc);
}

/* The decoder's side; -1 for a vector outside -lim-1 .. lim quarter
 * pixels (int8 for lim 127), which is damage, not motion. */
static int dec_vector(NvdrDecoder* dec, MvModels* m, uint8_t* same, int16_t* vx, int16_t* vy,
                      const int16_t* mx, const int16_t* my, int nbx, int b, int esc, int lim) {
    int px, py;
    mv_predict(vx, vy, mx, my, nbx, b, &px, &py);
    same[b] = !nvdr_dec_bit(dec, &m->same[mv_same_ctx(same, nbx, b)]);
    int ex = 0, ey = 0;
    if (!same[b]) {
        ex = mv_dec_component(dec, m, 0, 1, esc);
        ey = mv_dec_component(dec, m, 1, ex != 0, esc);
    }
    int x = px + ex, y = py + ey;
    if (x < -lim - 1 || x > lim || y < -lim - 1 || y > lim) return -1;
    vx[b] = (int16_t)x;
    vy[b] = (int16_t)y;
    return 0;
}

/* A block that does not use a list takes that list's prediction as its
 * vector, at no cost, so its neighbours' predictions stay continuous. */
static void inherit_vector(uint8_t* same, int16_t* vx, int16_t* vy, const int16_t* mx,
                           const int16_t* my, int nbx, int b) {
    int px, py;
    mv_predict(vx, vy, mx, my, nbx, b, &px, &py);
    vx[b] = (int16_t)px; vy[b] = (int16_t)py;
    same[b] = 1;
}

static uint8_t* pack_field(const int16_t* vx, const int16_t* vy, const int16_t* mx, const int16_t* my,
                           int nbx, int nby, int esc, size_t* out_len) {
    int nb = nbx * nby;
    uint8_t* same = (uint8_t*)malloc((size_t)nb);
    if (!same) return NULL;
    MvModels m;
    mv_models_init(&m);
    NvdrEncoder enc;
    if (nvdr_enc_init(&enc, (size_t)nb / 2 + 64) != 0) { free(same); return NULL; }
    for (int b = 0; b < nb; b++) enc_vector(&enc, &m, same, vx, vy, mx, my, nbx, b, esc);
    free(same);
    if (nvdr_enc_finish(&enc) != 0) { nvdr_enc_free(&enc); return NULL; }
    *out_len = enc.count;
    return enc.bytes;      /* ownership passes to the caller */
}

static int unpack_field(const uint8_t* packed, size_t len, int nbx, int nby, const int16_t* mx,
                        const int16_t* my, int esc, int lim, int16_t* vx, int16_t* vy) {
    int nb = nbx * nby;
    uint8_t* same = (uint8_t*)malloc((size_t)nb);
    if (!same) return -1;
    MvModels m;
    mv_models_init(&m);
    NvdrDecoder dec;
    nvdr_dec_init(&dec, packed, len);
    for (int b = 0; b < nb; b++)
        if (dec_vector(&dec, &m, same, vx, vy, mx, my, nbx, b, esc, lim) != 0) {
            free(same); return -1;
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
static void field_rd(const NvdrImage* cur, const Subpel* ref, int block, int lim,
                     const int16_t* mx, const int16_t* my, int lambda, int16_t* vx, int16_t* vy) {
    int nbx = (cur->width + block - 1) / block;
    int nby = (cur->height + block - 1) / block;
    for (int b = 0; b < nbx * nby; b++) {
        int x0 = (b % nbx) * block, y0 = (b / nbx) * block;
        int bw = cur->width - x0 < block ? cur->width - x0 : block;
        int bh = cur->height - y0 < block ? cur->height - y0 : block;
        int px, py;
        mv_predict(vx, vy, mx, my, nbx, b, &px, &py);
        int cand[9][2] = {
            { px, py }, { vx[b], vy[b] }, { mx[b], my[b] },
            { b % nbx ? vx[b - 1] : mx[b], b % nbx ? vy[b - 1] : my[b] },
            { b >= nbx ? vx[b - nbx] : mx[b], b >= nbx ? vy[b - nbx] : my[b] },
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
            if (cx < -lim || cx > lim || cy < -lim || cy > lim) continue;
            int cost = block_sad_q(cur, ref, x0, y0, bw, bh, cx, cy, best - rate) + rate;
            if (cost < best) { best = cost; bx = cx; by = cy; }
        }
        vx[b] = (int16_t)bx;
        vy[b] = (int16_t)by;
    }
}

static int field_blocks(int w, int h, int block) {
    return ((w + block - 1) / block) * ((h + block - 1) / block);
}

/* --------------------------------------------------- the motion model */

/*
 * THE MOTION MODEL
 * ----------------
 * A camera that zooms or turns moves every block by a slightly different
 * vector, varying smoothly across the frame, and a median of neighbours
 * pays for each step of that gradient: on a slow zoom the field was two
 * thirds of every predicted frame. So each field in a sequence carries an
 * affine model of the frame's motion,
 *
 *     mx = a0 + (a1 * cx + a2 * cy + 32768) >> 16
 *     my = b0 + (b1 * cx + b2 * cy + 32768) >> 16
 *
 * in quarter pixels at the block's centre (cx, cy), with a0 and b0 in
 * quarter pixels and the four slopes in 65536ths of a quarter pixel per
 * pixel, each an int16 ahead of the field (12 bytes a list; a list whose
 * model is the frame header's global vector, a plain translation, is
 * flagged in the header instead and costs nothing). Vectors are
 * then predicted as the model plus how far the neighbours stray from it
 * (see mv_predict). The encoder fits the model to its searched vectors by
 * least squares, twice more on the blocks that agree with the last fit,
 * so a moving object does not drag the camera's motion with it.
 */
#define MODEL_BYTES 12
/* Frame header byte 18: which lists carry an affine model. */
#define FRAME_MODEL0 1
#define FRAME_MODEL1 2

typedef struct { int a0, a1, a2, b0, b1, b2; } Model;

static void model_fill(const Model* m, int w, int h, int block, int16_t* mx, int16_t* my) {
    int nbx = (w + block - 1) / block, nby = (h + block - 1) / block;
    for (int by = 0; by < nby; by++)
        for (int bx = 0; bx < nbx; bx++) {
            int cx = bx * block + block / 2, cy = by * block + block / 2;
            int b = by * nbx + bx;
            /* In 64 bits: a slope times a coordinate overflows 32 on a
             * frame 65535 wide, and a damaged file can say that. */
            int64_t sx = ((int64_t)m->a1 * cx + (int64_t)m->a2 * cy + 32768) >> 16;
            int64_t sy = ((int64_t)m->b1 * cx + (int64_t)m->b2 * cy + 32768) >> 16;
            int64_t vx = m->a0 + sx, vy = m->b0 + sy;
            mx[b] = (int16_t)(vx < -NVDRV_MV_MAX ? -NVDRV_MV_MAX : vx > NVDRV_MV_MAX ? NVDRV_MV_MAX : vx);
            my[b] = (int16_t)(vy < -NVDRV_MV_MAX ? -NVDRV_MV_MAX : vy > NVDRV_MV_MAX ? NVDRV_MV_MAX : vy);
        }
}

static void model_put(const Model* m, uint8_t* p) {
    const int v[6] = { m->a0, m->a1, m->a2, m->b0, m->b1, m->b2 };
    for (int i = 0; i < 6; i++) put_u16v(p + 2 * i, (uint16_t)(int16_t)v[i]);
}

static void model_get(Model* m, const uint8_t* p) {
    m->a0 = (int16_t)get_u16v(p);     m->a1 = (int16_t)get_u16v(p + 2);
    m->a2 = (int16_t)get_u16v(p + 4); m->b0 = (int16_t)get_u16v(p + 6);
    m->b1 = (int16_t)get_u16v(p + 8); m->b2 = (int16_t)get_u16v(p + 10);
}

/* Least squares of v = c0 + c1 x + c2 y over the blocks in `use`. */
static int fit_plane(const int16_t* v, const uint8_t* use, int nbx, int nby, int block, double c[3]) {
    double A[3][4] = { { 0 } };
    int n = 0;
    for (int b = 0; b < nbx * nby; b++) {
        if (!use[b]) continue;
        double p[3] = { 1.0, (b % nbx) * block + block / 2, (b / nbx) * block + block / 2 };
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) A[i][j] += p[i] * p[j];
            A[i][3] += p[i] * v[b];
        }
        n++;
    }
    if (n < 6) return -1;
    for (int i = 0; i < 3; i++) {
        int piv = i;
        for (int r = i + 1; r < 3; r++) if (fabs(A[r][i]) > fabs(A[piv][i])) piv = r;
        for (int k = 0; k < 4; k++) { double t = A[i][k]; A[i][k] = A[piv][k]; A[piv][k] = t; }
        if (fabs(A[i][i]) < 1e-9) return -1;
        for (int r = 0; r < 3; r++) {
            if (r == i) continue;
            double f = A[r][i] / A[i][i];
            for (int k = i; k < 4; k++) A[r][k] -= f * A[i][k];
        }
    }
    for (int i = 0; i < 3; i++) c[i] = A[i][3] / A[i][i];
    return 0;
}

static int round_clamp(double v, int lim) {
    long r = lrint(v);
    return (int)(r < -lim ? -lim : (r > lim ? lim : r));
}

/* The model of a searched field; one translation by (gx, gy) quarter
 * pixels when the fit fails. */
static int model_fit(const int16_t* vx, const int16_t* vy, int w, int h, int block,
                     int gx, int gy, Model* m) {
    int nbx = (w + block - 1) / block, nby = (h + block - 1) / block, nb = nbx * nby;
    m->a0 = gx; m->b0 = gy; m->a1 = m->a2 = m->b1 = m->b2 = 0;
    uint8_t* use = (uint8_t*)malloc((size_t)nb);
    int* dev = (int*)malloc(sizeof(int) * (size_t)nb);
    int affine = 0;
    if (!use || !dev) { free(use); free(dev); return 0; }
    memset(use, 1, (size_t)nb);
    double cx[3], cy[3];
    int ok = 0;
    for (int pass = 0; pass < 3; pass++) {
        if (fit_plane(vx, use, nbx, nby, block, cx) != 0 || fit_plane(vy, use, nbx, nby, block, cy) != 0) break;
        ok = 1;
        /* Keep the blocks within 2.5 times the median deviation, and never
         * reject a block within one pixel. */
        for (int b = 0; b < nb; b++) {
            double px = (b % nbx) * block + block / 2, py = (b / nbx) * block + block / 2;
            double ex = vx[b] - (cx[0] + cx[1] * px + cx[2] * py);
            double ey = vy[b] - (cy[0] + cy[1] * px + cy[2] * py);
            dev[b] = (int)(fabs(ex) + fabs(ey));
        }
        int hist[64] = { 0 }, med = 0, acc = 0;
        for (int b = 0; b < nb; b++) hist[dev[b] < 63 ? dev[b] : 63]++;
        while (med < 63 && (acc += hist[med]) < nb / 2) med++;
        int keep = med * 5 / 2 > 4 ? med * 5 / 2 : 4;
        for (int b = 0; b < nb; b++) use[b] = dev[b] <= keep;
    }
    if (ok) {
        Model a;
        a.a1 = round_clamp(cx[1] * 65536.0, 32767); a.a2 = round_clamp(cx[2] * 65536.0, 32767);
        a.b1 = round_clamp(cy[1] * 65536.0, 32767); a.b2 = round_clamp(cy[2] * 65536.0, 32767);
        a.a0 = round_clamp(cx[0], 32767); a.b0 = round_clamp(cy[0], 32767);
        /* A field of noise (flat or periodic texture, where every offset
         * matches about as well) fits slopes that are not there, and a
         * wrong model costs every block. So the model has to predict
         * clearly more blocks to within a quarter pixel than the global
         * translation does. */
        int16_t* mx = (int16_t*)malloc(sizeof(int16_t) * (size_t)nb * 2);
        if (mx) {
            model_fill(&a, w, h, block, mx, mx + nb);
            int hit_model = 0, hit_trans = 0;
            for (int b = 0; b < nb; b++) {
                hit_model += abs(vx[b] - mx[b]) + abs(vy[b] - mx[nb + b]) <= 1;
                hit_trans += abs(vx[b] - gx) + abs(vy[b] - gy) <= 1;
            }
            if (hit_model > hit_trans + hit_trans / 8 + nb / 64) { *m = a; affine = 1; }
            free(mx);
        }
    }
    free(use); free(dev);
    return affine;
}

/* ------------------------------------------------------- B frames */

/*
 * BI-PREDICTION
 * -------------
 * A B frame sits between two decoded frames and each of its blocks is
 * predicted from the one before (forward), the one after (backward), or
 * the rounded mean of both. The mean is what makes B frames cheap: two
 * independent guesses at the same content average their noise and their
 * interpolation error down, and what one reference cannot see (the
 * background a moving object uncovers) the other usually can.
 *
 * The field codes, per block, the mode, then the vector of each list the
 * block uses against that list's own median prediction. A list the block
 * does not use takes its prediction as the vector, for free, so the
 * predictions of the blocks after it stay continuous. The mode is two
 * adaptive bits, "not the mean" and then "backward", each conditioned on
 * how many of the left and top neighbours had the same answer.
 */
#define MODE_BI  0
#define MODE_FWD 1
#define MODE_BWD 2

static int block_sad_bi(const NvdrImage* cur, const Subpel* r0, const Subpel* r1,
                        int x0, int y0, int bw, int bh, int f0x, int f0y, int f1x, int f1y,
                        int limit) {
    int acc = 0;
    for (int y = y0; y < y0 + bh; y++) {
        const unsigned char* a = cur->pixels + ((size_t)y * cur->width + x0) * 3;
        for (int x = x0; x < x0 + bw; x++, a += 3)
            for (int c = 0; c < 3; c++) {
                int p = (qsample(r0, x, y, f0x, f0y, c) + qsample(r1, x, y, f1x, f1y, c) + 1) >> 1;
                int d = a[c] - p;
                acc += d < 0 ? -d : d;
            }
        if (acc >= limit) return acc;
    }
    return acc;
}

static void block_predict_bi(const Subpel* r0, const Subpel* r1, NvdrImage* dst, int block,
                             const int16_t* v0x, const int16_t* v0y,
                             const int16_t* v1x, const int16_t* v1y, const uint8_t* mode) {
    int nbx = (dst->width + block - 1) / block;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < dst->height; y++) {
        size_t row = (size_t)(y / block) * nbx;
        unsigned char* out = dst->pixels + (size_t)y * dst->width * 3;
        for (int x = 0; x < dst->width; x++, out += 3) {
            size_t b = row + (size_t)(x / block);
            for (int c = 0; c < 3; c++) {
                int v;
                if (mode[b] == MODE_FWD) v = qsample(r0, x, y, v0x[b], v0y[b], c);
                else if (mode[b] == MODE_BWD) v = qsample(r1, x, y, v1x[b], v1y[b], c);
                else v = (qsample(r0, x, y, v0x[b], v0y[b], c) +
                          qsample(r1, x, y, v1x[b], v1y[b], c) + 1) >> 1;
                out[c] = (unsigned char)v;
            }
        }
    }
}

static int mode_ctx(const uint8_t* mode, int nbx, int b, int which) {
    int x = b % nbx, y = b / nbx, n = 0;
    if (x > 0) n += which ? mode[b - 1] == MODE_BWD : mode[b - 1] != MODE_BI;
    if (y > 0) n += which ? mode[b - nbx] == MODE_BWD : mode[b - nbx] != MODE_BI;
    return n;
}

typedef struct {
    MvModels l0, l1;
    uint16_t not_bi[3], bwd[3];
} BiModels;

static void bi_models_init(BiModels* m) {
    mv_models_init(&m->l0); mv_models_init(&m->l1);
    for (int i = 0; i < 3; i++) m->not_bi[i] = m->bwd[i] = NVDR_PROB_INIT;
}

/* The vectors of unused lists are overwritten with their predictions, as
 * the decoder will see them. */
static uint8_t* pack_field_bi(const uint8_t* mode, int16_t* v0x, int16_t* v0y,
                              int16_t* v1x, int16_t* v1y, int nbx, int nby,
                              const int16_t* m0x, const int16_t* m0y,
                              const int16_t* m1x, const int16_t* m1y, size_t* out_len) {
    int nb = nbx * nby;
    uint8_t* same = (uint8_t*)malloc((size_t)nb * 2);
    if (!same) return NULL;
    BiModels m;
    bi_models_init(&m);
    NvdrEncoder enc;
    if (nvdr_enc_init(&enc, (size_t)nb + 64) != 0) { free(same); return NULL; }
    for (int b = 0; b < nb; b++) {
        nvdr_enc_bit(&enc, &m.not_bi[mode_ctx(mode, nbx, b, 0)], mode[b] != MODE_BI);
        if (mode[b] != MODE_BI) nvdr_enc_bit(&enc, &m.bwd[mode_ctx(mode, nbx, b, 1)], mode[b] == MODE_BWD);
        if (mode[b] != MODE_BWD) enc_vector(&enc, &m.l0, same, v0x, v0y, m0x, m0y, nbx, b, MV_ESC_SEQ);
        else inherit_vector(same, v0x, v0y, m0x, m0y, nbx, b);
        if (mode[b] != MODE_FWD) enc_vector(&enc, &m.l1, same + nb, v1x, v1y, m1x, m1y, nbx, b, MV_ESC_SEQ);
        else inherit_vector(same + nb, v1x, v1y, m1x, m1y, nbx, b);
    }
    free(same);
    if (nvdr_enc_finish(&enc) != 0) { nvdr_enc_free(&enc); return NULL; }
    *out_len = enc.count;
    return enc.bytes;
}

static int unpack_field_bi(const uint8_t* packed, size_t len, int nbx, int nby,
                           const int16_t* m0x, const int16_t* m0y,
                           const int16_t* m1x, const int16_t* m1y, uint8_t* mode,
                           int16_t* v0x, int16_t* v0y, int16_t* v1x, int16_t* v1y) {
    int nb = nbx * nby;
    uint8_t* same = (uint8_t*)malloc((size_t)nb * 2);
    if (!same) return -1;
    BiModels m;
    bi_models_init(&m);
    NvdrDecoder dec;
    nvdr_dec_init(&dec, packed, len);
    int rc = 0;
    for (int b = 0; b < nb && rc == 0; b++) {
        mode[b] = MODE_BI;
        if (nvdr_dec_bit(&dec, &m.not_bi[mode_ctx(mode, nbx, b, 0)]))
            mode[b] = nvdr_dec_bit(&dec, &m.bwd[mode_ctx(mode, nbx, b, 1)]) ? MODE_BWD : MODE_FWD;
        if (mode[b] != MODE_BWD)
            rc = dec_vector(&dec, &m.l0, same, v0x, v0y, m0x, m0y, nbx, b, MV_ESC_SEQ, NVDRV_MV_MAX);
        else inherit_vector(same, v0x, v0y, m0x, m0y, nbx, b);
        if (rc) break;
        if (mode[b] != MODE_FWD)
            rc = dec_vector(&dec, &m.l1, same + nb, v1x, v1y, m1x, m1y, nbx, b, MV_ESC_SEQ, NVDRV_MV_MAX);
        else inherit_vector(same + nb, v1x, v1y, m1x, m1y, nbx, b);
    }
    free(same);
    return rc;
}

/*
 * Each list is searched on its own, as a P frame's field is; then this
 * pass walks the blocks in coding order, when both predictions are known,
 * and picks mode and vectors by error plus the bits they cost: forward or
 * backward alone, or the mean, each with the searched vectors or the
 * predicted ones (which cost a bit).
 */
static void bi_decide(const NvdrImage* cur, const Subpel* r0, const Subpel* r1, int block,
                      int lambda, const int16_t* m0x, const int16_t* m0y,
                      const int16_t* m1x, const int16_t* m1y,
                      const int16_t* s0x, const int16_t* s0y, const int16_t* s1x, const int16_t* s1y,
                      uint8_t* mode, int16_t* v0x, int16_t* v0y, int16_t* v1x, int16_t* v1y) {
    int nbx = (cur->width + block - 1) / block;
    int nby = (cur->height + block - 1) / block;
    for (int b = 0; b < nbx * nby; b++) {
        int x0 = (b % nbx) * block, y0 = (b / nbx) * block;
        int bw = cur->width - x0 < block ? cur->width - x0 : block;
        int bh = cur->height - y0 < block ? cur->height - y0 : block;
        int p0x, p0y, p1x, p1y;
        mv_predict(v0x, v0y, m0x, m0y, nbx, b, &p0x, &p0y);
        mv_predict(v1x, v1y, m1x, m1y, nbx, b, &p1x, &p1y);
        /* mode, then list 0's vector, then list 1's */
        int cand[8][5] = {
            { MODE_BI,  s0x[b], s0y[b], s1x[b], s1y[b] },
            { MODE_BI,  p0x, p0y, p1x, p1y },
            { MODE_BI,  s0x[b], s0y[b], p1x, p1y },
            { MODE_BI,  p0x, p0y, s1x[b], s1y[b] },
            { MODE_FWD, s0x[b], s0y[b], p1x, p1y },
            { MODE_FWD, p0x, p0y, p1x, p1y },
            { MODE_BWD, p0x, p0y, s1x[b], s1y[b] },
            { MODE_BWD, p0x, p0y, p1x, p1y },
        };
        int best = INT_MAX, bi = 1;
        for (int c = 0; c < 8; c++) {
            int md = cand[c][0];
            /* A mode like its neighbours' is nearly free once coded; one
             * unlike them costs more than its bits, since it also makes
             * the next blocks' modes dearer. Weighing a mismatch as 4 bits
             * instead of 1 was 1.5 to 2% smaller on the test clips. */
            int bits = 1 + 4 * ((b % nbx && mode[b - 1] != md) + (b >= nbx && mode[b - nbx] != md));
            if (md != MODE_BWD) bits += mv_bits(cand[c][1] - p0x, cand[c][2] - p0y);
            if (md != MODE_FWD) bits += mv_bits(cand[c][3] - p1x, cand[c][4] - p1y);
            int rate = lambda * bits;
            if (rate >= best) continue;
            int cost = rate + (md == MODE_FWD
                ? block_sad_q(cur, r0, x0, y0, bw, bh, cand[c][1], cand[c][2], best - rate)
                : md == MODE_BWD
                ? block_sad_q(cur, r1, x0, y0, bw, bh, cand[c][3], cand[c][4], best - rate)
                : block_sad_bi(cur, r0, r1, x0, y0, bw, bh, cand[c][1], cand[c][2],
                               cand[c][3], cand[c][4], best - rate));
            if (cost < best) { best = cost; bi = c; }
        }
        mode[b] = (uint8_t)cand[bi][0];
        v0x[b] = (int16_t)cand[bi][1]; v0y[b] = (int16_t)cand[bi][2];
        v1x[b] = (int16_t)cand[bi][3]; v1y[b] = (int16_t)cand[bi][4];
    }
}

/* ------------------------------------------------------------ encoder */

/*
 * FRAME ORDER
 * -----------
 * Frames arrive in display order and are written in coding order. Every
 * frame carries its display number, and nothing names a reference: a P
 * frame predicts from the decoded frame nearest before it in display
 * order, a B frame from the nearest before and the nearest after. The
 * encoder codes each group of pictures so that this rule picks the frames
 * it means: the anchor at the group's end first (from the anchor before
 * it), then the frame halfway between the two anchors, then recursively
 * the frame halfway through each half. Decoders keep the decoded frames
 * from the one last shown onward, and show a frame as soon as every frame
 * before it has been shown.
 */
typedef struct {
    int       display;
    int       kind;
    int       partial;
    NvdrImage img;
} Slot;

#define QCAP (NVDRV_MAX_B + 2 + NVDRV_MAX_LOOKAHEAD)

struct NvdrvEncoder {
    FILE*       f;
    NvdrvConfig cfg;
    int         width, height;
    uint32_t    count;       /* frames received */
    uint32_t    coded;       /* frames written */
    int         chroma420;   /* the last intra frame halved its colour */
    /* What the decoder holds: the encoder predicts from these and never
     * from the source, because they are all the decoder will have;
     * predicting from the source is how a codec drifts. */
    Slot        dpb[NVDRV_MAX_DPB];
    int         ndpb;
    /* Source frames after the last anchor, displays anchor + 1 on: the
     * next group, and the frames an intra frame looks ahead to. With them
     * the global motion from each source frame to the one before it,
     * summed from the last anchor: cum[i] for display anchor + i. */
    NvdrImage   pend[QCAP];
    int         npend;
    int         cumx[QCAP + 1], cumy[QCAP + 1];
    float       intra_scale; /* the next intra frame's step, as a fraction of frame.q */
    int8_t*     tile_q;      /* the next intra frame's tile offsets */
    int         use_tile_q;
    NvdrImage   last_src;
    int         anchor;      /* display number of the last anchor, -1 before any */
    NvdrImage   pred, error;
    int16_t     *s0x, *s0y, *s1x, *s1y, *v0x, *v0y, *v1x, *v1y, *m0x, *m0y, *m1x, *m1y;
    uint8_t*    mode;
    NvdrvReportFn report;
    void*       user;
};

static int alloc_image(NvdrImage* img, int w, int h) {
    img->width = w; img->height = h;
    img->pixels = (unsigned char*)calloc((size_t)w * h * 3, 1);
    return img->pixels ? 0 : -1;
}

/* Decode a frame's container at full quality, which is what later frames
 * predict from. */
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

/* The references the rule above picks, among `n` decoded frames. */
static void find_refs(const Slot* dpb, int n, int display, int* before, int* after) {
    *before = *after = -1;
    for (int i = 0; i < n; i++) {
        if (dpb[i].display < display && (*before < 0 || dpb[i].display > dpb[*before].display))
            *before = i;
        if (dpb[i].display > display && (*after < 0 || dpb[i].display < dpb[*after].display))
            *after = i;
    }
}

int nvdrv_encode_open(NvdrvEncoder** out, const char* path,
                      int width, int height, const NvdrvConfig* cfg) {
    *out = NULL;
    NvdrvEncoder* e = (NvdrvEncoder*)calloc(1, sizeof(*e));
    if (!e) return -1;
    e->cfg = cfg ? *cfg : nvdrv_default_config();
    e->width = width; e->height = height;
    e->anchor = -1;

    if (e->cfg.block < 0) e->cfg.block = (long)width * height >= 200000 ? 16 : 8;
    if (e->cfg.block > 128) e->cfg.block = 0;
    if (e->cfg.bframes < 0) e->cfg.bframes = 0;
    if (e->cfg.bframes > NVDRV_MAX_B) e->cfg.bframes = NVDRV_MAX_B;
    if (e->cfg.lookahead < 0) e->cfg.lookahead = 0;
    if (e->cfg.lookahead > NVDRV_MAX_LOOKAHEAD) e->cfg.lookahead = NVDRV_MAX_LOOKAHEAD;
    e->intra_scale = 1.0f;
    /* B frames need per-block vectors; a GOP shorter than a group bounds
     * the group anyway. */
    if (e->cfg.block == 0) e->cfg.bframes = 0;
    if (alloc_image(&e->pred, width, height) != 0 ||
        alloc_image(&e->error, width, height) != 0 ||
        alloc_image(&e->last_src, width, height) != 0) {
        nvdrv_encode_close(e); return -1;
    }
    if (e->cfg.block > 0) {
        size_t nb = (size_t)field_blocks(width, height, e->cfg.block);
        int16_t** v[12] = { &e->s0x, &e->s0y, &e->s1x, &e->s1y, &e->v0x, &e->v0y, &e->v1x, &e->v1y,
                            &e->m0x, &e->m0y, &e->m1x, &e->m1y };
        for (int i = 0; i < 12; i++)
            if (!(*v[i] = (int16_t*)malloc(nb * sizeof(int16_t)))) { nvdrv_encode_close(e); return -1; }
        if (!(e->mode = (uint8_t*)malloc(nb))) { nvdrv_encode_close(e); return -1; }
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
    h[16] = (uint8_t)e->cfg.bframes;      /* informational */
    if (fwrite(h, 1, sizeof(h), e->f) != sizeof(h)) { nvdrv_encode_close(e); return -1; }

    *out = e;
    return 0;
}

void nvdrv_encode_set_report(NvdrvEncoder* e, NvdrvReportFn fn, void* user) {
    e->report = fn; e->user = user;
}

/*
 * [kind u8][block u8][dx i16][dy i16][body length u32][dx1 i16][dy1 i16]
 * [display u32][flags u8][0 u8], then the body: for a frame with a field,
 * [u32 field length][field][container], otherwise the container. dx, dy
 * is the global vector toward the frame before, dx1, dy1 (B frames) the
 * one toward the frame after, both in whole pixels. A sequence's field
 * starts with the affine model of each list whose flag is set (bit 0 the
 * list before, bit 1 the list after), 12 bytes each; a list without one
 * takes its global vector as a plain translation.
 */
static int write_frame(NvdrvEncoder* e, int kind, int display, int block, int flags,
                       int dx, int dy, int dx1, int dy1,
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
    put_u16v(h + 10, (uint16_t)(int16_t)dx1);
    put_u16v(h + 12, (uint16_t)(int16_t)dy1);
    put_u32v(h + 14, (uint32_t)display);
    h[18] = (uint8_t)flags;
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

/* One list's field: global vector around `c`, blocks around it and zero,
 * refined to quarter pixels, the model fitted to them, then smoothed
 * against its bits. */
static int motion_list(NvdrvEncoder* e, const NvdrImage* cur, const NvdrImage* ref, int dist, int lambda,
                       int cx, int cy, int* gdx, int* gdy, int16_t* vx, int16_t* vy,
                       Model* model, int* affine, int16_t* mx, int16_t* my, Subpel* sp) {
    int block = e->cfg.block, lim = NVDRV_MV_MAX / 4 - 8;
    cx = clampi(cx, -lim, lim); cy = clampi(cy, -lim, lim);
    /* Next to its reference the whole search range, as a P frame always
     * searched; further away a smaller one around the summed motion. */
    find_shift_at(cur, ref, cx, cy, (cx || cy) ? 4 : e->cfg.search, gdx, gdy);
    *gdx = clampi(*gdx, -lim, lim); *gdy = clampi(*gdy, -lim, lim);
    block_search(cur, ref, block, dist, *gdx, *gdy, vx, vy);
    if (subpel_build(sp, ref) != 0) return -1;
    block_refine(cur, sp, block, NVDRV_MV_MAX, vx, vy);
    *affine = model_fit(vx, vy, cur->width, cur->height, block, *gdx * 4, *gdy * 4, model);
    model_fill(model, cur->width, cur->height, block, mx, my);
    if (lambda > 0) field_rd(cur, sp, block, NVDRV_MV_MAX, mx, my, lambda, vx, vy);
    return 0;
}

/* The affine models ahead of the field's coded bytes (NULL for a list
 * whose model is its global vector). */
static uint8_t* with_models(const Model* m0, const Model* m1, uint8_t* coded, size_t* len) {
    if (!coded) return NULL;
    size_t head = MODEL_BYTES * ((m0 != NULL) + (m1 != NULL));
    uint8_t* out = (uint8_t*)malloc(head + *len);
    if (out) {
        uint8_t* p = out;
        if (m0) { model_put(m0, p); p += MODEL_BYTES; }
        if (m1) { model_put(m1, p); p += MODEL_BYTES; }
        memcpy(p, coded, *len);
        *len += head;
    }
    free(coded);
    return out;
}

/* Summed source motion from display d to display r, both within the
 * current group. */
static void group_motion(const NvdrvEncoder* e, int d, int r, int* x, int* y) {
    int i = d - e->anchor, j = r - e->anchor;
    *x = e->cumx[i] - e->cumx[j];
    *y = e->cumy[i] - e->cumy[j];
}

static int q_for(const NvdrvEncoder* e, int kind, int level) {
    int q = e->cfg.frame.q;
    if (kind == NVDRV_INTRA) {
        int qi = (int)(q * e->intra_scale + 0.5f);
        return qi < 1 ? 1 : qi;
    }
    /* P frames coarser than the intra frames unless told otherwise: by
     * 1.2 when every frame is a P frame (on the clean clip that was 0.1 dB
     * better at equal rate across q 16-40), by 1.4 between B frames (2%
     * smaller at equal luma on the three test clips than 1.2, 4% than
     * 1.0). */
    int pq = e->cfg.pred_q > 0 ? e->cfg.pred_q
           : e->cfg.bframes > 0 ? (q * 7 + 2) / 5 : (q * 6 + 2) / 5;
    if (kind == NVDRV_PRED) return pq;
    return (int)(pq * (1.0f + e->cfg.b_q_step * (float)level) + 0.5f);
}

/* Code the source frame `src`, shown at `display`, and keep what the
 * decoder will hold. */
static int code_frame(NvdrvEncoder* e, const NvdrImage* src, int display, int force_intra, int level) {
    size_t npx = (size_t)e->width * e->height * 3;
    int before, after;
    find_refs(e->dpb, e->ndpb, display, &before, &after);
    int kind = force_intra || before < 0 ? NVDRV_INTRA
             : (after >= 0 && e->cfg.block > 0 ? NVDRV_BI : NVDRV_PRED);
    if (e->ndpb >= NVDRV_MAX_DPB) return -1;

    int block = 0, dx = 0, dy = 0, dx1 = 0, dy1 = 0;
    uint8_t* field = NULL;
    size_t field_len = 0;
    int nbx = e->cfg.block ? (e->width + e->cfg.block - 1) / e->cfg.block : 0;
    int nby = e->cfg.block ? (e->height + e->cfg.block - 1) / e->cfg.block : 0;
    const NvdrImage* ref = NULL;
    Model model0, model1;
    int aff0 = 0, aff1 = 0;
    /* A bit of field is worth more error the coarser the residual that
     * would have to correct it: mv_lambda is set at the default q of 24. */
    int lambda = e->cfg.mv_lambda;
    if (kind != NVDRV_INTRA) lambda = (e->cfg.mv_lambda * q_for(e, kind, level) + 12) / 24;

    if (kind == NVDRV_PRED) {
        const NvdrImage* r0 = &e->dpb[before].img;
        int cx, cy;
        group_motion(e, display, e->dpb[before].display, &cx, &cy);
        if (e->cfg.block > 0) {
            block = e->cfg.block;
            Subpel sp;
            if (motion_list(e, src, r0, display - e->dpb[before].display, lambda, cx, cy, &dx, &dy,
                            e->v0x, e->v0y, &model0, &aff0, e->m0x, e->m0y, &sp) != 0) return -1;
            block_predict(&sp, &e->pred, block, e->v0x, e->v0y);
            subpel_free(&sp);
            ref = &e->pred;
        } else {
            find_shift_at(src, r0, cx, cy, (cx || cy) ? 4 : e->cfg.search, &dx, &dy);
            if (dx || dy) { shift_into(r0, &e->pred, dx, dy); ref = &e->pred; }
            else ref = r0;
        }
        /* A cut, or anything else the bias to 128 would clip, goes intra. */
        double acc = 0.0;
        for (size_t i = 0; i < npx; i++) {
            int d = (int)src->pixels[i] - (int)ref->pixels[i];
            acc += d < 0 ? -d : d;
        }
        if (acc / (double)npx > e->cfg.intra_threshold) {
            kind = NVDRV_INTRA; block = 0; dx = dy = 0;
        }
    } else if (kind == NVDRV_BI) {
        block = e->cfg.block;
        const Slot* a = &e->dpb[before];
        const Slot* c = &e->dpb[after];
        int c0x, c0y, c1x, c1y;
        group_motion(e, display, a->display, &c0x, &c0y);
        group_motion(e, display, c->display, &c1x, &c1y);
        Subpel sp0, sp1;
        memset(&sp1, 0, sizeof(sp1));
        if (motion_list(e, src, &a->img, display - a->display, lambda, c0x, c0y, &dx, &dy,
                        e->s0x, e->s0y, &model0, &aff0, e->m0x, e->m0y, &sp0) != 0) return -1;
        if (motion_list(e, src, &c->img, c->display - display, lambda, c1x, c1y, &dx1, &dy1,
                        e->s1x, e->s1y, &model1, &aff1, e->m1x, e->m1y, &sp1) != 0) {
            subpel_free(&sp0); subpel_free(&sp1); return -1;
        }
        bi_decide(src, &sp0, &sp1, block, lambda, e->m0x, e->m0y, e->m1x, e->m1y,
                  e->s0x, e->s0y, e->s1x, e->s1y, e->mode, e->v0x, e->v0y, e->v1x, e->v1y);
        block_predict_bi(&sp0, &sp1, &e->pred, block, e->v0x, e->v0y, e->v1x, e->v1y, e->mode);
        subpel_free(&sp0); subpel_free(&sp1);
        ref = &e->pred;
    }

    const NvdrImage* to_code = src;
    if (kind != NVDRV_INTRA) {
        for (size_t i = 0; i < npx; i++)
            e->error.pixels[i] =
                (unsigned char)clamp255v((int)src->pixels[i] - (int)ref->pixels[i] + 128);
        to_code = &e->error;
        if (kind == NVDRV_PRED && block)
            field = with_models(aff0 ? &model0 : NULL, NULL,
                                pack_field(e->v0x, e->v0y, e->m0x, e->m0y, nbx, nby, MV_ESC_SEQ, &field_len),
                                &field_len);
        else if (kind == NVDRV_BI)
            field = with_models(aff0 ? &model0 : NULL, aff1 ? &model1 : NULL,
                                pack_field_bi(e->mode, e->v0x, e->v0y, e->v1x, e->v1y, nbx, nby,
                                              e->m0x, e->m0y, e->m1x, e->m1y, &field_len),
                                &field_len);
        if (block && !field) return -1;
    }

    NvdrConfig fcfg = e->cfg.frame;
    fcfg.q = q_for(e, kind, level);
    fcfg.tile_q = kind == NVDRV_INTRA && e->use_tile_q ? e->tile_q : NULL;
    if (kind != NVDRV_INTRA) {
        fcfg.residual = 1;
        /* When the intra frame halved its colour, as it does for anything
         * photographic, the residuals do too: the reference's colour is
         * already smooth, and a residual coded whole spends its bytes on
         * colour detail, and colour noise, the viewer does not see. Their
         * leaf seams are filtered as well, since over a continuous
         * prediction they land in the picture as they are. The filter is
         * the still decoder's, driven by the container's own flag, so
         * decoders need nothing new. Together, at equal luma: 6 to 8%
         * smaller on the clean test clips, 23% on the noisy one.
         *
         * A sequence whose intra frame keeps its colour whole (a drawing,
         * a screen) keeps both off: halving there piles colour error up
         * along the chain of references, and on the regression gate's
         * sawtooth texture the filter cost 9% for nothing. */
        int photo = e->chroma420 && e->cfg.frame.chroma420 == NVDR_CHROMA_AUTO;
        fcfg.chroma420 = photo ? NVDR_CHROMA_420 : e->cfg.frame.chroma420;
        fcfg.deblock = photo && e->cfg.frame.deblock;
    }

    uint8_t* blob = NULL;
    size_t len = 0;
    NvdrHeader fh;
    if (nvdr_encode_mem(&blob, &len, to_code, &fcfg, &fh) != 0) { free(field); return -1; }
    if (kind == NVDRV_INTRA) e->chroma420 = (fh.flags & NVDR_FLAG_CHROMA420) != 0;

    int flags = kind == NVDRV_INTRA ? 0 : (aff0 ? FRAME_MODEL0 : 0) | (kind == NVDRV_BI && aff1 ? FRAME_MODEL1 : 0);
    if (write_frame(e, kind, display, block, flags, dx, dy, dx1, dy1, field, field_len, blob, len) != 0) {
        free(blob); free(field); return -1;
    }
    free(field);

    /* Carry the decoder's state forward by decoding what was just written,
     * so the two sides hold the same bytes from here on. */
    Slot* s = &e->dpb[e->ndpb];
    if (alloc_image(&s->img, e->width, e->height) != 0) { free(blob); return -1; }
    int rc = reconstruct(blob, len, kind == NVDRV_INTRA ? &s->img : &e->error);
    free(blob);
    if (rc != 0) { nvdr_image_free(&s->img); return -1; }
    if (kind != NVDRV_INTRA)
        for (size_t i = 0; i < npx; i++)
            s->img.pixels[i] = (unsigned char)clamp255v(
                (int)e->error.pixels[i] - 128 + (int)ref->pixels[i]);
    s->display = display;
    s->kind = kind;
    e->ndpb++;
    e->coded++;

    if (e->report) {
        NvdrvFrameReport r;
        r.display = display; r.kind = kind; r.level = kind == NVDRV_BI ? level : 0;
        r.q = fcfg.q; r.dx = dx; r.dy = dy;
        r.bytes = len + NVDRV_FRAME_HEADER + (block ? 4 + field_len : 0);
        r.field_bytes = field_len;
        e->report(e->user, &r);
    }
    return 0;
}

/*
 * LOOKING AHEAD
 * -------------
 * An intra frame is coded as if nothing came after it, but the frames
 * that follow copy it: a background that stays on screen for the whole
 * group is paid for once here and inherited forty times, and every level
 * of quality it gets here is quality they do not have to pay for again.
 * So before an intra frame is coded, the encoder looks at up to
 * `lookahead` source frames after it and estimates how much of it they
 * reuse, the way x264's macroblock tree and AV1's temporal dependency
 * model do.
 *
 * Every 16x16 luma block of every frame gets an intra cost (the Hadamard
 * sum of its detail, what coding it from nothing would take) and an inter
 * cost (the Hadamard sum of its error against the best match in the frame
 * before, whole pixels, searched around the global motion). Walking back
 * from the last frame, each block passes on to the blocks it came from
 *
 *     (intra + inherited) * (1 - inter / intra)
 *
 * that is, the share of its information that was prediction and not new,
 * spread over the blocks its match overlaps. What reaches the intra frame
 * says, per block, how many times over it is reused; the step is refined
 * by `tpl_strength` sixths of a doubling for each doubling of
 * (intra + inherited) / intra, weighted over the frame by intra cost.
 */
#define TPL_B 16

static void to_luma(const NvdrImage* img, uint8_t* y) {
    size_t n = (size_t)img->width * img->height;
    for (size_t i = 0; i < n; i++) {
        const unsigned char* p = img->pixels + i * 3;
        y[i] = (uint8_t)((77 * p[0] + 150 * p[1] + 29 * p[2] + 128) >> 8);
    }
}

/* Sum of absolute 8x8 Hadamard coefficients of d, the DC left out when
 * `ac` is set. */
static int hadamard8(int d[64], int ac) {
    for (int r = 0; r < 8; r++) {
        int* v = d + 8 * r;
        for (int len = 1; len < 8; len <<= 1)
            for (int i = 0; i < 8; i += 2 * len)
                for (int j = i; j < i + len; j++) {
                    int a = v[j], b = v[j + len];
                    v[j] = a + b; v[j + len] = a - b;
                }
    }
    for (int c = 0; c < 8; c++)
        for (int len = 1; len < 8; len <<= 1)
            for (int i = 0; i < 8; i += 2 * len)
                for (int j = i; j < i + len; j++) {
                    int a = d[8 * j + c], b = d[8 * (j + len) + c];
                    d[8 * j + c] = a + b; d[8 * (j + len) + c] = a - b;
                }
    int acc = 0;
    for (int i = ac ? 1 : 0; i < 64; i++) acc += d[i] < 0 ? -d[i] : d[i];
    return acc;
}

/* Hadamard cost of block (x0, y0) of `cur` against `ref` moved by (dx,
 * dy), or of its own detail when ref is NULL. */
static int tpl_cost(const uint8_t* cur, const uint8_t* ref, int w, int h, int x0, int y0, int dx, int dy) {
    int acc = 0, d[64];
    for (int sy = 0; sy < TPL_B; sy += 8)
        for (int sx = 0; sx < TPL_B; sx += 8) {
            for (int j = 0; j < 8; j++)
                for (int i = 0; i < 8; i++) {
                    int x = x0 + sx + i, y = y0 + sy + j;
                    int c = cur[(size_t)y * w + x];
                    int r = ref ? ref[(size_t)clampi(y + dy, 0, h - 1) * w + clampi(x + dx, 0, w - 1)] : 0;
                    d[8 * j + i] = c - r;
                }
            acc += hadamard8(d, ref == NULL);
        }
    return acc;
}

static int tpl_sad(const uint8_t* cur, const uint8_t* ref, int w, int h, int x0, int y0, int dx, int dy, int limit) {
    int acc = 0;
    for (int y = y0; y < y0 + TPL_B; y++) {
        const uint8_t* a = cur + (size_t)y * w;
        const uint8_t* b = ref + (size_t)clampi(y + dy, 0, h - 1) * w;
        for (int x = x0; x < x0 + TPL_B; x++) {
            int v = a[x] - b[clampi(x + dx, 0, w - 1)];
            acc += v < 0 ? -v : v;
        }
        if (acc >= limit) return acc;
    }
    return acc;
}

/* The intra frame pend[first]'s step, as a fraction of frame.q, from the
 * `count` frames held from it on; with `tile_q`, per tile instead (the
 * frame's scale is then 1). */
static float tpl_scale(const NvdrvEncoder* e, int first, int count, int8_t* tile_q) {
    int w = e->width, h = e->height;
    int nbx = w / TPL_B, nby = h / TPL_B, nb = nbx * nby;
    int frames = count - 1 < e->cfg.lookahead ? count - 1 : e->cfg.lookahead;
    if (frames < 1 || nb < 1) return 1.0f;
    size_t npx = (size_t)w * h;
    uint8_t* luma = (uint8_t*)malloc(npx * (size_t)(frames + 1));
    float* intra = (float*)malloc(sizeof(float) * (size_t)nb * (frames + 1));
    float* prop = (float*)calloc((size_t)nb * (frames + 1), sizeof(float));
    int* mvx = (int*)malloc(sizeof(int) * (size_t)nb);
    int* mvy = (int*)malloc(sizeof(int) * (size_t)nb);
    float* inter = (float*)malloc(sizeof(float) * (size_t)nb);
    float scale = 1.0f;
    if (!luma || !intra || !prop || !mvx || !mvy || !inter) goto done;
    for (int k = 0; k <= frames; k++) {
        to_luma(&e->pend[first + k], luma + npx * k);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
        for (int b = 0; b < nb; b++)
            intra[(size_t)k * nb + b] = (float)tpl_cost(luma + npx * k, NULL, w, h,
                                                        (b % nbx) * TPL_B, (b / nbx) * TPL_B, 0, 0);
    }
    /* Back from the last frame: each passes on what it predicted. */
    for (int k = frames; k >= 1; k--) {
        const uint8_t* cur = luma + npx * k;
        const uint8_t* ref = luma + npx * (k - 1);
        int j = first + k + 1;                   /* cum index of this frame */
        int gx = e->cumx[j] - e->cumx[j - 1], gy = e->cumy[j] - e->cumy[j - 1];
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
        for (int b = 0; b < nb; b++) {
            int x0 = (b % nbx) * TPL_B, y0 = (b / nbx) * TPL_B;
            int best = INT_MAX, bx = 0, by = 0;
            for (int pass = 0; pass < 2; pass++) {
                int cx = pass ? 0 : gx, cy = pass ? 0 : gy;
                if (pass && !gx && !gy) break;
                for (int dy = cy - NVDRV_LOCAL_RANGE; dy <= cy + NVDRV_LOCAL_RANGE; dy++)
                    for (int dx = cx - NVDRV_LOCAL_RANGE; dx <= cx + NVDRV_LOCAL_RANGE; dx++) {
                        int c = tpl_sad(cur, ref, w, h, x0, y0, dx, dy, best);
                        if (c < best) { best = c; bx = dx; by = dy; }
                    }
            }
            mvx[b] = bx; mvy[b] = by;
            inter[b] = (float)tpl_cost(cur, ref, w, h, x0, y0, bx, by);
        }
        const float* in = intra + (size_t)k * nb;
        const float* pr = prop + (size_t)k * nb;
        float* back = prop + (size_t)(k - 1) * nb;
        for (int b = 0; b < nb; b++) {
            if (in[b] <= 0.0f) continue;
            float ratio = inter[b] < in[b] ? inter[b] / in[b] : 1.0f;
            float amount = (in[b] + pr[b]) * (1.0f - ratio);
            if (amount <= 0.0f) continue;
            /* Spread over the up to four blocks the match overlaps. */
            int px = (b % nbx) * TPL_B + mvx[b], py = (b / nbx) * TPL_B + mvy[b];
            int bx0 = px >= 0 ? px / TPL_B : -1 - (-px - 1) / TPL_B;
            int by0 = py >= 0 ? py / TPL_B : -1 - (-py - 1) / TPL_B;
            int fx = px - bx0 * TPL_B, fy = py - by0 * TPL_B;
            for (int t = 0; t < 4; t++) {
                int tx = bx0 + (t & 1), ty = by0 + (t >> 1);
                if (tx < 0 || ty < 0 || tx >= nbx || ty >= nby) continue;
                int ow = (t & 1) ? fx : TPL_B - fx, oh = (t >> 1) ? fy : TPL_B - fy;
                back[ty * nbx + tx] += amount * (float)(ow * oh) / (float)(TPL_B * TPL_B);
            }
        }
    }
    if (tile_q) {
        /* Per tile, the same average over the blocks it holds. */
        int tile = e->cfg.frame.max_block, per = tile / TPL_B;
        int tx_n = (w + tile - 1) / tile, ty_n = (h + tile - 1) / tile;
        for (int t = 0; t < tx_n * ty_n; t++) {
            double ws = 0.0, ls = 0.0;
            for (int j = 0; j < per; j++)
                for (int i = 0; i < per; i++) {
                    int bx = (t % tx_n) * per + i, by = (t / tx_n) * per + j;
                    if (bx >= nbx || by >= nby) continue;
                    double in = intra[by * nbx + bx] + 1.0;
                    ws += in;
                    ls += in * log2((in + prop[by * nbx + bx]) / in);
                }
            int d = ws > 0.0 ? (int)lrint(-e->cfg.tpl_strength * ls / ws) : 0;
            tile_q[t] = (int8_t)(d < -12 ? -12 : d > 0 ? 0 : d);
        }
        goto done;
    }
    /* The intra frame's reuse, per block as a doubling count, averaged
     * over the frame by what each block costs. */
    double wsum = 0.0, lsum = 0.0;
    for (int b = 0; b < nb; b++) {
        double in = intra[b] + 1.0;
        wsum += in;
        lsum += in * log2((in + prop[b]) / in);
    }
    if (wsum > 0.0) {
        double offset = e->cfg.tpl_strength * lsum / wsum;       /* sixths of a doubling */
        scale = (float)pow(2.0, -offset / 6.0);
        if (scale < 0.35f) scale = 0.35f;
    }
    if (getenv("TPLDBG")) fprintf(stderr, "tpl: %d frames, scale %.3f\n", frames, scale);
done:
    free(luma); free(intra); free(prop); free(mvx); free(mvy); free(inter);
    return scale;
}

/* The frames strictly between displays lo and hi, halfway first. */
static int code_between(NvdrvEncoder* e, int lo, int hi, int level) {
    /* pend[i] is display anchor + 1 + i, and the anchor is still lo. */
    if (hi - lo < 2) return 0;
    int mid = (lo + hi) / 2;
    if (code_frame(e, &e->pend[mid - e->anchor - 1], mid, 0, level) != 0) return -1;
    if (code_between(e, lo, mid, level + 1) != 0) return -1;
    return code_between(e, mid, hi, level + 1);
}

static void drop_before(NvdrvEncoder* e, int display) {
    int k = 0;
    for (int i = 0; i < e->ndpb; i++) {
        if (e->dpb[i].display < display) nvdr_image_free(&e->dpb[i].img);
        else e->dpb[k++] = e->dpb[i];
    }
    e->ndpb = k;
}

static int is_intra_slot(const NvdrvEncoder* e, int display) {
    return display == 0 || (e->cfg.gop > 0 && display % e->cfg.gop == 0);
}

/* The display number of the anchor that ends the next group. */
static int group_end(const NvdrvEncoder* e) {
    int a = e->anchor;
    if (a < 0) return 0;
    int g = a + e->cfg.bframes + 1;
    if (e->cfg.gop > 0) {
        int next_intra = (a / e->cfg.gop + 1) * e->cfg.gop;
        if (next_intra < g) g = next_intra;
    }
    return g;
}

/* Whether the next group can be coded: its frames have arrived, and so
 * have the frames an intra anchor looks ahead to, unless the stream is
 * closing. */
static int group_ready(const NvdrvEncoder* e, int closing) {
    if (e->npend == 0) return 0;
    if (closing) return 1;
    int g = group_end(e), last = e->anchor + e->npend;
    int ahead = is_intra_slot(e, g) && e->cfg.tpl_strength > 0 ? e->cfg.lookahead : 0;
    return last >= g + ahead;
}

static float tpl_scale(const NvdrvEncoder* e, int first, int count, int8_t* tile_q);

/* Code the next group: its anchor, then the B frames before it. */
static int flush_group(NvdrvEncoder* e, int closing) {
    int a = e->anchor, g = group_end(e);
    if (closing && g > a + e->npend) g = a + e->npend;
    int intra = is_intra_slot(e, g);
    int idx = g - a - 1;
    e->intra_scale = 1.0f;
    int tiles_on = e->cfg.frame.max_block >= 2 * TPL_B && !getenv("TPLFRAME");
    if (intra && e->cfg.tpl_strength > 0) {
        if (tiles_on && !e->tile_q) {
            int tile = e->cfg.frame.max_block;
            e->tile_q = (int8_t*)calloc((size_t)((e->width + tile - 1) / tile) * ((e->height + tile - 1) / tile), 1);
        }
        e->intra_scale = tpl_scale(e, idx, e->npend - idx, tiles_on ? e->tile_q : NULL);
    }
    int use_tiles = intra && tiles_on && e->tile_q && e->cfg.tpl_strength > 0;
    e->use_tile_q = use_tiles;
    if (code_frame(e, &e->pend[idx], g, intra, 0) != 0) return -1;
    e->use_tile_q = 0;
    e->intra_scale = 1.0f;
    if (a >= 0 && code_between(e, a, g, 1) != 0) return -1;
    /* Frames before the new anchor are never a reference again. */
    drop_before(e, g);
    /* The frames after the anchor move to the front, their buffers with
     * them, and the summed motion is measured from the new anchor. */
    int shift = g - a;
    NvdrImage keep[QCAP];
    for (int i = 0; i < e->npend; i++) keep[i] = e->pend[i];
    for (int i = 0; i < e->npend; i++) e->pend[i] = keep[(i + shift) % e->npend];
    for (int i = 0; i + shift <= e->npend; i++) {
        e->cumx[i] = e->cumx[i + shift] - e->cumx[shift];
        e->cumy[i] = e->cumy[i + shift] - e->cumy[shift];
    }
    e->npend -= shift;
    e->anchor = g;
    return 0;
}

int nvdrv_encode_frame(NvdrvEncoder* e, const NvdrImage* frame) {
    if (frame->width != e->width || frame->height != e->height) return -1;
    size_t npx = (size_t)e->width * e->height * 3;
    int display = (int)e->count;
    int i = display - e->anchor;
    if (e->npend >= QCAP || i > QCAP) return -1;

    /* Global motion from the frame before, measured on the sources: it
     * only centres the searches against the decoded references, and the
     * look ahead. */
    int sx = 0, sy = 0;
    if (display > 0 && (e->cfg.bframes > 0 || e->cfg.tpl_strength > 0))
        find_shift(frame, &e->last_src, e->cfg.search, &sx, &sy);
    if (i == 1) e->cumx[0] = e->cumy[0] = 0;
    e->cumx[i] = e->cumx[i - 1] + sx;
    e->cumy[i] = e->cumy[i - 1] + sy;
    NvdrImage* p = &e->pend[e->npend];
    if (!p->pixels && alloc_image(p, e->width, e->height) != 0) return -1;
    memcpy(p->pixels, frame->pixels, npx);
    e->npend++;
    memcpy(e->last_src.pixels, frame->pixels, npx);
    e->count++;
    while (group_ready(e, 0))
        if (flush_group(e, 0) != 0) return -1;
    return 0;
}

int nvdrv_encode_close(NvdrvEncoder* e) {
    int rc = 0;
    if (!e) return 0;
    if (e->f) {
        while (rc == 0 && e->npend > 0)
            if (flush_group(e, 1) != 0) rc = -1;
        if (fseek(e->f, 10, SEEK_SET) == 0) {
            uint8_t n[4];
            put_u32v(n, e->coded);
            if (fwrite(n, 1, 4, e->f) != 4) rc = -1;
        }
        if (fclose(e->f) != 0) rc = -1;
    }
    for (int i = 0; i < e->ndpb; i++) nvdr_image_free(&e->dpb[i].img);
    for (int i = 0; i < QCAP; i++) free(e->pend[i].pixels);
    free(e->pred.pixels); free(e->error.pixels); free(e->last_src.pixels);
    free(e->s0x); free(e->s0y); free(e->s1x); free(e->s1y);
    free(e->v0x); free(e->v0y); free(e->v1x); free(e->v1y);
    free(e->m0x); free(e->m0y); free(e->m1x); free(e->m1y);
    free(e->mode);
    free(e->tile_q);
    free(e);
    return rc;
}

/* ------------------------------------------------------------ decoder */

struct NvdrvDecoder {
    uint8_t*  data;
    size_t    size, pos;
    int       width, height;
    Slot      dpb[NVDRV_MAX_DPB];
    int       ndpb;
    int       next_out;      /* display number of the next frame to show */
    int       shown;         /* display number of the frame last shown */
    int       ended;
    NvdrImage pred, err;
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
    d->shown = -1;
    /* The working buffers are allocated at this size before a single frame
     * is read, so a damaged header gets refused here rather than trusted. */
    if (d->width <= 0 || d->height <= 0 ||
        (size_t)d->width * d->height > NVDR_MAX_PIXELS ||
        alloc_image(&d->pred, d->width, d->height) != 0 ||
        alloc_image(&d->err, d->width, d->height) != 0) {
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

/* The next frame in coding order into the held frames: 1 when one was
 * decoded, 0 at the end of the stream, -1 on damage. */
static int decode_one(NvdrvDecoder* d) {
    if (d->pos + NVDRV_FRAME_HEADER > d->size) return 0;
    const uint8_t* h = d->data + d->pos;
    int kind = h[0];
    int block = h[1];
    int dx = (int16_t)get_u16v(h + 2);
    int dy = (int16_t)get_u16v(h + 4);
    size_t len = get_u32v(h + 6);
    int dx1 = (int16_t)get_u16v(h + 10);
    int dy1 = (int16_t)get_u16v(h + 12);
    int flags = h[18];
    uint32_t display32 = get_u32v(h + 14);
    if (kind != NVDRV_INTRA && kind != NVDRV_PRED && kind != NVDRV_BI) return -1;
    if (kind == NVDRV_INTRA && block) return -1;
    if (flags & ~(FRAME_MODEL0 | FRAME_MODEL1) || h[19]) return -1;
    if (flags && (!block || (kind != NVDRV_BI && (flags & FRAME_MODEL1)))) return -1;
    if (kind == NVDRV_BI && (block < 4 || block > 128)) return -1;
    if (kind == NVDRV_PRED && block && (block < 4 || block > 128)) return -1;
    if (display32 > INT_MAX / 2) return -1;
    int display = (int)display32;
    /* A frame shown already, or twice, or more held frames than any
     * encoder makes, is damage. */
    if (display < d->next_out || d->ndpb >= NVDRV_MAX_DPB) return -1;
    for (int i = 0; i < d->ndpb; i++) if (d->dpb[i].display == display) return -1;
    int before, after;
    find_refs(d->dpb, d->ndpb, display, &before, &after);
    if (kind != NVDRV_INTRA && before < 0) return -1;
    if (kind == NVDRV_BI && after < 0) return -1;
    d->pos += NVDRV_FRAME_HEADER;

    size_t npx = (size_t)d->width * d->height * 3;
    const NvdrImage* ref = kind == NVDRV_INTRA ? NULL : &d->dpb[before].img;

    if (block) {
        /* The field has to arrive whole: without it there is no reference
         * to add the residual to, so a cut inside it ends the stream. */
        if (d->pos + 4 > d->size || len < 4) return 0;
        size_t field_len = get_u32v(d->data + d->pos);
        if (field_len > len - 4 || d->pos + 4 + field_len > d->size) return 0;
        int nbx = (d->width + block - 1) / block, nby = (d->height + block - 1) / block;
        size_t nb = (size_t)nbx * nby;
        /* Vectors of both lists, then the models' vectors of both. */
        int16_t* v = (int16_t*)malloc(nb * 8 * sizeof(int16_t));
        int16_t* mv = v + 4 * nb;
        uint8_t* mode = (uint8_t*)malloc(nb);
        Subpel sp0, sp1;
        memset(&sp0, 0, sizeof(sp0)); memset(&sp1, 0, sizeof(sp1));
        int rc = -1;
        const uint8_t* fp = d->data + d->pos + 4;
        size_t head = MODEL_BYTES * (!!(flags & FRAME_MODEL0) + !!(flags & FRAME_MODEL1));
        if (v && mode && field_len >= head) {
            Model m0 = { dx * 4, 0, 0, dy * 4, 0, 0 }, m1 = { dx1 * 4, 0, 0, dy1 * 4, 0, 0 };
            const uint8_t* mp = fp;
            if (flags & FRAME_MODEL0) { model_get(&m0, mp); mp += MODEL_BYTES; }
            if (flags & FRAME_MODEL1) model_get(&m1, mp);
            model_fill(&m0, d->width, d->height, block, mv, mv + nb);
            if (kind == NVDRV_BI) model_fill(&m1, d->width, d->height, block, mv + 2 * nb, mv + 3 * nb);
            if (kind == NVDRV_PRED) {
                if (unpack_field(fp + head, field_len - head, nbx, nby, mv, mv + nb, MV_ESC_SEQ,
                                 NVDRV_MV_MAX, v, v + nb) == 0 &&
                    subpel_build(&sp0, ref) == 0) {
                    block_predict(&sp0, &d->pred, block, v, v + nb);
                    rc = 0;
                }
            } else if (unpack_field_bi(fp + head, field_len - head, nbx, nby, mv, mv + nb, mv + 2 * nb,
                                       mv + 3 * nb, mode, v, v + nb, v + 2 * nb, v + 3 * nb) == 0 &&
                       subpel_build(&sp0, ref) == 0 &&
                       subpel_build(&sp1, &d->dpb[after].img) == 0) {
                block_predict_bi(&sp0, &sp1, &d->pred, block, v, v + nb, v + 2 * nb, v + 3 * nb, mode);
                rc = 0;
            }
        }
        subpel_free(&sp0); subpel_free(&sp1);
        free(v); free(mode);
        if (rc != 0) return -1;
        ref = &d->pred;
        d->pos += 4 + field_len;
        len -= 4 + field_len;
    } else if (kind == NVDRV_PRED && (dx || dy)) {
        shift_into(ref, &d->pred, dx, dy);
        ref = &d->pred;
    }

    /* A cut file ends mid-frame. The still decoder reads as far as the
     * bytes reach, so the last frame is shown at whatever quality arrived
     * instead of being dropped. */
    size_t have = d->size - d->pos;
    int partial = 0;
    if (len > have) { len = have; partial = 1; }
    if (len == 0) return 0;

    Slot* s = &d->dpb[d->ndpb];
    if (alloc_image(&s->img, d->width, d->height) != 0) return -1;
    /* A frame cut before its colour layer has no picture yet: the stream
     * ends there, it is not damaged. */
    if (reconstruct(d->data + d->pos, len, kind == NVDRV_INTRA ? &s->img : &d->err) != 0) {
        nvdr_image_free(&s->img);
        return partial ? 0 : -1;
    }
    if (kind != NVDRV_INTRA)
        for (size_t i = 0; i < npx; i++)
            s->img.pixels[i] = (unsigned char)clamp255v(
                (int)d->err.pixels[i] - 128 + (int)ref->pixels[i]);
    s->display = display;
    s->kind = kind;
    s->partial = partial;
    d->ndpb++;
    d->pos += len;
    return 1;
}

int nvdrv_decode_next(NvdrvDecoder* d, NvdrImage* out, int* kind_out, int* partial_out) {
    if (partial_out) *partial_out = 0;
    for (;;) {
        for (int i = 0; i < d->ndpb; i++) {
            if (d->dpb[i].display != d->next_out) continue;
            memcpy(out->pixels, d->dpb[i].img.pixels, (size_t)d->width * d->height * 3);
            if (kind_out) *kind_out = d->dpb[i].kind;
            if (partial_out) *partial_out = d->dpb[i].partial;
            d->shown = d->next_out++;
            /* Every frame still to come is shown after this one, so it
             * predicts from this one or something later. */
            int k = 0;
            for (int j = 0; j < d->ndpb; j++) {
                if (d->dpb[j].display < d->shown) nvdr_image_free(&d->dpb[j].img);
                else d->dpb[k++] = d->dpb[j];
            }
            d->ndpb = k;
            return 1;
        }
        if (!d->ended) {
            int rc = decode_one(d);
            if (rc < 0) return -1;
            if (rc == 0) d->ended = 1;
            continue;
        }
        /* The stream ended with frames missing (a cut file): skip to the
         * next one that did arrive. */
        int next = -1;
        for (int i = 0; i < d->ndpb; i++)
            if (d->dpb[i].display > d->next_out && (next < 0 || d->dpb[i].display < next))
                next = d->dpb[i].display;
        if (next < 0) return 0;
        d->next_out = next;
    }
}

int nvdrv_decode_display(const NvdrvDecoder* d) { return d->shown; }

void nvdrv_decode_close(NvdrvDecoder* d) {
    if (!d) return;
    free(d->data);
    for (int i = 0; i < d->ndpb; i++) nvdr_image_free(&d->dpb[i].img);
    free(d->pred.pixels);
    free(d->err.pixels);
    free(d);
}

/* ------------------------------------------------- one image from another */

/*
 * A predicted image outside any sequence: an album's photo coded against
 * the one before it. The same tools as a predicted frame — block motion in
 * quarter pixels, the residual as a v11 container with its colours
 * predicted as 128 and blocks that need nothing left alone — packed as
 *
 *   [block u8][global dx i16][global dy i16][field length u32][field][container]
 */
static int auto_block(int w, int h, int block) {
    if (block > 0 && block <= 128) return block;
    return (long)w * h >= 200000 ? 16 : 8;
}

int nvdrv_motion_find(const NvdrImage* ref, const NvdrImage* cur, const NvdrvConfig* cfg_in,
                      NvdrvMotion* m) {
    memset(m, 0, sizeof(*m));
    if (ref->width != cur->width || ref->height != cur->height) return -1;
    m->cfg = cfg_in ? *cfg_in : nvdrv_default_config();
    int w = cur->width, h = cur->height, block = auto_block(w, h, m->cfg.block);
    size_t npx = (size_t)w * h * 3;
    int nbx = (w + block - 1) / block, nby = (h + block - 1) / block;
    int rc = -1;
    int16_t* vx = (int16_t*)malloc((size_t)nbx * nby * sizeof(int16_t));
    int16_t* vy = (int16_t*)malloc((size_t)nbx * nby * sizeof(int16_t));
    int16_t* tx = (int16_t*)malloc((size_t)nbx * nby * sizeof(int16_t));
    int16_t* ty = (int16_t*)malloc((size_t)nbx * nby * sizeof(int16_t));
    NvdrImage pred = { NULL, w, h };
    Subpel sp;
    memset(&sp, 0, sizeof(sp));
    m->block = block;
    m->err.width = w; m->err.height = h;
    if (!vx || !vy || !tx || !ty || alloc_image(&pred, w, h) || alloc_image(&m->err, w, h)) goto done;

    find_shift(cur, ref, m->cfg.search, &m->dx, &m->dy);
    block_search(cur, ref, block, 1, m->dx, m->dy, vx, vy);
    if (subpel_build(&sp, ref) != 0) goto done;
    block_refine(cur, &sp, block, 127, vx, vy);
    /* An album's field is predicted from one translation, the global
     * vector, and carries no model. */
    for (int b = 0; b < nbx * nby; b++) { tx[b] = (int16_t)(m->dx * 4); ty[b] = (int16_t)(m->dy * 4); }
    if (m->cfg.mv_lambda > 0) field_rd(cur, &sp, block, 127, tx, ty, m->cfg.mv_lambda, vx, vy);
    block_predict(&sp, &pred, block, vx, vy);
    for (size_t i = 0; i < npx; i++)
        m->err.pixels[i] = (unsigned char)clamp255v((int)cur->pixels[i] - (int)pred.pixels[i] + 128);
    m->field = pack_field(vx, vy, tx, ty, nbx, nby, MV_ESC_ALBUM, &m->field_len);
    if (m->field) rc = 0;

done:
    subpel_free(&sp);
    free(vx); free(vy); free(tx); free(ty); free(pred.pixels);
    if (rc != 0) nvdrv_motion_free(m);
    return rc;
}

void nvdrv_motion_free(NvdrvMotion* m) {
    free(m->field); free(m->err.pixels);
    m->field = NULL; m->err.pixels = NULL;
}

int nvdrv_motion_encode(const NvdrvMotion* m, const NvdrImage* ref, int q, size_t limit,
                        uint8_t** out, size_t* out_len, NvdrImage* recon) {
    *out = NULL; *out_len = 0; recon->pixels = NULL;
    NvdrConfig fcfg = m->cfg.frame;
    fcfg.residual = 1;
    fcfg.deblock = 0;
    fcfg.band = 0;
    if (q > 0) fcfg.q = q;
    uint8_t* blob = NULL; size_t len = 0;
    if (nvdr_encode_mem(&blob, &len, &m->err, &fcfg, NULL) != 0) return -1;

    size_t total = 9 + m->field_len + len;
    if (total >= limit) { free(blob); return 1; }
    uint8_t* p = (uint8_t*)malloc(total);
    if (!p) { free(blob); return -1; }
    p[0] = (uint8_t)m->block;
    put_u16v(p + 1, (uint16_t)(int16_t)m->dx);
    put_u16v(p + 3, (uint16_t)(int16_t)m->dy);
    put_u32v(p + 5, (uint32_t)m->field_len);
    memcpy(p + 9, m->field, m->field_len);
    memcpy(p + 9 + m->field_len, blob, len);
    free(blob);

    /* What the decoder will show, from the bytes just written. */
    if (nvdrv_predict_decode(ref, p, total, recon, NULL) != 0) { free(p); return -1; }
    *out = p; *out_len = total;
    return 0;
}

int nvdrv_predict_encode(const NvdrImage* ref, const NvdrImage* cur, const NvdrvConfig* cfg_in,
                         uint8_t** out, size_t* out_len, NvdrImage* recon) {
    *out = NULL; *out_len = 0; recon->pixels = NULL;
    NvdrvMotion m;
    if (nvdrv_motion_find(ref, cur, cfg_in, &m) != 0) return -1;
    int rc = nvdrv_motion_encode(&m, ref, m.cfg.pred_q, (size_t)-1, out, out_len, recon);
    nvdrv_motion_free(&m);
    return rc == 0 ? 0 : -1;
}

int nvdrv_predict_decode(const NvdrImage* ref, const uint8_t* data, size_t len,
                         NvdrImage* out, int* partial_out) {
    out->pixels = NULL;
    if (partial_out) *partial_out = 0;
    if (len < 9) return -1;
    int w = ref->width, h = ref->height;
    int block = data[0];
    if (block < 4 || block > 128) return -1;
    int dx = (int16_t)get_u16v(data + 1), dy = (int16_t)get_u16v(data + 3);
    size_t field_len = get_u32v(data + 5);
    /* Without the whole field there is no reference to add a residual to. */
    if (field_len > len - 9) return -1;
    int nbx = (w + block - 1) / block, nby = (h + block - 1) / block;
    size_t npx = (size_t)w * h * 3;
    int rc = -1;
    int16_t* vx = (int16_t*)malloc((size_t)nbx * nby * sizeof(int16_t));
    int16_t* vy = (int16_t*)malloc((size_t)nbx * nby * sizeof(int16_t));
    int16_t* tx = (int16_t*)malloc((size_t)nbx * nby * sizeof(int16_t));
    int16_t* ty = (int16_t*)malloc((size_t)nbx * nby * sizeof(int16_t));
    NvdrImage pred = { NULL, w, h }, err = { NULL, w, h };
    Subpel sp;
    memset(&sp, 0, sizeof(sp));
    if (!vx || !vy || !tx || !ty || alloc_image(&pred, w, h) || alloc_image(&err, w, h)) goto done;
    for (int b = 0; b < nbx * nby; b++) { tx[b] = (int16_t)(dx * 4); ty[b] = (int16_t)(dy * 4); }
    if (unpack_field(data + 9, field_len, nbx, nby, tx, ty, MV_ESC_ALBUM, 127, vx, vy) != 0) goto done;
    if (subpel_build(&sp, ref) != 0) goto done;
    block_predict(&sp, &pred, block, vx, vy);
    if (reconstruct(data + 9 + field_len, len - 9 - field_len, &err) != 0) goto done;
    for (size_t i = 0; i < npx; i++)
        pred.pixels[i] = (unsigned char)clamp255v((int)err.pixels[i] - 128 + (int)pred.pixels[i]);
    *out = pred;
    pred.pixels = NULL;
    rc = 0;
done:
    subpel_free(&sp);
    free(vx); free(vy); free(tx); free(ty); free(pred.pixels); free(err.pixels);
    return rc;
}
