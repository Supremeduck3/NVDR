/*
 * MOTION COMPENSATION, BLOCK BY BLOCK
 * -----------------------------------
 * A sequence's reference is read at quarter pixels through HEVC's 8-tap
 * luma filters, across first with the sums unrounded, then down
 * (subpel_build_seq() in nvdrv.c). The encoder builds all sixteen phases
 * of a reference once, since its searches read every position many
 * times; a decoder reads each pixel of a frame once or twice, and
 * building the planes was 88% of decoding a 960x540 sequence. Here each
 * block filters only the region it reads, with the same sums and the same
 * rounding, so the pixels are the planes' to the bit; public/nvdrv.js's
 * predictRegionSeq() and obmcApply() are the same code, and this file is
 * compiled into nvdr.wasm too, so the page's player runs it.
 */
#include "mc.h"

#include <string.h>

#define MC_MAXB 129    /* a cell, 128 at most, and one more */

static const int DCTIF[4][8] = {
    { 0, 0, 0, 64, 0, 0, 0, 0 },
    { -1, 4, -10, 58, 17, -5, 1, 0 },
    { -1, 4, -11, 40, 40, -11, 4, -1 },
    { 0, 1, -5, 17, 58, -10, 4, -1 },
};

static _Thread_local int SG[(MC_MAXB + 7) * (MC_MAXB + 7) * 3];
static _Thread_local int HT[(MC_MAXB + 7) * MC_MAXB * 3];
static _Thread_local uint8_t OT0[MC_MAXB * MC_MAXB * 3], OT1[MC_MAXB * MC_MAXB * 3];

static inline int clip8(int v) { return v < 0 ? 0 : v > 255 ? 255 : v; }
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

/* The bw x bh region at (x0, y0) as the vector (fx, fy) predicts it, into
 * dst from `obase` with rows `os` bytes apart. */
static void predict_region(const uint8_t* src, int width, int height, int x0, int y0, int bw, int bh,
                           int fx, int fy, uint8_t* dst, size_t obase, size_t os) {
    size_t stride = (size_t)width * 3;
    int X0 = x0 + (fx >> 2), Y0 = y0 + (fy >> 2), px = fx & 3, py = fy & 3;
    int xs[MC_MAXB + 8];
    size_t ys[MC_MAXB + 8];
    /* Clamped source columns X0-3 .. X0+bw+3 and rows Y0-3 .. Y0+bh+3. */
    for (int i = 0; i < bw + 7; i++) xs[i] = clampi(X0 - 3 + i, 0, width - 1) * 3;
    for (int j = 0; j < bh + 7; j++) ys[j] = (size_t)clampi(Y0 - 3 + j, 0, height - 1) * stride;
    if (!px && !py) {
        for (int j = 0; j < bh; j++) {
            const uint8_t* row = src + ys[j + 3];
            uint8_t* o = dst + obase + j * os;
            for (int i = 0; i < bw; i++, o += 3) {
                const uint8_t* s = row + xs[i + 3];
                o[0] = s[0]; o[1] = s[1]; o[2] = s[2];
            }
        }
        return;
    }
    int gs = (bw + 7) * 3, n = bw * 3;
    for (int j = 0; j < bh + 7; j++) {
        const uint8_t* row = src + ys[j];
        int* g = SG + j * gs;
        for (int i = 0; i < bw + 7; i++, g += 3) {
            const uint8_t* s = row + xs[i];
            g[0] = s[0]; g[1] = s[1]; g[2] = s[2];
        }
    }
    const int *h = DCTIF[px], *v = DCTIF[py];
    if (!py) {
        for (int j = 0; j < bh; j++) {
            const int* g = SG + (j + 3) * gs;
            uint8_t* o = dst + obase + j * os;
            for (int k = 0; k < n; k++) {
                const int* q = g + k;
                int a = h[0] * q[0] + h[1] * q[3] + h[2] * q[6] + h[3] * q[9] +
                        h[4] * q[12] + h[5] * q[15] + h[6] * q[18] + h[7] * q[21];
                o[k] = (uint8_t)clip8((a + 32) >> 6);
            }
        }
        return;
    }
    if (!px) {
        for (int j = 0; j < bh; j++) {
            const int* g = SG + j * gs + 9;
            uint8_t* o = dst + obase + j * os;
            for (int k = 0; k < n; k++) {
                const int* q = g + k;
                int a = v[0] * q[0] + v[1] * q[gs] + v[2] * q[2 * gs] + v[3] * q[3 * gs] +
                        v[4] * q[4 * gs] + v[5] * q[5 * gs] + v[6] * q[6 * gs] + v[7] * q[7 * gs];
                o[k] = (uint8_t)clip8((a + 32) >> 6);
            }
        }
        return;
    }
    /* The sums across, unrounded, for all bh + 7 rows; then down them. */
    for (int j = 0; j < bh + 7; j++) {
        const int* g = SG + j * gs;
        int* l = HT + j * n;
        for (int k = 0; k < n; k++) {
            const int* q = g + k;
            l[k] = h[0] * q[0] + h[1] * q[3] + h[2] * q[6] + h[3] * q[9] +
                   h[4] * q[12] + h[5] * q[15] + h[6] * q[18] + h[7] * q[21];
        }
    }
    for (int j = 0; j < bh; j++) {
        const int* l = HT + j * n;
        uint8_t* o = dst + obase + j * os;
        for (int k = 0; k < n; k++) {
            const int* q = l + k;
            int a = v[0] * q[0] + v[1] * q[n] + v[2] * q[2 * n] + v[3] * q[3 * n] +
                    v[4] * q[4 * n] + v[5] * q[5 * n] + v[6] * q[6 * n] + v[7] * q[7 * n];
            /* Down a column of whole pixels the sums are 64p:
             * (64a + 2048) >> 12 is (a + 32) >> 6. */
            o[k] = (uint8_t)clip8((a + 2048) >> 12);
        }
    }
}

/* Every cell from one reference, but those whose mode is `skip`. */
static void predict_cells(const uint8_t* src, uint8_t* dst, int width, int height, int g, int nfx, int nfy,
                          const int16_t* vx, const int16_t* vy, const uint8_t* mode, int skip) {
    size_t stride = (size_t)width * 3;
    for (int fy = 0; fy < nfy; fy++) {
        int y0 = fy * g, bh = height - y0 < g ? height - y0 : g;
        for (int fx = 0; fx < nfx; fx++) {
            int x0 = fx * g, bw = width - x0 < g ? width - x0 : g, b = fy * nfx + fx;
            if (mode && mode[b] == skip) continue;
            predict_region(src, width, height, x0, y0, bw, bh, vx[b], vy[b], dst, y0 * stride + (size_t)x0 * 3, stride);
        }
    }
}

/* AV1's overlap masks, by half the cell's side (nvdrv.c's OBMC_M). */
static const uint8_t M1[1] = { 64 }, M2[2] = { 45, 64 }, M4[4] = { 39, 50, 59, 64 },
    M8[8] = { 36, 42, 48, 53, 57, 61, 64, 64 },
    M16[16] = { 34, 37, 40, 43, 46, 49, 52, 54, 56, 58, 60, 61, 64, 64, 64, 64 },
    M32[32] = { 33, 35, 36, 38, 40, 41, 43, 44, 45, 47, 48, 50, 51, 52, 53, 55,
                56, 57, 58, 59, 60, 60, 61, 62, 64, 64, 64, 64, 64, 64, 64, 64 };

/* Each cell blended near its edges with the prediction its neighbours'
 * motion makes of the same pixels, toward the cell above, below, left and
 * right in that order, skipping neighbours that move the same way:
 * obmc_apply() in nvdrv.c. */
static void obmc(const uint8_t* r0, const uint8_t* r1, uint8_t* dst, int width, int height, int g, int nfx, int nfy,
                 const int16_t* v0x, const int16_t* v0y, const int16_t* v1x, const int16_t* v1y, const uint8_t* mode) {
    int L = g >> 1 > 0 ? g >> 1 : 1;
    const uint8_t* mask = L >= 32 ? M32 : L >= 16 ? M16 : L >= 8 ? M8 : L >= 4 ? M4 : L >= 2 ? M2 : M1;
    size_t stride = (size_t)width * 3;
#define MD(b) (mode ? mode[b] : MC_FWD)
    for (int fy = 0; fy < nfy; fy++)
        for (int fx = 0; fx < nfx; fx++) {
            int b = fy * nfx + fx, x0 = fx * g, y0 = fy * g;
            int bw = width - x0 < g ? width - x0 : g, bh = height - y0 < g ? height - y0 : g;
            for (int side = 0; side < 4; side++) {
                int nb;
                if (side == 0) { if (fy == 0) continue; nb = b - nfx; }
                else if (side == 1) { if (fy + 1 >= nfy) continue; nb = b + nfx; }
                else if (side == 2) { if (fx == 0) continue; nb = b - 1; }
                else { if (fx + 1 >= nfx) continue; nb = b + 1; }
                int ma = MD(b), m = MD(nb);
                if (ma == m && (m == MC_BWD || (v0x[b] == v0x[nb] && v0y[b] == v0y[nb])) &&
                    (m == MC_FWD || (v1x[b] == v1x[nb] && v1y[b] == v1y[nb]))) continue;
                /* The neighbour's prediction of the whole cell, then the
                 * strip along the shared edge blended into it. */
                size_t ls = (size_t)bw * 3;
                if (m != MC_BWD) predict_region(r0, width, height, x0, y0, bw, bh, v0x[nb], v0y[nb], OT0, 0, ls);
                if (m != MC_FWD) predict_region(r1, width, height, x0, y0, bw, bh, v1x[nb], v1y[nb],
                                                m == MC_BWD ? OT0 : OT1, 0, ls);
                if (m != MC_FWD && m != MC_BWD)
                    for (size_t k = 0; k < (size_t)bh * ls; k++) OT0[k] = (uint8_t)((OT0[k] + OT1[k] + 1) >> 1);
                int vertical = side < 2, len = L < (vertical ? bh : bw) ? L : (vertical ? bh : bw);
                for (int d = 0; d < len; d++) {
                    int w = mask[d];
                    if (w == 64) continue;
                    int line = side == 0 ? d : side == 1 ? bh - 1 - d : side == 2 ? d : bw - 1 - d;
                    int count = vertical ? bw : bh;
                    for (int k = 0; k < count; k++) {
                        int i = vertical ? k : line, j = vertical ? line : k;
                        uint8_t* o = dst + (size_t)(y0 + j) * stride + (size_t)(x0 + i) * 3;
                        const uint8_t* t = OT0 + (size_t)j * ls + (size_t)i * 3;
                        for (int c = 0; c < 3; c++) o[c] = (uint8_t)((w * o[c] + (64 - w) * t[c] + 32) >> 6);
                    }
                }
            }
        }
#undef MD
}

void mc_predict_frame(const uint8_t* r0, const uint8_t* r1, uint8_t* dst, uint8_t* tmp,
                      int width, int height, int g, int nfx, int nfy,
                      const int16_t* v0x, const int16_t* v0y, const int16_t* v1x, const int16_t* v1y,
                      const uint8_t* mode, int obmc_on) {
    if (!mode) {
        predict_cells(r0, dst, width, height, g, nfx, nfy, v0x, v0y, NULL, -1);
    } else {
        /* Forward and mean cells from r0 into dst, backward and mean
         * cells from r1 into tmp; then backward cells take tmp's, mean
         * cells the rounded mean of both. */
        predict_cells(r0, dst, width, height, g, nfx, nfy, v0x, v0y, mode, MC_BWD);
        predict_cells(r1, tmp, width, height, g, nfx, nfy, v1x, v1y, mode, MC_FWD);
        size_t stride = (size_t)width * 3;
        for (int fy = 0; fy < nfy; fy++) {
            int y0 = fy * g, bh = height - y0 < g ? height - y0 : g;
            for (int fx = 0; fx < nfx; fx++) {
                int b = fy * nfx + fx, x0 = fx * g, bw = width - x0 < g ? width - x0 : g;
                if (mode[b] == MC_FWD) continue;
                for (int j = 0; j < bh; j++) {
                    size_t o = (size_t)(y0 + j) * stride + (size_t)x0 * 3;
                    if (mode[b] == MC_BWD) memcpy(dst + o, tmp + o, (size_t)bw * 3);
                    else for (int k = 0; k < bw * 3; k++) dst[o + k] = (uint8_t)((dst[o + k] + tmp[o + k] + 1) >> 1);
                }
            }
        }
    }
    if (obmc_on) obmc(r0, r1, dst, width, height, g, nfx, nfy, v0x, v0y, v1x, v1y, mode);
}
