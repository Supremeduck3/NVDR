/*
 * FILM GRAIN SYNTHESIS
 * --------------------
 * Sensor noise is the most expensive thing in a photograph to code and
 * the least worth keeping exactly: every grain is a random value, so a
 * codec pays full price for detail nobody could tell from another grain
 * of the same size and strength. A night sky at ISO 3200 spent two thirds
 * of its bytes that way.
 *
 * So the encoder takes the noise out (an edge-preserving filter, below),
 * codes the clean picture, and describes the noise it removed in 22
 * bytes: its strength at 16 levels of brightness (sensor noise grows with
 * the light), how far one grain spreads, and how strong it is in colour
 * against luma. The decoder draws grain with those statistics from a
 * fixed pseudo-random sequence and lays it over the picture. The grain is
 * not the original's, pixel for pixel, and PSNR against the noisy source
 * says so; it looks the same, which is what the bytes were being spent on.
 * AV1 does this for film; this is a smaller version of the same idea.
 *
 * Everything the decoder does is integer and mirrored in nvdr.js:
 *
 *   templates  one 64x64 block of grain per component: each value is the
 *              sum of four 11-bit draws of AV1's 16-bit LFSR (close to a
 *              Gaussian), filtered by one of five 3x3 kernels (the
 *              spread), then scaled to a spread of exactly 64.
 *   placement  the picture in 32x32 blocks, each reading the template at
 *              an offset hashed from the block's position, so the grain
 *              does not repeat on a visible grid.
 *   strength   luma grain at a pixel is template * sigma(Y) / 64, with
 *              sigma interpolated between the 16 points by the pixel's
 *              own brightness; colour grain is that times cb/32 or cr/32.
 */
#include "grain.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define T NVDR_GRAIN_TEMPLATE

/* Centre, edge and corner weights of the spread kernels, from none to
 * wide. */
static const int kernels[NVDR_GRAIN_KERNELS][3] = {
    { 1, 0, 0 }, { 8, 1, 0 }, { 4, 1, 0 }, { 4, 2, 1 }, { 2, 2, 1 }
};

void nvdr_grain_pack(const NvdrGrain* g, uint8_t out[NVDR_GRAIN_SIZE]) {
    out[0] = 1;   /* the version of these parameters */
    out[1] = (uint8_t)g->seed; out[2] = (uint8_t)(g->seed >> 8);
    out[3] = g->kernel; out[4] = g->cb; out[5] = g->cr;
    memcpy(out + 6, g->sigma, NVDR_GRAIN_POINTS);
}

int nvdr_grain_unpack(const uint8_t* in, size_t len, NvdrGrain* g) {
    if (len != NVDR_GRAIN_SIZE || in[0] != 1 || in[3] >= NVDR_GRAIN_KERNELS) return -1;
    g->seed = (uint16_t)(in[1] | (in[2] << 8));
    g->kernel = in[3]; g->cb = in[4]; g->cr = in[5];
    memcpy(g->sigma, in + 6, NVDR_GRAIN_POINTS);
    return 0;
}

/* AV1's grain generator: a 16-bit LFSR, taps 0, 1, 3, 12. */
static int lfsr(uint16_t* r, int bits) {
    unsigned bit = ((*r >> 0) ^ (*r >> 1) ^ (*r >> 3) ^ (*r >> 12)) & 1u;
    *r = (uint16_t)((*r >> 1) | (bit << 15));
    return (*r >> (16 - bits)) & ((1 << bits) - 1);
}

static int64_t floor_div(int64_t a, int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

static int64_t isqrt(int64_t v) {
    if (v <= 0) return 0;
    int64_t s = (int64_t)sqrt((double)v);
    while (s * s > v) s--;
    while ((s + 1) * (s + 1) <= v) s++;
    return s;
}

void nvdr_grain_templates(const NvdrGrain* g, NvdrGrainTemplates* t) {
    const int* k = kernels[g->kernel < NVDR_GRAIN_KERNELS ? g->kernel : 0];
    int32_t raw[(T + 2) * (T + 2)];
    int32_t f[T * T];
    for (int p = 0; p < 3; p++) {
        uint16_t r = (uint16_t)(g->seed ^ (0x5A5Au * (unsigned)(p + 1)));
        if (!r) r = 1;
        for (int i = 0; i < (T + 2) * (T + 2); i++)
            raw[i] = lfsr(&r, 11) + lfsr(&r, 11) + lfsr(&r, 11) + lfsr(&r, 11) - 4096;
        int64_t sum = 0;
        for (int y = 0; y < T; y++)
            for (int x = 0; x < T; x++) {
                const int32_t* c = raw + (y + 1) * (T + 2) + x + 1;
                int32_t v = k[0] * c[0] + k[1] * (c[-1] + c[1] + c[-(T + 2)] + c[T + 2]) +
                            k[2] * (c[-(T + 2) - 1] + c[-(T + 2) + 1] + c[T + 2 - 1] + c[T + 2 + 1]);
                f[y * T + x] = v;
                sum += v;
            }
        int64_t mean = floor_div(sum, T * T), var = 0;
        for (int i = 0; i < T * T; i++) var += (int64_t)(f[i] - mean) * (f[i] - mean);
        int64_t sd = isqrt(var / (T * T));
        if (sd < 1) sd = 1;
        for (int i = 0; i < T * T; i++) {
            int64_t v = (int64_t)(f[i] - mean) * 64 / sd;   /* truncates, as nvdr.js does */
            t->t[p][i] = (int16_t)(v < -255 ? -255 : (v > 255 ? 255 : v));
        }
    }
}

static uint32_t block_hash(uint16_t seed, int bx, int by, int p) {
    uint32_t h = (uint32_t)seed * 0x9E3779B1u ^ (uint32_t)bx * 0x85EBCA77u ^
                 (uint32_t)by * 0xC2B2AE3Du ^ (uint32_t)(p + 1) * 0x27D4EB2Fu;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12;
    return h;
}

static int round_div(int n, int d) { return n >= 0 ? (n + d / 2) / d : -((-n + d / 2) / d); }
static uint8_t clamp8(int v) { return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }

void nvdr_grain_apply(const NvdrGrain* g, const NvdrGrainTemplates* t,
                      uint8_t* y, uint8_t* cb, uint8_t* cr, int stride, int w, int h) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < h; j++) {
        uint8_t* ry = y + (size_t)j * stride;
        uint8_t* rb = cb + (size_t)j * stride;
        uint8_t* rr = cr + (size_t)j * stride;
        int off[3] = { 0, 0, 0 }, bx_at = -1;
        for (int i = 0; i < w; i++) {
            if ((i >> 5) != bx_at) {
                bx_at = i >> 5;
                for (int p = 0; p < 3; p++) {
                    uint32_t hh = block_hash(g->seed, bx_at, j >> 5, p);
                    int ox = (int)(hh & 31), oy = (int)((hh >> 5) & 31);
                    off[p] = (oy + (j & 31)) * T + ox - (bx_at << 5);
                }
            }
            int lum = ry[i], idx = lum / 17, frac = lum - idx * 17;
            int nxt = idx + 1 < NVDR_GRAIN_POINTS ? idx + 1 : NVDR_GRAIN_POINTS - 1;
            int s = g->sigma[idx] * (17 - frac) + g->sigma[nxt] * frac;   /* sigma x 8 x 17 */
            ry[i] = clamp8(lum + round_div(t->t[0][off[0] + i] * s, 64 * 8 * 17));
            rb[i] = clamp8(rb[i] + round_div(t->t[1][off[1] + i] * s * g->cb, 64 * 8 * 17 * 32));
            rr[i] = clamp8(rr[i] + round_div(t->t[2][off[2] + i] * s * g->cr, 64 * 8 * 17 * 32));
        }
    }
}

/* ============================================================ encoder */

static void to_ycc(const unsigned char* p, double* o) {
    double r = p[0], g = p[1], b = p[2];
    o[0] = 0.299 * r + 0.587 * g + 0.114 * b;
    o[1] = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
    o[2] = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;
}

/* ---------------------------------------------------- DCT denoising */

/* The orthonormal 8-point DCT, as a matrix. */
static double dct8[8][8];
static void dct8_init(void) {
    static int done = 0;
    if (done) return;
    for (int u = 0; u < 8; u++)
        for (int x = 0; x < 8; x++)
            dct8[u][x] = (u ? sqrt(2.0 / 8) : sqrt(1.0 / 8)) * cos((2 * x + 1) * u * 3.14159265358979 / 16);
    done = 1;
}

static void fdct8x8(const double* in, int stride, double* out) {
    double t[64];
    for (int y = 0; y < 8; y++)
        for (int u = 0; u < 8; u++) {
            double a = 0;
            for (int x = 0; x < 8; x++) a += dct8[u][x] * in[y * stride + x];
            t[y * 8 + u] = a;
        }
    for (int v = 0; v < 8; v++)
        for (int u = 0; u < 8; u++) {
            double a = 0;
            for (int y = 0; y < 8; y++) a += dct8[v][y] * t[y * 8 + u];
            out[v * 8 + u] = a;
        }
}

static void idct8x8(const double* in, double* out) {
    double t[64];
    for (int v = 0; v < 8; v++)
        for (int x = 0; x < 8; x++) {
            double a = 0;
            for (int u = 0; u < 8; u++) a += dct8[u][x] * in[v * 8 + u];
            t[v * 8 + x] = a;
        }
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++) {
            double a = 0;
            for (int v = 0; v < 8; v++) a += dct8[v][y] * t[v * 8 + x];
            out[y * 8 + x] = a;
        }
}

/*
 * The noise model, as libaom's grain estimator builds its own: measured in
 * pixels, on flat blocks. An 8x8 block has a plane fitted to it (smooth
 * shading and gradients are the picture) and what is left is noise, plus
 * texture where there is texture; the flattest quarter of the blocks of
 * each brightness are taken as noise alone. That gives the noise's
 * spread per brightness and how strongly neighbouring pixels share it
 * (demosaicing spreads a grain over more than a pixel).
 *
 * The denoiser works in the DCT, where it needs every coefficient's
 * share of the noise. Measuring that directly let texture in (a textured
 * photo's flattest blocks still carry their texture's low frequencies);
 * it is computed instead, taking the noise as separable first-order
 * autoregressive with that neighbour correlation rho: its covariance is
 * rho^|i-j| along each axis, so the variance of DCT coefficient (u, v) is
 * sigma^2 d_u d_v with d = diag(C R C^T).
 */
typedef struct {
    double shape[64];                  /* per coefficient, per unit of pixel sigma */
    double level[NVDR_GRAIN_POINTS];   /* the noise's pixel sigma, per brightness, from the DCT */
    double pixel[NVDR_GRAIN_POINTS];   /* the same, from flat blocks' pixels */
    double count[NVDR_GRAIN_POINTS];   /* blocks of each brightness */
    double typical;                    /* the measured level, before the fit (see GRAIN_MIN_LEVEL) */
    double rho;                        /* neighbour correlation */
} NoiseModel;

static int cmp_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}

/* The two measures of the noise err opposite ways. In the DCT, against a
 * first-order autoregressive shape, demosaiced sensor noise (which has
 * next to nothing at the highest frequencies) reads weaker than it is:
 * 23 for 45 on grey with that noise, 1.8 times. On flat blocks' pixels
 * the reading is right on grey (41 for 45) and too strong wherever the
 * flattest blocks still hold texture. So the DCT's decides whether there
 * is grain, and the strength used, both to filter and for the grain laid
 * back, is the pixels' reading capped at this multiple of the DCT's. On
 * montanha with sensor noise, filtering at the DCT's reading alone saved
 * 9% for 0.6 dB closer to the clean picture; at the capped one, 18% for
 * 1.2 dB. */
#define GRAIN_PIXEL_CAP 1.8

/* The lower quartile of the spread of plane-fit residuals of 8x8 blocks
 * of pure white Gaussian noise, as a fraction of its sigma (simulated). */
#define FLAT_QUARTILE_BIAS 0.9502

/*
 * Sensor noise is photon noise, whose variance grows with the light,
 * plus read noise, which does not: sigma^2 = a Y + b. Fitting that to the
 * levels measured per brightness, weighed by how many blocks each had,
 * keeps a brightness with few blocks (a night sky's bright ones are its
 * stars) from reading as noise; brightnesses more than 2.5 times off the
 * fit either way are dropped and it is fitted again. Replaces the levels
 * with the fit.
 */
static void fit_sensor(double* level, const double* count) {
    int use[NVDR_GRAIN_POINTS];
    for (int q = 0; q < NVDR_GRAIN_POINTS; q++) use[q] = level[q] >= 0 && count[q] >= 20;
    double a = 0, b = 0;
    for (int pass = 0; pass < 3; pass++) {
        double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int q = 0; q < NVDR_GRAIN_POINTS; q++) {
            if (!use[q]) continue;
            double wt = count[q], x = 17.0 * q, y = level[q] * level[q];
            sw += wt; sx += wt * x; sy += wt * y; sxx += wt * x * x; sxy += wt * x * y;
        }
        if (sw <= 0) return;
        double det = sw * sxx - sx * sx;
        a = det > 0 ? (sw * sxy - sx * sy) / det : 0;
        b = (sy - a * sx) / sw;
        if (a < 0) { a = 0; b = sy / sw; }
        if (b < 0) { b = 0; a = sxx > 0 ? sxy / sxx : 0; }
        for (int q = 0; q < NVDR_GRAIN_POINTS; q++) {
            if (level[q] < 0 || count[q] < 20) continue;
            double f = a * 17.0 * q + b, y = level[q] * level[q];
            use[q] = f > 0 && y < 2.5 * 2.5 * f && y > f / (2.5 * 2.5);
        }
    }
    for (int q = 0; q < NVDR_GRAIN_POINTS; q++) level[q] = sqrt(a * 17.0 * q + b);
}

static int noise_model(const double* p, const double* luma, int w, int h, NoiseModel* m) {
    int bw = w / 8, bh = h / 8;
    long total = (long)bw * bh;
    if (total < 64) return -1;
    int step = 1;
    while (total / ((long)step * step) > 60000) step++;
    int sw = (bw + step - 1) / step, sh = (bh + step - 1) / step, nb = sw * sh;
    double* sd = (double*)malloc(sizeof(double) * (size_t)nb);
    double* cr = (double*)malloc(sizeof(double) * (size_t)nb);
    int* bin = (int*)malloc(sizeof(int) * (size_t)nb);
    double* v = (double*)malloc(sizeof(double) * (size_t)nb);
    int rc = -1;
    if (!sd || !cr || !bin || !v) goto out;
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < nb; b++) {
        int x0 = (b % sw) * step * 8, y0 = (b / sw) * step * 8;
        /* Least squares plane a + b x + c y over x, y in -3.5..3.5. */
        double s = 0, sx = 0, sy = 0, mean = 0;
        for (int j = 0; j < 8; j++)
            for (int i = 0; i < 8; i++) {
                double val = p[(size_t)(y0 + j) * w + x0 + i];
                s += val; sx += val * (i - 3.5); sy += val * (j - 3.5);
                mean += luma[(size_t)(y0 + j) * w + x0 + i];
            }
        double a0 = s / 64, bx = sx / 168, by = sy / 168;   /* sum (i-3.5)^2 over the block = 168 */
        double r[64], e = 0, c1 = 0, c0 = 0;
        for (int j = 0; j < 8; j++)
            for (int i = 0; i < 8; i++) {
                r[j * 8 + i] = p[(size_t)(y0 + j) * w + x0 + i] - (a0 + bx * (i - 3.5) + by * (j - 3.5));
                e += r[j * 8 + i] * r[j * 8 + i];
            }
        for (int j = 0; j < 8; j++)
            for (int i = 0; i < 8; i++) {
                double q = r[j * 8 + i];
                c0 += q * q;
                if (i < 7) c1 += q * r[j * 8 + i + 1];
                if (j < 7) c1 += q * r[(j + 1) * 8 + i];
            }
        sd[b] = sqrt(e / 61);   /* 64 samples, three fitted */
        cr[b] = c0 > 0 ? (c1 / 112) / (c0 / 64) : 0;   /* 112 neighbour pairs */
        int q = (int)(mean / 64 / 17 + 0.5);
        bin[b] = q < 0 ? 0 : (q >= NVDR_GRAIN_POINTS ? NVDR_GRAIN_POINTS - 1 : q);
    }
    /* rho, and a pixel sigma, from the flattest quarter of each
     * brightness. */
    double rho_sum = 0; size_t rho_n = 0;
    for (int q = 0; q < NVDR_GRAIN_POINTS; q++) {
        size_t cnt = 0;
        m->pixel[q] = -1;
        for (int b = 0; b < nb; b++) if (bin[b] == q) v[cnt++] = sd[b];
        m->count[q] = (double)cnt;
        if (cnt < 20) continue;
        qsort(v, cnt, sizeof(double), cmp_double);
        double cut = v[cnt / 4];
        m->pixel[q] = cut / FLAT_QUARTILE_BIAS;
        for (int b = 0; b < nb; b++)
            if (bin[b] == q && sd[b] <= cut && sd[b] > 0) { rho_sum += cr[b]; rho_n++; }
    }
    if (!rho_n) goto out;
    double rho = rho_sum / (double)rho_n;
    rho = rho < 0 ? 0 : (rho > 0.95 ? 0.95 : rho);
    m->rho = rho;
    double dd[8];
    for (int u = 0; u < 8; u++) {
        double acc = 0;
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++) acc += dct8[u][i] * dct8[u][j] * pow(rho, abs(i - j));
        dd[u] = acc;
    }
    for (int k = 0; k < 64; k++) m->shape[k] = sqrt(dd[k / 8] * dd[k % 8]);

    /* The level, in the DCT, where texture is sparse and noise is not: a
     * block's level is the median over its 63 AC coefficients of |c| over
     * the noise that position holds per unit sigma (0.6745 sigma for
     * Gaussian noise), and a brightness's level the tenth percentile of
     * its blocks' (0.8203 of sigma for pure noise, simulated): the
     * flattest, where only noise is left, if there is noise. Measured on
     * pixels instead, a textured photo's flattest blocks still held their
     * texture and read as noise three times too strong. */
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < nb; b++) {
        int x0 = (b % sw) * step * 8, y0 = (b / sw) * step * 8;
        double c[64], a[63];
        fdct8x8(p + (size_t)y0 * w + x0, w, c);
        for (int k = 1; k < 64; k++) a[k - 1] = fabs(c[k]) / m->shape[k];
        qsort(a, 63, sizeof(double), cmp_double);
        sd[b] = a[31] / 0.6745;
    }
    int found = 0;
    for (int q = 0; q < NVDR_GRAIN_POINTS; q++) {
        size_t cnt = 0;
        for (int b = 0; b < nb; b++) if (bin[b] == q) v[cnt++] = sd[b];
        if (cnt < 20) { m->level[q] = -1; continue; }
        qsort(v, cnt, sizeof(double), cmp_double);
        m->level[q] = v[cnt / 10] / 0.8203;
        found = 1;
    }
    if (!found) goto out;
    for (int q = 0; q < NVDR_GRAIN_POINTS; q++) {
        if (m->level[q] >= 0) continue;
        for (int d = 1; d < NVDR_GRAIN_POINTS; d++) {
            if (q - d >= 0 && m->level[q - d] >= 0) { m->level[q] = m->level[q - d]; break; }
            if (q + d < NVDR_GRAIN_POINTS && m->level[q + d] >= 0) { m->level[q] = m->level[q + d]; break; }
        }
    }
    /* The typical level as measured, the median over blocks: the fit
     * below smooths the levels for filtering and grain, and on a clean,
     * textured photo can extrapolate a few noisy-looking brightnesses into
     * noise everywhere, so it does not decide whether there is grain. */
    {
        double half = 0, seen = 0;
        for (int q = 0; q < NVDR_GRAIN_POINTS; q++) half += m->count[q];
        half /= 2;
        m->typical = 0;
        double order[NVDR_GRAIN_POINTS];
        int idx[NVDR_GRAIN_POINTS];
        for (int q = 0; q < NVDR_GRAIN_POINTS; q++) { order[q] = m->level[q]; idx[q] = q; }
        for (int i = 1; i < NVDR_GRAIN_POINTS; i++)
            for (int j = i; j > 0 && order[j - 1] > order[j]; j--) {
                double t = order[j]; order[j] = order[j - 1]; order[j - 1] = t;
                int u = idx[j]; idx[j] = idx[j - 1]; idx[j - 1] = u;
            }
        for (int i = 0; i < NVDR_GRAIN_POINTS; i++) {
            seen += m->count[idx[i]];
            if (seen >= half) { m->typical = order[i]; break; }
        }
    }
    fit_sensor(m->level, m->count);
    fit_sensor(m->pixel, m->count);
    rc = 0;
out:
    free(sd); free(cr); free(bin); free(v);
    return rc;
}

static double model_level(const NoiseModel* m, double y) {
    double v = y / 17;
    int k = (int)v;
    if (k < 0) return m->level[0];
    if (k >= NVDR_GRAIN_POINTS - 1) return m->level[NVDR_GRAIN_POINTS - 1];
    return m->level[k] + (m->level[k + 1] - m->level[k]) * (v - k);
}

/*
 * Denoised in overlapping 8x8 DCT blocks, every block 4 pixels apart:
 * each AC coefficient c is scaled by c^2 / (c^2 + n^2), n the noise it
 * would hold, which leaves a coefficient well above the noise almost
 * whole and one at the noise mostly gone; the blocks are put back and
 * averaged, each weighed by how few coefficients it kept. Rows of blocks
 * 8 pixels apart never overlap, so they run in parallel in two passes.
 *
 * Hard thresholding at 2.7 n (BM3D's first stage, without its block
 * matching) was tried first: it took texture of the noise's strength
 * with the noise, and on montanha with sensor noise came out 3.5 dB
 * further from the clean picture than the noisy one was. This shrinkage
 * came out 1.1 dB closer.
 */
static void dct_denoise(const double* p, const double* luma, int w, int h, const NoiseModel* m, double* out) {
    size_t n = (size_t)w * h;
    double* acc = (double*)calloc(n, sizeof(double));
    double* wt = (double*)calloc(n, sizeof(double));
    if (!acc || !wt) { free(acc); free(wt); memcpy(out, p, n * sizeof(double)); return; }
    int nx = (w - 8) / 4 + 1, ny = (h - 8) / 4 + 1;
    for (int phase = 0; phase < 2; phase++) {
        #pragma omp parallel for schedule(dynamic, 1)
        for (int by = phase; by < ny + 1; by += 2) {
            /* The last row and column of blocks sit against the edge. */
            int y = by < ny ? by * 4 : h - 8;
            if (by == ny && y == (ny - 1) * 4) continue;
            for (int bx = 0; bx <= nx; bx++) {
                int x = bx < nx ? bx * 4 : w - 8;
                if (bx == nx && x == (nx - 1) * 4) continue;
                double c[64], r[64], mean = 0;
                for (int j = 0; j < 8; j++)
                    for (int i = 0; i < 8; i++) mean += luma[(size_t)(y + j) * w + x + i];
                double level = model_level(m, mean / 64);
                fdct8x8(p + (size_t)y * w + x, w, c);
                int kept = 0;
                for (int k = 1; k < 64; k++) {
                    double nz = level * m->shape[k], c2 = c[k] * c[k];
                    double gain = c2 / (c2 + nz * nz + 1e-12);
                    c[k] *= gain;
                    kept += gain > 0.5;
                }
                idct8x8(c, r);
                double weight = 1.0 / (1 + kept);
                for (int j = 0; j < 8; j++)
                    for (int i = 0; i < 8; i++) {
                        size_t at = (size_t)(y + j) * w + x + i;
                        acc[at] += weight * r[j * 8 + i];
                        wt[at] += weight;
                    }
            }
        }
    }
    for (size_t i = 0; i < n; i++) out[i] = wt[i] > 0 ? acc[i] / wt[i] : p[i];
    free(acc); free(wt);
}

/* Lag-one correlation of a template, the statistic a kernel is picked by. */
static double template_corr(const int16_t* t) {
    double a = 0, b = 0;
    for (int y = 0; y < T - 1; y++)
        for (int x = 0; x < T - 1; x++) {
            double v = t[y * T + x];
            a += v * (t[y * T + x + 1] + t[(y + 1) * T + x]) / 2;
            b += v * v;
        }
    return b > 0 ? a / b : 0;
}

/* Below this noise level (the typical AC coefficient's noise in the
 * flattest blocks, averaged over brightness) a picture is taken as clean.
 * Measured: the samples, JPEG photos, 0.5 to 2.5 and montanha at twice
 * its size 1.0; grey with sensor noise of sigma 5.7, 2.9; montanha at
 * twice its size with that noise, 3.9; a night sky 10. Small, densely
 * textured thumbnails are where texture and noise cannot be told apart
 * (one clean sample read 2.8, a clean crop of montanha 5.4), which is why
 * grain is off unless asked for: auto is meant for photographs at a
 * camera's resolution, where clean ones read under 1.5. */
#define GRAIN_MIN_LEVEL 3.0

int nvdr_grain_estimate(const NvdrImage* img, int force, NvdrImage* clean, NvdrGrain* g) {
    int w = img->width, h = img->height;
    size_t n = (size_t)w * h;
    clean->pixels = NULL;
    memset(g, 0, sizeof(*g));
    if (w < 16 || h < 16) return 0;
    dct8_init();
    double* p[3]; double* d[3];
    int rc = -1;
    for (int c = 0; c < 3; c++) { p[c] = (double*)malloc(sizeof(double) * n); d[c] = (double*)malloc(sizeof(double) * n); }
    for (int c = 0; c < 3; c++) if (!p[c] || !d[c]) goto done;
    for (size_t i = 0; i < n; i++) {
        double o[3];
        to_ycc(img->pixels + i * 3, o);
        for (int c = 0; c < 3; c++) p[c][i] = o[c];
    }
    NoiseModel model[3];
    for (int c = 0; c < 3; c++)
        if (noise_model(p[c], p[0], w, h, &model[c]) != 0) { rc = 0; goto done; }
    double typical = model[0].typical;
    dct8_init();
    if (getenv("NVDR_GRAIN_DEBUG")) {
        fprintf(stderr, "grain: luma noise level %.2f by brightness:", typical);
        for (int k = 0; k < NVDR_GRAIN_POINTS; k++) fprintf(stderr, " %.1f", model[0].level[k]);
        fprintf(stderr, "\n");
    }
    if (!force && typical < GRAIN_MIN_LEVEL) { rc = 0; goto done; }
    for (int c = 0; c < 3; c++) {
        NoiseModel strong = model[c];
        for (int k = 0; k < NVDR_GRAIN_POINTS; k++) {
            double cap = model[c].level[k] * GRAIN_PIXEL_CAP;
            strong.level[k] = model[c].pixel[k] < cap ? model[c].pixel[k] : cap;
        }
        dct_denoise(p[c], p[0], w, h, &strong, d[c]);
    }

    /* The grain: what was taken out, per level of the clean picture,
     * where it is flat enough that the residual is noise, not texture. */
    double s2[NVDR_GRAIN_POINTS] = { 0 }, s1[NVDR_GRAIN_POINTS] = { 0 };
    double cy = 0, ccb = 0, ccr = 0, corr = 0, corr_n = 0, sig[NVDR_GRAIN_POINTS];
    size_t used = 0;
    for (int y = 1; y < h - 1; y++)
        for (int x = 1; x < w - 1; x++) {
            size_t i = (size_t)y * w + x;
            double gx = d[0][i + 1] - d[0][i - 1], gy = d[0][i + w] - d[0][i - w];
            double lvl = model_level(&model[0], d[0][i]);
            if (gx * gx + gy * gy > 4 * lvl * lvl) continue;
            double r = p[0][i] - d[0][i];
            int k = (int)(d[0][i] / 17 + 0.5);
            k = k < 0 ? 0 : (k >= NVDR_GRAIN_POINTS ? NVDR_GRAIN_POINTS - 1 : k);
            s2[k] += r * r; s1[k] += 1;
            cy += r * r;
            ccb += (p[1][i] - d[1][i]) * (p[1][i] - d[1][i]);
            ccr += (p[2][i] - d[2][i]) * (p[2][i] - d[2][i]);
            corr += r * ((p[0][i + 1] - d[0][i + 1]) + (p[0][i + w] - d[0][i + w])) / 2;
            corr_n += r * r;
            used++;
        }
    if (used < 1000 || cy <= 0) { rc = 0; goto done; }
    for (int k = 0; k < NVDR_GRAIN_POINTS; k++) sig[k] = s1[k] >= 100 ? sqrt(s2[k] / s1[k]) : -1;
    for (int k = 0; k < NVDR_GRAIN_POINTS; k++) {
        if (sig[k] >= 0) continue;
        for (int dk = 1; dk < NVDR_GRAIN_POINTS; dk++) {
            if (k - dk >= 0 && s1[k - dk] >= 100) { sig[k] = sqrt(s2[k - dk] / s1[k - dk]); break; }
            if (k + dk < NVDR_GRAIN_POINTS && s1[k + dk] >= 100) { sig[k] = sqrt(s2[k + dk] / s1[k + dk]); break; }
        }
        if (sig[k] < 0) sig[k] = sqrt(cy / (double)used);
    }
    /* The grain's strength comes from the noise model, not from what the
     * filter took out: the shrinkage leaves part of the noise in weak
     * coefficients, and grain sized to what it removed came out visibly
     * weaker than the source's, in colour by a third. */
    double grain[3][NVDR_GRAIN_POINTS];
    for (int c = 0; c < 3; c++)
        for (int k = 0; k < NVDR_GRAIN_POINTS; k++) {
            double cap = model[c].level[k] * GRAIN_PIXEL_CAP;
            grain[c][k] = model[c].pixel[k] < cap ? model[c].pixel[k] : cap;
        }
    for (int k = 0; k < NVDR_GRAIN_POINTS; k++) {
        long v = lround(grain[0][k] * 8);
        g->sigma[k] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
    double csum[3] = { 0, 0, 0 };
    for (int k = 0; k < NVDR_GRAIN_POINTS; k++)
        for (int c = 0; c < 3; c++) csum[c] += grain[c][k];
    double lsum = csum[0];
    (void)sig; (void)ccb; (void)ccr;
    long vb = lround(lsum > 0 ? 32 * csum[1] / lsum : 0), vr = lround(lsum > 0 ? 32 * csum[2] / lsum : 0);
    g->cb = (uint8_t)(vb > 255 ? 255 : vb);
    g->cr = (uint8_t)(vr > 255 ? 255 : vr);
    g->seed = (uint16_t)((0x5EED ^ (unsigned)(w * 31 + h)) & 0xffff);
    if (!g->seed) g->seed = 1;
    /* The kernel whose grain spreads as far as the noise did. */
    double want = corr_n > 0 ? corr / corr_n : 0, best = 1e9;
    for (int k = 0; k < NVDR_GRAIN_KERNELS; k++) {
        NvdrGrain probe = *g;
        probe.kernel = (uint8_t)k;
        NvdrGrainTemplates t;
        nvdr_grain_templates(&probe, &t);
        double diff = fabs(template_corr(t.t[0]) - want);
        if (diff < best) { best = diff; g->kernel = (uint8_t)k; }
    }
    if (getenv("NVDR_GRAIN_DEBUG")) {
        fprintf(stderr, "grain: correlation %.2f -> kernel %d, cb %d cr %d, sigma x8:", want, g->kernel, g->cb, g->cr);
        for (int k = 0; k < NVDR_GRAIN_POINTS; k++) fprintf(stderr, " %d", g->sigma[k]);
        fprintf(stderr, "\n");
    }

    clean->pixels = (unsigned char*)malloc(n * 3);
    if (!clean->pixels) goto done;
    clean->width = w; clean->height = h;
    for (size_t i = 0; i < n; i++) {
        double yy = d[0][i], cb = d[1][i] - 128, cr = d[2][i] - 128;
        double rgb[3] = { yy + 1.402 * cr, yy - 0.344136 * cb - 0.714136 * cr, yy + 1.772 * cb };
        for (int c = 0; c < 3; c++) {
            long v = lround(rgb[c]);
            clean->pixels[i * 3 + c] = (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
    rc = 1;

done:
    for (int c = 0; c < 3; c++) { free(p[c]); free(d[c]); }
    return rc;
}
