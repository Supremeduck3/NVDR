/*
 * IN-LOOP RESTORATION
 * -------------------
 * The deblocking filter fixes one artefact, the step at a leaf's edge,
 * with a rule fixed in advance. What quantisation does inside a leaf
 * (ringing around an edge, texture flattened, a ramp turned into steps)
 * is left, and it differs from one picture to the next. So the encoder,
 * which holds the source, fits a filter to the picture it just decoded:
 * the least-squares filter that takes the decoded picture closest to the
 * source, written in a few dozen bytes. The decoder applies it after
 * deblocking, before the picture is shown or predicted from, so in a
 * sequence every later frame predicts from the restored picture: the
 * filter is in the loop.
 *
 * Kind 1 is VVC's adaptive loop filter, simplified:
 *
 *   shape     a 7x7 diamond for luma (12 coefficients, each for a pair of
 *             pixels opposite the centre) and a 5x5 one for colour (6).
 *   clipping  each neighbour enters as its difference from the centre,
 *             clipped to +-bound: a neighbour across a strong edge counts
 *             no more than one across a weak one, so the filter smooths
 *             noise without blurring the edge. One bound per component,
 *             one of four.
 *   classes   luma is split in 4x4 blocks by the direction of its
 *             gradient (none, horizontal, vertical, either diagonal) and
 *             by how busy it is (five levels): 25 classes, which the
 *             encoder merges into as many filters as pay for themselves.
 *   units     each 64x64 unit (in luma pixels) is filtered or not, as the
 *             encoder found better.
 *
 *     out = x + (sum_i c_i * (clip(a_i - x) + clip(b_i - x)) + 64) >> 7
 *
 * all in integers, mirrored in nvdr.js. The encoder's side (fit, below)
 * accumulates the correlations per class for each clip bound, merges
 * classes greedily by how much least-squares gain a merge loses, and
 * weighs the error each filter removes against its bits at the lambda
 * the rest of the picture was coded at.
 *
 * A LEARNED FILTER goes in as kind 2 (see restore.h): its own parse,
 * apply and fit, the same unit map, the same place in the decoder. The
 * encoder tries every kind it has and keeps the cheapest, per component,
 * so a network that only beats the linear filter on some pictures costs
 * nothing on the others.
 */
#include "restore.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* The bound a clipped difference is held to, by index. 255 is none. */
static const int CLIP[4] = { 255, 32, 12, 4 };

/* A tap's offset; its pair is the opposite one. */
static const int8_t TAPS_Y[NVDR_ALF_TAPS_Y][2] = {
    { 0, -3 }, { -1, -2 }, { 0, -2 }, { 1, -2 }, { -2, -1 }, { -1, -1 },
    { 0, -1 }, { 1, -1 }, { 2, -1 }, { -3, 0 }, { -2, 0 }, { -1, 0 }
};
static const int8_t TAPS_C[NVDR_ALF_TAPS_C][2] = {
    { 0, -2 }, { -1, -1 }, { 0, -1 }, { 1, -1 }, { -2, 0 }, { -1, 0 }
};
#define PAD 3

/* Activity per pixel (the mean of |2x - l - r| + |2x - u - d| over a 4x4
 * block) at which each level starts. */
static const int ACT_AT[4] = { 3, 8, 16, 32 };

static int clamp_u8(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }
static int clip_to(int v, int b) { return v < -b ? -b : (v > b ? b : v); }

/* ------------------------------------------------------------------ bits */

typedef struct {
    const uint8_t* p;
    size_t len, pos;          /* in bits */
    int bad;
} BitReader;

static unsigned read_bits(BitReader* r, int n) {
    unsigned v = 0;
    for (int i = 0; i < n; i++) {
        if (r->pos >= r->len) { r->bad = 1; return 0; }
        v = (v << 1) | ((r->p[r->pos >> 3] >> (7 - (r->pos & 7))) & 1);
        r->pos++;
    }
    return v;
}

/* Exp-Golomb of order k; at most 16 leading zeros. */
static unsigned read_eg(BitReader* r, int k) {
    int z = 0;
    while (!read_bits(r, 1)) {
        if (r->bad || ++z > 16) { r->bad = 1; return 0; }
    }
    return (((1u << z) - 1) << k) + read_bits(r, z + k);
}

static int read_signed(BitReader* r, int k) {
    int m = (int)read_eg(r, k);
    if (m && read_bits(r, 1)) m = -m;
    return m;
}

static int bits_for(int n) { int b = 0; while ((1 << b) < n) b++; return b; }

/* ----------------------------------------------------------------- parse */

static int unit_count(int p, int unit) { return (p + unit - 1) / unit; }

int nvdr_restore_parse(const uint8_t* in, size_t len, const NvdrRestorePlane* planes, NvdrRestore* r) {
    memset(r, 0, sizeof(*r));
    BitReader br = { in, len * 8, 0, 0 };
    for (int c = 0; c < 3; c++) {
        NvdrRestoreComp* rc = &r->comp[c];
        rc->kind = (int)read_bits(&br, 3);
        if (br.bad || rc->kind >= NVDR_RESTORE_KINDS) return -1;
        if (rc->kind == NVDR_RESTORE_NONE) continue;
        rc->clip = (int)read_bits(&br, 2);
        int k = (int)read_bits(&br, 2);
        int taps = c == 0 ? NVDR_ALF_TAPS_Y : NVDR_ALF_TAPS_C;
        rc->nfilters = 1;
        if (c == 0) {
            rc->nfilters = (int)read_bits(&br, 5) + 1;
            if (rc->nfilters > NVDR_ALF_CLASSES) return -1;
            int b = bits_for(rc->nfilters);
            for (int i = 0; i < NVDR_ALF_CLASSES; i++) {
                rc->classmap[i] = (uint8_t)read_bits(&br, b);
                if (rc->classmap[i] >= rc->nfilters) return -1;
            }
        }
        for (int f = 0; f < rc->nfilters; f++)
            for (int t = 0; t < taps; t++) {
                int v = read_signed(&br, k);
                if (v < -127 || v > 127) return -1;
                rc->coef[f][t] = (int8_t)v;
            }
        rc->units_x = unit_count(planes[c].pw, planes[c].unit);
        rc->units_y = unit_count(planes[c].ph, planes[c].unit);
        rc->all_units = (int)read_bits(&br, 1);
        if (!rc->all_units) {
            rc->unit_bits = in;
            rc->unit_bit = br.pos;
            size_t n = (size_t)rc->units_x * rc->units_y;
            if (n > br.len - br.pos) return -1;
            br.pos += n;
        }
        if (br.bad) return -1;
    }
    return 0;
}

/* ----------------------------------------------------------------- apply */

/* The plane with PAD pixels of its edge repeated around it. */
static uint8_t* padded(const uint8_t* p, int pw, int ph) {
    int s = pw + 2 * PAD;
    uint8_t* q = (uint8_t*)malloc((size_t)s * (ph + 2 * PAD));
    if (!q) return NULL;
    for (int y = -PAD; y < ph + PAD; y++) {
        int sy = y < 0 ? 0 : (y >= ph ? ph - 1 : y);
        uint8_t* row = q + (size_t)(y + PAD) * s;
        const uint8_t* from = p + (size_t)sy * pw;
        memcpy(row + PAD, from, (size_t)pw);
        for (int x = 0; x < PAD; x++) { row[x] = from[0]; row[PAD + pw + x] = from[pw - 1]; }
    }
    return q;
}

/* Luma's class of the 4x4 block at (x, y): direction * 5 + activity. */
static int block_class(const uint8_t* q, int s, int x, int y) {
    int gv = 0, gh = 0, g1 = 0, g2 = 0;
    for (int j = 0; j < 4; j++) {
        const uint8_t* r = q + (size_t)(y + j + PAD) * s + x + PAD;
        for (int i = 0; i < 4; i++) {
            int c2 = 2 * r[i];
            gh += abs(c2 - r[i - 1] - r[i + 1]);
            gv += abs(c2 - r[i - s] - r[i + s]);
            g1 += abs(c2 - r[i - s - 1] - r[i + s + 1]);
            g2 += abs(c2 - r[i - s + 1] - r[i + s - 1]);
        }
    }
    int avg = (gv + gh) >> 4, act = 0;
    while (act < 4 && avg >= ACT_AT[act]) act++;
    int hvmax = gh > gv ? gh : gv, hvmin = gh > gv ? gv : gh;
    int dmax = g1 > g2 ? g1 : g2, dmin = g1 > g2 ? g2 : g1;
    int dir = 0;
    /* hvmax / hvmin against dmax / dmin, crossed to stay in integers */
    if ((long)hvmax * dmin >= (long)dmax * hvmin) {
        if (hvmax > 2 * hvmin) dir = gh > gv ? 1 : 2;
    } else if (dmax > 2 * dmin) dir = g1 > g2 ? 3 : 4;
    return dir * 5 + act;
}

static int unit_on(const NvdrRestoreComp* rc, int ux, int uy) {
    if (rc->all_units) return 1;
    size_t b = rc->unit_bit + (size_t)uy * rc->units_x + ux;
    return (rc->unit_bits[b >> 3] >> (7 - (b & 7))) & 1;
}

/* One pixel through filter `c`, reading the padded plane at `r`. */
static int filter_px(const uint8_t* r, int s, const int8_t* c, const int8_t (*taps)[2], int n, int b) {
    int x = r[0], sum = 0;
    for (int t = 0; t < n; t++) {
        int o = taps[t][1] * s + taps[t][0];
        sum += c[t] * (clip_to(r[o] - x, b) + clip_to(r[-o] - x, b));
    }
    return clamp_u8(x + ((sum + 64) >> NVDR_ALF_SHIFT));
}

/* Filters every 4x4 block of the units that are on, from `q` (padded)
 * into `out`. */
static void alf_run(const NvdrRestoreComp* rc, int comp, const uint8_t* q, uint8_t* out,
                    const NvdrRestorePlane* g) {
    int s = g->pw + 2 * PAD, b = CLIP[rc->clip];
    int n = comp == 0 ? NVDR_ALF_TAPS_Y : NVDR_ALF_TAPS_C;
    const int8_t (*taps)[2] = comp == 0 ? TAPS_Y : TAPS_C;
    for (int y = 0; y < g->ph; y += 4)
        for (int x = 0; x < g->pw; x += 4) {
            if (!unit_on(rc, x / g->unit, y / g->unit)) continue;
            const int8_t* c = rc->coef[comp == 0 ? rc->classmap[block_class(q, s, x, y)] : 0];
            for (int j = 0; j < 4; j++) {
                const uint8_t* r = q + (size_t)(y + j + PAD) * s + x + PAD;
                uint8_t* o = out + (size_t)(y + j) * g->pw + x;
                for (int i = 0; i < 4; i++) o[i] = (uint8_t)filter_px(r + i, s, c, taps, n, b);
            }
        }
}

int nvdr_restore_apply(const NvdrRestoreComp* rc, int comp, uint8_t* plane, const NvdrRestorePlane* g) {
    if (rc->kind != NVDR_RESTORE_ALF) return 0;
    uint8_t* q = padded(plane, g->pw, g->ph);
    if (!q) return -1;
    alf_run(rc, comp, q, plane, g);
    free(q);
    return 0;
}

/* =============================================================== encoder */

static int bits_put(NvdrBits* b, unsigned v, int n) {
    for (int i = n - 1; i >= 0; i--) {
        if (b->bit == 0) {
            if (b->count == b->cap) {
                size_t cap = b->cap ? b->cap * 2 : 256;
                uint8_t* p = (uint8_t*)realloc(b->bytes, cap);
                if (!p) return -1;
                b->bytes = p; b->cap = cap;
            }
            b->bytes[b->count++] = 0;
        }
        if ((v >> i) & 1) b->bytes[b->count - 1] |= (uint8_t)(0x80 >> b->bit);
        b->bit = (b->bit + 1) & 7;
    }
    return 0;
}

void nvdr_bits_free(NvdrBits* b) { free(b->bytes); memset(b, 0, sizeof(*b)); }

static int eg_len(unsigned v, int k) {
    int z = 0;
    while (v >= (((1u << (z + 1)) - 1) << k)) z++;
    return 2 * z + 1 + k;
}
static int signed_len(int v, int k) { return eg_len((unsigned)abs(v), k) + (v != 0); }

static int put_eg(NvdrBits* b, unsigned v, int k) {
    int z = 0;
    while (v >= (((1u << (z + 1)) - 1) << k)) z++;
    if (bits_put(b, 0, z) || bits_put(b, 1, 1)) return -1;
    return bits_put(b, v - (((1u << z) - 1) << k), z + k);
}
static int put_signed(NvdrBits* b, int v, int k) {
    if (put_eg(b, (unsigned)abs(v), k)) return -1;
    return v ? bits_put(b, v < 0, 1) : 0;
}

#define MAXT NVDR_ALF_TAPS_Y

/* Least-squares statistics of one class: R = sum f f', p = sum f e,
 * with f the clipped pair sums and e the source minus the centre. */
typedef struct {
    double R[MAXT][MAXT];
    double p[MAXT];
    double n;
} Stats;

static void stats_add(Stats* a, const Stats* b, int t) {
    for (int i = 0; i < t; i++) {
        for (int j = 0; j < t; j++) a->R[i][j] += b->R[i][j];
        a->p[i] += b->p[i];
    }
    a->n += b->n;
}

/* Solves R w = p, lightly regularised; returns the squared error it
 * removes, p'w. */
static double solve(const Stats* s, int t, double* w) {
    double A[MAXT][MAXT + 1];
    double tr = 0;
    for (int i = 0; i < t; i++) tr += s->R[i][i];
    double ridge = 1e-6 * tr / t + 1e-9;
    for (int i = 0; i < t; i++) {
        for (int j = 0; j < t; j++) A[i][j] = s->R[i][j];
        A[i][i] += ridge;
        A[i][t] = s->p[i];
    }
    for (int i = 0; i < t; i++) {
        int piv = i;
        for (int r = i + 1; r < t; r++) if (fabs(A[r][i]) > fabs(A[piv][i])) piv = r;
        if (fabs(A[piv][i]) < 1e-12) { for (int k = 0; k < t; k++) w[k] = 0; return 0; }
        if (piv != i) for (int k = 0; k <= t; k++) { double x = A[i][k]; A[i][k] = A[piv][k]; A[piv][k] = x; }
        for (int r = i + 1; r < t; r++) {
            double f = A[r][i] / A[i][i];
            for (int k = i; k <= t; k++) A[r][k] -= f * A[i][k];
        }
    }
    for (int i = t - 1; i >= 0; i--) {
        double v = A[i][t];
        for (int k = i + 1; k < t; k++) v -= A[i][k] * w[k];
        w[i] = v / A[i][i];
    }
    double g = 0;
    for (int i = 0; i < t; i++) g += s->p[i] * w[i];
    return g;
}

/* The candidate filters for one clip bound. */
typedef struct {
    int clip, k, nfilters;
    uint8_t classmap[NVDR_ALF_CLASSES];
    int8_t coef[NVDR_ALF_CLASSES][MAXT];
} Cand;

static int coef_bits(const Cand* c, int taps, int k) {
    int b = 0;
    for (int f = 0; f < c->nfilters; f++)
        for (int t = 0; t < taps; t++) b += signed_len(c->coef[f][t], k);
    return b;
}

/*
 * What a component's error is measured against. Usually the source on
 * the plane's own canvas; for colour at half size, the full-size source
 * after the decoder's upsampling (upsample() below), because a filter that
 * brings the half-size plane closer to the 2x2 means it was coded from
 * made the full-size colour worse on all six samples: the means are not
 * what the eye, or PSNR, sees.
 */
typedef struct {
    const double* src;
    int w, h;                /* the plane's visible size */
    int up;
    int fpw, fw, fh;         /* with `up`, the full-size canvas's stride and visible size */
} Target;

/* The decoder's 4:2:0 upsampling (upsample_plane() in nvdr.c): the two
 * nearest samples each way, 9 3 3 1 in sixteenths, edges held. */
typedef struct { int a, b; } Pair;
static Pair up_pair(int x, int n) {
    int c = x >> 1, o = (x & 1) ? c + 1 : c - 1;
    if (o < 0) o = 0;
    if (o >= n) o = n - 1;
    Pair p = { c, o };
    return p;
}

/* The sums one class gathers: R in integers (exact, and several times
 * faster than in doubles), p in doubles. */
typedef struct {
    int64_t R[MAXT][MAXT];
    double p[MAXT];
    double n;
} Acc;

static void add_sample(Acc* S, const int* f, double e, int n) {
    for (int i = 0; i < n; i++) {
        int64_t fi = f[i];
        for (int j = i; j < n; j++) S->R[i][j] += fi * f[j];
        S->p[i] += fi * e;
    }
    S->n += 1;
}

/* Which pixels the statistics are taken from: every one (1), half in a
 * quincunx (2), every other one each way (4), or every fourth (16). */
#ifndef SUB_CLIP
#define SUB_CLIP 16
#endif
#ifndef SUB_FIT
#define SUB_FIT 2
#endif
#ifndef SUB_FIT_UP
#define SUB_FIT_UP 4
#endif
static int sampled(int x, int y, int sub) {
    switch (sub) {
    case 1: return 1;
    case 2: return ((x + y) & 1) == 0;
    case 4: return ((x | y) & 1) == 0;
    default: return ((x | y) & 3) == 0;
    }
}

/*
 * Stats for every class at one clip bound, over the visible pixels of the
 * units `mask` allows (NULL for all), one in `sub` of them, scaled back up.
 */
static int gather(const uint8_t* q, const uint8_t* cls, const Target* T, const NvdrRestorePlane* g,
                  int comp, int bound, const uint8_t* mask, int ux, int sub, Stats* st) {
    int s = g->pw + 2 * PAD, n = comp == 0 ? NVDR_ALF_TAPS_Y : NVDR_ALF_TAPS_C;
    int nc = comp == 0 ? NVDR_ALF_CLASSES : 1;
    const int8_t (*taps)[2] = comp == 0 ? TAPS_Y : TAPS_C;
    Acc* acc = (Acc*)calloc((size_t)nc, sizeof(Acc));
    if (!acc) return -1;
    int off[MAXT], f[MAXT];
    for (int t = 0; t < n; t++) off[t] = taps[t][1] * s + taps[t][0];
    /* Features are exact integers; with `up` they are 16 times the
     * upsampled ones, and the error is scaled to match. */
    double scale = 1;
    if (!T->up) {
        for (int y = 0; y < T->h; y++)
            for (int x = 0; x < T->w; x++) {
                if (!sampled(x, y, sub) || (mask && !mask[(y / g->unit) * ux + x / g->unit])) continue;
                const uint8_t* r = q + (size_t)(y + PAD) * s + x + PAD;
                int c = r[0];
                for (int t = 0; t < n; t++) f[t] = clip_to(r[off[t]] - c, bound) + clip_to(r[-off[t]] - c, bound);
                add_sample(&acc[comp == 0 ? cls[(y >> 2) * (g->pw >> 2) + (x >> 2)] : 0], f,
                           T->src[(size_t)y * g->pw + x] - c, n);
            }
    } else {
        /* The features at half size, then carried to full size by the
         * upsampling's weights: the filter's effect is linear in them. */
        scale = 16;
        size_t np = (size_t)g->pw * g->ph;
        int* F = (int*)calloc(np * n, sizeof(int));
        if (!F) { free(acc); return -1; }
        for (int y = 0; y < T->h; y++)
            for (int x = 0; x < T->w; x++) {
                const uint8_t* r = q + (size_t)(y + PAD) * s + x + PAD;
                int c = r[0];
                for (int t = 0; t < n; t++)
                    F[(size_t)t * np + (size_t)y * g->pw + x] = clip_to(r[off[t]] - c, bound) + clip_to(r[-off[t]] - c, bound);
            }
        const uint8_t* r = q + PAD * (size_t)s + PAD;
        for (int y = 0; y < T->fh; y++) {
            Pair py = up_pair(y, T->h);
            for (int x = 0; x < T->fw; x++) {
                if (!sampled(x, y, sub)) continue;
                Pair px = up_pair(x, T->w);
                if (mask && !mask[(py.a / g->unit) * ux + px.a / g->unit]) continue;
                size_t i0 = (size_t)py.a * g->pw + px.a, i1 = (size_t)py.a * g->pw + px.b;
                size_t i2 = (size_t)py.b * g->pw + px.a, i3 = (size_t)py.b * g->pw + px.b;
                for (int t = 0; t < n; t++) {
                    const int* Ft = F + (size_t)t * np;
                    f[t] = 9 * Ft[i0] + 3 * Ft[i1] + 3 * Ft[i2] + Ft[i3];
                }
                int d16 = 9 * r[py.a * s + px.a] + 3 * r[py.a * s + px.b] + 3 * r[py.b * s + px.a] + r[py.b * s + px.b];
                add_sample(&acc[0], f, 16 * T->src[(size_t)y * T->fpw + x] - d16, n);
            }
        }
        free(F);
    }
    /* back to the features' own scale (R and p over scale^2), and to every
     * pixel from one in `sub` */
    double k2 = (double)sub / (scale * scale);
    for (int k = 0; k < nc; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) st[k].R[i][j] = (double)acc[k].R[i < j ? i : j][i < j ? j : i] * k2;
            st[k].p[i] = acc[k].p[i] * k2;
        }
        st[k].n = acc[k].n * sub;
    }
    free(acc);
    return 0;
}

/*
 * Merges the classes into filters greedily, each step joining the two
 * groups whose merge loses the least gain, and keeps the number of
 * filters that pays best at `lambda`; the result quantised into `out`.
 */
static double design(const Stats* st, int comp, double lambda, Cand* out) {
    int n = comp == 0 ? NVDR_ALF_TAPS_Y : NVDR_ALF_TAPS_C;
    int nc = comp == 0 ? NVDR_ALF_CLASSES : 1;
    Stats grp[NVDR_ALF_CLASSES];
    double gain[NVDR_ALF_CLASSES], w[MAXT];
    int of[NVDR_ALF_CLASSES];     /* class -> group */
    int alive[NVDR_ALF_CLASSES];
    for (int k = 0; k < nc; k++) { grp[k] = st[k]; gain[k] = solve(&grp[k], n, w); of[k] = k; alive[k] = 1; }
    int best_map[NVDR_ALF_CLASSES];
    double best = 1e300;
    /* what each pair of groups would gain merged, kept until one changes */
    double joint[NVDR_ALF_CLASSES][NVDR_ALF_CLASSES];
    uint8_t known[NVDR_ALF_CLASSES][NVDR_ALF_CLASSES];
    memset(known, 0, sizeof(known));
    int groups = nc;
    for (;;) {
        double total = 0;
        for (int k = 0; k < nc; k++) if (alive[k]) total += gain[k];
        /* a filter's coefficients run about five bits each */
        double bits = groups * n * 5.0 + (nc > 1 ? nc * bits_for(groups) : 0);
        double cost = -total + lambda * bits;
        if (cost < best) { best = cost; memcpy(best_map, of, sizeof(of)); }
        if (groups == 1) break;
        int ba = -1, bb = -1;
        double bl = 1e300;
        for (int a = 0; a < nc; a++) {
            if (!alive[a]) continue;
            for (int b = a + 1; b < nc; b++) {
                if (!alive[b]) continue;
                if (!known[a][b]) {
                    Stats m = grp[a];
                    stats_add(&m, &grp[b], n);
                    joint[a][b] = solve(&m, n, w);
                    known[a][b] = 1;
                }
                double loss = gain[a] + gain[b] - joint[a][b];
                if (loss < bl) { bl = loss; ba = a; bb = b; }
            }
        }
        stats_add(&grp[ba], &grp[bb], n);
        gain[ba] = joint[ba][bb];
        alive[bb] = 0;
        for (int k = 0; k < nc; k++) { known[ba][k] = known[k][ba] = 0; if (of[k] == bb) of[k] = ba; }
        groups--;
    }
    /* Number the chosen groups in order of first class, and fit each. */
    int id[NVDR_ALF_CLASSES];
    for (int k = 0; k < nc; k++) id[k] = -1;
    out->nfilters = 0;
    for (int k = 0; k < nc; k++) {
        int gidx = best_map[k];
        if (id[gidx] < 0) id[gidx] = out->nfilters++;
        out->classmap[k] = (uint8_t)id[gidx];
    }
    for (int f = 0; f < out->nfilters; f++) {
        Stats m;
        memset(&m, 0, sizeof(m));
        for (int k = 0; k < nc; k++) if (out->classmap[k] == f) stats_add(&m, &st[k], n);
        solve(&m, n, w);
        for (int t = 0; t < n; t++) {
            double v = floor(w[t] * (1 << NVDR_ALF_SHIFT) + 0.5);
            out->coef[f][t] = (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
        }
    }
    int bk = 0, bb = 1 << 30;
    for (int k = 0; k < 4; k++) { int b = coef_bits(out, n, k); if (b < bb) { bb = b; bk = k; } }
    out->k = bk;
    return best;
}

static int cand_bits(const Cand* c, int comp, int units, int all) {
    int n = comp == 0 ? NVDR_ALF_TAPS_Y : NVDR_ALF_TAPS_C;
    int b = 3 + 2 + 2 + coef_bits(c, n, c->k) + 1 + (all ? 0 : units);
    if (comp == 0) b += 5 + NVDR_ALF_CLASSES * bits_for(c->nfilters);
    return b;
}

static void to_comp(const Cand* c, NvdrRestoreComp* rc) {
    memset(rc, 0, sizeof(*rc));
    rc->kind = NVDR_RESTORE_ALF;
    rc->clip = c->clip;
    rc->nfilters = c->nfilters;
    memcpy(rc->classmap, c->classmap, sizeof(rc->classmap));
    memcpy(rc->coef, c->coef, sizeof(rc->coef));
    rc->all_units = 1;
}

/* Squared error against the target, per unit. */
static void unit_sse(const uint8_t* p, const Target* T, const NvdrRestorePlane* g, int ux, double* sse) {
    int uy = (g->ph + g->unit - 1) / g->unit;
    memset(sse, 0, sizeof(double) * ux * uy);
    if (!T->up) {
        for (int y = 0; y < T->h; y++)
            for (int x = 0; x < T->w; x++) {
                double d = T->src[(size_t)y * g->pw + x] - p[(size_t)y * g->pw + x];
                sse[(y / g->unit) * ux + x / g->unit] += d * d;
            }
        return;
    }
    for (int y = 0; y < T->fh; y++) {
        Pair py = up_pair(y, T->h);
        const uint8_t* r0 = p + (size_t)py.a * g->pw;
        const uint8_t* r1 = p + (size_t)py.b * g->pw;
        for (int x = 0; x < T->fw; x++) {
            Pair px = up_pair(x, T->w);
            int v = (9 * r0[px.a] + 3 * r0[px.b] + 3 * r1[px.a] + r1[px.b] + 8) >> 4;
            double d = T->src[(size_t)y * T->fpw + x] - v;
            sse[(py.a / g->unit) * ux + px.a / g->unit] += d * d;
        }
    }
}

int nvdr_restore_fit(NvdrRestoreFit* fit, int comp, uint8_t* plane, const NvdrRestorePlane* g, int w, int h) {
    Target T = { fit->src[comp], w, h, 0, 0, 0, 0 };
    if (comp && fit->full_src[comp]) {
        T.up = 1; T.src = fit->full_src[comp];
        T.fpw = fit->full_pw; T.fw = fit->full_w; T.fh = fit->full_h;
    }
    double lambda = fit->lambda[comp];
    int ux = (g->pw + g->unit - 1) / g->unit, units = ux * ((g->ph + g->unit - 1) / g->unit);
    size_t npx = (size_t)g->pw * g->ph;
    uint8_t* q = padded(plane, g->pw, g->ph);
    uint8_t* cls = (uint8_t*)malloc(npx / 16 + 1);
    uint8_t* trial = (uint8_t*)malloc(npx);
    uint8_t* best_px = (uint8_t*)malloc(npx);
    uint8_t* mask = (uint8_t*)malloc((size_t)units);
    uint8_t* best_mask = (uint8_t*)malloc((size_t)units);
    double* off_sse = (double*)malloc(sizeof(double) * units);
    double* on_sse = (double*)malloc(sizeof(double) * units);
    Stats* st = (Stats*)malloc(sizeof(Stats) * NVDR_ALF_CLASSES);
    int rc = -1;
    if (!q || !cls || !trial || !best_px || !mask || !best_mask || !off_sse || !on_sse || !st) goto done;
    int s = g->pw + 2 * PAD;
    if (comp == 0)
        for (int y = 0; y < g->ph; y += 4)
            for (int x = 0; x < g->pw; x += 4) cls[(y >> 2) * (g->pw >> 2) + (x >> 2)] = (uint8_t)block_class(q, s, x, y);
    unit_sse(plane, &T, g, ux, off_sse);
    double d_off = 0;
    for (int u = 0; u < units; u++) d_off += off_sse[u];

    /* The clip bound by the gain design() expects from a sixteenth of the
     * pixels; then the filter from half of them (a quarter of the colour
     * judged at full size), and once more from the units it turned out to
     * be on in. */
    int clip = 0;
    double est = 1e300;
    for (int k = 0; k < 4; k++) {
        if (gather(q, cls, &T, g, comp, CLIP[k], NULL, ux, SUB_CLIP, st) != 0) goto done;
        Cand c;
        memset(&c, 0, sizeof(c));
        double e = design(st, comp, lambda, &c);
        if (e < est) { est = e; clip = k; }
    }
    double best_cost = d_off + lambda * 3;
    Cand best;
    int best_all = 1, found = 0;
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 1 && (!found || best_all)) break;
        if (gather(q, cls, &T, g, comp, CLIP[clip], pass == 1 ? best_mask : NULL, ux, T.up ? SUB_FIT_UP : SUB_FIT, st) != 0) goto done;
        Cand c;
        memset(&c, 0, sizeof(c));
        c.clip = clip;
        design(st, comp, lambda, &c);
        NvdrRestoreComp rcomp;
        to_comp(&c, &rcomp);
        memcpy(trial, plane, npx);
        alf_run(&rcomp, comp, q, trial, g);
        unit_sse(trial, &T, g, ux, on_sse);
        /* A unit is filtered where that removes more than the bit its
         * flag costs. */
        double d = 0;
        for (int u = 0; u < units; u++) {
            mask[u] = on_sse[u] + lambda < off_sse[u];
            d += mask[u] ? on_sse[u] : off_sse[u];
        }
        double d_all = 0;
        for (int u = 0; u < units; u++) d_all += on_sse[u];
        double cost_map = d + lambda * cand_bits(&c, comp, units, 0);
        double cost_all = d_all + lambda * cand_bits(&c, comp, units, 1);
        int use_all = cost_all <= cost_map;
        double cost = use_all ? cost_all : cost_map;
        if (cost < best_cost) {
            best_cost = cost; best = c; best_all = use_all; found = 1;
            if (use_all) memset(best_mask, 1, (size_t)units);
            else memcpy(best_mask, mask, (size_t)units);
            /* what the decoder will show: filtered units, the rest as it was */
            for (int y = 0; y < g->ph; y++)
                for (int x = 0; x < g->pw; x++) {
                    size_t at = (size_t)y * g->pw + x;
                    best_px[at] = best_mask[(y / g->unit) * ux + x / g->unit] ? trial[at] : plane[at];
                }
        }
    }

    NvdrBits* b = &fit->out;
    if (!found) {
        if (bits_put(b, NVDR_RESTORE_NONE, 3)) goto done;
        fit->gain[comp] = 0;
        fit->bits[comp] = 3;
        rc = 0;
        goto done;
    }
    int n = comp == 0 ? NVDR_ALF_TAPS_Y : NVDR_ALF_TAPS_C;
    size_t before = b->count * 8 - (b->bit ? 8 - b->bit : 0);
    if (bits_put(b, NVDR_RESTORE_ALF, 3) || bits_put(b, (unsigned)best.clip, 2) || bits_put(b, (unsigned)best.k, 2))
        goto done;
    if (comp == 0) {
        if (bits_put(b, (unsigned)(best.nfilters - 1), 5)) goto done;
        int bf = bits_for(best.nfilters);
        for (int i = 0; i < NVDR_ALF_CLASSES; i++) if (bits_put(b, best.classmap[i], bf)) goto done;
    }
    for (int f = 0; f < best.nfilters; f++)
        for (int t = 0; t < n; t++) if (put_signed(b, best.coef[f][t], best.k)) goto done;
    if (bits_put(b, (unsigned)best_all, 1)) goto done;
    if (!best_all) for (int u = 0; u < units; u++) if (bits_put(b, best_mask[u], 1)) goto done;
    size_t after = b->count * 8 - (b->bit ? 8 - b->bit : 0);
    fit->bits[comp] = after - before;
    double d_best = 0;
    unit_sse(best_px, &T, g, ux, on_sse);
    for (int u = 0; u < units; u++) d_best += on_sse[u];
    fit->gain[comp] = d_off - d_best;
    memcpy(plane, best_px, npx);
    rc = 0;

done:
    free(q); free(cls); free(trial); free(best_px); free(mask); free(best_mask);
    free(off_sse); free(on_sse); free(st);
    if (rc) fit->failed = 1;
    return rc;
}
