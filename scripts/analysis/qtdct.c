/*
 * The quadtree as the transform's partition, measured before it is built.
 *
 * scripts/analysis/texture showed flat rectangles under a fixed 8x8 DCT
 * make it worse: their edges land in the residual. Here the quadtree is
 * the transform's partition instead. Every leaf is a square of 4 to 32
 * pixels on a power-of-two grid; its colour is predicted from the pixels
 * already decoded above and to its left, and what the prediction misses
 * is coded with a DCT of the leaf's own size. A leaf with no AC
 * coefficients is exactly a flat rectangle, so the old codec is the
 * special case where texture is never sent.
 *
 * Whether a node splits is decided by rate and distortion: the leaf and
 * its four children are both coded against the adaptive models as they
 * stand, and the cheaper of D + lambda * R wins. The same models then
 * code the chosen tree for real, so the byte count is the coder's.
 *
 *   qtdct <image> [--q Q,Q,...] [--lambda K] [--max N] [--dc-only]
 *                 [--pred dc|none] [--out img]
 *
 * lambda is K * Q^2, the usual relation between a uniform quantiser's step
 * and the slope of its rate-distortion curve.
 */
#include "nvdr.h"
#include "entropy.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MIN_N    4
#define MAX_N    32
#define NSIZES   4          /* 4, 8, 16, 32 */
#define POS_CTX  15
#define MAG_UNARY 14

static int size_class(int n) { return n == 4 ? 0 : n == 8 ? 1 : n == 16 ? 2 : 3; }

/* --------------------------------------------------------------- DCT */

static double* dct_m[NSIZES];     /* dct_m[s][u * n + x] */
static int*    scan[NSIZES];      /* diagonal scan: position index u + v * n */
static uint8_t* posctx[NSIZES];   /* context of each scan position */

static void tables_init(void) {
    for (int s = 0; s < NSIZES; s++) {
        int n = MIN_N << s;
        dct_m[s] = malloc(sizeof(double) * n * n);
        for (int u = 0; u < n; u++)
            for (int x = 0; x < n; x++)
                dct_m[s][u * n + x] = (u ? sqrt(2.0 / n) : sqrt(1.0 / n)) *
                                      cos((2 * x + 1) * u * M_PI / (2.0 * n));
        scan[s] = malloc(sizeof(int) * n * n);
        posctx[s] = malloc((size_t)n * n);
        int k = 0;
        for (int d = 0; d <= 2 * (n - 1); d++)
            for (int v = 0; v < n; v++) {
                int u = d - v;
                if (u < 0 || u >= n) continue;
                scan[s][k] = v * n + u;
                posctx[s][k] = (uint8_t)(d < 8 ? d : 8 + ((d - 8) / 4 < 6 ? (d - 8) / 4 : 6));
                k++;
            }
    }
}

static void fdct(int s, const double* in, double* out) {
    int n = MIN_N << s; const double* m = dct_m[s];
    double t[MAX_N * MAX_N];
    for (int y = 0; y < n; y++)
        for (int u = 0; u < n; u++) {
            double a = 0; for (int x = 0; x < n; x++) a += m[u * n + x] * in[y * n + x];
            t[y * n + u] = a;
        }
    for (int v = 0; v < n; v++)
        for (int u = 0; u < n; u++) {
            double a = 0; for (int y = 0; y < n; y++) a += m[v * n + y] * t[y * n + u];
            out[v * n + u] = a;
        }
}

static void idct(int s, const double* in, double* out) {
    int n = MIN_N << s; const double* m = dct_m[s];
    double t[MAX_N * MAX_N];
    for (int v = 0; v < n; v++)
        for (int x = 0; x < n; x++) {
            double a = 0; for (int u = 0; u < n; u++) a += m[u * n + x] * in[v * n + u];
            t[v * n + x] = a;
        }
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            double a = 0; for (int v = 0; v < n; v++) a += m[v * n + y] * t[v * n + x];
            out[y * n + x] = a;
        }
}

/* ------------------------------------------------------------ models */

typedef struct {
    uint16_t split[NSIZES];
    uint16_t cbf[NSIZES][3];
    uint16_t sig[NSIZES][3][POS_CTX];
    uint16_t last[NSIZES][3][POS_CTX];
    uint16_t gt1[3][4];
    uint16_t mag[3][MAG_UNARY];
} Models;

static void models_init(Models* m) {
    uint16_t* p = (uint16_t*)m;
    for (size_t i = 0; i < sizeof(*m) / 2; i++) p[i] = NVDR_PROB_INIT;
}

/* Cost of a bit in 1/256ths of a bit, without touching the model. */
static double bitcost[2][1 << NVDR_PROB_BITS];
static void bitcost_init(void) {
    for (int p = 1; p < (1 << NVDR_PROB_BITS); p++) {
        double p0 = (double)p / (1 << NVDR_PROB_BITS);
        bitcost[0][p] = -log2(p0);
        bitcost[1][p] = -log2(1.0 - p0);
    }
    bitcost[0][0] = bitcost[1][0] = 16;
}

/* One coder front end for both uses: `enc` NULL means count bits against
 * the models as they are, without adapting them. */
typedef struct { NvdrEncoder* enc; Models* m; double bits; } Sink;

static void put_bit(Sink* s, uint16_t* p, int bit) {
    if (s->enc) nvdr_enc_bit(s->enc, p, bit);
    else s->bits += bitcost[bit][*p];
}
static void put_direct(Sink* s, uint32_t v, int n) {
    if (s->enc) nvdr_enc_direct(s->enc, v, n);
    else s->bits += n;
}

static void put_mag(Sink* s, int c, int ctx, int a) {
    put_bit(s, &s->m->gt1[c][ctx], a > 1);
    if (a == 1) return;
    int r = a - 2;
    for (int i = 0; i < MAG_UNARY; i++) {
        put_bit(s, &s->m->mag[c][i], r > i);
        if (r <= i) return;
    }
    unsigned v = (unsigned)(r - MAG_UNARY) + 1;
    int n = 0; while ((v >> n) > 1) n++;
    put_direct(s, 0, n);
    put_direct(s, v, n + 1);
}

/* Levels in scan order; returns nothing, only emits. */
static void put_block(Sink* s, int sc, int c, const int* lv, int count) {
    int last = -1;
    for (int i = 0; i < count; i++) if (lv[i]) last = i;
    put_bit(s, &s->m->cbf[sc][c], last >= 0);
    if (last < 0) return;
    int g = 0;
    for (int i = 0; i <= last; i++) {
        int a = lv[i] < 0 ? -lv[i] : lv[i];
        int pc = posctx[sc][i];
        if (i < count - 1) put_bit(s, &s->m->sig[sc][c][pc], a != 0);
        if (!a) continue;
        if (i < count - 1) put_bit(s, &s->m->last[sc][c][pc], i == last);
        put_mag(s, c, g < 3 ? g : 3, a);
        put_direct(s, lv[i] < 0, 1);
        if (a > 1) g++;
    }
}

/* ------------------------------------------------------------- codec */

typedef struct {
    int w, h, pw, ph;             /* image, and padded to a multiple of MIN_N */
    double* src[3];               /* YCbCr, padded by replication */
    double* rec[3];               /* reconstruction so far */
    double Q, chroma_q, deadzone, lambda;
    int max_n, dc_only, pred;
    uint8_t* split;               /* decisions, one per node, by level */
    int levels;
    Models m;
} Ctx;

static size_t node_index(const Ctx* c, int x, int y, int n) {
    /* One grid per size; each is addressed by the node's top-left. */
    int s = size_class(n);
    int gw = (c->pw + n - 1) / n, gh = (c->ph + n - 1) / n;
    size_t base = 0;
    for (int k = 0; k < s; k++) {
        int m = MIN_N << k;
        base += (size_t)((c->pw + m - 1) / m) * ((c->ph + m - 1) / m);
    }
    (void)gh;
    return base + (size_t)(y / n) * gw + (x / n);
}

/* The colour a leaf is predicted as: the mean of the decoded row above and
 * column to the left, whichever exist. The flat fill of the old codec,
 * except that the decoder can compute it and it costs nothing to send. */
static double predict_dc(const Ctx* c, int ch, int x, int y, int n) {
    if (!c->pred) return ch ? 0.0 : 128.0;
    double sum = 0; int k = 0;
    if (y > 0) for (int i = 0; i < n && x + i < c->pw; i++) { sum += c->rec[ch][(size_t)(y - 1) * c->pw + x + i]; k++; }
    if (x > 0) for (int i = 0; i < n && y + i < c->ph; i++) { sum += c->rec[ch][(size_t)(y + i) * c->pw + x - 1]; k++; }
    return k ? sum / k : (ch ? 0.0 : 128.0);
}

/* Code one leaf: predict, transform, quantise, emit, reconstruct into rec.
 * Returns the squared error over the pixels inside the image. */
static double code_leaf(Ctx* c, Sink* s, int x, int y, int n) {
    int sc = size_class(n), cnt = n * n;
    double err = 0;
    for (int ch = 0; ch < 3; ch++) {
        double p = predict_dc(c, ch, x, y, n);
        double blk[MAX_N * MAX_N], co[MAX_N * MAX_N], px[MAX_N * MAX_N];
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                blk[j * n + i] = c->src[ch][(size_t)(y + j) * c->pw + x + i] - p;
        fdct(sc, blk, co);
        double q = c->Q * (ch ? c->chroma_q : 1.0);
        int lv[MAX_N * MAX_N];
        for (int i = 0; i < cnt; i++) {
            if (c->dc_only && i > 0) { lv[i] = 0; continue; }
            double v = co[scan[sc][i]] / q;
            int a = (int)(fabs(v) + (1.0 - c->deadzone) - 0.5);
            if (a < 0) a = 0;
            lv[i] = v < 0 ? -a : a;
        }
        put_block(s, sc, ch, lv, cnt);
        double dq[MAX_N * MAX_N];
        memset(dq, 0, sizeof(double) * cnt);
        for (int i = 0; i < cnt; i++) dq[scan[sc][i]] = lv[i] * q;
        idct(sc, dq, px);
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++) {
                size_t at = (size_t)(y + j) * c->pw + x + i;
                double r = px[j * n + i] + p;
                c->rec[ch][at] = r;
                if (x + i < c->w && y + j < c->h) { double d = c->src[ch][at] - r; err += d * d; }
            }
    }
    return err;
}

static void save_region(const Ctx* c, int x, int y, int n, double* buf) {
    for (int ch = 0; ch < 3; ch++)
        for (int j = 0; j < n; j++)
            memcpy(buf + ((size_t)ch * n + j) * n, c->rec[ch] + (size_t)(y + j) * c->pw + x, sizeof(double) * n);
}
static void load_region(Ctx* c, int x, int y, int n, const double* buf) {
    for (int ch = 0; ch < 3; ch++)
        for (int j = 0; j < n; j++)
            memcpy(c->rec[ch] + (size_t)(y + j) * c->pw + x, buf + ((size_t)ch * n + j) * n, sizeof(double) * n);
}

/* A node that reaches past the padded canvas must split; one wholly past
 * it does not exist. */
static int inside(const Ctx* c, int x, int y) { return x < c->pw && y < c->ph; }
static int whole(const Ctx* c, int x, int y, int n) { return x + n <= c->pw && y + n <= c->ph; }

static double search(Ctx* c, int x, int y, int n) {
    size_t id = node_index(c, x, y, n);
    int can_leaf = whole(c, x, y, n);
    int can_split = n > MIN_N;
    double leaf_cost = 1e300, split_cost = 1e300;

    double* saved = NULL;
    if (can_leaf) {
        Sink s = { NULL, &c->m, 0 };
        if (can_split) put_bit(&s, &c->m.split[size_class(n)], 0);
        double d = code_leaf(c, &s, x, y, n);
        leaf_cost = d + c->lambda * s.bits;
        if (!can_split) { c->split[id] = 0; return leaf_cost; }
        saved = malloc(sizeof(double) * 3 * n * n);
        save_region(c, x, y, n, saved);
    }
    {
        Sink s = { NULL, &c->m, 0 };
        if (can_leaf) put_bit(&s, &c->m.split[size_class(n)], 1);
        split_cost = c->lambda * s.bits;
        int h = n / 2;
        for (int k = 0; k < 4; k++) {
            int cx = x + (k & 1) * h, cy = y + (k >> 1) * h;
            if (inside(c, cx, cy)) split_cost += search(c, cx, cy, h);
        }
    }
    if (can_leaf && leaf_cost <= split_cost) {
        load_region(c, x, y, n, saved);
        c->split[id] = 0;
        free(saved);
        return leaf_cost;
    }
    free(saved);
    c->split[id] = 1;
    return split_cost;
}

static void emit(Ctx* c, Sink* s, int x, int y, int n) {
    size_t id = node_index(c, x, y, n);
    int can_leaf = whole(c, x, y, n);
    if (n > MIN_N && can_leaf) put_bit(s, &c->m.split[size_class(n)], c->split[id]);
    if (n > MIN_N && (c->split[id] || !can_leaf)) {
        int h = n / 2;
        for (int k = 0; k < 4; k++) {
            int cx = x + (k & 1) * h, cy = y + (k >> 1) * h;
            if (inside(c, cx, cy)) emit(c, s, cx, cy, h);
        }
        return;
    }
    code_leaf(c, s, x, y, n);
}

/* ------------------------------------------------------------ colour */

static void to_ycc(const unsigned char* p, double* o) {
    double r = p[0], g = p[1], b = p[2];
    o[0] = 0.299 * r + 0.587 * g + 0.114 * b;
    o[1] = -0.168736 * r - 0.331264 * g + 0.5 * b;
    o[2] = 0.5 * r - 0.418688 * g - 0.081312 * b;
}
static unsigned char clamp8(double v) { return v < 0 ? 0 : v > 255 ? 255 : (unsigned char)(v + 0.5); }
static void from_ycc(const double* o, unsigned char* p) {
    p[0] = clamp8(o[0] + 1.402 * o[2]);
    p[1] = clamp8(o[0] - 0.344136 * o[1] - 0.714136 * o[2]);
    p[2] = clamp8(o[0] + 1.772 * o[1]);
}

static double psnr_rgb(const NvdrImage* a, const NvdrImage* b) {
    double se = 0; size_t n = (size_t)a->width * a->height * 3;
    for (size_t i = 0; i < n; i++) { double d = (double)a->pixels[i] - b->pixels[i]; se += d * d; }
    return se ? 10 * log10(255.0 * 255.0 * n / se) : 99.0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <image> [--q Q,Q,...] [--lambda K] [--max N] [--dc-only] "
                        "[--pred dc|none] [--chroma-q F] [--deadzone F] [--out img]\n", argv[0]);
        return 2;
    }
    const char* qlist = "4,6,8,12,16,24,32,48";
    const char* out = NULL;
    double K = 0.12, chroma_q = 1.5, deadzone = 0.33;
    int max_n = MAX_N, dc_only = 0, pred = 1;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--q") && i + 1 < argc) qlist = argv[++i];
        else if (!strcmp(argv[i], "--lambda") && i + 1 < argc) K = atof(argv[++i]);
        else if (!strcmp(argv[i], "--max") && i + 1 < argc) max_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dc-only")) dc_only = 1;
        else if (!strcmp(argv[i], "--pred") && i + 1 < argc) pred = strcmp(argv[++i], "none") != 0;
        else if (!strcmp(argv[i], "--chroma-q") && i + 1 < argc) chroma_q = atof(argv[++i]);
        else if (!strcmp(argv[i], "--deadzone") && i + 1 < argc) deadzone = atof(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    }
    if (max_n < MIN_N) max_n = MIN_N;
    if (max_n > MAX_N) max_n = MAX_N;

    NvdrImage img;
    if (nvdr_image_load(&img, argv[1]) != 0) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    tables_init();
    bitcost_init();

    Ctx c;
    memset(&c, 0, sizeof c);
    c.w = img.width; c.h = img.height;
    c.pw = (c.w + MIN_N - 1) / MIN_N * MIN_N;
    c.ph = (c.h + MIN_N - 1) / MIN_N * MIN_N;
    for (int ch = 0; ch < 3; ch++) {
        c.src[ch] = malloc(sizeof(double) * c.pw * c.ph);
        c.rec[ch] = malloc(sizeof(double) * c.pw * c.ph);
    }
    for (int y = 0; y < c.ph; y++)
        for (int x = 0; x < c.pw; x++) {
            int sx = x < c.w ? x : c.w - 1, sy = y < c.h ? y : c.h - 1;
            double o[3]; to_ycc(img.pixels + ((size_t)sy * c.w + sx) * 3, o);
            for (int ch = 0; ch < 3; ch++) c.src[ch][(size_t)y * c.pw + x] = o[ch];
        }
    size_t nodes = 0;
    for (int k = 0; k < NSIZES; k++) {
        int m = MIN_N << k;
        nodes += (size_t)((c.pw + m - 1) / m) * ((c.ph + m - 1) / m);
    }
    c.split = calloc(nodes, 1);
    c.chroma_q = chroma_q; c.deadzone = deadzone; c.max_n = max_n; c.dc_only = dc_only; c.pred = pred;

    printf("%s  %dx%d  max %d  lambda %.3f*Q^2%s\n", argv[1], c.w, c.h, max_n, K, dc_only ? "  dc-only" : "");
    printf("      Q     bytes     PSNR   leaves 4/8/16/32\n");

    char buf[256]; snprintf(buf, sizeof buf, "%s", qlist);
    for (char* t = strtok(buf, ","); t; t = strtok(NULL, ",")) {
        c.Q = atof(t);
        c.lambda = K * c.Q * c.Q;
        models_init(&c.m);
        NvdrEncoder enc;
        nvdr_enc_init(&enc, 1 << 16);
        Sink s = { &enc, &c.m, 0 };
        /* Search a CTU against the models as they stand, then code it for
         * real, which is what adapts them for the next one. */
        for (int y = 0; y < c.ph; y += max_n)
            for (int x = 0; x < c.pw; x += max_n) {
                search(&c, x, y, max_n);
                emit(&c, &s, x, y, max_n);
            }
        nvdr_enc_finish(&enc);

        long leaves[NSIZES] = {0};
        for (int k = 0; k < NSIZES; k++) (void)k;
        /* Count leaves by walking the decisions. */
        for (int y = 0; y < c.ph; y += max_n)
            for (int x = 0; x < c.pw; x += max_n) {
                int stack[256][3], sp = 0;
                stack[sp][0] = x; stack[sp][1] = y; stack[sp][2] = max_n; sp++;
                while (sp) {
                    sp--; int nx = stack[sp][0], ny = stack[sp][1], nn = stack[sp][2];
                    if (!inside(&c, nx, ny)) continue;
                    size_t id = node_index(&c, nx, ny, nn);
                    if (nn > MIN_N && (c.split[id] || !whole(&c, nx, ny, nn))) {
                        int h = nn / 2;
                        for (int k = 0; k < 4; k++) {
                            stack[sp][0] = nx + (k & 1) * h; stack[sp][1] = ny + (k >> 1) * h; stack[sp][2] = h; sp++;
                        }
                    } else leaves[size_class(nn)]++;
                }
            }

        NvdrImage rec = { malloc((size_t)c.w * c.h * 3), c.w, c.h };
        for (int y = 0; y < c.h; y++)
            for (int x = 0; x < c.w; x++) {
                double o[3];
                for (int ch = 0; ch < 3; ch++) o[ch] = c.rec[ch][(size_t)y * c.pw + x];
                from_ycc(o, rec.pixels + ((size_t)y * c.w + x) * 3);
            }
        /* 16 bytes stands in for a header: dimensions, Q, tree depth. */
        printf("  %5.1f  %8zu  %6.2f dB  %ld/%ld/%ld/%ld\n", c.Q, enc.count + 16, psnr_rgb(&img, &rec),
               leaves[0], leaves[1], leaves[2], leaves[3]);
        if (out) nvdr_image_write(&rec, out);
        free(rec.pixels);
        nvdr_enc_free(&enc);
    }
    return 0;
}
