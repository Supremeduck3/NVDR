/*
 * A texture layer over the rectangles, measured before it is built.
 *
 * The quadtree codes structure well and texture badly: flat rectangles
 * cannot hold detail finer than the smallest tile, so the finest level
 * spends most of its bytes subdividing into grain for little PSNR. This
 * takes the rectangles at some cut — anchor only, up to R1, R1 with its
 * colour corrections but no new splits, or the full container — renders
 * them flat, and codes what is left with an 8x8 DCT through the same
 * adaptive arithmetic coder the rectangles use. Bytes are real: the base
 * is the container's own prefix and the texture is the coder's output.
 *
 *   texture <image> [--base anchor|r1|r1c|full] [--q Q,Q,...] [--out img]
 *
 * The coefficient coding is a small CABAC: a coded-block flag conditioned
 * on the left and top blocks, then in zigzag order a significance flag
 * and, after each significant coefficient, a last flag, both conditioned
 * on position; magnitudes as an adaptive greater-than-one flag, adaptive
 * unary, and an Exp-Golomb escape.
 */
#include "nvdr.h"
#include "entropy.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define B 8
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define POS_CTX 22
#define MAG_UNARY 14

static const int zigzag[64] = {
     0, 1, 8,16, 9, 2, 3,10,17,24,32,25,18,11, 4, 5,
    12,19,26,33,40,48,41,34,27,20,13, 6, 7,14,21,28,
    35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63 };

static int pos_ctx(int i) { return i < 16 ? i : 16 + (i - 16) / 8; }

typedef struct {
    uint16_t cbf[3][3];
    uint16_t sig[3][POS_CTX];
    uint16_t last[3][POS_CTX];
    uint16_t gt1[3][4];
    uint16_t mag[3][MAG_UNARY];
} TexModels;

static void models_init(TexModels* m) {
    uint16_t* p = (uint16_t*)m;
    for (size_t i = 0; i < sizeof(*m) / 2; i++) p[i] = NVDR_PROB_INIT;
}

static double dct_c[B][B];   /* dct_c[u][x] */

static void dct_init(void) {
    for (int u = 0; u < B; u++)
        for (int x = 0; x < B; x++)
            dct_c[u][x] = (u ? sqrt(2.0 / B) : sqrt(1.0 / B)) * cos((2 * x + 1) * u * M_PI / (2 * B));
}

static void fdct(const double in[64], double out[64]) {
    double t[64];
    for (int y = 0; y < B; y++)
        for (int u = 0; u < B; u++) {
            double s = 0; for (int x = 0; x < B; x++) s += dct_c[u][x] * in[y * B + x];
            t[y * B + u] = s;
        }
    for (int v = 0; v < B; v++)
        for (int u = 0; u < B; u++) {
            double s = 0; for (int y = 0; y < B; y++) s += dct_c[v][y] * t[y * B + u];
            out[v * B + u] = s;
        }
}

static void idct(const double in[64], double out[64]) {
    double t[64];
    for (int v = 0; v < B; v++)
        for (int x = 0; x < B; x++) {
            double s = 0; for (int u = 0; u < B; u++) s += dct_c[u][x] * in[v * B + u];
            t[v * B + x] = s;
        }
    for (int y = 0; y < B; y++)
        for (int x = 0; x < B; x++) {
            double s = 0; for (int v = 0; v < B; v++) s += dct_c[v][y] * t[v * B + x];
            out[y * B + x] = s;
        }
}

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

static void enc_mag(NvdrEncoder* e, TexModels* m, int ch, int ctx, int a) {
    nvdr_enc_bit(e, &m->gt1[ch][ctx], a > 1);
    if (a == 1) return;
    int r = a - 2;
    for (int i = 0; i < MAG_UNARY; i++) {
        nvdr_enc_bit(e, &m->mag[ch][i], r > i);
        if (r <= i) return;
    }
    /* Exp-Golomb order 0 on the rest. */
    unsigned v = (unsigned)(r - MAG_UNARY) + 1;
    int n = 0; while ((v >> n) > 1) n++;
    nvdr_enc_direct(e, 0, n);
    nvdr_enc_direct(e, v, n + 1);
}

static double psnr_rgb(const NvdrImage* a, const NvdrImage* b) {
    double se = 0; size_t n = (size_t)a->width * a->height * 3;
    for (size_t i = 0; i < n; i++) { double d = (double)a->pixels[i] - b->pixels[i]; se += d * d; }
    return se ? 10 * log10(255.0 * 255.0 * n / se) : 99.0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <image> [--base anchor|r1|r1c|full] [--q Q,Q,...] "
                        "[--chroma-q F] [--deadzone F] [--out img]\n", argv[0]);
        return 2;
    }
    const char* base_mode = "r1c";
    const char* qlist = "0,4,6,8,12,16,24";
    const char* out = NULL;
    double chroma_q = 1.5, deadzone = 0.33;
    for (int i = 2; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--base")) base_mode = argv[++i];
        else if (!strcmp(argv[i], "--q")) qlist = argv[++i];
        else if (!strcmp(argv[i], "--out")) out = argv[++i];
        else if (!strcmp(argv[i], "--chroma-q")) chroma_q = atof(argv[++i]);
        else if (!strcmp(argv[i], "--deadzone")) deadzone = atof(argv[++i]);
    }

    NvdrImage src;
    if (nvdr_image_load(&src, argv[1]) != 0) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    int w = src.width, h = src.height;
    dct_init();

    NvdrConfig cfg = nvdr_default_config();
    int levels = NVDR_LEVELS;
    int none = !strcmp(base_mode, "none");
    if (!strcmp(base_mode, "anchor")) levels = 1;
    else if (!strcmp(base_mode, "r1")) levels = 2;
    else if (!strcmp(base_mode, "r1c")) cfg.tolerance[2] = cfg.tolerance[1];

    uint8_t* blob; size_t len; NvdrHeader hdr;
    if (nvdr_encode_mem(&blob, &len, &src, &cfg, &hdr) != 0) { fprintf(stderr, "encode failed\n"); return 1; }
    size_t base_bytes = NVDR_HEADER_SIZE;
    for (int k = 0; k < levels; k++) base_bytes += hdr.stored_bytes[k];
    if (levels == NVDR_LEVELS) base_bytes = len;

    NvdrPyramid pyr; NvdrHeader dh;
    if (nvdr_decode_mem(blob, len, &pyr, &dh) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    NvdrImage base = { calloc((size_t)w * h * 3, 1), w, h };
    for (int k = 0; k < levels && k < pyr.levels_present; k++) nvdr_render_level(&pyr.level[k], &base);
    /* No rectangles at all: the DCT alone, against a flat grey, with its
     * DC predicted from the block to the left the way JPEG does. */
    if (none) { memset(base.pixels, 128, (size_t)w * h * 3); base_bytes = NVDR_HEADER_SIZE; }
    nvdr_pyramid_free(&pyr);
    free(blob);

    /* The residual in YCbCr, one plane per channel. */
    double* res[3];
    for (int c = 0; c < 3; c++) res[c] = calloc((size_t)w * h, sizeof(double));
    for (size_t i = 0; i < (size_t)w * h; i++) {
        double a[3], b[3];
        to_ycc(src.pixels + i * 3, a); to_ycc(base.pixels + i * 3, b);
        for (int c = 0; c < 3; c++) res[c][i] = a[c] - b[c];
    }

    printf("%s  %dx%d  base %s: %zu B, %.2f dB\n", argv[1], w, h, base_mode, base_bytes, psnr_rgb(&src, &base));
    printf("      Q   texture     total     PSNR   coded blocks\n");

    int nbx = (w + B - 1) / B, nby = (h + B - 1) / B;
    char buf[256]; snprintf(buf, sizeof buf, "%s", qlist);
    for (char* t = strtok(buf, ","); t; t = strtok(NULL, ",")) {
        double Q = atof(t);
        NvdrImage rec = { malloc((size_t)w * h * 3), w, h };
        if (Q <= 0) {
            memcpy(rec.pixels, base.pixels, (size_t)w * h * 3);
            printf("  %5.1f  %8d  %8zu  %6.2f dB\n", Q, 0, base_bytes, psnr_rgb(&src, &rec));
            free(rec.pixels); continue;
        }
        double* outp[3];
        for (int c = 0; c < 3; c++) outp[c] = calloc((size_t)w * h, sizeof(double));
        TexModels m; models_init(&m);
        NvdrEncoder enc; nvdr_enc_init(&enc, 1 << 16);
        uint8_t* coded = calloc((size_t)nbx * nby * 3, 1);
        long ncoded = 0;
        int prev_dc[3] = {0, 0, 0};

        for (int by = 0; by < nby; by++)
            for (int bx = 0; bx < nbx; bx++)
                for (int c = 0; c < 3; c++) {
                    double blk[64] = {0}, co[64];
                    for (int y = 0; y < B; y++)
                        for (int x = 0; x < B; x++) {
                            int px = bx * B + x, py = by * B + y;
                            if (px < w && py < h) blk[y * B + x] = res[c][(size_t)py * w + px];
                        }
                    fdct(blk, co);
                    double q = Q * (c ? chroma_q : 1.0);
                    /* DC as a difference from the left block's quantised DC. */
                    int dc_pred = bx > 0 ? prev_dc[c] : 0;
                    int lv[64], last = -1;
                    for (int i = 0; i < 64; i++) {
                        double v = co[zigzag[i]] / q - (i == 0 ? dc_pred : 0);
                        int a = (int)(fabs(v) + (1.0 - deadzone) - 0.5);
                        if (a < 0) a = 0;
                        lv[i] = v < 0 ? -a : a;
                        if (a) last = i;
                    }
                    prev_dc[c] = lv[0] + dc_pred;
                    size_t bi = ((size_t)by * nbx + bx) * 3 + c;
                    int ctx = (bx > 0 && coded[bi - 3]) + (by > 0 && coded[bi - (size_t)nbx * 3]);
                    coded[bi] = last >= 0;
                    nvdr_enc_bit(&enc, &m.cbf[c][ctx], coded[bi]);
                    if (last < 0) {
                        /* Nothing coded, but a predicted DC still lands. */
                        for (int y = 0; y < B; y++)
                            for (int x = 0; x < B; x++) {
                                int ppx = bx * B + x, ppy = by * B + y;
                                if (ppx < w && ppy < h) outp[c][(size_t)ppy * w + ppx] = dc_pred * q / B;
                            }
                        continue;
                    }
                    ncoded++;
                    int gctx = 0;
                    for (int i = 0; i <= last; i++) {
                        int a = lv[i] < 0 ? -lv[i] : lv[i];
                        if (i < 63) nvdr_enc_bit(&enc, &m.sig[c][pos_ctx(i)], a != 0);
                        if (!a) continue;
                        if (i < 63) nvdr_enc_bit(&enc, &m.last[c][pos_ctx(i)], i == last);
                        enc_mag(&enc, &m, c, gctx < 3 ? gctx : 3, a);
                        nvdr_enc_direct(&enc, lv[i] < 0, 1);
                        if (a > 1) gctx++;
                    }
                    double deq[64] = {0}, px[64];
                    for (int i = 0; i <= last; i++) deq[zigzag[i]] = lv[i] * q;
                    deq[0] = (lv[0] + dc_pred) * q;
                    idct(deq, px);
                    for (int y = 0; y < B; y++)
                        for (int x = 0; x < B; x++) {
                            int ppx = bx * B + x, ppy = by * B + y;
                            if (ppx < w && ppy < h) outp[c][(size_t)ppy * w + ppx] = px[y * B + x];
                        }
                }
        nvdr_enc_finish(&enc);

        for (size_t i = 0; i < (size_t)w * h; i++) {
            double b[3];
            to_ycc(base.pixels + i * 3, b);
            for (int c = 0; c < 3; c++) b[c] += outp[c][i];
            from_ycc(b, rec.pixels + i * 3);
        }
        printf("  %5.1f  %8zu  %8zu  %6.2f dB  %5.1f%%\n", Q, enc.count, base_bytes + enc.count,
               psnr_rgb(&src, &rec), 100.0 * ncoded / ((double)nbx * nby * 3));
        if (out) nvdr_image_write(&rec, out);
        nvdr_enc_free(&enc);
        for (int c = 0; c < 3; c++) free(outp[c]);
        free(coded); free(rec.pixels);
    }
    return 0;
}
