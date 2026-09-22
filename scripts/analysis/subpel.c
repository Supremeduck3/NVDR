/*
 * What sub-pixel motion is worth to a predicted frame, measured before it
 * is built.
 *
 * Takes two consecutive frames. Every block is searched at whole pixels,
 * then refined to half and to quarter pixels with bilinear interpolation.
 * The prediction at each precision is subtracted from the second frame
 * and the residual coded as a v10 still, the way nvdrv codes a predicted
 * frame. Reports the residual's energy, its bytes and the PSNR of what
 * the decoder would show. The reference is the source frame, not a
 * reconstruction, so this is the ceiling of what precision can buy.
 *
 *   subpel <frame0> <frame1> [--block N] [--q Q] [--range R] [--sixtap]
 *
 * --sixtap interpolates with H.264's 6-tap filter instead of bilinear.
 */
#include "nvdr.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int W, H;
static int sixtap = 0;

static int px_at(const unsigned char* ref, int x, int y, int c) {
    x = x < 0 ? 0 : x >= W ? W - 1 : x;
    y = y < 0 ? 0 : y >= H ? H - 1 : y;
    return ref[((size_t)y * W + x) * 3 + c];
}

/* H.264's luma interpolation: a 6-tap filter (1,-5,20,20,-5,1)/32 for half
 * pixels, horizontally, vertically, or both; quarter pixels as the rounded
 * average of the two nearest whole or half samples. Kept at full
 * precision between the passes for the centre sample. */
static int half_h(const unsigned char* r, int x, int y, int c) {
    return px_at(r,x-2,y,c) - 5*px_at(r,x-1,y,c) + 20*px_at(r,x,y,c) + 20*px_at(r,x+1,y,c) - 5*px_at(r,x+2,y,c) + px_at(r,x+3,y,c);
}
static int clip8(int v) { return v < 0 ? 0 : v > 255 ? 255 : v; }
/* Sample on the half-pixel lattice: hx, hy in half pixels. */
static int sample_half(const unsigned char* r, int hx, int hy, int c) {
    int x = hx >> 1, y = hy >> 1, fx = hx & 1, fy = hy & 1;
    if (!fx && !fy) return px_at(r, x, y, c);
    if (fx && !fy) return clip8((half_h(r, x, y, c) + 16) >> 5);
    if (!fx && fy) {
        int v = px_at(r,x,y-2,c) - 5*px_at(r,x,y-1,c) + 20*px_at(r,x,y,c) + 20*px_at(r,x,y+1,c) - 5*px_at(r,x,y+2,c) + px_at(r,x,y+3,c);
        return clip8((v + 16) >> 5);
    }
    int v = half_h(r,x,y-2,c) - 5*half_h(r,x,y-1,c) + 20*half_h(r,x,y,c) + 20*half_h(r,x,y+1,c) - 5*half_h(r,x,y+2,c) + half_h(r,x,y+3,c);
    return clip8((v + 512) >> 10);
}
static int sample6(const unsigned char* r, int x, int y, int fx, int fy, int c) {
    int qx = x * 4 + fx, qy = y * 4 + fy;           /* position in quarter pixels */
    int hx0 = qx >> 1, hy0 = qy >> 1;                /* half-lattice below */
    if (!(qx & 1) && !(qy & 1)) return sample_half(r, hx0, hy0, c);
    /* quarter: average the two nearest half-lattice samples */
    int ax = hx0 + (qx & 1), ay = hy0 + (qy & 1);
    if ((qx & 1) && (qy & 1)) return (sample_half(r, hx0 + 1, hy0, c) + sample_half(r, hx0, hy0 + 1, c) + 1) >> 1;
    return (sample_half(r, hx0, hy0, c) + sample_half(r, ax, ay, c) + 1) >> 1;
}

/* The reference at (x + fx/4, y + fy/4), fx and fy in quarter pixels,
 * bilinear, edges held. */
static int sample(const unsigned char* ref, int x, int y, int fx, int fy, int c) {
    if (sixtap) return sample6(ref, x, y, fx, fy, c);
    int ix = x + (fx >> 2), iy = y + (fy >> 2);    /* floor for negatives */
    int ax = fx & 3, ay = fy & 3;
    int x0 = ix < 0 ? 0 : ix >= W ? W - 1 : ix, x1 = ix + 1 < 0 ? 0 : ix + 1 >= W ? W - 1 : ix + 1;
    int y0 = iy < 0 ? 0 : iy >= H ? H - 1 : iy, y1 = iy + 1 < 0 ? 0 : iy + 1 >= H ? H - 1 : iy + 1;
    int a = ref[((size_t)y0 * W + x0) * 3 + c], b = ref[((size_t)y0 * W + x1) * 3 + c];
    int d = ref[((size_t)y1 * W + x0) * 3 + c], e = ref[((size_t)y1 * W + x1) * 3 + c];
    return ((4 - ax) * (4 - ay) * a + ax * (4 - ay) * b + (4 - ax) * ay * d + ax * ay * e + 8) >> 4;
}

static long block_sad(const unsigned char* cur, const unsigned char* ref,
                      int bx, int by, int n, int fx, int fy, long limit) {
    long acc = 0;
    for (int y = by; y < by + n && y < H; y++) {
        for (int x = bx; x < bx + n && x < W; x++)
            for (int c = 0; c < 3; c++) {
                int d = cur[((size_t)y * W + x) * 3 + c] - sample(ref, x, y, fx, fy, c);
                acc += d < 0 ? -d : d;
            }
        if (acc >= limit) return acc;
    }
    return acc;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <frame0> <frame1> [--block N] [--q Q] [--range R]\n", argv[0]); return 2; }
    int n = 16, range = 12;
    NvdrConfig cfg = nvdr_default_config();
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--block")) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--q")) cfg.q = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--range")) range = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sixtap")) { sixtap = 1; }
    }
    NvdrImage a, b;
    if (nvdr_image_load(&a, argv[1]) || nvdr_image_load(&b, argv[2])) { fprintf(stderr, "cannot read\n"); return 1; }
    W = a.width; H = a.height;
    int nbx = (W + n - 1) / n, nby = (H + n - 1) / n;
    int* vx = calloc((size_t)nbx * nby, sizeof(int));
    int* vy = calloc((size_t)nbx * nby, sizeof(int));

    printf("%s -> %s  %dx%d  block %d  q %d\n", argv[1], argv[2], W, H, n, cfg.q);
    printf("  precision   MSE of residual   residual bytes   PSNR shown\n");
    for (int prec = 0; prec < 3; prec++) {          /* 0 whole, 1 half, 2 quarter */
        for (int bi = 0; bi < nbx * nby; bi++) {
            int bx = (bi % nbx) * n, by = (bi / nbx) * n;
            long best = LONG_MAX; int bfx = 0, bfy = 0;
            if (prec == 0) {
                for (int dy = -range; dy <= range; dy++)
                    for (int dx = -range; dx <= range; dx++) {
                        long s = block_sad(b.pixels, a.pixels, bx, by, n, dx * 4, dy * 4, best);
                        if (s < best) { best = s; bfx = dx * 4; bfy = dy * 4; }
                    }
            } else {
                /* Refine around the previous precision's vector. */
                int stepq = prec == 1 ? 2 : 1;
                int cx = vx[bi], cy = vy[bi];
                for (int dy = -stepq; dy <= stepq; dy += stepq)
                    for (int dx = -stepq; dx <= stepq; dx += stepq) {
                        long s = block_sad(b.pixels, a.pixels, bx, by, n, cx + dx, cy + dy, best);
                        if (s < best) { best = s; bfx = cx + dx; bfy = cy + dy; }
                    }
            }
            vx[bi] = bfx; vy[bi] = bfy;
        }
        NvdrImage res = { malloc((size_t)W * H * 3), W, H };
        unsigned char* pred = malloc((size_t)W * H * 3);
        double se = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                int bi = (y / n) * nbx + x / n;
                for (int c = 0; c < 3; c++) {
                    size_t at = ((size_t)y * W + x) * 3 + c;
                    int p = sample(a.pixels, x, y, vx[bi], vy[bi], c);
                    pred[at] = (unsigned char)p;
                    int d = b.pixels[at] - p;
                    se += (double)d * d;
                    int v = d + 128;
                    res.pixels[at] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
                }
            }
        uint8_t* blob; size_t len;
        nvdr_encode_mem(&blob, &len, &res, &cfg, NULL);
        NvdrImage dec;
        nvdr_decode_mem(blob, len, -1, &dec, NULL, NULL);
        NvdrImage shown = { malloc((size_t)W * H * 3), W, H };
        for (size_t i = 0; i < (size_t)W * H * 3; i++) {
            int v = pred[i] + dec.pixels[i] - 128;
            shown.pixels[i] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
        static const char* names[] = { "whole", "half", "quarter" };
        printf("  %-9s   %15.2f   %14zu   %7.2f dB\n", names[prec], se / ((double)W * H * 3), len,
               nvdr_psnr(&b, &shown));
        free(blob); nvdr_image_free(&dec); free(res.pixels); free(pred); free(shown.pixels);
    }
    return 0;
}
