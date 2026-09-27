/*
 * One image's rate-distortion curve for the still codec: encode at each
 * step given, decode back, print bytes, PSNR-Y (BT.601 luma of both
 * pictures in full precision) and PSNR-RGB. `scripts/analysis/bdrate.py`
 * turns two builds' curves into a BD-rate.
 *
 *   rdcurve <image> [--directional] [--no-adaptive-tx] q1 q2 ...
 */
#define _POSIX_C_SOURCE 200809L
#include "nvdr.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double psnr_y(const NvdrImage* a, const NvdrImage* b) {
    double mse = 0.0;
    size_t n = (size_t)a->width * a->height;
    for (size_t i = 0; i < n; i++) {
        const unsigned char *p = a->pixels + 3 * i, *q = b->pixels + 3 * i;
        double ya = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
        double yb = 0.299 * q[0] + 0.587 * q[1] + 0.114 * q[2];
        mse += (ya - yb) * (ya - yb);
    }
    mse /= (double)n;
    return mse <= 0.0 ? 99.0 : 10.0 * log10(255.0 * 255.0 / mse);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <image> [--directional] [--no-adaptive-tx] q...\n", argv[0]); return 2; }
    NvdrImage img;
    if (nvdr_image_load(&img, argv[1]) != 0) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    NvdrConfig cfg = nvdr_default_config();
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--directional")) { cfg.directional = 1; continue; }
        if (!strcmp(argv[i], "--no-adaptive-tx")) { cfg.adaptive_tx = 0; continue; }
        cfg.q = atoi(argv[i]);
        uint8_t* buf = NULL; size_t len = 0; NvdrHeader h;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (nvdr_encode_mem(&buf, &len, &img, &cfg, &h) != 0) { fprintf(stderr, "encode failed\n"); return 1; }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        NvdrImage out;
        if (nvdr_decode_mem(buf, len, -1, &out, NULL, NULL) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        printf("%d %zu %.4f %.4f %.3f\n", cfg.q, len, psnr_y(&img, &out), nvdr_psnr(&img, &out),
               (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9);
        free(buf);
        nvdr_image_free(&out);
    }
    nvdr_image_free(&img);
    return 0;
}
