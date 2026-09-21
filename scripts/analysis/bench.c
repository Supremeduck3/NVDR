/*
 * Time the encoder and nothing else.
 *
 * `nvdr_encode` measures badly for this: its wall clock includes decoding
 * the JPEG in and decoding the container back to report PSNR, which
 * together are most of it, so a change that halves a phase of the encoder
 * shows up as a couple of percent. This loads once and encodes in a loop.
 */
#define _POSIX_C_SOURCE 200809L
#include "nvdr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <image> [reps] [--gradient F]\n", argv[0]); return 2; }
    int reps = argc > 2 && argv[2][0] != '-' ? atoi(argv[2]) : 5;
    NvdrConfig cfg = nvdr_default_config();
    for (int i = 2; i < argc - 1; i++)
        if (!strcmp(argv[i], "--gradient")) cfg.gradient = (float)atof(argv[i+1]);

    NvdrImage img;
    if (nvdr_image_load(&img, argv[1]) != 0) { fprintf(stderr, "cannot read\n"); return 1; }

    /* One untimed run so the page cache and the allocator are warm. */
    uint8_t* blob = NULL; size_t len = 0; NvdrHeader h;
    if (nvdr_encode_mem(&blob, &len, &img, &cfg, &h) != 0) return 1;
    free(blob);

    double enc = 0.0;
    for (int i = 0; i < reps; i++) {
        double t0 = now_s();
        if (nvdr_encode_mem(&blob, &len, &img, &cfg, &h) != 0) return 1;
        enc += now_s() - t0;
        if (i + 1 < reps) free(blob);
    }

    double dec = 0.0;
    for (int i = 0; i < reps; i++) {
        NvdrPyramid pyr; NvdrHeader hh;
        double t0 = now_s();
        if (nvdr_decode_mem(blob, len, &pyr, &hh) != 0) return 1;
        dec += now_s() - t0;
        nvdr_pyramid_free(&pyr);
    }
    free(blob);

    double mpx = (double)img.width * img.height / 1e6;
    printf("%-26s %5.2f Mpx  encode %7.1f ms (%5.1f ms/Mpx)  decode %6.1f ms  %zu B\n",
           argv[1], mpx, enc / reps * 1e3, enc / reps * 1e3 / mpx, dec / reps * 1e3, len);
    nvdr_image_free(&img);
    return 0;
}
