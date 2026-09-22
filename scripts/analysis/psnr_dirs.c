/*
 * Mean PSNR between two folders of frames, paired in name order.
 *
 * Both codecs in a comparison are measured by the same code against the
 * same reference, so neither gets a metric the other does not. Reports the
 * mean of per-frame PSNR, which is what nvdrv_decode prints, and the PSNR
 * of the mean squared error over the whole clip, which weights a bad frame
 * the same as a good one.
 */
#include "nvdr.h"
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

static int cmp(const void* a, const void* b) { return strcmp(*(char* const*)a, *(char* const*)b); }

static char** list(const char* dir, int* n) {
    DIR* d = opendir(dir); if (!d) return NULL;
    char** v = NULL; int c = 0, cap = 0; struct dirent* e;
    while ((e = readdir(d))) {
        const char* dot = strrchr(e->d_name, '.');
        if (!dot || (strcasecmp(dot, ".png") && strcasecmp(dot, ".jpg") && strcasecmp(dot, ".ppm"))) continue;
        if (c == cap) { cap = cap ? cap * 2 : 64; v = realloc(v, cap * sizeof(char*)); }
        v[c] = malloc(strlen(dir) + strlen(e->d_name) + 2);
        sprintf(v[c++], "%s/%s", dir, e->d_name);
    }
    closedir(d); qsort(v, c, sizeof(char*), cmp); *n = c; return v;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <reference-dir> <test-dir> [--smooth W]\n", argv[0]); return 2; }
    float smooth = 0.0f;
    for (int i = 3; i < argc - 1; i++) if (!strcmp(argv[i], "--smooth")) smooth = (float)atof(argv[i + 1]);
    int na, nb; char** a = list(argv[1], &na); char** b = list(argv[2], &nb);
    if (!a || !b || !na || !nb) { fprintf(stderr, "empty folder\n"); return 1; }
    int n = na < nb ? na : nb;
    double sum_db = 0, sum_mse = 0, worst = 1e9;
    for (int i = 0; i < n; i++) {
        NvdrImage x, y;
        if (nvdr_image_load(&x, a[i]) || nvdr_image_load(&y, b[i])) { fprintf(stderr, "cannot read pair %d\n", i); return 1; }
        if (x.width != y.width || x.height != y.height) { fprintf(stderr, "size mismatch at %d\n", i); return 1; }
        if (smooth > 0) nvdr_smooth(&y, smooth);
        double db = nvdr_psnr(&x, &y);
        sum_db += db; if (db < worst) worst = db;
        sum_mse += 255.0 * 255.0 / pow(10.0, db / 10.0);
        nvdr_image_free(&x); nvdr_image_free(&y);
    }
    printf("%d quadros  PSNR medio %.2f dB  PSNR do clipe %.2f dB  pior quadro %.2f dB\n",
           n, sum_db / n, 10.0 * log10(255.0 * 255.0 / (sum_mse / n)), worst);
    return 0;
}
