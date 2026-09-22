/*
 * Flicker: how much the decoded video changes where the source did not.
 *
 * Over every pair of consecutive frames, takes the pixels whose source
 * value is identical in both, and reports how many of them changed in
 * the decoded frames and by how much on average. A static background
 * that the codec re-codes a little differently every frame shows up here
 * as shimmer, even when each frame on its own scores well.
 *
 *   flicker <source-dir> <decoded-dir>
 */
#include "nvdr.h"

#include <dirent.h>
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
    if (argc < 3) { fprintf(stderr, "usage: %s <source-dir> <decoded-dir>\n", argv[0]); return 2; }
    int na, nb; char** a = list(argv[1], &na); char** b = list(argv[2], &nb);
    if (!a || !b) { fprintf(stderr, "empty folder\n"); return 1; }
    int n = na < nb ? na : nb;
    NvdrImage sp, dp;
    if (nvdr_image_load(&sp, a[0]) || nvdr_image_load(&dp, b[0])) return 1;
    double changed = 0, total = 0, sum = 0;
    for (int i = 1; i < n; i++) {
        NvdrImage sc, dc;
        if (nvdr_image_load(&sc, a[i]) || nvdr_image_load(&dc, b[i])) return 1;
        size_t m = (size_t)sc.width * sc.height * 3;
        for (size_t k = 0; k < m; k++) {
            if (sc.pixels[k] != sp.pixels[k]) continue;
            int d = dc.pixels[k] - dp.pixels[k];
            total++;
            if (d) { changed++; sum += d < 0 ? -d : d; }
        }
        nvdr_image_free(&sp); nvdr_image_free(&dp);
        sp = sc; dp = dc;
    }
    printf("static source samples %.0f: %.2f%% changed in the decode, mean |change| %.3f levels\n",
           total, total ? 100.0 * changed / total : 0.0, total ? sum / total : 0.0);
    return 0;
}
