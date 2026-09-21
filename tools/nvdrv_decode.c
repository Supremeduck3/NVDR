/*
 * nvdrv_decode — play a sequence back to files, and measure it.
 *
 * With --compare pointing at the frames it was built from, it reports the
 * PSNR of every frame. That number is the one that catches drift: a codec
 * that predicts from the wrong thing looks fine on frame 1 and walks away
 * from the source by frame 50, and only a per-frame measurement shows it.
 */
#include "nvdrv.h"

#include <strings.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cmp_name(const void* a, const void* b) {
    return strcmp(*(const char**)a, *(const char**)b);
}

static void free_frames(char** names, int n) {
    for (int i = 0; i < n; i++) free(names[i]);
    free(names);
}

static char** list_frames(const char* dir, int* count) {
    DIR* d = opendir(dir);
    if (!d) return NULL;
    char** names = NULL; int n = 0, cap = 0;
    struct dirent* e;
    while ((e = readdir(d))) {
        const char* dot = strrchr(e->d_name, '.');
        if (!dot) continue;
        if (strcasecmp(dot, ".png") && strcasecmp(dot, ".jpg") &&
            strcasecmp(dot, ".jpeg") && strcasecmp(dot, ".ppm") &&
            strcasecmp(dot, ".bmp") && strcasecmp(dot, ".tga")) continue;
        if (n == cap) { cap = cap ? cap*2 : 64; names = (char**)realloc(names, (size_t)cap*sizeof(char*)); }
        size_t len = strlen(dir) + strlen(e->d_name) + 2;
        names[n] = (char*)malloc(len);
        snprintf(names[n], len, "%s/%s", dir, e->d_name);
        n++;
    }
    closedir(d);
    qsort(names, (size_t)n, sizeof(char*), cmp_name);
    *count = n;
    return names;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s <in.nvdrv> [options]\n"
            "  --out DIR       write every frame as DIR/f%%03d.png\n"
            "  --compare DIR   PSNR of each frame against the source frames\n"
            "  --smooth W      seam blend on the written frames (default %.2f)\n",
            argv[0], NVDR_SMOOTH_DEFAULT);
        return 2;
    }
    const char* in = argv[1];
    const char* outdir = NULL;
    const char* cmpdir = NULL;
    float smooth = NVDR_SMOOTH_DEFAULT;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--out") && i+1 < argc) outdir = argv[++i];
        else if (!strcmp(argv[i], "--compare") && i+1 < argc) cmpdir = argv[++i];
        else if (!strcmp(argv[i], "--smooth") && i+1 < argc) smooth = (float)atof(argv[++i]);
    }

    NvdrvDecoder* dec;
    NvdrvInfo info;
    if (nvdrv_decode_open(&dec, in, &info) != 0) {
        fprintf(stderr, "cannot decode '%s' (not an NVDRV file)\n", in);
        return 1;
    }

    char** srcs = NULL; int nsrc = 0;
    if (cmpdir) srcs = list_frames(cmpdir, &nsrc);

    printf("%s  %dx%d  %d frames claimed  %d fps  gop %d\n",
           in, info.width, info.height, info.frame_count, info.fps, info.gop);
    if (cmpdir) printf("  frame  tipo     PSNR\n");

    NvdrImage f;
    f.width = info.width; f.height = info.height;
    f.pixels = (unsigned char*)malloc((size_t)f.width * f.height * 3);
    if (!f.pixels) return 1;

    int i = 0, kind = 0, partial = 0, rc;
    double psnr_sum = 0.0; int psnr_n = 0;
    while ((rc = nvdrv_decode_next(dec, &f, &kind, &partial)) == 1) {
        if (cmpdir && i < nsrc) {
            NvdrImage src;
            if (nvdr_image_load(&src, srcs[i]) == 0) {
                NvdrImage shown = f;
                unsigned char* copy = NULL;
                if (smooth > 0.0f) {
                    copy = (unsigned char*)malloc((size_t)f.width*f.height*3);
                    memcpy(copy, f.pixels, (size_t)f.width*f.height*3);
                    shown.pixels = copy;
                    nvdr_smooth(&shown, smooth);
                }
                double p = nvdr_psnr(&src, &shown);
                psnr_sum += p; psnr_n++;
                printf("  %5d  %-6s %6.2f dB%s\n", i,
                       kind == NVDRV_INTRA ? "INTRA" : "pred", p,
                       partial ? "   (quadro parcial)" : "");
                free(copy);
                nvdr_image_free(&src);
            }
        }
        if (outdir) {
            char path[512];
            snprintf(path, sizeof(path), "%s/f%03d.png", outdir, i);
            NvdrImage shown = f;
            unsigned char* copy = (unsigned char*)malloc((size_t)f.width*f.height*3);
            memcpy(copy, f.pixels, (size_t)f.width*f.height*3);
            shown.pixels = copy;
            if (smooth > 0.0f) nvdr_smooth(&shown, smooth);
            nvdr_image_write(&shown, path);
            free(copy);
        }
        i++;
    }
    if (rc < 0) fprintf(stderr, "stream went bad after %d frames\n", i);

    printf("\n  %d frames decodificados", i);
    if (psnr_n) printf(", PSNR medio %.2f dB", psnr_sum / psnr_n);
    printf("\n");

    free(f.pixels);
    if (srcs) free_frames(srcs, nsrc);
    nvdrv_decode_close(dec);
    return 0;
}
