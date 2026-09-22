/*
 * nvdrv_encode — build a sequence container from a directory of frames,
 * and report what each frame cost and how it was coded.
 *
 * The per-frame report is the point, the same way it is for stills. A
 * predicted frame that keeps coming out the size of an intra one means the
 * prediction is not working, and that has to be visible rather than
 * averaged away into a bitrate.
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
    char** names = NULL;
    int n = 0, cap = 0;
    struct dirent* e;
    while ((e = readdir(d))) {
        const char* dot = strrchr(e->d_name, '.');
        if (!dot) continue;
        if (strcasecmp(dot, ".png") && strcasecmp(dot, ".jpg") &&
            strcasecmp(dot, ".jpeg") && strcasecmp(dot, ".ppm") &&
            strcasecmp(dot, ".bmp") && strcasecmp(dot, ".tga")) continue;
        if (n == cap) {
            cap = cap ? cap * 2 : 64;
            names = (char**)realloc(names, (size_t)cap * sizeof(char*));
        }
        size_t len = strlen(dir) + strlen(e->d_name) + 2;
        names[n] = (char*)malloc(len);
        snprintf(names[n], len, "%s/%s", dir, e->d_name);
        n++;
    }
    closedir(d);
    /* Sorted by name, which is what the frame numbering is for. */
    qsort(names, (size_t)n, sizeof(char*), cmp_name);
    *count = n;
    return names;
}

static void usage(const char* a0) {
    fprintf(stderr,
        "usage: %s <frames-dir> <out.nvdrv> [options]\n"
        "  --gop N          force an intra frame every N (default 48; 0 = only the first)\n"
        "  --search N       global motion search half-width in px (default 12; 0 disables)\n"
        "  --intra-thresh F mean absolute error above which a frame goes intra (default 24)\n"
        "  --block N        per-block motion with NxN blocks (0 = one global vector)\n"
        "  --fps N          recorded in the header (default 24)\n"
        "  --tolerance A,B,C  per-frame tolerance, coarse to fine\n"
        "  --gradient F     per-frame ramp multiplier (0 disables)\n"
        "  --limit N        stop after N frames\n", a0);
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 2; }
    const char* dir = argv[1];
    const char* out = argv[2];
    NvdrvConfig cfg = nvdrv_default_config();
    int limit = 0;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--gop") && i+1 < argc) cfg.gop = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--search") && i+1 < argc) cfg.search = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--block") && i+1 < argc) cfg.block = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--intra-thresh") && i+1 < argc) cfg.intra_threshold = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--fps") && i+1 < argc) cfg.fps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gradient") && i+1 < argc) cfg.frame.gradient = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--limit") && i+1 < argc) limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tolerance") && i+1 < argc) {
            char buf[128]; snprintf(buf, sizeof(buf), "%s", argv[++i]);
            int k = 0;
            for (char* t = strtok(buf, ","); t && k < NVDR_LEVELS; t = strtok(NULL, ","))
                cfg.frame.tolerance[k++] = (float)atof(t);
            if (k != NVDR_LEVELS) { fprintf(stderr, "--tolerance needs %d values\n", NVDR_LEVELS); return 2; }
        } else { usage(argv[0]); return 2; }
    }

    int n = 0;
    char** names = list_frames(dir, &n);
    if (!names || n == 0) { fprintf(stderr, "no frames in '%s'\n", dir); return 1; }
    if (limit > 0 && limit < n) n = limit;

    NvdrImage first;
    if (nvdr_image_load(&first, names[0]) != 0) {
        fprintf(stderr, "cannot read %s\n", names[0]); return 1;
    }

    NvdrvEncoder* enc;
    if (nvdrv_encode_open(&enc, out, first.width, first.height, &cfg) != 0) {
        fprintf(stderr, "cannot open '%s' for writing\n", out);
        free_frames(names, n);
        nvdr_image_free(&first);
        return 1;
    }

    printf("%d frames  %dx%d  gop %d  search %d  block %d\n", n, first.width, first.height,
           cfg.gop, cfg.search, cfg.block);
    printf("  frame  tipo      bytes    mv     acumulado\n");

    size_t total = NVDRV_HEADER_SIZE, intra_total = 0;
    int intra_count = 0;
    for (int i = 0; i < n; i++) {
        NvdrImage f;
        if (i == 0) f = first;
        else if (nvdr_image_load(&f, names[i]) != 0) {
            fprintf(stderr, "cannot read %s\n", names[i]); break;
        }
        if (f.width != first.width || f.height != first.height) {
            fprintf(stderr, "%s is %dx%d, expected %dx%d\n",
                    names[i], f.width, f.height, first.width, first.height);
            nvdr_image_free(&f);
            break;
        }
        int kind = 0, dx = 0, dy = 0;
        size_t bytes = 0;
        if (nvdrv_encode_frame(enc, &f, &kind, &bytes, &dx, &dy) != 0) {
            fprintf(stderr, "encode failed on %s\n", names[i]);
            nvdr_image_free(&f);
            break;
        }
        total += bytes;
        if (kind == NVDRV_INTRA) { intra_count++; intra_total += bytes; }
        printf("  %5d  %-6s %8zu  %+3d%+3d  %10zu\n",
               i, kind == NVDRV_INTRA ? "INTRA" : "pred", bytes, dx, dy, total);
        nvdr_image_free(&f);
    }

    int close_rc = nvdrv_encode_close(enc);
    free_frames(names, n);
    if (close_rc != 0) { fprintf(stderr, "close failed\n"); return 1; }

    printf("\n  %zu bytes total, %d intra (%zu bytes), %d preditos (%zu bytes)\n",
           total, intra_count, intra_total, n - intra_count, total - intra_total - NVDRV_HEADER_SIZE);
    if (n > intra_count && intra_count > 0)
        printf("  quadro predito medio %zu B contra intra medio %zu B\n",
               (total - intra_total - NVDRV_HEADER_SIZE) / (size_t)(n - intra_count),
               intra_total / (size_t)intra_count);
    return 0;
}
