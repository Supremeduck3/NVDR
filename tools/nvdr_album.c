/*
 * nvdr_album — many still images in one .nvda file, each coded with what
 * the codec learned on the ones before it (see src/nvda.h).
 *
 *   nvdr_album pack   <out.nvda> <images...> [--q N] [--band N] [--window N] [--distance D] [--fluid] [--no-predict]
 *   nvdr_album unpack <in.nvda> <out-dir>    [--compare <dir>] [--only N]
 *
 * pack reports every image's bytes against what it costs coded alone, and
 * whether it was coded alone or predicted from the one before, so what
 * the album is worth is printed rather than assumed.
 * unpack writes each image as <out-dir>/<name>.png; --compare looks for a
 * file of the same name in <dir> and reports PSNR.
 */
#include "nvda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char* a0) {
    fprintf(stderr,
        "usage: %s pack <out.nvda> <images...> [--q N] [--band N] [--window N] [--distance D] [--fluid] [--no-predict]\n"
        "       %s unpack <in.nvda> <out-dir> [--compare <dir>] [--only N]\n"
        "  --window N     how many earlier photos a photo may be predicted from (default 8)\n"
        "  --distance D   skip earlier photos whose thumbnails differ by more than D\n"
        "                 grey levels (default 10; larger tries more, slower)\n"
        "  --fluid        carry the fluid context between photos coded alone (the album\n"
        "                 must then be read in order)\n"
        "  --no-predict   never predict a photo from another\n"
        "  --only N       decode photo N alone, and only what it depends on\n", a0, a0);
}

static const char* base_name(const char* p) {
    const char* s = strrchr(p, '/');
    const char* b = strrchr(p, '\\');
    if (b && (!s || b > s)) s = b;
    return s ? s + 1 : p;
}

/* The stored name without its extension, with anything that could climb
 * out of the output directory replaced. */
static void safe_stem(const char* name, char* out, size_t cap) {
    size_t k = 0;
    for (const char* p = name; *p && k + 1 < cap; p++) {
        char c = *p;
        if (c == '/' || c == '\\' || c == ':' || (unsigned char)c < 32) c = '_';
        out[k++] = c;
    }
    out[k] = 0;
    char* dot = strrchr(out, '.');
    if (dot && dot != out) *dot = 0;
    if (!out[0] || !strcmp(out, ".") || !strcmp(out, "..")) snprintf(out, cap, "imagem");
}

static int pack(int argc, char** argv) {
    NvdrConfig cfg = nvdr_default_config();
    NvdaOptions opt = nvda_default_options();
    const char* out = argv[2];
    const char** paths = (const char**)calloc((size_t)argc, sizeof(char*));
    int n = 0;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--q") && i + 1 < argc) cfg.q = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--band") && i + 1 < argc) cfg.band = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--window") && i + 1 < argc) opt.window = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--distance") && i + 1 < argc) opt.max_distance = atof(argv[++i]);
        else if (!strcmp(argv[i], "--fluid")) opt.fluid = 1;
        else if (!strcmp(argv[i], "--no-predict")) opt.predict = 0;
        else paths[n++] = argv[i];
    }
    if (!n || n > NVDA_MAX_IMAGES) { usage(argv[0]); return 2; }
    NvdrImage* imgs = (NvdrImage*)calloc((size_t)n, sizeof(NvdrImage));
    const char** names = (const char**)calloc((size_t)n, sizeof(char*));
    NvdaReport* rep = (NvdaReport*)calloc((size_t)n, sizeof(NvdaReport));
    for (int i = 0; i < n; i++) {
        if (nvdr_image_load(&imgs[i], paths[i]) != 0 || imgs[i].width > 65535 || imgs[i].height > 65535) {
            fprintf(stderr, "cannot read image '%s'\n", paths[i]);
            return 1;
        }
        names[i] = base_name(paths[i]);
    }
    if (nvda_write(out, n, names, imgs, &cfg, &opt, rep) != 0) {
        fprintf(stderr, "cannot write '%s'\n", out);
        return 1;
    }
    /* Read the album back the way a viewer would, for the PSNR column. */
    FILE* f = fopen(out, "rb");
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* data = (uint8_t*)malloc((size_t)sz);
    size_t got = fread(data, 1, (size_t)sz, f);
    fclose(f);
    NvdaReader r;
    nvda_open(&r, data, got);
    printf("%s  %d imagens  janela %d%s%s\n", out, n, opt.window,
           opt.predict ? ", previsao entre imagens" : "", opt.fluid ? ", codebook fluido" : "");
    printf("  %-32s %-12s %9s %9s %8s %9s\n", "imagem", "tipo", "bytes", "sozinha", "ganho", "PSNR");
    size_t tb = 0, tc = 0;
    for (int i = 0; i < n; i++) {
        NvdrImage dec;
        double p = 0;
        if (nvda_decode(&r, i, &dec, NULL) == 1) { p = nvdr_psnr(&imgs[i], &dec); nvdr_image_free(&dec); }
        char kind[32];
        if (rep[i].kind == NVDA_KIND_PRED) snprintf(kind, sizeof kind, "prevista #%d", i - rep[i].ref + 1);
        else snprintf(kind, sizeof kind, "sozinha");
        printf("  %-32.32s %-12s %9zu %9zu %+7.2f%% %6.2f dB\n", names[i], kind, rep[i].bytes, rep[i].alone,
               rep[i].alone ? 100.0 * ((double)rep[i].bytes - rep[i].alone) / rep[i].alone : 0.0, p);
        tb += rep[i].bytes; tc += rep[i].alone;
        nvdr_image_free(&imgs[i]);
    }
    printf("  %-32s %-12s %9zu %9zu %+7.2f%%   arquivo %ld B\n", "total", "", tb, tc,
           tc ? 100.0 * ((double)tb - tc) / tc : 0.0, sz);
    nvda_close(&r);
    free(data); free(imgs); free(names); free(rep); free(paths);
    return 0;
}

static int unpack(int argc, char** argv) {
    const char* in = argv[2];
    const char* dir = argc > 3 ? argv[3] : NULL;
    const char* cmp = NULL;
    int only = -1;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--compare") && i + 1 < argc) cmp = argv[++i];
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only = atoi(argv[++i]) - 1;
    }
    if (!dir) { usage(argv[0]); return 2; }
    FILE* f = fopen(in, "rb");
    if (!f) { fprintf(stderr, "cannot open '%s'\n", in); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* data = (uint8_t*)malloc(sz > 0 ? (size_t)sz : 1);
    size_t got = fread(data, 1, (size_t)(sz > 0 ? sz : 0), f);
    fclose(f);
    NvdaReader r;
    if (nvda_open(&r, data, got) != 0) { fprintf(stderr, "'%s' is not an NVDA album\n", in); free(data); return 1; }
    printf("%s  %u imagens  janela %d%s\n", in, r.count, r.window,
           (r.flags & NVDA_FLAG_FLUID) ? "  codebook fluido" : "");
    int rc = 1, partial;
    char name[256], stem[256], path[1024];
    for (int i = only >= 0 ? only : 0; i < (int)r.count && (only < 0 || i == only); i++) {
        NvdrImage img;
        rc = nvda_decode(&r, i, &img, &partial);
        if (rc != 1) break;
        nvda_name(&r, i, name, sizeof name);
        safe_stem(name, stem, sizeof stem);
        snprintf(path, sizeof path, "%s/%03d_%s.png", dir, i, stem);
        nvdr_image_write(&img, path);
        const NvdaEntry* x = &r.index[i];
        char kind[32];
        if (x->kind == NVDA_KIND_PRED) snprintf(kind, sizeof kind, "prevista #%d", i - x->ref + 1);
        else snprintf(kind, sizeof kind, "sozinha");
        printf("  %3d  %-32.32s %-12s %dx%d%s", i + 1, name, kind, img.width, img.height,
               partial ? "  (parcial)" : "");
        if (cmp) {
            char src[1024]; NvdrImage s;
            snprintf(src, sizeof src, "%s/%s", cmp, name);
            if (nvdr_image_load(&s, src) == 0) { printf("  psnr %.2f dB", nvdr_psnr(&s, &img)); nvdr_image_free(&s); }
        }
        printf("\n");
        nvdr_image_free(&img);
        if (partial) { rc = 0; break; }
    }
    if (rc < 0) fprintf(stderr, "album is damaged\n");
    nvda_close(&r);
    free(data);
    return rc < 0 ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc >= 4 && !strcmp(argv[1], "pack")) return pack(argc, argv);
    if (argc >= 4 && !strcmp(argv[1], "unpack")) return unpack(argc, argv);
    usage(argv[0]);
    return 2;
}
