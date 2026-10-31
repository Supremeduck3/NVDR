/*
 * nvdrv_encode — build a sequence container from a directory of frames,
 * and report what each frame cost and how it was coded.
 *
 * The per-frame report is the point, the same way it is for stills. A
 * predicted frame that keeps coming out the size of an intra one means the
 * prediction is not working, and that has to be visible rather than
 * averaged away into a bitrate.
 *
 * A sequence longer than one GOP is coded in stretches, one per GOP, as
 * many at once as there are cores (`--jobs`), and the stretches joined
 * into the file a single pass writes, to the byte (STRETCHES in nvdrv.c).
 * Most of the encoder runs on one core, so on a clip of several GOPs this
 * is close to that many times faster.
 */
#include "nvdrv.h"

#ifdef _OPENMP
#include <omp.h>
#endif
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
        "  --search N       global motion search half-width in px (default 24; 0 disables)\n"
        "  --intra-thresh F mean absolute error above which a frame goes intra (default 24)\n"
        "  --block N        per-block motion with NxN blocks (default 16 from 0.2 Mpx\n"
        "                   up, 8 below; 0 = one global vector)\n"
        "  --fps N          recorded in the header (default 24)\n"
        "  --q N            quantiser step for every frame (default 24)\n"
        "  --pred-q N       quantiser step for P frames (default 1.4 x q with B frames,\n"
        "                   1.2 x q without)\n"
        "  --skip-k F       how readily a predicted frame leaves a block as it was\n"
        "                   (default 0.25; 0 corrects every block)\n"
        "  --mv-lambda N    weight of one motion-field bit against block error\n"
        "                   (default 24 with blocks of 32 and up, 16 below;\n"
        "                   0 keeps each block's best-matching vector)\n"
        "  --bframes N      B frames between anchors (default 7; 0 = P frames only)\n"
        "  --b-q-step F     each level of B frames is 1 + F times coarser than P (default 0.5)\n"
        "  --lookahead N    frames looked ahead of each intra frame (default 16)\n"
        "  --tpl F          how much their reuse refines the intra frame (default 1; 0 = off)\n"
        "  --refs N         anchors a P frame may predict from, 1 or 2 (default 2)\n"
        "  --fast           try the other transform types only on the two cheapest\n"
        "                   modes of a leaf: much faster, slightly bigger\n"
        "  --tf N           source frames each side averaged into each anchor (default 7; 0 = off)\n"
        "  --tf-strength F  how far a match may differ, in units of the noise (default 4)\n"
        "  --no-restore     send no restoration filter with any frame\n"
        "  --jobs N         GOPs coded at once (default: one per core; 1 = in one pass)\n"
        "  --limit N        stop after N frames\n", a0);
}

typedef struct { size_t total, intra_total; int intra_count, count; } Totals;

static const char* kind_name(int kind) {
    return kind == NVDRV_INTRA ? "INTRA" : kind == NVDRV_PRED ? "pred" : kind == NVDRV_PRED2 ? "pred2" : "bi";
}

/* Frames are reported in coding order. */
static void print_report(Totals* t, const NvdrvFrameReport* r) {
    t->total += r->bytes;
    t->count++;
    if (r->kind == NVDRV_INTRA) { t->intra_count++; t->intra_total += r->bytes; }
    char kind[16];
    if (r->kind == NVDRV_BI) snprintf(kind, sizeof(kind), "bi%d", r->level);
    else snprintf(kind, sizeof(kind), "%s", kind_name(r->kind));
    printf("  %5d  %-6s %8zu %7zu  %+3d%+3d  q%-4d %10zu\n",
           r->display, kind, r->bytes, r->field_bytes, r->dx, r->dy, r->q, t->total);
    fflush(stdout);
}

/* A stretch's reports: printed as they come when it is the only one,
 * kept to be printed in order otherwise. */
typedef struct {
    Totals* live;
    NvdrvFrameReport* r;
    int n, cap;
} Reports;

static void report(void* user, const NvdrvFrameReport* r) {
    Reports* s = (Reports*)user;
    if (s->n == s->cap) {
        int cap = s->cap ? s->cap * 2 : 64;
        NvdrvFrameReport* g = (NvdrvFrameReport*)realloc(s->r, (size_t)cap * sizeof(*g));
        if (!g) return;
        s->r = g; s->cap = cap;
    }
    s->r[s->n++] = *r;
    if (s->live) print_report(s->live, r);
}

/* Codes the stretch `cfg` names (the whole sequence when it names none)
 * of frames names[0..n) into `path`. On failure, the reason is in `err`. */
static int code_stretch(char** names, int n, const NvdrvConfig* cfg, int width, int height,
                        const char* path, Reports* reps, char* err, size_t errlen) {
    NvdrvEncoder* enc;
    if (nvdrv_encode_open(&enc, path, width, height, cfg) != 0) {
        snprintf(err, errlen, "cannot open '%s' for writing", path);
        return -1;
    }
    nvdrv_encode_set_report(enc, report, reps);
    int first, last, rc = 0;
    nvdrv_encode_span(enc, &first, &last);
    if (last < 0 || last > n - 1) last = n - 1;
    for (int i = first; i <= last && rc == 0; i++) {
        NvdrImage f;
        if (nvdr_image_load(&f, names[i]) != 0) {
            snprintf(err, errlen, "cannot read %s", names[i]);
            rc = -1; break;
        }
        if (f.width != width || f.height != height) {
            snprintf(err, errlen, "%s is %dx%d, expected %dx%d", names[i], f.width, f.height, width, height);
            rc = -1;
        } else if (nvdrv_encode_frame(enc, &f) != 0) {
            snprintf(err, errlen, "encode failed on %s", names[i]);
            rc = -1;
        }
        nvdr_image_free(&f);
    }
    if (nvdrv_encode_close(enc) != 0 && rc == 0) {
        snprintf(err, errlen, "close failed");
        rc = -1;
    }
    return rc;
}

/* The varints of a frame header (COMPACT FRAMES in nvdrv.c). */
static int put_varint(uint8_t* p, uint32_t v) {
    int n = 0;
    while (v >= 0x80) { p[n++] = (uint8_t)(v | 0x80); v >>= 7; }
    p[n++] = (uint8_t)v;
    return n;
}

static int varint_len(const uint8_t* p, size_t avail) {
    for (size_t i = 0; i < 5 && i < avail; i++) if (!(p[i] & 0x80)) return (int)i + 1;
    return -1;
}

static uint32_t zigzag(int v) { return v >= 0 ? 2u * (uint32_t)v : 2u * (uint32_t)(-(long)v) - 1u; }

/*
 * The stretches' files into one: the first whole, the others after their
 * header. Each stretch numbers its first frame from 0, so that number is
 * rewritten as the change from the last frame of the stretch before
 * (STRETCHES in nvdrv.c); then the frame count goes in the header.
 */
static int join(const char* out, char** parts, Reports* reps, int count) {
    FILE* o = fopen(out, "wb");
    if (!o) return -1;
    int rc = 0, last = 0;
    uint32_t frames = 0;
    for (int k = 0; k < count && rc == 0; k++) {
        FILE* f = fopen(parts[k], "rb");
        if (!f) { rc = -1; break; }
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        uint8_t* data = size >= NVDRV_HEADER_SIZE ? (uint8_t*)malloc((size_t)size) : NULL;
        if (!data || fread(data, 1, (size_t)size, f) != (size_t)size) rc = -1;
        fclose(f);
        if (rc == 0 && k == 0) {
            if (fwrite(data, 1, (size_t)size, o) != (size_t)size) rc = -1;
        } else if (rc == 0 && reps[k].n > 0) {
            size_t at = NVDRV_HEADER_SIZE + 1;
            int old = varint_len(data + at, (size_t)size - at);
            uint8_t d[5];
            int now = put_varint(d, zigzag(reps[k].r[0].display - last));
            if (old < 0 ||
                fwrite(data + NVDRV_HEADER_SIZE, 1, 1, o) != 1 ||
                fwrite(d, 1, (size_t)now, o) != (size_t)now ||
                fwrite(data + at + old, 1, (size_t)size - at - (size_t)old, o) != (size_t)size - at - (size_t)old)
                rc = -1;
            reps[k].r[0].bytes += (size_t)now - (size_t)old;
        }
        free(data);
        if (reps[k].n > 0) last = reps[k].r[reps[k].n - 1].display;
        frames += (uint32_t)reps[k].n;
    }
    uint8_t c[4] = { (uint8_t)frames, (uint8_t)(frames >> 8), (uint8_t)(frames >> 16), (uint8_t)(frames >> 24) };
    if (rc == 0 && (fseek(o, 10, SEEK_SET) != 0 || fwrite(c, 1, 4, o) != 4)) rc = -1;
    if (fclose(o) != 0) rc = -1;
    return rc;
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 2; }
    const char* dir = argv[1];
    const char* out = argv[2];
    NvdrvConfig cfg = nvdrv_default_config();
    int limit = 0, jobs = 0;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--gop") && i+1 < argc) cfg.gop = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--search") && i+1 < argc) cfg.search = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--block") && i+1 < argc) cfg.block = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--intra-thresh") && i+1 < argc) cfg.intra_threshold = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--fps") && i+1 < argc) cfg.fps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--q") && i+1 < argc) cfg.frame.q = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pred-q") && i+1 < argc) cfg.pred_q = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--skip-k") && i+1 < argc) cfg.frame.skip_k = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--mv-lambda") && i+1 < argc) cfg.mv_lambda = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bframes") && i+1 < argc) cfg.bframes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--b-q-step") && i+1 < argc) cfg.b_q_step = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--lookahead") && i+1 < argc) cfg.lookahead = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tpl") && i+1 < argc) cfg.tpl_strength = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--fast")) cfg.frame.tx_modes = 2;
        else if (!strcmp(argv[i], "--refs") && i+1 < argc) cfg.refs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tf") && i+1 < argc) cfg.tf_radius = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tf-strength") && i+1 < argc) cfg.tf_strength = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--tf-levels") && i+1 < argc) cfg.tf_levels = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--limit") && i+1 < argc) limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--jobs") && i+1 < argc) jobs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-restore")) cfg.frame.restore = 0;
    }
#ifdef _OPENMP
    if (jobs <= 0) jobs = omp_get_max_threads();
#else
    jobs = 1;
#endif

    int n = 0;
    char** names = list_frames(dir, &n);
    if (!names || n == 0) { fprintf(stderr, "no frames in '%s'\n", dir); return 1; }
    if (limit > 0 && limit < n) n = limit;

    NvdrImage first;
    if (nvdr_image_load(&first, names[0]) != 0) {
        fprintf(stderr, "cannot read %s\n", names[0]);
        free_frames(names, n);
        return 1;
    }
    int width = first.width, height = first.height;
    nvdr_image_free(&first);

    /* A stretch from each intra frame the GOP places, but for one on the
     * last frame, which the stretch before it ends with. */
    int count = 1;
    if (cfg.gop > 0 && jobs > 1)
        while ((long)count * cfg.gop < n - 1) count++;

    printf("%d frames  %dx%d  gop %d  search %d  block %d  bframes %d\n", n, width, height,
           cfg.gop, cfg.search, cfg.block, cfg.bframes);
    if (count > 1) printf("  %d trechos de um GOP, %d ao mesmo tempo\n", count, jobs < count ? jobs : count);
    printf("  frame  tipo      bytes   campo    mv     q         acumulado\n");

    Totals t = { NVDRV_HEADER_SIZE, 0, 0, 0 };
    Reports* reps = (Reports*)calloc((size_t)count, sizeof(Reports));
    char** parts = (char**)calloc((size_t)count, sizeof(char*));
    char (*errs)[256] = (char (*)[256])calloc((size_t)count, 256);
    int failed = 0;
    if (!reps || !parts || !errs) { fprintf(stderr, "out of memory\n"); return 1; }

    if (count == 1) {
        reps[0].live = &t;
        failed = code_stretch(names, n, &cfg, width, height, out, &reps[0], errs[0], sizeof(errs[0])) != 0;
    } else {
        for (int k = 0; k < count && !failed; k++) {
            size_t len = strlen(out) + 16;
            if (!(parts[k] = (char*)malloc(len))) failed = 1;
            else snprintf(parts[k], len, "%s.part%d", out, k);
        }
        /* The cores left over when there are fewer stretches than cores go
         * to the encoder's own threads, inside each stretch. */
        int outer = jobs < count ? jobs : count, inner = jobs / outer;
#ifdef _OPENMP
        if (inner > 1) omp_set_max_active_levels(2);
#endif
        if (!failed) {
            #pragma omp parallel for schedule(dynamic, 1) num_threads(outer)
            for (int k = 0; k < count; k++) {
#ifdef _OPENMP
                omp_set_num_threads(inner);
#endif
                NvdrvConfig c = cfg;
                c.from = k * cfg.gop;
                c.to = k + 1 < count ? (k + 1) * cfg.gop : 0;
                if (code_stretch(names, n, &c, width, height, parts[k], &reps[k], errs[k], sizeof(errs[k])) != 0) {
                    #pragma omp atomic write
                    failed = 1;
                }
            }
        }
        (void)inner;
        if (!failed && join(out, parts, reps, count) != 0) {
            snprintf(errs[0], sizeof(errs[0]), "cannot write '%s'", out);
            failed = 1;
        }
        if (!failed)
            for (int k = 0; k < count; k++)
                for (int i = 0; i < reps[k].n; i++) print_report(&t, &reps[k].r[i]);
        for (int k = 0; k < count; k++) if (parts[k]) { remove(parts[k]); free(parts[k]); }
    }
    free_frames(names, n);
    for (int k = 0; k < count; k++) {
        if (errs[k][0]) fprintf(stderr, "%s\n", errs[k]);
        free(reps[k].r);
    }
    free(reps); free(parts); free(errs);
    if (failed) return 1;

    size_t inter = t.total - t.intra_total - NVDRV_HEADER_SIZE;
    int ninter = t.count - t.intra_count;
    printf("\n  %zu bytes total, %d intra (%zu bytes), %d preditos (%zu bytes)\n",
           t.total, t.intra_count, t.intra_total, ninter, inter);
    if (ninter > 0 && t.intra_count > 0)
        printf("  quadro predito medio %zu B contra intra medio %zu B\n",
               inter / (size_t)ninter, t.intra_total / (size_t)t.intra_count);
    return 0;
}
