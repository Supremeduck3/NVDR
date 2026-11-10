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
 *
 * The same stretches can be coded on other machines (worker.js, with
 * server.js sharing them out): --plan lists them, and --from/--to codes
 * one from a folder holding just the frames it reads, ending with a line
 * that says what to join and how to check the joins.
 *
 * --kbps aims at a bitrate instead of a quantiser: the same step for the
 * whole sequence (constant quality, the bytes going where the pictures
 * need them), chosen so the file comes to that average rate. Two probes
 * on a sample of the stretches, coded the fast way, give the step a
 * trial and the slope of bytes against it; the sequence is then coded
 * once, and once more only if it missed by more than a step's worth.
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
        "       %s <frames-dir> --plan [options]\n"
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
        "                   modes of a leaf, ranked without RDOQ: much faster,\n"
        "                   slightly bigger\n"
        "  --tf N           source frames each side averaged into each anchor (default 7; 0 = off)\n"
        "  --tf-strength F  how far a match may differ, in units of the noise (default 4)\n"
        "  --no-restore     send no restoration filter with any frame\n"
        "  --audio FILE     carry FILE after the frames as the sound track: a file a\n"
        "                   browser plays (.webm, .ogg, .m4a, .mp3...), as long as the frames\n"
        "  --kbps N         aim at an average of N kbit/s at --fps instead of a fixed --q\n"
        "  --jobs N         GOPs coded at once (default: one per core; 1 = in one pass)\n"
        "  --limit N        stop after N frames\n"
        "  --plan           list the stretches (from, to, first and last frame read) and stop\n"
        "  --from F --to T  code only the stretch from intra frame F to intra frame T\n"
        "                   (0: the end), from a folder holding the frames it reads\n"
        "  --version        the sequence format this encoder writes\n", a0, a0);
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
    NvdrvFrameReport* r;     /* the frames written */
    int n, cap;
    uint32_t start;          /* check of the intra frame it starts with, unwritten */
} Reports;

static void report(void* user, const NvdrvFrameReport* r) {
    Reports* s = (Reports*)user;
    if (!r->written) { s->start = r->check; return; }
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
 * into `path`, frame d of the sequence being names[d - offset], up to
 * names[n - 1]. On failure, the reason is in `err`. */
static int code_stretch(char** names, int n, int offset, const NvdrvConfig* cfg, int width, int height,
                        const char* path, Reports* reps, char* err, size_t errlen) {
    int first, last, rc = 0;
    if (nvdrv_stretch_span(cfg, width, height, &first, &last) != 0) {
        snprintf(err, errlen, "not a stretch: --from and --to are intra frames of the GOP, in order");
        return -1;
    }
    if (last < 0 || last > offset + n - 1) last = offset + n - 1;
    if (first < offset) {
        snprintf(err, errlen, "the stretch reads from frame %d", first);
        return -1;
    }
    NvdrvEncoder* enc;
    if (nvdrv_encode_open(&enc, path, width, height, cfg) != 0) {
        snprintf(err, errlen, "cannot open '%s' for writing", path);
        return -1;
    }
    nvdrv_encode_set_report(enc, report, reps);
    for (int d = first; d <= last && rc == 0; d++) {
        const char* name = names[d - offset];
        NvdrImage f;
        if (nvdr_image_load(&f, name) != 0) {
            snprintf(err, errlen, "cannot read %s", name);
            rc = -1; break;
        }
        if (f.width != width || f.height != height) {
            snprintf(err, errlen, "%s is %dx%d, expected %dx%d", name, f.width, f.height, width, height);
            rc = -1;
        } else if (nvdrv_encode_frame(enc, &f) != 0) {
            snprintf(err, errlen, "encode failed on %s", name);
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

/*
 * The sound track `path` after the frames of `out` (SOUND in nvdrv.h): a
 * file a browser plays, its type told by its name.
 */
static int append_sound(const char* out, const char* path) {
    static const char* types[][2] = {
        { ".webm", "audio/webm" }, { ".ogg", "audio/ogg" }, { ".opus", "audio/ogg" }, { ".m4a", "audio/mp4" },
        { ".mp4", "audio/mp4" }, { ".aac", "audio/aac" }, { ".mp3", "audio/mpeg" }, { ".wav", "audio/wav" },
    };
    const char* dot = strrchr(path, '.');
    const char* mime = NULL;
    for (size_t i = 0; dot && i < sizeof(types) / sizeof(types[0]); i++)
        if (!strcasecmp(dot, types[i][0])) mime = types[i][1];
    if (!mime) { fprintf(stderr, "--audio: a .webm, .ogg, .opus, .m4a, .mp4, .aac, .mp3 or .wav file\n"); return -1; }
    FILE* a = fopen(path, "rb");
    if (!a) { fprintf(stderr, "cannot read %s\n", path); return -1; }
    fseek(a, 0, SEEK_END);
    long size = ftell(a);
    fseek(a, 0, SEEK_SET);
    uint8_t* data = size > 0 ? (uint8_t*)malloc((size_t)size) : NULL;
    int ok = data && fread(data, 1, (size_t)size, a) == (size_t)size;
    fclose(a);
    FILE* o = ok ? fopen(out, "ab") : NULL;
    if (o) {
        uint8_t head[6] = { NVDRV_SOUND_MARK, 'N', 'V', 'A', 'U', (uint8_t)strlen(mime) };
        uint8_t len[4] = { (uint8_t)size, (uint8_t)(size >> 8), (uint8_t)(size >> 16), (uint8_t)(size >> 24) };
        ok = fwrite(head, 1, 6, o) == 6 && fwrite(mime, 1, strlen(mime), o) == strlen(mime) &&
             fwrite(len, 1, 4, o) == 4 && fwrite(data, 1, (size_t)size, o) == (size_t)size;
        if (fclose(o) != 0) ok = 0;
    } else ok = 0;
    free(data);
    if (!ok) fprintf(stderr, "cannot add %s to %s\n", path, out);
    return ok ? 0 : -1;
}

/*
 * RATE
 * ----
 * What the whole sequence would cost at cfg's step, in bytes, from the
 * stretches k = 0, stride, 2 * stride...: a probe for --kbps. Intra
 * frames are most of the bytes and a sample holds more or fewer of them
 * than its share (the first stretch has two), so the two kinds are
 * averaged apart and each scaled to how many the sequence has. Nothing
 * is kept.
 */
#include <math.h>
static int probe(char** names, int n, const NvdrvConfig* cfg, int width, int height, const char* out,
                 int count, int jobs, int stride, double* bytes) {
    int failed = 0;
    double bi = 0, bp = 0;
    int ni = 0, np = 0;
    #pragma omp parallel for schedule(dynamic, 1) num_threads(jobs < count ? jobs : count) reduction(+:bi, bp, ni, np)
    for (int k = 0; k < count; k += stride) {
        NvdrvConfig c = *cfg;
        if (count > 1) { c.from = k * cfg->gop; c.to = k + 1 < count ? (k + 1) * cfg->gop : 0; }
        char path[4096], err[256];
        snprintf(path, sizeof(path), "%s.probe%d", out, k);
        Reports r = { NULL, NULL, 0, 0, 0 };
        if (code_stretch(names, n, 0, &c, width, height, path, &r, err, sizeof(err)) != 0) {
            #pragma omp atomic write
            failed = 1;
        }
        for (int i = 0; i < r.n; i++) {
            if (r.r[i].kind == NVDRV_INTRA) { bi += (double)r.r[i].bytes; ni++; }
            else { bp += (double)r.r[i].bytes; np++; }
        }
        free(r.r);
        remove(path);
    }
    if (failed || ni + np == 0) return -1;
    /* The intra frames the GOP places; a cut the encoder finds is one more. */
    int intras = cfg->gop > 0 ? (n - 1) / cfg->gop + 1 : 1;
    if (intras > n) intras = n;
    *bytes = (ni ? bi / ni * intras : 0) + (np ? bp / np * (n - intras) : 0);
    return 0;
}

/* The step that should bring `n` frames to `target` bytes: see the top of
 * the file. `slope` takes the exponent found, for the correction after. */
static int step_for_rate(char** names, int n, const NvdrvConfig* cfg, int width, int height, const char* out,
                         int count, int jobs, double target, double* slope) {
    NvdrvConfig c = *cfg;
    c.frame.tx_modes = 2; c.frame.fast_rank = 1;   /* --fast: a probe is an estimate */
    /* About half the sequence, in stretches spread along it. */
    int stride = count >= 4 ? 2 : 1;
    double b0, b1;
    int q0 = cfg->frame.q;
    if (probe(names, n, &c, width, height, out, count, jobs, stride, &b0) != 0) return -1;
    /* Bytes go roughly as 1 / step: the second probe where that says. */
    int q1 = (int)(q0 * b0 / target + 0.5);
    q1 = q1 < 4 ? 4 : q1 > 200 ? 200 : q1;
    if (q1 == q0) q1 = b0 > target ? q0 + 1 : q0 - 1;
    c.frame.q = q1;
    if (probe(names, n, &c, width, height, out, count, jobs, stride, &b1) != 0) return -1;
    /* The slope the two probes show, where they are far enough apart to
     * show one: between neighbouring steps the difference is mostly the
     * rounding of the predicted frames' steps, and bytes go as 1 / step
     * to within a few percent (-0.95 on the clip this was tried on). */
    double e = -1.0;
    if (fabs(log((double)q1 / q0)) > 0.25) {
        e = log(b1 / b0) / log((double)q1 / q0);
        if (!(e < -0.7)) e = -0.7;
        if (e < -1.6) e = -1.6;
    }
    *slope = e;
    int q = (int)(q1 * pow(target / b1, 1.0 / e) + 0.5);
    return q < 4 ? 4 : q > 200 ? 200 : q;
}

int main(int argc, char** argv) {
    if (argc == 2 && !strcmp(argv[1], "--version")) {
        printf("nvdrv_encode, sequence format %d\n", NVDRV_VERSION);
        return 0;
    }
    int plan = 0;
    for (int i = 2; i < argc; i++) if (!strcmp(argv[i], "--plan")) plan = 1;
    if (argc < 3) { usage(argv[0]); return 2; }
    const char* dir = argv[1];
    const char* out = argv[2];
    NvdrvConfig cfg = nvdrv_default_config();
    int limit = 0, jobs = 0, stretch = 0;
    double kbps = 0;
    const char* audio = NULL;

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
        else if (!strcmp(argv[i], "--fast")) { cfg.frame.tx_modes = 2; cfg.frame.fast_rank = 1; }
        else if (!strcmp(argv[i], "--refs") && i+1 < argc) cfg.refs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tf") && i+1 < argc) cfg.tf_radius = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tf-strength") && i+1 < argc) cfg.tf_strength = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--tf-levels") && i+1 < argc) cfg.tf_levels = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--limit") && i+1 < argc) limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--jobs") && i+1 < argc) jobs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kbps") && i+1 < argc) kbps = atof(argv[++i]);
        else if (!strcmp(argv[i], "--audio") && i+1 < argc) audio = argv[++i];
        else if (!strcmp(argv[i], "--from") && i+1 < argc) { cfg.from = atoi(argv[++i]); stretch = 1; }
        else if (!strcmp(argv[i], "--to") && i+1 < argc) { cfg.to = atoi(argv[++i]); stretch = 1; }
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
    if (cfg.gop > 0 && (jobs > 1 || plan || kbps > 0))
        while ((long)count * cfg.gop < n - 1) count++;
    if (audio && (plan || stretch)) { fprintf(stderr, "--audio goes with a whole sequence\n"); return 1; }
    if (kbps > 0 && (plan || stretch || cfg.fps <= 0)) {
        fprintf(stderr, "--kbps codes a whole sequence, at a known --fps\n");
        return 1;
    }

    if (plan) {
        for (int k = 0; k < count; k++) {
            NvdrvConfig c = cfg;
            c.from = k * cfg.gop;
            c.to = k + 1 < count ? (k + 1) * cfg.gop : 0;
            int first, last;
            if (nvdrv_stretch_span(&c, width, height, &first, &last) != 0) return 1;
            printf("trecho %d %d %d %d\n", c.from, c.to, first, last < 0 || last > n - 1 ? n - 1 : last);
        }
        free_frames(names, n);
        return 0;
    }
    if (stretch) {
        /* One stretch, from a folder whose first frame is the first it reads. */
        int first, last;
        if (nvdrv_stretch_span(&cfg, width, height, &first, &last) != 0) {
            fprintf(stderr, "not a stretch: --from and --to are intra frames of the GOP, in order\n");
            return 1;
        }
        /* Fewer frames than it looks ahead to is the end of the sequence,
         * which a single pass meets the same way; fewer than its last intra
         * frame is a folder short of frames. */
        if (cfg.to > 0 && n < cfg.to - first + 1) {
            fprintf(stderr, "the stretch ends at frame %d, the folder holds %d to %d\n", cfg.to, first, first + n - 1);
            return 1;
        }
        Totals t = { NVDRV_HEADER_SIZE, 0, 0, 0 };
        Reports r = { &t, NULL, 0, 0, 0 };
        char err[256] = "";
        int rc = code_stretch(names, n, first, &cfg, width, height, out, &r, err, sizeof(err));
        free_frames(names, n);
        if (rc != 0) { fprintf(stderr, "%s\n", err); free(r.r); return 1; }
        /* What the coordinator joins: the first and last display numbers
         * written, how many frames, and the checks of the intra frames at
         * either end, which must match the next and previous stretches'. */
        uint32_t end = 0;
        for (int i = 0; i < r.n; i++) if (cfg.to > 0 && r.r[i].display == cfg.to) end = r.r[i].check;
        printf("trecho %d %d %d %08x %08x\n", r.n ? r.r[0].display : 0, r.n ? r.r[r.n - 1].display : 0,
               r.n, (unsigned)r.start, (unsigned)end);
        free(r.r);
        return 0;
    }

    double target = kbps * 125.0 * n / (cfg.fps > 0 ? cfg.fps : 1), slope = -1.0;
    int tries = 0;
    if (kbps > 0) {
        int q = step_for_rate(names, n, &cfg, width, height, out, count, jobs, target, &slope);
        if (q < 0) { fprintf(stderr, "could not probe the sequence for --kbps\n"); return 1; }
        cfg.frame.q = q;
    }
again:
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
        failed = code_stretch(names, n, 0, &cfg, width, height, out, &reps[0], errs[0], sizeof(errs[0])) != 0;
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
                if (code_stretch(names, n, 0, &c, width, height, parts[k], &reps[k], errs[k], sizeof(errs[k])) != 0) {
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
    for (int k = 0; k < count; k++) {
        if (errs[k][0]) fprintf(stderr, "%s\n", errs[k]);
        free(reps[k].r);
    }
    free(reps); free(parts); free(errs);
    if (failed) { free_frames(names, n); return 1; }
    if (kbps > 0) {
        /* A step is 4 to 5% of the bytes near the default: past that, the
         * probes misjudged, and what was just coded says by how much. */
        double got = (double)t.total;
        int q = (int)(cfg.frame.q * pow(target / got, 1.0 / slope) + 0.5);
        q = q < 4 ? 4 : q > 200 ? 200 : q;
        printf("\n  taxa: %.0f kbit/s com q %d (alvo %.0f)\n", got / 125.0 * cfg.fps / n, cfg.frame.q, kbps);
        if (fabs(got / target - 1.0) > 0.06 && q != cfg.frame.q && tries++ < 1) {
            printf("  fora do alvo: de novo com q %d\n\n", q);
            cfg.frame.q = q;
            goto again;
        }
    }
    free_frames(names, n);
    if (audio && append_sound(out, audio) != 0) return 1;

    size_t inter = t.total - t.intra_total - NVDRV_HEADER_SIZE;
    int ninter = t.count - t.intra_count;
    printf("\n  %zu bytes total, %d intra (%zu bytes), %d preditos (%zu bytes)\n",
           t.total, t.intra_count, t.intra_total, ninter, inter);
    if (ninter > 0 && t.intra_count > 0)
        printf("  quadro predito medio %zu B contra intra medio %zu B\n",
               inter / (size_t)ninter, t.intra_total / (size_t)t.intra_count);
    return 0;
}
