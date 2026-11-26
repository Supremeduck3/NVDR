/*
 * seqpriors — the models a sequence starts from (PRIORS in src/nvdrv.c),
 * as src/seqpriors.h.
 *
 *   NVDRV_DUMP=run.bin nvdrv_encode clip/ out.nvdrv --jobs 1 ...   (as many as wanted)
 *   seqpriors src/seqpriors.h run.bin...
 *
 * With NVDRV_DUMP set the encoder appends what each run of models ended
 * with: a class of predicted frame's context at the end of each group of
 * pictures (tag 0 to 3, frame_class() in nvdrv.c), and every motion
 * field's models (tag 4 for P frames, 5 for B). This averages each
 * probability over the runs that used it, as scripts/analysis/priors.c
 * does for stills. A class's context no run used takes the stills'
 * directional prior, a field's even odds. One file at a time per dump:
 * --jobs 1, or a dump per encode.
 *
 *   gcc -O2 -Isrc -Ivendor -o output/seqpriors scripts/analysis/seqpriors.c \
 *       src/entropy.c src/grain.c src/restore.c src/extras.c -lm -lz
 */
#include "../../src/nvdr.c"

#define CLASSES 4
#define NP (sizeof(ColourModels) / sizeof(NvdrProb) + 2 * sizeof(TextureModels) / sizeof(NvdrProb))
#define FN 64    /* NVDRV_FIELD_PRIOR_COUNT: nvdrv.c checks it against its FieldModels */

static int to256(double sum, int used, int fallback) {
    int v = used ? (int)(sum / used / (1 << NVDR_STATE_BITS) * 256 + 0.5) : fallback;
    return v < 1 ? 1 : v > 255 ? 255 : v;
}

static void row(FILE* h, const int* v, size_t n) {
    fprintf(h, "    {");
    for (size_t j = 0; j < n; j++) fprintf(h, "%s%d,", j % 24 ? "" : "\n        ", v[j]);
    fprintf(h, "\n    },\n");
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <out.h> <dump>...\n", argv[0]); return 2; }
    static double sum[CLASSES][NP], fsum[2][FN];
    static int used[CLASSES][NP], fused[2][FN], runs[CLASSES], fields[2];
    for (int i = 2; i < argc; i++) {
        FILE* f = fopen(argv[i], "rb");
        if (!f) { fprintf(stderr, "cannot read %s\n", argv[i]); return 1; }
        int tag;
        while (fread(&tag, sizeof(int), 1, f) == 1) {
            if (tag >= 0 && tag < CLASSES) {
                struct NvdrContext c;
                if (fread(&c, sizeof(c), 1, f) != 1) break;
                if (!c.valid) continue;
                NvdrProb all[NP];
                size_t nc = sizeof(ColourModels) / sizeof(NvdrProb), nt = sizeof(TextureModels) / sizeof(NvdrProb);
                memcpy(all, &c.cm, sizeof(ColourModels));
                memcpy(all + nc, &c.tm, sizeof(TextureModels));
                memcpy(all + nc + nt, &c.tm2, sizeof(TextureModels));
                /* A context counts once it has seen a symbol past what its
                 * prior was taken as. */
                for (size_t j = 0; j < NP; j++)
                    if ((all[j] >> 16) > 48 || ((all[j] >> 16) && (all[j] >> 16) < 48)) {
                        sum[tag][j] += all[j] & 0xffff; used[tag][j]++;
                    }
                runs[tag]++;
            } else if (tag == 4 || tag == 5) {
                NvdrProb m[FN];
                if (fread(m, sizeof(m), 1, f) != 1) break;
                for (int j = 0; j < FN; j++)
                    if ((m[j] >> 16) > 48 || ((m[j] >> 16) && (m[j] >> 16) < 48)) {
                        fsum[tag - 4][j] += m[j] & 0xffff; fused[tag - 4][j]++;
                    }
                fields[tag - 4]++;
            } else {
                fprintf(stderr, "%s: tag %d makes no sense\n", argv[i], tag); return 1;
            }
        }
        fclose(f);
    }
    FILE* h = fopen(argv[1], "w");
    if (!h) return 1;
    fprintf(h, "/* The models a sequence's frames and motion fields start from (PRIORS in\n"
               " * nvdrv.c): written by scripts/analysis/seqpriors.c. In 256ths. */\n"
               "#define NVDRV_CLASS_PRIOR_COUNT %zu\n"
               "static const uint8_t NVDRV_CLASS_PRIORS[%d][NVDRV_CLASS_PRIOR_COUNT] = {\n", (size_t)NP, CLASSES);
    static int v[NP];
    for (int k = 0; k < CLASSES; k++) {
        for (size_t j = 0; j < NP; j++) v[j] = to256(sum[k][j], used[k][j], NVDR_PRIORS[1][j]);
        row(h, v, NP);
        fprintf(stderr, "class %d: %d runs\n", k, runs[k]);
    }
    fprintf(h, "};\n#define NVDRV_FIELD_PRIOR_COUNT %d\n"
               "static const uint8_t NVDRV_FIELD_PRIORS[2][NVDRV_FIELD_PRIOR_COUNT] = {\n", FN);
    for (int k = 0; k < 2; k++) {
        for (int j = 0; j < FN; j++) v[j] = to256(fsum[k][j], fused[k][j], 128);
        row(h, v, FN);
        fprintf(stderr, "%s fields: %d\n", k ? "B" : "P", fields[k]);
    }
    fprintf(h, "};\n");
    fclose(h);
    return 0;
}
