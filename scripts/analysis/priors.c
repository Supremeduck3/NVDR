/*
 * priors — the models a slice starts from (src/priors.h).
 *
 *   priors <out.h> <image>...
 *
 * A slice cannot start from what the slice before it learnt, since it has
 * to decode without it, and from even odds every slice pays again for
 * the slow start. So it starts from what pictures in general teach: each
 * image here is coded whole, progressive and directional, at q 16, 24
 * and 40, and the probability every context ends at is averaged over the
 * runs that used it, to 8 bits. A context no run used stays at even
 * odds. The order is the structs': ColourModels, then the low band's
 * TextureModels, then the high band's.
 *
 * Built by hand, as the tables are committed:
 *   gcc -O2 -Isrc -Ivendor -fopenmp -o output/priors scripts/analysis/priors.c \
 *       src/entropy.c src/grain.c src/restore.c -lm -lz
 */
#include "../../src/nvdr.c"

#define NP (sizeof(ColourModels) / sizeof(NvdrProb) + 2 * sizeof(TextureModels) / sizeof(NvdrProb))

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <out.h> <image>...\n", argv[0]); return 2; }
    static double sum[2][NP];
    static int used[2][NP];
    const int qs[3] = { 16, 24, 40 };
    for (int i = 2; i < argc; i++) {
        NvdrImage img;
        if (nvdr_image_load(&img, argv[i]) != 0) { fprintf(stderr, "skip %s\n", argv[i]); continue; }
        for (int mode = 0; mode < 2; mode++)
            for (int k = 0; k < 3; k++) {
                NvdrConfig c = nvdr_default_config();
                c.q = qs[k];
                c.directional = mode;
                c.restore = 0;
                if (mode) { c.tx_modes = 2; c.fast_rank = 1; }
                NvdrContext* ctx = nvdr_context_new();
                uint8_t* buf; size_t len;
                if (ctx && nvdr_encode_mem_ctx(&buf, &len, &img, &c, NULL, ctx) == 0) {
                    free(buf);
                    NvdrProb all[NP];
                    size_t nc = sizeof(ColourModels) / sizeof(NvdrProb), nt = sizeof(TextureModels) / sizeof(NvdrProb);
                    memcpy(all, &ctx->cm, sizeof(ColourModels));
                    memcpy(all + nc, &ctx->tm, sizeof(TextureModels));
                    memcpy(all + nc + nt, &ctx->tm2, sizeof(TextureModels));
                    for (size_t j = 0; j < NP; j++)
                        if (all[j] >> 16) { sum[mode][j] += (double)(all[j] & 0xffff) / (1 << NVDR_STATE_BITS); used[mode][j]++; }
                }
                nvdr_context_free(ctx);
            }
        fprintf(stderr, "%s\n", argv[i]);
        nvdr_image_free(&img);
    }
    FILE* h = fopen(argv[1], "w");
    if (!h) return 1;
    fprintf(h, "/* The models a slice starts from: see scripts/analysis/priors.c, which\n"
               " * wrote this. [0] progressive, [1] directional; the probability of a 0,\n"
               " * in 256ths, in the order of ColourModels, the low band's TextureModels\n"
               " * and the high band's. */\n"
               "#define NVDR_PRIOR_COUNT %zu\nstatic const uint8_t NVDR_PRIORS[2][NVDR_PRIOR_COUNT] = {\n", (size_t)NP);
    for (int mode = 0; mode < 2; mode++) {
        fprintf(h, "    {");
        for (size_t j = 0; j < NP; j++) {
            int v = used[mode][j] ? (int)(sum[mode][j] / used[mode][j] * 256 + 0.5) : 128;
            v = v < 1 ? 1 : v > 255 ? 255 : v;
            fprintf(h, "%s%d,", j % 24 ? "" : "\n        ", v);
        }
        fprintf(h, "\n    },\n");
    }
    fprintf(h, "};\n");
    fclose(h);
    return 0;
}
