/*
 * nvdr_encode — build a v10 container and report what each layer costs
 * and buys.
 *
 * The report is read back from the file the way a consumer would see it,
 * not taken from the encoder's own state: layer 0 alone, then both.
 */
#include "nvdr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char* argv0) {
    fprintf(stderr,
        "usage: %s <image> <out.nvdr> [options]\n"
        "  --q N            quantiser step, the quality knob (default 24;\n"
        "                   smaller is better and larger)\n"
        "  --chroma-q F     chroma step relative to luma (default 1.0)\n"
        "  --deadzone F     0..0.5, how readily small coefficients round to\n"
        "                   zero (default 0.1)\n"
        "  --lambda F       rate-distortion slope, times q^2 (default 0.12)\n"
        "  --max-block N    largest leaf, 4..32 (default 32)\n"
        "  --min-block N    smallest leaf, 4..max (default 4)\n"
        "  --no-deblock     leave the seams between leaves unfiltered\n"
        "  --band N         0..32, where texture splits between its low and high\n"
        "                   layers (default 8; 0 keeps it in one layer)\n",
        argv0);
}

static const char* layer_name(int k) { return k == 0 ? "COR" : k == 1 ? "TEX-BAIXA" : "TEX-ALTA"; }

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 2; }
    const char* in_path  = argv[1];
    const char* out_path = argv[2];
    NvdrConfig cfg = nvdr_default_config();

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--q") && i + 1 < argc) cfg.q = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chroma-q") && i + 1 < argc) cfg.chroma_q = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--deadzone") && i + 1 < argc) cfg.deadzone = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--lambda") && i + 1 < argc) cfg.lambda_k = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--max-block") && i + 1 < argc) cfg.max_block = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--min-block") && i + 1 < argc) cfg.min_block = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-deblock")) cfg.deblock = 0;
        else if (!strcmp(argv[i], "--band") && i + 1 < argc) cfg.band = atoi(argv[++i]);
        else { fprintf(stderr, "unknown option '%s'\n", argv[i]); usage(argv[0]); return 2; }
    }

    NvdrImage source;
    if (nvdr_image_load(&source, in_path) != 0) {
        fprintf(stderr, "cannot read image '%s'\n", in_path);
        return 1;
    }
    if (source.width > 65535 || source.height > 65535) {
        fprintf(stderr, "image exceeds the 65535px the container addresses\n");
        nvdr_image_free(&source);
        return 1;
    }

    NvdrHeader hdr;
    if (nvdr_encode_file(out_path, &source, &cfg, &hdr) != 0) {
        fprintf(stderr, "encode failed\n");
        nvdr_image_free(&source);
        return 1;
    }

    printf("%s  %dx%d  q %d/%d  blocks %d..%d\n", in_path, source.width, source.height,
           hdr.q_luma, hdr.q_chroma, hdr.min_block, hdr.max_block);
    printf("  layer       stored  cumulative     PSNR\n");
    size_t cumulative = NVDR_HEADER_SIZE;
    for (int k = 0; k < NVDR_LAYERS; k++) {
        NvdrImage shown;
        if (nvdr_decode_file(out_path, k, &shown, NULL, NULL) != 0) {
            fprintf(stderr, "wrote a container that does not read back\n");
            nvdr_image_free(&source);
            return 1;
        }
        cumulative += hdr.stored_bytes[k];
        printf("  %-8s %10u  %10zu   %6.2f dB\n", layer_name(k), hdr.stored_bytes[k],
               cumulative, nvdr_psnr(&source, &shown));
        nvdr_image_free(&shown);
    }
    printf("  leaves 4/8/16/32: %u/%u/%u/%u, %u with texture\n",
           hdr.leaves[0], hdr.leaves[1], hdr.leaves[2], hdr.leaves[3], hdr.textured);

    nvdr_image_free(&source);
    return 0;
}
