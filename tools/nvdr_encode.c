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
        "                   layers (default 8; 0 keeps it in one layer)\n"
        "  --chroma M       auto (default), 420 or 444: colour at half resolution\n"
        "                   each way or whole; auto halves it unless that costs\n"
        "                   more than it saves (graphics, hard colour edges)\n"
        "  --grain M        off (default), auto or on: take the sensor noise out,\n"
        "                   code the clean picture, and have the decoder lay the\n"
        "                   same kind of grain back (auto: only if noisy)\n"
        "  --quiet          write the file and skip the per-layer report, which\n"
        "                   decodes it three times\n"
        "  --no-rdoq        round texture levels with the dead zone instead of\n"
        "                   choosing them by rate-distortion\n"
        "  --tile-q-test    give every tile a step offset from a fixed pattern,\n"
        "                   to exercise the decoders' per-tile steps\n",
        argv0);
}

static const char* layer_name(int k) { return k == 0 ? "COR" : k == 1 ? "TEX-BAIXA" : "TEX-ALTA"; }

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 2; }
    const char* in_path  = argv[1];
    const char* out_path = argv[2];
    NvdrConfig cfg = nvdr_default_config();
    int quiet = 0, tile_q_test = 0;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--q") && i + 1 < argc) cfg.q = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chroma-q") && i + 1 < argc) cfg.chroma_q = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--deadzone") && i + 1 < argc) cfg.deadzone = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--lambda") && i + 1 < argc) cfg.lambda_k = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--max-block") && i + 1 < argc) cfg.max_block = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--min-block") && i + 1 < argc) cfg.min_block = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-deblock")) cfg.deblock = 0;
        else if (!strcmp(argv[i], "--band") && i + 1 < argc) cfg.band = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
        else if (!strcmp(argv[i], "--tile-q-test")) tile_q_test = 1;
        else if (!strcmp(argv[i], "--no-rdoq")) cfg.rdoq = 0;
        else if (!strcmp(argv[i], "--grain") && i + 1 < argc) {
            const char* v = argv[++i];
            if (!strcmp(v, "off")) cfg.grain = NVDR_GRAIN_OFF;
            else if (!strcmp(v, "auto")) cfg.grain = NVDR_GRAIN_AUTO;
            else if (!strcmp(v, "on")) cfg.grain = NVDR_GRAIN_ON;
            else { fprintf(stderr, "--grain takes off, auto or on\n"); return 2; }
        }
        else if (!strcmp(argv[i], "--chroma") && i + 1 < argc) {
            const char* v = argv[++i];
            if (!strcmp(v, "420")) cfg.chroma420 = NVDR_CHROMA_420;
            else if (!strcmp(v, "444")) cfg.chroma420 = NVDR_CHROMA_444;
            else if (!strcmp(v, "auto")) cfg.chroma420 = NVDR_CHROMA_AUTO;
            else { fprintf(stderr, "--chroma takes auto, 420 or 444\n"); return 2; }
        }
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

    /* Offsets from -12 to 12 in a pattern that changes every tile, so the
     * steps, the deblocking and the differences between tiles are all
     * exercised. */
    int8_t* tq = NULL;
    if (tile_q_test) {
        int tile = cfg.max_block;
        size_t tiles = (size_t)((source.width + tile - 1) / tile) * ((source.height + tile - 1) / tile);
        tq = (int8_t*)malloc(tiles);
        if (!tq) { nvdr_image_free(&source); return 1; }
        for (size_t t = 0; t < tiles; t++) tq[t] = (int8_t)((int)((t * 7) % 25) - 12);
        cfg.tile_q = tq;
    }

    NvdrHeader hdr;
    int enc_rc = nvdr_encode_file(out_path, &source, &cfg, &hdr);
    free(tq);
    if (enc_rc != 0) {
        fprintf(stderr, "encode failed\n");
        nvdr_image_free(&source);
        return 1;
    }

    /* The report decodes the file once per layer: whoever only wants the
     * file skips it. */
    if (quiet) { nvdr_image_free(&source); return 0; }

    printf("%s  %dx%d  q %d/%d  blocks %d..%d  colour %s\n", in_path, source.width, source.height,
           hdr.q_luma, hdr.q_chroma, hdr.min_block, hdr.max_block,
           (hdr.flags & NVDR_FLAG_CHROMA420) ? "4:2:0" : "4:4:4");
    if (hdr.flags & NVDR_FLAG_GRAIN) {
        printf("  grain: kernel %d, colour %d/%d, luma sigma x8", hdr.grain.kernel, hdr.grain.cb, hdr.grain.cr);
        for (int k = 0; k < NVDR_GRAIN_POINTS; k++) printf(" %d", hdr.grain.sigma[k]);
        printf("\n");
    }
    printf("  layer       stored  cumulative     PSNR\n");
    /* One decode per layer, independent of each other: side by side. */
    double psnr[NVDR_LAYERS];
    int failed = 0;
    #pragma omp parallel for reduction(|:failed)
    for (int k = 0; k < NVDR_LAYERS; k++) {
        NvdrImage shown;
        if (nvdr_decode_file(out_path, k, &shown, NULL, NULL) != 0) { failed = 1; continue; }
        psnr[k] = nvdr_psnr(&source, &shown);
        nvdr_image_free(&shown);
    }
    if (failed) {
        fprintf(stderr, "wrote a container that does not read back\n");
        nvdr_image_free(&source);
        return 1;
    }
    size_t cumulative = NVDR_HEADER_SIZE + hdr.grain_len;
    for (int k = 0; k < NVDR_LAYERS; k++) {
        cumulative += hdr.stored_bytes[k];
        printf("  %-8s %10u  %10zu   %6.2f dB\n", layer_name(k), hdr.stored_bytes[k],
               cumulative, psnr[k]);
    }
    printf("  leaves 4/8/16/32: %u/%u/%u/%u, %u with texture\n",
           hdr.leaves[0], hdr.leaves[1], hdr.leaves[2], hdr.leaves[3], hdr.textured);
    double planes = hdr.plane_bits[0] + hdr.plane_bits[1] + hdr.plane_bits[2];
    if (planes > 0)
        printf("  planes Y/Cb/Cr: %.1f%%/%.1f%%/%.1f%% of the leaves' bits\n",
               100 * hdr.plane_bits[0] / planes, 100 * hdr.plane_bits[1] / planes,
               100 * hdr.plane_bits[2] / planes);

    nvdr_image_free(&source);
    return 0;
}
