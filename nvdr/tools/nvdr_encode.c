/*
 * nvdr_encode — build a PRS container from an image and report what each
 * layer actually costs and buys.
 *
 * The report is the point. The spec makes three testable claims about the
 * residual stack (§1.3): that the anchor carries the bulk of the
 * information, that R1 and R2 are far lower entropy than the layer they
 * correct, and that the three together reconstruct the target exactly.
 * Every run prints the numbers that confirm or refute them on real data.
 */
#include "nvdr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char* argv0) {
    fprintf(stderr,
        "usage: %s <image> <out.nvdr> [options]\n"
        "  --anchor-bits N   palette is 2^N entries (default 4, the spec's int4)\n"
        "  --r1-step N       coarse residual quantisation step (default 16)\n"
        "  --homogeneity F   split regions less uniform than F (default 0.020)\n"
        "  --min-tile N      smallest tile edge (default 2)\n"
        "  --max-depth N     deepest subdivision (default 12)\n",
        argv0);
}

static double psnr_at(const NvdrImage* source, const NvdrGeometry* geo,
                      const NvdrStack* stack, NvdrLevel level,
                      unsigned char* leaf_rgb, NvdrImage* scratch) {
    nvdr_stack_resolve(stack, level, leaf_rgb);
    nvdr_render(geo, leaf_rgb, scratch);
    return nvdr_psnr(source, scratch);
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 2; }

    const char* in_path  = argv[1];
    const char* out_path = argv[2];

    NvdrBuildConfig  build = nvdr_default_build_config();
    NvdrEncodeConfig enc   = nvdr_default_encode_config();

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--anchor-bits") && i + 1 < argc) enc.anchor_bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--r1-step") && i + 1 < argc) enc.r1_step = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--homogeneity") && i + 1 < argc) build.homogeneity = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--min-tile") && i + 1 < argc) build.min_tile = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-depth") && i + 1 < argc) build.max_depth = atoi(argv[++i]);
        else { usage(argv[0]); return 2; }
    }
    if (enc.anchor_bits < 1 || enc.anchor_bits > 8) {
        fprintf(stderr, "anchor-bits must be between 1 and 8\n");
        return 2;
    }
    if (enc.r1_step < 1 || enc.r1_step > 255) {
        fprintf(stderr, "r1-step must be between 1 and 255\n");
        return 2;
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

    NvdrGeometry geo;
    NvdrStack    stack;
    if (nvdr_encode_image(&source, &build, &enc, &geo, &stack) != 0) {
        fprintf(stderr, "encode failed\n");
        nvdr_image_free(&source);
        return 1;
    }

    if (nvdr_container_write(out_path, &geo, &stack, source.width, source.height) != 0) {
        fprintf(stderr, "cannot write '%s'\n", out_path);
        nvdr_stack_free(&stack);
        nvdr_geometry_free(&geo);
        nvdr_image_free(&source);
        return 1;
    }

    /* --- what each level is worth --- */
    unsigned char* leaf_rgb = (unsigned char*)malloc((size_t)geo.leaf_count * 3);
    NvdrImage scratch;
    scratch.width = source.width;
    scratch.height = source.height;
    scratch.pixels = (unsigned char*)calloc((size_t)source.width * source.height * 3, 1);

    double psnr_anchor = 0, psnr_r1 = 0, psnr_r2 = 0;
    if (leaf_rgb && scratch.pixels) {
        psnr_anchor = psnr_at(&source, &geo, &stack, NVDR_LEVEL_ANCHOR, leaf_rgb, &scratch);
        psnr_r1     = psnr_at(&source, &geo, &stack, NVDR_LEVEL_R1,     leaf_rgb, &scratch);
        psnr_r2     = psnr_at(&source, &geo, &stack, NVDR_LEVEL_R2,     leaf_rgb, &scratch);
    }

    size_t geo_bits   = ((size_t)geo.node_count + 7) / 8;
    size_t token_bits = ((size_t)geo.leaf_count * enc.anchor_bits + 7) / 8;
    size_t anchor_sz  = 1 + (size_t)stack.anchor_palette_n * 3 + geo_bits + token_bits;
    size_t residual_sz = (size_t)geo.leaf_count * 3;

    printf("%s  %dx%d\n", in_path, source.width, source.height);
    printf("  quadtree      %u nodes, %u leaves\n", geo.node_count, geo.leaf_count);
    printf("  layer            bytes    cumulative    PSNR\n");
    printf("  ANCHOR       %9zu    %9zu   %6.2f dB   (geometry %zu + tokens %zu + palette %d)\n",
           anchor_sz, anchor_sz, psnr_anchor, geo_bits, token_bits, stack.anchor_palette_n);
    printf("  R1           %9zu    %9zu   %6.2f dB\n",
           residual_sz, anchor_sz + residual_sz, psnr_r1);
    printf("  R2           %9zu    %9zu   %6.2f dB\n",
           residual_sz, anchor_sz + residual_sz * 2, psnr_r2);

    free(leaf_rgb);
    free(scratch.pixels);
    nvdr_stack_free(&stack);
    nvdr_geometry_free(&geo);
    nvdr_image_free(&source);
    return 0;
}
