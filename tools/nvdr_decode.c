/*
 * nvdr_decode — render a v10 container as far as its bytes reach.
 *
 * Point it at a complete file and it renders at full quality; point it at
 * one cut short and it renders what arrived, says how much, and exits 0.
 * Only a file with no usable header, or no colour at all, has no picture.
 */
#include "nvdr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "usage: %s <in.nvdr> <out.png|out.ppm> [--layer N] [--compare source]\n"
            "  --layer N      0 renders colour only, 1 adds texture (default);\n"
            "                 the file may carry less, and then less is what you get\n"
            "  --compare IMG  also report PSNR against IMG\n",
            argv[0]);
        return 2;
    }
    const char* in_path = argv[1];
    const char* out_path = argv[2];
    const char* compare_path = NULL;
    int layer = NVDR_LAYERS - 1;
    for (int i = 3; i < argc; i++) {
        if ((!strcmp(argv[i], "--layer") || !strcmp(argv[i], "--level")) && i + 1 < argc)
            layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--compare") && i + 1 < argc) compare_path = argv[++i];
    }
    if (layer < 0) layer = 0;
    if (layer > NVDR_LAYERS - 1) layer = NVDR_LAYERS - 1;

    NvdrImage out;
    NvdrHeader hdr;
    NvdrDecodeInfo info;
    if (nvdr_decode_file(in_path, layer, &out, &hdr, &info) != 0) {
        fprintf(stderr, "cannot decode '%s' (no colour layer, or not an NVDR v10 file)\n", in_path);
        return 1;
    }
    if (nvdr_image_write(&out, out_path) != 0) {
        fprintf(stderr, "cannot write '%s'\n", out_path);
        nvdr_image_free(&out);
        return 1;
    }

    printf("%s  %dx%d  cor %d/%d tiles", in_path, hdr.width, hdr.height,
           info.tiles_complete[0], info.tiles);
    if (layer >= 1) printf("  textura %d/%d tiles", info.tiles_complete[1], info.tiles);
    if (compare_path) {
        NvdrImage source;
        if (nvdr_image_load(&source, compare_path) == 0) {
            printf("  psnr %.2f dB", nvdr_psnr(&source, &out));
            nvdr_image_free(&source);
        }
    }
    printf("\n");
    nvdr_image_free(&out);
    return 0;
}
