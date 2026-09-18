/*
 * nvdr_decode — render a PRS container at whatever level its bytes reach.
 *
 * This is the consumer side of the spec's central invariant (§9.2): the
 * display is never interrupted. Point it at a complete file and it renders
 * at full quality; point it at one truncated mid-stream and it renders at
 * the last level whose bytes are all present, says so, and exits 0. No
 * input short of a missing anchor produces no picture.
 */
#include "nvdr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* level_name(int level) {
    switch (level) {
        case 0:  return "ANCHOR";
        case 1:  return "ANCHOR+R1";
        default: return "ANCHOR+R1+R2";
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "usage: %s <in.nvdr> <out.ppm> [--level N] [--compare source]\n"
            "  --level N      cap the render at level N; the file may carry\n"
            "                 less, and then less is what you get\n"
            "  --compare IMG  also report PSNR against IMG\n",
            argv[0]);
        return 2;
    }

    const char* in_path      = argv[1];
    const char* out_path     = argv[2];
    const char* compare_path = NULL;
    int requested = NVDR_LEVELS - 1;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--level") && i + 1 < argc) requested = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--compare") && i + 1 < argc) compare_path = argv[++i];
    }
    if (requested < 0) requested = 0;
    if (requested > NVDR_LEVELS - 1) requested = NVDR_LEVELS - 1;

    NvdrPyramid pyr;
    NvdrHeader  hdr;
    if (nvdr_decode_file(in_path, &pyr, &hdr) != 0) {
        fprintf(stderr, "cannot decode '%s' (anchor missing or not an NVDR file)\n", in_path);
        return 1;
    }

    int available = pyr.levels_present - 1;
    int level = requested < available ? requested : available;

    NvdrImage out;
    out.width  = hdr.width;
    out.height = hdr.height;
    out.pixels = (unsigned char*)calloc((size_t)hdr.width * hdr.height * 3, 1);
    if (!out.pixels) {
        fprintf(stderr, "out of memory\n");
        nvdr_pyramid_free(&pyr);
        return 1;
    }

    nvdr_render_level(&pyr.level[level], &out);

    if (nvdr_image_write_ppm(&out, out_path) != 0) {
        fprintf(stderr, "cannot write '%s'\n", out_path);
        free(out.pixels);
        nvdr_pyramid_free(&pyr);
        return 1;
    }

    printf("%s  %dx%d  %u rects  rendered at %s",
           in_path, hdr.width, hdr.height, pyr.level[level].count, level_name(level));
    if (available < requested)
        printf("  (file carries only %s)", level_name(available));

    if (compare_path) {
        NvdrImage source;
        if (nvdr_image_load(&source, compare_path) == 0) {
            printf("  psnr %.2f dB", nvdr_psnr(&source, &out));
            nvdr_image_free(&source);
        }
    }
    printf("\n");

    free(out.pixels);
    nvdr_pyramid_free(&pyr);
    return 0;
}
