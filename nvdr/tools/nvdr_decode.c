/*
 * nvdr_decode — render a PRS container at whatever level its bytes reach.
 *
 * This is the consumer side of the spec's central invariant (§9.2): the
 * display is never interrupted. Point it at a complete file and it renders
 * at full quality; point it at one truncated mid-stream and it renders at
 * the last level whose bytes are all present, says so, and exits 0. There
 * is no input short of a missing anchor that produces no picture.
 */
#include "nvdr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* level_name(NvdrLevel level) {
    switch (level) {
        case NVDR_LEVEL_ANCHOR: return "ANCHOR";
        case NVDR_LEVEL_R1:     return "ANCHOR+R1";
        default:                return "ANCHOR+R1+R2";
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "usage: %s <in.nvdr> <out.ppm> [--level N] [--compare source]\n"
            "  --level N      cap the render at level N (0/1/2); the file may\n"
            "                 still carry less, and then less is what you get\n"
            "  --compare IMG  also report PSNR against IMG\n",
            argv[0]);
        return 2;
    }

    const char* in_path      = argv[1];
    const char* out_path     = argv[2];
    const char* compare_path = NULL;
    int requested = NVDR_LEVEL_R2;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--level") && i + 1 < argc) requested = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--compare") && i + 1 < argc) compare_path = argv[++i];
    }
    if (requested < 0) requested = 0;
    if (requested > NVDR_LEVEL_R2) requested = NVDR_LEVEL_R2;

    NvdrGeometry geo;
    NvdrStack    stack;
    NvdrHeader   hdr;
    NvdrLevel    available;

    if (nvdr_container_read(in_path, &geo, &stack, &hdr, &available) != 0) {
        fprintf(stderr, "cannot decode '%s' (anchor missing or not an NVDR file)\n", in_path);
        return 1;
    }

    NvdrLevel level = (NvdrLevel)(requested < (int)available ? requested : (int)available);

    unsigned char* leaf_rgb = (unsigned char*)malloc((size_t)geo.leaf_count * 3);
    NvdrImage out;
    out.width  = hdr.width;
    out.height = hdr.height;
    out.pixels = (unsigned char*)calloc((size_t)hdr.width * hdr.height * 3, 1);
    if (!leaf_rgb || !out.pixels) {
        fprintf(stderr, "out of memory\n");
        free(leaf_rgb); free(out.pixels);
        nvdr_stack_free(&stack); nvdr_geometry_free(&geo);
        return 1;
    }

    nvdr_stack_resolve(&stack, level, leaf_rgb);
    nvdr_render(&geo, leaf_rgb, &out);

    if (nvdr_image_write_ppm(&out, out_path) != 0) {
        fprintf(stderr, "cannot write '%s'\n", out_path);
        free(leaf_rgb); free(out.pixels);
        nvdr_stack_free(&stack); nvdr_geometry_free(&geo);
        return 1;
    }

    printf("%s  %dx%d  %u leaves  rendered at %s",
           in_path, hdr.width, hdr.height, geo.leaf_count, level_name(level));
    if ((int)available < requested)
        printf("  (file carries only %s)", level_name(available));

    if (compare_path) {
        NvdrImage source;
        if (nvdr_image_load(&source, compare_path) == 0) {
            printf("  psnr %.2f dB", nvdr_psnr(&source, &out));
            nvdr_image_free(&source);
        }
    }
    printf("\n");

    free(leaf_rgb);
    free(out.pixels);
    nvdr_stack_free(&stack);
    nvdr_geometry_free(&geo);
    return 0;
}
