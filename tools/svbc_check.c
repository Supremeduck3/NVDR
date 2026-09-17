/*
 * svbc_check — independent verifier for a .svbc against its source image.
 *
 * Decodes the container through svbc_read() (the same reader a third-party
 * consumer would use, and until now dead code), rasterises it, and checks
 * the properties that have to hold for the file to be worth anything:
 *
 *   - every leaf's token indexes a real codebook entry
 *   - the leaves cover every pixel of the declared canvas
 *   - the reconstruction's PSNR against the source image
 *
 * Exit code is 0 only when the structural checks pass and PSNR clears the
 * floor given by --min-psnr (default 0, i.e. report only).
 *
 * Usage: svbc_check <source image> <file.svbc> [--min-psnr X] [--quiet]
 */
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "svbc_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: svbc_check <source image> <file.svbc> "
                        "[--min-psnr X] [--quiet]\n");
        return 2;
    }
    const char* src_path  = argv[1];
    const char* svbc_path = argv[2];
    double min_psnr = 0.0;
    int quiet = 0;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--min-psnr") && i + 1 < argc) min_psnr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
    }

    int sw, sh, sc;
    unsigned char* src = stbi_load(src_path, &sw, &sh, &sc, 3);
    if (!src) {
        fprintf(stderr, "svbc_check: cannot read source '%s'\n", src_path);
        return 2;
    }

    SVBCFile f;
    if (svbc_read(&f, svbc_path) != 0) {
        fprintf(stderr, "svbc_check: cannot read container '%s'\n", svbc_path);
        stbi_image_free(src);
        return 1;
    }

    int w = f.header.img_width, h = f.header.img_height;
    int failures = 0;

    if (w != sw || h != sh) {
        printf("FAIL  canvas %dx%d does not match source %dx%d\n", w, h, sw, sh);
        failures++;
    }

    unsigned char* recon = (unsigned char*)calloc((size_t)w * h * 3, 1);
    unsigned char* cover = (unsigned char*)calloc((size_t)w * h, 1);
    if (!recon || !cover) {
        fprintf(stderr, "svbc_check: out of memory\n");
        free(recon); free(cover); svbc_free(&f); stbi_image_free(src);
        return 2;
    }

    long bad_tokens = 0, out_of_bounds = 0;
    for (uint32_t i = 0; i < f.header.node_count; i++) {
        const SVBC_Node* n = &f.nodes[i];
        if (n->token_id >= f.header.codebook_count) { bad_tokens++; continue; }
        if (n->x + n->w > w || n->y + n->h > h) out_of_bounds++;

        const SVBC_Color* c = &f.palette[n->token_id];
        int y1 = n->y + n->h > h ? h : n->y + n->h;
        int x1 = n->x + n->w > w ? w : n->x + n->w;
        for (int y = n->y; y < y1; y++) {
            for (int x = n->x; x < x1; x++) {
                size_t p = (size_t)y * w + x;
                recon[p * 3 + 0] = c->r;
                recon[p * 3 + 1] = c->g;
                recon[p * 3 + 2] = c->b;
                if (cover[p] < 255) cover[p]++;
            }
        }
    }

    long uncovered = 0, overlapped = 0;
    for (size_t p = 0; p < (size_t)w * h; p++) {
        if (cover[p] == 0) uncovered++;
        else if (cover[p] > 1) overlapped++;
    }

    double mse = 0.0;
    if (w == sw && h == sh) {
        for (size_t p = 0; p < (size_t)w * h * 3; p++) {
            double d = (double)src[p] - (double)recon[p];
            mse += d * d;
        }
        mse /= (double)((size_t)w * h * 3);
    }
    double psnr = mse > 0.0 ? 10.0 * log10(255.0 * 255.0 / mse) : 99.0;

    if (bad_tokens)   { printf("FAIL  %ld leaves point outside the codebook\n", bad_tokens); failures++; }
    if (uncovered)    { printf("FAIL  %ld pixels are covered by no leaf\n", uncovered); failures++; }
    if (out_of_bounds){ printf("WARN  %ld leaves extend past the canvas\n", out_of_bounds); }
    if (psnr < min_psnr) {
        printf("FAIL  PSNR %.2f dB is below the %.2f dB floor\n", psnr, min_psnr);
        failures++;
    }

    if (!quiet) {
        printf("%-28s %5dx%-5d nodes %7u  codebook %5u  "
               "psnr %5.2f dB  overlap %.3f%%  %s\n",
               svbc_path, w, h, f.header.node_count, f.header.codebook_count,
               psnr, 100.0 * (double)overlapped / (double)((size_t)w * h),
               failures ? "FAIL" : "ok");
    }

    free(recon); free(cover);
    svbc_free(&f);
    stbi_image_free(src);
    return failures ? 1 : 0;
}
