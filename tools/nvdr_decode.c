/*
 * nvdr_decode — render a v10 container as far as its bytes reach.
 *
 * Point it at a complete file and it renders at full quality; point it at
 * one cut short and it renders what arrived, says how much, and exits 0.
 * Only a file with no usable header, or no colour at all, has no picture.
 * With --missing, the byte ranges named never arrived (a datagram link
 * that lost packets): each layer's part of each slice decodes up to its
 * first missing byte.
 */
#include "nvdr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "usage: %s <in.nvdr> <out.png|out.ppm> [--layer N] [--compare source]\n"
            "  --layer N      0 renders colour only, 1 adds the low texture band,\n"
            "                 2 the high one too (default);\n"
            "                 the file may carry less, and then less is what you get\n"
            "  --compare IMG  also report PSNR against IMG\n"
            "  --missing A-B,C-D,...  byte ranges [A, B) that never arrived\n",
            argv[0]);
        return 2;
    }
    const char* in_path = argv[1];
    const char* out_path = argv[2];
    const char* compare_path = NULL;
    const char* missing = NULL;
    int layer = NVDR_LAYERS - 1;
    for (int i = 3; i < argc; i++) {
        if ((!strcmp(argv[i], "--layer") || !strcmp(argv[i], "--level")) && i + 1 < argc)
            layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--compare") && i + 1 < argc) compare_path = argv[++i];
        else if (!strcmp(argv[i], "--missing") && i + 1 < argc) missing = argv[++i];
    }
    if (layer < 0) layer = 0;
    if (layer > NVDR_LAYERS - 1) layer = NVDR_LAYERS - 1;

    NvdrImage out;
    NvdrHeader hdr;
    NvdrDecodeInfo info;
    int rc;
    if (missing) {
        size_t n = 1;
        for (const char* p = missing; *p; p++) n += *p == ',';
        size_t* holes = (size_t*)malloc(sizeof(size_t) * 2 * n);
        int nh = 0;
        for (const char* p = missing; holes && *p;) {
            char* e;
            unsigned long long a = strtoull(p, &e, 10), b;
            if (*e != '-') { fprintf(stderr, "--missing wants A-B,C-D,...\n"); free(holes); return 2; }
            b = strtoull(e + 1, &e, 10);
            holes[2 * nh] = (size_t)a; holes[2 * nh + 1] = (size_t)b; nh++;
            p = *e == ',' ? e + 1 : e;
            if (*e && *e != ',') { fprintf(stderr, "--missing wants A-B,C-D,...\n"); free(holes); return 2; }
        }
        FILE* f = fopen(in_path, "rb");
        uint8_t* data = NULL;
        size_t size = 0;
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (sz > 0 && (data = (uint8_t*)malloc((size_t)sz))) size = fread(data, 1, (size_t)sz, f);
            fclose(f);
        }
        rc = data && holes ? nvdr_decode_mem_lossy(data, size, holes, nh, layer, &out, &hdr, &info) : -1;
        free(data);
        free(holes);
    } else rc = nvdr_decode_file(in_path, layer, &out, &hdr, &info);
    if (rc != 0) {
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
    if (layer >= 1) printf("  textura %d/%d", info.tiles_complete[1], info.tiles);
    if (layer >= 2 && hdr.band) printf("  alta %d/%d", info.tiles_complete[2], info.tiles);
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
