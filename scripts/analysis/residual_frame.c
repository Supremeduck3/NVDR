/*
 * Coding a frame as a residual against the previous one, measured end to
 * end with the codec exactly as it stands.
 *
 * No new format is needed to find out whether this direction works. The
 * prediction error is an image: encode frame N-1, decode it to get the
 * reference a decoder would hold, subtract it from frame N, bias by 128,
 * and hand that to the encoder. Decoding it and adding the reference back
 * reconstructs frame N, so the container size and the final PSNR are
 * directly comparable to encoding frame N on its own.
 *
 * With --search N the reference is first shifted by the whole-frame
 * translation that minimises absolute error, searched over +/-N pixels.
 * That is the crudest possible motion compensation — one vector for the
 * entire frame — and it is here because a co-located reference measured
 * worthless under a 3 px pan, so the question of whether motion is
 * optional needed an answer rather than an opinion.
 */
#include "nvdr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int clamp255(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

/* Whole-frame translation minimising mean absolute difference. */
static void find_shift(const NvdrImage* cur, const NvdrImage* ref, int range,
                       int* best_dx, int* best_dy) {
    double best = 1e30;
    *best_dx = *best_dy = 0;
    for (int dy = -range; dy <= range; dy++) {
        for (int dx = -range; dx <= range; dx++) {
            double acc = 0.0; long n = 0;
            /* Sample on a lattice: a full pass per candidate is not worth
             * the accuracy at this stage. */
            for (int y = range; y < cur->height - range; y += 3) {
                for (int x = range; x < cur->width - range; x += 3) {
                    const unsigned char* a = cur->pixels + ((size_t)y*cur->width + x)*3;
                    const unsigned char* b = ref->pixels + ((size_t)(y+dy)*ref->width + (x+dx))*3;
                    int d = a[0]-b[0]; acc += d<0?-d:d;
                    d = a[1]-b[1]; acc += d<0?-d:d;
                    d = a[2]-b[2]; acc += d<0?-d:d;
                    n++;
                }
            }
            if (n && acc / n < best) { best = acc / n; *best_dx = dx; *best_dy = dy; }
        }
    }
}

static void shift_image(const NvdrImage* src, NvdrImage* dst, int dx, int dy) {
    for (int y = 0; y < dst->height; y++)
        for (int x = 0; x < dst->width; x++) {
            int sx = x + dx, sy = y + dy;
            if (sx < 0) sx = 0; if (sx >= src->width) sx = src->width - 1;
            if (sy < 0) sy = 0; if (sy >= src->height) sy = src->height - 1;
            memcpy(dst->pixels + ((size_t)y*dst->width + x)*3,
                   src->pixels + ((size_t)sy*src->width + sx)*3, 3);
        }
}

static long file_size(const char* p) {
    FILE* f = fopen(p, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); long n = ftell(f); fclose(f); return n;
}

/* Encode an image, decode it back, and render the reconstruction. */
static int roundtrip(const NvdrImage* in, const NvdrConfig* cfg,
                     const char* path, NvdrImage* out) {
    NvdrHeader hdr;
    if (nvdr_encode_file(path, in, cfg, &hdr) != 0) return -1;
    NvdrPyramid pyr;
    if (nvdr_decode_file(path, &pyr, &hdr) != 0) return -1;
    out->width = in->width; out->height = in->height;
    out->pixels = (unsigned char*)calloc((size_t)in->width * in->height * 3, 1);
    if (!out->pixels) return -1;
    for (int k = 0; k < pyr.levels_present; k++) nvdr_render_level(&pyr.level[k], out);
    nvdr_pyramid_free(&pyr);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <prev.ppm> <cur.ppm> [--search N]\n", argv[0]);
        return 2;
    }
    int range = 0;
    for (int i = 3; i < argc - 1; i++)
        if (!strcmp(argv[i], "--search")) range = atoi(argv[i + 1]);

    NvdrConfig cfg = nvdr_default_config();
    NvdrImage prev, cur;
    if (nvdr_image_load(&prev, argv[1]) != 0 || nvdr_image_load(&cur, argv[2]) != 0) return 1;

    /* The reference a decoder would hold after the previous frame. */
    NvdrImage ref;
    if (roundtrip(&prev, &cfg, "/tmp/_rf_prev.nvdr", &ref) != 0) return 1;

    int dx = 0, dy = 0;
    if (range) {
        find_shift(&cur, &ref, range, &dx, &dy);
        NvdrImage moved;
        moved.width = ref.width; moved.height = ref.height;
        moved.pixels = (unsigned char*)malloc((size_t)ref.width*ref.height*3);
        shift_image(&ref, &moved, dx, dy);
        free(ref.pixels);
        ref = moved;
    }

    /* Intra: the frame on its own, as the codec does it today. */
    NvdrImage rec_intra;
    if (roundtrip(&cur, &cfg, "/tmp/_rf_intra.nvdr", &rec_intra) != 0) return 1;
    long size_intra = file_size("/tmp/_rf_intra.nvdr");
    double psnr_intra = nvdr_psnr(&cur, &rec_intra);

    /* Inter: the prediction error, biased to the middle of the range. */
    NvdrImage err;
    err.width = cur.width; err.height = cur.height;
    err.pixels = (unsigned char*)malloc((size_t)cur.width*cur.height*3);
    size_t n = (size_t)cur.width * cur.height * 3;
    for (size_t i = 0; i < n; i++)
        err.pixels[i] = (unsigned char)clamp255(cur.pixels[i] - ref.pixels[i] + 128);

    NvdrImage rec_err;
    if (roundtrip(&err, &cfg, "/tmp/_rf_inter.nvdr", &rec_err) != 0) return 1;
    long size_inter = file_size("/tmp/_rf_inter.nvdr");

    /* Add the reference back to get frame N. */
    NvdrImage rec_inter;
    rec_inter.width = cur.width; rec_inter.height = cur.height;
    rec_inter.pixels = (unsigned char*)malloc(n);
    for (size_t i = 0; i < n; i++)
        rec_inter.pixels[i] = (unsigned char)clamp255(rec_err.pixels[i] - 128 + ref.pixels[i]);
    double psnr_inter = nvdr_psnr(&cur, &rec_inter);

    const char* name = strrchr(argv[2], '/');
    printf("%-10s intra %7ld B %5.2f dB | inter %7ld B %5.2f dB | %+5.1f%% bytes, "
           "%+5.2f dB",
           name ? name + 1 : argv[2], size_intra, psnr_intra, size_inter, psnr_inter,
           100.0 * (size_inter - size_intra) / size_intra, psnr_inter - psnr_intra);
    if (range) printf(" | shift %+d%+d", dx, dy);
    printf("\n");
    return 0;
}
