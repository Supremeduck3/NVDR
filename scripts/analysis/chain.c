/*
 * Does the error accumulate?
 *
 * Every measurement of frame N against frame N-1 is taken with a clean
 * reference. A real sequence never has one: frame 3 is predicted from a
 * reconstruction of a reconstruction of a reconstruction, and if each step
 * loses a little the picture walks away from the source until the next
 * intra frame rescues it. That drift is the failure mode that decides
 * whether residual-against-previous-frame is a codec or a demo, so it is
 * measured over a real chain rather than one step.
 *
 * Frame 0 is coded on its own. Every frame after it is coded as the error
 * against what the decoder actually holds, with one global motion vector,
 * and the decoder's state is carried forward exactly as a decoder would.
 */
#include "nvdr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int clamp255(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

static void find_shift(const NvdrImage* cur, const NvdrImage* ref, int range,
                       int* bdx, int* bdy) {
    double best = 1e30; *bdx = *bdy = 0;
    for (int dy = -range; dy <= range; dy++)
        for (int dx = -range; dx <= range; dx++) {
            double acc = 0; long n = 0;
            for (int y = range; y < cur->height - range; y += 3)
                for (int x = range; x < cur->width - range; x += 3) {
                    const unsigned char* a = cur->pixels + ((size_t)y*cur->width+x)*3;
                    const unsigned char* b = ref->pixels + ((size_t)(y+dy)*ref->width+(x+dx))*3;
                    for (int c = 0; c < 3; c++) { int d = a[c]-b[c]; acc += d<0?-d:d; }
                    n++;
                }
            if (n && acc/n < best) { best = acc/n; *bdx = dx; *bdy = dy; }
        }
}

static long file_size(const char* p) {
    FILE* f = fopen(p,"rb"); if(!f) return -1;
    fseek(f,0,SEEK_END); long n = ftell(f); fclose(f); return n;
}

static int roundtrip(const NvdrImage* in, const NvdrConfig* cfg,
                     const char* path, NvdrImage* out) {
    NvdrHeader hdr;
    if (nvdr_encode_file(path, in, cfg, &hdr) != 0) return -1;
    NvdrPyramid pyr;
    if (nvdr_decode_file(path, &pyr, &hdr) != 0) return -1;
    out->width = in->width; out->height = in->height;
    out->pixels = (unsigned char*)calloc((size_t)in->width*in->height*3, 1);
    if (!out->pixels) return -1;
    for (int k = 0; k < pyr.levels_present; k++) nvdr_render_level(&pyr.level[k], out);
    nvdr_pyramid_free(&pyr);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <dir> <frames> [--search N]\n", argv[0]); return 2; }
    const char* dir = argv[1];
    int frames = atoi(argv[2]);
    int range = 0;
    for (int i = 3; i < argc - 1; i++)
        if (!strcmp(argv[i], "--search")) range = atoi(argv[i+1]);

    NvdrConfig cfg = nvdr_default_config();
    NvdrImage state; state.pixels = NULL;
    long total_inter = 0, total_intra = 0;
    char path[512];

    printf("  quadro       bytes     PSNR   intra seria   deriva\n");
    double first_psnr = 0;
    for (int n = 0; n < frames; n++) {
        snprintf(path, sizeof(path), "%s/f%03d.ppm", dir, n);
        NvdrImage cur;
        if (nvdr_image_load(&cur, path) != 0) { fprintf(stderr, "cannot read %s\n", path); return 1; }

        /* What coding it on its own would have cost, for the comparison. */
        NvdrImage ri;
        if (roundtrip(&cur, &cfg, "/tmp/_ch_intra.nvdr", &ri) != 0) return 1;
        long si = file_size("/tmp/_ch_intra.nvdr");
        total_intra += si;
        double pi = nvdr_psnr(&cur, &ri);
        free(ri.pixels);

        long bytes; double psnr;
        if (!state.pixels) {
            /* Frame 0 has nothing to predict from. */
            NvdrImage rec;
            if (roundtrip(&cur, &cfg, "/tmp/_ch.nvdr", &rec) != 0) return 1;
            bytes = file_size("/tmp/_ch.nvdr");
            psnr = nvdr_psnr(&cur, &rec);
            state = rec;
        } else {
            NvdrImage ref = state;
            int dx = 0, dy = 0;
            if (range) {
                find_shift(&cur, &state, range, &dx, &dy);
                ref.width = state.width; ref.height = state.height;
                ref.pixels = (unsigned char*)malloc((size_t)ref.width*ref.height*3);
                for (int y = 0; y < ref.height; y++)
                    for (int x = 0; x < ref.width; x++) {
                        int sx = x+dx, sy = y+dy;
                        if (sx<0) sx=0; if (sx>=state.width) sx=state.width-1;
                        if (sy<0) sy=0; if (sy>=state.height) sy=state.height-1;
                        memcpy(ref.pixels+((size_t)y*ref.width+x)*3,
                               state.pixels+((size_t)sy*state.width+sx)*3, 3);
                    }
            }
            size_t np = (size_t)cur.width*cur.height*3;
            NvdrImage err; err.width = cur.width; err.height = cur.height;
            err.pixels = (unsigned char*)malloc(np);
            for (size_t i = 0; i < np; i++)
                err.pixels[i] = (unsigned char)clamp255(cur.pixels[i] - ref.pixels[i] + 128);

            NvdrImage rec_err;
            if (roundtrip(&err, &cfg, "/tmp/_ch.nvdr", &rec_err) != 0) return 1;
            bytes = file_size("/tmp/_ch.nvdr");

            /* The decoder's new state: reference plus decoded error. This
             * is what the next frame predicts from, drift included. */
            NvdrImage next; next.width = cur.width; next.height = cur.height;
            next.pixels = (unsigned char*)malloc(np);
            for (size_t i = 0; i < np; i++)
                next.pixels[i] = (unsigned char)clamp255(rec_err.pixels[i] - 128 + ref.pixels[i]);
            psnr = nvdr_psnr(&cur, &next);

            free(rec_err.pixels); free(err.pixels);
            if (range) free(ref.pixels);
            free(state.pixels);
            state = next;
        }
        total_inter += bytes;
        if (n == 0) first_psnr = psnr;
        printf("  %3d      %9ld  %6.2f dB   %8ld B     %+5.2f dB\n",
               n, bytes, psnr, si, psnr - first_psnr);
        nvdr_image_free(&cur);
    }
    printf("\n  total %ld B contra %ld B intra  (%+.1f%%)\n",
           total_inter, total_intra,
           100.0 * (total_inter - total_intra) / total_intra);
    return 0;
}
