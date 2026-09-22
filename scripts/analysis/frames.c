/*
 * Synthesise a frame sequence from a still image.
 *
 * There is no video in this repository and no tool here that can decode
 * one, so the inter-frame measurements have to start from something built
 * rather than filmed. This crops a moving window out of a larger picture,
 * which reproduces the two motions that dominate real footage — the camera
 * translating and the camera zooming — plus a region that moves against a
 * still background.
 *
 * Read the numbers it feeds knowing what it leaves out: a real sensor adds
 * noise that changes every frame, real footage arrives already compressed
 * so its artefacts move too, and real light changes. All three make frames
 * differ where this generator makes them identical, so a skip rate
 * measured on these frames is an upper bound. `--noise N` puts a
 * deterministic per-frame perturbation back so the gap is at least
 * visible.
 */
#include "nvdr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned rnd(unsigned* s) { *s = *s * 1664525u + 1013904223u; return *s >> 16; }

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr,
            "usage: %s <image> <outdir> <mode> [frames] [--noise N]\n"
            "  mode: static | pan | zoom | object | mixed\n", argv[0]);
        return 2;
    }
    const char* in = argv[1];
    const char* dir = argv[2];
    const char* mode = argv[3];
    int frames = argc > 4 && argv[4][0] != '-' ? atoi(argv[4]) : 8;
    int noise = 0;
    for (int i = 4; i < argc - 1; i++)
        if (!strcmp(argv[i], "--noise")) noise = atoi(argv[i + 1]);

    NvdrImage src;
    if (nvdr_image_load(&src, in) != 0) { fprintf(stderr, "cannot read %s\n", in); return 1; }

    /* A window small enough to leave room for the motion to move it. */
    int W = src.width * 2 / 3, H = src.height * 2 / 3;
    NvdrImage f;
    f.width = W; f.height = H;
    f.pixels = (unsigned char*)malloc((size_t)W * H * 3);
    if (!f.pixels) return 1;

    for (int n = 0; n < frames; n++) {
        int ox = (src.width - W) / 2, oy = (src.height - H) / 2;
        double scale = 1.0;
        if (!strcmp(mode, "pan") || !strcmp(mode, "mixed")) ox += n * 3;
        if (!strcmp(mode, "zoom")) scale = 1.0 - n * 0.01;

        unsigned seed = 12345u + (unsigned)n * 7919u;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int sx = ox + (int)(x * scale), sy = oy + (int)(y * scale);
                /* One region slides across an otherwise still frame. In
                 * `mixed` the camera pans at the same time, so the region
                 * and the background move in different directions and no
                 * single vector can describe the frame. */
                if (!strcmp(mode, "object") || !strcmp(mode, "mixed")) {
                    int bx = W / 6 + n * 5, by = H / 2;
                    if (x >= bx && x < bx + W / 5 && y >= by && y < by + H / 5) {
                        sx = ox + (x - bx) + W / 3;
                        sy = oy + (y - by) + H / 3;
                    }
                }
                if (sx < 0) sx = 0; if (sx >= src.width) sx = src.width - 1;
                if (sy < 0) sy = 0; if (sy >= src.height) sy = src.height - 1;
                const unsigned char* p = src.pixels + ((size_t)sy * src.width + sx) * 3;
                unsigned char* q = f.pixels + ((size_t)y * W + x) * 3;
                for (int c = 0; c < 3; c++) {
                    int v = p[c];
                    if (noise) v += (int)(rnd(&seed) % (unsigned)(2 * noise + 1)) - noise;
                    q[c] = (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
                }
            }
        }
        char path[512];
        snprintf(path, sizeof(path), "%s/f%03d.ppm", dir, n);
        if (nvdr_image_write_ppm(&f, path) != 0) { fprintf(stderr, "write failed\n"); return 1; }
    }
    printf("%d frames %dx%d, mode %s, noise %d -> %s\n", frames, W, H, mode, noise, dir);
    free(f.pixels);
    nvdr_image_free(&src);
    return 0;
}
