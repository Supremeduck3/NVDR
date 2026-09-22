/*
 * How much of a predicted frame is spent re-coding the codec's own error?
 *
 * A predicted frame codes cur - ref. With ref the previous frame's
 * reconstruction, that residual is the genuine change between frames plus
 * whatever error the previous frame left behind, moved along. Coding the
 * same frame against the previous *source* instead removes the second
 * term. The byte difference between the two is what the loop spends
 * correcting itself, at the same tolerance, with the same motion.
 */
#include "nvdr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int c255(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

static void shift(const NvdrImage* s, NvdrImage* d, int dx, int dy) {
    for (int y = 0; y < d->height; y++) for (int x = 0; x < d->width; x++) {
        int sx = x + dx, sy = y + dy;
        if (sx < 0) sx = 0; if (sx >= s->width) sx = s->width - 1;
        if (sy < 0) sy = 0; if (sy >= s->height) sy = s->height - 1;
        memcpy(d->pixels + ((size_t)y * d->width + x) * 3, s->pixels + ((size_t)sy * s->width + sx) * 3, 3);
    }
}
static double mad(const NvdrImage* a, const NvdrImage* b) {
    double acc = 0; size_t n = (size_t)a->width * a->height * 3;
    for (size_t i = 0; i < n; i += 5) { int d = a->pixels[i] - b->pixels[i]; acc += d < 0 ? -d : d; }
    return acc / (n / 5);
}
static size_t code(const NvdrImage* cur, const NvdrImage* ref, const NvdrConfig* cfg) {
    size_t n = (size_t)cur->width * cur->height * 3;
    NvdrImage e = { malloc(n), cur->width, cur->height };
    for (size_t i = 0; i < n; i++) e.pixels[i] = (unsigned char)c255(cur->pixels[i] - ref->pixels[i] + 128);
    uint8_t* blob; size_t len; NvdrHeader h;
    nvdr_encode_mem(&blob, &len, &e, cfg, &h);
    free(blob); free(e.pixels);
    return len;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <prev> <cur> <tol-a,b,c>\n", argv[0]); return 2; }
    NvdrConfig cfg = nvdr_default_config();
    sscanf(argv[3], "%f,%f,%f", &cfg.tolerance[0], &cfg.tolerance[1], &cfg.tolerance[2]);
    NvdrImage prev, cur;
    if (nvdr_image_load(&prev, argv[1]) || nvdr_image_load(&cur, argv[2])) return 1;

    /* The previous frame as a decoder would hold it, coded intra here. */
    uint8_t* blob; size_t len; NvdrHeader h; NvdrPyramid pyr;
    nvdr_encode_mem(&blob, &len, &prev, &cfg, &h);
    nvdr_decode_mem(blob, len, &pyr, &h);
    NvdrImage recon = { calloc((size_t)prev.width * prev.height * 3, 1), prev.width, prev.height };
    for (int k = 0; k < pyr.levels_present; k++) nvdr_render_level(&pyr.level[k], &recon);

    /* One motion vector, found against the clean source so both residuals
     * use exactly the same motion. */
    int bdx = 0, bdy = 0; double best = 1e9;
    NvdrImage tmp = { malloc((size_t)cur.width * cur.height * 3), cur.width, cur.height };
    for (int dy = -4; dy <= 4; dy++) for (int dx = -4; dx <= 4; dx++) {
        shift(&prev, &tmp, dx, dy); double m = mad(&cur, &tmp);
        if (m < best) { best = m; bdx = dx; bdy = dy; }
    }
    NvdrImage ref_src = { malloc((size_t)cur.width * cur.height * 3), cur.width, cur.height };
    NvdrImage ref_rec = { malloc((size_t)cur.width * cur.height * 3), cur.width, cur.height };
    shift(&prev, &ref_src, bdx, bdy);
    shift(&recon, &ref_rec, bdx, bdy);

    size_t open = code(&cur, &ref_src, &cfg), closed = code(&cur, &ref_rec, &cfg);
    printf("  vetor %+d%+d | erro da referencia %.2f niveis | mudanca real %.2f niveis | "
           "contra a fonte %6zu B, contra a reconstrucao %6zu B -> %.0f%% do quadro e o proprio erro\n",
           bdx, bdy, mad(&ref_src, &ref_rec), mad(&cur, &ref_src), open, closed,
           100.0 * (closed - open) / closed);
    return 0;
}
