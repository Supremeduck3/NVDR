/*
 * What reusing the previous frame would be worth.
 *
 * The tree stops descending wherever a region is within tolerance of its
 * own mean colour. For video the rule gains a second way to stop: the
 * region may already be within tolerance of what the previous frame is
 * showing there, in which case nothing needs to be sent for it at all —
 * not a colour, not a subtree, not a split bit beyond the one that says
 * "copy".
 *
 * So this is not an estimate. Walking the tree with that rule produces
 * exactly the leaf count the scheme would produce, against the leaf count
 * the encoder produces today. The predictor is the previous frame
 * *decoded*, not its source, because a decoder only ever has the former.
 *
 * No motion compensation here. Every gain reported is what a fixed
 * co-located reference buys; searching for motion can only raise it, and
 * costs vectors this does not pay for.
 */
#include "nvdr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const NvdrImage* g_cur;
static const NvdrImage* g_ref;
static float g_weber, g_pivot, g_tol;
static long n_intra, n_inter, n_skip, px_skip, px_total;

/* The same perceptual metric the tree uses, but measured against a fixed
 * reference image instead of against the region's own mean. */
static double deviation_from_ref(int x, int y, int w, int h) {
    int x1 = x + w > g_cur->width ? g_cur->width : x + w;
    int y1 = y + h > g_cur->height ? g_cur->height : y + h;
    if (x1 <= x || y1 <= y) return 0.0;
    double acc = 0.0, luma_acc = 0.0;
    long n = 0;
    for (int py = y; py < y1; py++) {
        const unsigned char* a = g_cur->pixels + ((size_t)py * g_cur->width + x) * 3;
        const unsigned char* b = g_ref->pixels + ((size_t)py * g_ref->width + x) * 3;
        for (int px = x; px < x1; px++, a += 3, b += 3) {
            int dr = a[0]-b[0], dg = a[1]-b[1], db = a[2]-b[2];
            if (dr < 0) dr = -dr; if (dg < 0) dg = -dg; if (db < 0) db = -db;
            acc += 0.30*dr + 0.59*dg + 0.11*db;
            luma_acc += 0.30*b[0] + 0.59*b[1] + 0.11*b[2];
            n++;
        }
    }
    double luma = luma_acc / n;
    return (acc / n) / (255.0 * (luma + g_weber) / (g_pivot + g_weber));
}

/* Today's rule: descend while the region strays from its own mean. */
static void walk_intra(const NvdrTree* t, int32_t idx) {
    const NvdrNode* n = &t->nodes[idx];
    if (n->first_child < 0 || n->deviation <= g_tol) { n_intra++; return; }
    for (int i = 0; i < 4; i++) walk_intra(t, n->first_child + i);
}

/* With a reference: a region already explained by the previous frame ends
 * the descent, whatever its own deviation says. */
static void walk_inter(const NvdrTree* t, int32_t idx) {
    const NvdrNode* n = &t->nodes[idx];
    if (deviation_from_ref(n->x, n->y, n->w, n->h) <= g_tol) {
        n_inter++; n_skip++; px_skip += (long)n->w * n->h;
        return;
    }
    if (n->first_child < 0 || n->deviation <= g_tol) { n_inter++; return; }
    for (int i = 0; i < 4; i++) walk_inter(t, n->first_child + i);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <prev.ppm> <cur.ppm>\n", argv[0]); return 2; }

    NvdrConfig cfg = nvdr_default_config();
    g_tol = cfg.tolerance[NVDR_LEVELS - 1];
    g_weber = cfg.weber;

    /* The predictor is the previous frame as a decoder would hold it. */
    NvdrImage prev_src;
    if (nvdr_image_load(&prev_src, argv[1]) != 0) return 1;
    NvdrHeader hdr;
    if (nvdr_encode_file("/tmp/_if_prev.nvdr", &prev_src, &cfg, &hdr) != 0) return 1;
    NvdrPyramid pyr;
    if (nvdr_decode_file("/tmp/_if_prev.nvdr", &pyr, &hdr) != 0) return 1;
    NvdrImage ref;
    ref.width = prev_src.width; ref.height = prev_src.height;
    ref.pixels = (unsigned char*)calloc((size_t)ref.width * ref.height * 3, 1);
    for (int k = 0; k < pyr.levels_present; k++) nvdr_render_level(&pyr.level[k], &ref);

    NvdrImage cur;
    if (nvdr_image_load(&cur, argv[2]) != 0) return 1;
    g_cur = &cur; g_ref = &ref;

    /* The tree's Weber pivot is the image's own mean luminance. */
    double sum = 0.0;
    long np = (long)cur.width * cur.height;
    for (long i = 0; i < np; i++) {
        const unsigned char* p = cur.pixels + i * 3;
        sum += 0.30*p[0] + 0.59*p[1] + 0.11*p[2];
    }
    g_pivot = (float)(sum / np);
    px_total = np;

    NvdrTree tree;
    if (nvdr_tree_build(&tree, &cur, &cfg) != 0) return 1;
    walk_intra(&tree, 0);
    walk_inter(&tree, 0);

    printf("%-22s intra %7ld folhas | inter %7ld (-%4.1f%%) | "
           "%ld copiadas cobrindo %4.1f%% da area | prev->cur psnr %5.2f dB\n",
           argv[2], n_intra, n_inter,
           n_intra ? 100.0 * (n_intra - n_inter) / n_intra : 0.0,
           n_skip, 100.0 * px_skip / px_total, nvdr_psnr(&cur, &ref));
    return 0;
}
