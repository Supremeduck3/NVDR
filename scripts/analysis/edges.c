/*
 * How much of a level is boundary.
 *
 * A leaf stops for one of two reasons: its region is uniform enough, or it
 * ran out of room to split. The second kind is a staircase step across an
 * edge the axis-aligned grid cannot follow, and it is exactly what a
 * primitive that could cut along a line would replace. Counting them
 * bounds what any such primitive can be worth.
 */
#include "nvdr.h"
#include <stdio.h>
#include <stdlib.h>

static long forced, uniform, forced_px, uniform_px;
static double forced_dev;

static void walk(const NvdrTree* t, int32_t idx, float tol, int min_tile) {
    const NvdrNode* n = &t->nodes[idx];
    if (n->first_child >= 0 && n->deviation > tol) {
        for (int i = 0; i < 4; i++) walk(t, n->first_child + i, tol, min_tile);
        return;
    }
    long px = (long)n->w * n->h;
    if (n->deviation > tol) { forced++; forced_px += px; forced_dev += n->deviation; }
    else { uniform++; uniform_px += px; }
}

int main(int argc, char** argv) {
    NvdrImage img;
    if (argc < 2 || nvdr_image_load(&img, argv[1]) != 0) return 1;
    NvdrConfig cfg = nvdr_default_config();
    NvdrTree tree;
    if (nvdr_tree_build(&tree, &img, &cfg) != 0) return 1;
    walk(&tree, 0, cfg.tolerance[NVDR_LEVELS - 1], cfg.min_tile);
    long total = forced + uniform;
    printf("%-24s %7ld folhas | %6.2f%% sao degrau de borda "
           "(%.2f%% da area, desvio medio %.3f)\n",
           argv[1], total, total ? 100.0 * forced / total : 0.0,
           (forced_px + uniform_px) ? 100.0 * forced_px / (forced_px + uniform_px) : 0.0,
           forced ? forced_dev / forced : 0.0);
    nvdr_tree_free(&tree);
    nvdr_image_free(&img);
    return 0;
}
