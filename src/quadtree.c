#include "quadtree.h"
#include "color.h"
#include "sat.h"
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int quadtree_init(QuadTree* qt, int max_nodes) {
    qt->nodes = (QuadNode*)malloc(sizeof(QuadNode) * (size_t)max_nodes);
    if (!qt->nodes) return -1;
    qt->count    = 0;
    qt->capacity = max_nodes;
    return 0;
}

void quadtree_free(QuadTree* qt) {
    if (qt->nodes) {
        free(qt->nodes);
        qt->nodes = NULL;
    }
    qt->count    = 0;
    qt->capacity = 0;
}

// Aloca um nó do pool, retorna índice ou -1 se cheio
static int alloc_node(QuadTree* qt) {
    int idx;
#pragma omp atomic capture
    {
        idx = qt->count;
        qt->count++;
    }
    if (idx >= qt->capacity) {
#pragma omp atomic update
        qt->count--; // Rollback using update to decrement atomically
        return -1;
    }
    memset(&qt->nodes[idx], 0, sizeof(QuadNode));
    qt->nodes[idx].children[0] = -1;
    qt->nodes[idx].children[1] = -1;
    qt->nodes[idx].children[2] = -1;
    qt->nodes[idx].children[3] = -1;
    return idx;
}

static int psq_layer(int y, int h, int img_height) {
    int cy = y + h / 2;
    float ratio = (float)cy / (float)img_height;
    if (ratio < 0.3f)  return 0; // LAYER_SKY
    if (ratio < 0.7f)  return 1; // LAYER_MIDGROUND
    return 2; // LAYER_FOREGROUND
}

static float psq_threshold(const SVGConfig* cfg, int layer_id) {
    if (layer_id == 0) return cfg->homo_threshold * cfg->psq_sky_mult;
    if (layer_id == 1) return cfg->homo_threshold * cfg->psq_midground_mult;
    return cfg->homo_threshold;
}

static int build_recursive(QuadTree* qt, const Image* img, const SAT* sat,
                           const SVGConfig* cfg,
                           int x, int y, int w, int h, int depth) {
    int idx = alloc_node(qt);
    if (idx < 0) return -1;

    QuadNode* node = &qt->nodes[idx];
    node->x     = x;
    node->y     = y;
    node->w     = w;
    node->h     = h;
    node->depth = depth;
    node->layer_id = psq_layer(y, h, img->height);

    float local_thresh = psq_threshold(cfg, node->layer_id);

    // Calcula cor média da região
    sat_avg(sat, x, y, w, h, &node->avg_r, &node->avg_g, &node->avg_b);

    // Edge Snapping: at leaf-size regions with high contrast edges,
    // use dominant side color instead of blurred average
    if (w <= cfg->min_tile_size * 4 && h <= cfg->min_tile_size * 4) {
        color_edge_snap(img, x, y, w, h,
                        &node->avg_r, &node->avg_g, &node->avg_b, 40);
    }

    // Calcula homogeneidade (pixel-level YCbCr — visually calibrated)
    node->homogeneity = color_homogeneity(img, x, y, w, h,
                                          node->avg_r, node->avg_g, node->avg_b);
                                          
    if (node->homogeneity > local_thresh &&
        w > cfg->min_tile_size &&
        h > cfg->min_tile_size &&
        depth < cfg->max_depth) {

        int hw = w / 2;
        int hh = h / 2;
        int rw = w - hw;   // metade direita (trata largura ímpar)
        int rh = h - hh;   // metade inferior (trata altura ímpar)

        node->is_leaf = 0;

        int c0, c1, c2, c3;
        // OpenMP Task-based parallel recursion. depth<5 limits task explosion.
#pragma omp task shared(qt, c0) if(depth < 5)
        c0 = build_recursive(qt, img, sat, cfg, x,      y,      hw, hh, depth + 1);
#pragma omp task shared(qt, c1) if(depth < 5)
        c1 = build_recursive(qt, img, sat, cfg, x + hw, y,      rw, hh, depth + 1);
#pragma omp task shared(qt, c2) if(depth < 5)
        c2 = build_recursive(qt, img, sat, cfg, x,      y + hh, hw, rh, depth + 1);
#pragma omp task shared(qt, c3) if(depth < 5)
        c3 = build_recursive(qt, img, sat, cfg, x + hw, y + hh, rw, rh, depth + 1);

#pragma omp taskwait

        // Re-acquire pointer after wait sync
        node = &qt->nodes[idx];
        node->children[0] = c0;
        node->children[1] = c1;
        node->children[2] = c2;
        node->children[3] = c3;

        // Re-acquire pointer (pool não faz realloc, mas boa prática)
        node = &qt->nodes[idx];

        // Verifica se todas alocações foram bem sucedidas
        for (int i = 0; i < 4; i++) {
            if (node->children[i] < 0) return -1;
        }

        // Unified Engine (v0.14 Section 1): Early coalescing during build
        // If all 4 children are leaves with similar colors, collapse immediately
        int all_leaves = 1;
        for (int i = 0; i < 4; i++) {
            if (!qt->nodes[node->children[i]].is_leaf) { all_leaves = 0; break; }
        }
        if (all_leaves) {
            QuadNode* c0 = &qt->nodes[node->children[0]];
            int can_merge = 1;
            for (int i = 1; i < 4; i++) {
                QuadNode* ci = &qt->nodes[node->children[i]];
                int dr = abs((int)ci->avg_r - (int)c0->avg_r);
                int dg = abs((int)ci->avg_g - (int)c0->avg_g);
                int db = abs((int)ci->avg_b - (int)c0->avg_b);
                if (dr + dg + db > 12) { can_merge = 0; break; }
            }
            if (can_merge) {
                node->is_leaf = 1;
                for (int i = 0; i < 4; i++) node->children[i] = -1;
            }
        }
    } else {
        node->is_leaf = 1;
    }

    return idx;
}

int quadtree_build(QuadTree* qt, const Image* img, const SAT* sat,
                   const SVGConfig* cfg) {
    int root = -1;
#pragma omp parallel
    {
#pragma omp single
        {
            root = build_recursive(qt, img, sat, cfg, 0, 0, img->width, img->height, 0);
        }
    }
    return root;
}
