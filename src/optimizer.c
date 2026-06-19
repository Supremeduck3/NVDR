#include "optimizer.h"
#include "color_hash.h"
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void optimizer_coalesce(QuadTree* qt, int node_idx, int color_threshold) {
    if (node_idx < 0) return;
    QuadNode* node = &qt->nodes[node_idx];
    if (node->is_leaf) return;

    // Recurse into children first (bottom-up)
    for (int i = 0; i < 4; i++) {
        optimizer_coalesce(qt, node->children[i], color_threshold);
    }

    // Check if all children are now leaves
    for (int i = 0; i < 4; i++) {
        if (node->children[i] < 0 || !qt->nodes[node->children[i]].is_leaf) {
            return;
        }
    }

    // Check color similarity against the first child
    QuadNode* first_child = &qt->nodes[node->children[0]];
    for (int i = 1; i < 4; i++) {
        QuadNode* child = &qt->nodes[node->children[i]];
        int diff = abs((int)child->avg_r - (int)first_child->avg_r) +
                   abs((int)child->avg_g - (int)first_child->avg_g) +
                   abs((int)child->avg_b - (int)first_child->avg_b);
        if (diff > color_threshold) return;
    }

    // All children are similar enough — merge back into parent leaf
    node->is_leaf = 1;
    node->avg_r = first_child->avg_r;
    node->avg_g = first_child->avg_g;
    node->avg_b = first_child->avg_b;
    for (int i = 0; i < 4; i++) {
        node->children[i] = -1;
    }
}

void optimizer_quantize(QuadTree* qt, int step) {
    if (step < 2) return;
#pragma omp parallel for
    for (int i = 0; i < qt->count; i++) {
        if (!qt->nodes[i].is_leaf) continue;
        int r = qt->nodes[i].avg_r;
        int g = qt->nodes[i].avg_g;
        int b = qt->nodes[i].avg_b;
        
        // Luminance approximation
        int lum = (299 * r + 587 * g + 114 * b) / 1000;

        // Zero-Threshold Clamp (Mura effect / pure black crush)
        if (lum < 5) {
            r = 0; g = 0; b = 0;
        } else {
            // Perceptual Shadow Quantization (PSQ)
            int effective_step = step;
            if (lum < 26) {
                effective_step = 2; // Finer step for heavy shadows (0-10%)
            } else if (lum < 64) {
                effective_step = 4; // Finer step for mid-shadows
            }
            
            r = ((r + effective_step / 2) / effective_step) * effective_step;
            g = ((g + effective_step / 2) / effective_step) * effective_step;
            b = ((b + effective_step / 2) / effective_step) * effective_step;
        }
        
        if (r > 255) r = 255;
        if (g > 255) g = 255;
        if (b > 255) b = 255;
        
        qt->nodes[i].avg_r = (unsigned char)r;
        qt->nodes[i].avg_g = (unsigned char)g;
        qt->nodes[i].avg_b = (unsigned char)b;
    }
}

static int count_recursive(const QuadTree* qt, int node_idx) {
    if (node_idx < 0) return 0;
    const QuadNode* node = &qt->nodes[node_idx];
    if (node->is_leaf) return 1;
    int count = 0;
    for (int i = 0; i < 4; i++) {
        count += count_recursive(qt, node->children[i]);
    }
    return count;
}

int optimizer_count_leaves(const QuadTree* qt, int node_idx) {
    return count_recursive(qt, node_idx);
}

typedef struct {
    unsigned char r, g, b;
    int weight;
} ColorWeight;

void optimizer_apply_ilut(QuadTree* qt, int max_colors) {
    int cap = 4096;
    ColorWeight* cw = malloc(cap * sizeof(ColorWeight));
    if (!cw) return;
    int unique_count = 0;

    // Hash table for O(1) color dedup
    ColorHashTable cht;
    cht_init(&cht, cap);
    
    // Pass 1: Gather area-weighted colors (O(N) with hash table)
    for (int i = 0; i < qt->count; i++) {
        if (!qt->nodes[i].is_leaf) continue;
        QuadNode* n = &qt->nodes[i];
        int area = n->w * n->h;
        int idx;
        if (cht_find_or_insert(&cht, n->avg_r, n->avg_g, n->avg_b, unique_count, &idx)) {
            // Found existing color
            cw[idx].weight += area;
        } else {
            // New color inserted
            if (unique_count >= cap) {
                cap *= 2;
                ColorWeight* grown = realloc(cw, cap * sizeof(ColorWeight));
                if (!grown) {
                    free(cw);
                    cht_free(&cht);
                    return;
                }
                cw = grown;
            }
            cw[unique_count].r = n->avg_r;
            cw[unique_count].g = n->avg_g;
            cw[unique_count].b = n->avg_b;
            cw[unique_count].weight = area;
            unique_count++;
        }
    }
    cht_free(&cht);
    
    // Pass 2: Select N=256 palette using Pareto-weighted Perceptual Salience (Frequency * Perceptual Distance)
    int palette_size = 0;
    int max_palette = unique_count > max_colors ? max_colors : unique_count;
    ColorWeight* palette = malloc(max_palette * sizeof(ColorWeight));
    if (!palette) {
        free(cw);
        return;
    }
    
    // Select the absolute most frequent color as the first anchor
    int best_idx = 0;
    for(int i=1; i<unique_count; i++) {
        if(cw[i].weight > cw[best_idx].weight) best_idx = i;
    }
    palette[palette_size++] = cw[best_idx];
    cw[best_idx].weight = -1; // mark as used
    
    // Greedily select colors that maximize (weight * distance_to_closest_selected)
    while(palette_size < max_palette) {
        int highest_salience_idx = -1;
        long long max_salience = -1;
        
        for(int i=0; i<unique_count; i++) {
            if(cw[i].weight < 0) continue;
            
            // Find distance to closest color already in palette
            long min_dist = 255*255*10;
            for(int p=0; p<palette_size; p++) {
                int dr = cw[i].r - palette[p].r;
                int dg = cw[i].g - palette[p].g;
                int db = cw[i].b - palette[p].b;
                int dist = dr*dr*3 + dg*dg*4 + db*db*2;
                if(dist < min_dist) min_dist = dist;
            }
            
            long long salience = (long long)cw[i].weight * min_dist;
            if(salience > max_salience) {
                max_salience = salience;
                highest_salience_idx = i;
            }
        }
        
        if (highest_salience_idx == -1) break;
        palette[palette_size++] = cw[highest_salience_idx];
        cw[highest_salience_idx].weight = -1;
    }
    
    // Pass 3: Map all nodes to nearest neighbor in generated iLUT Palette
#pragma omp parallel for
    for (int i = 0; i < qt->count; i++) {
        if (!qt->nodes[i].is_leaf) continue;
        QuadNode* n = &qt->nodes[i];
        
        int best_r = palette[0].r, best_g = palette[0].g, best_b = palette[0].b;
        long min_dist = 255*255*10;
        
        for (int p = 0; p < palette_size; p++) {
            int dr = n->avg_r - palette[p].r;
            int dg = n->avg_g - palette[p].g;
            int db = n->avg_b - palette[p].b;
            int dist = dr*dr*3 + dg*dg*4 + db*db*2; // Weighted perceptual distance
            if (dist < min_dist) {
                min_dist = dist;
                best_r = palette[p].r;
                best_g = palette[p].g;
                best_b = palette[p].b;
            }
        }
        
        n->avg_r = best_r;
        n->avg_g = best_g;
        n->avg_b = best_b;
    }
    free(palette);
    free(cw);
}

static void clear_is_leaf(QuadTree* qt, int node_idx) {
    if (node_idx == -1) return;
    QuadNode* n = &qt->nodes[node_idx];
    n->is_leaf = 0;
    for (int i = 0; i < 4; i++) {
        if (n->children[i] != -1) {
            clear_is_leaf(qt, n->children[i]);
        }
    }
}

// Fluid System W_i Culling
static int cull_recursive(QuadTree* qt, int node_idx, float base_contrast) {
    QuadNode* n = &qt->nodes[node_idx];
    if (n->is_leaf) return 1;

    int all_leaves = 1;
    for (int i = 0; i < 4; i++) {
        if (!cull_recursive(qt, n->children[i], base_contrast)) {
            all_leaves = 0;
        }
    }

    if (all_leaves) {
        // Evaluate perceptual distance of children to parent average
        long max_dist = 0;
        for (int i = 0; i < 4; i++) {
            QuadNode* child = &qt->nodes[n->children[i]];
            long dr = child->avg_r - n->avg_r;
            long dg = child->avg_g - n->avg_g;
            long db = child->avg_b - n->avg_b;
            long dist = dr*dr*3 + dg*dg*4 + db*db*2;
            if (dist > max_dist) max_dist = dist;
        }

        // PSQ Multipliers - higher tolerance for distant layers
        float layer_mult = 1.0f;
        if (n->layer_id == 0) layer_mult = 4.0f;
        else if (n->layer_id == 1) layer_mult = 2.0f;

        // Fluid weighting: smaller nodes tolerate higher destruction
        float dynamic_limit = base_contrast * layer_mult;
        if (n->w <= 4) dynamic_limit *= 4.0f;
        else if (n->w <= 8) dynamic_limit *= 2.0f;
        
        long threshold_sq = (long)(dynamic_limit * dynamic_limit);
        
        if (max_dist <= threshold_sq) {
            for (int i = 0; i < 4; i++) {
                clear_is_leaf(qt, n->children[i]);
            }
            n->is_leaf = 1; // Cull details, merge into larger anchor block
            return 1;
        }
    }
    return 0;
}

void optimizer_apply_importance_cull(QuadTree* qt, float w_i_threshold) {
    cull_recursive(qt, 0, w_i_threshold);
}

// --- Rectangular Coalescing (v0.14 Unified Engine) ---
// Sort comparator: by y, then x
static QuadNode* s_rc_nodes = NULL; // module-local context for qsort comparators

static int cmp_leaf_yx(const void* a, const void* b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    QuadNode* na = &s_rc_nodes[ia];
    QuadNode* nb = &s_rc_nodes[ib];
    if (na->y != nb->y) return na->y - nb->y;
    return na->x - nb->x;
}

// Sort comparator: by x, then y (for vertical merge)
static int cmp_leaf_xy(const void* a, const void* b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    QuadNode* na = &s_rc_nodes[ia];
    QuadNode* nb = &s_rc_nodes[ib];
    if (na->x != nb->x) return na->x - nb->x;
    return na->y - nb->y;
}

void optimizer_rect_coalesce(QuadTree* qt, int color_threshold) {
    // Collect all leaf indices
    int leaf_cap = 4096;
    int* leaves = (int*)malloc(leaf_cap * sizeof(int));
    if (!leaves) return;
    int leaf_count = 0;

    for (int i = 0; i < qt->count; i++) {
        if (!qt->nodes[i].is_leaf) continue;
        if (leaf_count >= leaf_cap) {
            leaf_cap *= 2;
            int* grown = (int*)realloc(leaves, leaf_cap * sizeof(int));
            if (!grown) {
                free(leaves);
                return;
            }
            leaves = grown;
        }
        leaves[leaf_count++] = i;
    }

    if (leaf_count < 2) { free(leaves); return; }

    // Sort leaves by (y, x) for horizontal merge pass
    s_rc_nodes = qt->nodes;
    qsort(leaves, leaf_count, sizeof(int), cmp_leaf_yx);

    // Horizontal merge: same y, same h, same color, x1+w1==x2
    for (int i = 0; i < leaf_count - 1; i++) {
        QuadNode* a = &qt->nodes[leaves[i]];
        if (!a->is_leaf) continue;

        for (int j = i + 1; j < leaf_count; j++) {
            QuadNode* b = &qt->nodes[leaves[j]];
            if (!b->is_leaf) continue;
            if (b->y != a->y) break;           // different row
            if (b->x != a->x + a->w) continue; // not adjacent
            if (b->h != a->h) continue;         // different height

            int matched = 0;
            if (color_threshold == 0) {
                matched = (a->avg_r == b->avg_r && a->avg_g == b->avg_g && a->avg_b == b->avg_b);
            } else {
                // PSQ: relax color threshold for sky/midground to merge large patches
                int layer_mult = 1;
                if (a->layer_id == 0) layer_mult = 3;
                else if (a->layer_id == 1) layer_mult = 2; // integer approx for 1.5
                
                int thresh = color_threshold * layer_mult;
                matched = (abs((int)a->avg_r - (int)b->avg_r) <= thresh &&
                           abs((int)a->avg_g - (int)b->avg_g) <= thresh &&
                           abs((int)a->avg_b - (int)b->avg_b) <= thresh);
            }
            if (!matched) continue;

            // Merge b into a
            a->w += b->w;
            b->is_leaf = 0; // consumed
        }
    }

    // Re-collect surviving leaves for vertical pass
    leaf_count = 0;
    for (int i = 0; i < qt->count; i++) {
        if (!qt->nodes[i].is_leaf) continue;
        if (leaf_count >= leaf_cap) {
            leaf_cap *= 2;
            int* grown = (int*)realloc(leaves, leaf_cap * sizeof(int));
            if (!grown) {
                free(leaves);
                return;
            }
            leaves = grown;
        }
        leaves[leaf_count++] = i;
    }

    if (leaf_count < 2) { free(leaves); return; }

    // Sort by (x, y) for vertical merge using O(n log n) qsort
    qsort(leaves, leaf_count, sizeof(int), cmp_leaf_xy);

    // Vertical merge: same x, same w, same color, y1+h1==y2
    for (int i = 0; i < leaf_count - 1; i++) {
        QuadNode* a = &qt->nodes[leaves[i]];
        if (!a->is_leaf) continue;

        for (int j = i + 1; j < leaf_count; j++) {
            QuadNode* b = &qt->nodes[leaves[j]];
            if (!b->is_leaf) continue;
            if (b->x != a->x) break;           // different column
            if (b->y != a->y + a->h) continue; // not adjacent
            if (b->w != a->w) continue;         // different width

            int matched = 0;
            if (color_threshold == 0) {
                matched = (a->avg_r == b->avg_r && a->avg_g == b->avg_g && a->avg_b == b->avg_b);
            } else {
                int layer_mult = 1;
                if (a->layer_id == 0) layer_mult = 3;
                else if (a->layer_id == 1) layer_mult = 2; // integer approx for 1.5
                
                int thresh = color_threshold * layer_mult;
                matched = (abs((int)a->avg_r - (int)b->avg_r) <= thresh &&
                           abs((int)a->avg_g - (int)b->avg_g) <= thresh &&
                           abs((int)a->avg_b - (int)b->avg_b) <= thresh);
            }
            if (!matched) continue;

            // Merge b into a
            a->h += b->h;
            b->is_leaf = 0; // consumed
        }
    }

    free(leaves);
}

// --- Border Blend Detection (O(N log N)) ---
// Detects adjacent leaf pairs with visible color difference and sets
// SVBC_BLEND_RIGHT / SVBC_BLEND_BELOW flags on their layer_id.
#define BLEND_COLOR_THRESHOLD 12  // min RGB diff to trigger blend (of 255)

void optimizer_detect_blends(QuadTree* qt) {
    // Collect leaves
    int leaf_cap = 4096;
    int* leaves = (int*)malloc(leaf_cap * sizeof(int));
    if (!leaves) return;
    int leaf_count = 0;

    for (int i = 0; i < qt->count; i++) {
        if (!qt->nodes[i].is_leaf) continue;
        if (leaf_count >= leaf_cap) {
            leaf_cap *= 2;
            int* grown = (int*)realloc(leaves, leaf_cap * sizeof(int));
            if (!grown) { free(leaves); return; }
            leaves = grown;
        }
        leaves[leaf_count++] = i;
    }

    if (leaf_count < 2) { free(leaves); return; }

    // Sort by (y, x) for horizontal neighbor detection
    s_rc_nodes = qt->nodes;
    qsort(leaves, leaf_count, sizeof(int), cmp_leaf_yx);

    // Detect right neighbors: for each leaf, find one at (x+w, y) with same h
    for (int i = 0; i < leaf_count; i++) {
        QuadNode* a = &qt->nodes[leaves[i]];
        // Scan forward for right neighbor (sorted by y,x so it's close)
        for (int j = i + 1; j < leaf_count; j++) {
            QuadNode* b = &qt->nodes[leaves[j]];
            if (b->y > a->y) break;              // past this row
            if (b->y != a->y) continue;
            if (b->x != a->x + a->w) continue;   // not adjacent right
            // Check color difference
            int diff = abs((int)a->avg_r - (int)b->avg_r)
                     + abs((int)a->avg_g - (int)b->avg_g)
                     + abs((int)a->avg_b - (int)b->avg_b);
            if (diff > BLEND_COLOR_THRESHOLD) {
                a->layer_id |= 0x40;  // SVBC_BLEND_RIGHT
            }
            break; // only first right neighbor
        }
    }

    // Re-sort by (x, y) for vertical neighbor detection
    qsort(leaves, leaf_count, sizeof(int), cmp_leaf_xy);

    for (int i = 0; i < leaf_count; i++) {
        QuadNode* a = &qt->nodes[leaves[i]];
        for (int j = i + 1; j < leaf_count; j++) {
            QuadNode* b = &qt->nodes[leaves[j]];
            if (b->x > a->x) break;              // past this column
            if (b->x != a->x) continue;
            if (b->y != a->y + a->h) continue;   // not adjacent below
            int diff = abs((int)a->avg_r - (int)b->avg_r)
                     + abs((int)a->avg_g - (int)b->avg_g)
                     + abs((int)a->avg_b - (int)b->avg_b);
            if (diff > BLEND_COLOR_THRESHOLD) {
                a->layer_id |= 0x80;  // SVBC_BLEND_BELOW
            }
            break;
        }
    }

    free(leaves);
}

