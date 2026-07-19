#include "svbc_writer.h"
#include "color.h"
#include "color_hash.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cmp_node_area_desc(const void* a, const void* b) {
    const SVBC_Node* na = a;
    const SVBC_Node* nb = b;
    int area_a = na->w * na->h;
    int area_b = nb->w * nb->h;
    return area_b - area_a;
}

// Removed vq_snap_anchor due to extreme loss of visual fidelity

typedef struct {
    unsigned char r, g, b;
    int weight;
} CWeight;

int svbc_write(const char* path,
               const QuadTree* qt,
               int img_width, int img_height) {

    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    // 1. Gather leaves
    uint32_t leaf_count = 0;
    for (int i = 0; i < qt->count; i++) {
        if (qt->nodes[i].is_leaf) leaf_count++;
    }
    
    if (leaf_count == 0) {
        fclose(f);
        return 0;
    }

    SVBC_Node* flat_nodes = (SVBC_Node*)malloc(leaf_count * sizeof(SVBC_Node));
    if (!flat_nodes) { fclose(f); return -1; }
    
    int n_idx = 0;
    // B10: calloc instead of malloc so the unused weight field of
    // any pre-allocated slot is defined and won't be read uninitialized
    // when codebook assembly picks a slot before this loop touches it.
    int cap = 4096;
    CWeight* cw = calloc((size_t)cap, sizeof(CWeight));
    if (!cw) {
        free(flat_nodes);
        fclose(f);
        return -1;
    }
    int unique_count = 0;

    // Hash table for O(1) color dedup
    ColorHashTable cht;
    cht_init(&cht, cap);

    // Prepare structural parsing and extract global frequency/diversity distribution
    for (int i = 0; i < qt->count; i++) {
        const QuadNode* src = &qt->nodes[i];
        if (!src->is_leaf) continue;
        
        flat_nodes[n_idx].x = (uint16_t)src->x;
        flat_nodes[n_idx].y = (uint16_t)src->y;
        flat_nodes[n_idx].w = (uint16_t)src->w;
        flat_nodes[n_idx].h = (uint16_t)src->h;
        flat_nodes[n_idx].layer_id = (uint8_t)src->layer_id;
        
        // Record structural color mass to define codebook
        int area = src->w * src->h;
        int idx;
        if (cht_find_or_insert(&cht, src->avg_r, src->avg_g, src->avg_b, unique_count, &idx)) {
            cw[idx].weight += area;
        } else {
            if (unique_count >= cap) {
                cap *= 2;
                CWeight* grown = realloc(cw, cap * sizeof(CWeight));
                if (!grown) {
                    free(cw);
                    free(flat_nodes);
                    cht_free(&cht);
                    fclose(f);
                    return -1;
                }
                cw = grown;
            }
            cw[unique_count].r = src->avg_r;
            cw[unique_count].g = src->avg_g;
            cw[unique_count].b = src->avg_b;
            cw[unique_count].weight = area;
            unique_count++;
        }
        n_idx++;
    }
    cht_free(&cht);

    // 2. Build SVBC Universal Codebook (Fluid Protocol up to 65535 colors)
    int max_colors = 65535; // Bumped to 65535 max to preserve Ultra fidelity
    int max_palette = unique_count > max_colors ? max_colors : unique_count;
    SVBC_Color* palette = malloc(max_palette * sizeof(SVBC_Color));
    if (!palette) {
        free(cw);
        free(flat_nodes);
        fclose(f);
        return -1;
    }
    int palette_size = 0;

    // Construir codebook em espaço YCbCr: escolhe as cores mais pesadas,
    // mas todas as comparações subsequentes (assignment) serão feitas em YCbCr.
    if (unique_count <= max_palette) {
        for (int i = 0; i < unique_count; i++) {
            palette[i].r = cw[i].r;
            palette[i].g = cw[i].g;
            palette[i].b = cw[i].b;
            palette_size++;
        }
    } else {
        for (int i = 0; i < max_palette; i++) {
            int best_idx = -1;
            int max_w = -1;
            for (int j = 0; j < unique_count; j++) {
                if (cw[j].weight > max_w) {
                    max_w = cw[j].weight;
                    best_idx = j;
                }
            }
            if (best_idx == -1) break;
            palette[i].r = cw[best_idx].r;
            palette[i].g = cw[best_idx].g;
            palette[i].b = cw[best_idx].b;
            cw[best_idx].weight = -1;
            palette_size++;
        }
    }
    
    // 3. Assign Index/Tokens nativamente em espaço perceptual YCbCr
    n_idx = 0;
    for (int i = 0; i < qt->count; i++) {
        const QuadNode* src = &qt->nodes[i];
        if (!src->is_leaf) continue;
        
        long min_dist = 255L*255L*10L;
        int best_p = 0;

        int y_src, cb_src, cr_src;
        rgb_to_ycbcr(src->avg_r, src->avg_g, src->avg_b, &y_src, &cb_src, &cr_src);

        for (int p=0; p<palette_size; p++) {
            int y_p, cb_p, cr_p;
            rgb_to_ycbcr(palette[p].r, palette[p].g, palette[p].b, &y_p, &cb_p, &cr_p);

            int dY  = y_src  - y_p;
            int dCb = cb_src - cb_p;
            int dCr = cr_src - cr_p;
            long dist = labs(dY) * 7L + labs(dCb) * 2L + labs(dCr) * 2L;
            if (dist < min_dist) {
                min_dist = dist;
                best_p = p;
                if (dist == 0) break; // Exact match shortcut, O(1) mostly
            }
        }
        flat_nodes[n_idx].token_id = (uint16_t)best_p;
        n_idx++;
    }

    // 4. Sort for PRS Fractal Parsing (Largest physical objects first)
    qsort(flat_nodes, leaf_count, sizeof(SVBC_Node), cmp_node_area_desc);

    // 5. Build Header and Flush payload physically
    SVBC_Header hdr;
    memcpy(hdr.magic, "SVBC", 4);
    hdr.version    = 3;
    hdr._pad[0]    = 0;
    hdr.codebook_count = (uint16_t)palette_size;
    hdr.img_width  = (uint16_t)img_width;
    hdr.img_height = (uint16_t)img_height;
    hdr.node_count = leaf_count;
    hdr._reserved  = 0;

    fwrite(&hdr, sizeof(SVBC_Header), 1, f);
    fwrite(palette, sizeof(SVBC_Color), palette_size, f);
    fwrite(flat_nodes, sizeof(SVBC_Node), leaf_count, f);

    fclose(f);
    free(cw);
    free(palette);
    free(flat_nodes);
    return 0;
}
