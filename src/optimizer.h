#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "quadtree.h"

// Coalesces sibling leaf nodes with similar colors back into parent
// color_threshold: max sum of absolute RGB diffs to merge (e.g., 12)
void optimizer_coalesce(QuadTree* qt, int node_idx, int color_threshold);

// Quantizes all leaf node colors to reduce unique color count
// step: quantization step (e.g., 4 rounds to nearest multiple of 4)
void optimizer_quantize(QuadTree* qt, int step);

// Returns the number of leaf nodes in the subtree
int optimizer_count_leaves(const QuadTree* qt, int node_idx);

// Sub-pass B: iLUT Pareto-weighted Palette (N=256)
void optimizer_apply_ilut(QuadTree* qt, int max_colors);

// Fluid System W_i: Drops noisy frequency leaves
void optimizer_apply_importance_cull(QuadTree* qt, float w_i_threshold);

// Rectangular Coalescing (v0.14 Unified Engine):
// Merges adjacent same-color leaf nodes into wider/taller rectangles
// Eliminates fragmented edges and reduces SVG element count
void optimizer_rect_coalesce(QuadTree* qt, int color_threshold);

// Border Blend: detects adjacent leaves with color difference and
// sets SVBC_BLEND_RIGHT / SVBC_BLEND_BELOW flags on layer_id.
// Must be called AFTER rect_coalesce (uses final leaf geometry).
void optimizer_detect_blends(QuadTree* qt);

#endif /* OPTIMIZER_H */
