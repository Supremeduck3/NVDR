#ifndef CONTOUR_H
#define CONTOUR_H

#include "quadtree.h"
#include "optimizer.h"
#include "nvdr_types.h"
#include <stdio.h>

/*
 * contour.h — Smooth boundary extraction and Bezier fitting
 *
 * Replaces the staircase-of-rects artifact on circles and curved shapes
 * with smooth SVG paths using:
 *   1. Edge cancellation  → extracts outer boundary polygon
 *   2. Douglas-Peucker    → removes staircase intermediate vertices
 *   3. Angle-aware Bezier → smooth curves, sharp 90° corners kept sharp
 *
 * Usage in main.c:
 *   Instead of write_layer() for the anchor layer, call
 *   contour_write_layer_smooth() which automatically selects contour
 *   paths for large curved regions and falls back to rects for small ones.
 */

/* Write a single color region as a smooth Bezier <path>.
 * nodes[]  : array of QuadNode* belonging to this color region
 * count    : number of nodes
 * css_class: CSS class string for fill (e.g. "c5")
 * Returns 1 if path was written, 0 if fallback to rects is needed. */
int contour_write_smooth_path(FILE* f, const char* css_class,
                               QuadNode** nodes, int count);

/* Drop-in replacement for write_layer() in main.c for the anchor layer.
 * Uses smooth contour paths for large regions, rects for small ones. */
void contour_write_layer_smooth(FILE* f, IntList* layer, QuadTree* qt,
                                 PaletteEntry* palette, int palette_size,
                                 int* node_to_palette);

#endif /* CONTOUR_H */
