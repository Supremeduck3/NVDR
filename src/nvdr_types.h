#ifndef NVDR_TYPES_H
#define NVDR_TYPES_H

#include "quadtree.h"

/*
 * nvdr_types.h — Shared pipeline types for NVDR renderer
 *
 * Central definitions used across main.c, contour.c, and future modules.
 * Avoids scattered typedefs and cross-module coupling.
 */

// GCT: Unified Palette Entry (Geometry + Color as atomic token)
typedef struct {
    unsigned char r, g, b;
    int has_gradient;
    int gradient_id;
    int freq;
    char hex_fill[16];
} PaletteEntry;

// Index list for PRS layer node references
typedef struct {
    int* node_indices;
    int count;
} IntList;

// Per-node gradient information (detected in main.c)
typedef struct {
    int has_gradient;
    int grad_dir;             // 0=vertical, 1=horizontal
    unsigned char top_r, top_g, top_b;   // top/left
    unsigned char bot_r, bot_g, bot_b;   // bottom/right
    int gradient_id;
} NodeGradient;

#endif /* NVDR_TYPES_H */
