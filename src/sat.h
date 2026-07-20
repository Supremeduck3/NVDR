#ifndef SAT_H
#define SAT_H

#include "image_io.h"
#include <stdint.h>

/*
 * sat.h — Summed Area Table (Integral Image)
 *
 * Pre-computes prefix sums over R, G, B channels (and their squares)
 * so that any rectangular region's average color and variance can be
 * queried in O(1) instead of O(W×H).
 *
 * Memory: 6 × W × H × 8 bytes (int64_t).
 * For a 4K image (3840×2160): ~240 MB. Acceptable for desktop use.
 * For larger images, fall back to direct pixel iteration.
 */

typedef struct SAT_tag {
    int64_t* sum_r;    // prefix sum of R
    int64_t* sum_g;    // prefix sum of G
    int64_t* sum_b;    // prefix sum of B
    int width, height;
} SAT;

// Build all 6 integral images from the source image.
// Returns 0 on success, -1 on allocation failure.
int  sat_build(SAT* sat, const Image* img);

// Free all SAT buffers.
void sat_free(SAT* sat);

// O(1) average color of rectangle [x, y, x+w, y+h).
// Clamped to image bounds internally.
void sat_avg(const SAT* sat, int x, int y, int w, int h,
             unsigned char* r, unsigned char* g, unsigned char* b);

#endif /* SAT_H */
