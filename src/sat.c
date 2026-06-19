#include "sat.h"
#include "color.h"
#include <stdlib.h>
#include <math.h>

// Helper: index into (width+1) × (height+1) padded table
// Padding row 0 and col 0 with zeros avoids boundary checks.
#define IDX(sat, px, py) ((py) * ((sat)->width + 1) + (px))

int sat_build(SAT* sat, const Image* img) {
    sat->width  = img->width;
    sat->height = img->height;

    size_t table_size = (size_t)(img->width + 1) * (size_t)(img->height + 1);

    sat->sum_r  = (int64_t*)calloc(table_size, sizeof(int64_t));
    sat->sum_g  = (int64_t*)calloc(table_size, sizeof(int64_t));
    sat->sum_b  = (int64_t*)calloc(table_size, sizeof(int64_t));
    sat->sum_r2 = (int64_t*)calloc(table_size, sizeof(int64_t));
    sat->sum_g2 = (int64_t*)calloc(table_size, sizeof(int64_t));
    sat->sum_b2 = (int64_t*)calloc(table_size, sizeof(int64_t));

    if (!sat->sum_r || !sat->sum_g || !sat->sum_b ||
        !sat->sum_r2 || !sat->sum_g2 || !sat->sum_b2) {
        sat_free(sat);
        return -1;
    }

    // Build integral images.
    // SAT[y][x] = pixel(x-1,y-1) + SAT[y][x-1] + SAT[y-1][x] - SAT[y-1][x-1]
    // Tables are (W+1)×(H+1) with row 0 and col 0 = 0 (no boundary checks needed).
    for (int py = 1; py <= img->height; py++) {
        for (int px = 1; px <= img->width; px++) {
            const unsigned char* p = image_pixel(img, px - 1, py - 1);
            int r = p[0], g = p[1], b = p[2];
            size_t i   = IDX(sat, px, py);
            size_t il  = IDX(sat, px - 1, py);
            size_t it  = IDX(sat, px, py - 1);
            size_t ilt = IDX(sat, px - 1, py - 1);

            sat->sum_r[i]  = r           + sat->sum_r[il]  + sat->sum_r[it]  - sat->sum_r[ilt];
            sat->sum_g[i]  = g           + sat->sum_g[il]  + sat->sum_g[it]  - sat->sum_g[ilt];
            sat->sum_b[i]  = b           + sat->sum_b[il]  + sat->sum_b[it]  - sat->sum_b[ilt];
            sat->sum_r2[i] = (int64_t)r*r + sat->sum_r2[il] + sat->sum_r2[it] - sat->sum_r2[ilt];
            sat->sum_g2[i] = (int64_t)g*g + sat->sum_g2[il] + sat->sum_g2[it] - sat->sum_g2[ilt];
            sat->sum_b2[i] = (int64_t)b*b + sat->sum_b2[il] + sat->sum_b2[it] - sat->sum_b2[ilt];
        }
    }

    return 0;
}

void sat_free(SAT* sat) {
    free(sat->sum_r);  sat->sum_r  = NULL;
    free(sat->sum_g);  sat->sum_g  = NULL;
    free(sat->sum_b);  sat->sum_b  = NULL;
    free(sat->sum_r2); sat->sum_r2 = NULL;
    free(sat->sum_g2); sat->sum_g2 = NULL;
    free(sat->sum_b2); sat->sum_b2 = NULL;
    sat->width = sat->height = 0;
}

// Query the sum of a channel over [x, y, x+w, y+h) using the SAT.
// Uses the inclusion-exclusion formula:
//   sum = SAT[y+h][x+w] - SAT[y][x+w] - SAT[y+h][x] + SAT[y][x]
static inline int64_t sat_rect_sum(const int64_t* table, const SAT* sat,
                                    int x, int y, int w, int h) {
    // Coordinates in the padded table (1-indexed)
    int x1 = x, y1 = y, x2 = x + w, y2 = y + h;

    // Clamp to image bounds
    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 > sat->width)  x2 = sat->width;
    if (y2 > sat->height) y2 = sat->height;

    return table[IDX(sat, x2, y2)]
         - table[IDX(sat, x1, y2)]
         - table[IDX(sat, x2, y1)]
         + table[IDX(sat, x1, y1)];
}

void sat_avg(const SAT* sat, int x, int y, int w, int h,
             unsigned char* r, unsigned char* g, unsigned char* b) {
    // Clamp region
    int x2 = x + w, y2 = y + h;
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x2 > sat->width)  x2 = sat->width;
    if (y2 > sat->height) y2 = sat->height;

    int cw = x2 - x, ch = y2 - y;
    int count = cw * ch;
    if (count <= 0) { *r = *g = *b = 0; return; }

    int64_t sr = sat_rect_sum(sat->sum_r, sat, x, y, cw, ch);
    int64_t sg = sat_rect_sum(sat->sum_g, sat, x, y, cw, ch);
    int64_t sb = sat_rect_sum(sat->sum_b, sat, x, y, cw, ch);

    *r = (unsigned char)(sr / count);
    *g = (unsigned char)(sg / count);
    *b = (unsigned char)(sb / count);
}

float sat_homogeneity(const SAT* sat, int x, int y, int w, int h,
                      unsigned char avg_r, unsigned char avg_g, unsigned char avg_b) {
    if (w <= 0 || h <= 0) return 0.0f;

    // Clamp region
    int x2 = x + w, y2 = y + h;
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x2 > sat->width)  x2 = sat->width;
    if (y2 > sat->height) y2 = sat->height;

    int cw = x2 - x, ch = y2 - y;
    int count = cw * ch;
    if (count <= 0) return 0.0f;

    // Variance = E[X²] - E[X]²
    // For each channel: var_c = (sum_c2 / N) - (avg_c)²
    int64_t sr2 = sat_rect_sum(sat->sum_r2, sat, x, y, cw, ch);
    int64_t sg2 = sat_rect_sum(sat->sum_g2, sat, x, y, cw, ch);
    int64_t sb2 = sat_rect_sum(sat->sum_b2, sat, x, y, cw, ch);

    float var_r = (float)sr2 / (float)count - (float)avg_r * (float)avg_r;
    float var_g = (float)sg2 / (float)count - (float)avg_g * (float)avg_g;
    float var_b = (float)sb2 / (float)count - (float)avg_b * (float)avg_b;

    if (var_r < 0.0f) var_r = 0.0f;
    if (var_g < 0.0f) var_g = 0.0f;
    if (var_b < 0.0f) var_b = 0.0f;

    // Convert RGB variance to YCbCr-weighted standard deviation.
    // BT.601 weights: Y ≈ 0.299R + 0.587G + 0.114B
    // Perceptual weighting of std dev matches color_homogeneity():
    //   Y=0.70, Cb=0.15, Cr=0.15
    // For simplicity, we use luminance-weighted channel variance:
    float std_r = sqrtf(var_r);
    float std_g = sqrtf(var_g);
    float std_b = sqrtf(var_b);

    // Convert channel std devs to YCbCr-approx std dev
    // Y  ~= 0.299*R + 0.587*G + 0.114*B
    // We approximate the perceptual spread as weighted sum of channel stds
    float perceptual_std = std_r * 0.299f + std_g * 0.587f + std_b * 0.114f;

    // Scale to 0..1 range (same as color_homogeneity)
    // The old function weighted dY*0.70 + dCb*0.15 + dCr*0.15.
    // A typical "uniform" block has std ~0, "chaotic" block ~40-60.
    // We normalize by 255 for consistent scale.
    return perceptual_std / 255.0f;
}
