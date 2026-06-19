#include "color.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void rgb_to_ycbcr(unsigned char r, unsigned char g, unsigned char b,
                  int* Y, int* Cb, int* Cr) {
    int y  =  ( 66 * r + 129 * g +  25 * b + 128) >> 8;
    int cb = (-38 * r -  74 * g + 112 * b + 128) >> 8;
    int cr = (112 * r -  94 * g -  18 * b + 128) >> 8;
    y  += 16;
    cb += 128;
    cr += 128;
    *Y = y;
    *Cb = cb;
    *Cr = cr;
}

void color_avg(const Image* img, int x, int y, int w, int h,
               unsigned char* r, unsigned char* g, unsigned char* b) {
    unsigned long long sum_r = 0, sum_g = 0, sum_b = 0;
    int count = 0;

    int x1 = (x + w > img->width)  ? img->width  : x + w;
    int y1 = (y + h > img->height) ? img->height : y + h;

    for (int py = y; py < y1; py++) {
        for (int px = x; px < x1; px++) {
            unsigned char* p = image_pixel(img, px, py);
            sum_r += p[0];
            sum_g += p[1];
            sum_b += p[2];
            count++;
        }
    }

    if (count > 0) {
        *r = (unsigned char)(sum_r / count);
        *g = (unsigned char)(sum_g / count);
        *b = (unsigned char)(sum_b / count);
    } else {
        *r = *g = *b = 0;
    }
}

float color_homogeneity(const Image* img, int x, int y, int w, int h,
                        unsigned char avg_r, unsigned char avg_g, unsigned char avg_b) {
    if (w <= 0 || h <= 0) return 0.0f;

    float total_diff = 0.0f;
    int   count      = 0;

    int x1 = (x + w > img->width)  ? img->width  : x + w;
    int y1 = (y + h > img->height) ? img->height : y + h;

    int avgY, avgCb, avgCr;
    rgb_to_ycbcr(avg_r, avg_g, avg_b, &avgY, &avgCb, &avgCr);

    for (int py = y; py < y1; py++) {
        for (int px = x; px < x1; px++) {
            const unsigned char* p = image_pixel(img, px, py);
            int yPix, cbPix, crPix;
            rgb_to_ycbcr(p[0], p[1], p[2], &yPix, &cbPix, &crPix);

            int dY  = yPix - avgY;
            int dCb = cbPix - avgCb;
            int dCr = crPix - avgCr;

            float diff = fabsf((float)dY) * 0.70f
                       + fabsf((float)dCb) * 0.15f
                       + fabsf((float)dCr) * 0.15f;

            total_diff += diff;
            count++;
        }
    }

    if (count == 0) return 0.0f;
    return (total_diff / (float)count) / 255.0f;
}

void color_to_hex(unsigned char r, unsigned char g, unsigned char b, char* buf) {
    // Forma curta #rgb quando ambos nibbles são iguais (e.g. #aabb00 → #ab0)
    if ((r >> 4) == (r & 0xF) &&
        (g >> 4) == (g & 0xF) &&
        (b >> 4) == (b & 0xF)) {
        snprintf(buf, 5, "#%x%x%x", r >> 4, g >> 4, b >> 4);
    } else {
        snprintf(buf, 8, "#%02x%02x%02x", r, g, b);
    }
}

int color_gradient_v(const Image* img, int x, int y, int w, int h,
                     unsigned char* top_r, unsigned char* top_g, unsigned char* top_b,
                     unsigned char* bot_r, unsigned char* bot_g, unsigned char* bot_b,
                     int threshold) {
    int half_h = h / 2;
    if (half_h < 1) return 0;

    // Média da metade superior
    color_avg(img, x, y, w, half_h, top_r, top_g, top_b);
    // Média da metade inferior
    color_avg(img, x, y + half_h, w, h - half_h, bot_r, bot_g, bot_b);

    int diff = abs((int)*top_r - (int)*bot_r) +
               abs((int)*top_g - (int)*bot_g) +
               abs((int)*top_b - (int)*bot_b);
    return diff > threshold;
}

int color_gradient_h(const Image* img, int x, int y, int w, int h,
                     unsigned char* left_r, unsigned char* left_g, unsigned char* left_b,
                     unsigned char* right_r, unsigned char* right_g, unsigned char* right_b,
                     int threshold) {
    int half_w = w / 2;
    if (half_w < 1) return 0;

    // Média da metade esquerda
    color_avg(img, x, y, half_w, h, left_r, left_g, left_b);
    // Média da metade direita
    color_avg(img, x + half_w, y, w - half_w, h, right_r, right_g, right_b);

    int diff = abs((int)*left_r - (int)*right_r) +
               abs((int)*left_g - (int)*right_g) +
               abs((int)*left_b - (int)*right_b);
    return diff > threshold;
}

int color_edge_snap(const Image* img, int x, int y, int w, int h,
                    unsigned char* r, unsigned char* g, unsigned char* b,
                    int edge_threshold) {
    if (w < 2 || h < 2) return 0;

    int x1 = (x + w > img->width)  ? img->width  : x + w;
    int y1 = (y + h > img->height) ? img->height : y + h;

    // Find the maximum single-row or single-column gradient jump
    // Horizontal scan: find the row with max avg color difference top vs bottom half
    int best_split_h = -1;
    int best_diff_h = 0;

    // Check horizontal split (top half vs bottom half at each possible y-split)
    // Simplified: just check the center split
    int mid_y = y + h / 2;
    {
        unsigned long long tr = 0, tg = 0, tb = 0, tc = 0;
        unsigned long long br = 0, bg = 0, bb = 0, bc = 0;
        for (int py = y; py < mid_y && py < y1; py++) {
            for (int px = x; px < x1; px++) {
                unsigned char* p = image_pixel(img, px, py);
                tr += p[0]; tg += p[1]; tb += p[2]; tc++;
            }
        }
        for (int py = mid_y; py < y1; py++) {
            for (int px = x; px < x1; px++) {
                unsigned char* p = image_pixel(img, px, py);
                br += p[0]; bg += p[1]; bb += p[2]; bc++;
            }
        }
        if (tc > 0 && bc > 0) {
            int diff = abs((int)(tr/tc) - (int)(br/bc)) +
                       abs((int)(tg/tc) - (int)(bg/bc)) +
                       abs((int)(tb/tc) - (int)(bb/bc));
            if (diff > best_diff_h) {
                best_diff_h = diff;
                best_split_h = mid_y;
            }
        }
    }

    // Vertical scan: left half vs right half
    int best_diff_v = 0;
    int mid_x = x + w / 2;
    {
        unsigned long long lr = 0, lg = 0, lb = 0, lc = 0;
        unsigned long long rr = 0, rg = 0, rb = 0, rc = 0;
        for (int py = y; py < y1; py++) {
            for (int px = x; px < mid_x && px < x1; px++) {
                unsigned char* p = image_pixel(img, px, py);
                lr += p[0]; lg += p[1]; lb += p[2]; lc++;
            }
            for (int px = mid_x; px < x1; px++) {
                unsigned char* p = image_pixel(img, px, py);
                rr += p[0]; rg += p[1]; rb += p[2]; rc++;
            }
        }
        if (lc > 0 && rc > 0) {
            int diff = abs((int)(lr/lc) - (int)(rr/rc)) +
                       abs((int)(lg/lc) - (int)(rg/rc)) +
                       abs((int)(lb/lc) - (int)(rb/rc));
            if (diff > best_diff_v) {
                best_diff_v = diff;
            }
        }
    }

    int best_diff = best_diff_h > best_diff_v ? best_diff_h : best_diff_v;
    if (best_diff < edge_threshold) return 0;

    // Edge detected! Use dominant color (the side with more area)
    // For simplicity, pick the color of the larger half
    // Re-sample the two halves and pick the one with more pixels
    if (best_diff_h >= best_diff_v && best_split_h > 0) {
        // Horizontal edge: pick top or bottom based on area
        int top_area = (best_split_h - y) * w;
        int bot_area = (y + h - best_split_h) * w;
        if (top_area >= bot_area) {
            color_avg(img, x, y, w, best_split_h - y, r, g, b);
        } else {
            color_avg(img, x, best_split_h, w, y + h - best_split_h, r, g, b);
        }
    } else {
        // Vertical edge: pick left or right
        int left_area = (mid_x - x) * h;
        int right_area = (x + w - mid_x) * h;
        if (left_area >= right_area) {
            color_avg(img, x, y, mid_x - x, h, r, g, b);
        } else {
            color_avg(img, mid_x, y, x + w - mid_x, h, r, g, b);
        }
    }
    return 1;
}
