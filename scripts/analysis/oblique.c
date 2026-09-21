/*
 * The ceiling for cutting a rectangle along a line instead of into four.
 *
 * Measured the way the ramp was, before building anything: for every leaf
 * the encoder actually produces, how much of its squared error would a
 * single straight cut remove, against what a quadtree split costs to do
 * the same job? A line is the local form of every shape's boundary, so
 * this bounds what any shape primitive can be worth without having to
 * recognise a shape.
 */
#include "nvdr.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <string.h>

static const NvdrImage* g_img;
static double sse_flat, sse_cut, sse_quad;
static long leaves, cut_wins;

static double region_sse(int x, int y, int w, int h) {
    int x1 = x + w > g_img->width ? g_img->width : x + w;
    int y1 = y + h > g_img->height ? g_img->height : y + h;
    if (x1 <= x || y1 <= y) return 0.0;
    double s[3] = {0,0,0}; long n = 0;
    for (int py = y; py < y1; py++) {
        const unsigned char* p = g_img->pixels + ((size_t)py*g_img->width + x)*3;
        for (int px = x; px < x1; px++, p += 3) { for (int c=0;c<3;c++) s[c]+=p[c]; n++; }
    }
    double m[3]; for (int c=0;c<3;c++) m[c]=s[c]/n;
    double sse = 0;
    for (int py = y; py < y1; py++) {
        const unsigned char* p = g_img->pixels + ((size_t)py*g_img->width + x)*3;
        for (int px = x; px < x1; px++, p += 3)
            for (int c=0;c<3;c++) { double d=p[c]-m[c]; sse += d*d; }
    }
    return sse;
}

/* Best two-region fit over a line, searched coarsely over angle and offset. */
static double best_line_sse(int x, int y, int w, int h) {
    int x1 = x + w > g_img->width ? g_img->width : x + w;
    int y1 = y + h > g_img->height ? g_img->height : y + h;
    if (x1 <= x || y1 <= y) return 0.0;
    double best = 1e30;
    const int ANGLES = 8, OFFSETS = 7;
    for (int a = 0; a < ANGLES; a++) {
        double th = M_PI * a / ANGLES, nx = cos(th), ny = sin(th);
        /* Range of the projection over the rectangle's corners. */
        double lo = 1e30, hi = -1e30;
        double cx[4] = {x, (double)x1, x, (double)x1}, cy[4] = {y, y, (double)y1, (double)y1};
        for (int k=0;k<4;k++){ double t=nx*cx[k]+ny*cy[k]; if(t<lo)lo=t; if(t>hi)hi=t; }
        for (int o = 1; o <= OFFSETS; o++) {
            double thr = lo + (hi - lo) * o / (OFFSETS + 1.0);
            double sA[3]={0,0,0}, sB[3]={0,0,0}; long nA=0, nB=0;
            for (int py = y; py < y1; py++) {
                const unsigned char* p = g_img->pixels + ((size_t)py*g_img->width + x)*3;
                for (int px = x; px < x1; px++, p += 3) {
                    if (nx*px + ny*py < thr) { for(int c=0;c<3;c++) sA[c]+=p[c]; nA++; }
                    else                     { for(int c=0;c<3;c++) sB[c]+=p[c]; nB++; }
                }
            }
            if (!nA || !nB) continue;
            double mA[3], mB[3];
            for (int c=0;c<3;c++){ mA[c]=sA[c]/nA; mB[c]=sB[c]/nB; }
            double sse = 0;
            for (int py = y; py < y1; py++) {
                const unsigned char* p = g_img->pixels + ((size_t)py*g_img->width + x)*3;
                for (int px = x; px < x1; px++, p += 3) {
                    const double* m = (nx*px + ny*py < thr) ? mA : mB;
                    for (int c=0;c<3;c++){ double d=p[c]-m[c]; sse += d*d; }
                }
            }
            if (sse < best) best = sse;
        }
    }
    return best >= 1e29 ? region_sse(x, y, w, h) : best;
}

/*
 * At every node the encoder decides to split, compare the two refinement
 * steps head to head on that node alone: one oblique cut into two regions,
 * or the 4-way split it actually does. No recursion into what either would
 * need afterwards — that comparison cannot be made without building the
 * alternative encoder, and pretending otherwise collapses the whole image
 * to two rectangles at the root.
 */
static long decisions, cut_wins;
static double rate_cut, rate_quad;

static void walk(const NvdrTree* t, int32_t idx, float tol) {
    const NvdrNode* n = &t->nodes[idx];
    if (n->first_child < 0 || n->deviation <= tol) return;

    if (n->w >= 4 && n->h >= 4) {
        decisions++;
        double flat = region_sse(n->x, n->y, n->w, n->h);
        double line = best_line_sse(n->x, n->y, n->w, n->h);
        int hw=n->w/2, hh=n->h/2, rw=n->w-hw, rh=n->h-hh;
        double quad = region_sse(n->x,n->y,hw,hh) + region_sse(n->x+hw,n->y,rw,hh)
                    + region_sse(n->x,n->y+hh,hw,rh) + region_sse(n->x+hw,n->y+hh,rw,rh);
        sse_flat += flat; sse_cut += line; sse_quad += quad;
        if (line < quad) cut_wins++;
        /* Rough bit accounting: a residual runs about 12 bits for three
         * channels, a split flag 1, a quantised line angle plus offset
         * about 10. */
        rate_quad += 4 * 12.0 + 4 * 1.0;
        rate_cut  += 2 * 12.0 + 1.0 + 10.0;
    }
    for (int i = 0; i < 4; i++) walk(t, n->first_child + i, tol);
}

int main(int argc, char** argv) {
    NvdrImage img;
    if (argc < 2 || nvdr_image_load(&img, argv[1]) != 0) return 1;
    g_img = &img;
    NvdrConfig cfg = nvdr_default_config();
    float tol = argc > 2 ? (float)atof(argv[2]) : cfg.tolerance[NVDR_LEVELS-1];
    NvdrTree tree;
    if (nvdr_tree_build(&tree, &img, &cfg) != 0) return 1;
    walk(&tree, 0, tol);
    printf("%-24s %6ld decisoes | erro removido: corte %5.1f%%  split %5.1f%%"
           "  (%.2fx) | custo %.2fx | corte vence em %4.1f%%\n",
           argv[1], decisions,
           sse_flat>0 ? 100.0*(1.0-sse_cut/sse_flat) : 0.0,
           sse_flat>0 ? 100.0*(1.0-sse_quad/sse_flat) : 0.0,
           (sse_flat-sse_cut) > 0 && (sse_flat-sse_quad) > 0
               ? (sse_flat-sse_cut)/(sse_flat-sse_quad) : 0.0,
           rate_quad > 0 ? rate_cut/rate_quad : 0.0,
           decisions ? 100.0*cut_wins/decisions : 0.0);
    return 0;
}
