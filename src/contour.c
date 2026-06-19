#include "contour.h"
#include "color.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ===========================================================
 * PART 1 — Edge extraction and interior-edge cancellation
 *
 * Every axis-aligned rect contributes 4 directed boundary edges
 * (clockwise winding). Edges shared by two same-color rects
 * appear once in each direction and cancel. The remaining
 * (uncancelled) edges form the outer boundary.
 * =========================================================== */

typedef struct { int x1, y1, x2, y2; } Edge;

/* Canonical sort key: smaller endpoint first so antiparallel
 * pairs sort adjacent to each other. */
typedef struct {
    int cx1, cy1, cx2, cy2;
    int orig; /* index into flat edge array */
} CEKey;

static int cekey_cmp(const void* a, const void* b) {
    const CEKey* ka = (const CEKey*)a;
    const CEKey* kb = (const CEKey*)b;
    if (ka->cx1 != kb->cx1) return ka->cx1 - kb->cx1;
    if (ka->cy1 != kb->cy1) return ka->cy1 - kb->cy1;
    if (ka->cx2 != kb->cx2) return ka->cx2 - kb->cx2;
    return ka->cy2 - kb->cy2;
}

static void make_cekey(const Edge* e, int idx, CEKey* out) {
    out->orig = idx;
    /* Choose the endpoint that comes first lexicographically */
    if (e->x1 < e->x2 || (e->x1 == e->x2 && e->y1 < e->y2)) {
        out->cx1 = e->x1; out->cy1 = e->y1;
        out->cx2 = e->x2; out->cy2 = e->y2;
    } else {
        out->cx1 = e->x2; out->cy1 = e->y2;
        out->cx2 = e->x1; out->cy2 = e->y1;
    }
}

/* Build boundary edge list for a set of same-color leaf nodes.
 * Caller must free *out_edges.
 * Returns boundary edge count, or 0 on failure. */
static int extract_boundary(QuadNode** nodes, int count,
                             Edge** out_edges) {
    int total = count * 4;
    Edge*  edges = (Edge*)malloc(total * sizeof(Edge));
    CEKey* keys  = (CEKey*)malloc(total * sizeof(CEKey));
    if (!edges || !keys) { free(edges); free(keys); return 0; }

    int ei = 0;
    for (int i = 0; i < count; i++) {
        int x = nodes[i]->x, y = nodes[i]->y;
        int w = nodes[i]->w, h = nodes[i]->h;
        edges[ei++] = (Edge){x,   y,   x+w, y  }; /* top    */
        edges[ei++] = (Edge){x+w, y,   x+w, y+h}; /* right  */
        edges[ei++] = (Edge){x+w, y+h, x,   y+h}; /* bottom */
        edges[ei++] = (Edge){x,   y+h, x,   y  }; /* left   */
    }

    for (int i = 0; i < total; i++) make_cekey(&edges[i], i, &keys[i]);
    qsort(keys, total, sizeof(CEKey), cekey_cmp);

    /* Mark interior edges: adjacent pairs with same canonical key */
    char* interior = (char*)calloc(total, 1);
    for (int i = 0; i < total - 1; i++) {
        if (cekey_cmp(&keys[i], &keys[i+1]) == 0) {
            interior[keys[i].orig]   = 1;
            interior[keys[i+1].orig] = 1;
            i++; /* consume the pair */
        }
    }

    int bc = 0;
    for (int i = 0; i < total; i++) bc += !interior[i];

    Edge* boundary = (Edge*)malloc(bc * sizeof(Edge));
    if (!boundary) { free(edges); free(keys); free(interior); return 0; }

    int bi = 0;
    for (int i = 0; i < total; i++) {
        if (!interior[i]) boundary[bi++] = edges[i];
    }

    free(edges); free(keys); free(interior);
    *out_edges = boundary;
    return bc;
}

/* ===========================================================
 * PART 2 — Chain boundary edges into a closed polygon
 *
 * Uses sort + binary-search for O(n log n) chaining.
 * Produces the longest polygon found (outer boundary).
 * =========================================================== */

typedef struct { int x, y; } Vtx;

/* Sort boundary edges by start point for binary search */
static int edge_start_cmp(const void* a, const void* b) {
    const Edge* ea = (const Edge*)a;
    const Edge* eb = (const Edge*)b;
    if (ea->x1 != eb->x1) return ea->x1 - eb->x1;
    return ea->y1 - eb->y1;
}

/* Find first unused edge whose (x1,y1) matches (tx, ty).
 * Edges are sorted, used[] tracks consumed edges.
 * Returns index or -1. */
static int find_next(Edge* edges, int count, char* used,
                     int tx, int ty) {
    /* Binary search for first edge with matching x1 */
    int lo = 0, hi = count - 1, start = -1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (edges[mid].x1 < tx || (edges[mid].x1 == tx && edges[mid].y1 < ty))
            lo = mid + 1;
        else if (edges[mid].x1 > tx || (edges[mid].x1 == tx && edges[mid].y1 > ty))
            hi = mid - 1;
        else { start = mid; break; }
    }
    if (start < 0) return -1;

    /* Scan forward for matching (x1,y1) */
    for (int i = start;
         i < count && edges[i].x1 == tx && edges[i].y1 == ty;
         i++) {
        if (!used[i]) return i;
    }
    return -1;
}

/* Chain edges into a polygon. Returns vertex count.
 * *out_vtx must be freed by caller.
 * Limit of MAX_BOUNDARY_EDGES prevents O(n²) in degenerate inputs. */
#define MAX_BOUNDARY_EDGES 2000

static int chain_polygon(Edge* edges, int count, Vtx** out_vtx) {
    if (count < 3 || count > MAX_BOUNDARY_EDGES) return 0;

    /* Sort by start point for fast lookup */
    qsort(edges, count, sizeof(Edge), edge_start_cmp);

    char* used = (char*)calloc(count, 1);
    Vtx* vtx   = (Vtx*)malloc((count + 2) * sizeof(Vtx));
    if (!used || !vtx) { free(used); free(vtx); return 0; }

    int vi = 0;
    used[0] = 1;
    int ox = edges[0].x1, oy = edges[0].y1;
    int cx = edges[0].x2, cy = edges[0].y2;
    vtx[vi++] = (Vtx){ox, oy};

    int bail = count + 2;
    while ((cx != ox || cy != oy) && bail-- > 0) {
        int ni = find_next(edges, count, used, cx, cy);
        if (ni < 0) break;
        vtx[vi++] = (Vtx){cx, cy};
        used[ni]  = 1;
        cx = edges[ni].x2;
        cy = edges[ni].y2;
    }

    free(used);
    *out_vtx = vtx;
    return vi;
}

/* ===========================================================
 * PART 3 — Douglas-Peucker simplification (closed polygon)
 *
 * Tolerance ~1.5px removes single-pixel staircase steps while
 * preserving real corners and large-scale shape.
 * =========================================================== */

static float pt_seg_dist(float px, float py,
                          float ax, float ay,
                          float bx, float by) {
    float dx = bx - ax, dy = by - ay;
    float len2 = dx*dx + dy*dy;
    if (len2 < 1e-6f) {
        float ex = px-ax, ey = py-ay;
        return sqrtf(ex*ex + ey*ey);
    }
    float t = ((px-ax)*dx + (py-ay)*dy) / len2;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    float qx = ax + t*dx, qy = ay + t*dy;
    float fx = px - qx,   fy = py - qy;
    return sqrtf(fx*fx + fy*fy);
}

static void dp_recurse(Vtx* pts, int n, float tol, char* keep) {
    if (n < 3) return;
    float max_d = 0.0f;
    int   max_i = 0;
    for (int i = 1; i < n - 1; i++) {
        float d = pt_seg_dist((float)pts[i].x, (float)pts[i].y,
                              (float)pts[0].x, (float)pts[0].y,
                              (float)pts[n-1].x, (float)pts[n-1].y);
        if (d > max_d) { max_d = d; max_i = i; }
    }
    if (max_d >= tol) {
        keep[max_i] = 1;
        dp_recurse(pts,        max_i + 1, tol, keep);
        dp_recurse(pts+max_i, n - max_i,  tol, keep);
    }
}

/* Simplify closed polygon in-place. Returns new vertex count. */
static int dp_simplify_closed(Vtx* vtx, int count, float tol) {
    if (count < 4) return count;

    /* Find the vertex farthest from any chord to use as the "open" point */
    int split = 0;
    float max_d = 0.0f;
    for (int i = 0; i < count; i++) {
        /* Distance from midpoint-opposite chord */
        int j = (i + count/2) % count;
        float d = pt_seg_dist((float)vtx[i].x, (float)vtx[i].y,
                              (float)vtx[j].x, (float)vtx[j].y,
                              (float)vtx[(j+1)%count].x,
                              (float)vtx[(j+1)%count].y);
        if (d > max_d) { max_d = d; split = i; }
    }

    /* Rotate so split is first */
    Vtx* tmp = (Vtx*)malloc(count * sizeof(Vtx));
    if (!tmp) return count;
    for (int i = 0; i < count; i++) tmp[i] = vtx[(split + i) % count];
    memcpy(vtx, tmp, count * sizeof(Vtx));
    free(tmp);

    /* Run D-P on the "open" version (first == last conceptually) */
    char* keep = (char*)calloc(count, 1);
    keep[0] = 1;
    keep[count-1] = 1;
    dp_recurse(vtx, count, tol, keep);

    int out = 0;
    for (int i = 0; i < count; i++) {
        if (keep[i]) vtx[out++] = vtx[i];
    }
    free(keep);
    return out;
}

/* ===========================================================
 * PART 4 — Angle-aware Bezier path output
 *
 * For each polygon vertex:
 *   - If the turn angle is close to 90° (sharp corner) → LineTo
 *   - Otherwise (smooth curve) → midpoint quadratic Bezier
 *
 * This means rectangles keep sharp corners, circles become smooth.
 * =========================================================== */

#define SHARP_ANGLE_DEG 60.0f   /* turns sharper than this → LineTo */

static float turn_angle_deg(Vtx prev, Vtx curr, Vtx next) {
    float ax = (float)(curr.x - prev.x), ay = (float)(curr.y - prev.y);
    float bx = (float)(next.x - curr.x), by = (float)(next.y - curr.y);
    float la = sqrtf(ax*ax + ay*ay), lb = sqrtf(bx*bx + by*by);
    if (la < 1e-4f || lb < 1e-4f) return 180.0f;
    float dot = (ax*bx + ay*by) / (la * lb);
    if (dot >  1.0f) dot =  1.0f;
    if (dot < -1.0f) dot = -1.0f;
    /* angle between the two edge directions (0 = straight, 180 = U-turn) */
    return acosf(dot) * (180.0f / 3.14159265f);
}

static void write_bezier_path(FILE* f, const char* css_class,
                               Vtx* vtx, int count) {
    /* First point: midpoint between last and first vertex */
    float mx0 = (vtx[count-1].x + vtx[0].x) * 0.5f;
    float my0 = (vtx[count-1].y + vtx[0].y) * 0.5f;

    fprintf(f, "<path class=\"%s\" d=\"M%.2f %.2f", css_class, mx0, my0);

    for (int i = 0; i < count; i++) {
        int prev = (i - 1 + count) % count;
        int next = (i + 1) % count;

        float ang = turn_angle_deg(vtx[prev], vtx[i], vtx[next]);

        /* Midpoint to the next vertex (always the on-curve landing point) */
        float nmx = (vtx[i].x + vtx[next].x) * 0.5f;
        float nmy = (vtx[i].y + vtx[next].y) * 0.5f;

        if (ang > SHARP_ANGLE_DEG) {
            /* Sharp corner: go straight to vertex, then straight to midpoint */
            fprintf(f, "L%d %d L%.2f %.2f",
                    vtx[i].x, vtx[i].y, nmx, nmy);
        } else {
            /* Smooth curve: quadratic Bezier, control at vertex */
            fprintf(f, "Q%d %d %.2f %.2f",
                    vtx[i].x, vtx[i].y, nmx, nmy);
        }
    }

    fprintf(f, "Z\"/>\n");
}

/* ===========================================================
 * PUBLIC API
 * =========================================================== */

int contour_write_smooth_path(FILE* f, const char* css_class,
                               QuadNode** nodes, int count) {
    if (count < 4) return 0;

    Edge* boundary = NULL;
    int bc = extract_boundary(nodes, count, &boundary);
    if (bc < 4 || bc > MAX_BOUNDARY_EDGES) {
        free(boundary);
        return 0;
    }

    Vtx* vtx = NULL;
    int vc = chain_polygon(boundary, bc, &vtx);
    free(boundary);
    if (vc < 4) { free(vtx); return 0; }

    /* D-P tolerance: 1.5px removes single-pixel staircase steps */
    vc = dp_simplify_closed(vtx, vc, 1.5f);
    if (vc < 3) { free(vtx); return 0; }

    write_bezier_path(f, css_class, vtx, vc);
    free(vtx);
    return 1;
}

/* Drop-in replacement for write_layer() for the anchor layer.
 * Groups nodes by palette index, tries contour for each group. */
void contour_write_layer_smooth(FILE* f, IntList* layer, QuadTree* qt,
                                 PaletteEntry* palette, int palette_size,
                                 int* node_to_palette) {
    /* Temporary node-pointer buffer (reused per color) */
    QuadNode** tmp = (QuadNode**)malloc(layer->count * sizeof(QuadNode*));
    if (!tmp) return;

    char css_class[16];

    for (int p = 0; p < palette_size; p++) {
        int tc = 0;
        for (int i = 0; i < layer->count; i++) {
            int ni = layer->node_indices[i];
            if (node_to_palette[ni] == p)
                tmp[tc++] = &qt->nodes[ni];
        }
        if (tc == 0) continue;

        snprintf(css_class, sizeof(css_class), "c%d", p);

        /* Gradient fills: always emit as individual rects (no contour) */
        if (palette[p].has_gradient) {
            for (int i = 0; i < tc; i++) {
                fprintf(f,
                    "<rect class=\"%s\" x=\"%d\" y=\"%d\""
                    " width=\"%d\" height=\"%d\"/>\n",
                    css_class,
                    tmp[i]->x, tmp[i]->y,
                    tmp[i]->w, tmp[i]->h);
            }
            continue;
        }

        /* Try smooth contour path first.
         * Fall back to the original merged-path approach if it fails. */
        if (!contour_write_smooth_path(f, css_class, tmp, tc)) {
            /* Fallback: sort by (y,x) and emit merged <path> with H/V moves */
            /* (same logic as the original write_layer) */
            for (int a = 0; a < tc - 1; a++) {
                for (int b = a + 1; b < tc; b++) {
                    if (tmp[b]->y < tmp[a]->y ||
                        (tmp[b]->y == tmp[a]->y && tmp[b]->x < tmp[a]->x)) {
                        QuadNode* t = tmp[a]; tmp[a] = tmp[b]; tmp[b] = t;
                    }
                }
            }
            fprintf(f, "<path class=\"%s\" d=\"", css_class);
            int cx = tmp[0]->x, cy = tmp[0]->y;
            int cw = tmp[0]->w, ch = tmp[0]->h;
            int pen_x = 0, pen_y = 0;
            for (int i = 1; i < tc; i++) {
                QuadNode* n = tmp[i];
                if (n->y == cy && n->h == ch && cx + cw == n->x) {
                    cw += n->w;
                } else {
                    fprintf(f, "m%d %dh%dv%dh-%dz",
                            cx - pen_x, cy - pen_y, cw, ch, cw);
                    pen_x = cx; pen_y = cy;
                    cx = n->x; cy = n->y; cw = n->w; ch = n->h;
                }
            }
            fprintf(f, "m%d %dh%dv%dh-%dz",
                    cx - pen_x, cy - pen_y, cw, ch, cw);
            fprintf(f, "\"/>\n");
        }
    }

    free(tmp);
}
