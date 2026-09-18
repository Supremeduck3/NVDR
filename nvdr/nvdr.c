/*
 * NVDR — Progressive Residual Stack, still-image implementation.
 * See nvdr.h for what this is and how it maps onto spec v0.14.1 §1.2.
 */
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "nvdr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <zlib.h>
#include "entropy.h"

/* ================================================================ image */

int nvdr_image_load(NvdrImage* img, const char* path) {
    int channels;
    img->pixels = stbi_load(path, &img->width, &img->height, &channels, 3);
    return img->pixels ? 0 : -1;
}

int nvdr_image_write_ppm(const NvdrImage* img, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P6\n%d %d\n255\n", img->width, img->height);
    size_t n = (size_t)img->width * img->height * 3;
    size_t written = fwrite(img->pixels, 1, n, f);
    fclose(f);
    return written == n ? 0 : -1;
}

/*
 * PNG writer. PPM is trivial to emit but opens in almost nothing, which
 * makes eyeballing a decode harder than it needs to be. zlib is already
 * linked for the container, so a real PNG costs one deflate and three
 * CRCs.
 */
static void png_chunk(FILE* f, const char* type, const uint8_t* data, size_t len) {
    uint8_t size[4] = {
        (uint8_t)(len >> 24), (uint8_t)(len >> 16), (uint8_t)(len >> 8), (uint8_t)len
    };
    fwrite(size, 1, 4, f);
    fwrite(type, 1, 4, f);
    if (len) fwrite(data, 1, len, f);

    uLong crc = crc32(0, (const Bytef*)type, 4);
    if (len) crc = crc32(crc, (const Bytef*)data, (uInt)len);
    uint8_t crc_bytes[4] = {
        (uint8_t)(crc >> 24), (uint8_t)(crc >> 16), (uint8_t)(crc >> 8), (uint8_t)crc
    };
    fwrite(crc_bytes, 1, 4, f);
}

int nvdr_image_write_png(const NvdrImage* img, const char* path) {
    /* Every scanline is prefixed with filter type 0 — no prediction. The
     * data is flat rectangles, so a filter would buy little. */
    size_t stride = (size_t)img->width * 3;
    size_t raw_size = (stride + 1) * (size_t)img->height;
    uint8_t* raw = (uint8_t*)malloc(raw_size);
    if (!raw) return -1;
    for (int y = 0; y < img->height; y++) {
        raw[(stride + 1) * (size_t)y] = 0;
        memcpy(raw + (stride + 1) * (size_t)y + 1, img->pixels + stride * (size_t)y, stride);
    }

    uLongf packed_size = compressBound((uLong)raw_size);
    uint8_t* packed = (uint8_t*)malloc(packed_size);
    if (!packed || compress2(packed, &packed_size, raw, (uLong)raw_size, 6) != Z_OK) {
        free(raw); free(packed);
        return -1;
    }
    free(raw);

    FILE* f = fopen(path, "wb");
    if (!f) { free(packed); return -1; }

    static const uint8_t signature[8] = { 137, 'P', 'N', 'G', 13, 10, 26, 10 };
    fwrite(signature, 1, 8, f);

    uint8_t ihdr[13];
    uint32_t w = (uint32_t)img->width, h = (uint32_t)img->height;
    ihdr[0] = (uint8_t)(w >> 24); ihdr[1] = (uint8_t)(w >> 16);
    ihdr[2] = (uint8_t)(w >> 8);  ihdr[3] = (uint8_t)w;
    ihdr[4] = (uint8_t)(h >> 24); ihdr[5] = (uint8_t)(h >> 16);
    ihdr[6] = (uint8_t)(h >> 8);  ihdr[7] = (uint8_t)h;
    ihdr[8] = 8;    /* bit depth */
    ihdr[9] = 2;    /* truecolour */
    ihdr[10] = ihdr[11] = ihdr[12] = 0;
    png_chunk(f, "IHDR", ihdr, sizeof(ihdr));
    png_chunk(f, "IDAT", packed, packed_size);
    png_chunk(f, "IEND", NULL, 0);

    fclose(f);
    free(packed);
    return 0;
}

int nvdr_image_write(const NvdrImage* img, const char* path) {
    size_t len = strlen(path);
    if (len >= 4 && strcmp(path + len - 4, ".png") == 0)
        return nvdr_image_write_png(img, path);
    return nvdr_image_write_ppm(img, path);
}

void nvdr_image_free(NvdrImage* img) {
    free(img->pixels);
    img->pixels = NULL;
    img->width = img->height = 0;
}

double nvdr_psnr(const NvdrImage* a, const NvdrImage* b) {
    if (a->width != b->width || a->height != b->height) return -1.0;
    size_t n = (size_t)a->width * a->height * 3;
    double mse = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a->pixels[i] - (double)b->pixels[i];
        mse += d * d;
    }
    mse /= (double)n;
    return mse <= 0.0 ? 99.0 : 10.0 * log10(255.0 * 255.0 / mse);
}

/* ============================================================== bit i/o */

typedef struct {
    uint8_t* bytes;
    size_t   capacity;
    size_t   bit_count;
} BitWriter;

static int bw_init(BitWriter* bw, size_t expected_bits) {
    bw->capacity = (expected_bits + 7) / 8 + 64;
    bw->bytes = (uint8_t*)calloc(bw->capacity, 1);
    bw->bit_count = 0;
    return bw->bytes ? 0 : -1;
}

static int bw_put(BitWriter* bw, int bit) {
    size_t byte = bw->bit_count >> 3;
    if (byte >= bw->capacity) {
        size_t grown = bw->capacity * 2;
        uint8_t* next = (uint8_t*)realloc(bw->bytes, grown);
        if (!next) return -1;
        memset(next + bw->capacity, 0, grown - bw->capacity);
        bw->bytes = next;
        bw->capacity = grown;
    }
    if (bit) bw->bytes[byte] |= (uint8_t)(1u << (bw->bit_count & 7));
    bw->bit_count++;
    return 0;
}

typedef struct {
    const uint8_t* bytes;
    size_t         bit_count;
    size_t         bit_pos;
    int            overrun;
} BitReader;

static int br_get(BitReader* br) {
    if (br->bit_pos >= br->bit_count) { br->overrun = 1; return 0; }
    int bit = (br->bytes[br->bit_pos >> 3] >> (br->bit_pos & 7)) & 1;
    br->bit_pos++;
    return bit;
}

/* =========================================================== tree build */

NvdrConfig nvdr_default_config(void) {
    NvdrConfig cfg;
    cfg.min_tile     = 2;
    cfg.max_depth    = 12;
    cfg.tolerance[0] = 0.090f;   /* anchor: only genuinely flat regions stay */
    cfg.tolerance[1] = 0.040f;
    cfg.tolerance[2] = 0.018f;   /* the tree is built to this */
    cfg.anchor_bits  = 4;        /* the spec's int4 anchor */
    cfg.step[0]      = 0;        /* level 0 is palette-coded, not residual */
    /* A level's step is sized to the correction it carries. R1 moves a
     * rectangle from its parent's colour to its own, which is a large
     * jump, so a fine step there spends bits resolving differences the
     * next level will overwrite anyway. Swept over the sample set: 16/4
     * lands 35KB below a 2/2 stack at the same PSNR. */
    cfg.step[1]      = 16;
    cfg.step[2]      = 4;
    cfg.codec        = NVDR_COMPRESS_ARITH;
    return cfg;
}

static unsigned char* pixel_at(const NvdrImage* img, int x, int y) {
    return img->pixels + ((size_t)y * img->width + x) * 3;
}

/*
 * Mean colour of a region and the mean perceptual distance from it,
 * normalised to 0..1 so a tolerance reads the same at any bit depth. The
 * deviation accumulates in double because a 4K region is millions of
 * terms and a float accumulator starts dropping the small ones.
 */
static float region_stats(const NvdrImage* img, int x, int y, int w, int h,
                          uint8_t* out_r, uint8_t* out_g, uint8_t* out_b) {
    int x1 = x + w > img->width  ? img->width  : x + w;
    int y1 = y + h > img->height ? img->height : y + h;
    if (x1 <= x || y1 <= y) { *out_r = *out_g = *out_b = 0; return 0.0f; }

    uint64_t sum_r = 0, sum_g = 0, sum_b = 0;
    uint32_t count = 0;
    for (int py = y; py < y1; py++) {
        const unsigned char* p = pixel_at(img, x, py);
        for (int px = x; px < x1; px++, p += 3) {
            sum_r += p[0]; sum_g += p[1]; sum_b += p[2];
            count++;
        }
    }
    uint8_t mr = (uint8_t)((sum_r + count / 2) / count);
    uint8_t mg = (uint8_t)((sum_g + count / 2) / count);
    uint8_t mb = (uint8_t)((sum_b + count / 2) / count);
    *out_r = mr; *out_g = mg; *out_b = mb;

    double deviation = 0.0;
    for (int py = y; py < y1; py++) {
        const unsigned char* p = pixel_at(img, x, py);
        for (int px = x; px < x1; px++, p += 3) {
            int dr = (int)p[0] - mr, dg = (int)p[1] - mg, db = (int)p[2] - mb;
            if (dr < 0) dr = -dr;
            if (dg < 0) dg = -dg;
            if (db < 0) db = -db;
            deviation += 0.30 * dr + 0.59 * dg + 0.11 * db;
        }
    }
    return (float)(deviation / count / 255.0);
}

static int32_t tree_alloc(NvdrTree* tree) {
    if (tree->count == tree->capacity) {
        uint32_t grown = tree->capacity ? tree->capacity * 2 : 8192;
        NvdrNode* nodes = (NvdrNode*)realloc(tree->nodes, (size_t)grown * sizeof(NvdrNode));
        if (!nodes) return -1;
        tree->nodes = nodes;
        tree->capacity = grown;
    }
    return (int32_t)tree->count++;
}

static int tree_build_rec(NvdrTree* tree, const NvdrImage* img,
                          const NvdrConfig* cfg,
                          int32_t idx, int x, int y, int w, int h, int depth) {
    NvdrNode* node = &tree->nodes[idx];
    node->x = (uint16_t)x; node->y = (uint16_t)y;
    node->w = (uint16_t)w; node->h = (uint16_t)h;
    node->first_child = -1;
    node->deviation = region_stats(img, x, y, w, h, &node->r, &node->g, &node->b);

    int splittable = w > cfg->min_tile && h > cfg->min_tile && depth < cfg->max_depth;
    if (!splittable || node->deviation <= cfg->tolerance[NVDR_LEVELS - 1]) return 0;

    int32_t first = tree_alloc(tree);
    if (first < 0) return -1;
    for (int i = 1; i < 4; i++) if (tree_alloc(tree) < 0) return -1;
    tree->nodes[idx].first_child = first;

    int hw = w / 2, hh = h / 2, rw = w - hw, rh = h - hh;
    int rc = 0;
    rc |= tree_build_rec(tree, img, cfg, first + 0, x,      y,      hw, hh, depth + 1);
    rc |= tree_build_rec(tree, img, cfg, first + 1, x + hw, y,      rw, hh, depth + 1);
    rc |= tree_build_rec(tree, img, cfg, first + 2, x,      y + hh, hw, rh, depth + 1);
    rc |= tree_build_rec(tree, img, cfg, first + 3, x + hw, y + hh, rw, rh, depth + 1);
    return rc;
}

int nvdr_tree_build(NvdrTree* tree, const NvdrImage* img, const NvdrConfig* cfg) {
    memset(tree, 0, sizeof(*tree));
    int32_t root = tree_alloc(tree);
    if (root < 0) return -1;
    if (tree_build_rec(tree, img, cfg, root, 0, 0, img->width, img->height, 0) != 0) {
        nvdr_tree_free(tree);
        return -1;
    }
    return 0;
}

void nvdr_tree_free(NvdrTree* tree) {
    free(tree->nodes);
    memset(tree, 0, sizeof(*tree));
}

/* ======================================================== anchor palette */

/*
 * Candidates come from a 5-bit-per-channel histogram of the level-0 leaf
 * colours weighted by area. That bounds the candidate set at 32768
 * whatever the image size, and costs one pass.
 */
typedef struct { uint8_t r, g, b; double weight; } Candidate;

static int nearest_entry(const unsigned char* palette, int n, int r, int g, int b) {
    int best = 0;
    long best_dist = -1;
    for (int i = 0; i < n; i++) {
        long dr = r - palette[i * 3 + 0];
        long dg = g - palette[i * 3 + 1];
        long db = b - palette[i * 3 + 2];
        long d = dr * dr * 3 + dg * dg * 4 + db * db * 2;
        if (best_dist < 0 || d < best_dist) { best_dist = d; best = i; }
    }
    return best;
}

static int build_palette(const NvdrTree* tree, const uint32_t* leaves,
                         uint32_t leaf_count, int wanted,
                         unsigned char* palette_out) {
    enum { BINS = 32768 };
    double* weight = (double*)calloc(BINS, sizeof(double));
    double* sum_r  = (double*)calloc(BINS, sizeof(double));
    double* sum_g  = (double*)calloc(BINS, sizeof(double));
    double* sum_b  = (double*)calloc(BINS, sizeof(double));
    Candidate* cands = (Candidate*)malloc(BINS * sizeof(Candidate));
    if (!weight || !sum_r || !sum_g || !sum_b || !cands) {
        free(weight); free(sum_r); free(sum_g); free(sum_b); free(cands);
        return -1;
    }

    for (uint32_t i = 0; i < leaf_count; i++) {
        const NvdrNode* n = &tree->nodes[leaves[i]];
        double area = (double)n->w * (double)n->h;
        int bin = ((n->r >> 3) << 10) | ((n->g >> 3) << 5) | (n->b >> 3);
        weight[bin] += area;
        sum_r[bin] += area * n->r;
        sum_g[bin] += area * n->g;
        sum_b[bin] += area * n->b;
    }

    int cand_count = 0;
    for (int i = 0; i < BINS; i++) {
        if (weight[i] <= 0.0) continue;
        Candidate* c = &cands[cand_count++];
        c->r = (uint8_t)(sum_r[i] / weight[i] + 0.5);
        c->g = (uint8_t)(sum_g[i] / weight[i] + 0.5);
        c->b = (uint8_t)(sum_b[i] / weight[i] + 0.5);
        c->weight = weight[i];
    }
    free(weight); free(sum_r); free(sum_g); free(sum_b);
    if (cand_count == 0) { free(cands); return -1; }
    if (wanted > cand_count) wanted = cand_count;

    /* Greedy max-salience: heaviest colour first, then whichever candidate
     * maximises weight x distance-to-nearest-chosen. A candidate's distance
     * to the set only shrinks, so it is carried forward rather than
     * recomputed — O(wanted * cand_count) instead of a cubic loop. */
    double* min_dist = (double*)malloc((size_t)cand_count * sizeof(double));
    char*   taken    = (char*)calloc((size_t)cand_count, 1);
    if (!min_dist || !taken) { free(cands); free(min_dist); free(taken); return -1; }
    for (int i = 0; i < cand_count; i++) min_dist[i] = 1e30;

    int best = 0;
    for (int i = 1; i < cand_count; i++)
        if (cands[i].weight > cands[best].weight) best = i;

    int chosen = 0;
    for (;;) {
        palette_out[chosen * 3 + 0] = cands[best].r;
        palette_out[chosen * 3 + 1] = cands[best].g;
        palette_out[chosen * 3 + 2] = cands[best].b;
        taken[best] = 1;
        if (++chosen >= wanted) break;

        int next = -1;
        double best_salience = -1.0;
        for (int i = 0; i < cand_count; i++) {
            if (taken[i]) continue;
            double dr = (double)cands[i].r - cands[best].r;
            double dg = (double)cands[i].g - cands[best].g;
            double db = (double)cands[i].b - cands[best].b;
            double d = dr * dr * 3.0 + dg * dg * 4.0 + db * db * 2.0;
            if (d < min_dist[i]) min_dist[i] = d;
            double salience = cands[i].weight * min_dist[i];
            if (salience > best_salience) { best_salience = salience; next = i; }
        }
        if (next < 0) break;
        best = next;
    }

    free(min_dist); free(taken);

    /*
     * Lloyd refinement. The greedy pass above maximises spread, which is a
     * good seeding — it is essentially k-means++ without the randomness —
     * but spread is not the objective. The objective is the area-weighted
     * error of assigning every leaf colour to its nearest entry, and that
     * is what these iterations actually minimise.
     *
     * Runs over the histogram candidates rather than the leaves, so the
     * cost is bounded by the 32768 bins regardless of image size.
     */
    double* cell_r = (double*)calloc((size_t)chosen, sizeof(double));
    double* cell_g = (double*)calloc((size_t)chosen, sizeof(double));
    double* cell_b = (double*)calloc((size_t)chosen, sizeof(double));
    double* mass  = (double*)calloc((size_t)chosen, sizeof(double));
    if (cell_r && cell_g && cell_b && mass) {
        for (int iteration = 0; iteration < 12; iteration++) {
            memset(cell_r, 0, (size_t)chosen * sizeof(double));
            memset(cell_g, 0, (size_t)chosen * sizeof(double));
            memset(cell_b, 0, (size_t)chosen * sizeof(double));
            memset(mass,  0, (size_t)chosen * sizeof(double));

            for (int i = 0; i < cand_count; i++) {
                int nearest = nearest_entry(palette_out, chosen,
                                            cands[i].r, cands[i].g, cands[i].b);
                double w = cands[i].weight;
                cell_r[nearest] += w * cands[i].r;
                cell_g[nearest] += w * cands[i].g;
                cell_b[nearest] += w * cands[i].b;
                mass[nearest]  += w;
            }

            int moved = 0;
            for (int p = 0; p < chosen; p++) {
                if (mass[p] <= 0.0) continue;   /* empty cell keeps its seed */
                unsigned char r = (unsigned char)(cell_r[p] / mass[p] + 0.5);
                unsigned char g = (unsigned char)(cell_g[p] / mass[p] + 0.5);
                unsigned char b = (unsigned char)(cell_b[p] / mass[p] + 0.5);
                if (r != palette_out[p * 3 + 0] || g != palette_out[p * 3 + 1] ||
                    b != palette_out[p * 3 + 2]) moved = 1;
                palette_out[p * 3 + 0] = r;
                palette_out[p * 3 + 1] = g;
                palette_out[p * 3 + 2] = b;
            }
            if (!moved) break;
        }
    }
    free(cell_r); free(cell_g); free(cell_b); free(mass);

    free(cands);
    return chosen;
}

/* ============================================================== helpers */

static int clamp_u8(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }
static int clamp_i8(int v) { return v < -127 ? -127 : (v > 127 ? 127 : v); }
static int div_round(int n, int d) {
    return n >= 0 ? (n + d / 2) / d : -((-n + d / 2) / d);
}

typedef struct {
    uint32_t* items;
    uint32_t  count;
    uint32_t  capacity;
} IndexList;

static int il_push(IndexList* list, uint32_t value) {
    if (list->count == list->capacity) {
        uint32_t grown = list->capacity ? list->capacity * 2 : 4096;
        uint32_t* items = (uint32_t*)realloc(list->items, (size_t)grown * sizeof(uint32_t));
        if (!items) return -1;
        list->items = items;
        list->capacity = grown;
    }
    list->items[list->count++] = value;
    return 0;
}

/* ============================================================== encoder */

/*
 * Cut a level out of the tree. A node stops here if it has no children or
 * if its region is already within `tolerance`; otherwise the four children
 * are visited. One bit per visited node records which way it went, and the
 * decoder replays the same walk from the canvas rectangle alone.
 */
static int cut_level(const NvdrTree* tree, int32_t idx, float tolerance,
                     BitWriter* bw, IndexList* out) {
    const NvdrNode* node = &tree->nodes[idx];
    int split = node->first_child >= 0 && node->deviation > tolerance;
    if (bw_put(bw, split) != 0) return -1;
    if (!split) return il_push(out, (uint32_t)idx);
    for (int i = 0; i < 4; i++)
        if (cut_level(tree, node->first_child + i, tolerance, bw, out) != 0) return -1;
    return 0;
}

/*
 * Re-code a level's split bitstream through the arithmetic coder, giving
 * each bit the area context of the rectangle it decides. The tree shape is
 * recovered by walking it again rather than stored, so nothing extra is
 * carried: the walk is deterministic from the canvas rectangle down.
 */
static void transcode_split(BitReader* br, NvdrEncoder* enc, NvdrModels* m,
                            int w, int h) {
    int bit = br_get(br);
    nvdr_enc_bit(enc, &m->split[nvdr_area_context(w, h)], bit);
    if (!bit) return;
    int hw = w / 2, hh = h / 2, rw = w - hw, rh = h - hh;
    transcode_split(br, enc, m, hw, hh);
    transcode_split(br, enc, m, rw, hh);
    transcode_split(br, enc, m, hw, rh);
    transcode_split(br, enc, m, rw, rh);
}

static void put_u16(uint8_t* p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void put_u32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static uint16_t get_u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t get_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int nvdr_encode_file(const char* out_path, const NvdrImage* img,
                     const NvdrConfig* cfg, NvdrHeader* hdr_out) {
    NvdrTree tree;
    if (nvdr_tree_build(&tree, img, cfg) != 0) return -1;

    BitWriter  bits[NVDR_LEVELS];
    IndexList  leaves[NVDR_LEVELS];
    uint8_t*   recon[NVDR_LEVELS];       /* reconstructed rgb per leaf */
    int8_t*    residual[NVDR_LEVELS];    /* 3 planes, level >= 1 only */
    uint8_t*   split_ctx[NVDR_LEVELS];   /* 1 when the parent subdivided */
    unsigned char* palette = NULL;
    uint8_t*   tokens = NULL;
    int rc = -1;

    memset(bits, 0, sizeof(bits));
    memset(leaves, 0, sizeof(leaves));
    memset(recon, 0, sizeof(recon));
    memset(residual, 0, sizeof(residual));
    memset(split_ctx, 0, sizeof(split_ctx));

    for (int k = 0; k < NVDR_LEVELS; k++)
        if (bw_init(&bits[k], tree.count + 1024) != 0) goto done;

    /* --- level 0: cut the whole tree at the coarse tolerance --- */
    if (cut_level(&tree, 0, cfg->tolerance[0], &bits[0], &leaves[0]) != 0) goto done;

    int palette_n = 1 << cfg->anchor_bits;
    palette = (unsigned char*)calloc((size_t)palette_n * 3, 1);
    if (!palette) goto done;
    palette_n = build_palette(&tree, leaves[0].items, leaves[0].count,
                              palette_n, palette);
    if (palette_n <= 0) goto done;

    tokens = (uint8_t*)malloc(leaves[0].count);
    recon[0] = (uint8_t*)malloc((size_t)leaves[0].count * 3);
    if (!tokens || !recon[0]) goto done;
    for (uint32_t i = 0; i < leaves[0].count; i++) {
        const NvdrNode* n = &tree.nodes[leaves[0].items[i]];
        int token = nearest_entry(palette, palette_n, n->r, n->g, n->b);
        tokens[i] = (uint8_t)token;
        memcpy(recon[0] + (size_t)i * 3, palette + token * 3, 3);
    }

    /* --- levels 1..N: re-cut each previous leaf at a finer tolerance --- */
    for (int k = 1; k < NVDR_LEVELS; k++) {
        /* Expanding one previous leaf may yield many leaves here; they all
         * inherit that leaf's displayed colour as the base of their delta. */
        uint8_t* base = NULL;
        size_t base_capacity = 0, base_count = 0, ctx_capacity = 0;

        for (uint32_t i = 0; i < leaves[k - 1].count; i++) {
            uint32_t before = leaves[k].count;
            if (cut_level(&tree, (int32_t)leaves[k - 1].items[i],
                          cfg->tolerance[k], &bits[k], &leaves[k]) != 0) {
                free(base); goto done;
            }
            uint32_t produced = leaves[k].count - before;
            if (base_count + produced > base_capacity) {
                size_t grown = (base_capacity ? base_capacity * 2 : 8192);
                while (grown < base_count + produced) grown *= 2;
                uint8_t* next = (uint8_t*)realloc(base, grown * 3);
                if (!next) { free(base); goto done; }
                base = next;
                base_capacity = grown;
            }
            /* A rectangle whose parent subdivided carries a genuinely new
             * colour; one whose parent did not carries a small correction
             * to a colour already close. The entropy coder is given that
             * distinction as context; it is recomputed on the decoder side
             * from the same split bitstream, never transmitted. */
            if (base_count + produced > ctx_capacity) {
                size_t grown = ctx_capacity ? ctx_capacity * 2 : 8192;
                while (grown < base_count + produced) grown *= 2;
                uint8_t* next = (uint8_t*)realloc(split_ctx[k], grown);
                if (!next) { free(base); goto done; }
                split_ctx[k] = next;
                ctx_capacity = grown;
            }
            for (uint32_t j = 0; j < produced; j++) {
                memcpy(base + (base_count + j) * 3, recon[k - 1] + (size_t)i * 3, 3);
                split_ctx[k][base_count + j] = (uint8_t)(produced > 1);
            }
            base_count += produced;
        }

        recon[k] = (uint8_t*)malloc((size_t)leaves[k].count * 3);
        residual[k] = (int8_t*)malloc((size_t)leaves[k].count * 3);
        if (!recon[k] || !residual[k]) { free(base); goto done; }

        for (uint32_t i = 0; i < leaves[k].count; i++) {
            const NvdrNode* n = &tree.nodes[leaves[k].items[i]];
            int target[3] = { n->r, n->g, n->b };
            for (int c = 0; c < 3; c++) {
                int delta = target[c] - (int)base[(size_t)i * 3 + c];
                int q = clamp_i8(div_round(delta, cfg->step[k]));
                /* Planar: all of channel c together, so the entropy coder
                 * sees long runs of near-identical deltas. */
                residual[k][(size_t)c * leaves[k].count + i] = (int8_t)q;
                recon[k][(size_t)i * 3 + c] =
                    (uint8_t)clamp_u8((int)base[(size_t)i * 3 + c] + q * cfg->step[k]);
            }
        }
        free(base);
    }

    /* ------------------------------- write ------------------------------- */
    {
        /* Each level is assembled whole, then deflated on its own. Sharing
         * one deflate stream across levels would compress better and would
         * make every prefix undecodable, which is the one thing this format
         * is not allowed to give up. */
        uint8_t* raw[NVDR_LEVELS] = { NULL, NULL, NULL };
        uint8_t* stored[NVDR_LEVELS] = { NULL, NULL, NULL };
        size_t raw_size[NVDR_LEVELS] = { 0, 0, 0 };
        size_t stored_size[NVDR_LEVELS] = { 0, 0, 0 };
        int wrote = 0;

        size_t token_bytes = ((size_t)leaves[0].count * cfg->anchor_bits + 7) / 8;
        size_t bits0_bytes = (bits[0].bit_count + 7) / 8;
        raw_size[0] = 1 + (size_t)palette_n * 3 + bits0_bytes + token_bytes;
        raw[0] = (uint8_t*)malloc(raw_size[0]);
        if (!raw[0]) goto write_done;
        {
            size_t at = 0;
            /* Stored biased by one: a full 256-entry palette would
             * otherwise wrap to zero in this byte, which is exactly what
             * --anchor-bits 8 used to produce. A palette is never empty,
             * so the bias costs nothing. */
            raw[0][at++] = (uint8_t)(palette_n - 1);
            memcpy(raw[0] + at, palette, (size_t)palette_n * 3);
            at += (size_t)palette_n * 3;
            memcpy(raw[0] + at, bits[0].bytes, bits0_bytes);
            at += bits0_bytes;
            memset(raw[0] + at, 0, token_bytes);
            for (uint32_t i = 0; i < leaves[0].count; i++) {
                size_t bit = (size_t)i * cfg->anchor_bits;
                for (int b = 0; b < cfg->anchor_bits; b++)
                    if (tokens[i] & (1u << b))
                        raw[0][at + ((bit + b) >> 3)] |= (uint8_t)(1u << ((bit + b) & 7));
            }
        }

        for (int k = 1; k < NVDR_LEVELS; k++) {
            size_t bits_bytes = (bits[k].bit_count + 7) / 8;
            size_t residual_bytes = (size_t)leaves[k].count * 3;
            raw_size[k] = bits_bytes + residual_bytes;
            raw[k] = (uint8_t*)malloc(raw_size[k]);
            if (!raw[k]) goto write_done;
            memcpy(raw[k], bits[k].bytes, bits_bytes);
            memcpy(raw[k] + bits_bytes, residual[k], residual_bytes);
        }

        if (cfg->codec == NVDR_COMPRESS_ARITH) {
            /* Each level gets its own coder and its own fresh model: a
             * level has to decode without the ones after it, so adaptation
             * cannot carry across the boundary. */
            for (int k = 0; k < NVDR_LEVELS; k++) {
                NvdrModels models;
                NvdrEncoder ae;
                nvdr_models_init(&models);
                if (nvdr_enc_init(&ae, raw_size[k] / 2 + 1024) != 0) goto write_done;

                size_t prefix = 0;
                if (k == 0) {
                    /* The palette rides ahead of the coded stream: 48 bytes
                     * of genuinely incompressible colour are not worth
                     * modelling. */
                    prefix = 1 + (size_t)palette_n * 3;
                }

                BitReader br;
                br.bytes = bits[k].bytes;
                br.bit_count = bits[k].bit_count;
                br.bit_pos = 0;
                br.overrun = 0;

                if (k == 0) {
                    transcode_split(&br, &ae, &models, img->width, img->height);
                    for (uint32_t i = 0; i < leaves[0].count; i++)
                        nvdr_enc_tree(&ae, models.token, tokens[i], cfg->anchor_bits);
                } else {
                    for (uint32_t i = 0; i < leaves[k - 1].count; i++) {
                        const NvdrNode* n = &tree.nodes[leaves[k - 1].items[i]];
                        transcode_split(&br, &ae, &models, n->w, n->h);
                    }
                    uint32_t count = leaves[k].count;
                    for (int c = 0; c < 3; c++) {
                        int prev = 0;
                        for (uint32_t i = 0; i < count; i++) {
                            int value = residual[k][(size_t)c * count + i];
                            int neighbour = c > 0
                                ? residual[k][(size_t)(c - 1) * count + i] : prev;
                            prev = value;
                            nvdr_enc_residual(&ae, &models, value,
                                              split_ctx[k][i], c,
                                              nvdr_prev_context(neighbour));
                        }
                    }
                }

                if (nvdr_enc_finish(&ae) != 0) { nvdr_enc_free(&ae); goto write_done; }

                stored_size[k] = prefix + ae.count;
                stored[k] = (uint8_t*)malloc(stored_size[k]);
                if (!stored[k]) { nvdr_enc_free(&ae); goto write_done; }
                if (prefix) memcpy(stored[k], raw[k], prefix);
                memcpy(stored[k] + prefix, ae.bytes, ae.count);
                nvdr_enc_free(&ae);
            }
        } else {
            for (int k = 0; k < NVDR_LEVELS; k++) {
                uLongf bound = compressBound((uLong)raw_size[k]);
                stored[k] = (uint8_t*)malloc(bound);
                if (!stored[k]) goto write_done;
                if (compress2(stored[k], &bound, raw[k], (uLong)raw_size[k], 9) != Z_OK)
                    goto write_done;
                stored_size[k] = bound;
            }
        }

        FILE* f = fopen(out_path, "wb");
        if (!f) goto write_done;

        uint8_t header[NVDR_HEADER_SIZE];
        memset(header, 0, sizeof(header));
        memcpy(header, NVDR_MAGIC, 4);
        header[4] = NVDR_VERSION;
        header[5] = (uint8_t)cfg->codec;
        put_u16(header + 6, (uint16_t)img->width);
        put_u16(header + 8, (uint16_t)img->height);
        header[10] = (uint8_t)cfg->anchor_bits;
        for (int k = 0; k < NVDR_LEVELS; k++) header[11 + k] = (uint8_t)cfg->step[k];
        for (int k = 0; k < NVDR_LEVELS; k++) {
            put_u32(header + 16 + k * 4, leaves[k].count);
            put_u32(header + 28 + k * 4, (uint32_t)bits[k].bit_count);
            put_u32(header + 40 + k * 4, (uint32_t)raw_size[k]);
            put_u32(header + 52 + k * 4, (uint32_t)stored_size[k]);
        }
        fwrite(header, 1, sizeof(header), f);
        for (int k = 0; k < NVDR_LEVELS; k++)
            fwrite(stored[k], 1, stored_size[k], f);
        fclose(f);
        wrote = 1;

        if (hdr_out) {
            memset(hdr_out, 0, sizeof(*hdr_out));
            hdr_out->width = (uint16_t)img->width;
            hdr_out->height = (uint16_t)img->height;
            hdr_out->anchor_bits = (uint8_t)cfg->anchor_bits;
            hdr_out->compression = (uint8_t)cfg->codec;
            for (int k = 0; k < NVDR_LEVELS; k++) {
                hdr_out->leaf_count[k]   = leaves[k].count;
                hdr_out->split_bits[k]   = (uint32_t)bits[k].bit_count;
                hdr_out->raw_bytes[k]    = (uint32_t)raw_size[k];
                hdr_out->stored_bytes[k] = (uint32_t)stored_size[k];
                hdr_out->step[k]         = (uint8_t)cfg->step[k];
            }
        }

write_done:
        for (int k = 0; k < NVDR_LEVELS; k++) { free(raw[k]); free(stored[k]); }
        if (!wrote) goto done;
    }
    rc = 0;

done:
    for (int k = 0; k < NVDR_LEVELS; k++) {
        free(bits[k].bytes);
        free(leaves[k].items);
        free(recon[k]);
        free(residual[k]);
        free(split_ctx[k]);
    }
    free(palette);
    free(tokens);
    nvdr_tree_free(&tree);
    return rc;
}

/* ============================================================== decoder */

typedef struct {
    NvdrLevelData* out;
    uint32_t       capacity;
} RectSink;

static int sink_push(RectSink* sink, int x, int y, int w, int h) {
    NvdrLevelData* d = sink->out;
    if (d->count == sink->capacity) {
        uint32_t grown = sink->capacity ? sink->capacity * 2 : 4096;
        uint16_t* nx = (uint16_t*)realloc(d->x, (size_t)grown * sizeof(uint16_t));
        uint16_t* ny = (uint16_t*)realloc(d->y, (size_t)grown * sizeof(uint16_t));
        uint16_t* nw = (uint16_t*)realloc(d->w, (size_t)grown * sizeof(uint16_t));
        uint16_t* nh = (uint16_t*)realloc(d->h, (size_t)grown * sizeof(uint16_t));
        if (nx) d->x = nx;
        if (ny) d->y = ny;
        if (nw) d->w = nw;
        if (nh) d->h = nh;
        if (!nx || !ny || !nw || !nh) return -1;
        sink->capacity = grown;
    }
    d->x[d->count] = (uint16_t)x;
    d->y[d->count] = (uint16_t)y;
    d->w[d->count] = (uint16_t)w;
    d->h[d->count] = (uint16_t)h;
    d->count++;
    return 0;
}

/* The mirror of cut_level: one bit decides split or stop, geometry is
 * recomputed rather than read. */
static int replay(BitReader* br, RectSink* sink, int x, int y, int w, int h) {
    if (br->overrun) return -1;
    if (!br_get(br)) return sink_push(sink, x, y, w, h);
    int hw = w / 2, hh = h / 2, rw = w - hw, rh = h - hh;
    if (replay(br, sink, x,      y,      hw, hh) != 0) return -1;
    if (replay(br, sink, x + hw, y,      rw, hh) != 0) return -1;
    if (replay(br, sink, x,      y + hh, hw, rh) != 0) return -1;
    if (replay(br, sink, x + hw, y + hh, rw, rh) != 0) return -1;
    return 0;
}

/* The mirror of transcode_split: same walk, same area contexts. */
static int replay_arith(NvdrDecoder* dec, NvdrModels* m, RectSink* sink,
                        int x, int y, int w, int h) {
    if (dec->overrun) return -1;
    if (!nvdr_dec_bit(dec, &m->split[nvdr_area_context(w, h)]))
        return sink_push(sink, x, y, w, h);
    int hw = w / 2, hh = h / 2, rw = w - hw, rh = h - hh;
    if (replay_arith(dec, m, sink, x,      y,      hw, hh) != 0) return -1;
    if (replay_arith(dec, m, sink, x + hw, y,      rw, hh) != 0) return -1;
    if (replay_arith(dec, m, sink, x,      y + hh, hw, rh) != 0) return -1;
    if (replay_arith(dec, m, sink, x + hw, y + hh, rw, rh) != 0) return -1;
    return 0;
}

void nvdr_pyramid_free(NvdrPyramid* pyr) {
    for (int k = 0; k < NVDR_LEVELS; k++) {
        free(pyr->level[k].x); free(pyr->level[k].y);
        free(pyr->level[k].w); free(pyr->level[k].h);
        free(pyr->level[k].rgb);
    }
    free(pyr->palette);
    memset(pyr, 0, sizeof(*pyr));
}

/*
 * Read one level's stream: `stored_bytes` compressed bytes off disk,
 * inflated to exactly `raw_bytes`. Returns NULL when the bytes are not all
 * there, which is the ordinary outcome for a truncated file rather than an
 * error — the caller simply stops at the previous level.
 */
static uint8_t* read_stream(FILE* f, long* available, uint32_t stored_bytes,
                            uint32_t raw_bytes, uint8_t compression) {
    if (stored_bytes == 0 || *available < (long)stored_bytes) return NULL;

    uint8_t* packed = (uint8_t*)malloc(stored_bytes);
    if (!packed) return NULL;
    if (fread(packed, 1, stored_bytes, f) != stored_bytes) { free(packed); return NULL; }
    *available -= (long)stored_bytes;

    if (compression != NVDR_COMPRESS_DEFLATE) return packed;

    uint8_t* raw = (uint8_t*)malloc(raw_bytes ? raw_bytes : 1);
    if (!raw) { free(packed); return NULL; }
    uLongf produced = raw_bytes;
    int rc = uncompress(raw, &produced, packed, (uLong)stored_bytes);
    free(packed);
    if (rc != Z_OK || produced != raw_bytes) { free(raw); return NULL; }
    return raw;
}

int nvdr_decode_file(const char* path, NvdrPyramid* pyr, NvdrHeader* hdr) {
    memset(pyr, 0, sizeof(*pyr));

    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t header[NVDR_HEADER_SIZE];
    if (file_size < NVDR_HEADER_SIZE ||
        fread(header, 1, sizeof(header), f) != sizeof(header) ||
        memcmp(header, NVDR_MAGIC, 4) != 0 || header[4] != NVDR_VERSION) {
        fclose(f);
        return -1;
    }

    memset(hdr, 0, sizeof(*hdr));
    hdr->width       = get_u16(header + 6);
    hdr->height      = get_u16(header + 8);
    hdr->anchor_bits = header[10];
    hdr->compression = header[5];
    for (int k = 0; k < NVDR_LEVELS; k++) {
        hdr->step[k]         = header[11 + k];
        hdr->leaf_count[k]   = get_u32(header + 16 + k * 4);
        hdr->split_bits[k]   = get_u32(header + 28 + k * 4);
        hdr->raw_bytes[k]    = get_u32(header + 40 + k * 4);
        hdr->stored_bytes[k] = get_u32(header + 52 + k * 4);
    }
    pyr->anchor_bits = hdr->anchor_bits;
    for (int k = 0; k < NVDR_LEVELS; k++) pyr->step[k] = hdr->step[k];

    long available = file_size - NVDR_HEADER_SIZE;

    /* --- level 0 is the contract; without it there is no picture --- */
    uint8_t* stream = read_stream(f, &available, hdr->stored_bytes[0],
                                  hdr->raw_bytes[0], hdr->compression);
    if (!stream) { fclose(f); return -1; }

    size_t off = 0;
    pyr->palette_count = (int)stream[off++] + 1;   /* stored biased by one */
    pyr->palette = (unsigned char*)malloc((size_t)pyr->palette_count * 3);
    if (!pyr->palette) { free(stream); fclose(f); return -1; }
    memcpy(pyr->palette, stream + off, (size_t)pyr->palette_count * 3);
    off += (size_t)pyr->palette_count * 3;

    RectSink sink = { &pyr->level[0], 0 };
    uint32_t* tokens = (uint32_t*)malloc((size_t)hdr->leaf_count[0] * sizeof(uint32_t));
    if (!tokens) { free(stream); fclose(f); nvdr_pyramid_free(pyr); return -1; }

    if (hdr->compression == NVDR_COMPRESS_ARITH) {
        NvdrModels models;
        NvdrDecoder ad;
        nvdr_models_init(&models);
        nvdr_dec_init(&ad, stream + off, hdr->stored_bytes[0] - off);
        if (replay_arith(&ad, &models, &sink, 0, 0, hdr->width, hdr->height) != 0 ||
            ad.overrun || pyr->level[0].count != hdr->leaf_count[0]) {
            free(tokens); free(stream); fclose(f); nvdr_pyramid_free(pyr); return -1;
        }
        for (uint32_t i = 0; i < hdr->leaf_count[0]; i++)
            tokens[i] = nvdr_dec_tree(&ad, models.token, hdr->anchor_bits);
        if (ad.overrun) {
            free(tokens); free(stream); fclose(f); nvdr_pyramid_free(pyr); return -1;
        }
    } else {
        BitReader br = { stream + off, hdr->split_bits[0], 0, 0 };
        if (replay(&br, &sink, 0, 0, hdr->width, hdr->height) != 0 || br.overrun ||
            pyr->level[0].count != hdr->leaf_count[0]) {
            free(tokens); free(stream); fclose(f); nvdr_pyramid_free(pyr); return -1;
        }
        off += (hdr->split_bits[0] + 7) / 8;
        for (uint32_t i = 0; i < hdr->leaf_count[0]; i++) {
            size_t bit = (size_t)i * hdr->anchor_bits;
            uint32_t token = 0;
            for (int b = 0; b < hdr->anchor_bits; b++)
                if (stream[off + ((bit + b) >> 3)] & (1u << ((bit + b) & 7)))
                    token |= (1u << b);
            tokens[i] = token;
        }
    }

    pyr->level[0].rgb = (uint8_t*)malloc((size_t)pyr->level[0].count * 3);
    if (!pyr->level[0].rgb) {
        free(tokens); free(stream); fclose(f); nvdr_pyramid_free(pyr); return -1;
    }
    for (uint32_t i = 0; i < pyr->level[0].count; i++) {
        uint32_t token = tokens[i] < (uint32_t)pyr->palette_count ? tokens[i] : 0;
        memcpy(pyr->level[0].rgb + (size_t)i * 3, pyr->palette + token * 3, 3);
    }
    free(tokens);
    free(stream);
    pyr->levels_present = 1;

    /* --- every further level is a bonus the bytes may not have paid for -- */
    for (int k = 1; k < NVDR_LEVELS; k++) {
        uint8_t* buf = read_stream(f, &available, hdr->stored_bytes[k],
                                   hdr->raw_bytes[k], hdr->compression);
        if (!buf) break;

        BitReader lbr = { buf, hdr->split_bits[k], 0, 0 };
        RectSink lsink = { &pyr->level[k], 0 };
        const NvdrLevelData* prev = &pyr->level[k - 1];
        const int arith = hdr->compression == NVDR_COMPRESS_ARITH;

        NvdrModels models;
        NvdrDecoder ad;
        if (arith) {
            nvdr_models_init(&models);
            nvdr_dec_init(&ad, buf, hdr->stored_bytes[k]);
        }

        /* Track which previous rectangle each new one came from, so the
         * delta has the colour that was on screen there as its base, and
         * whether that rectangle subdivided, which is the entropy
         * context — derived here, never read from the file. */
        uint8_t* base = (uint8_t*)malloc((size_t)hdr->leaf_count[k] * 3);
        uint8_t* split_ctx = (uint8_t*)malloc(hdr->leaf_count[k]);
        if (!base || !split_ctx) { free(base); free(split_ctx); free(buf); break; }

        int failed = 0;
        for (uint32_t i = 0; i < prev->count && !failed; i++) {
            uint32_t before = pyr->level[k].count;
            int rc = arith
                ? replay_arith(&ad, &models, &lsink, prev->x[i], prev->y[i],
                               prev->w[i], prev->h[i])
                : replay(&lbr, &lsink, prev->x[i], prev->y[i],
                         prev->w[i], prev->h[i]);
            if (rc != 0 || (arith ? ad.overrun : lbr.overrun)) { failed = 1; break; }

            uint32_t produced = pyr->level[k].count - before;
            for (uint32_t j = before; j < pyr->level[k].count; j++) {
                if (j >= hdr->leaf_count[k]) { failed = 1; break; }
                memcpy(base + (size_t)j * 3, prev->rgb + (size_t)i * 3, 3);
                split_ctx[j] = (uint8_t)(produced > 1);
            }
        }
        if (failed || pyr->level[k].count != hdr->leaf_count[k]) {
            free(base); free(split_ctx); free(buf);
            free(pyr->level[k].x); free(pyr->level[k].y);
            free(pyr->level[k].w); free(pyr->level[k].h);
            memset(&pyr->level[k], 0, sizeof(pyr->level[k]));
            break;
        }

        uint32_t n = pyr->level[k].count;
        int8_t* residual = (int8_t*)malloc((size_t)n * 3);
        pyr->level[k].rgb = (uint8_t*)malloc((size_t)n * 3);
        if (!residual || !pyr->level[k].rgb) {
            free(residual); free(base); free(split_ctx); free(buf); break;
        }

        if (arith) {
            for (int c = 0; c < 3; c++) {
                int prev = 0;
                for (uint32_t i = 0; i < n; i++) {
                    int neighbour = c > 0
                        ? residual[(size_t)(c - 1) * n + i] : prev;
                    int value = nvdr_dec_residual(&ad, &models, split_ctx[i], c,
                                                  nvdr_prev_context(neighbour));
                    residual[(size_t)c * n + i] = (int8_t)value;
                    prev = value;
                }
            }
            if (ad.overrun) {
                free(residual); free(base); free(split_ctx); free(buf);
                free(pyr->level[k].rgb);
                free(pyr->level[k].x); free(pyr->level[k].y);
                free(pyr->level[k].w); free(pyr->level[k].h);
                memset(&pyr->level[k], 0, sizeof(pyr->level[k]));
                break;
            }
        } else {
            memcpy(residual, buf + (hdr->split_bits[k] + 7) / 8, (size_t)n * 3);
        }

        for (uint32_t i = 0; i < n; i++) {
            for (int c = 0; c < 3; c++) {
                int value = (int)base[(size_t)i * 3 + c] +
                            (int)residual[(size_t)c * n + i] * hdr->step[k];
                pyr->level[k].rgb[(size_t)i * 3 + c] = (uint8_t)clamp_u8(value);
            }
        }

        free(residual);
        free(base);
        free(split_ctx);
        free(buf);
        pyr->levels_present = k + 1;
    }

    fclose(f);
    return 0;
}

/* =============================================================== render */

void nvdr_render_level(const NvdrLevelData* level, NvdrImage* out) {
    for (uint32_t i = 0; i < level->count; i++) {
        const uint8_t* rgb = level->rgb + (size_t)i * 3;
        int x1 = level->x[i] + level->w[i];
        int y1 = level->y[i] + level->h[i];
        if (x1 > out->width)  x1 = out->width;
        if (y1 > out->height) y1 = out->height;
        for (int y = level->y[i]; y < y1; y++) {
            unsigned char* row = out->pixels + ((size_t)y * out->width + level->x[i]) * 3;
            for (int x = level->x[i]; x < x1; x++, row += 3) {
                row[0] = rgb[0]; row[1] = rgb[1]; row[2] = rgb[2];
            }
        }
    }
}
