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
    cfg.weber        = 64.0f;
    cfg.texture      = 0.0f;   /* off: see the crossover in the README */
    cfg.order        = NVDR_ORDER_AREA;
    cfg.chroma       = 2;
    /*
     * Ramps on. Rate-matched against the flat encoder across the sample
     * set they are worth +0.07 to +1.08 dB, and they get there with fewer
     * rectangles, so there are fewer seams to smooth as well.
     *
     * The slope step is deliberately coarse. The slope is coded roughly
     * unary, so its cost grows with its magnitude, and a fine step prices
     * the large ramps — the ones that actually explain a region — out of
     * the rate-distortion test. Swept at matched rate: step 16 lands 0.3 dB
     * above step 2 and stays flat from there.
     */
    /*
     * 280 was calibrated against the loosened tolerances that came with
     * it and did not survive them. A ramp pays off in proportion to the
     * area it covers, so at the tight tolerances the rectangles are small
     * and each ramp explains less; the same lambda then buys quality at a
     * rate the flat encoder was not asked for. 600 gains everywhere
     * measured — +0.40 to +2.43 dB — for 2 to 33% more bytes.
     */
    cfg.gradient     = 600.0f;
    cfg.gradient_step = 16;
    /*
     * These were briefly loosened to 0.140/0.070/0.040, on the grounds
     * that a rectangle able to ramp can stop subdividing earlier. Measured
     * on six photographs it looked like a straight gain. It was not: those
     * three numbers are one step apart, so the change shifted the whole
     * pyramid down a level — the new level 2 came out with exactly the
     * rectangle count the old level 1 had — and threw away a level of
     * refinement.
     *
     * What hid it is that tolerance behaves completely differently
     * depending on whether an image's tree saturates. Between tolerance
     * 0.060 and 0.040 a star field goes from 4507 rectangles to 103891,
     * because noise has detail at every scale and there is no tolerance at
     * which it is resolved; the knob is a smooth, powerful rate dial and
     * every setting looks reasonable. A picture of flat shapes goes from
     * 1474 to 1744 over the same interval and from 16 to 16 across the
     * entire range, because its structure is finite. There the knob is a
     * cliff: below saturation it buys nothing, above it the edges — which
     * carry half the squared error in 1% of the pixels — are destroyed.
     * `circulos` lost 5.5 dB to save 7% of its bytes.
     *
     * So this is a rate control that has to stay separate from the ramp,
     * and it stays where it was measured to belong.
     */
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
static float region_stats_full(const NvdrImage* img, int x, int y, int w, int h,
                               float weber, float pivot, double* out_raw,
                               uint8_t* out_r, uint8_t* out_g, uint8_t* out_b) {
    int x1 = x + w > img->width  ? img->width  : x + w;
    int y1 = y + h > img->height ? img->height : y + h;
    if (x1 <= x || y1 <= y) {
        *out_r = *out_g = *out_b = 0;
        if (out_raw) *out_raw = 0.0;
        return 0.0f;
    }

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
    /* Weber normalisation: the same absolute deviation counts for more in
     * a dark region than a bright one, which is the whole difference
     * between "this region is uniform" and "this region looks uniform". */
    double mean_deviation = deviation / count;
    if (out_raw) *out_raw = mean_deviation;

    double luma = 0.30 * mr + 0.59 * mg + 0.11 * mb;
    double denominator = 255.0 * (luma + weber) / (pivot + weber);
    return (float)(mean_deviation / denominator);
}

/* --------------------------------------------------------- texture map --
 *
 * How much of a region's variation is fine grain rather than structure.
 *
 * Deviation alone cannot tell distant grass from a face: both are far from
 * uniform. What separates them is the scale of the variation. Grass varies
 * as much inside a 4x4 window as it does across the whole patch, so
 * subdividing reproduces noise; a face varies mostly across the region and
 * barely within a window, so subdividing resolves it.
 *
 * Two earlier attempts measured this by splitting a node and looking at how
 * much its children's deviation dropped. Both failed, for the same reason:
 * that drop is confounded with scale. It is small at the top of the tree
 * whatever the content, so a threshold scaled by it only bit the middle of
 * the range, and a hard floor on it pruned the root and collapsed the
 * image to one rectangle.
 *
 * A 4x4 deviation map answers the question directly and at a fixed scale.
 * Summed once into an integral image, any region's mean local deviation is
 * an O(1) query however large the region.
 */
typedef struct {
    double* sat;        /* (bw+1) x (bh+1) integral image over block deviations */
    int     bw, bh;     /* block grid size */
} NvdrTextureMap;

#define NVDR_TEXTURE_BLOCK 4

static void texture_map_free(NvdrTextureMap* map) {
    free(map->sat);
    map->sat = NULL;
}

static int texture_map_build(NvdrTextureMap* map, const NvdrImage* img) {
    map->bw = (img->width  + NVDR_TEXTURE_BLOCK - 1) / NVDR_TEXTURE_BLOCK;
    map->bh = (img->height + NVDR_TEXTURE_BLOCK - 1) / NVDR_TEXTURE_BLOCK;
    map->sat = (double*)calloc((size_t)(map->bw + 1) * (map->bh + 1), sizeof(double));
    if (!map->sat) return -1;

    for (int by = 0; by < map->bh; by++) {
        for (int bx = 0; bx < map->bw; bx++) {
            int x0 = bx * NVDR_TEXTURE_BLOCK, y0 = by * NVDR_TEXTURE_BLOCK;
            int x1 = x0 + NVDR_TEXTURE_BLOCK, y1 = y0 + NVDR_TEXTURE_BLOCK;
            if (x1 > img->width)  x1 = img->width;
            if (y1 > img->height) y1 = img->height;

            double sr = 0, sg = 0, sb = 0;
            int n = 0;
            for (int y = y0; y < y1; y++)
                for (int x = x0; x < x1; x++) {
                    const unsigned char* p = pixel_at(img, x, y);
                    sr += p[0]; sg += p[1]; sb += p[2]; n++;
                }
            double dev = 0.0;
            if (n) {
                double mr = sr / n, mg = sg / n, mb = sb / n;
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++) {
                        const unsigned char* p = pixel_at(img, x, y);
                        dev += fabs(p[0] - mr) * 0.30 + fabs(p[1] - mg) * 0.59
                             + fabs(p[2] - mb) * 0.11;
                    }
                dev /= n;
            }
            size_t i = (size_t)(by + 1) * (map->bw + 1) + (bx + 1);
            map->sat[i] = dev;
        }
    }

    for (int by = 1; by <= map->bh; by++)
        for (int bx = 1; bx <= map->bw; bx++) {
            size_t w1 = (size_t)(map->bw + 1);
            map->sat[by * w1 + bx] += map->sat[by * w1 + bx - 1]
                                    + map->sat[(by - 1) * w1 + bx]
                                    - map->sat[(by - 1) * w1 + bx - 1];
        }
    return 0;
}

static double texture_map_mean(const NvdrTextureMap* map, int x, int y, int w, int h) {
    int bx0 = x / NVDR_TEXTURE_BLOCK, by0 = y / NVDR_TEXTURE_BLOCK;
    int bx1 = (x + w + NVDR_TEXTURE_BLOCK - 1) / NVDR_TEXTURE_BLOCK;
    int by1 = (y + h + NVDR_TEXTURE_BLOCK - 1) / NVDR_TEXTURE_BLOCK;
    if (bx1 > map->bw) bx1 = map->bw;
    if (by1 > map->bh) by1 = map->bh;
    if (bx1 <= bx0 || by1 <= by0) return 0.0;

    size_t w1 = (size_t)(map->bw + 1);
    double sum = map->sat[(size_t)by1 * w1 + bx1]
               - map->sat[(size_t)by0 * w1 + bx1]
               - map->sat[(size_t)by1 * w1 + bx0]
               + map->sat[(size_t)by0 * w1 + bx0];
    return sum / ((double)(bx1 - bx0) * (double)(by1 - by0));
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
                          const NvdrConfig* cfg, float pivot,
                          const NvdrTextureMap* texture,
                          int32_t idx, int x, int y, int w, int h, int depth) {
    NvdrNode* node = &tree->nodes[idx];
    node->x = (uint16_t)x; node->y = (uint16_t)y;
    node->w = (uint16_t)w; node->h = (uint16_t)h;
    node->first_child = -1;

    double raw = 0.0;
    node->deviation = region_stats_full(img, x, y, w, h, cfg->weber, pivot, &raw,
                                        &node->r, &node->g, &node->b);

    /* The share of this region's variation that lives inside 4x4 windows.
     * Near 1 the region is fine grain and subdividing reproduces noise;
     * near 0 the variation is structure and subdividing resolves it. */
    node->penalty = 1.0f;
    if (cfg->texture > 0.0f && raw > 0.0) {
        double grain = texture_map_mean(texture, x, y, w, h) / raw;
        if (grain > 1.0) grain = 1.0;
        if (grain < 0.0) grain = 0.0;
        node->penalty = 1.0f + cfg->texture * (float)grain;
    }

    /* In fine grain the deviation never falls, so a scaled threshold never
     * bites and the tree descends to min_tile regardless — measured, the
     * high-grain band sat at 6.1 px per leaf whatever the penalty. What
     * stops it is raising the floor it descends to, so grain is spent at a
     * coarser resolution while structure keeps the full one. */
    int floor_tile = (int)(cfg->min_tile * node->penalty + 0.5f);
    if (floor_tile < cfg->min_tile) floor_tile = cfg->min_tile;

    int splittable = w > floor_tile && h > floor_tile && depth < cfg->max_depth;
    if (!splittable ||
        node->deviation <= cfg->tolerance[NVDR_LEVELS - 1]) return 0;

    int32_t first = tree_alloc(tree);
    if (first < 0) return -1;
    for (int i = 1; i < 4; i++) if (tree_alloc(tree) < 0) return -1;
    tree->nodes[idx].first_child = first;

    int hw = w / 2, hh = h / 2, rw = w - hw, rh = h - hh;
    int rc = 0;
    rc |= tree_build_rec(tree, img, cfg, pivot, texture, first + 0, x,      y,      hw, hh, depth + 1);
    rc |= tree_build_rec(tree, img, cfg, pivot, texture, first + 1, x + hw, y,      rw, hh, depth + 1);
    rc |= tree_build_rec(tree, img, cfg, pivot, texture, first + 2, x,      y + hh, hw, rh, depth + 1);
    rc |= tree_build_rec(tree, img, cfg, pivot, texture, first + 3, x + hw, y + hh, rw, rh, depth + 1);
    return rc;
}

int nvdr_tree_build(NvdrTree* tree, const NvdrImage* img, const NvdrConfig* cfg) {
    memset(tree, 0, sizeof(*tree));

    /* The Weber denominator pivots on the image's own mean luminance
     * rather than on mid-grey. Pivoting on a constant would tighten every
     * dark image and loosen every bright one, which is a quality setting
     * wearing a reallocation costume — measured, it cost macarrao.jpg
     * 1.35 dB. Pivoting on the image keeps the average tolerance where it
     * was and changes only how it is distributed. */
    double sum_luma = 0.0;
    size_t pixels = (size_t)img->width * img->height;
    for (size_t i = 0; i < pixels; i++) {
        const unsigned char* p = img->pixels + i * 3;
        sum_luma += 0.30 * p[0] + 0.59 * p[1] + 0.11 * p[2];
    }
    float pivot = pixels ? (float)(sum_luma / (double)pixels) : 128.0f;

    NvdrTextureMap texture;
    memset(&texture, 0, sizeof(texture));
    if (cfg->texture > 0.0f && texture_map_build(&texture, img) != 0) {
        nvdr_tree_free(tree);
        return -1;
    }

    int32_t root = tree_alloc(tree);
    if (root < 0) { texture_map_free(&texture); nvdr_tree_free(tree); return -1; }
    int rc = tree_build_rec(tree, img, cfg, pivot, &texture, root,
                            0, 0, img->width, img->height, 0);
    texture_map_free(&texture);
    if (rc != 0) { nvdr_tree_free(tree); return -1; }
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

/*
 * BT.601 in fixed point, the integer form both sides must agree on down to
 * the rounding. Floating point here would be a portability bug waiting to
 * happen: encoder and decoder have to land on the same byte.
 */
static void rgb_to_ycc(const uint8_t* rgb, uint8_t* ycc) {
    int r = rgb[0], g = rgb[1], b = rgb[2];
    ycc[0] = (uint8_t)clamp_u8((( 66 * r + 129 * g +  25 * b + 128) >> 8) + 16);
    ycc[1] = (uint8_t)clamp_u8(((-38 * r -  74 * g + 112 * b + 128) >> 8) + 128);
    ycc[2] = (uint8_t)clamp_u8(((112 * r -  94 * g -  18 * b + 128) >> 8) + 128);
}

static void ycc_to_rgb(const uint8_t* ycc, uint8_t* rgb) {
    int c = (int)ycc[0] - 16, d = (int)ycc[1] - 128, e = (int)ycc[2] - 128;
    rgb[0] = (uint8_t)clamp_u8((298 * c + 409 * e + 128) >> 8);
    rgb[1] = (uint8_t)clamp_u8((298 * c - 100 * d - 208 * e + 128) >> 8);
    rgb[2] = (uint8_t)clamp_u8((298 * c + 516 * d + 128) >> 8);
}

/* Luma keeps the level's step; the two chroma channels are coarser. */
static int channel_step(const NvdrConfig* cfg, int k, int channel) {
    if (cfg->chroma <= 0 || channel == 0) return cfg->step[k];
    return cfg->step[k] * cfg->chroma;
}
static int clamp_i8(int v) { return v < -127 ? -127 : (v > 127 ? 127 : v); }
static int div_round(int n, int d) {
    return n >= 0 ? (n + d / 2) / d : -((-n + d / 2) / d);
}

/*
 * The ramp, in integers so encoder, decoder and the JS port land on the
 * same byte.
 *
 * `span` is the total edge-to-edge change: position 0 sits at -span/2 and
 * position len-1 at +span/2, so the mean offset is zero and the DC the
 * residual already paid for stays the region mean. Floating point here
 * would be a portability bug — the ramp is evaluated per pixel on both
 * sides and has to agree exactly.
 */
static int ramp_offset(int span, int pos, int len) {
    if (span == 0 || len < 2) return 0;
    return div_round(span * (2 * pos - len + 1), 2 * (len - 1));
}

/* The colour one rectangle actually shows at a given pixel, ramp included.
 * Both the encoder and the decoder predict a child from this, so they have
 * to compute it the same way. */
static const int8_t nvdr_no_slope[3] = { 0, 0, 0 };

static void chain_sample(const uint8_t* chain, int axis, const int8_t* slope,
                         int slope_step, int rx, int ry, int rw, int rh,
                         int px, int py, uint8_t* out) {
    memcpy(out, chain, 3);
    if (!axis) return;
    int len = axis == 1 ? rw : rh;
    int pos = axis == 1 ? px - rx : py - ry;
    for (int c = 0; c < 3; c++)
        out[c] = (uint8_t)clamp_u8((int)out[c] +
                                   ramp_offset(slope[c] * slope_step, pos, len));
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
 * One refinement unit, emitted with its colours interleaved into its
 * geometry.
 *
 * The split bits used to go first and the residuals after. That made a
 * half-delivered unit worthless — the decoder had the shape and no
 * colours — so the whole unit was rolled back, and the truncation
 * granularity of a level was one unit rather than one rectangle. On an
 * image whose previous level has thirteen rectangles that meant thirteen
 * all-or-nothing chunks: a star field sat at 21.59 dB through the first
 * quarter of its container and only then moved.
 *
 * Interleaved, a cut anywhere leaves a valid partial subtree. The
 * rectangles that arrived keep their own colour; the region below the cut
 * falls back to what its parent was showing there. The symbols are the
 * same ones in the same per-model order, so the models adapt identically
 * and this costs nothing — only the interleaving changes, which is why it
 * still needed a container version.
 */
typedef struct {
    NvdrEncoder*    enc;
    NvdrModels*     m;
    const NvdrTree* tree;
    const int8_t*   residual;
    const uint8_t*  axis;
    const int8_t*   slope;
    float           tolerance;
    uint32_t        at;          /* next slot in the level's arrays */
    int             split_ctx;   /* the unit's root decision */
    int             prev0;
} EmitCtx;

static void emit_unit(EmitCtx* e, int32_t idx, int depth) {
    const NvdrNode* n = &e->tree->nodes[idx];
    int split = n->first_child >= 0 && n->deviation > e->tolerance;
    nvdr_enc_bit(e->enc, &e->m->split[nvdr_area_context(n->w, n->h)], split);
    if (depth == 0) e->split_ctx = split;

    if (split) {
        for (int i = 0; i < 4; i++) emit_unit(e, n->first_child + i, depth + 1);
        return;
    }

    uint32_t i = e->at++;
    for (int c = 0; c < 3; c++) {
        int value = e->residual[(size_t)i * 3 + c];
        int neighbour = c > 0 ? e->residual[(size_t)i * 3 + c - 1] : e->prev0;
        nvdr_enc_residual(e->enc, e->m, value, e->split_ctx, c,
                          nvdr_prev_context(neighbour));
    }
    e->prev0 = e->residual[(size_t)i * 3];

    if (e->axis) {
        int ctx = nvdr_area_context(n->w, n->h);
        int a = e->axis[i];
        nvdr_enc_bit(e->enc, &e->m->grad[ctx], a != 0);
        if (a) {
            nvdr_enc_bit(e->enc, &e->m->grad_axis[ctx], a == 2);
            for (int c = 0; c < 3; c++)
                nvdr_enc_slope(e->enc, e->m, e->slope[(size_t)i * 3 + c], c);
        }
    }
}

/*
 * Re-code a level's split bitstream through the arithmetic coder, giving
 * each bit the area context of the rectangle it decides. The tree shape is
 * recovered by walking it again rather than stored, so nothing extra is
 * carried: the walk is deterministic from the canvas rectangle down.
 *
 * Only level 0 uses this now; levels above it interleave through
 * emit_unit above.
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

/*
 * One refinement unit: a rectangle from the previous level, the split bits
 * that decide how it subdivides, and the rectangles it produces.
 *
 * The level used to be one stream — every split bit, then three planes of
 * residual — which made a truncated level worthless: a prefix gave the red
 * channel and no green or blue. Cut into units, a prefix is a whole number
 * of finished refinements, and the units that never arrived simply keep
 * the rectangle they had at the level before.
 */
typedef struct {
    uint32_t leaf_start, leaf_end;   /* range in the level's leaf array */
    size_t   bit_start, bit_end;     /* range in the level's split bitstream */
    uint32_t area;
    uint32_t index;                  /* position at the previous level */
} Unit;

/* Largest first, ties broken by position so both sides agree exactly. */
static int cmp_unit_area(const void* a, const void* b) {
    const Unit* ua = (const Unit*)a;
    const Unit* ub = (const Unit*)b;
    if (ua->area != ub->area) return ua->area > ub->area ? -1 : 1;
    return ua->index < ub->index ? -1 : (ua->index > ub->index);
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

/*
 * Decide flat or ramp for one rectangle, and fit the ramp if it wins.
 *
 * The DC is not up for negotiation here: it is the quantised region mean
 * the residual has already paid for, and the ramp has zero mean by
 * construction, so the two never fight over the same bits. What the ramp
 * has to justify is its own cost — an axis bit plus three slopes — against
 * the squared error it removes.
 *
 * Both axes are fitted by least squares and then re-evaluated with the
 * exact integer ramp the renderer will use, because the quantisation of
 * the slope is coarse enough that the fitted optimum and the coded one are
 * not always the same choice. A ramp that comes out all zeros costs four
 * bits and buys nothing, so the comparison rejects it on its own.
 *
 * Returns 0 for flat, 1 for a ramp along x, 2 along y.
 */
static int fit_ramp(const uint8_t* space, int stride, int x, int y, int w, int h,
                    const uint8_t* dc, int slope_step, double lambda,
                    int8_t* slope_out) {
    slope_out[0] = slope_out[1] = slope_out[2] = 0;
    if (w < 2 && h < 2) return 0;

    double flat_sse = 0.0;
    for (int yy = 0; yy < h; yy++) {
        const uint8_t* row = space + ((size_t)(y + yy) * stride + x) * 3;
        for (int xx = 0; xx < w; xx++, row += 3)
            for (int c = 0; c < 3; c++) {
                double d = (double)row[c] - (double)dc[c];
                flat_sse += d * d;
            }
    }

    int    best_axis = 0;
    double best_cost = flat_sse;
    int8_t best_slope[3] = { 0, 0, 0 };

    for (int axis = 1; axis <= 2; axis++) {
        int len = axis == 1 ? w : h;
        if (len < 2) continue;

        /* Least squares against t = 2*pos - len + 1, which is the ramp
         * shape without its scale. */
        double num[3] = { 0.0, 0.0, 0.0 }, den = 0.0;
        for (int yy = 0; yy < h; yy++) {
            const uint8_t* row = space + ((size_t)(y + yy) * stride + x) * 3;
            for (int xx = 0; xx < w; xx++, row += 3) {
                double t = (double)(2 * (axis == 1 ? xx : yy) - len + 1);
                den += t * t;
                for (int c = 0; c < 3; c++)
                    num[c] += ((double)row[c] - (double)dc[c]) * t;
            }
        }
        if (den <= 0.0) continue;

        int8_t q[3];
        int any = 0;
        for (int c = 0; c < 3; c++) {
            double span = 2.0 * (double)(len - 1) * num[c] / den;
            q[c] = (int8_t)clamp_i8(div_round((int)(span < 0 ? span - 0.5 : span + 0.5),
                                              slope_step));
            if (q[c]) any = 1;
        }
        if (!any) continue;

        double sse = 0.0;
        for (int yy = 0; yy < h; yy++) {
            const uint8_t* row = space + ((size_t)(y + yy) * stride + x) * 3;
            for (int xx = 0; xx < w; xx++, row += 3) {
                int pos = axis == 1 ? xx : yy;
                for (int c = 0; c < 3; c++) {
                    int fill = clamp_u8((int)dc[c] +
                        ramp_offset(q[c] * slope_step, pos, len));
                    double d = (double)row[c] - (double)fill;
                    sse += d * d;
                }
            }
        }

        /* One axis bit plus the three slopes; the flag bit is paid either
         * way, so it drops out of the comparison. */
        double bits = 1.0;
        for (int c = 0; c < 3; c++) bits += nvdr_slope_bits(q[c]);

        double cost = sse + lambda * bits;
        if (cost < best_cost) {
            best_cost = cost;
            best_axis = axis;
            memcpy(best_slope, q, 3);
        }
    }

    memcpy(slope_out, best_slope, 3);
    return best_axis;
}

int nvdr_encode_file(const char* out_path, const NvdrImage* img,
                     const NvdrConfig* cfg, NvdrHeader* hdr_out) {
    /* The deflate path writes its split bits in tree order, so it cannot
     * also carry a reordered emission. It exists for comparison, and the
     * comparison is fair either way. */
    NvdrConfig local = *cfg;
    if (local.codec != NVDR_COMPRESS_ARITH) {
        local.order = NVDR_ORDER_DFS;
        /* Ramps live in the coded stream, not in the raw one deflate
         * packs, so the comparison codec keeps every rectangle flat. */
        local.gradient = 0.0f;
    }
    if (local.gradient_step < 1) local.gradient_step = 1;
    cfg = &local;

    NvdrTree tree;
    if (nvdr_tree_build(&tree, img, cfg) != 0) return -1;

    BitWriter  bits[NVDR_LEVELS];
    IndexList  leaves[NVDR_LEVELS];
    uint8_t*   recon[NVDR_LEVELS];       /* reconstruction in the chain space */
    int8_t*    residual[NVDR_LEVELS];    /* 3 planes, level >= 1 only */
    uint8_t*   split_ctx[NVDR_LEVELS];   /* 1 when the parent subdivided */
    Unit*      units[NVDR_LEVELS];
    uint32_t   unit_count[NVDR_LEVELS];
    uint8_t*   axis[NVDR_LEVELS];        /* 0 flat, 1 ramp along x, 2 along y */
    int8_t*    slope[NVDR_LEVELS];       /* 3 per rectangle, chain space */
    unsigned char* palette = NULL;
    uint8_t*   tokens = NULL;
    uint8_t*   space_img = NULL;         /* the source in the chain's space */
    int rc = -1;

    memset(axis, 0, sizeof(axis));
    memset(slope, 0, sizeof(slope));
    memset(bits, 0, sizeof(bits));
    memset(leaves, 0, sizeof(leaves));
    memset(recon, 0, sizeof(recon));
    memset(residual, 0, sizeof(residual));
    memset(split_ctx, 0, sizeof(split_ctx));
    memset(units, 0, sizeof(units));
    memset(unit_count, 0, sizeof(unit_count));

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
        if (cfg->chroma > 0) rgb_to_ycc(palette + token * 3, recon[0] + (size_t)i * 3);
        else memcpy(recon[0] + (size_t)i * 3, palette + token * 3, 3);
    }

    /* The ramp is fitted against real pixels, and it has to be fitted in
     * the space the residuals run in — a slope in Y is not a slope in R. */
    if (cfg->gradient > 0.0f) {
        size_t pixels = (size_t)img->width * img->height;
        space_img = (uint8_t*)malloc(pixels * 3);
        if (!space_img) goto done;
        for (size_t p = 0; p < pixels; p++) {
            if (cfg->chroma > 0) rgb_to_ycc(img->pixels + p * 3, space_img + p * 3);
            else memcpy(space_img + p * 3, img->pixels + p * 3, 3);
        }
    }

    /* --- levels 1..N: re-cut each previous leaf at a finer tolerance --- */
    for (int k = 1; k < NVDR_LEVELS; k++) {
        uint32_t count = leaves[k - 1].count;
        units[k] = (Unit*)calloc(count ? count : 1, sizeof(Unit));
        if (!units[k]) goto done;
        unit_count[k] = count;

        /* Pass 1: cut every unit, recording where its leaves and its split
         * bits landed, so they can be emitted in a different order later. */
        for (uint32_t i = 0; i < count; i++) {
            const NvdrNode* parent = &tree.nodes[leaves[k - 1].items[i]];
            Unit* unit = &units[k][i];
            unit->index = i;
            unit->area = (uint32_t)parent->w * (uint32_t)parent->h;
            unit->leaf_start = leaves[k].count;
            unit->bit_start = bits[k].bit_count;
            if (cut_level(&tree, (int32_t)leaves[k - 1].items[i],
                          cfg->tolerance[k], &bits[k], &leaves[k]) != 0) goto done;
            unit->leaf_end = leaves[k].count;
            unit->bit_end = bits[k].bit_count;
        }

        if (cfg->order == NVDR_ORDER_AREA)
            qsort(units[k], count, sizeof(Unit), cmp_unit_area);

        /* Pass 2: rebuild the level in emission order. Everything
         * downstream — residuals, contexts, the next level's units — is
         * indexed by this order, and the decoder reproduces it by sorting
         * the rectangles it already has. */
        uint32_t total = leaves[k].count;
        uint32_t* ordered = (uint32_t*)malloc((size_t)(total ? total : 1) * sizeof(uint32_t));
        uint8_t* base = (uint8_t*)malloc((size_t)(total ? total : 1) * 3);
        split_ctx[k] = (uint8_t*)malloc(total ? total : 1);
        if (!ordered || !base || !split_ctx[k]) { free(ordered); free(base); goto done; }

        uint32_t at = 0;
        for (uint32_t u = 0; u < count; u++) {
            Unit* unit = &units[k][u];
            uint32_t produced = unit->leaf_end - unit->leaf_start;
            uint32_t new_start = at;
            const NvdrNode* parent = &tree.nodes[leaves[k - 1].items[unit->index]];
            for (uint32_t j = 0; j < produced; j++, at++) {
                ordered[at] = leaves[k].items[unit->leaf_start + j];
                const NvdrNode* child = &tree.nodes[ordered[at]];
                /* Predict the child from what the parent actually shows at
                 * the child's centre, ramp included. Re-deriving it here is
                 * what keeps a ramped parent from handing every child the
                 * same wrong colour. */
                chain_sample(recon[k - 1] + (size_t)unit->index * 3,
                             axis[k - 1] ? axis[k - 1][unit->index] : 0,
                             slope[k - 1] ? slope[k - 1] + (size_t)unit->index * 3
                                          : nvdr_no_slope,
                             cfg->gradient_step,
                             parent->x, parent->y, parent->w, parent->h,
                             child->x + child->w / 2, child->y + child->h / 2,
                             base + (size_t)at * 3);
                /* A rectangle whose parent subdivided carries a genuinely
                 * new colour; one whose parent did not carries a small
                 * correction to a colour already close. The coder is given
                 * that distinction, recomputed on the decoder side from the
                 * same split bits and never transmitted. */
                split_ctx[k][at] = (uint8_t)(produced > 1);
            }
            unit->leaf_start = new_start;
            unit->leaf_end = at;
        }
        free(leaves[k].items);
        leaves[k].items = ordered;
        leaves[k].capacity = total;

        recon[k] = (uint8_t*)malloc((size_t)(total ? total : 1) * 3);
        residual[k] = (int8_t*)malloc((size_t)(total ? total : 1) * 3);
        if (!recon[k] || !residual[k]) { free(base); goto done; }

        for (uint32_t i = 0; i < total; i++) {
            const NvdrNode* n = &tree.nodes[leaves[k].items[i]];
            uint8_t node_rgb[3] = { n->r, n->g, n->b };
            uint8_t target[3];
            if (cfg->chroma > 0) rgb_to_ycc(node_rgb, target);
            else memcpy(target, node_rgb, 3);

            for (int c = 0; c < 3; c++) {
                int step = channel_step(cfg, k, c);
                int delta = (int)target[c] - (int)base[(size_t)i * 3 + c];
                int q = clamp_i8(div_round(delta, step));
                /* Interleaved by rectangle rather than planar: a prefix has
                 * to end on a whole rectangle, and the coder's contexts are
                 * explicit so the grouping costs it nothing. */
                residual[k][(size_t)i * 3 + c] = (int8_t)q;
                recon[k][(size_t)i * 3 + c] =
                    (uint8_t)clamp_u8((int)base[(size_t)i * 3 + c] + q * step);
            }
        }
        free(base);

        if (cfg->gradient > 0.0f) {
            axis[k] = (uint8_t*)calloc(total ? total : 1, 1);
            slope[k] = (int8_t*)calloc((size_t)(total ? total : 1) * 3, 1);
            if (!axis[k] || !slope[k]) goto done;
            for (uint32_t i = 0; i < total; i++) {
                const NvdrNode* n = &tree.nodes[leaves[k].items[i]];
                axis[k][i] = (uint8_t)fit_ramp(space_img, img->width,
                                               n->x, n->y, n->w, n->h,
                                               recon[k] + (size_t)i * 3,
                                               cfg->gradient_step,
                                               (double)cfg->gradient,
                                               slope[k] + (size_t)i * 3);
            }
        }
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
            /* Ramps only exist on the arith path, where nothing parses
             * `raw` — but it is still the number the report calls the
             * stream before entropy coding, so it has to include them. */
            size_t ramp_bytes = axis[k] ? (size_t)leaves[k].count * 4 : 0;
            raw_size[k] = bits_bytes + residual_bytes + ramp_bytes;
            raw[k] = (uint8_t*)malloc(raw_size[k]);
            if (!raw[k]) goto write_done;
            memcpy(raw[k], bits[k].bytes, bits_bytes);
            memcpy(raw[k] + bits_bytes, residual[k], residual_bytes);
            if (ramp_bytes) {
                uint8_t* at = raw[k] + bits_bytes + residual_bytes;
                for (uint32_t i = 0; i < leaves[k].count; i++) {
                    *at++ = axis[k][i];
                    memcpy(at, slope[k] + (size_t)i * 3, 3);
                    at += 3;
                }
            }
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
                    /* One unit at a time: its split bits, then the
                     * residuals of every rectangle it produced. A decoder
                     * that runs out of bytes stops on a unit boundary with
                     * everything before it whole. */
                    EmitCtx e;
                    memset(&e, 0, sizeof(e));
                    e.enc = &ae;
                    e.m = &models;
                    e.tree = &tree;
                    e.residual = residual[k];
                    e.axis = axis[k];
                    e.slope = slope[k];
                    e.tolerance = cfg->tolerance[k];
                    for (uint32_t u = 0; u < unit_count[k]; u++) {
                        const Unit* unit = &units[k][u];
                        e.at = unit->leaf_start;
                        emit_unit(&e, (int32_t)leaves[k - 1].items[unit->index], 0);
                        if (e.at != unit->leaf_end) goto write_done;
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
        header[14] = (uint8_t)cfg->order;
        header[15] = (uint8_t)(cfg->chroma > 0 ? NVDR_SPACE_YCC : NVDR_SPACE_RGB);
        /* Zero means every rectangle is flat and no ramp data is coded. */
        header[67] = (uint8_t)(cfg->gradient > 0.0f ? cfg->gradient_step : 0);
        for (int k = 0; k < NVDR_LEVELS; k++)
            header[64 + k] = (uint8_t)channel_step(cfg, k, 1);
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
            hdr_out->order = (uint8_t)cfg->order;
            hdr_out->space = (uint8_t)(cfg->chroma > 0 ? NVDR_SPACE_YCC : NVDR_SPACE_RGB);
            hdr_out->gradient_step = (uint8_t)(cfg->gradient > 0.0f ? cfg->gradient_step : 0);
            for (int k = 0; k < NVDR_LEVELS; k++)
                hdr_out->chroma_step[k] = (uint8_t)channel_step(cfg, k, 1);
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
        free(units[k]);
        free(axis[k]);
        free(slope[k]);
    }
    free(palette);
    free(tokens);
    free(space_img);
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

/*
 * The mirror of emit_unit: one unit, geometry and colour interleaved.
 *
 * Once the bytes run out the walk stops reading and paints the rest of the
 * region from what the parent was showing there, so a unit cut in half
 * still contributes everything that arrived. The overrun flag is checked
 * immediately after every symbol and the symbol that tripped it is thrown
 * away, which bounds the damage of a cut to the few rectangles inside the
 * coder's five-byte lookahead.
 */
typedef struct {
    NvdrDecoder*         dec;
    NvdrModels*          m;
    RectSink*            sink;
    NvdrLevelData*       level;
    const NvdrLevelData* prev;
    uint32_t             prev_index;
    uint8_t*             chain;
    uint8_t*             axis;
    int8_t*              slope;
    size_t               cap;
    int                  step, chroma_step, slope_step;
    int                  ramped;
    int                  split_ctx;
    int                  prev0;
    int                  stopped;
    uint32_t             delivered;   /* rectangles whose colour really arrived */
} UnitCtx;

/* What the parent shows at a point, which is what an undelivered region
 * falls back to and what a delivered one is predicted from. */
static void unit_parent_colour(const UnitCtx* d, int px, int py, uint8_t* out) {
    uint32_t i = d->prev_index;
    chain_sample(d->prev->chain + (size_t)i * 3,
                 d->prev->axis ? d->prev->axis[i] : 0,
                 d->prev->slope ? d->prev->slope + (size_t)i * 3 : nvdr_no_slope,
                 d->prev->slope_step,
                 d->prev->x[i], d->prev->y[i], d->prev->w[i], d->prev->h[i],
                 px, py, out);
}

static int replay_unit(UnitCtx* d, int x, int y, int w, int h, int depth) {
    int split = 0;
    if (!d->stopped) {
        if ((size_t)d->level->count + 1 >= d->cap) {
            d->stopped = 1;
        } else {
            split = nvdr_dec_bit(d->dec, &d->m->split[nvdr_area_context(w, h)]);
            if (d->dec->overrun) { d->stopped = 1; split = 0; }
            else if (depth == 0) d->split_ctx = split;
        }
    }

    if (split) {
        int hw = w / 2, hh = h / 2, rw = w - hw, rh = h - hh;
        if (replay_unit(d, x,      y,      hw, hh, depth + 1) != 0) return -1;
        if (replay_unit(d, x + hw, y,      rw, hh, depth + 1) != 0) return -1;
        if (replay_unit(d, x,      y + hh, hw, rh, depth + 1) != 0) return -1;
        if (replay_unit(d, x + hw, y + hh, rw, rh, depth + 1) != 0) return -1;
        return 0;
    }

    if (sink_push(d->sink, x, y, w, h) != 0) return -1;
    uint32_t j = d->level->count - 1;
    uint8_t parent[3];
    unit_parent_colour(d, x + w / 2, y + h / 2, parent);

    if (d->stopped) {
        /* Never arrived: show what the level before showed here. The
         * parent's ramp is sampled at this rectangle's centre rather than
         * carried, so the fallback is flat but continuous with it. */
        memcpy(d->chain + (size_t)j * 3, parent, 3);
        if (d->ramped) d->axis[j] = 0;
        return 0;
    }

    int8_t raw[3];
    for (int c = 0; c < 3; c++) {
        int neighbour = c > 0 ? raw[c - 1] : d->prev0;
        raw[c] = (int8_t)nvdr_dec_residual(d->dec, d->m, d->split_ctx, c,
                                           nvdr_prev_context(neighbour));
    }
    if (d->dec->overrun) {
        d->stopped = 1;
        memcpy(d->chain + (size_t)j * 3, parent, 3);
        if (d->ramped) d->axis[j] = 0;
        return 0;
    }
    d->prev0 = raw[0];

    for (int c = 0; c < 3; c++)
        d->chain[(size_t)j * 3 + c] = (uint8_t)clamp_u8(
            (int)parent[c] + (int)raw[c] * (c == 0 ? d->step : d->chroma_step));

    if (d->ramped) {
        int ctx = nvdr_area_context(w, h);
        d->axis[j] = 0;
        if (nvdr_dec_bit(d->dec, &d->m->grad[ctx]) && !d->dec->overrun) {
            int a = nvdr_dec_bit(d->dec, &d->m->grad_axis[ctx]) ? 2 : 1;
            int8_t sl[3];
            for (int c = 0; c < 3; c++)
                sl[c] = (int8_t)clamp_i8(nvdr_dec_slope(d->dec, d->m, c));
            if (!d->dec->overrun) {
                d->axis[j] = (uint8_t)a;
                memcpy(d->slope + (size_t)j * 3, sl, 3);
            }
        }
        if (d->dec->overrun) {
            d->stopped = 1;
            memcpy(d->chain + (size_t)j * 3, parent, 3);
            d->axis[j] = 0;
            return 0;
        }
    }

    d->delivered++;
    return 0;
}

void nvdr_pyramid_free(NvdrPyramid* pyr) {
    for (int k = 0; k < NVDR_LEVELS; k++) {
        free(pyr->level[k].x); free(pyr->level[k].y);
        free(pyr->level[k].w); free(pyr->level[k].h);
        free(pyr->level[k].rgb);
        free(pyr->level[k].chain);
        free(pyr->level[k].axis);
        free(pyr->level[k].slope);
    }
    free(pyr->palette);
    memset(pyr, 0, sizeof(*pyr));
}

/*
 * Read one level's stream.
 *
 * An arithmetic stream decodes as far as its bytes go, so a short read is
 * handed over as-is and the unit loop stops wherever it runs out. A deflate
 * stream has no such property — a prefix of it inflates to nothing — so
 * that one stays all or nothing.
 */
static uint8_t* read_stream(FILE* f, long* available, uint32_t stored_bytes,
                            uint32_t raw_bytes, uint8_t compression,
                            size_t* out_size) {
    if (stored_bytes == 0 || *available <= 0) return NULL;

    size_t want = stored_bytes;
    if (*available < (long)stored_bytes) {
        if (compression == NVDR_COMPRESS_DEFLATE) return NULL;
        want = (size_t)*available;
    }

    uint8_t* packed = (uint8_t*)malloc(want);
    if (!packed) return NULL;
    if (fread(packed, 1, want, f) != want) { free(packed); return NULL; }
    *available -= (long)want;
    if (out_size) *out_size = want;

    if (compression != NVDR_COMPRESS_DEFLATE) return packed;

    uint8_t* raw = (uint8_t*)malloc(raw_bytes ? raw_bytes : 1);
    if (!raw) { free(packed); return NULL; }
    uLongf produced = raw_bytes;
    int rc = uncompress(raw, &produced, packed, (uLong)want);
    free(packed);
    if (rc != Z_OK || produced != raw_bytes) { free(raw); return NULL; }
    if (out_size) *out_size = raw_bytes;
    return raw;
}

int nvdr_decode_file(const char* path, NvdrPyramid* pyr, NvdrHeader* hdr) {
    memset(pyr, 0, sizeof(*pyr));
    pyr->last_level_fraction = 1.0;

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
    hdr->order       = header[14];
    hdr->space       = header[15];
    hdr->gradient_step = header[67];
    for (int k = 0; k < NVDR_LEVELS; k++)
        hdr->chroma_step[k] = header[64 + k] ? header[64 + k] : hdr->step[k];
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
    size_t anchor_size = 0;
    uint8_t* stream = read_stream(f, &available, hdr->stored_bytes[0],
                                  hdr->raw_bytes[0], hdr->compression, &anchor_size);
    /*
     * The anchor is the contract: a partial one is no picture at all.
     *
     * Only the arith path can hand over a partial stream — a deflate one
     * that did not arrive whole comes back NULL — and on that path
     * `anchor_size` is the inflated size, which says nothing about how
     * many bytes arrived. Comparing the two used to reject every deflate
     * container whose anchor actually compressed.
     */
    int anchor_short = hdr->compression != NVDR_COMPRESS_DEFLATE &&
                       anchor_size < hdr->stored_bytes[0];
    if (!stream || anchor_short) {
        free(stream); fclose(f); return -1;
    }

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
    pyr->level[0].chain = (uint8_t*)malloc((size_t)pyr->level[0].count * 3);
    if (!pyr->level[0].rgb || !pyr->level[0].chain) {
        free(tokens); free(stream); fclose(f); nvdr_pyramid_free(pyr); return -1;
    }
    for (uint32_t i = 0; i < pyr->level[0].count; i++) {
        uint32_t token = tokens[i] < (uint32_t)pyr->palette_count ? tokens[i] : 0;
        memcpy(pyr->level[0].rgb + (size_t)i * 3, pyr->palette + token * 3, 3);
        if (hdr->space == NVDR_SPACE_YCC)
            rgb_to_ycc(pyr->palette + token * 3, pyr->level[0].chain + (size_t)i * 3);
        else
            memcpy(pyr->level[0].chain + (size_t)i * 3, pyr->palette + token * 3, 3);
    }
    free(tokens);
    free(stream);
    pyr->level[0].space = hdr->space;
    pyr->levels_present = 1;

    /* --- every further level is a bonus the bytes may not have paid for -- */
    for (int k = 1; k < NVDR_LEVELS; k++) {
        size_t buf_size = 0;
        uint8_t* buf = read_stream(f, &available, hdr->stored_bytes[k],
                                   hdr->raw_bytes[k], hdr->compression, &buf_size);
        if (!buf) break;

        const NvdrLevelData* prev = &pyr->level[k - 1];
        const int arith = hdr->compression == NVDR_COMPRESS_ARITH;

        /* The same unit order the encoder used, derived from rectangles
         * already decoded rather than read from the file. */
        Unit* order = (Unit*)calloc(prev->count ? prev->count : 1, sizeof(Unit));
        if (!order) { free(buf); break; }
        for (uint32_t i = 0; i < prev->count; i++) {
            order[i].index = i;
            order[i].area = (uint32_t)prev->w[i] * (uint32_t)prev->h[i];
        }
        if (hdr->order == NVDR_ORDER_AREA)
            qsort(order, prev->count, sizeof(Unit), cmp_unit_area);

        size_t cap = (size_t)hdr->leaf_count[k] + prev->count + 1;
        uint8_t* chain = (uint8_t*)malloc(cap * 3);
        uint8_t* rgb = (uint8_t*)malloc(cap * 3);
        uint8_t* lvl_axis = hdr->gradient_step ? (uint8_t*)calloc(cap, 1) : NULL;
        int8_t* lvl_slope = hdr->gradient_step ? (int8_t*)calloc(cap * 3, 1) : NULL;
        if (!chain || !rgb || (hdr->gradient_step && (!lvl_axis || !lvl_slope))) {
            free(chain); free(rgb); free(lvl_axis); free(lvl_slope);
            free(order); free(buf); break;
        }

        RectSink sink = { &pyr->level[k], 0 };
        NvdrModels models;
        NvdrDecoder ad;
        BitReader lbr = { buf, hdr->split_bits[k], 0, 0 };
        if (arith) {
            nvdr_models_init(&models);
            nvdr_dec_init(&ad, buf, buf_size);
        }

        uint32_t processed = 0;
        int stopped = 0;

        UnitCtx d;
        memset(&d, 0, sizeof(d));
        d.dec = &ad;
        d.m = &models;
        d.sink = &sink;
        d.level = &pyr->level[k];
        d.prev = prev;
        d.chain = chain;
        d.axis = lvl_axis;
        d.slope = lvl_slope;
        d.cap = cap;
        d.step = hdr->step[k];
        d.chroma_step = hdr->chroma_step[k];
        d.slope_step = hdr->gradient_step;
        d.ramped = lvl_axis != NULL;

        for (uint32_t u = 0; u < prev->count; u++) {
            uint32_t i = order[u].index;

            if (stopped) {
                /* Past the end of what arrived: this unit keeps the
                 * rectangle and colour it had at the level before, ramp
                 * included, so it renders exactly as that level did. */
                if (sink_push(&sink, prev->x[i], prev->y[i], prev->w[i], prev->h[i]) != 0)
                    break;
                uint32_t j = pyr->level[k].count - 1;
                size_t at = (size_t)j * 3;
                memcpy(chain + at, prev->chain + (size_t)i * 3, 3);
                memcpy(rgb + at, prev->rgb + (size_t)i * 3, 3);
                if (lvl_axis && prev->axis) {
                    lvl_axis[j] = prev->axis[i];
                    memcpy(lvl_slope + at, prev->slope + (size_t)i * 3, 3);
                }
                continue;
            }

            uint32_t before = pyr->level[k].count;
            d.prev_index = i;
            d.stopped = 0;
            d.delivered = 0;

            if (!arith) {
                /* Deflate is all or nothing, so it keeps the old planar
                 * layout: every split bit, then every residual. */
                if (replay(&lbr, &sink, prev->x[i], prev->y[i],
                           prev->w[i], prev->h[i]) != 0 || lbr.overrun ||
                    pyr->level[k].count > cap) {
                    pyr->level[k].count = before;
                    stopped = 1;
                    u--;
                    continue;
                }
                const int8_t* res = (const int8_t*)(buf + (hdr->split_bits[k] + 7) / 8);
                for (uint32_t j = before; j < pyr->level[k].count; j++) {
                    uint8_t parent[3];
                    chain_sample(prev->chain + (size_t)i * 3,
                                 prev->axis ? prev->axis[i] : 0,
                                 prev->slope ? prev->slope + (size_t)i * 3 : nvdr_no_slope,
                                 prev->slope_step,
                                 prev->x[i], prev->y[i], prev->w[i], prev->h[i],
                                 pyr->level[k].x[j] + pyr->level[k].w[j] / 2,
                                 pyr->level[k].y[j] + pyr->level[k].h[j] / 2,
                                 parent);
                    for (int c = 0; c < 3; c++) {
                        int step = c == 0 ? hdr->step[k] : hdr->chroma_step[k];
                        chain[(size_t)j * 3 + c] = (uint8_t)clamp_u8(
                            (int)parent[c] + (int)res[(size_t)j * 3 + c] * step);
                    }
                }
            } else {
                if (replay_unit(&d, prev->x[i], prev->y[i],
                                prev->w[i], prev->h[i], 0) != 0) break;
                if (d.delivered == 0) {
                    /* Not one rectangle of this unit arrived. Roll it
                     * back and let the absent path above cover it. */
                    pyr->level[k].count = before;
                    stopped = 1;
                    u--;
                    continue;
                }
                if (d.stopped) stopped = 1;   /* this unit is the last, in part */
            }

            for (uint32_t j = before; j < pyr->level[k].count; j++) {
                if (hdr->space == NVDR_SPACE_YCC)
                    ycc_to_rgb(chain + (size_t)j * 3, rgb + (size_t)j * 3);
                else
                    memcpy(rgb + (size_t)j * 3, chain + (size_t)j * 3, 3);
            }
            processed++;
        }

        free(order);
        free(buf);

        if (processed == 0) {
            free(rgb); free(chain); free(lvl_axis); free(lvl_slope);
            free(pyr->level[k].x); free(pyr->level[k].y);
            free(pyr->level[k].w); free(pyr->level[k].h);
            memset(&pyr->level[k], 0, sizeof(pyr->level[k]));
            break;
        }

        pyr->level[k].rgb = rgb;
        pyr->level[k].chain = chain;
        pyr->level[k].axis = lvl_axis;
        pyr->level[k].slope = lvl_slope;
        pyr->level[k].slope_step = hdr->gradient_step;
        pyr->level[k].space = hdr->space;
        pyr->levels_present = k + 1;
        pyr->last_level_fraction = prev->count
            ? (double)processed / (double)prev->count : 1.0;
        if (stopped) break;
    }

    fclose(f);
    return 0;
}

/* =============================================================== render */

void nvdr_smooth(NvdrImage* img, float weight) {
    if (weight <= 0.0f || img->width < 2 || img->height < 2) return;

    size_t n = (size_t)img->width * img->height * 3;
    unsigned char* source = (unsigned char*)malloc(n);
    if (!source) return;
    memcpy(source, img->pixels, n);

    const int W = img->width, H = img->height;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            size_t p = ((size_t)y * W + x) * 3;
            for (int c = 0; c < 3; c++) {
                double acc = source[p + c];
                double total = 1.0;
                if (x > 0)     { acc += weight * source[p - 3 + c];         total += weight; }
                if (x < W - 1) { acc += weight * source[p + 3 + c];         total += weight; }
                if (y > 0)     { acc += weight * source[p - (size_t)W*3 + c]; total += weight; }
                if (y < H - 1) { acc += weight * source[p + (size_t)W*3 + c]; total += weight; }
                img->pixels[p + c] = (unsigned char)clamp_u8((int)(acc / total + 0.5));
            }
        }
    }
    free(source);
}

void nvdr_render_level(const NvdrLevelData* level, NvdrImage* out) {
    for (uint32_t i = 0; i < level->count; i++) {
        const uint8_t* rgb = level->rgb + (size_t)i * 3;
        int x1 = level->x[i] + level->w[i];
        int y1 = level->y[i] + level->h[i];
        if (x1 > out->width)  x1 = out->width;
        if (y1 > out->height) y1 = out->height;

        int axis = level->axis ? level->axis[i] : 0;
        if (!axis) {
            for (int y = level->y[i]; y < y1; y++) {
                unsigned char* row =
                    out->pixels + ((size_t)y * out->width + level->x[i]) * 3;
                for (int x = level->x[i]; x < x1; x++, row += 3) {
                    row[0] = rgb[0]; row[1] = rgb[1]; row[2] = rgb[2];
                }
            }
            continue;
        }

        /* A ramp varies along one axis only, so the whole rectangle is one
         * row (or one column) of colours repeated. It is evaluated in the
         * chain's space and converted per position, not per pixel. */
        const uint8_t* chain = level->chain + (size_t)i * 3;
        const int8_t* sl = level->slope + (size_t)i * 3;
        int len = axis == 1 ? level->w[i] : level->h[i];
        for (int y = level->y[i]; y < y1; y++) {
            unsigned char* row =
                out->pixels + ((size_t)y * out->width + level->x[i]) * 3;
            uint8_t fill[3];
            if (axis == 2) {
                uint8_t c3[3];
                for (int c = 0; c < 3; c++)
                    c3[c] = (uint8_t)clamp_u8((int)chain[c] +
                        ramp_offset(sl[c] * level->slope_step,
                                    y - level->y[i], len));
                if (level->space == NVDR_SPACE_YCC) ycc_to_rgb(c3, fill);
                else memcpy(fill, c3, 3);
            }
            for (int x = level->x[i]; x < x1; x++, row += 3) {
                if (axis == 1) {
                    uint8_t c3[3];
                    for (int c = 0; c < 3; c++)
                        c3[c] = (uint8_t)clamp_u8((int)chain[c] +
                            ramp_offset(sl[c] * level->slope_step,
                                        x - level->x[i], len));
                    if (level->space == NVDR_SPACE_YCC) ycc_to_rgb(c3, fill);
                    else memcpy(fill, c3, 3);
                }
                row[0] = fill[0]; row[1] = fill[1]; row[2] = fill[2];
            }
        }
    }
}
