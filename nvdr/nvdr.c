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

void nvdr_image_free(NvdrImage* img) {
    free(img->pixels);
    img->pixels = NULL;
    img->width = img->height = 0;
}

static unsigned char* pixel_at(const NvdrImage* img, int x, int y) {
    return img->pixels + ((size_t)y * img->width + x) * 3;
}

/* ============================================================== bit i/o */

typedef struct {
    uint8_t* bytes;
    size_t   capacity;
    size_t   bit_pos;
} BitWriter;

static int bw_init(BitWriter* bw, size_t expected_bits) {
    bw->capacity = (expected_bits + 7) / 8 + 64;
    bw->bytes = (uint8_t*)calloc(bw->capacity, 1);
    bw->bit_pos = 0;
    return bw->bytes ? 0 : -1;
}

static int bw_put(BitWriter* bw, int bit) {
    size_t byte = bw->bit_pos >> 3;
    if (byte >= bw->capacity) {
        size_t grown = bw->capacity * 2;
        uint8_t* next = (uint8_t*)realloc(bw->bytes, grown);
        if (!next) return -1;
        memset(next + bw->capacity, 0, grown - bw->capacity);
        bw->bytes = next;
        bw->capacity = grown;
    }
    if (bit) bw->bytes[byte] |= (uint8_t)(1u << (bw->bit_pos & 7));
    bw->bit_pos++;
    return 0;
}

typedef struct {
    const uint8_t* bytes;
    size_t         bit_count;
    size_t         bit_pos;
} BitReader;

static int br_get(BitReader* br) {
    if (br->bit_pos >= br->bit_count) return -1;
    size_t byte = br->bit_pos >> 3;
    int bit = (br->bytes[byte] >> (br->bit_pos & 7)) & 1;
    br->bit_pos++;
    return bit;
}

/* =========================================================== homogeneity */

/*
 * Mean absolute deviation from the region's own mean, weighted towards
 * luma the way the eye is. Returned normalised to 0..1 so the threshold
 * reads the same regardless of bit depth.
 */
static float region_stats(const NvdrImage* img, int x, int y, int w, int h,
                          unsigned char* out_r, unsigned char* out_g,
                          unsigned char* out_b) {
    int x1 = x + w > img->width  ? img->width  : x + w;
    int y1 = y + h > img->height ? img->height : y + h;
    if (x1 <= x || y1 <= y) {
        *out_r = *out_g = *out_b = 0;
        return 0.0f;
    }

    uint64_t sum_r = 0, sum_g = 0, sum_b = 0;
    uint32_t count = 0;
    for (int py = y; py < y1; py++) {
        const unsigned char* row = pixel_at(img, x, py);
        for (int px = x; px < x1; px++, row += 3) {
            sum_r += row[0];
            sum_g += row[1];
            sum_b += row[2];
            count++;
        }
    }
    unsigned char mr = (unsigned char)((sum_r + count / 2) / count);
    unsigned char mg = (unsigned char)((sum_g + count / 2) / count);
    unsigned char mb = (unsigned char)((sum_b + count / 2) / count);
    *out_r = mr; *out_g = mg; *out_b = mb;

    /* Deviation accumulates in double: a 4K region is 8M terms, and a
     * float accumulator would start dropping small ones. */
    double deviation = 0.0;
    for (int py = y; py < y1; py++) {
        const unsigned char* row = pixel_at(img, x, py);
        for (int px = x; px < x1; px++, row += 3) {
            int dr = (int)row[0] - (int)mr;
            int dg = (int)row[1] - (int)mg;
            int db = (int)row[2] - (int)mb;
            if (dr < 0) dr = -dr;
            if (dg < 0) dg = -dg;
            if (db < 0) db = -db;
            deviation += 0.30 * dr + 0.59 * dg + 0.11 * db;
        }
    }
    return (float)(deviation / (double)count / 255.0);
}

/* ============================================================= geometry */

NvdrBuildConfig nvdr_default_build_config(void) {
    NvdrBuildConfig cfg;
    cfg.min_tile     = 2;
    cfg.max_depth    = 12;
    cfg.homogeneity  = 0.020f;
    return cfg;
}

typedef struct {
    const NvdrImage*       img;
    const NvdrBuildConfig* cfg;
    BitWriter              bits;
    NvdrLeaf*              leaves;
    uint32_t               leaf_count;
    uint32_t               leaf_capacity;
    unsigned char*         exact;         /* 3 bytes per leaf */
    uint32_t               node_count;
    int                    failed;
} BuildCtx;

static int push_leaf(BuildCtx* ctx, int x, int y, int w, int h,
                     unsigned char r, unsigned char g, unsigned char b) {
    if (ctx->leaf_count == ctx->leaf_capacity) {
        uint32_t grown = ctx->leaf_capacity ? ctx->leaf_capacity * 2 : 4096;
        NvdrLeaf* leaves = (NvdrLeaf*)realloc(ctx->leaves, grown * sizeof(NvdrLeaf));
        unsigned char* exact = (unsigned char*)realloc(ctx->exact, (size_t)grown * 3);
        if (!leaves || !exact) {
            ctx->leaves = leaves ? leaves : ctx->leaves;
            ctx->exact  = exact  ? exact  : ctx->exact;
            return -1;
        }
        ctx->leaves = leaves;
        ctx->exact = exact;
        ctx->leaf_capacity = grown;
    }
    NvdrLeaf* leaf = &ctx->leaves[ctx->leaf_count];
    leaf->x = (uint16_t)x; leaf->y = (uint16_t)y;
    leaf->w = (uint16_t)w; leaf->h = (uint16_t)h;
    unsigned char* rgb = ctx->exact + (size_t)ctx->leaf_count * 3;
    rgb[0] = r; rgb[1] = g; rgb[2] = b;
    ctx->leaf_count++;
    return 0;
}

static void build_rec(BuildCtx* ctx, int x, int y, int w, int h, int depth) {
    if (ctx->failed) return;
    ctx->node_count++;

    unsigned char r, g, b;
    float deviation = region_stats(ctx->img, x, y, w, h, &r, &g, &b);

    int splittable = w > ctx->cfg->min_tile && h > ctx->cfg->min_tile &&
                     depth < ctx->cfg->max_depth;
    int split = splittable && deviation > ctx->cfg->homogeneity;

    if (bw_put(&ctx->bits, split) != 0) { ctx->failed = 1; return; }

    if (!split) {
        if (push_leaf(ctx, x, y, w, h, r, g, b) != 0) ctx->failed = 1;
        return;
    }

    int hw = w / 2, hh = h / 2;
    int rw = w - hw, rh = h - hh;
    build_rec(ctx, x,      y,      hw, hh, depth + 1);
    build_rec(ctx, x + hw, y,      rw, hh, depth + 1);
    build_rec(ctx, x,      y + hh, hw, rh, depth + 1);
    build_rec(ctx, x + hw, y + hh, rw, rh, depth + 1);
}

/* The exact colours live inside the build context; the caller gets them
 * through this out-parameter because they are the encoder's input, not
 * part of the geometry that ships in the file. */
int nvdr_geometry_build_ex(NvdrGeometry* geo, unsigned char** exact_out,
                           const NvdrImage* img, const NvdrBuildConfig* cfg) {
    BuildCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.img = img;
    ctx.cfg = cfg;
    if (bw_init(&ctx.bits, (size_t)img->width * img->height / 4 + 1024) != 0)
        return -1;

    build_rec(&ctx, 0, 0, img->width, img->height, 0);

    if (ctx.failed) {
        free(ctx.bits.bytes);
        free(ctx.leaves);
        free(ctx.exact);
        return -1;
    }

    geo->split_bits = ctx.bits.bytes;
    geo->node_count = ctx.node_count;
    geo->leaves     = ctx.leaves;
    geo->leaf_count = ctx.leaf_count;
    *exact_out      = ctx.exact;
    return 0;
}

int nvdr_geometry_build(NvdrGeometry* geo, const NvdrImage* img,
                        const NvdrBuildConfig* cfg) {
    unsigned char* exact = NULL;
    int rc = nvdr_geometry_build_ex(geo, &exact, img, cfg);
    free(exact);
    return rc;
}

typedef struct {
    BitReader  br;
    NvdrLeaf*  leaves;
    uint32_t   leaf_count;
    uint32_t   leaf_capacity;
    int        failed;
} DecodeCtx;

static void decode_rec(DecodeCtx* ctx, int x, int y, int w, int h) {
    if (ctx->failed) return;
    int split = br_get(&ctx->br);
    if (split < 0) { ctx->failed = 1; return; }

    if (!split) {
        if (ctx->leaf_count == ctx->leaf_capacity) {
            uint32_t grown = ctx->leaf_capacity ? ctx->leaf_capacity * 2 : 4096;
            NvdrLeaf* leaves = (NvdrLeaf*)realloc(ctx->leaves, grown * sizeof(NvdrLeaf));
            if (!leaves) { ctx->failed = 1; return; }
            ctx->leaves = leaves;
            ctx->leaf_capacity = grown;
        }
        NvdrLeaf* leaf = &ctx->leaves[ctx->leaf_count++];
        leaf->x = (uint16_t)x; leaf->y = (uint16_t)y;
        leaf->w = (uint16_t)w; leaf->h = (uint16_t)h;
        return;
    }

    int hw = w / 2, hh = h / 2;
    int rw = w - hw, rh = h - hh;
    decode_rec(ctx, x,      y,      hw, hh);
    decode_rec(ctx, x + hw, y,      rw, hh);
    decode_rec(ctx, x,      y + hh, hw, rh);
    decode_rec(ctx, x + hw, y + hh, rw, rh);
}

int nvdr_geometry_decode(NvdrGeometry* geo, const uint8_t* split_bits,
                         uint32_t node_count, int width, int height) {
    DecodeCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.br.bytes     = split_bits;
    ctx.br.bit_count = node_count;

    decode_rec(&ctx, 0, 0, width, height);
    if (ctx.failed) { free(ctx.leaves); return -1; }

    geo->split_bits = NULL;          /* borrowed from the caller's buffer */
    geo->node_count = node_count;
    geo->leaves     = ctx.leaves;
    geo->leaf_count = ctx.leaf_count;
    return 0;
}

void nvdr_geometry_free(NvdrGeometry* geo) {
    free(geo->split_bits);
    free(geo->leaves);
    memset(geo, 0, sizeof(*geo));
}

/* ======================================================== palette build */

/*
 * Candidate colours come from a 5-bit-per-channel histogram of the leaf
 * colours, weighted by the area each leaf covers. That bounds the
 * candidate set at 32768 regardless of image size and costs one pass.
 */
typedef struct {
    unsigned char r, g, b;
    double        weight;
} Candidate;

static int build_palette(const unsigned char* exact, const NvdrLeaf* leaves,
                         uint32_t leaf_count, int palette_n,
                         unsigned char* palette_out) {
    enum { BINS = 32768 };
    double*  weight = (double*)calloc(BINS, sizeof(double));
    double*  sum_r  = (double*)calloc(BINS, sizeof(double));
    double*  sum_g  = (double*)calloc(BINS, sizeof(double));
    double*  sum_b  = (double*)calloc(BINS, sizeof(double));
    if (!weight || !sum_r || !sum_g || !sum_b) {
        free(weight); free(sum_r); free(sum_g); free(sum_b);
        return -1;
    }

    for (uint32_t i = 0; i < leaf_count; i++) {
        const unsigned char* c = exact + (size_t)i * 3;
        double area = (double)leaves[i].w * (double)leaves[i].h;
        int bin = ((c[0] >> 3) << 10) | ((c[1] >> 3) << 5) | (c[2] >> 3);
        weight[bin] += area;
        sum_r[bin]  += area * c[0];
        sum_g[bin]  += area * c[1];
        sum_b[bin]  += area * c[2];
    }

    Candidate* cands = (Candidate*)malloc(BINS * sizeof(Candidate));
    if (!cands) {
        free(weight); free(sum_r); free(sum_g); free(sum_b);
        return -1;
    }
    int cand_count = 0;
    for (int i = 0; i < BINS; i++) {
        if (weight[i] <= 0.0) continue;
        Candidate* c = &cands[cand_count++];
        c->r = (unsigned char)(sum_r[i] / weight[i] + 0.5);
        c->g = (unsigned char)(sum_g[i] / weight[i] + 0.5);
        c->b = (unsigned char)(sum_b[i] / weight[i] + 0.5);
        c->weight = weight[i];
    }
    free(weight); free(sum_r); free(sum_g); free(sum_b);

    if (cand_count == 0) { free(cands); return -1; }
    if (palette_n > cand_count) palette_n = cand_count;

    /* Greedy max-salience selection: heaviest colour first, then whichever
     * candidate maximises weight x distance-to-nearest-already-chosen. The
     * per-candidate minimum only ever shrinks, so it is carried forward
     * instead of recomputed — O(palette_n * cand_count) overall. */
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
        chosen++;
        if (chosen >= palette_n) break;

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

    free(cands); free(min_dist); free(taken);
    return chosen;
}

static int nearest_entry(const unsigned char* palette, int n,
                         int r, int g, int b) {
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

/* ======================================================= residual stack */

NvdrEncodeConfig nvdr_default_encode_config(void) {
    NvdrEncodeConfig cfg;
    cfg.anchor_bits = 4;   /* the spec's int4 anchor */
    cfg.r1_step     = 16;
    cfg.r2_step     = 1;   /* closes the gap exactly */
    return cfg;
}

static int clamp_i8(int v) {
    if (v >  127) return  127;
    if (v < -127) return -127;
    return v;
}

static int div_round(int numerator, int denominator) {
    if (numerator >= 0) return (numerator + denominator / 2) / denominator;
    return -((-numerator + denominator / 2) / denominator);
}

int nvdr_stack_encode(NvdrStack* stack, const unsigned char* exact,
                      uint32_t leaf_count, const NvdrEncodeConfig* cfg) {
    memset(stack, 0, sizeof(*stack));
    stack->leaf_count  = leaf_count;
    stack->anchor_bits = cfg->anchor_bits;
    stack->r1_step     = cfg->r1_step;
    stack->r2_step     = cfg->r2_step;

    int palette_n = 1 << cfg->anchor_bits;
    stack->anchor_palette = (unsigned char*)calloc((size_t)palette_n * 3, 1);
    stack->anchor_tokens  = (uint8_t*)calloc(leaf_count, 1);
    for (int c = 0; c < 3; c++) {
        stack->r1[c] = (int8_t*)calloc(leaf_count, 1);
        stack->r2[c] = (int8_t*)calloc(leaf_count, 1);
    }
    if (!stack->anchor_palette || !stack->anchor_tokens ||
        !stack->r1[0] || !stack->r1[1] || !stack->r1[2] ||
        !stack->r2[0] || !stack->r2[1] || !stack->r2[2]) {
        nvdr_stack_free(stack);
        return -1;
    }
    return palette_n;   /* caller fills the palette, then calls _finish */
}

/*
 * The encode is split so the palette can be built from leaf areas, which
 * the stack itself does not carry. Callers use nvdr_encode_image below;
 * this pair stays internal.
 */
static void stack_fill_layers(NvdrStack* stack, const unsigned char* exact) {
    for (uint32_t i = 0; i < stack->leaf_count; i++) {
        const unsigned char* target = exact + (size_t)i * 3;
        int token = nearest_entry(stack->anchor_palette,
                                  stack->anchor_palette_n,
                                  target[0], target[1], target[2]);
        stack->anchor_tokens[i] = (uint8_t)token;
        const unsigned char* anchor = stack->anchor_palette + token * 3;

        for (int c = 0; c < 3; c++) {
            /* R1: coarse correction of the anchor towards the truth */
            int residual = (int)target[c] - (int)anchor[c];
            int q1 = clamp_i8(div_round(residual, stack->r1_step));
            stack->r1[c][i] = (int8_t)q1;

            /* R2: whatever R1 could not express. With r2_step = 1 this is
             * exact, which is the spec's "no approximation in the
             * reconstruction chain when all layers are present". */
            int remainder = residual - q1 * stack->r1_step;
            int q2 = clamp_i8(div_round(remainder, stack->r2_step));
            stack->r2[c][i] = (int8_t)q2;
        }
    }
}

void nvdr_stack_resolve(const NvdrStack* stack, NvdrLevel level,
                        unsigned char* out_rgb) {
    for (uint32_t i = 0; i < stack->leaf_count; i++) {
        const unsigned char* anchor =
            stack->anchor_palette + (size_t)stack->anchor_tokens[i] * 3;
        for (int c = 0; c < 3; c++) {
            int value = anchor[c];
            if (level >= NVDR_LEVEL_R1 && stack->r1[c])
                value += (int)stack->r1[c][i] * stack->r1_step;
            if (level >= NVDR_LEVEL_R2 && stack->r2[c])
                value += (int)stack->r2[c][i] * stack->r2_step;
            if (value < 0) value = 0;
            if (value > 255) value = 255;
            out_rgb[(size_t)i * 3 + c] = (unsigned char)value;
        }
    }
}

void nvdr_stack_free(NvdrStack* stack) {
    free(stack->anchor_palette);
    free(stack->anchor_tokens);
    for (int c = 0; c < 3; c++) { free(stack->r1[c]); free(stack->r2[c]); }
    memset(stack, 0, sizeof(*stack));
}

/* ------------------------------------------------ one-call encode path */

int nvdr_encode_image(const NvdrImage* img, const NvdrBuildConfig* build_cfg,
                      const NvdrEncodeConfig* enc_cfg,
                      NvdrGeometry* geo, NvdrStack* stack) {
    unsigned char* exact = NULL;
    if (nvdr_geometry_build_ex(geo, &exact, img, build_cfg) != 0) return -1;

    if (nvdr_stack_encode(stack, exact, geo->leaf_count, enc_cfg) < 0) {
        free(exact);
        nvdr_geometry_free(geo);
        return -1;
    }

    int palette_n = build_palette(exact, geo->leaves, geo->leaf_count,
                                  1 << enc_cfg->anchor_bits,
                                  stack->anchor_palette);
    if (palette_n <= 0) {
        free(exact);
        nvdr_stack_free(stack);
        nvdr_geometry_free(geo);
        return -1;
    }
    stack->anchor_palette_n = palette_n;

    stack_fill_layers(stack, exact);
    free(exact);
    return 0;
}

/* ============================================================ container */

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

static size_t anchor_stream_size(const NvdrStack* stack, uint32_t node_count) {
    size_t palette = 1 + (size_t)stack->anchor_palette_n * 3;
    size_t bits    = ((size_t)node_count + 7) / 8;
    size_t tokens  = ((size_t)stack->leaf_count * stack->anchor_bits + 7) / 8;
    return palette + bits + tokens;
}

int nvdr_container_write(const char* path, const NvdrGeometry* geo,
                         const NvdrStack* stack, int width, int height) {
    size_t anchor_bytes = anchor_stream_size(stack, geo->node_count);
    size_t residual_bytes = (size_t)stack->leaf_count * 3;

    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    uint8_t header[NVDR_HEADER_SIZE];
    memset(header, 0, sizeof(header));
    memcpy(header, NVDR_MAGIC, 4);
    header[4] = NVDR_VERSION;
    header[5] = 0;
    put_u16(header +  6, (uint16_t)width);
    put_u16(header +  8, (uint16_t)height);
    put_u32(header + 10, geo->node_count);
    put_u32(header + 14, stack->leaf_count);
    put_u32(header + 18, (uint32_t)anchor_bytes);
    put_u32(header + 22, (uint32_t)residual_bytes);
    put_u32(header + 26, (uint32_t)residual_bytes);
    header[30] = (uint8_t)stack->anchor_bits;
    header[31] = (uint8_t)stack->r1_step;
    /* r2_step rides in the flags byte; it is 1 in every sane configuration
     * and the format reserves room to widen this later. */
    header[5] = (uint8_t)stack->r2_step;
    fwrite(header, 1, sizeof(header), f);

    /* ANCHOR: palette, then geometry bits, then packed tokens. */
    uint8_t palette_n = (uint8_t)stack->anchor_palette_n;
    fwrite(&palette_n, 1, 1, f);
    fwrite(stack->anchor_palette, 3, stack->anchor_palette_n, f);
    fwrite(geo->split_bits, 1, ((size_t)geo->node_count + 7) / 8, f);

    size_t token_bytes = ((size_t)stack->leaf_count * stack->anchor_bits + 7) / 8;
    uint8_t* packed = (uint8_t*)calloc(token_bytes, 1);
    if (!packed) { fclose(f); return -1; }
    for (uint32_t i = 0; i < stack->leaf_count; i++) {
        size_t bit = (size_t)i * stack->anchor_bits;
        uint32_t token = stack->anchor_tokens[i];
        for (int b = 0; b < stack->anchor_bits; b++) {
            if (token & (1u << b)) packed[(bit + b) >> 3] |= (uint8_t)(1u << ((bit + b) & 7));
        }
    }
    fwrite(packed, 1, token_bytes, f);
    free(packed);

    /* R1 and R2, planar per channel — neighbouring leaves carry similar
     * residuals, so keeping each channel contiguous gives the entropy
     * coder long stretches of near-identical bytes to work with. */
    for (int c = 0; c < 3; c++) fwrite(stack->r1[c], 1, stack->leaf_count, f);
    for (int c = 0; c < 3; c++) fwrite(stack->r2[c], 1, stack->leaf_count, f);

    fclose(f);
    return 0;
}

int nvdr_container_read(const char* path, NvdrGeometry* geo, NvdrStack* stack,
                        NvdrHeader* hdr, NvdrLevel* available_level) {
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

    hdr->width        = get_u16(header + 6);
    hdr->height       = get_u16(header + 8);
    hdr->node_count   = get_u32(header + 10);
    hdr->leaf_count   = get_u32(header + 14);
    hdr->anchor_bytes = get_u32(header + 18);
    hdr->r1_bytes     = get_u32(header + 22);
    hdr->r2_bytes     = get_u32(header + 26);
    hdr->anchor_bits  = header[30];
    hdr->r1_step      = header[31];
    hdr->r2_step      = header[5];

    /* The anchor is the contract: without it there is nothing to show. */
    long have = file_size - NVDR_HEADER_SIZE;
    if (have < (long)hdr->anchor_bytes) { fclose(f); return -1; }

    memset(stack, 0, sizeof(*stack));
    stack->leaf_count  = hdr->leaf_count;
    stack->anchor_bits = hdr->anchor_bits;
    stack->r1_step     = hdr->r1_step;
    stack->r2_step     = hdr->r2_step;

    uint8_t palette_n = 0;
    if (fread(&palette_n, 1, 1, f) != 1) { fclose(f); return -1; }
    stack->anchor_palette_n = palette_n;
    stack->anchor_palette = (unsigned char*)malloc((size_t)palette_n * 3);
    if (!stack->anchor_palette ||
        fread(stack->anchor_palette, 3, palette_n, f) != palette_n) {
        fclose(f); nvdr_stack_free(stack); return -1;
    }

    size_t bit_bytes = ((size_t)hdr->node_count + 7) / 8;
    uint8_t* split_bits = (uint8_t*)malloc(bit_bytes);
    if (!split_bits || fread(split_bits, 1, bit_bytes, f) != bit_bytes) {
        free(split_bits); fclose(f); nvdr_stack_free(stack); return -1;
    }
    if (nvdr_geometry_decode(geo, split_bits, hdr->node_count,
                             hdr->width, hdr->height) != 0) {
        free(split_bits); fclose(f); nvdr_stack_free(stack); return -1;
    }
    free(split_bits);

    size_t token_bytes = ((size_t)hdr->leaf_count * hdr->anchor_bits + 7) / 8;
    uint8_t* packed = (uint8_t*)malloc(token_bytes);
    stack->anchor_tokens = (uint8_t*)calloc(hdr->leaf_count, 1);
    if (!packed || !stack->anchor_tokens ||
        fread(packed, 1, token_bytes, f) != token_bytes) {
        free(packed); fclose(f); nvdr_stack_free(stack); nvdr_geometry_free(geo);
        return -1;
    }
    for (uint32_t i = 0; i < hdr->leaf_count; i++) {
        size_t bit = (size_t)i * hdr->anchor_bits;
        uint32_t token = 0;
        for (int b = 0; b < hdr->anchor_bits; b++) {
            if (packed[(bit + b) >> 3] & (1u << ((bit + b) & 7))) token |= (1u << b);
        }
        stack->anchor_tokens[i] = (uint8_t)token;
    }
    free(packed);

    *available_level = NVDR_LEVEL_ANCHOR;

    /* Everything past here is a bonus. A file that stops mid-residual
     * simply decodes at the level whose bytes are all present — no error,
     * no partial layer, no stall. */
    long consumed = NVDR_HEADER_SIZE + (long)hdr->anchor_bytes;
    long remaining = file_size - consumed;

    if (remaining >= (long)hdr->r1_bytes && hdr->r1_bytes > 0) {
        int ok = 1;
        for (int c = 0; c < 3 && ok; c++) {
            stack->r1[c] = (int8_t*)malloc(hdr->leaf_count);
            if (!stack->r1[c] ||
                fread(stack->r1[c], 1, hdr->leaf_count, f) != hdr->leaf_count) ok = 0;
        }
        if (ok) {
            *available_level = NVDR_LEVEL_R1;
            remaining -= (long)hdr->r1_bytes;
        } else {
            for (int c = 0; c < 3; c++) { free(stack->r1[c]); stack->r1[c] = NULL; }
            remaining = -1;
        }
    }

    if (remaining >= (long)hdr->r2_bytes && hdr->r2_bytes > 0 &&
        *available_level == NVDR_LEVEL_R1) {
        int ok = 1;
        for (int c = 0; c < 3 && ok; c++) {
            stack->r2[c] = (int8_t*)malloc(hdr->leaf_count);
            if (!stack->r2[c] ||
                fread(stack->r2[c], 1, hdr->leaf_count, f) != hdr->leaf_count) ok = 0;
        }
        if (ok) *available_level = NVDR_LEVEL_R2;
        else for (int c = 0; c < 3; c++) { free(stack->r2[c]); stack->r2[c] = NULL; }
    }

    fclose(f);
    return 0;
}

/* ============================================================== render */

void nvdr_render(const NvdrGeometry* geo, const unsigned char* leaf_rgb,
                 NvdrImage* out) {
    for (uint32_t i = 0; i < geo->leaf_count; i++) {
        const NvdrLeaf* leaf = &geo->leaves[i];
        const unsigned char* rgb = leaf_rgb + (size_t)i * 3;
        int x1 = leaf->x + leaf->w > out->width  ? out->width  : leaf->x + leaf->w;
        int y1 = leaf->y + leaf->h > out->height ? out->height : leaf->y + leaf->h;
        for (int y = leaf->y; y < y1; y++) {
            unsigned char* row = out->pixels + ((size_t)y * out->width + leaf->x) * 3;
            for (int x = leaf->x; x < x1; x++, row += 3) {
                row[0] = rgb[0]; row[1] = rgb[1]; row[2] = rgb[2];
            }
        }
    }
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
    if (mse <= 0.0) return 99.0;
    return 10.0 * log10(255.0 * 255.0 / mse);
}
