/*
 * NVDR — Progressive Residual Stack, still-image implementation
 *
 * This is the PRS of NVDR spec v0.14.1 §1.2 built for real, on the one
 * domain that can be built and measured without a GPU: static images.
 *
 * The spec's decomposition is:
 *
 *     ANCHOR = quantize_coarse(L)
 *     R1     = quantize_mid (L - dequant(ANCHOR))
 *     R2     = quantize_fine(L - dequant(ANCHOR) - dequant(R1))
 *
 * with the guarantee that ANCHOR alone always decodes, that each further
 * layer is a pure delta on top of what came before, and that the three
 * together reconstruct L exactly (§12: "the mathematical structure is
 * exact"). Nothing here is a metaphor for that — the three layers below
 * are that decomposition.
 *
 * In the spec L is a diffusion latent and the three layers are int4/int8/
 * fp16 quantizations of it. Here L is the exact per-leaf colour of a
 * quadtree over the source image, and the three layers are a 2^k-entry
 * palette (the int4 analogue, and k is a knob), a coarse signed residual,
 * and a fine signed residual that closes the gap exactly.
 *
 * What carries over unchanged is the property the spec is actually about:
 * every prefix of the container is decodable. Truncate the file anywhere
 * past the anchor and it still renders, at the quality the surviving bytes
 * pay for. There is no such thing as a partially-decodable failure.
 */
#ifndef NVDR_H
#define NVDR_H

#include <stdint.h>
#include <stddef.h>

/* ---------------------------------------------------------------- image */

typedef struct {
    unsigned char* pixels;   /* RGB, row-major, 3 bytes per pixel */
    int width, height;
} NvdrImage;

int  nvdr_image_load(NvdrImage* img, const char* path);
int  nvdr_image_write_ppm(const NvdrImage* img, const char* path);
void nvdr_image_free(NvdrImage* img);

/* ------------------------------------------------------------- geometry */

/*
 * The quadtree is stored in depth-first pre-order, which is what lets the
 * geometry travel as one bit per node: 1 = this node splits, 0 = this node
 * is a leaf. The decoder replays the same subdivision rule from the canvas
 * size, so coordinates are never transmitted. For a typical photo this is
 * the difference between 8 bytes and 1 bit per leaf.
 */
typedef struct {
    uint16_t x, y, w, h;
} NvdrLeaf;

typedef struct {
    uint8_t*  split_bits;    /* 1 bit per node, DFS pre-order */
    uint32_t  node_count;
    NvdrLeaf* leaves;
    uint32_t  leaf_count;
} NvdrGeometry;

typedef struct {
    int   min_tile;          /* stop splitting at this size */
    int   max_depth;
    float homogeneity;       /* split while a region is less uniform than this */
} NvdrBuildConfig;

NvdrBuildConfig nvdr_default_build_config(void);

/* Build the quadtree over `img` and collect its leaves in DFS pre-order. */
int  nvdr_geometry_build(NvdrGeometry* geo, const NvdrImage* img,
                         const NvdrBuildConfig* cfg);

/* Replay a split bitstream back into leaves. The inverse of the above. */
int  nvdr_geometry_decode(NvdrGeometry* geo, const uint8_t* split_bits,
                          uint32_t node_count, int width, int height);

void nvdr_geometry_free(NvdrGeometry* geo);

/* ------------------------------------------------------- residual stack */

/*
 * PRS_LEVEL_ANCHOR is the contract: it is always present and always
 * decodes. The other two are bonuses delivered by whatever bytes arrived.
 */
typedef enum {
    NVDR_LEVEL_ANCHOR = 0,
    NVDR_LEVEL_R1     = 1,
    NVDR_LEVEL_R2     = 2
} NvdrLevel;

typedef struct {
    /* ANCHOR — palette index per leaf, packed at anchor_bits per entry */
    int            anchor_bits;       /* 4 in the spec's int4 sense; tunable */
    unsigned char* anchor_palette;    /* 3 bytes per entry */
    int            anchor_palette_n;
    uint8_t*       anchor_tokens;     /* one per leaf, unpacked in memory */

    /* R1 — coarse signed residual, one int8 per channel per leaf */
    int8_t* r1[3];
    int     r1_step;

    /* R2 — fine signed residual; with step 1 it closes the gap exactly */
    int8_t* r2[3];
    int     r2_step;

    uint32_t leaf_count;
} NvdrStack;

typedef struct {
    int anchor_bits;   /* palette size is 1 << anchor_bits */
    int r1_step;       /* quantisation step of the coarse residual */
    int r2_step;       /* 1 means anchor + R1 + R2 is bit-exact */
} NvdrEncodeConfig;

NvdrEncodeConfig nvdr_default_encode_config(void);

/*
 * Decompose the exact per-leaf colours into the three layers. `exact` is
 * 3 * leaf_count bytes: the mean colour of each leaf's region.
 */
int  nvdr_stack_encode(NvdrStack* stack, const unsigned char* exact,
                       uint32_t leaf_count, const NvdrEncodeConfig* cfg);

/*
 * Accumulate the layers up to `level` into an RGB colour per leaf. This is
 * §3.1's synthesis loop: start from the anchor, add each residual that is
 * actually present, never wait for one that is not.
 */
void nvdr_stack_resolve(const NvdrStack* stack, NvdrLevel level,
                        unsigned char* out_rgb);

void nvdr_stack_free(NvdrStack* stack);

/*
 * Build geometry and all three layers in one call. This is the encoder
 * proper; the pieces above are exposed for tools that need one of them.
 */
int nvdr_encode_image(const NvdrImage* img, const NvdrBuildConfig* build_cfg,
                      const NvdrEncodeConfig* enc_cfg,
                      NvdrGeometry* geo, NvdrStack* stack);

/* Geometry build that also hands back the exact per-leaf colours. */
int nvdr_geometry_build_ex(NvdrGeometry* geo, unsigned char** exact_out,
                           const NvdrImage* img, const NvdrBuildConfig* cfg);

/* ------------------------------------------------------------ container */

#define NVDR_MAGIC   "NVDR"
#define NVDR_VERSION 1
#define NVDR_HEADER_SIZE 32

typedef struct {
    uint16_t width, height;
    uint32_t node_count;
    uint32_t leaf_count;
    uint32_t anchor_bytes;
    uint32_t r1_bytes;
    uint32_t r2_bytes;
    uint8_t  anchor_bits;
    uint8_t  r1_step;
    uint8_t  r2_step;
} NvdrHeader;

int nvdr_container_write(const char* path, const NvdrGeometry* geo,
                         const NvdrStack* stack, int width, int height);

/*
 * Read whatever is there. `available_level` reports how far the bytes on
 * disk actually reach — a file truncated mid-R1 decodes at the anchor, and
 * reports so, rather than failing.
 */
int nvdr_container_read(const char* path, NvdrGeometry* geo, NvdrStack* stack,
                        NvdrHeader* hdr, NvdrLevel* available_level);

/* --------------------------------------------------------------- render */

/* Paint one flat colour per leaf onto an RGB canvas. */
void nvdr_render(const NvdrGeometry* geo, const unsigned char* leaf_rgb,
                 NvdrImage* out);

double nvdr_psnr(const NvdrImage* a, const NvdrImage* b);

#endif /* NVDR_H */
