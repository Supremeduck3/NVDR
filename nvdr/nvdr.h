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
 * together reconstruct L to within a rounding step.
 *
 * WHAT L IS, AND WHY IT MOVED
 * ---------------------------
 * The first cut of this file made L the colour of each leaf of a fixed
 * quadtree: the anchor picked colours from a 16-entry palette and the two
 * residuals corrected them. That is a faithful reading of the spec, and it
 * measured badly — the residuals drove PSNR from 23.3 dB to 26.3 dB and
 * then stopped dead, because 26.3 dB was not a colour limit at all. It was
 * the geometry: one flat colour per leaf, and the leaf set never changed.
 * The stack was refining the dimension that was already nearly solved.
 *
 * That is an artefact of the domain, not of the spec. In the spec L is a
 * diffusion latent — a fixed 64x64x4 grid — so the support is constant and
 * quantisation precision really is the only axis left. An image has no such
 * fixed support. Its dominant error is where the tiles are.
 *
 * So here a level is a *tolerance*, and refining means both subdividing and
 * recolouring. Level 0 prunes the tree wherever a region is uniform enough
 * for a coarse tolerance; each further level lowers the tolerance, so some
 * leaves split into their subtrees. Every leaf at level k — whether it is
 * newly split or the same rectangle as before — carries one signed delta
 * against the colour level k-1 displayed at that spot. Geometric refinement
 * and colour refinement become the same operation, and the residual is a
 * parent-to-child delta, which is where the low entropy the spec counts on
 * actually lives.
 *
 * The guarantee is unchanged and is the point: every prefix of the
 * container past the anchor decodes. Truncate the file anywhere and it
 * still renders, at the quality the surviving bytes pay for.
 *
 * Each level is entropy-coded on its own rather than the container being
 * compressed as a whole. That is not a packaging detail — a prefix of one
 * deflate stream does not decode, so compressing everything together would
 * buy smaller files by destroying the property the format exists for. Per
 * level, both hold: the bytes on disk are the bytes on the wire, and any
 * prefix still ends on a level boundary that decodes.
 */
#ifndef NVDR_H
#define NVDR_H

#include <stdint.h>
#include <stddef.h>

#define NVDR_LEVELS 3

/* ---------------------------------------------------------------- image */

typedef struct {
    unsigned char* pixels;   /* RGB, row-major, 3 bytes per pixel */
    int width, height;
} NvdrImage;

int  nvdr_image_load(NvdrImage* img, const char* path);
int  nvdr_image_write_ppm(const NvdrImage* img, const char* path);
int  nvdr_image_write_png(const NvdrImage* img, const char* path);

/* Picks the writer from the extension: .png gets a PNG, anything else PPM. */
int  nvdr_image_write(const NvdrImage* img, const char* path);
void nvdr_image_free(NvdrImage* img);

double nvdr_psnr(const NvdrImage* a, const NvdrImage* b);

/* ----------------------------------------------------------------- tree */

/*
 * The full quadtree, built once at the finest tolerance. Every node keeps
 * the mean colour of its region and how far the region strays from it, so
 * a level can be cut out of the tree by thresholding that deviation
 * without ever touching pixels again.
 */
typedef struct {
    uint16_t x, y, w, h;
    int32_t  first_child;    /* -1 when this node was never split */
    float    deviation;      /* mean perceptual distance from the mean colour */
    float    penalty;        /* how much harder this region is to justify splitting */
    uint8_t  r, g, b;
} NvdrNode;

typedef struct {
    NvdrNode* nodes;
    uint32_t  count;
    uint32_t  capacity;
} NvdrTree;

typedef struct {
    int   min_tile;
    int   max_depth;
    /*
     * How much the tolerance tightens in dark regions.
     *
     * Deviation used to be measured against a constant 255, which makes
     * the metric one of absolute difference. The eye does not work that
     * way: an error of 7 on a pixel of value 16 is obvious, the same error
     * on a pixel of value 240 is invisible. Measured on
     * samples/montanha_pessoas.jpg, the darkest eighth of the image was
     * carrying 41% relative error against 1.3% in the brightest — dark
     * regions were being collapsed into single rectangles while the metric
     * reported them as uniform.
     *
     * The denominator is now `255 * (luma + weber) / (128 + weber)`, which
     * leaves mid-grey exactly where it was and tightens or loosens either
     * side of it. Smaller values push harder; a very large value reproduces
     * the old absolute metric.
     */
    float weber;
    /*
     * How coarsely fine grain is allowed to be resolved.
     *
     * Deviation says a region is not uniform; it does not say whether
     * subdividing would help. Distant grass varies as much inside a 4x4
     * window as across the whole patch, so splitting reproduces noise. A
     * face varies across the region and barely within a window, so
     * splitting resolves it. The ratio between the two is the grain of a
     * region, and it raises that region's minimum tile by
     * `1 + texture * grain`.
     *
     * This is a rate control, not a free improvement: it caps how fine
     * anything can get, so it cannot reach the high-quality end at all.
     * Below roughly half the default rate it beats plain tolerance by 3-4
     * dB; above that it is strictly worse. Zero, the default, disables it.
     */
    float texture;
    /*
     * The order refinement units are emitted in, which is also the order a
     * truncated stream delivers them. 0 keeps the depth-first order the
     * tree produces; 1 sends the largest rectangles first, so a prefix
     * covers the whole canvas coarsely instead of one corner finely.
     * Both sides derive it from the previous level, so nothing is sent.
     */
    int   order;
    /*
     * How much coarser the two chroma channels are quantised than luma.
     *
     * Residuals used to be RGB deltas with one step for all three, which
     * spends the same precision on colour as on brightness even though the
     * eye has far less resolution for the first. It also wastes bits on
     * redundancy: R, G and B move together, which is exactly why
     * conditioning each residual on the previous plane was worth 3.4% —
     * that gain was the coder recovering correlation the representation
     * should not have had.
     *
     * 1 keeps chroma as fine as luma. 0 drops the transform entirely and
     * codes RGB, for comparison.
     */
    int   chroma;
    float tolerance[NVDR_LEVELS];   /* strictly decreasing: coarse to fine */
    int   anchor_bits;              /* anchor palette is 1 << anchor_bits */
    int   step[NVDR_LEVELS];        /* residual quantisation step per level */
    int   codec;                    /* NVDR_COMPRESS_DEFLATE or _ARITH */
} NvdrConfig;

NvdrConfig nvdr_default_config(void);

int  nvdr_tree_build(NvdrTree* tree, const NvdrImage* img, const NvdrConfig* cfg);
void nvdr_tree_free(NvdrTree* tree);

/* ------------------------------------------------------------- pyramid */

/*
 * One decoded or encoded level: the rectangles visible at that tolerance
 * and the colour each of them shows.
 */
typedef struct {
    uint16_t* x;
    uint16_t* y;
    uint16_t* w;
    uint16_t* h;
    uint8_t*  rgb;        /* 3 bytes per rectangle, what gets painted */
    /*
     * The same reconstruction in the space the residuals run in. The chain
     * carries this rather than rgb because the round trip through RGB is
     * lossy by a few units, and re-deriving it at every level would let
     * that drift accumulate.
     */
    uint8_t*  chain;
    uint32_t  count;
} NvdrLevelData;

#define NVDR_ORDER_DFS   0
#define NVDR_ORDER_AREA  1

/* Colour space the residual chain runs in. */
#define NVDR_SPACE_RGB   0
#define NVDR_SPACE_YCC   1

typedef struct {
    NvdrLevelData level[NVDR_LEVELS];
    int           levels_present;   /* 1, 2 or 3 */
    /*
     * How much of the last level actually arrived. 1.0 when the stream was
     * complete; less when it was cut, in which case the units that never
     * came keep the rectangle and colour they had at the level before.
     */
    double        last_level_fraction;

    /* Anchor palette, shared by level 0 only. */
    unsigned char* palette;
    int            palette_count;
    int            anchor_bits;
    int            step[NVDR_LEVELS];
} NvdrPyramid;

void nvdr_pyramid_free(NvdrPyramid* pyr);

/* ------------------------------------------------------------ container */

#define NVDR_MAGIC       "NVDR"
#define NVDR_VERSION     7
#define NVDR_HEADER_SIZE 72

/* Compression applied to each level stream independently. */
#define NVDR_COMPRESS_NONE    0
#define NVDR_COMPRESS_DEFLATE 1
#define NVDR_COMPRESS_ARITH   2   /* adaptive arithmetic coding, see entropy.h */

typedef struct {
    uint16_t width, height;
    uint32_t leaf_count[NVDR_LEVELS];
    uint32_t split_bits[NVDR_LEVELS];   /* bits, not bytes */
    uint32_t raw_bytes[NVDR_LEVELS];    /* stream size once inflated */
    uint32_t stored_bytes[NVDR_LEVELS]; /* stream size on disk and on the wire */
    uint8_t  anchor_bits;
    uint8_t  compression;
    uint8_t  order;
    uint8_t  space;
    uint8_t  step[NVDR_LEVELS];
    uint8_t  chroma_step[NVDR_LEVELS];
} NvdrHeader;

/* Encode an image straight to a container. */
int nvdr_encode_file(const char* out_path, const NvdrImage* img,
                     const NvdrConfig* cfg, NvdrHeader* hdr_out);

/*
 * Read whatever is there. `levels_present` on the returned pyramid says how
 * far the bytes on disk actually reach: a file truncated mid-stream decodes
 * at the last level whose bytes are all present, rather than failing.
 */
int nvdr_decode_file(const char* path, NvdrPyramid* pyr, NvdrHeader* hdr);

/* --------------------------------------------------------------- render */

/* Paint the rectangles of one level onto an RGB canvas. */
void nvdr_render_level(const NvdrLevelData* level, NvdrImage* out);

/*
 * Soften the seams between rectangles, in place.
 *
 * Every pixel is averaged with its four neighbours at `weight` each. Inside
 * a rectangle the neighbours carry the same colour, so the average returns
 * it unchanged and the pass does nothing; only the one-pixel band along a
 * seam moves. That makes this exactly a boundary blend without needing to
 * know where the boundaries are.
 *
 * Costs no bytes and changes no format: it is a choice the decoder makes.
 * 0 disables it; 0.40 is the measured optimum.
 */
#define NVDR_SMOOTH_DEFAULT 0.40f
void nvdr_smooth(NvdrImage* img, float weight);

#endif /* NVDR_H */
