/*
 * NVDR — still images, format v10: the quadtree as the partition of a
 * transform.
 *
 * WHAT CHANGED FROM v9, AND WHY
 * -----------------------------
 * Up to v9 an image was a stack of flat rectangles, each level a finer
 * quadtree with a colour correction per rectangle. Measured against a
 * DCT at the same size it was 6 to 11 dB behind on every sample, and the
 * reason was structural: a flat rectangle cannot hold texture finer than
 * the smallest tile, so the finest level spent most of the file
 * subdividing into grain (see README, "A texture layer, measured"). v9
 * lives on in reference/nvdr_v9 so the numbers measured with it stay
 * reproducible.
 *
 * In v10 the quadtree keeps its job of deciding where detail is, and each
 * of its leaves carries two things:
 *
 *   - a colour, predicted from the neighbouring leaves and corrected by a
 *     small delta. A leaf with nothing else is exactly a v9 rectangle.
 *   - optionally, texture: the DCT of what the flat colour misses, at the
 *     leaf's own size (4, 8, 16 or 32 pixels square).
 *
 * The encoder chooses the tree by rate and distortion: a node splits only
 * when its four children, coded for real against the adaptive models,
 * cost less in error + lambda * bits than the node coded whole.
 *
 * THREE LAYERS, AND THE TRUNCATION GUARANTEE
 * ------------------------------------------
 * Layer 0 is the tree and every leaf's colour: a complete picture of flat
 * rectangles on its own. Layer 1 is every leaf's low-frequency texture,
 * layer 2 the rest (format v11; the split is `band`). Each is one
 * arithmetic stream, coded 32x32 tile by tile in raster order, and a
 * stream cut short decodes every tile that arrived whole. A tile of layer
 * 0 that did not arrive is painted neutral grey; a tile of a texture layer
 * that did not arrive shows what the layer before gave it. So any prefix
 * of the file past the header decodes, and every further byte adds to it,
 * and since the coarse texture of the whole picture comes before any of
 * its detail, a half-delivered file is sharp everywhere rather than at the
 * top.
 *
 * That guarantee costs one thing. A leaf's colour is predicted from the
 * flat colours around it, never from their texture, so that layer 0
 * decodes without layer 1. Predicting from the full reconstruction would
 * be slightly better and would make layer 0 depend on layer 1.
 *
 * EXACTNESS
 * ---------
 * The C decoder and the browser's must produce the same pixels, and a
 * video's predicted frames are added to what the decoder holds, so a
 * one-bit difference compounds. Everything the decoder computes is
 * integer: the inverse transform is HEVC's integer approximation of the
 * DCT with fixed shifts, the colour conversion is fixed point, and the
 * prediction is an integer mean. The forward transform only runs in the
 * encoder and is free to use floating point.
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
int  nvdr_image_write_png(const NvdrImage* img, const char* path);
/* PNG when the path ends in .png, PPM otherwise. */
int  nvdr_image_write(const NvdrImage* img, const char* path);
void nvdr_image_free(NvdrImage* img);
double nvdr_psnr(const NvdrImage* a, const NvdrImage* b);

/* --------------------------------------------------------------- config */

#define NVDR_LAYERS      3
#define NVDR_MIN_BLOCK   4
#define NVDR_MAX_BLOCK   32

typedef struct {
    /* Quantiser step for luma, in orthonormal DCT units. It is the one
     * quality knob: halving it costs roughly twice the bytes and buys
     * about 5 dB. */
    int   q;
    /* Chroma's step relative to luma's. */
    float chroma_q;
    /* How far below one half a coefficient must fall to round to zero.
     * 0 rounds to nearest; larger values trade small coefficients for
     * bytes. */
    float deadzone;
    /* lambda = lambda_k * q^2, the slope the split decision weighs bits
     * against squared error at. */
    float lambda_k;
    int   max_block;   /* 4..32, a power of two: the tile size */
    int   min_block;   /* 4..max_block */
    /* The image is a residual centred on 128, as a sequence's predicted
     * frames are. Every leaf's colour is then predicted as 128 rather than
     * from its neighbours: a residual's neighbours say nothing about it,
     * and on the clean clip predicting from them cost 13% more colour
     * bytes and 5% more texture bytes. Carried in the header's flags. */
    int   residual;
    /* Filter the seams between leaves after decoding (see deblock() in
     * nvdr.c). Carried in the header's flags. */
    int   deblock;
    /* In a residual, how readily a leaf is left uncorrected, as a multiple
     * of lambda: its bits are weighed at skip_k * lambda against the error
     * it would remove. 0 codes every leaf. */
    float skip_k;
    /* Where texture splits between its two layers: low frequencies are
     * u + v <= max(1, n * band / 32) in an n x n leaf. 0 keeps all texture
     * in one layer (see band_split() in nvdr.c). */
    int   band;
    /* Colour at half resolution each way (4:2:0): Cb and Cr get their own
     * quadtree over a half-size canvas, tile by tile after luma's, and the
     * decoder scales them back up. Carried in the header's flags. Needs
     * max_block of 8 or more. NVDR_CHROMA_AUTO (the default) picks per
     * image: see nvdr_encode_mem_ctx(). */
    int   chroma420;
    /* Film grain synthesis (see grain.c): the encoder takes the sensor
     * noise out, codes the clean picture, and stores 22 bytes that let the
     * decoder lay statistically the same grain back over it. NVDR_GRAIN_OFF,
     * _AUTO (only when the picture is measurably noisy) or _ON. Carried in
     * the header's flags. */
    int   grain;
} NvdrConfig;

#define NVDR_GRAIN_OFF   0
#define NVDR_GRAIN_AUTO  1
#define NVDR_GRAIN_ON    2

/* The grain a decoder lays over the picture: see grain.c. */
#define NVDR_GRAIN_POINTS 16
#define NVDR_GRAIN_SIZE   22
typedef struct {
    uint16_t seed;
    uint8_t  kernel;                    /* how far a grain spreads, 0..NVDR_GRAIN_KERNELS-1 */
    uint8_t  cb, cr;                    /* colour grain, x32 of luma's */
    uint8_t  sigma[NVDR_GRAIN_POINTS];  /* luma grain x8 at Y = 17 k */
} NvdrGrain;

#define NVDR_CHROMA_444  0
#define NVDR_CHROMA_AUTO 1
#define NVDR_CHROMA_420  2

NvdrConfig nvdr_default_config(void);

/* ------------------------------------------------------------ container */

/* The largest canvas the decoders will allocate for: 134 Mpx, well past
 * any real image, well short of what a damaged header could ask for. */
#define NVDR_MAX_PIXELS  ((size_t)1 << 27)

#define NVDR_MAGIC       "NVDR"
#define NVDR_VERSION     11
#define NVDR_HEADER_SIZE 32
#define NVDR_FLAG_RESIDUAL 0x01         /* colours predicted as 128 */
#define NVDR_FLAG_DEBLOCK  0x02         /* leaf seams filtered after decoding */
#define NVDR_FLAG_CHROMA420 0x04        /* colour in its own half-resolution tree */
#define NVDR_FLAG_GRAIN    0x08         /* grain parameters follow the header */

typedef struct {
    uint16_t width, height;
    uint8_t  max_block, min_block;
    uint16_t q_luma, q_chroma;
    uint8_t  flags;                     /* NVDR_FLAG_* */
    uint8_t  band;
    uint8_t  grain_len;                 /* bytes of grain parameters after the header */
    NvdrGrain grain;
    uint32_t stored_bytes[NVDR_LAYERS];
    /* Filled by the encoder only: leaves of 4, 8, 16 and 32 pixels, and
     * how many of them carry texture. */
    uint32_t leaves[4];
    uint32_t textured;
    /* and what the colours and textures of Y, Cb and Cr cost, in bits as
     * the models priced them (the tree's split flags are in none) */
    double   plane_bits[3];
} NvdrHeader;

int nvdr_encode_mem(uint8_t** out_buf, size_t* out_len, const NvdrImage* img,
                    const NvdrConfig* cfg, NvdrHeader* hdr_out);
int nvdr_encode_file(const char* out_path, const NvdrImage* img,
                     const NvdrConfig* cfg, NvdrHeader* hdr_out);

typedef struct {
    int layers_present;               /* 1 or 2: how far the bytes reached */
    int tiles;                        /* tiles in the image */
    int tiles_complete[NVDR_LAYERS];  /* how many of them each layer carried */
} NvdrDecodeInfo;

/*
 * Decode as far as the bytes reach, rendering layers up to `max_layer`
 * (-1 for all of them) into `out`, which the call allocates. Fewer bytes
 * than the container holds is the truncation case, not an error. Returns
 * -1 only when there is no picture at all: a header that is missing,
 * damaged, or asks for more than NVDR_MAX_PIXELS.
 */
int nvdr_decode_mem(const uint8_t* data, size_t size, int max_layer,
                    NvdrImage* out, NvdrHeader* hdr, NvdrDecodeInfo* info);
int nvdr_decode_file(const char* path, int max_layer,
                     NvdrImage* out, NvdrHeader* hdr, NvdrDecodeInfo* info);

/*
 * The fluid context: the adaptive models one container leaves behind,
 * carried into the next so it does not start from even odds (see
 * nvdr.c). An empty or reset context behaves exactly like none. Encoder
 * and decoder must feed their contexts the same containers in the same
 * order, each decoded whole.
 */
typedef struct NvdrContext NvdrContext;
NvdrContext* nvdr_context_new(void);
void nvdr_context_free(NvdrContext* ctx);
void nvdr_context_reset(NvdrContext* ctx);
void nvdr_context_copy(NvdrContext* dst, const NvdrContext* src);
int  nvdr_context_equal(const NvdrContext* a, const NvdrContext* b);

int nvdr_encode_mem_ctx(uint8_t** out_buf, size_t* out_len, const NvdrImage* img,
                        const NvdrConfig* cfg, NvdrHeader* hdr_out, NvdrContext* ctx);
int nvdr_decode_mem_ctx(const uint8_t* data, size_t size, int max_layer, NvdrImage* out,
                        NvdrHeader* hdr, NvdrDecodeInfo* info, NvdrContext* ctx);

#endif /* NVDR_H */
