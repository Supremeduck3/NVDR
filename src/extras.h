/*
 * What rides after a container's layers: a picture's transparency and
 * its colour profile.
 *
 * A decoder reads a container by the sizes in its header, so whatever
 * follows the last layer is not read: every decoder since v10 shows a
 * file with extras as the same picture, opaque and in sRGB. The extras
 * are chunks, any number, in any order:
 *
 *   [tag, 4 ASCII][length u32 LE][data]
 *
 *   "ALPH"  the alpha plane, itself an NVDR container of the picture's
 *           size: a grey picture whose green is the alpha. Coded like
 *           the picture (the same mode, a finer step: an edge's alpha
 *           off by a few levels shows as a halo).
 *   "ICCP"  the source's ICC profile, as it was found.
 *
 * They come last, so a progressive file shows its colours first and
 * turns transparent when its end arrives, and a file cut short is
 * opaque. An unknown tag is skipped.
 */
#ifndef NVDR_EXTRAS_H
#define NVDR_EXTRAS_H

#include "nvdr.h"

#define NVDR_EXTRA_ALPHA "ALPH"
#define NVDR_EXTRA_ICC   "ICCP"

/* The chunk `tag` of the file: 1 with where its data is and how long, 0
 * when the file has none (or was cut before it), -1 when it is no
 * container. */
int nvdr_extra_find(const uint8_t* data, size_t size, const char* tag, size_t* at, size_t* len);
/* The file with one more chunk at its end; `*buf` is reallocated. */
int nvdr_extra_append(uint8_t** buf, size_t* len, const char* tag, const uint8_t* data, size_t n);

/* An image with its alpha: `*alpha` is width * height bytes, or NULL when
 * every pixel is opaque. Where a pixel is fully transparent its colour is
 * replaced by the mean of the pixels that are not, which costs nothing
 * to code and is never seen. The same EXIF turning as nvdr_image_load(). */
int nvdr_image_load_alpha(NvdrImage* img, uint8_t** alpha, const char* path);
/* A JPEG's or a PNG's ICC profile, malloc'd, or NULL when it has none. */
uint8_t* nvdr_icc_read(const char* path, size_t* len);

/* Codes `alpha` (the picture's size) the way `cfg` codes the picture and
 * appends it to the file as its ALPH chunk. */
int nvdr_alpha_append(uint8_t** buf, size_t* len, const uint8_t* alpha, int width, int height,
                      const NvdrConfig* cfg);
/* The file's alpha plane, width * height bytes: 1 with `*alpha` malloc'd,
 * 0 when it has none, -1 when it has one that does not decode. */
int nvdr_alpha_decode(const uint8_t* data, size_t size, uint8_t** alpha);

/* A PNG with the alpha and the profile, either NULL for none. */
int nvdr_image_write_png_extra(const NvdrImage* img, const uint8_t* alpha, const uint8_t* icc, size_t icc_len,
                               const char* path);

#endif
