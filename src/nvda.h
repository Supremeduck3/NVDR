/*
 * NVDA — an album: many still images in one file, the fluid context
 * carried from each to the next.
 *
 *   [header 16B]  "NVDA", version 2, flags, image count u32, 6 reserved
 *   per image:    name length u16, name (UTF-8, at most 255 bytes),
 *                 kind u8, payload length u32, payload
 *
 * kind 0: the payload is an NVDR container, the image on its own.
 * kind 1: the image is predicted from the one before it, which must be
 *         the same size: nvdrv_predict_encode()'s payload, block motion in
 *         quarter pixels plus a residual container. This is the reuse that
 *         pays: a photo that repeats most of the one before it (a burst,
 *         the same scene, screenshots) costs only what changed.
 *
 * With NVDA_FLAG_FLUID every container starts from the adaptive models
 * the one before it ended with (nvdr.h, the fluid context), so the album
 * reads in order. What that is worth, measured on the six samples: 0.7%
 * of the bytes, up to 1.5% on one image. The models adapt within a few
 * dozen symbols, so starting from even odds costs an image little. It is
 * on because it costs nothing either. Only kind-0 images carry it.
 *
 * With NVDA_FLAG_PREDICT the encoder tries both kinds for every image the
 * size of the one before, at equal quality: the prediction is coded with a
 * finer step until it is within 0.1 dB of the image coded alone, and is
 * kept if it is still smaller.
 *
 * A file cut short gives every image before the cut whole and the one
 * the cut lands in as far as its bytes reach.
 */
#ifndef NVDA_H
#define NVDA_H

#include "nvdr.h"

#define NVDA_MAGIC       "NVDA"
#define NVDA_VERSION     2
#define NVDA_HEADER_SIZE 16
#define NVDA_FLAG_FLUID  0x01
#define NVDA_FLAG_PREDICT 0x02   /* informational: the encoder tried kind 1 */
#define NVDA_KIND_INTRA  0
#define NVDA_KIND_PRED   1
#define NVDA_MAX_IMAGES  4096

/* Encode `count` images into `path`. `bytes_out`, when given, receives each
 * image's payload size, `cold_out` what it would have been coded alone
 * with no context (encoded a second time, for the report), and `kind_out`
 * each image's kind. */
int nvda_write(const char* path, int count, const char* const* names, const NvdrImage* imgs,
               const NvdrConfig* cfg, int fluid, int predict,
               size_t* bytes_out, size_t* cold_out, int* kind_out);

typedef struct {
    const uint8_t* data;
    size_t         size, pos;
    uint32_t       count, index;
    int            flags;
    NvdrContext*   ctx;
    NvdrImage      prev;    /* the last image, what a kind-1 image predicts from */
} NvdaReader;

/* 0 on success; -1 when this is not an album. `data` must outlive it. */
int  nvda_open(NvdaReader* r, const uint8_t* data, size_t size);
void nvda_close(NvdaReader* r);

/* The next image: 1 with `out` allocated, 0 at the end (or at a cut before
 * any of this image's colour), -1 on damage. `name` gets the stored name,
 * NUL-terminated and cut to `cap`. */
int  nvda_next(NvdaReader* r, char* name, size_t cap, NvdrImage* out, int* kind,
               int* partial);

#endif /* NVDA_H */
