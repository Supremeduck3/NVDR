/*
 * NVDA — an album: many still images in one file, where a photo that
 * repeats an earlier one is coded as what changed.
 *
 *   [header 16B]  "NVDA", version 3, flags, image count u32, window u8,
 *                 5 reserved
 *   [index]       16 bytes per image:
 *                   payload offset u32 (from the start of the file)
 *                   payload length u32
 *                   kind u8, reference distance u8 (1..window; 0 when alone)
 *                   width u16, height u16, name length u16
 *   [images]      per image: its name (UTF-8), then its payload
 *
 * kind 0: the payload is an NVDR container, the image on its own.
 * kind 1: the image is predicted from the image `reference distance`
 *         places before it, which must be the same size: block motion in
 *         quarter pixels plus a residual container (nvdrv_predict_encode).
 *
 * WHY AN INDEX, AND A WINDOW
 * --------------------------
 * A predicted photo does not exist on its own: showing it means decoding
 * the photo it was predicted from, and that one's reference in turn. The
 * index says, before a single image is read, where every image is and
 * what it depends on, so a viewer that wants one photo fetches that photo
 * and its chain of references and nothing else, with HTTP range requests
 * when the file is on a server. The window bounds how far back a
 * reference may reach, so a decoder holds at most `window` images, and a
 * chain is only as long as the photos really repeat each other.
 *
 * THE FLUID CONTEXT
 * -----------------
 * With NVDA_FLAG_FLUID, every photo coded on its own starts from the
 * models the one before it left (nvdr.h). That is worth 0.7% on the
 * samples and makes every photo depend on all the ones before it, so it
 * is off unless asked for: an album a site serves one photo at a time
 * wants each photo reachable on its own.
 *
 * The encoder tries, for every photo, the photo alone and a prediction
 * from the most similar earlier photos of the same size in the window, at
 * equal quality: a prediction is coded with a finer step until it is
 * within 0.1 dB of the photo alone, and is kept only if it is still
 * smaller. An album is never bigger than its photos coded alone.
 */
#ifndef NVDA_H
#define NVDA_H

#include "nvdr.h"

#define NVDA_MAGIC        "NVDA"
#define NVDA_VERSION      3
#define NVDA_HEADER_SIZE  16
#define NVDA_ENTRY_SIZE   16
#define NVDA_FLAG_FLUID   0x01
#define NVDA_FLAG_PREDICT 0x02   /* informational: the encoder tried kind 1 */
#define NVDA_KIND_INTRA   0
#define NVDA_KIND_PRED    1
#define NVDA_MAX_IMAGES   4096
#define NVDA_MAX_WINDOW   32
#define NVDA_DEFAULT_WINDOW 8

typedef struct {
    int    fluid;      /* carry the fluid context between photos coded alone */
    int    predict;    /* try predicting photos from earlier ones */
    int    window;     /* how many earlier photos a prediction may reach, 1..32 */
    int    candidates; /* how many of them, the most similar first, to try in full */
} NvdaOptions;

NvdaOptions nvda_default_options(void);

typedef struct {
    size_t bytes;      /* payload */
    size_t alone;      /* the photo coded alone, for the report */
    int    kind;
    int    ref;        /* reference distance, 0 when alone */
} NvdaReport;

/* Encode `count` images into `path`. `report`, when given, gets one entry
 * per image. */
int nvda_write(const char* path, int count, const char* const* names, const NvdrImage* imgs,
               const NvdrConfig* cfg, const NvdaOptions* opt, NvdaReport* report);

typedef struct {
    uint32_t offset, length;
    int      kind, ref, width, height, name_len;
} NvdaEntry;

typedef struct {
    const uint8_t* data;
    size_t         size;
    uint32_t       count;
    int            flags, window;
    NvdaEntry*     index;
    /* Decoded images kept for references: the last `window` decoded, by
     * position. */
    NvdrImage*     ring;
    int*           ring_of;     /* which image each ring slot holds, -1 empty */
    NvdrContext*   ctx;
    int            next_fluid;  /* with the fluid context: the next image in order */
} NvdaReader;

/*
 * Open an album from the bytes available. The header and the whole index
 * must be there; images need not be. 0 on success, -1 when it is not an
 * album or its index is damaged.
 */
int  nvda_open(NvdaReader* r, const uint8_t* data, size_t size);
void nvda_close(NvdaReader* r);

/*
 * Decode image `i`, decoding the images its chain of references needs
 * first when they are not already held. 1 with `out` allocated; 0 when its
 * bytes (or a reference's) have not arrived; -1 on damage. With the fluid
 * context images must be read in order, and any other order returns -1.
 * `partial` is set when the image's own bytes were cut short.
 */
int  nvda_decode(NvdaReader* r, int i, NvdrImage* out, int* partial);

/* The image's stored name, NUL-terminated and cut to `cap`. */
void nvda_name(const NvdaReader* r, int i, char* name, size_t cap);

#endif /* NVDA_H */
