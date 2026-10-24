/*
 * An NVDR file dressed as a PNG: an ordinary PNG whose picture is a
 * preview, carrying the file untouched in its private `nvDr` chunk. See
 * pngwrap.c for the layout and how the preview is made.
 */
#ifndef NVDR_PNGWRAP_H
#define NVDR_PNGWRAP_H

#include "nvdr.h"

typedef struct {
    int    side;       /* the preview's longer side at most (0 whole) */
    int    colours;    /* palette size, 2..256, 0 for full colour */
    int    refine;     /* k-means rounds on the palette */
    int    dither;     /* percent of the colour error diffused */
    double budget;     /* the preview costs at most this times the file (0: none) */
} NvdrPngOptions;

typedef struct {
    size_t bytes;              /* the whole PNG */
    int    width, height;      /* the preview's */
    int    colours, dithered;
} NvdrPngResult;

NvdrPngOptions nvdr_png_options(void);

/* Writes `data` (an NVDR image, album or video, len bytes) as a PNG at
 * `path`, with `picture` as its preview. 0, or -1 when it cannot. Safe to
 * call from several threads at once. */
int nvdr_png_write(const char* path, const uint8_t* data, size_t len, const NvdrImage* picture,
                   const NvdrPngOptions* o, NvdrPngResult* res);

/* Where the NVDR file inside a PNG starts and how long it is: 1 found,
 * 0 none, -1 not a PNG or a damaged chunk. */
int nvdr_png_find(const uint8_t* png, size_t len, size_t* at, size_t* n);

#endif
