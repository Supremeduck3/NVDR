/*
 * In-loop restoration: internal to the codec (nvdr.c), mirrored by the
 * restoration functions in public/nvdr.js. See restore.c.
 *
 * A container with NVDR_FLAG_RESTORE carries, after the grain parameters
 * and before layer 0, `restore_len` bytes (header bytes 30 and 31) that
 * say how to restore each of Y, Cb and Cr once the picture is decoded.
 * Each component names a KIND and then that kind's own parameters, so a
 * new restoration method is a new kind, parsed and applied by its own
 * pair of functions, next to the old ones rather than instead of them:
 *
 *   0  none
 *   1  ALF: a classified, clipped linear filter fitted to the picture by
 *      the encoder (restore.c)
 *   2  reserved for a learned filter: a network whose weights both sides
 *      hold, named by an id, with a few per-picture parameters (a
 *      strength, a class map) in its payload. Its apply() goes where
 *      kind 1's does and its fit() in nvdr_restore_fit()'s list of
 *      candidates, which keeps whichever kind costs least.
 *
 * Every kind works on one component's decoded 8-bit plane, reads only
 * the unfiltered plane and writes a filtered one, and is exact in
 * integers, identical in C, JS and WebAssembly. Each also carries a map
 * of 64x64 units (in luma pixels) it is on in, so the encoder can leave
 * alone what a filter would harm.
 */
#ifndef NVDR_RESTORE_H
#define NVDR_RESTORE_H

#include <stddef.h>
#include <stdint.h>

#define NVDR_RESTORE_NONE  0
#define NVDR_RESTORE_ALF   1
#define NVDR_RESTORE_KINDS 2    /* kinds this decoder knows */

#define NVDR_ALF_CLASSES   25   /* 5 directions x 5 activities, luma */
#define NVDR_ALF_TAPS_Y    12   /* 7x7 diamond: 12 symmetric pairs */
#define NVDR_ALF_TAPS_C    6    /* 5x5 diamond: 6 symmetric pairs */
#define NVDR_ALF_SHIFT     7    /* coefficients in 128ths */
#define NVDR_RESTORE_UNIT  64   /* the on/off unit, in luma pixels */

/* One component's restoration, as read from the container. The unit map
 * points into the container's bytes (bit `unit_bit` onward, MSB first). */
typedef struct {
    int kind;
    int clip;                                  /* index into the clip bounds */
    int nfilters;
    uint8_t classmap[NVDR_ALF_CLASSES];        /* class -> filter */
    int8_t coef[NVDR_ALF_CLASSES][NVDR_ALF_TAPS_Y];
    int all_units;                             /* on everywhere, no map */
    int units_x, units_y;
    const uint8_t* unit_bits;
    size_t unit_bit;
} NvdrRestoreComp;

typedef struct {
    NvdrRestoreComp comp[3];
} NvdrRestore;

/* The geometry of one component's plane: its canvas (padded) and how
 * many of its pixels make one unit (64 for luma and 4:4:4 colour, 32 for
 * colour at half size). */
typedef struct { int pw, ph, unit; } NvdrRestorePlane;

/* 0, or -1 when the bytes are not restoration parameters this decoder
 * knows or run past `len`. */
int  nvdr_restore_parse(const uint8_t* in, size_t len, const NvdrRestorePlane* planes, NvdrRestore* r);

/* Restores one component's plane in place. Does nothing for a component
 * whose kind is none; -1 when out of memory. */
int  nvdr_restore_apply(const NvdrRestoreComp* rc, int comp, uint8_t* plane, const NvdrRestorePlane* g);

/* ---------------------------------------------------------------- encoder */

typedef struct {
    uint8_t* bytes;
    size_t count, cap;
    int bit;                 /* bits used in the last byte, 0..7 */
} NvdrBits;

/* What the encoder hands the decoder when it fits: the source of each
 * component on its canvas, the lambda bits are weighed at, and where the
 * parameters are written. */
typedef struct {
    const double* src[3];
    /* Colour at half size is judged after upsampling, against these:
     * the full-size source on the luma canvas (stride full_pw, visible
     * full_w x full_h). NULL when colour is whole. */
    const double* full_src[3];
    int full_pw, full_w, full_h;
    double lambda[3];
    NvdrBits out;
    int failed;
    /* filled in: squared error removed per component, and bits spent */
    double gain[3];
    size_t bits[3];
} NvdrRestoreFit;

/* Chooses a restoration for one component's decoded plane (visible area
 * w x h), writes it to fit->out, and restores the plane in place as the
 * decoder will. Components must come in order, 0 to 2. */
int  nvdr_restore_fit(NvdrRestoreFit* fit, int comp, uint8_t* plane, const NvdrRestorePlane* g, int w, int h);
void nvdr_bits_free(NvdrBits* b);

#endif
