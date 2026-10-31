/*
 * A sequence frame's motion-compensated prediction, made block by block
 * from the reference frames themselves (mc.c): what the decoders do,
 * where the encoder, which reads every position many times while it
 * searches, builds the sixteen subpixel planes of each reference once.
 * The pixels are the planes' to the bit.
 */
#ifndef NVDR_MC_H
#define NVDR_MC_H

#include <stddef.h>
#include <stdint.h>

/* A cell's mode in a frame with two references, as nvdrv.c's MODE_. */
#define MC_BI  0
#define MC_FWD 1     /* the first reference only */
#define MC_BWD 2     /* the second only */

/*
 * The prediction of a width x height RGB frame into `dst`, from `r0` and,
 * where `mode` is given, `r1` (NULL `mode`: every cell from r0 alone).
 * The field is on a grid of g x g cells, nfx x nfy of them, with vectors
 * in quarter pixels; `tmp` is width * height * 3 bytes of scratch, used
 * only with two references. `obmc` blends each cell's edges with its
 * neighbours' predictions, as FRAME_OBMC asks.
 */
void mc_predict_frame(const uint8_t* r0, const uint8_t* r1, uint8_t* dst, uint8_t* tmp,
                      int width, int height, int g, int nfx, int nfy,
                      const int16_t* v0x, const int16_t* v0y, const int16_t* v1x, const int16_t* v1y,
                      const uint8_t* mode, int obmc);

#endif
