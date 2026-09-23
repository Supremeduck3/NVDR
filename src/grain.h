/*
 * Film grain synthesis: internal to the codec (nvdr.c), mirrored by the
 * grain functions in public/nvdr.js. See grain.c.
 */
#ifndef NVDR_GRAIN_H
#define NVDR_GRAIN_H

#include "nvdr.h"

#define NVDR_GRAIN_TEMPLATE 64
#define NVDR_GRAIN_KERNELS  5

/* One 64x64 template of unit grain (std 64) per component. */
typedef struct {
    int16_t t[3][NVDR_GRAIN_TEMPLATE * NVDR_GRAIN_TEMPLATE];
} NvdrGrainTemplates;

void nvdr_grain_pack(const NvdrGrain* g, uint8_t out[NVDR_GRAIN_SIZE]);
/* 0, or -1 when the bytes are not grain parameters this decoder knows. */
int  nvdr_grain_unpack(const uint8_t* in, size_t len, NvdrGrain* g);

void nvdr_grain_templates(const NvdrGrain* g, NvdrGrainTemplates* t);

/* Lays the grain over planes of YCbCr at full resolution, `stride` apart,
 * in place. Every noise value depends on the luma before grain. */
void nvdr_grain_apply(const NvdrGrain* g, const NvdrGrainTemplates* t,
                      uint8_t* y, uint8_t* cb, uint8_t* cr, int stride, int w, int h);

/*
 * Encoder side. Measures the picture's noise; when it is worth modelling
 * (always with `force`, otherwise when its luma spread passes a threshold)
 * writes the denoised picture to `clean` (allocated), fills `g` and
 * returns 1. Returns 0 when the picture is left as it is, -1 on failure.
 */
int  nvdr_grain_estimate(const NvdrImage* img, int force, NvdrImage* clean, NvdrGrain* g);

#endif
