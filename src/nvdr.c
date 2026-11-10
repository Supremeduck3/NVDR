/*
 * NVDR still images, format v10. See nvdr.h for the design; NOTEBOOK.md
 * for the measurements behind it.
 */
/* NVDR_WASM builds the decoder for the browser (wasm/): no files, no
 * image formats, no zlib; everything else is the same code. */
#ifndef NVDR_WASM
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

#include "nvdr.h"
#include "grain.h"
#include "restore.h"
#include "entropy.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef NVDR_WASM
#include <zlib.h>
#endif

/*
 * One translation unit, in parts: everything here is static and inlined
 * across what used to be one file of four thousand lines, so the parts
 * are included, in the order they depend on each other, and not
 * compiled apart.
 */
#include "nvdr_image.inc"
#include "nvdr_transform.inc"
#include "nvdr_models.inc"
#include "nvdr_planes.inc"
#include "nvdr_header.inc"
#include "nvdr_encode.inc"
#include "nvdr_decode.inc"
