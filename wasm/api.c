/*
 * The decoder's face to JavaScript (public/nvdr-wasm.js). One input
 * buffer, decoded as many times as the page wants views of it:
 *
 *   p = nvdr_wasm_input(n)       room for the container, written by JS
 *   b = nvdr_wasm_base(w, h)     room for the picture a container with
 *                                NVDR_FLAG_INTER is predicted from, RGB,
 *                                written by JS after the input
 *   r = nvdr_wasm_decode(n, L)   decode its first n bytes up to layer L;
 *                                r points at a Result, 0 when no picture
 *   nvdr_wasm_release()          give back everything since the input
 */
#include <stdint.h>
#include <stdlib.h>
#include "nvdr.h"

uintptr_t nvdr_heap_top(void);
void nvdr_heap_rewind(uintptr_t to);

typedef struct {
    uint32_t rgb;                 /* pointer to width * height * 3 bytes */
    int32_t  width, height;
    int32_t  tiles, layers_present;
    int32_t  tiles_complete[NVDR_LAYERS];
} Result;

static const uint8_t* input;
static uintptr_t after_input;
static Result result;
static NvdrImage base;

__attribute__((export_name("nvdr_wasm_input")))
uint8_t* nvdr_wasm_input(uint32_t n) {
    nvdr_heap_rewind(0);
    uint8_t* p = (uint8_t*)malloc(n ? n : 1);
    input = p;
    base.pixels = NULL;
    after_input = nvdr_heap_top();
    return p;
}

__attribute__((export_name("nvdr_wasm_base")))
uint8_t* nvdr_wasm_base(int32_t w, int32_t h) {
    base.pixels = (unsigned char*)malloc((size_t)w * h * 3);
    base.width = w; base.height = h;
    after_input = nvdr_heap_top();
    return base.pixels;
}

__attribute__((export_name("nvdr_wasm_decode")))
Result* nvdr_wasm_decode(uint32_t n, int32_t max_layer) {
    NvdrImage out;
    NvdrDecodeInfo info;
    int rc = base.pixels ? nvdr_decode_mem_base(input, n, &base, &out, NULL, &info)
                         : nvdr_decode_mem(input, n, max_layer, &out, NULL, &info);
    if (rc != 0) return 0;
    result.rgb = (uint32_t)(uintptr_t)out.pixels;
    result.width = out.width; result.height = out.height;
    result.tiles = info.tiles; result.layers_present = info.layers_present;
    for (int k = 0; k < NVDR_LAYERS; k++) result.tiles_complete[k] = info.tiles_complete[k];
    return &result;
}

__attribute__((export_name("nvdr_wasm_release")))
void nvdr_wasm_release(void) { nvdr_heap_rewind(after_input); }
