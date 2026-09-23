/*
 * The decoder's face to JavaScript (public/nvdr-wasm.js). One input
 * buffer, decoded as many times as the page wants views of it:
 *
 *   p = nvdr_wasm_input(n)       room for the container, written by JS
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

__attribute__((export_name("nvdr_wasm_input")))
uint8_t* nvdr_wasm_input(uint32_t n) {
    nvdr_heap_rewind(0);
    uint8_t* p = (uint8_t*)malloc(n ? n : 1);
    input = p;
    after_input = nvdr_heap_top();
    return p;
}

__attribute__((export_name("nvdr_wasm_decode")))
Result* nvdr_wasm_decode(uint32_t n, int32_t max_layer) {
    NvdrImage out;
    NvdrDecodeInfo info;
    if (nvdr_decode_mem(input, n, max_layer, &out, NULL, &info) != 0) return 0;
    result.rgb = (uint32_t)(uintptr_t)out.pixels;
    result.width = out.width; result.height = out.height;
    result.tiles = info.tiles; result.layers_present = info.layers_present;
    for (int k = 0; k < NVDR_LAYERS; k++) result.tiles_complete[k] = info.tiles_complete[k];
    return &result;
}

__attribute__((export_name("nvdr_wasm_release")))
void nvdr_wasm_release(void) { nvdr_heap_rewind(after_input); }
