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
 *
 * and a pool of contexts outside that heap, for a sequence's frames,
 * whose models carry from one to the next:
 *
 *   k = nvdr_wasm_ctx_alloc()    an empty context, -1 when none is free
 *   nvdr_wasm_ctx_reset(k)       empty it again
 *   nvdr_wasm_ctx_free(k)        give it back
 *   nvdr_wasm_use_ctx(k)         the next decode starts from it and
 *                                leaves the frame's models in it
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
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

#define CTX_SLOTS 64
#define CTX_BYTES 16384
static uint64_t ctx_mem[CTX_SLOTS][CTX_BYTES / 8];
static uint8_t ctx_used[CTX_SLOTS];
static int use_ctx = -1;

__attribute__((export_name("nvdr_wasm_ctx_alloc")))
int32_t nvdr_wasm_ctx_alloc(void) {
    if (nvdr_context_size() > CTX_BYTES) return -1;
    for (int k = 0; k < CTX_SLOTS; k++)
        if (!ctx_used[k]) { ctx_used[k] = 1; memset(ctx_mem[k], 0, CTX_BYTES); return k; }
    return -1;
}

__attribute__((export_name("nvdr_wasm_ctx_reset")))
void nvdr_wasm_ctx_reset(int32_t k) { if (k >= 0 && k < CTX_SLOTS) memset(ctx_mem[k], 0, CTX_BYTES); }

__attribute__((export_name("nvdr_wasm_ctx_free")))
void nvdr_wasm_ctx_free(int32_t k) { if (k >= 0 && k < CTX_SLOTS) ctx_used[k] = 0; }

__attribute__((export_name("nvdr_wasm_use_ctx")))
void nvdr_wasm_use_ctx(int32_t k) { use_ctx = k >= 0 && k < CTX_SLOTS && ctx_used[k] ? k : -1; }

__attribute__((export_name("nvdr_wasm_input")))
uint8_t* nvdr_wasm_input(uint32_t n) {
    nvdr_heap_rewind(0);
    uint8_t* p = (uint8_t*)malloc(n ? n : 1);
    input = p;
    base.pixels = NULL;
    use_ctx = -1;
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
    NvdrContext* ctx = use_ctx >= 0 ? (NvdrContext*)ctx_mem[use_ctx] : NULL;
    use_ctx = -1;    /* one decode only: another view must not move it again */
    int rc = base.pixels || ctx ? nvdr_decode_mem_base(input, n, base.pixels ? &base : NULL, &out, NULL, &info, ctx)
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
