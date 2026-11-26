/*
 * The decoder's face to JavaScript (public/nvdr-wasm.js). One input
 * buffer, decoded as many times as the page wants views of it:
 *
 *   p = nvdr_wasm_input(n)       room for the container, written by JS
 *                                (it takes the place of the one before)
 *   b = nvdr_wasm_base(w, h)     room for the picture a container with
 *                                NVDR_FLAG_INTER is predicted from, RGB,
 *                                written by JS after the input
 *   h = nvdr_wasm_holes(k)       room for k byte ranges [start, end) that
 *                                never arrived, as 2k uint32, written by
 *                                JS; decodes until the next input then
 *                                read around them (nvdr_decode_mem_lossy)
 *   r = nvdr_wasm_decode(n, L)   decode its first n bytes up to layer L;
 *                                r points at a Result, 0 when no picture
 *   nvdr_wasm_release()          give back the picture of the last decode
 *
 * a pool of contexts, for containers whose models carry from one to the
 * next (a fluid album's photos):
 *
 *   k = nvdr_wasm_ctx_alloc()    an empty context, -1 when none is free
 *   nvdr_wasm_ctx_reset(k)       empty it again
 *   nvdr_wasm_ctx_free(k)        give it back
 *   nvdr_wasm_use_ctx(k)         the next decode starts from it and
 *                                leaves the frame's models in it
 *
 * and the sequence decoder of src/nvdrv.c, and its one picture predicted
 * from another (an album's), further down.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "nvdr.h"
#include "nvdrv.h"

typedef struct {
    uint32_t rgb;                 /* pointer to width * height * 3 bytes */
    int32_t  width, height;
    int32_t  tiles, layers_present;
    int32_t  tiles_complete[NVDR_LAYERS];
} Result;

static const uint8_t* input;
static size_t* holes;
static int nholes;
static NvdrImage last;      /* the last decode's picture, until released */
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
    free((void*)input); free(base.pixels); free(holes);
    free(last.pixels); last.pixels = NULL;
    base.pixels = NULL;
    holes = NULL; nholes = 0;
    use_ctx = -1;
    uint8_t* p = (uint8_t*)malloc(n ? n : 1);
    input = p;
    return p;
}

__attribute__((export_name("nvdr_wasm_base")))
uint8_t* nvdr_wasm_base(int32_t w, int32_t h) {
    free(base.pixels);
    base.pixels = (unsigned char*)malloc((size_t)w * h * 3);
    base.width = w; base.height = h;
    return base.pixels;
}

__attribute__((export_name("nvdr_wasm_holes")))
uint32_t* nvdr_wasm_holes(int32_t k) {
    _Static_assert(sizeof(size_t) == sizeof(uint32_t), "wasm32: a byte offset is 32 bits");
    free(holes);
    holes = (size_t*)malloc((size_t)(k > 0 ? k : 1) * 2 * sizeof(size_t));
    nholes = holes && k > 0 ? k : 0;
    return (uint32_t*)holes;
}

__attribute__((export_name("nvdr_wasm_decode")))
Result* nvdr_wasm_decode(uint32_t n, int32_t max_layer) {
    NvdrImage out;
    NvdrDecodeInfo info;
    NvdrContext* ctx = use_ctx >= 0 ? (NvdrContext*)ctx_mem[use_ctx] : NULL;
    use_ctx = -1;    /* one decode only: another view must not move it again */
    int rc = nholes ? nvdr_decode_mem_lossy(input, n, holes, nholes, max_layer, &out, NULL, &info)
           : base.pixels || ctx ? nvdr_decode_mem_base(input, n, base.pixels ? &base : NULL, &out, NULL, &info, ctx)
                                : nvdr_decode_mem(input, n, max_layer, &out, NULL, &info);
    if (rc != 0) return 0;
    free(last.pixels);
    last = out;
    result.rgb = (uint32_t)(uintptr_t)out.pixels;
    result.width = out.width; result.height = out.height;
    result.tiles = info.tiles; result.layers_present = info.layers_present;
    for (int k = 0; k < NVDR_LAYERS; k++) result.tiles_complete[k] = info.tiles_complete[k];
    return &result;
}

__attribute__((export_name("nvdr_wasm_release")))
void nvdr_wasm_release(void) { free(last.pixels); last.pixels = NULL; }

/* Room of the decoder's own heap, for what JS hands the calls below. */
__attribute__((export_name("nvdr_wasm_alloc")))
void* nvdr_wasm_alloc(uint32_t n) { return malloc(n ? n : 1); }

__attribute__((export_name("nvdr_wasm_free")))
void nvdr_wasm_free(void* p) { free(p); }

/*
 * A sequence, for public/nvdrv.js, decoded by the command-line decoder's
 * code (nvdrv_decode_next()):
 *
 *   p = nvdr_wasm_alloc(n)           the file, written by JS
 *   s = nvdr_wasm_seq_open(p, n)     a decoder, which keeps p; 0 when the
 *                                    header makes no sense (p is freed)
 *   f = nvdr_wasm_seq_next(s)        the next frame in display order:
 *                                    f points at a SeqFrame, whose rc is
 *                                    1 for a frame, 0 at the end, -1 on
 *                                    damage; the pixels stay until the
 *                                    next call
 *   nvdr_wasm_seq_close(s)
 */
typedef struct {
    int32_t rc, display, kind, partial;
    uint32_t rgb;
    int32_t width, height, fps, frames;
} SeqFrame;

typedef struct {
    SeqFrame f;              /* first: s is where JS reads the size from */
    NvdrvDecoder* dec;
    NvdrImage out;
} Seq;

__attribute__((export_name("nvdr_wasm_seq_open")))
Seq* nvdr_wasm_seq_open(uint8_t* data, uint32_t n) {
    Seq* s = (Seq*)calloc(1, sizeof(*s));
    NvdrvInfo info;
    if (!s) { free(data); return 0; }
    if (nvdrv_decode_open_mem(&s->dec, data, n, &info) != 0) { free(s); return 0; }
    s->out.width = info.width; s->out.height = info.height;
    s->out.pixels = (unsigned char*)malloc((size_t)info.width * info.height * 3);
    if (!s->out.pixels) { nvdrv_decode_close(s->dec); free(s); return 0; }
    s->f.width = info.width; s->f.height = info.height;
    s->f.fps = info.fps; s->f.frames = info.frame_count;
    s->f.rgb = (uint32_t)(uintptr_t)s->out.pixels;
    return s;
}

__attribute__((export_name("nvdr_wasm_seq_next")))
SeqFrame* nvdr_wasm_seq_next(Seq* s) {
    int kind = 0, partial = 0;
    s->f.rc = nvdrv_decode_next(s->dec, &s->out, &kind, &partial);
    s->f.kind = kind; s->f.partial = partial;
    s->f.display = nvdrv_decode_display(s->dec);
    return &s->f;
}

__attribute__((export_name("nvdr_wasm_seq_close")))
void nvdr_wasm_seq_close(Seq* s) {
    if (!s) return;
    nvdrv_decode_close(s->dec);
    free(s->out.pixels);
    free(s);
}

/*
 * One picture predicted from another of its size (nvdrv_predict_decode()),
 * an album's photo:
 *
 *   r = nvdr_wasm_alloc(w * h * 3), d = nvdr_wasm_alloc(n), both by JS
 *   o = nvdr_wasm_predict(r, w, h, d, n)   the picture, w * h * 3 bytes
 *                                          to give back with
 *                                          nvdr_wasm_free(); 0 when it
 *                                          does not decode
 */
__attribute__((export_name("nvdr_wasm_predict")))
uint8_t* nvdr_wasm_predict(uint8_t* ref, int32_t w, int32_t h, const uint8_t* data, uint32_t n) {
    NvdrImage r = { ref, w, h }, out;
    int partial;
    if (nvdrv_predict_decode(&r, data, n, &out, &partial) != 0) return 0;
    return out.pixels;
}
