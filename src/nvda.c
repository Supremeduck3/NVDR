/*
 * NVDA albums. See nvda.h.
 */
#include "nvda.h"
#include "nvdrv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void put_u16a(uint8_t* p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void put_u32a(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static uint32_t get_u16a(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8); }
static uint32_t get_u32a(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static double sse(const NvdrImage* a, const NvdrImage* b) {
    double e = 0;
    size_t n = (size_t)a->width * a->height * 3;
    for (size_t i = 0; i < n; i++) { double d = (double)a->pixels[i] - b->pixels[i]; e += d * d; }
    return e;
}

static int copy_image(NvdrImage* dst, const NvdrImage* src) {
    size_t n = (size_t)src->width * src->height * 3;
    unsigned char* p = (unsigned char*)realloc(dst->pixels, n);
    if (!p) return -1;
    memcpy(p, src->pixels, n);
    dst->pixels = p; dst->width = src->width; dst->height = src->height;
    return 0;
}

int nvda_write(const char* path, int count, const char* const* names, const NvdrImage* imgs,
               const NvdrConfig* cfg, int fluid, int predict,
               size_t* bytes_out, size_t* cold_out, int* kind_out) {
    if (count < 0 || count > NVDA_MAX_IMAGES) return -1;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    NvdrContext* ctx = fluid ? nvdr_context_new() : NULL;
    NvdrContext* trial = fluid ? nvdr_context_new() : NULL;
    NvdrContext* check = fluid ? nvdr_context_new() : NULL;
    NvdrImage prev = { NULL, 0, 0 };
    NvdrvConfig vcfg = nvdrv_default_config();
    vcfg.frame = *cfg;
    int rc = (fluid && (!ctx || !trial || !check)) ? -1 : 0;
    uint8_t h[NVDA_HEADER_SIZE];
    memset(h, 0, sizeof(h));
    memcpy(h, NVDA_MAGIC, 4);
    h[4] = NVDA_VERSION;
    h[5] = (uint8_t)((fluid ? NVDA_FLAG_FLUID : 0) | (predict ? NVDA_FLAG_PREDICT : 0));
    put_u32a(h + 6, (uint32_t)count);
    if (rc == 0 && fwrite(h, 1, sizeof(h), f) != sizeof(h)) rc = -1;
    for (int i = 0; i < count && rc == 0; i++) {
        const NvdrImage* img = &imgs[i];
        /* On its own, from the running context. */
        if (ctx) { nvdr_context_copy(trial, ctx); nvdr_context_copy(check, ctx); }
        uint8_t* blob; size_t len;
        if (nvdr_encode_mem_ctx(&blob, &len, img, cfg, NULL, trial) != 0) { rc = -1; break; }
        NvdrImage shown;
        if (nvdr_decode_mem_ctx(blob, len, -1, &shown, NULL, NULL, check) != 0) { free(blob); rc = -1; break; }
        int kind = NVDA_KIND_INTRA;

        /* From the image before, when there is one of the same size,
         * compared at equal quality: the prediction is coded with a finer
         * step until its squared error is within 0.1 dB (2.3%) of the
         * image coded alone, and wins if it is then still smaller. At the
         * same step it would win on bytes and lose on quality, which is
         * not a comparison. */
        if (predict && prev.pixels && prev.width == img->width && prev.height == img->height) {
            double di = sse(img, &shown);
            static const float scales[] = { 1.0f, 0.85f, 0.72f, 0.6f, 0.5f, 0.42f };
            for (int t = 0; t < 6; t++) {
                vcfg.pred_q = (int)(cfg->q * scales[t] + 0.5f);
                if (vcfg.pred_q < 1) vcfg.pred_q = 1;
                uint8_t* pb; size_t plen; NvdrImage precon;
                if (nvdrv_predict_encode(&prev, img, &vcfg, &pb, &plen, &precon) != 0) break;
                double dp = sse(img, &precon);
                if (plen >= len) { free(pb); nvdr_image_free(&precon); break; }   /* only grows from here */
                if (dp <= di * 1.0233) {
                    free(blob); nvdr_image_free(&shown);
                    blob = pb; len = plen; shown = precon;
                    kind = NVDA_KIND_PRED;
                    break;
                }
                free(pb); nvdr_image_free(&precon);
            }
        }
        if (kind == NVDA_KIND_INTRA && ctx) nvdr_context_copy(ctx, trial);

        if (cold_out) {
            uint8_t* cold; size_t clen;
            if (nvdr_encode_mem(&cold, &clen, img, cfg, NULL) != 0) { free(blob); rc = -1; break; }
            free(cold);
            cold_out[i] = clen;
        }
        if (bytes_out) bytes_out[i] = len;
        if (kind_out) kind_out[i] = kind;
        size_t nl = names && names[i] ? strlen(names[i]) : 0;
        if (nl > 255) nl = 255;
        uint8_t n[2], k = (uint8_t)kind, l[4];
        put_u16a(n, (uint32_t)nl);
        put_u32a(l, (uint32_t)len);
        if (fwrite(n, 1, 2, f) != 2 || (nl && fwrite(names[i], 1, nl, f) != nl) ||
            fwrite(&k, 1, 1, f) != 1 || fwrite(l, 1, 4, f) != 4 || fwrite(blob, 1, len, f) != len) rc = -1;
        free(blob);
        if (rc == 0 && copy_image(&prev, &shown) != 0) rc = -1;
        nvdr_image_free(&shown);
    }
    nvdr_context_free(ctx); nvdr_context_free(trial); nvdr_context_free(check);
    free(prev.pixels);
    if (fclose(f) != 0) rc = -1;
    return rc;
}

int nvda_open(NvdaReader* r, const uint8_t* data, size_t size) {
    memset(r, 0, sizeof(*r));
    if (size < NVDA_HEADER_SIZE || memcmp(data, NVDA_MAGIC, 4) != 0 || data[4] != NVDA_VERSION)
        return -1;
    if (data[5] & ~(NVDA_FLAG_FLUID | NVDA_FLAG_PREDICT)) return -1;
    r->data = data; r->size = size; r->pos = NVDA_HEADER_SIZE;
    r->flags = data[5];
    r->count = get_u32a(data + 6);
    if (r->count > NVDA_MAX_IMAGES) return -1;
    if ((r->flags & NVDA_FLAG_FLUID) && !(r->ctx = nvdr_context_new())) return -1;
    return 0;
}

void nvda_close(NvdaReader* r) {
    nvdr_context_free(r->ctx);
    r->ctx = NULL;
    free(r->prev.pixels);
    r->prev.pixels = NULL;
}

int nvda_next(NvdaReader* r, char* name, size_t cap, NvdrImage* out, int* kind_out, int* partial) {
    out->pixels = NULL;
    if (partial) *partial = 0;
    if (r->index >= r->count || r->pos + 2 > r->size) return 0;
    size_t nl = get_u16a(r->data + r->pos);
    if (nl > 255) return -1;
    if (r->pos + 2 + nl + 5 > r->size) return 0;
    if (name && cap) {
        size_t k = nl < cap - 1 ? nl : cap - 1;
        memcpy(name, r->data + r->pos + 2, k);
        name[k] = 0;
    }
    int kind = r->data[r->pos + 2 + nl];
    if (kind != NVDA_KIND_INTRA && kind != NVDA_KIND_PRED) return -1;
    size_t len = get_u32a(r->data + r->pos + 2 + nl + 1);
    size_t at = r->pos + 2 + nl + 5;
    size_t have = r->size - at;
    int cut = 0;
    if (len > have) { len = have; cut = 1; }
    int ok;
    if (kind == NVDA_KIND_INTRA) {
        ok = nvdr_decode_mem_ctx(r->data + at, len, -1, out, NULL, NULL, r->ctx) == 0;
    } else {
        /* A prediction needs the image before it. */
        if (!r->prev.pixels) return -1;
        ok = nvdrv_predict_decode(&r->prev, r->data + at, len, out, NULL) == 0;
    }
    if (!ok) return cut ? 0 : -1;
    if (copy_image(&r->prev, out) != 0) { nvdr_image_free(out); return -1; }
    if (kind_out) *kind_out = kind;
    if (partial) *partial = cut;
    r->pos = at + len;
    r->index++;
    if (cut) r->index = r->count;
    return 1;
}
