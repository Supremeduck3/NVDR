/*
 * NVDA albums. See nvda.h.
 */
#include "nvda.h"

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

int nvda_write(const char* path, int count, const char* const* names, const NvdrImage* imgs,
               const NvdrConfig* cfg, int fluid, size_t* bytes_out, size_t* cold_out) {
    if (count < 0 || count > NVDA_MAX_IMAGES) return -1;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    NvdrContext* ctx = fluid ? nvdr_context_new() : NULL;
    int rc = (fluid && !ctx) ? -1 : 0;
    uint8_t h[NVDA_HEADER_SIZE];
    memset(h, 0, sizeof(h));
    memcpy(h, NVDA_MAGIC, 4);
    h[4] = NVDA_VERSION;
    h[5] = fluid ? NVDA_FLAG_FLUID : 0;
    put_u32a(h + 6, (uint32_t)count);
    if (rc == 0 && fwrite(h, 1, sizeof(h), f) != sizeof(h)) rc = -1;
    for (int i = 0; i < count && rc == 0; i++) {
        uint8_t* blob; size_t len;
        if (nvdr_encode_mem_ctx(&blob, &len, &imgs[i], cfg, NULL, ctx) != 0) { rc = -1; break; }
        if (cold_out) {
            if (fluid) {
                uint8_t* cold; size_t clen;
                if (nvdr_encode_mem(&cold, &clen, &imgs[i], cfg, NULL) != 0) { free(blob); rc = -1; break; }
                free(cold);
                cold_out[i] = clen;
            } else cold_out[i] = len;
        }
        if (bytes_out) bytes_out[i] = len;
        size_t nl = names && names[i] ? strlen(names[i]) : 0;
        if (nl > 255) nl = 255;
        uint8_t n[2], l[4];
        put_u16a(n, (uint32_t)nl);
        put_u32a(l, (uint32_t)len);
        if (fwrite(n, 1, 2, f) != 2 || (nl && fwrite(names[i], 1, nl, f) != nl) ||
            fwrite(l, 1, 4, f) != 4 || fwrite(blob, 1, len, f) != len) rc = -1;
        free(blob);
    }
    nvdr_context_free(ctx);
    if (fclose(f) != 0) rc = -1;
    return rc;
}

int nvda_open(NvdaReader* r, const uint8_t* data, size_t size) {
    memset(r, 0, sizeof(*r));
    if (size < NVDA_HEADER_SIZE || memcmp(data, NVDA_MAGIC, 4) != 0 || data[4] != NVDA_VERSION)
        return -1;
    if (data[5] & ~NVDA_FLAG_FLUID) return -1;
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
}

int nvda_next(NvdaReader* r, char* name, size_t cap, NvdrImage* out, NvdrHeader* hdr, int* partial) {
    out->pixels = NULL;
    if (partial) *partial = 0;
    if (r->index >= r->count || r->pos + 2 > r->size) return 0;
    size_t nl = get_u16a(r->data + r->pos);
    if (nl > 255) return -1;
    if (r->pos + 2 + nl + 4 > r->size) return 0;
    if (name && cap) {
        size_t k = nl < cap - 1 ? nl : cap - 1;
        memcpy(name, r->data + r->pos + 2, k);
        name[k] = 0;
    }
    size_t len = get_u32a(r->data + r->pos + 2 + nl);
    size_t at = r->pos + 2 + nl + 4;
    size_t have = r->size - at;
    int cut = 0;
    if (len > have) { len = have; cut = 1; }
    /* A context the image before did not leave whole cannot decode this
     * one: after a cut there is nothing more to read anyway. */
    if (nvdr_decode_mem_ctx(r->data + at, len, -1, out, hdr, NULL, r->ctx) != 0)
        return cut ? 0 : -1;
    if (partial) *partial = cut;
    r->pos = at + len;
    r->index++;
    if (cut) r->index = r->count;
    return 1;
}
