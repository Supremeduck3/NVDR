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

NvdaOptions nvda_default_options(void) {
    NvdaOptions o;
    o.fluid = 0;
    o.predict = 1;
    o.window = NVDA_DEFAULT_WINDOW;
    /* Trying the two most similar earlier photos in full found the best
     * reference on every burst measured; the full trial is the expensive
     * part, the ranking is cheap. */
    o.candidates = 2;
    /* Photos whose thumbnails differ by more than this (mean grey levels)
     * are different pictures, and a trial would only cost time: a large
     * photo's trial is three times its encode. Predictions measured to win
     * came up to 7.6 (a pan, 44 times smaller); pictures of nothing in
     * common from 12.7. */
    o.max_distance = 10.0;
    return o;
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

/* ------------------------------------------------------------ similarity */

/*
 * Ranking the earlier photos is cheap: a grey thumbnail at 1/8 scale, and
 * the smallest mean difference over shifts of up to 2 thumbnail pixels
 * (16 image pixels), so a pan does not look like a different scene.
 */
#define THUMB 8

typedef struct { int w, h; float* p; } Thumb;

static void thumb_make(Thumb* t, const NvdrImage* img) {
    t->w = img->width / THUMB; t->h = img->height / THUMB;
    if (t->w < 1) t->w = 1;
    if (t->h < 1) t->h = 1;
    t->p = (float*)calloc((size_t)t->w * t->h, sizeof(float));
    if (!t->p) return;
    for (int y = 0; y < t->h; y++)
        for (int x = 0; x < t->w; x++) {
            double s = 0; int n = 0;
            for (int j = y * THUMB; j < (y + 1) * THUMB && j < img->height; j++)
                for (int i = x * THUMB; i < (x + 1) * THUMB && i < img->width; i++) {
                    const unsigned char* q = img->pixels + ((size_t)j * img->width + i) * 3;
                    s += 0.299 * q[0] + 0.587 * q[1] + 0.114 * q[2]; n++;
                }
            t->p[(size_t)y * t->w + x] = n ? (float)(s / n) : 0.0f;
        }
}

static double thumb_distance(const Thumb* a, const Thumb* b) {
    if (!a->p || !b->p || a->w != b->w || a->h != b->h) return 1e30;
    double best = 1e30;
    for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++) {
            double s = 0; long n = 0;
            for (int y = 2; y < a->h - 2; y++)
                for (int x = 2; x < a->w - 2; x++) {
                    double d = a->p[(size_t)y * a->w + x] - b->p[(size_t)(y + dy) * b->w + x + dx];
                    s += d < 0 ? -d : d; n++;
                }
            if (n && s / n < best) best = s / n;
        }
    return best;
}

/* ---------------------------------------------------------------- writer */

/*
 * The best prediction of `img` from `ref` at the quality `target` (squared
 * error) the photo alone reached: the step is made finer until the error
 * is within 2.3% (0.1 dB), and the attempt ends once it would cost more
 * than `limit` bytes, since it only grows from there. The motion search
 * does not depend on the step and is done once.
 */
static int best_prediction(const NvdrImage* ref, const NvdrImage* img, const NvdrConfig* cfg,
                           double target, size_t limit, uint8_t** out, size_t* out_len,
                           NvdrImage* shown) {
    static const float scales[] = { 1.0f, 0.85f, 0.72f, 0.6f, 0.5f, 0.42f };
    NvdrvConfig vcfg = nvdrv_default_config();
    vcfg.frame = *cfg;
    NvdrvMotion m;
    if (nvdrv_motion_find(ref, img, &vcfg, &m) != 0) return 0;
    int found = 0;
    for (int t = 0; t < 6 && !found; t++) {
        int q = (int)(cfg->q * scales[t] + 0.5f);
        if (q < 1) q = 1;
        uint8_t* pb; size_t plen; NvdrImage precon;
        if (nvdrv_motion_encode(&m, ref, q, limit, &pb, &plen, &precon) != 0) break;
        if (sse(img, &precon) <= target * 1.0233) {
            *out = pb; *out_len = plen; *shown = precon;
            found = 1;
        } else {
            free(pb); nvdr_image_free(&precon);
        }
    }
    nvdrv_motion_free(&m);
    return found;
}

int nvda_write(const char* path, int count, const char* const* names, const NvdrImage* imgs,
               const NvdrConfig* cfg, const NvdaOptions* opt_in, NvdaReport* report) {
    NvdaOptions opt = opt_in ? *opt_in : nvda_default_options();
    if (opt.window < 1) opt.window = 1;
    if (opt.window > NVDA_MAX_WINDOW) opt.window = NVDA_MAX_WINDOW;
    if (opt.candidates < 1) opt.candidates = 1;
    if (!(opt.max_distance >= 0)) opt.max_distance = 0;
    if (count < 0 || count > NVDA_MAX_IMAGES) return -1;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    int rc = 0;
    NvdrContext* ctx = opt.fluid ? nvdr_context_new() : NULL;
    NvdrContext* trial = opt.fluid ? nvdr_context_new() : NULL;
    NvdrImage* shown_all = (NvdrImage*)calloc((size_t)count + 1, sizeof(NvdrImage));
    Thumb* thumbs = (Thumb*)calloc((size_t)count + 1, sizeof(Thumb));
    uint8_t* index = (uint8_t*)calloc((size_t)count + 1, NVDA_ENTRY_SIZE);
    if ((opt.fluid && (!ctx || !trial)) || !shown_all || !thumbs || !index) rc = -1;

    uint8_t h[NVDA_HEADER_SIZE];
    memset(h, 0, sizeof(h));
    memcpy(h, NVDA_MAGIC, 4);
    h[4] = NVDA_VERSION;
    h[5] = (uint8_t)((opt.fluid ? NVDA_FLAG_FLUID : 0) | (opt.predict ? NVDA_FLAG_PREDICT : 0));
    put_u32a(h + 6, (uint32_t)count);
    h[10] = (uint8_t)opt.window;
    /* The index is written once every offset is known. */
    if (rc == 0 && (fwrite(h, 1, sizeof(h), f) != sizeof(h) ||
                    fwrite(index, NVDA_ENTRY_SIZE, (size_t)count, f) != (size_t)count)) rc = -1;
    size_t pos = NVDA_HEADER_SIZE + (size_t)count * NVDA_ENTRY_SIZE;

    for (int i = 0; i < count && rc == 0; i++) {
        const NvdrImage* img = &imgs[i];
        if (ctx) nvdr_context_copy(trial, ctx);
        uint8_t* blob; size_t len;
        if (nvdr_encode_mem_ctx(&blob, &len, img, cfg, NULL, trial) != 0) { rc = -1; break; }
        size_t alone = len;
        NvdrImage shown;
        {
            NvdrContext* check = ctx ? nvdr_context_new() : NULL;
            if (check) nvdr_context_copy(check, ctx);
            int d = nvdr_decode_mem_ctx(blob, len, -1, &shown, NULL, NULL, check);
            nvdr_context_free(check);
            if (d != 0) { free(blob); rc = -1; break; }
        }
        int kind = NVDA_KIND_INTRA, ref = 0;
        if (opt.predict) {
            thumb_make(&thumbs[i], img);
            /* The earlier photos in the window, the most similar first. */
            int cand[NVDA_MAX_WINDOW]; double dist[NVDA_MAX_WINDOW]; int nc = 0;
            for (int d = 1; d <= opt.window && i - d >= 0; d++) {
                const NvdrImage* r = &shown_all[i - d];
                if (!r->pixels || r->width != img->width || r->height != img->height) continue;
                double s = thumb_distance(&thumbs[i], &thumbs[i - d]);
                if (s > opt.max_distance) continue;
                int k = nc++;
                while (k > 0 && dist[k - 1] > s) { cand[k] = cand[k - 1]; dist[k] = dist[k - 1]; k--; }
                cand[k] = d; dist[k] = s;
            }
            double di = sse(img, &shown);
            for (int k = 0; k < nc && k < opt.candidates; k++) {
                uint8_t* pb; size_t plen; NvdrImage precon;
                int ok = best_prediction(&shown_all[i - cand[k]], img, cfg, di, len, &pb, &plen, &precon);
                if (ok) {
                    free(blob); nvdr_image_free(&shown);
                    blob = pb; len = plen; shown = precon;
                    kind = NVDA_KIND_PRED; ref = cand[k];
                }
            }
        }
        if (kind == NVDA_KIND_INTRA && ctx) nvdr_context_copy(ctx, trial);

        size_t nl = names && names[i] ? strlen(names[i]) : 0;
        if (nl > 255) nl = 255;
        if ((nl && fwrite(names[i], 1, nl, f) != nl) || fwrite(blob, 1, len, f) != len) rc = -1;
        uint8_t* e = index + (size_t)i * NVDA_ENTRY_SIZE;
        put_u32a(e, (uint32_t)(pos + nl));
        put_u32a(e + 4, (uint32_t)len);
        e[8] = (uint8_t)kind;
        e[9] = (uint8_t)ref;
        put_u16a(e + 10, (uint32_t)img->width);
        put_u16a(e + 12, (uint32_t)img->height);
        put_u16a(e + 14, (uint32_t)nl);
        pos += nl + len;
        free(blob);
        if (report) { report[i].bytes = len; report[i].alone = alone; report[i].kind = kind; report[i].ref = ref; }

        shown_all[i] = shown;
        /* Only the window is ever referenced again. */
        if (i - opt.window >= 0) { nvdr_image_free(&shown_all[i - opt.window]); free(thumbs[i - opt.window].p); thumbs[i - opt.window].p = NULL; }
    }
    if (rc == 0 && (fseek(f, NVDA_HEADER_SIZE, SEEK_SET) != 0 ||
                    fwrite(index, NVDA_ENTRY_SIZE, (size_t)count, f) != (size_t)count)) rc = -1;
    if (fclose(f) != 0) rc = -1;
    for (int i = 0; i < count; i++) { nvdr_image_free(&shown_all[i]); free(thumbs[i].p); }
    free(shown_all); free(thumbs); free(index);
    nvdr_context_free(ctx); nvdr_context_free(trial);
    return rc;
}

/* ---------------------------------------------------------------- reader */

int nvda_open(NvdaReader* r, const uint8_t* data, size_t size) {
    memset(r, 0, sizeof(*r));
    if (size < NVDA_HEADER_SIZE || memcmp(data, NVDA_MAGIC, 4) != 0 || data[4] != NVDA_VERSION)
        return -1;
    if (data[5] & ~(NVDA_FLAG_FLUID | NVDA_FLAG_PREDICT)) return -1;
    r->data = data; r->size = size;
    r->flags = data[5];
    r->count = get_u32a(data + 6);
    r->window = data[10];
    if (r->count > NVDA_MAX_IMAGES || r->window < 1 || r->window > NVDA_MAX_WINDOW) return -1;
    size_t table = NVDA_HEADER_SIZE + (size_t)r->count * NVDA_ENTRY_SIZE;
    if (size < table) return -1;   /* the whole index has to be there */
    r->index = (NvdaEntry*)calloc(r->count ? r->count : 1, sizeof(NvdaEntry));
    r->ring = (NvdrImage*)calloc((size_t)r->window, sizeof(NvdrImage));
    r->ring_of = (int*)malloc(sizeof(int) * (size_t)r->window);
    if (!r->index || !r->ring || !r->ring_of) { nvda_close(r); return -1; }
    for (int k = 0; k < r->window; k++) r->ring_of[k] = -1;
    for (uint32_t i = 0; i < r->count; i++) {
        const uint8_t* e = data + NVDA_HEADER_SIZE + (size_t)i * NVDA_ENTRY_SIZE;
        NvdaEntry* x = &r->index[i];
        x->offset = get_u32a(e); x->length = get_u32a(e + 4);
        x->kind = e[8]; x->ref = e[9];
        x->width = (int)get_u16a(e + 10); x->height = (int)get_u16a(e + 12);
        x->name_len = (int)get_u16a(e + 14);
        int ok = x->name_len <= 255 && x->offset >= table + (uint32_t)x->name_len &&
                 x->width > 0 && x->height > 0 &&
                 (size_t)x->width * x->height <= NVDR_MAX_PIXELS;
        if (x->kind == NVDA_KIND_INTRA) ok &= x->ref == 0;
        else if (x->kind == NVDA_KIND_PRED) {
            ok &= x->ref >= 1 && x->ref <= r->window && (uint32_t)x->ref <= i;
            if (ok) {
                const NvdaEntry* y = &r->index[i - x->ref];
                ok &= y->width == x->width && y->height == x->height;
            }
        } else ok = 0;
        if (!ok) { nvda_close(r); return -1; }
    }
    if ((r->flags & NVDA_FLAG_FLUID) && !(r->ctx = nvdr_context_new())) { nvda_close(r); return -1; }
    return 0;
}

void nvda_close(NvdaReader* r) {
    if (r->ring) for (int k = 0; k < r->window; k++) nvdr_image_free(&r->ring[k]);
    free(r->ring); free(r->ring_of); free(r->index);
    nvdr_context_free(r->ctx);
    r->ring = NULL; r->ring_of = NULL; r->index = NULL; r->ctx = NULL;
}

void nvda_name(const NvdaReader* r, int i, char* name, size_t cap) {
    if (!cap) return;
    name[0] = 0;
    if (i < 0 || (uint32_t)i >= r->count) return;
    const NvdaEntry* x = &r->index[i];
    size_t at = x->offset - (size_t)x->name_len;
    if (at + (size_t)x->name_len > r->size) return;
    size_t k = (size_t)x->name_len < cap - 1 ? (size_t)x->name_len : cap - 1;
    memcpy(name, r->data + at, k);
    name[k] = 0;
}

static const NvdrImage* ring_get(const NvdaReader* r, int i) {
    int s = i % r->window;
    return r->ring_of[s] == i ? &r->ring[s] : NULL;
}

int nvda_decode(NvdaReader* r, int i, NvdrImage* out, int* partial) {
    out->pixels = NULL;
    if (partial) *partial = 0;
    if (i < 0 || (uint32_t)i >= r->count) return -1;
    const NvdrImage* held = ring_get(r, i);
    if (held) return copy_image(out, held) == 0 ? 1 : -1;

    const NvdaEntry* x = &r->index[i];
    if (r->ctx) {
        /* Each photo's models come from the one before: in order only. */
        if (i != r->next_fluid) return -1;
    }
    if (x->offset >= r->size) return 0;
    size_t len = x->length, have = r->size - x->offset;
    int cut = 0;
    if (len > have) { len = have; cut = 1; }

    if (x->kind == NVDA_KIND_INTRA) {
        NvdrHeader hh;
        if (nvdr_decode_mem_ctx(r->data + x->offset, len, -1, out, &hh, NULL,
                                x->kind == NVDA_KIND_INTRA ? r->ctx : NULL) != 0)
            return cut ? 0 : -1;
        if (hh.width != x->width || hh.height != x->height) { nvdr_image_free(out); return -1; }
    } else {
        int j = i - x->ref;
        NvdrImage ref;
        int p = 0, rc = nvda_decode(r, j, &ref, &p);
        if (rc != 1) return rc;
        if (p) { nvdr_image_free(&ref); return 0; }   /* a cut reference predicts nothing exact */
        rc = nvdrv_predict_decode(&ref, r->data + x->offset, len, out, NULL);
        nvdr_image_free(&ref);
        if (rc != 0) return cut ? 0 : -1;
    }
    if (partial) *partial = cut;
    if (r->ctx) r->next_fluid = i + 1;
    if (!cut) {
        int s = i % r->window;
        if (copy_image(&r->ring[s], out) != 0) { nvdr_image_free(out); return -1; }
        r->ring_of[s] = i;
    }
    return 1;
}
