#include "nvdrv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- bytes */

static void put_u16v(uint8_t* p, uint16_t v) { p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); }
static void put_u32v(uint8_t* p, uint32_t v) {
    p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); p[2]=(uint8_t)(v>>16); p[3]=(uint8_t)(v>>24);
}
static uint16_t get_u16v(const uint8_t* p) { return (uint16_t)(p[0] | (p[1]<<8)); }
static uint32_t get_u32v(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}
static int clamp255v(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

NvdrvConfig nvdrv_default_config(void) {
    NvdrvConfig c;
    c.frame = nvdr_default_config();
    /* Two seconds at 24fps. Short enough to join a stream quickly, long
     * enough that the intra frames — four times the size of a predicted
     * one — do not dominate the bitrate. */
    c.gop = 48;
    /* Wide enough for a brisk handheld pan at this resolution. The search
     * is global, so its cost is one pass per candidate over a lattice, not
     * per block. */
    c.search = 12;
    c.intra_threshold = 24.0f;
    c.fps = 24;
    return c;
}

/* ------------------------------------------------------- motion search */

/*
 * One translation for the whole frame, minimising mean absolute
 * difference. Coarse pass on a stride-4 lattice over the full range, then
 * a fine pass over the neighbourhood of the winner — a full search at
 * every offset costs more than the vector is worth at this stage.
 */
static double shift_cost(const NvdrImage* cur, const NvdrImage* ref,
                         int dx, int dy, int step, double give_up) {
    double acc = 0.0; long n = 0;
    int m = 8;
    for (int y = m; y < cur->height - m; y += step) {
        const unsigned char* a = cur->pixels + ((size_t)y * cur->width) * 3;
        int sy = y + dy;
        if (sy < 0) sy = 0;
        if (sy >= ref->height) sy = ref->height - 1;
        const unsigned char* b = ref->pixels + ((size_t)sy * ref->width) * 3;
        for (int x = m; x < cur->width - m; x += step) {
            int sx = x + dx;
            if (sx < 0) sx = 0;
            if (sx >= ref->width) sx = ref->width - 1;
            const unsigned char* pa = a + (size_t)x * 3;
            const unsigned char* pb = b + (size_t)sx * 3;
            for (int c = 0; c < 3; c++) { int d = pa[c]-pb[c]; acc += d < 0 ? -d : d; }
            n++;
        }
        if (n > 4096 && acc / n > give_up) return acc / n;   /* already lost */
    }
    return n ? acc / n : 1e30;
}

static void find_shift(const NvdrImage* cur, const NvdrImage* ref, int range,
                       int* bdx, int* bdy) {
    *bdx = 0; *bdy = 0;
    if (range <= 0) return;
    double best = shift_cost(cur, ref, 0, 0, 4, 1e30);
    for (int dy = -range; dy <= range; dy += 2)
        for (int dx = -range; dx <= range; dx += 2) {
            double c = shift_cost(cur, ref, dx, dy, 4, best);
            if (c < best) { best = c; *bdx = dx; *bdy = dy; }
        }
    int cx = *bdx, cy = *bdy;
    best = shift_cost(cur, ref, cx, cy, 2, 1e30);
    for (int dy = cy - 2; dy <= cy + 2; dy++)
        for (int dx = cx - 2; dx <= cx + 2; dx++) {
            double c = shift_cost(cur, ref, dx, dy, 2, best);
            if (c < best) { best = c; *bdx = dx; *bdy = dy; }
        }
}

/* The reference translated by (dx, dy), edges held rather than wrapped. */
static void shift_into(const NvdrImage* src, NvdrImage* dst, int dx, int dy) {
    for (int y = 0; y < dst->height; y++) {
        int sy = y + dy;
        if (sy < 0) sy = 0;
        if (sy >= src->height) sy = src->height - 1;
        for (int x = 0; x < dst->width; x++) {
            int sx = x + dx;
            if (sx < 0) sx = 0;
            if (sx >= src->width) sx = src->width - 1;
            memcpy(dst->pixels + ((size_t)y*dst->width + x)*3,
                   src->pixels + ((size_t)sy*src->width + sx)*3, 3);
        }
    }
}

/* ------------------------------------------------------------ encoder */

struct NvdrvEncoder {
    FILE*       f;
    NvdrvConfig cfg;
    int         width, height;
    uint32_t    count;
    /* What the decoder holds after the last frame. The encoder predicts
     * from this and never from the source, because this is all the decoder
     * will have; predicting from the source is how a codec drifts. */
    NvdrImage   state;
    NvdrImage   scratch;   /* the shifted reference */
    NvdrImage   error;     /* the biased prediction error */
};

static int alloc_image(NvdrImage* img, int w, int h) {
    img->width = w; img->height = h;
    img->pixels = (unsigned char*)calloc((size_t)w * h * 3, 1);
    return img->pixels ? 0 : -1;
}

/* Decode a container and render it flat, which is what the next frame
 * predicts from. Smoothing is a display choice and must not enter the
 * loop, or the two sides would have to agree about it too. */
static int reconstruct(const uint8_t* blob, size_t len, NvdrImage* out) {
    NvdrPyramid pyr;
    NvdrHeader hdr;
    if (nvdr_decode_mem(blob, len, &pyr, &hdr) != 0) return -1;
    /* A frame's container has to be the size of the sequence it sits in.
     * Rendering clips, so a mismatch is not unsafe, but it is not a frame
     * of this sequence either. */
    if (hdr.width != out->width || hdr.height != out->height) {
        nvdr_pyramid_free(&pyr);
        return -1;
    }
    memset(out->pixels, 0, (size_t)out->width * out->height * 3);
    for (int k = 0; k < pyr.levels_present; k++) nvdr_render_level(&pyr.level[k], out);
    nvdr_pyramid_free(&pyr);
    return 0;
}

int nvdrv_encode_open(NvdrvEncoder** out, const char* path,
                      int width, int height, const NvdrvConfig* cfg) {
    *out = NULL;
    NvdrvEncoder* e = (NvdrvEncoder*)calloc(1, sizeof(*e));
    if (!e) return -1;
    e->cfg = cfg ? *cfg : nvdrv_default_config();
    e->width = width; e->height = height;

    if (alloc_image(&e->state, width, height) != 0 ||
        alloc_image(&e->scratch, width, height) != 0 ||
        alloc_image(&e->error, width, height) != 0) {
        nvdrv_encode_close(e); return -1;
    }

    e->f = fopen(path, "wb");
    if (!e->f) { nvdrv_encode_close(e); return -1; }

    uint8_t h[NVDRV_HEADER_SIZE];
    memset(h, 0, sizeof(h));
    memcpy(h, NVDRV_MAGIC, 4);
    h[4] = NVDRV_VERSION;
    put_u16v(h + 6, (uint16_t)width);
    put_u16v(h + 8, (uint16_t)height);
    /* Frame count is patched on close; a stream that never closes still
     * decodes, since frames are self-delimiting. */
    put_u32v(h + 10, 0);
    h[14] = (uint8_t)(e->cfg.fps > 0 && e->cfg.fps < 256 ? e->cfg.fps : 24);
    h[15] = (uint8_t)(e->cfg.gop > 255 ? 255 : e->cfg.gop);
    if (fwrite(h, 1, sizeof(h), e->f) != sizeof(h)) { nvdrv_encode_close(e); return -1; }

    *out = e;
    return 0;
}

static int write_frame(NvdrvEncoder* e, int kind, int dx, int dy,
                       const uint8_t* payload, size_t len) {
    uint8_t h[NVDRV_FRAME_HEADER];
    memset(h, 0, sizeof(h));
    h[0] = (uint8_t)kind;
    put_u16v(h + 2, (uint16_t)(int16_t)dx);
    put_u16v(h + 4, (uint16_t)(int16_t)dy);
    put_u32v(h + 6, (uint32_t)len);
    if (fwrite(h, 1, sizeof(h), e->f) != sizeof(h)) return -1;
    if (fwrite(payload, 1, len, e->f) != len) return -1;
    return 0;
}

int nvdrv_encode_frame(NvdrvEncoder* e, const NvdrImage* frame,
                       int* kind_out, size_t* bytes_out,
                       int* dx_out, int* dy_out) {
    if (frame->width != e->width || frame->height != e->height) return -1;
    size_t npx = (size_t)e->width * e->height * 3;

    int forced_intra = (e->count == 0) ||
                       (e->cfg.gop > 0 && (int)(e->count % (uint32_t)e->cfg.gop) == 0);

    int dx = 0, dy = 0;
    const NvdrImage* ref = NULL;
    double mad = 1e30;

    if (!forced_intra) {
        find_shift(frame, &e->state, e->cfg.search, &dx, &dy);
        if (dx || dy) { shift_into(&e->state, &e->scratch, dx, dy); ref = &e->scratch; }
        else ref = &e->state;

        double acc = 0.0;
        for (size_t i = 0; i < npx; i++) {
            int d = (int)frame->pixels[i] - (int)ref->pixels[i];
            acc += d < 0 ? -d : d;
        }
        mad = acc / (double)npx;
    }

    /* A cut, or anything else the bias to 128 would clip, goes intra. */
    int kind = (forced_intra || mad > e->cfg.intra_threshold)
               ? NVDRV_INTRA : NVDRV_PRED;

    const NvdrImage* to_code = frame;
    if (kind == NVDRV_PRED) {
        for (size_t i = 0; i < npx; i++)
            e->error.pixels[i] =
                (unsigned char)clamp255v((int)frame->pixels[i] - (int)ref->pixels[i] + 128);
        to_code = &e->error;
    } else {
        dx = dy = 0;
    }

    uint8_t* blob = NULL;
    size_t len = 0;
    NvdrHeader fh;
    if (nvdr_encode_mem(&blob, &len, to_code, &e->cfg.frame, &fh) != 0) return -1;

    if (write_frame(e, kind, dx, dy, blob, len) != 0) { free(blob); return -1; }

    /* Carry the decoder's state forward by decoding what was just written,
     * so the two sides hold the same bytes from here on. */
    int rc;
    if (kind == NVDRV_INTRA) {
        rc = reconstruct(blob, len, &e->state);
    } else {
        rc = reconstruct(blob, len, &e->error);
        if (rc == 0) {
            /* `ref` may alias e->state, so the sum is written into scratch
             * and swapped in rather than updated in place. */
            for (size_t i = 0; i < npx; i++)
                e->scratch.pixels[i] = (unsigned char)clamp255v(
                    (int)e->error.pixels[i] - 128 + (int)ref->pixels[i]);
            unsigned char* tmp = e->state.pixels;
            e->state.pixels = e->scratch.pixels;
            e->scratch.pixels = tmp;
        }
    }
    free(blob);
    if (rc != 0) return -1;

    e->count++;
    if (kind_out) *kind_out = kind;
    if (bytes_out) *bytes_out = len + NVDRV_FRAME_HEADER;
    if (dx_out) *dx_out = dx;
    if (dy_out) *dy_out = dy;
    return 0;
}

int nvdrv_encode_close(NvdrvEncoder* e) {
    int rc = 0;
    if (!e) return 0;
    if (e->f) {
        if (fseek(e->f, 10, SEEK_SET) == 0) {
            uint8_t n[4];
            put_u32v(n, e->count);
            if (fwrite(n, 1, 4, e->f) != 4) rc = -1;
        }
        if (fclose(e->f) != 0) rc = -1;
    }
    free(e->state.pixels);
    free(e->scratch.pixels);
    free(e->error.pixels);
    free(e);
    return rc;
}

/* ------------------------------------------------------------ decoder */

struct NvdrvDecoder {
    uint8_t*  data;
    size_t    size, pos;
    int       width, height;
    NvdrImage state;
    NvdrImage scratch;
};

int nvdrv_decode_open(NvdrvDecoder** out, const char* path, NvdrvInfo* info) {
    *out = NULL;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size < NVDRV_HEADER_SIZE) { fclose(f); return -1; }

    uint8_t* data = (uint8_t*)malloc((size_t)size);
    if (!data) { fclose(f); return -1; }
    size_t got = fread(data, 1, (size_t)size, f);
    fclose(f);
    if (got < NVDRV_HEADER_SIZE ||
        memcmp(data, NVDRV_MAGIC, 4) != 0 || data[4] != NVDRV_VERSION) {
        free(data); return -1;
    }

    NvdrvDecoder* d = (NvdrvDecoder*)calloc(1, sizeof(*d));
    if (!d) { free(data); return -1; }
    d->data = data; d->size = got; d->pos = NVDRV_HEADER_SIZE;
    d->width = get_u16v(data + 6);
    d->height = get_u16v(data + 8);
    /* Every frame buffer is allocated at this size before a single frame
     * is read, so a damaged header gets refused here rather than trusted. */
    if (d->width <= 0 || d->height <= 0 ||
        (size_t)d->width * d->height > NVDR_MAX_PIXELS ||
        alloc_image(&d->state, d->width, d->height) != 0 ||
        alloc_image(&d->scratch, d->width, d->height) != 0) {
        nvdrv_decode_close(d); return -1;
    }
    if (info) {
        info->width = d->width;
        info->height = d->height;
        info->frame_count = (int)get_u32v(data + 10);
        info->fps = data[14];
        info->gop = data[15];
    }
    *out = d;
    return 0;
}

int nvdrv_decode_next(NvdrvDecoder* d, NvdrImage* out,
                      int* kind_out, int* partial_out) {
    if (partial_out) *partial_out = 0;
    if (d->pos + NVDRV_FRAME_HEADER > d->size) return 0;

    const uint8_t* h = d->data + d->pos;
    int kind = h[0];
    int dx = (int16_t)get_u16v(h + 2);
    int dy = (int16_t)get_u16v(h + 4);
    size_t len = get_u32v(h + 6);
    if (kind != NVDRV_INTRA && kind != NVDRV_PRED) return -1;
    d->pos += NVDRV_FRAME_HEADER;

    /* A cut file ends mid-frame. The still decoder reads as far as the
     * bytes reach, so the last frame is shown at whatever quality arrived
     * instead of being dropped. */
    size_t have = d->size - d->pos;
    int partial = 0;
    if (len > have) { len = have; partial = 1; }
    if (len == 0) return 0;

    size_t npx = (size_t)d->width * d->height * 3;

    if (kind == NVDRV_INTRA) {
        if (reconstruct(d->data + d->pos, len, &d->state) != 0) return -1;
    } else {
        const NvdrImage* ref = &d->state;
        if (dx || dy) { shift_into(&d->state, &d->scratch, dx, dy); ref = &d->scratch; }
        NvdrImage err;
        err.width = d->width; err.height = d->height;
        err.pixels = (unsigned char*)malloc(npx);
        if (!err.pixels) return -1;
        if (reconstruct(d->data + d->pos, len, &err) != 0) { free(err.pixels); return -1; }
        for (size_t i = 0; i < npx; i++)
            err.pixels[i] = (unsigned char)clamp255v(
                (int)err.pixels[i] - 128 + (int)ref->pixels[i]);
        memcpy(d->state.pixels, err.pixels, npx);
        free(err.pixels);
    }

    d->pos += len;
    memcpy(out->pixels, d->state.pixels, npx);
    if (kind_out) *kind_out = kind;
    if (partial_out) *partial_out = partial;
    return 1;
}

void nvdrv_decode_close(NvdrvDecoder* d) {
    if (!d) return;
    free(d->data);
    free(d->state.pixels);
    free(d->scratch.pixels);
    free(d);
}
