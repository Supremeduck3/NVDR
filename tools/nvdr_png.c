/*
 * nvdr_png — an NVDR file dressed as a PNG, and back.
 *
 *   nvdr_png <in.nvdr|in.nvda|in.nvdrv> <out.png> [--preview N]
 *   nvdr_png --extract <in.png> <out>
 *
 * The PNG is an ordinary one: any viewer, browser or phone shows its
 * picture, a preview decoded from the file (the image, an album's first
 * photo, a video's first frame) and shrunk to N pixels on its longer side
 * (0 keeps it whole). The NVDR file rides along untouched in a private
 * ancillary chunk, `nvDr`, which PNG readers skip:
 *
 *   n  lower case: ancillary, a reader may ignore it
 *   v  lower case: private, not a registered chunk
 *   D  upper case: as the PNG specification requires of the third letter
 *   r  lower case: safe to copy, an editor that keeps the picture may keep it
 *
 * The chunk comes straight after IHDR, so its data always starts at byte
 * 41 of the PNG (signature 8, IHDR 25, the chunk's length and type 8): a
 * viewer streaming the PNG can start decoding the NVDR inside it as soon
 * as it arrives, as nvdr-img.js does. What it holds names itself by its
 * own magic (NVDR, NVDA or NVDV). public/nvdr-png.js reads and writes the
 * same layout.
 *
 * Services that recompress what they are sent (most social networks and
 * chat apps) drop ancillary chunks, and what arrives is the preview alone.
 * E-mail, file hosting, and anything that passes the file as a file keep
 * it whole.
 */
#include "nvdr.h"
#include "nvda.h"
#include "nvdrv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#define PNG_PREVIEW 320   /* default longer side of the preview, in pixels */
#define PNG_COLOURS 256   /* default palette size; 0 for full colour */

/* A tEXt chunk saying what the file is, for whoever looks inside. */
static const char note[] = "Comment\0This PNG carries an NVDR file in its nvDr chunk; "
                           "an NVDR viewer shows it, nvdr_png --extract takes it out.";

static const uint8_t PNG_SIG[8] = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };

static uint8_t* read_all(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    size_t cap = 1 << 16, n = 0;
    uint8_t* buf = malloc(cap);
    for (size_t got; buf && (got = fread(buf + n, 1, cap - n, f)) > 0;) {
        n += got;
        if (n == cap) {
            uint8_t* grown = realloc(buf, cap *= 2);
            if (!grown) free(buf);
            buf = grown;
        }
    }
    fclose(f);
    *len = n;
    return buf;
}

static void put32(uint8_t* p, uint32_t v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }
static uint32_t get32(const uint8_t* p) { return (uint32_t)p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3]; }

static int put_chunk(FILE* f, const char* type, const uint8_t* data, size_t len) {
    uint8_t head[8];
    put32(head, (uint32_t)len);
    memcpy(head + 4, type, 4);
    uLong crc = crc32(0, head + 4, 4);
    if (len) crc = crc32(crc, data, (uInt)len);
    uint8_t tail[4];
    put32(tail, (uint32_t)crc);
    return fwrite(head, 1, 8, f) == 8 && (!len || fwrite(data, 1, len, f) == len) && fwrite(tail, 1, 4, f) == 4 ? 0 : -1;
}

/* The picture to show: the image, an album's first photo, a video's
 * first frame. */
static int preview_of(const char* path, const uint8_t* data, size_t len, NvdrImage* out) {
    if (len >= 4 && !memcmp(data, NVDR_MAGIC, 4)) {
        NvdrHeader hdr;
        NvdrDecodeInfo info;
        return nvdr_decode_mem(data, len, -1, out, &hdr, &info);
    }
    if (len >= 4 && !memcmp(data, NVDA_MAGIC, 4)) {
        NvdaReader r;
        if (nvda_open(&r, data, len) != 0) return -1;
        int partial = 0, ok = nvda_decode(&r, 0, out, &partial);
        nvda_close(&r);
        return ok == 1 ? 0 : -1;
    }
    if (len >= 4 && !memcmp(data, NVDRV_MAGIC, 4)) {
        NvdrvDecoder* dec;
        NvdrvInfo info;
        if (nvdrv_decode_open(&dec, path, &info) != 0) return -1;
        out->width = info.width;
        out->height = info.height;
        out->pixels = malloc((size_t)info.width * info.height * 3);
        int kind, partial, ok = out->pixels ? nvdrv_decode_next(dec, out, &kind, &partial) : -1;
        nvdrv_decode_close(dec);
        if (ok == 1) return 0;
        free(out->pixels);
        return -1;
    }
    return -1;
}

/* Averaged down so the longer side is at most `side` (box filter over
 * each output pixel's footprint). */
static NvdrImage shrink(const NvdrImage* in, int side) {
    int big = in->width > in->height ? in->width : in->height;
    if (side <= 0 || big <= side) {
        NvdrImage copy = { malloc((size_t)in->width * in->height * 3), in->width, in->height };
        if (copy.pixels) memcpy(copy.pixels, in->pixels, (size_t)in->width * in->height * 3);
        return copy;
    }
    NvdrImage o;
    o.width = (int)((long)in->width * side / big);
    o.height = (int)((long)in->height * side / big);
    if (o.width < 1) o.width = 1;
    if (o.height < 1) o.height = 1;
    o.pixels = malloc((size_t)o.width * o.height * 3);
    if (!o.pixels) return o;
    for (int y = 0; y < o.height; y++) {
        int y0 = (int)((long)y * in->height / o.height), y1 = (int)((long)(y + 1) * in->height / o.height);
        for (int x = 0; x < o.width; x++) {
            int x0 = (int)((long)x * in->width / o.width), x1 = (int)((long)(x + 1) * in->width / o.width);
            long s[3] = { 0, 0, 0 }, n = (long)(x1 - x0) * (y1 - y0);
            for (int j = y0; j < y1; j++)
                for (int i = x0; i < x1; i++)
                    for (int c = 0; c < 3; c++) s[c] += in->pixels[((size_t)j * in->width + i) * 3 + c];
            for (int c = 0; c < 3; c++) o.pixels[((size_t)y * o.width + x) * 3 + c] = (uint8_t)((s[c] + n / 2) / n);
        }
    }
    return o;
}

static int paeth(int a, int b, int c) {
    int p = a + b - c, pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
    return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

/* The preview's scanlines (`bpp` bytes a pixel), each with whichever of
 * the five PNG filters leaves the smallest sum of magnitudes, the usual
 * heuristic (libpng's); palette indices are not numbers, so they go
 * unfiltered, as the PNG specification advises. */
static uint8_t* filtered(const uint8_t* px, int width, int height, int bpp, size_t* len) {
    size_t row = (size_t)width * bpp;
    uint8_t* out = malloc((row + 1) * height);
    uint8_t* tries = malloc(row * 5);
    if (!out || !tries) { free(out); free(tries); return NULL; }
    int filters = bpp == 1 ? 1 : 5;
    for (int y = 0; y < height; y++) {
        const uint8_t* cur = px + (size_t)y * row;
        const uint8_t* up = y ? cur - row : NULL;
        long best = -1;
        int pick = 0;
        for (int f = 0; f < filters; f++) {
            uint8_t* t = tries + (size_t)f * row;
            long sum = 0;
            for (size_t i = 0; i < row; i++) {
                int a = i >= (size_t)bpp ? cur[i - bpp] : 0, b = up ? up[i] : 0, c = up && i >= (size_t)bpp ? up[i - bpp] : 0;
                int p = f == 0 ? 0 : f == 1 ? a : f == 2 ? b : f == 3 ? (a + b) / 2 : paeth(a, b, c);
                t[i] = (uint8_t)(cur[i] - p);
                sum += t[i] < 128 ? t[i] : 256 - t[i];
            }
            if (best < 0 || sum < best) { best = sum; pick = f; }
        }
        out[(size_t)y * (row + 1)] = (uint8_t)pick;
        memcpy(out + (size_t)y * (row + 1) + 1, tries + (size_t)pick * row, row);
    }
    free(tries);
    *len = (row + 1) * height;
    return out;
}

/* Median cut (Heckbert, "Color image quantization for frame buffer
 * display", SIGGRAPH 1982) on a 15-bit histogram: the box with the most
 * pixels times its longest side is split at its median along that side,
 * until there are `colours` boxes; each colour is its box's mean, and
 * every pixel takes the nearest colour. Returns the palette size, and
 * fills `index` with one byte a pixel. */
typedef struct { int lo[3], hi[3]; long count; } Box;

static void box_fit(Box* b, const long* hist) {
    int lo[3] = { 31, 31, 31 }, hi[3] = { 0, 0, 0 };
    long n = 0;
    for (int r = b->lo[0]; r <= b->hi[0]; r++)
        for (int g = b->lo[1]; g <= b->hi[1]; g++)
            for (int u = b->lo[2]; u <= b->hi[2]; u++) {
                long h = hist[r << 10 | g << 5 | u];
                if (!h) continue;
                n += h;
                int v[3] = { r, g, u };
                for (int c = 0; c < 3; c++) { if (v[c] < lo[c]) lo[c] = v[c]; if (v[c] > hi[c]) hi[c] = v[c]; }
            }
    b->count = n;
    if (n) for (int c = 0; c < 3; c++) { b->lo[c] = lo[c]; b->hi[c] = hi[c]; }
}

static int quantise_palette(const NvdrImage* img, int colours, uint8_t pal[256][3], uint8_t* index) {
    size_t n = (size_t)img->width * img->height;
    long* hist = calloc(32768, sizeof(long));
    double (*sum)[3] = calloc(32768, sizeof *sum);
    uint8_t* near = malloc(32768);
    Box* box = malloc(sizeof(Box) * 256);
    int nb = 0;
    if (!hist || !sum || !near || !box) goto out;
    for (size_t i = 0; i < n; i++) {
        const uint8_t* p = img->pixels + i * 3;
        int k = (p[0] >> 3) << 10 | (p[1] >> 3) << 5 | p[2] >> 3;
        hist[k]++;
        for (int c = 0; c < 3; c++) sum[k][c] += p[c];
    }
    box[0] = (Box){ { 0, 0, 0 }, { 31, 31, 31 }, 0 };
    box_fit(&box[0], hist);
    nb = 1;
    while (nb < colours) {
        int pick = -1, axis = 0;
        double worth = 0;
        for (int i = 0; i < nb; i++) {
            int a = 0;
            for (int c = 1; c < 3; c++) if (box[i].hi[c] - box[i].lo[c] > box[i].hi[a] - box[i].lo[a]) a = c;
            double w = (double)box[i].count * (box[i].hi[a] - box[i].lo[a]);
            if (w > worth) { worth = w; pick = i; axis = a; }
        }
        if (pick < 0) break;   /* every box is one bin */
        /* the median along `axis` */
        Box* b = &box[pick];
        long half = b->count / 2, acc = 0;
        int cut = b->lo[axis];
        for (int v = b->lo[axis]; v < b->hi[axis]; v++) {
            Box slab = *b;
            slab.lo[axis] = slab.hi[axis] = v;
            box_fit(&slab, hist);
            acc += slab.count;
            cut = v;
            if (acc >= half) break;
        }
        Box upper = *b;
        b->hi[axis] = cut;
        upper.lo[axis] = cut + 1;
        box_fit(b, hist);
        box_fit(&upper, hist);
        box[nb++] = upper;
    }
    for (int i = 0; i < nb; i++) {
        double s[3] = { 0, 0, 0 };
        long c = 0;
        for (int r = box[i].lo[0]; r <= box[i].hi[0]; r++)
            for (int g = box[i].lo[1]; g <= box[i].hi[1]; g++)
                for (int u = box[i].lo[2]; u <= box[i].hi[2]; u++) {
                    int k = r << 10 | g << 5 | u;
                    c += hist[k];
                    for (int j = 0; j < 3; j++) s[j] += sum[k][j];
                }
        for (int j = 0; j < 3; j++) pal[i][j] = (uint8_t)(c ? s[j] / c + 0.5 : 0);
    }
    for (int k = 0; k < 32768; k++) {
        if (!hist[k]) continue;
        double v[3] = { sum[k][0] / hist[k], sum[k][1] / hist[k], sum[k][2] / hist[k] }, best = 1e300;
        for (int i = 0; i < nb; i++) {
            double d = 0;
            for (int j = 0; j < 3; j++) d += (v[j] - pal[i][j]) * (v[j] - pal[i][j]);
            if (d < best) { best = d; near[k] = (uint8_t)i; }
        }
    }
    for (size_t i = 0; i < n; i++) {
        const uint8_t* p = img->pixels + i * 3;
        index[i] = near[(p[0] >> 3) << 10 | (p[1] >> 3) << 5 | p[2] >> 3];
    }
out:
    free(hist); free(sum); free(near); free(box);
    return nb;
}

static int wrap(const char* in_path, const char* out_path, int side, int colours) {
    size_t len;
    uint8_t* data = read_all(in_path, &len);
    if (!data) { fprintf(stderr, "cannot read '%s'\n", in_path); return 1; }
    if (len > 0x7fffffff - 12) { fprintf(stderr, "'%s' is too large for a PNG chunk\n", in_path); free(data); return 1; }
    NvdrImage full;
    if (preview_of(in_path, data, len, &full) != 0) {
        fprintf(stderr, "'%s' is not an NVDR image, album or video this build can decode\n", in_path);
        free(data);
        return 1;
    }
    NvdrImage pv = shrink(&full, side);
    nvdr_image_free(&full);
    size_t raw_len = 0;
    uint8_t pal[256][3];
    int np = 0;
    uint8_t* index = colours > 0 && pv.pixels ? malloc((size_t)pv.width * pv.height) : NULL;
    if (index) np = quantise_palette(&pv, colours, pal, index);
    uint8_t* raw = !pv.pixels ? NULL
                 : np ? filtered(index, pv.width, pv.height, 1, &raw_len)
                 : filtered(pv.pixels, pv.width, pv.height, 3, &raw_len);
    free(index);
    uLongf packed_len = raw ? compressBound((uLong)raw_len) : 0;
    uint8_t* packed = raw ? malloc(packed_len) : NULL;
    int ok = packed && compress2(packed, &packed_len, raw, (uLong)raw_len, 9) == Z_OK;
    free(raw);
    FILE* f = ok ? fopen(out_path, "wb") : NULL;
    if (f) {
        uint8_t ihdr[13];
        put32(ihdr, (uint32_t)pv.width);
        put32(ihdr + 4, (uint32_t)pv.height);
        ihdr[8] = 8; ihdr[9] = np ? 3 : 2; ihdr[10] = ihdr[11] = ihdr[12] = 0;   /* 8-bit palette or RGB */
        ok = fwrite(PNG_SIG, 1, 8, f) == 8
          && put_chunk(f, "IHDR", ihdr, 13) == 0
          && put_chunk(f, "nvDr", data, len) == 0
          && put_chunk(f, "tEXt", (const uint8_t*)note, sizeof note - 1) == 0
          && (!np || put_chunk(f, "PLTE", &pal[0][0], (size_t)np * 3) == 0)
          && put_chunk(f, "IDAT", packed, packed_len) == 0
          && put_chunk(f, "IEND", NULL, 0) == 0;
        ok = fclose(f) == 0 && ok;
    }
    if (ok) {
        size_t png = 8 + 25 + (12 + len) + (12 + sizeof note - 1) + (np ? 12 + (size_t)np * 3 : 0) + (12 + packed_len) + 12;
        printf("%s -> %s  %zu bytes: %zu NVDR + %zu preview %dx%d",
               in_path, out_path, png, len, png - len, pv.width, pv.height);
        if (np) printf(", %d colours", np);
        printf("\n");
    } else fprintf(stderr, "cannot write '%s'\n", out_path);
    free(packed);
    nvdr_image_free(&pv);
    free(data);
    return ok ? 0 : 1;
}

static int extract(const char* in_path, const char* out_path) {
    size_t len;
    uint8_t* png = read_all(in_path, &len);
    if (!png || len < 8 || memcmp(png, PNG_SIG, 8)) {
        fprintf(stderr, "'%s' is not a PNG\n", in_path);
        free(png);
        return 1;
    }
    for (size_t at = 8; at + 12 <= len;) {
        size_t n = get32(png + at);
        if (n > len - at - 12) break;
        if (!memcmp(png + at + 4, "nvDr", 4)) {
            uLong crc = crc32(crc32(0, png + at + 4, 4), png + at + 8, (uInt)n);
            if (crc != get32(png + at + 8 + n)) { fprintf(stderr, "the nvDr chunk is damaged\n"); free(png); return 1; }
            FILE* f = fopen(out_path, "wb");
            int ok = f && fwrite(png + at + 8, 1, n, f) == n;
            if (f) ok = fclose(f) == 0 && ok;
            if (ok) printf("%s -> %s  %zu bytes\n", in_path, out_path, n);
            else fprintf(stderr, "cannot write '%s'\n", out_path);
            free(png);
            return ok ? 0 : 1;
        }
        if (!memcmp(png + at + 4, "IEND", 4)) break;
        at += 12 + n;
    }
    fprintf(stderr, "'%s' carries no NVDR file (no nvDr chunk)\n", in_path);
    free(png);
    return 1;
}

int main(int argc, char** argv) {
    if (argc >= 4 && !strcmp(argv[1], "--extract")) return extract(argv[2], argv[3]);
    if (argc < 3 || argv[1][0] == '-') {
        fprintf(stderr,
            "usage: %s <in.nvdr|in.nvda|in.nvdrv> <out.png> [--preview N] [--colours N]\n"
            "       %s --extract <in.png> <out>\n"
            "  --preview N  the preview's longer side in pixels (default %d, 0 whole)\n"
            "  --colours N  the preview's palette, 2..256 (default %d), 0 for full colour\n",
            argv[0], argv[0], PNG_PREVIEW, PNG_COLOURS);
        return 2;
    }
    int side = PNG_PREVIEW, colours = PNG_COLOURS;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--preview") && i + 1 < argc) side = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "--colours") || !strcmp(argv[i], "--colors")) && i + 1 < argc) colours = atoi(argv[++i]);
    }
    if (colours == 1 || colours < 0) colours = 2;
    if (colours > 256) colours = 256;
    return wrap(argv[1], argv[2], side, colours);
}
