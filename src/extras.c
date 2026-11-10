/*
 * What rides after a container's layers (extras.h): the chunks, the alpha
 * plane coded as a picture of its own, the colour profile carried as it
 * was found, and the PNG that shows both.
 */
#include "extras.h"

#include "stb_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

static uint32_t le32(const uint8_t* p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint32_t be32(const uint8_t* p) { return (uint32_t)p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3]; }

int nvdr_extra_find(const uint8_t* data, size_t size, const char* tag, size_t* at, size_t* len) {
    size_t p = nvdr_container_bytes(data, size);
    if (!p) return -1;
    while (p + 8 <= size) {
        size_t n = le32(data + p + 4);
        if (n > size - p - 8) return 0;   /* cut short */
        if (!memcmp(data + p, tag, 4)) { *at = p + 8; *len = n; return 1; }
        p += 8 + n;
    }
    return 0;
}

int nvdr_extra_append(uint8_t** buf, size_t* len, const char* tag, const uint8_t* data, size_t n) {
    if (n > 0xffffffffu) return -1;
    uint8_t* g = (uint8_t*)realloc(*buf, *len + 8 + n);
    if (!g) return -1;
    memcpy(g + *len, tag, 4);
    g[*len + 4] = (uint8_t)n; g[*len + 5] = (uint8_t)(n >> 8);
    g[*len + 6] = (uint8_t)(n >> 16); g[*len + 7] = (uint8_t)(n >> 24);
    if (n) memcpy(g + *len + 8, data, n);
    *buf = g;
    *len += 8 + n;
    return 0;
}

/* ---------------------------------------------------------------- alpha */

int nvdr_image_load_alpha(NvdrImage* img, uint8_t** alpha, const char* path) {
    *alpha = NULL;
    int w, h, channels;
    unsigned char* p = stbi_load(path, &w, &h, &channels, 4);
    if (!p) return -1;
    size_t n = (size_t)w * h;
    NvdrImage a;
    img->width = a.width = w; img->height = a.height = h;
    img->pixels = (unsigned char*)malloc(n * 3);
    a.pixels = (unsigned char*)malloc(n * 3);
    if (!img->pixels || !a.pixels) { free(img->pixels); free(a.pixels); stbi_image_free(p); return -1; }
    /* The mean of what is seen at all, for what is not. */
    double sum[3] = { 0, 0, 0 };
    size_t seen = 0, clear = 0;
    for (size_t i = 0; i < n; i++) {
        if (p[i * 4 + 3] != 255) clear++;
        if (!p[i * 4 + 3]) continue;
        for (int c = 0; c < 3; c++) sum[c] += p[i * 4 + c];
        seen++;
    }
    unsigned char mean[3] = { 128, 128, 128 };
    if (seen) for (int c = 0; c < 3; c++) mean[c] = (unsigned char)(sum[c] / (double)seen + 0.5);
    for (size_t i = 0; i < n; i++) {
        memcpy(img->pixels + i * 3, p[i * 4 + 3] ? p + i * 4 : mean, 3);
        memset(a.pixels + i * 3, p[i * 4 + 3], 3);
    }
    stbi_image_free(p);
    int o = nvdr_exif_orientation(path);
    if (nvdr_image_orient(img, o) != 0 || nvdr_image_orient(&a, o) != 0) {
        free(img->pixels); free(a.pixels); img->pixels = NULL;
        return -1;
    }
    if (clear) {
        n = (size_t)a.width * a.height;
        if (!(*alpha = (uint8_t*)malloc(n))) { free(img->pixels); free(a.pixels); img->pixels = NULL; return -1; }
        for (size_t i = 0; i < n; i++) (*alpha)[i] = a.pixels[i * 3];
    }
    free(a.pixels);
    return 0;
}

int nvdr_alpha_append(uint8_t** buf, size_t* len, const uint8_t* alpha, int width, int height,
                      const NvdrConfig* cfg) {
    NvdrConfig c = *cfg;
    /* Half the picture's step. Its colour is flat grey, which 4:2:0 codes
     * in a few bytes; it has no grain, no base and no steps per tile. */
    c.q = cfg->q / 2 < 4 ? 4 : cfg->q / 2;
    c.chroma420 = NVDR_CHROMA_420;
    c.grain = NVDR_GRAIN_OFF;
    c.tile_q = NULL;
    c.base = NULL;
    c.residual = 0;
    c.priors = 1;
    NvdrImage g;
    g.width = width; g.height = height;
    size_t n = (size_t)width * height;
    if (!(g.pixels = (unsigned char*)malloc(n * 3))) return -1;
    for (size_t i = 0; i < n; i++) memset(g.pixels + i * 3, alpha[i], 3);
    uint8_t* coded;
    size_t coded_len;
    int rc = nvdr_encode_mem(&coded, &coded_len, &g, &c, NULL);
    free(g.pixels);
    if (rc != 0) return -1;
    rc = nvdr_extra_append(buf, len, NVDR_EXTRA_ALPHA, coded, coded_len);
    free(coded);
    return rc;
}

int nvdr_alpha_decode(const uint8_t* data, size_t size, uint8_t** alpha) {
    *alpha = NULL;
    size_t at, len;
    int found = nvdr_extra_find(data, size, NVDR_EXTRA_ALPHA, &at, &len);
    if (found <= 0) return found;
    NvdrImage g;
    NvdrHeader h;
    if (nvdr_decode_mem(data + at, len, -1, &g, &h, NULL) != 0) return -1;
    int w = data[6] | data[7] << 8, hh = data[8] | data[9] << 8;
    size_t n = (size_t)w * hh;
    if (g.width != w || g.height != hh || !(*alpha = (uint8_t*)malloc(n))) { nvdr_image_free(&g); return -1; }
    for (size_t i = 0; i < n; i++) (*alpha)[i] = g.pixels[i * 3 + 1];
    nvdr_image_free(&g);
    return 1;
}

/* ---------------------------------------------------------- the profile */

static uint8_t* read_file(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* d = size > 0 ? (uint8_t*)malloc((size_t)size) : NULL;
    if (d && fread(d, 1, (size_t)size, f) != (size_t)size) { free(d); d = NULL; }
    fclose(f);
    *len = d ? (size_t)size : 0;
    return d;
}

uint8_t* nvdr_icc_read(const char* path, size_t* len) {
    size_t n;
    uint8_t* d = read_file(path, &n);
    uint8_t* icc = NULL;
    *len = 0;
    if (!d) return NULL;
    if (n > 4 && d[0] == 0xff && d[1] == 0xd8) {
        /* JPEG: APP2 segments "ICC_PROFILE\0", numbered from 1, each with
         * how many there are; the profile is their data in that order. */
        for (int want = 1, count = 1; want <= count; want++) {
            int got = 0;
            for (size_t p = 2; p + 4 <= n && d[p] == 0xff && d[p + 1] != 0xda;) {
                size_t seg = (size_t)d[p + 2] << 8 | d[p + 3];
                if (seg < 2 || p + 2 + seg > n) break;
                if (d[p + 1] == 0xe2 && seg >= 16 && !memcmp(d + p + 4, "ICC_PROFILE\0", 12) && d[p + 16] == want) {
                    count = d[p + 17];
                    size_t m = seg - 16;
                    uint8_t* g = (uint8_t*)realloc(icc, *len + m);
                    if (!g) { free(icc); free(d); *len = 0; return NULL; }
                    icc = g;
                    memcpy(icc + *len, d + p + 18, m);
                    *len += m;
                    got = 1;
                    break;
                }
                p += 2 + seg;
            }
            if (!got) { free(icc); icc = NULL; *len = 0; break; }
        }
    } else if (n > 8 && !memcmp(d, "\x89PNG\r\n\x1a\n", 8)) {
        /* PNG: iCCP, a name, a zero, a compression method and the profile deflated. */
        for (size_t p = 8; p + 12 <= n;) {
            size_t m = be32(d + p);
            if (m > n - p - 12) break;
            if (!memcmp(d + p + 4, "IDAT", 4)) break;
            if (!memcmp(d + p + 4, "iCCP", 4)) {
                const uint8_t* c = d + p + 8;
                size_t k = 0;
                while (k < m && c[k]) k++;
                if (k + 2 < m) {
                    uLongf cap = 1 << 16;
                    for (int tries = 0; tries < 8; tries++, cap *= 4) {
                        uint8_t* g = (uint8_t*)realloc(icc, cap);
                        if (!g) break;
                        icc = g;
                        uLongf out = cap;
                        int rc = uncompress(icc, &out, c + k + 2, (uLong)(m - k - 2));
                        if (rc == Z_OK) { *len = out; break; }
                        if (rc != Z_BUF_ERROR) break;
                    }
                    if (!*len) { free(icc); icc = NULL; }
                }
                break;
            }
            p += 12 + m;
        }
    }
    free(d);
    return icc;
}

/* ------------------------------------------------------------------ PNG */

static void chunk(FILE* f, const char* type, const uint8_t* data, size_t len) {
    uint8_t n[4] = { (uint8_t)(len >> 24), (uint8_t)(len >> 16), (uint8_t)(len >> 8), (uint8_t)len };
    fwrite(n, 1, 4, f);
    fwrite(type, 1, 4, f);
    if (len) fwrite(data, 1, len, f);
    uLong crc = crc32(0, (const Bytef*)type, 4);
    if (len) crc = crc32(crc, (const Bytef*)data, (uInt)len);
    uint8_t c[4] = { (uint8_t)(crc >> 24), (uint8_t)(crc >> 16), (uint8_t)(crc >> 8), (uint8_t)crc };
    fwrite(c, 1, 4, f);
}

int nvdr_image_write_png_extra(const NvdrImage* img, const uint8_t* alpha, const uint8_t* icc, size_t icc_len,
                               const char* path) {
    int ch = alpha ? 4 : 3;
    size_t stride = (size_t)img->width * ch, raw_size = (stride + 1) * (size_t)img->height;
    uint8_t* raw = (uint8_t*)malloc(raw_size);
    if (!raw) return -1;
    for (int y = 0; y < img->height; y++) {
        uint8_t* row = raw + (stride + 1) * (size_t)y;
        *row++ = 0;
        const uint8_t* s = img->pixels + (size_t)y * img->width * 3;
        if (!alpha) memcpy(row, s, stride);
        else for (int x = 0; x < img->width; x++, row += 4, s += 3) {
            memcpy(row, s, 3);
            row[3] = alpha[(size_t)y * img->width + x];
        }
    }
    uLongf packed_size = compressBound((uLong)raw_size);
    uint8_t* packed = (uint8_t*)malloc(packed_size);
    if (!packed || compress2(packed, &packed_size, raw, (uLong)raw_size, 6) != Z_OK) { free(raw); free(packed); return -1; }
    free(raw);
    /* The profile as iCCP: a name, 0 for deflate, the profile deflated. */
    uint8_t* prof = NULL;
    size_t prof_len = 0;
    if (icc && icc_len) {
        static const char name[] = "ICC profile";
        uLongf z = compressBound((uLong)icc_len);
        prof = (uint8_t*)malloc(sizeof(name) + 1 + z);
        if (prof && compress2(prof + sizeof(name) + 1, &z, icc, (uLong)icc_len, 9) == Z_OK) {
            memcpy(prof, name, sizeof(name));
            prof[sizeof(name)] = 0;
            prof_len = sizeof(name) + 1 + z;
        }
    }
    FILE* f = fopen(path, "wb");
    if (!f) { free(packed); free(prof); return -1; }
    static const uint8_t signature[8] = { 137, 'P', 'N', 'G', 13, 10, 26, 10 };
    fwrite(signature, 1, 8, f);
    uint32_t w = (uint32_t)img->width, h = (uint32_t)img->height;
    uint8_t ihdr[13] = { (uint8_t)(w >> 24), (uint8_t)(w >> 16), (uint8_t)(w >> 8), (uint8_t)w,
                         (uint8_t)(h >> 24), (uint8_t)(h >> 16), (uint8_t)(h >> 8), (uint8_t)h,
                         8, (uint8_t)(alpha ? 6 : 2), 0, 0, 0 };
    chunk(f, "IHDR", ihdr, sizeof(ihdr));
    if (prof_len) chunk(f, "iCCP", prof, prof_len);
    chunk(f, "IDAT", packed, packed_size);
    chunk(f, "IEND", NULL, 0);
    int rc = fclose(f) == 0 ? 0 : -1;
    free(packed);
    free(prof);
    return rc;
}
