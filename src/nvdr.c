/*
 * NVDR still images, format v10. See nvdr.h for the design; README.md
 * for the measurements behind it.
 */
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "nvdr.h"
#include "entropy.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

/* ================================================================ image */

int nvdr_image_load(NvdrImage* img, const char* path) {
    int channels;
    img->pixels = stbi_load(path, &img->width, &img->height, &channels, 3);
    return img->pixels ? 0 : -1;
}

int nvdr_image_write_ppm(const NvdrImage* img, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P6\n%d %d\n255\n", img->width, img->height);
    size_t n = (size_t)img->width * img->height * 3;
    size_t written = fwrite(img->pixels, 1, n, f);
    fclose(f);
    return written == n ? 0 : -1;
}

/*
 * PNG writer. PPM is trivial to emit but opens in almost nothing, which
 * makes eyeballing a decode harder than it needs to be. zlib is already
 * linked for the container, so a real PNG costs one deflate and three
 * CRCs.
 */
static void png_chunk(FILE* f, const char* type, const uint8_t* data, size_t len) {
    uint8_t size[4] = {
        (uint8_t)(len >> 24), (uint8_t)(len >> 16), (uint8_t)(len >> 8), (uint8_t)len
    };
    fwrite(size, 1, 4, f);
    fwrite(type, 1, 4, f);
    if (len) fwrite(data, 1, len, f);

    uLong crc = crc32(0, (const Bytef*)type, 4);
    if (len) crc = crc32(crc, (const Bytef*)data, (uInt)len);
    uint8_t crc_bytes[4] = {
        (uint8_t)(crc >> 24), (uint8_t)(crc >> 16), (uint8_t)(crc >> 8), (uint8_t)crc
    };
    fwrite(crc_bytes, 1, 4, f);
}

int nvdr_image_write_png(const NvdrImage* img, const char* path) {
    /* Every scanline is prefixed with filter type 0 — no prediction. A
     * filter would shrink the file; nothing here depends on its size. */
    size_t stride = (size_t)img->width * 3;
    size_t raw_size = (stride + 1) * (size_t)img->height;
    uint8_t* raw = (uint8_t*)malloc(raw_size);
    if (!raw) return -1;
    for (int y = 0; y < img->height; y++) {
        raw[(stride + 1) * (size_t)y] = 0;
        memcpy(raw + (stride + 1) * (size_t)y + 1, img->pixels + stride * (size_t)y, stride);
    }

    uLongf packed_size = compressBound((uLong)raw_size);
    uint8_t* packed = (uint8_t*)malloc(packed_size);
    if (!packed || compress2(packed, &packed_size, raw, (uLong)raw_size, 6) != Z_OK) {
        free(raw); free(packed);
        return -1;
    }
    free(raw);

    FILE* f = fopen(path, "wb");
    if (!f) { free(packed); return -1; }

    static const uint8_t signature[8] = { 137, 'P', 'N', 'G', 13, 10, 26, 10 };
    fwrite(signature, 1, 8, f);

    uint8_t ihdr[13];
    uint32_t w = (uint32_t)img->width, h = (uint32_t)img->height;
    ihdr[0] = (uint8_t)(w >> 24); ihdr[1] = (uint8_t)(w >> 16);
    ihdr[2] = (uint8_t)(w >> 8);  ihdr[3] = (uint8_t)w;
    ihdr[4] = (uint8_t)(h >> 24); ihdr[5] = (uint8_t)(h >> 16);
    ihdr[6] = (uint8_t)(h >> 8);  ihdr[7] = (uint8_t)h;
    ihdr[8] = 8;    /* bit depth */
    ihdr[9] = 2;    /* truecolour */
    ihdr[10] = ihdr[11] = ihdr[12] = 0;
    png_chunk(f, "IHDR", ihdr, sizeof(ihdr));
    png_chunk(f, "IDAT", packed, packed_size);
    png_chunk(f, "IEND", NULL, 0);

    fclose(f);
    free(packed);
    return 0;
}

int nvdr_image_write(const NvdrImage* img, const char* path) {
    size_t len = strlen(path);
    if (len >= 4 && strcmp(path + len - 4, ".png") == 0)
        return nvdr_image_write_png(img, path);
    return nvdr_image_write_ppm(img, path);
}

void nvdr_image_free(NvdrImage* img) {
    free(img->pixels);
    img->pixels = NULL;
    img->width = img->height = 0;
}

double nvdr_psnr(const NvdrImage* a, const NvdrImage* b) {
    if (a->width != b->width || a->height != b->height) return -1.0;
    size_t n = (size_t)a->width * a->height * 3;
    double mse = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a->pixels[i] - (double)b->pixels[i];
        mse += d * d;
    }
    mse /= (double)n;
    return mse <= 0.0 ? 99.0 : 10.0 * log10(255.0 * 255.0 / mse);
}

/* ============================================================ constants */

#define NSIZES     4           /* 4, 8, 16, 32 */
#define POS_CTX    15
#define MAG_UNARY  14
#define EG_LIMIT   24          /* longest Exp-Golomb prefix a decoder accepts */
#define COEF_MAX   32767

static int size_class(int n) { return n == 4 ? 0 : n == 8 ? 1 : n == 16 ? 2 : 3; }
static int log2_int(int n) { int k = 0; while ((1 << k) < n) k++; return k; }
static int clamp_u8(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }
static int clamp_coef(long v) { return v < -COEF_MAX ? -COEF_MAX : (v > COEF_MAX ? COEF_MAX : (int)v); }

NvdrConfig nvdr_default_config(void) {
    NvdrConfig c;
    /* Chosen so the samples come out the size v9 made them, or smaller:
     * the gain is taken as quality at the same bytes. */
    c.q = 24;
    /* Measured on the samples: chroma at luma's step beat 1.5x and 2.5x. */
    c.chroma_q = 1.0f;
    c.deadzone = 0.1f;
    /* The split decision is flat across 0.06..0.3; this is the middle. */
    c.lambda_k = 0.12f;
    c.max_block = NVDR_MAX_BLOCK;
    c.min_block = NVDR_MIN_BLOCK;
    return c;
}

/* ============================================================ transform */

/*
 * HEVC's integer DCT. Every entry of the 4-, 8-, 16- and 32-point matrices
 * is one of these 33 values, 90.5 * cos(j * pi / 64) nudged so the rows
 * stay close to orthogonal, with the sign cos() gives it. Row 0 is 64,
 * 90.5 / sqrt(2). The matrix is T = 64 * sqrt(N) * C, C the orthonormal
 * DCT, so a round trip divides by 4096 * N, which for these N is a shift.
 */
static const int cos_table[33] = {
    90, 90, 90, 90, 89, 88, 87, 85, 83, 82, 80, 78, 75, 73, 70, 67,
    64, 61, 57, 54, 50, 46, 43, 38, 36, 31, 25, 22, 18, 13,  9,  4, 0
};

static int tmat[NSIZES][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];   /* [k * n + x] */
static int scan_pos[NSIZES][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
static uint8_t scan_ctx[NSIZES][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
static double bitcost[2][1 << NVDR_PROB_BITS];

static int cos_entry(int j) {
    j &= 127;
    if (j <= 32) return cos_table[j];
    if (j <= 64) return -cos_table[64 - j];
    if (j <= 96) return -cos_table[j - 64];
    return cos_table[128 - j];
}

/* Built once, the same way the JS decoder builds them; nothing here
 * depends on floating point. */
static void tables_init(void) {
    static int done = 0;
    if (done) return;
    for (int s = 0; s < NSIZES; s++) {
        int n = NVDR_MIN_BLOCK << s, unit = NVDR_MAX_BLOCK / n;
        for (int k = 0; k < n; k++)
            for (int x = 0; x < n; x++)
                tmat[s][k * n + x] = k ? cos_entry((2 * x + 1) * k * unit) : 64;
        /* Diagonal scan, low frequencies first; the context of a position
         * is its diagonal, fine at the low end where most coefficients
         * live and coarser past it. */
        int at = 0;
        for (int d = 0; d <= 2 * (n - 1); d++)
            for (int v = 0; v < n; v++) {
                int u = d - v;
                if (u < 0 || u >= n) continue;
                scan_pos[s][at] = v * n + u;
                scan_ctx[s][at] = (uint8_t)(d < 8 ? d : 8 + ((d - 8) / 4 < 6 ? (d - 8) / 4 : 6));
                at++;
            }
    }
    for (int p = 1; p < (1 << NVDR_PROB_BITS); p++) {
        double p0 = (double)p / (1 << NVDR_PROB_BITS);
        bitcost[0][p] = -log2(p0);
        bitcost[1][p] = -log2(1.0 - p0);
    }
    bitcost[0][0] = bitcost[1][0] = 16.0;
    done = 1;
}

/* Encoder only: orthonormal coefficients of an n x n block. */
static void forward_dct(int s, const double* in, double* out) {
    int n = NVDR_MIN_BLOCK << s;
    const int* t = tmat[s];
    double tmp[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    double scale = 1.0 / (4096.0 * n);
    for (int y = 0; y < n; y++)
        for (int u = 0; u < n; u++) {
            double a = 0;
            for (int x = 0; x < n; x++) a += t[u * n + x] * in[y * n + x];
            tmp[y * n + u] = a;
        }
    for (int v = 0; v < n; v++)
        for (int u = 0; u < n; u++) {
            double a = 0;
            for (int y = 0; y < n; y++) a += t[v * n + y] * tmp[y * n + u];
            out[v * n + u] = a * scale;
        }
}

/*
 * The decoder's inverse, and the encoder's too, so both reconstruct the
 * same pixels. Columns first with a shift of 6 and a clip to 16 bits,
 * then rows with a shift of 6 + log2(n): 4096 * n in all. The clip only
 * bites on damaged input, and it is what keeps every product inside 32
 * bits: 32767 * 90 * 32 is under 2^27.
 */
static void inverse_dct(int s, const int* in, int* out) {
    int n = NVDR_MIN_BLOCK << s;
    const int* t = tmat[s];
    int tmp[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    int shift2 = 6 + log2_int(n);
    for (int y = 0; y < n; y++)
        for (int u = 0; u < n; u++) {
            int a = 0;
            for (int v = 0; v < n; v++) a += t[v * n + y] * in[v * n + u];
            tmp[y * n + u] = clamp_coef((a + 32) >> 6);
        }
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            int a = 0;
            for (int u = 0; u < n; u++) a += t[u * n + x] * tmp[y * n + u];
            out[y * n + x] = (a + (1 << (shift2 - 1))) >> shift2;
        }
}

/* Round a / n to nearest, halves away from zero, n a power of two. */
static int div_round(int a, int n) {
    int k = log2_int(n);
    return a >= 0 ? (a + n / 2) >> k : -((-a + n / 2) >> k);
}

/* =============================================================== colour */

/* Encoder side, full precision. Chroma is offset by 128 so every plane
 * lives in 0..255 and 128 is neutral in all three. */
static void rgb_to_ycc(const unsigned char* p, double* o) {
    double r = p[0], g = p[1], b = p[2];
    o[0] = 0.299 * r + 0.587 * g + 0.114 * b;
    o[1] = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
    o[2] = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;
}

/* Decoder side, 16-bit fixed point, identical in nvdr.js. */
static void ycc_to_rgb(int y, int cb, int cr, unsigned char* p) {
    cb -= 128; cr -= 128;
    p[0] = (unsigned char)clamp_u8(y + ((91881 * cr + 32768) >> 16));
    p[1] = (unsigned char)clamp_u8(y + ((-22554 * cb - 46802 * cr + 32768) >> 16));
    p[2] = (unsigned char)clamp_u8(y + ((116130 * cb + 32768) >> 16));
}

/* =============================================================== models */

typedef struct {
    uint16_t split[NSIZES];
    uint16_t dc_zero[NSIZES][3];
    uint16_t dc_sign[3];
    uint16_t dc_mag[3][MAG_UNARY];
} ColourModels;

typedef struct {
    uint16_t cbf[NSIZES][3];
    uint16_t sig[NSIZES][3][POS_CTX];
    uint16_t last[NSIZES][3][POS_CTX];
    uint16_t gt1[3][4];
    uint16_t mag[3][MAG_UNARY];
} TextureModels;

static void models_fill(uint16_t* p, size_t count) {
    for (size_t i = 0; i < count; i++) p[i] = NVDR_PROB_INIT;
}

/*
 * One front end for coding and for costing. With `enc` set a symbol is
 * written and its model adapts; without, its cost in bits against the
 * model as it stands is added up and nothing changes. The encoder's
 * search costs every candidate this way, then codes the winner for real.
 */
typedef struct {
    NvdrEncoder* enc;
    double       bits;
} Sink;

static void put_bit(Sink* s, uint16_t* p, int bit) {
    if (s->enc) nvdr_enc_bit(s->enc, p, bit);
    else s->bits += bitcost[bit][*p];
}

static void put_direct(Sink* s, uint32_t v, int n) {
    if (s->enc) nvdr_enc_direct(s->enc, v, n);
    else s->bits += n;
}

/* Magnitude above the unary run: order-0 Exp-Golomb. */
static void put_escape(Sink* s, unsigned r) {
    unsigned v = r + 1;
    int n = 0;
    while ((v >> n) > 1) n++;
    put_direct(s, 0, n);
    put_direct(s, v, n + 1);
}

static int get_escape(NvdrDecoder* d, int* corrupt) {
    int n = 0;
    while (!nvdr_dec_direct(d, 1)) {
        if (++n > EG_LIMIT || d->overrun) { *corrupt = 1; return 0; }
    }
    unsigned v = n ? ((1u << n) | nvdr_dec_direct(d, n)) : 1u;
    return (int)(v - 1);
}

static void put_dc(Sink* s, ColourModels* m, int sc, int c, int v) {
    put_bit(s, &m->dc_zero[sc][c], v != 0);
    if (!v) return;
    put_bit(s, &m->dc_sign[c], v < 0);
    int r = (v < 0 ? -v : v) - 1;
    for (int i = 0; i < MAG_UNARY; i++) {
        put_bit(s, &m->dc_mag[c][i], r > i);
        if (r <= i) return;
    }
    put_escape(s, (unsigned)(r - MAG_UNARY));
}

static int get_dc(NvdrDecoder* d, ColourModels* m, int sc, int c, int* corrupt) {
    if (!nvdr_dec_bit(d, &m->dc_zero[sc][c])) return 0;
    int neg = nvdr_dec_bit(d, &m->dc_sign[c]);
    int r = 0, i = 0;
    for (; i < MAG_UNARY; i++) {
        if (!nvdr_dec_bit(d, &m->dc_mag[c][i])) break;
        r = i + 1;
    }
    if (i == MAG_UNARY) r = MAG_UNARY + get_escape(d, corrupt);
    if (r > COEF_MAX) { *corrupt = 1; r = COEF_MAX; }
    return neg ? -(r + 1) : r + 1;
}

static void put_mag(Sink* s, TextureModels* m, int c, int g, int a) {
    put_bit(s, &m->gt1[c][g], a > 1);
    if (a == 1) return;
    int r = a - 2;
    for (int i = 0; i < MAG_UNARY; i++) {
        put_bit(s, &m->mag[c][i], r > i);
        if (r <= i) return;
    }
    put_escape(s, (unsigned)(r - MAG_UNARY));
}

/* A block's AC levels in scan order, positions 1..count-1. */
static void put_texture(Sink* s, TextureModels* m, int sc, int c, const int* lv, int count) {
    int last = 0;
    for (int i = 1; i < count; i++) if (lv[i]) last = i;
    put_bit(s, &m->cbf[sc][c], last > 0);
    if (!last) return;
    int g = 0;
    for (int i = 1; i <= last; i++) {
        int a = lv[i] < 0 ? -lv[i] : lv[i];
        int pc = scan_ctx[sc][i];
        if (i < count - 1) put_bit(s, &m->sig[sc][c][pc], a != 0);
        if (!a) continue;
        if (i < count - 1) put_bit(s, &m->last[sc][c][pc], i == last);
        put_mag(s, m, c, g < 3 ? g : 3, a);
        put_direct(s, lv[i] < 0, 1);
        if (a > 1) g++;
    }
}

/* Returns 0 when the block has no texture. */
static int get_texture(NvdrDecoder* d, TextureModels* m, int sc, int c, int* lv, int count,
                       int* corrupt) {
    memset(lv, 0, sizeof(int) * count);
    if (!nvdr_dec_bit(d, &m->cbf[sc][c])) return 0;
    int g = 0;
    for (int i = 1; i < count; i++) {
        int pc = scan_ctx[sc][i];
        int sig = i < count - 1 ? nvdr_dec_bit(d, &m->sig[sc][c][pc]) : 1;
        if (!sig) continue;
        int last = i < count - 1 ? nvdr_dec_bit(d, &m->last[sc][c][pc]) : 1;
        int a;
        if (!nvdr_dec_bit(d, &m->gt1[c][g < 3 ? g : 3])) a = 1;
        else {
            int r = 0, k = 0;
            for (; k < MAG_UNARY; k++) {
                if (!nvdr_dec_bit(d, &m->mag[c][k])) break;
                r = k + 1;
            }
            if (k == MAG_UNARY) r = MAG_UNARY + get_escape(d, corrupt);
            if (r > COEF_MAX) { *corrupt = 1; r = COEF_MAX; }
            a = r + 2;
            g++;
        }
        lv[i] = nvdr_dec_direct(d, 1) ? -a : a;
        if (last || *corrupt || d->overrun) break;
    }
    return 1;
}

/* ============================================================== planes */

/*
 * The canvas both sides work on: the image padded to a multiple of the
 * smallest block, so every leaf is whole, in three 8-bit planes. `flat`
 * holds each leaf's colour and nothing else — it is what layer 0 shows
 * and the only thing predictions read. `full` adds the texture.
 */
typedef struct {
    int w, h, pw, ph, tile, min_block;
    uint8_t* flat[3];
    uint8_t* full[3];
} Canvas;

static int canvas_init(Canvas* cv, int w, int h, int tile, int min_block) {
    memset(cv, 0, sizeof(*cv));
    cv->w = w; cv->h = h; cv->tile = tile; cv->min_block = min_block;
    cv->pw = (w + min_block - 1) / min_block * min_block;
    cv->ph = (h + min_block - 1) / min_block * min_block;
    for (int c = 0; c < 3; c++) {
        cv->flat[c] = (uint8_t*)malloc((size_t)cv->pw * cv->ph);
        cv->full[c] = (uint8_t*)malloc((size_t)cv->pw * cv->ph);
        if (!cv->flat[c] || !cv->full[c]) return -1;
    }
    return 0;
}

static void canvas_free(Canvas* cv) {
    for (int c = 0; c < 3; c++) { free(cv->flat[c]); free(cv->full[c]); }
}

/*
 * A leaf's predicted colour: the rounded mean of the flat colours in the
 * row just above it and the column just to its left, whichever exist on
 * the canvas; 128 at the top-left corner. Integer, and read only from
 * `flat`, so layer 0 needs nothing else to decode.
 */
static int predict(const Canvas* cv, int c, int x, int y, int n) {
    const uint8_t* p = cv->flat[c];
    int sum = 0, k = 0;
    if (y > 0) {
        int x1 = x + n < cv->pw ? x + n : cv->pw;
        for (int i = x; i < x1; i++) sum += p[(size_t)(y - 1) * cv->pw + i];
        k += x1 - x;
    }
    if (x > 0) {
        int y1 = y + n < cv->ph ? y + n : cv->ph;
        for (int j = y; j < y1; j++) sum += p[(size_t)j * cv->pw + x - 1];
        k += y1 - y;
    }
    return k ? (sum + k / 2) / k : 128;
}

static void fill(uint8_t* p, int stride, int x, int y, int w, int h, int v) {
    for (int j = y; j < y + h; j++) memset(p + (size_t)j * stride + x, v, (size_t)w);
}

/*
 * A tile of layer 0 that never arrived is painted neutral grey. Painting
 * it with the colour its neighbours predict looks smoother and is not
 * safe: on a gradient it carries the top rows' colour down the picture,
 * worse than grey, so a longer prefix could show a worse picture. Grey
 * cannot be made worse by a tile arriving.
 */
static void tile_fallback(Canvas* cv, int tx, int ty) {
    int w = tx + cv->tile < cv->pw ? cv->tile : cv->pw - tx;
    int h = ty + cv->tile < cv->ph ? cv->tile : cv->ph - ty;
    for (int c = 0; c < 3; c++) {
        fill(cv->flat[c], cv->pw, tx, ty, w, h, 128);
        fill(cv->full[c], cv->pw, tx, ty, w, h, 128);
    }
}

/* Adds a leaf's texture from its levels. `lv` is in scan order. */
static void apply_texture(Canvas* cv, int c, int x, int y, int n, const int* lv, int step) {
    int s = size_class(n), count = n * n;
    int coef[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK], res[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    memset(coef, 0, sizeof(int) * count);
    for (int i = 1; i < count; i++) coef[scan_pos[s][i]] = clamp_coef((long)lv[i] * step);
    inverse_dct(s, coef, res);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            size_t at = (size_t)(y + j) * cv->pw + x + i;
            cv->full[c][at] = (uint8_t)clamp_u8(cv->flat[c][at] + res[j * n + i]);
        }
}

/* ============================================================== header */

static void put_u16(uint8_t* p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void put_u32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static uint32_t get_u16(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8); }
static uint32_t get_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int valid_block(int n) { return n == 4 || n == 8 || n == 16 || n == 32; }

static int read_header(const uint8_t* data, size_t size, NvdrHeader* h) {
    if (size < NVDR_HEADER_SIZE || memcmp(data, NVDR_MAGIC, 4) != 0 || data[4] != NVDR_VERSION)
        return -1;
    memset(h, 0, sizeof(*h));
    h->width = (uint16_t)get_u16(data + 6);
    h->height = (uint16_t)get_u16(data + 8);
    h->max_block = data[10];
    h->min_block = data[11];
    h->q_luma = (uint16_t)get_u16(data + 12);
    h->q_chroma = (uint16_t)get_u16(data + 14);
    h->stored_bytes[0] = get_u32(data + 16);
    h->stored_bytes[1] = get_u32(data + 20);
    if (!h->width || !h->height || (size_t)h->width * h->height > NVDR_MAX_PIXELS) return -1;
    if (!valid_block(h->max_block) || !valid_block(h->min_block) || h->min_block > h->max_block)
        return -1;
    if (!h->q_luma || !h->q_chroma) return -1;
    if (h->stored_bytes[0] > 0x7fffffffu || h->stored_bytes[1] > 0x7fffffffu) return -1;
    return 0;
}

/* ============================================================= encoder */

typedef struct {
    Canvas         cv;
    double*        src[3];      /* YCbCr, padded by replication */
    int            step[3];
    double         deadzone, lambda;
    ColourModels   cm;
    TextureModels  tm;
    int*           split;       /* one decision per node, per size */
    size_t         grid_base[NSIZES];
    int            grid_w[NSIZES];
} Enc;

static size_t node_id(const Enc* e, int x, int y, int n) {
    int s = size_class(n);
    return e->grid_base[s] + (size_t)(y / n) * e->grid_w[s] + (size_t)(x / n);
}

static int node_whole(const Canvas* cv, int x, int y, int n) { return x + n <= cv->pw && y + n <= cv->ph; }
static int node_exists(const Canvas* cv, int x, int y) { return x < cv->pw && y < cv->ph; }

static int quantise(double v, double dz) {
    int a = (int)(fabs(v) + 0.5 - dz);
    if (a < 0) a = 0;
    if (a > COEF_MAX) a = COEF_MAX;
    return v < 0 ? -a : a;
}

/*
 * Code one leaf into both layers' sinks: its three colours, then its
 * three textures, reconstructing into the canvas exactly as the decoder
 * will. Returns the squared error over the pixels inside the image.
 */
static double code_leaf(Enc* e, Sink* s0, Sink* s1, int x, int y, int n, int* textured) {
    Canvas* cv = &e->cv;
    int sc = size_class(n), count = n * n;
    double err = 0.0;
    int any = 0;
    for (int c = 0; c < 3; c++) {
        int step = e->step[c];
        int pred = predict(cv, c, x, y, n);

        double sum = 0.0;
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++) sum += e->src[c][(size_t)(y + j) * cv->pw + x + i];
        /* The orthonormal DC of (block - prediction) is n * its mean. */
        double dc = (sum / count - pred) * n;
        int dl = quantise(dc / step, 0.0);
        put_dc(s0, &e->cm, sc, c, dl);
        int colour = clamp_u8(pred + div_round(clamp_coef((long)dl * step), n));
        fill(cv->flat[c], cv->pw, x, y, n, n, colour);

        double blk[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK], co[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                blk[j * n + i] = e->src[c][(size_t)(y + j) * cv->pw + x + i] - colour;
        forward_dct(sc, blk, co);
        int lv[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
        lv[0] = 0;
        int nonzero = 0;
        for (int i = 1; i < count; i++) {
            lv[i] = quantise(co[scan_pos[sc][i]] / step, e->deadzone);
            nonzero |= lv[i];
        }
        put_texture(s1, &e->tm, sc, c, lv, count);
        if (nonzero) { apply_texture(cv, c, x, y, n, lv, step); any = 1; }
        else fill(cv->full[c], cv->pw, x, y, n, n, colour);

        for (int j = 0; j < n && y + j < cv->h; j++)
            for (int i = 0; i < n && x + i < cv->w; i++) {
                size_t at = (size_t)(y + j) * cv->pw + x + i;
                double d = e->src[c][at] - cv->full[c][at];
                err += d * d;
            }
    }
    if (textured) *textured = any;
    return err;
}

static void save_block(const Canvas* cv, int x, int y, int n, uint8_t* buf) {
    for (int c = 0; c < 3; c++)
        for (int j = 0; j < n; j++) {
            memcpy(buf, cv->flat[c] + (size_t)(y + j) * cv->pw + x, (size_t)n); buf += n;
            memcpy(buf, cv->full[c] + (size_t)(y + j) * cv->pw + x, (size_t)n); buf += n;
        }
}

static void load_block(Canvas* cv, int x, int y, int n, const uint8_t* buf) {
    for (int c = 0; c < 3; c++)
        for (int j = 0; j < n; j++) {
            memcpy(cv->flat[c] + (size_t)(y + j) * cv->pw + x, buf, (size_t)n); buf += n;
            memcpy(cv->full[c] + (size_t)(y + j) * cv->pw + x, buf, (size_t)n); buf += n;
        }
}

/*
 * Rate-distortion search over one node. The node is coded whole, then as
 * four children, both against the models as they stand, and the cheaper
 * stays in the canvas. Children see their earlier siblings' colours, as
 * the decoder will. A node that runs past the canvas has no whole option
 * and must split; one wholly past it does not exist.
 */
static double search(Enc* e, int x, int y, int n) {
    Canvas* cv = &e->cv;
    size_t id = node_id(e, x, y, n);
    int whole = node_whole(cv, x, y, n);
    int can_split = n > cv->min_block;
    double whole_cost = 1e300, split_cost = 0.0;
    uint8_t* kept = NULL;

    if (whole) {
        Sink s0 = { NULL, 0 }, s1 = { NULL, 0 };
        if (can_split) put_bit(&s0, &e->cm.split[size_class(n)], 0);
        double d = code_leaf(e, &s0, &s1, x, y, n, NULL);
        whole_cost = d + e->lambda * (s0.bits + s1.bits);
        if (!can_split) { e->split[id] = 0; return whole_cost; }
        kept = (uint8_t*)malloc((size_t)6 * n * n);
        if (kept) save_block(cv, x, y, n, kept);
    }

    Sink s0 = { NULL, 0 };
    if (whole) put_bit(&s0, &e->cm.split[size_class(n)], 1);
    split_cost = e->lambda * s0.bits;
    int h = n / 2;
    for (int k = 0; k < 4; k++) {
        int cx = x + (k & 1) * h, cy = y + (k >> 1) * h;
        if (node_exists(cv, cx, cy)) split_cost += search(e, cx, cy, h);
    }

    if (whole && kept && whole_cost <= split_cost) {
        load_block(cv, x, y, n, kept);
        free(kept);
        e->split[id] = 0;
        return whole_cost;
    }
    free(kept);
    e->split[id] = 1;
    return split_cost;
}

static void emit(Enc* e, Sink* s0, Sink* s1, int x, int y, int n, NvdrHeader* h) {
    Canvas* cv = &e->cv;
    size_t id = node_id(e, x, y, n);
    int whole = node_whole(cv, x, y, n);
    if (n > cv->min_block && whole) put_bit(s0, &e->cm.split[size_class(n)], e->split[id]);
    if (n > cv->min_block && (e->split[id] || !whole)) {
        int half = n / 2;
        for (int k = 0; k < 4; k++) {
            int cx = x + (k & 1) * half, cy = y + (k >> 1) * half;
            if (node_exists(cv, cx, cy)) emit(e, s0, s1, cx, cy, half, h);
        }
        return;
    }
    int textured = 0;
    code_leaf(e, s0, s1, x, y, n, &textured);
    h->leaves[size_class(n)]++;
    h->textured += (uint32_t)textured;
}

int nvdr_encode_mem(uint8_t** out_buf, size_t* out_len, const NvdrImage* img,
                    const NvdrConfig* cfg_in, NvdrHeader* hdr_out) {
    *out_buf = NULL; *out_len = 0;
    tables_init();
    NvdrConfig cfg = cfg_in ? *cfg_in : nvdr_default_config();
    if (img->width <= 0 || img->height <= 0 || img->width > 65535 || img->height > 65535 ||
        (size_t)img->width * img->height > NVDR_MAX_PIXELS) return -1;
    if (!valid_block(cfg.max_block)) cfg.max_block = NVDR_MAX_BLOCK;
    if (!valid_block(cfg.min_block) || cfg.min_block > cfg.max_block) cfg.min_block = NVDR_MIN_BLOCK;
    if (cfg.q < 1) cfg.q = 1;
    if (cfg.q > 4095) cfg.q = 4095;

    Enc e;
    memset(&e, 0, sizeof(e));
    int rc = -1;
    NvdrEncoder enc0, enc1;
    memset(&enc0, 0, sizeof(enc0));
    memset(&enc1, 0, sizeof(enc1));
    NvdrHeader h;
    memset(&h, 0, sizeof(h));

    if (canvas_init(&e.cv, img->width, img->height, cfg.max_block, cfg.min_block) != 0) goto done;
    Canvas* cv = &e.cv;
    for (int c = 0; c < 3; c++) {
        e.src[c] = (double*)malloc(sizeof(double) * cv->pw * cv->ph);
        if (!e.src[c]) goto done;
    }
    for (int y = 0; y < cv->ph; y++)
        for (int x = 0; x < cv->pw; x++) {
            int sx = x < cv->w ? x : cv->w - 1, sy = y < cv->h ? y : cv->h - 1;
            double o[3];
            rgb_to_ycc(img->pixels + ((size_t)sy * cv->w + sx) * 3, o);
            for (int c = 0; c < 3; c++) e.src[c][(size_t)y * cv->pw + x] = o[c];
        }

    int qc = (int)(cfg.q * cfg.chroma_q + 0.5f);
    if (qc < 1) qc = 1;
    if (qc > 4095) qc = 4095;
    e.step[0] = cfg.q; e.step[1] = e.step[2] = qc;
    e.deadzone = cfg.deadzone;
    e.lambda = cfg.lambda_k * (double)cfg.q * cfg.q;

    size_t nodes = 0;
    for (int s = 0; s < NSIZES; s++) {
        int n = NVDR_MIN_BLOCK << s;
        e.grid_base[s] = nodes;
        e.grid_w[s] = (cv->pw + n - 1) / n;
        nodes += (size_t)e.grid_w[s] * ((cv->ph + n - 1) / n);
    }
    e.split = (int*)calloc(nodes, sizeof(int));
    if (!e.split) goto done;

    models_fill((uint16_t*)&e.cm, sizeof(e.cm) / sizeof(uint16_t));
    models_fill((uint16_t*)&e.tm, sizeof(e.tm) / sizeof(uint16_t));
    if (nvdr_enc_init(&enc0, 1 << 14) != 0 || nvdr_enc_init(&enc1, 1 << 16) != 0) goto done;
    Sink s0 = { &enc0, 0 }, s1 = { &enc1, 0 };

    /* Each tile is searched against the models as they stand, then coded
     * for real, which is what adapts them for the next. */
    for (int ty = 0; ty < cv->ph; ty += cv->tile)
        for (int tx = 0; tx < cv->pw; tx += cv->tile) {
            search(&e, tx, ty, cv->tile);
            emit(&e, &s0, &s1, tx, ty, cv->tile, &h);
        }
    if (nvdr_enc_finish(&enc0) != 0 || nvdr_enc_finish(&enc1) != 0) goto done;

    size_t total = NVDR_HEADER_SIZE + enc0.count + enc1.count;
    uint8_t* buf = (uint8_t*)malloc(total);
    if (!buf) goto done;
    memset(buf, 0, NVDR_HEADER_SIZE);
    memcpy(buf, NVDR_MAGIC, 4);
    buf[4] = NVDR_VERSION;
    put_u16(buf + 6, (uint32_t)img->width);
    put_u16(buf + 8, (uint32_t)img->height);
    buf[10] = (uint8_t)cfg.max_block;
    buf[11] = (uint8_t)cfg.min_block;
    put_u16(buf + 12, (uint32_t)e.step[0]);
    put_u16(buf + 14, (uint32_t)e.step[1]);
    put_u32(buf + 16, (uint32_t)enc0.count);
    put_u32(buf + 20, (uint32_t)enc1.count);
    memcpy(buf + NVDR_HEADER_SIZE, enc0.bytes, enc0.count);
    memcpy(buf + NVDR_HEADER_SIZE + enc0.count, enc1.bytes, enc1.count);

    h.width = (uint16_t)img->width; h.height = (uint16_t)img->height;
    h.max_block = (uint8_t)cfg.max_block; h.min_block = (uint8_t)cfg.min_block;
    h.q_luma = (uint16_t)e.step[0]; h.q_chroma = (uint16_t)e.step[1];
    h.stored_bytes[0] = (uint32_t)enc0.count;
    h.stored_bytes[1] = (uint32_t)enc1.count;
    if (hdr_out) *hdr_out = h;
    *out_buf = buf;
    *out_len = total;
    rc = 0;

done:
    nvdr_enc_free(&enc0);
    nvdr_enc_free(&enc1);
    for (int c = 0; c < 3; c++) free(e.src[c]);
    free(e.split);
    canvas_free(&e.cv);
    return rc;
}

int nvdr_encode_file(const char* out_path, const NvdrImage* img,
                     const NvdrConfig* cfg, NvdrHeader* hdr_out) {
    uint8_t* buf; size_t len;
    if (nvdr_encode_mem(&buf, &len, img, cfg, hdr_out) != 0) return -1;
    FILE* f = fopen(out_path, "wb");
    int rc = -1;
    if (f) {
        rc = fwrite(buf, 1, len, f) == len ? 0 : -1;
        if (fclose(f) != 0) rc = -1;
    }
    free(buf);
    return rc;
}

/* ============================================================= decoder */

typedef struct { uint16_t x, y; uint8_t n; } Leaf;

typedef struct {
    Canvas*        cv;
    NvdrDecoder*   d;
    ColourModels*  cm;
    const int*     step;
    Leaf*          leaves;
    size_t         count, cap;
    int            corrupt;
} Layer0;

static int push_leaf(Layer0* L, int x, int y, int n) {
    if (L->count == L->cap) {
        size_t cap = L->cap ? L->cap * 2 : 4096;
        Leaf* p = (Leaf*)realloc(L->leaves, cap * sizeof(Leaf));
        if (!p) return -1;
        L->leaves = p; L->cap = cap;
    }
    L->leaves[L->count].x = (uint16_t)x;
    L->leaves[L->count].y = (uint16_t)y;
    L->leaves[L->count].n = (uint8_t)n;
    L->count++;
    return 0;
}

static int read_node(Layer0* L, int x, int y, int n) {
    Canvas* cv = L->cv;
    int whole = node_whole(cv, x, y, n);
    int split = 0;
    if (n > cv->min_block) split = whole ? nvdr_dec_bit(L->d, &L->cm->split[size_class(n)]) : 1;
    if (L->d->overrun || L->corrupt) return 0;
    if (split) {
        int h = n / 2;
        for (int k = 0; k < 4; k++) {
            int cx = x + (k & 1) * h, cy = y + (k >> 1) * h;
            if (node_exists(cv, cx, cy) && read_node(L, cx, cy, h) != 0) return -1;
            if (L->d->overrun || L->corrupt) return 0;
        }
        return 0;
    }
    int sc = size_class(n);
    for (int c = 0; c < 3; c++) {
        int pred = predict(cv, c, x, y, n);
        int dl = get_dc(L->d, L->cm, sc, c, &L->corrupt);
        int colour = clamp_u8(pred + div_round(clamp_coef((long)dl * L->step[c]), n));
        fill(cv->flat[c], cv->pw, x, y, n, n, colour);
        fill(cv->full[c], cv->pw, x, y, n, n, colour);
    }
    return push_leaf(L, x, y, n);
}

int nvdr_decode_mem(const uint8_t* data, size_t size, int max_layer,
                    NvdrImage* out, NvdrHeader* hdr_out, NvdrDecodeInfo* info) {
    out->pixels = NULL; out->width = out->height = 0;
    tables_init();
    NvdrHeader h;
    if (read_header(data, size, &h) != 0) return -1;
    if (hdr_out) *hdr_out = h;

    size_t avail0 = size - NVDR_HEADER_SIZE;
    if (avail0 > h.stored_bytes[0]) avail0 = h.stored_bytes[0];
    /* The range decoder primes itself with five bytes; fewer than that
     * and not one symbol of layer 0 can be read. */
    if (avail0 < 5) return -1;
    size_t off1 = NVDR_HEADER_SIZE + (size_t)h.stored_bytes[0];
    size_t avail1 = size > off1 ? size - off1 : 0;
    if (avail1 > h.stored_bytes[1]) avail1 = h.stored_bytes[1];

    Canvas cv;
    int rc = -1;
    Layer0 L;
    memset(&L, 0, sizeof(L));
    size_t* tile_start = NULL;
    if (canvas_init(&cv, h.width, h.height, h.max_block, h.min_block) != 0) goto done;
    int tiles_x = (cv.pw + cv.tile - 1) / cv.tile, tiles_y = (cv.ph + cv.tile - 1) / cv.tile;
    int tiles = tiles_x * tiles_y;
    tile_start = (size_t*)malloc(sizeof(size_t) * (tiles + 1));
    if (!tile_start) goto done;
    int step[3] = { h.q_luma, h.q_chroma, h.q_chroma };

    /* Layer 0: every tile that arrives whole; the rest predicted. */
    ColourModels cm;
    models_fill((uint16_t*)&cm, sizeof(cm) / sizeof(uint16_t));
    NvdrDecoder d0;
    nvdr_dec_init(&d0, data + NVDR_HEADER_SIZE, avail0);
    L.cv = &cv; L.d = &d0; L.cm = &cm; L.step = step;
    int complete0 = 0, stopped = 0;
    for (int t = 0; t < tiles; t++) {
        int tx = (t % tiles_x) * cv.tile, ty = (t / tiles_x) * cv.tile;
        tile_start[t] = L.count;
        if (!stopped) {
            if (read_node(&L, tx, ty, cv.tile) != 0) goto done;
            if (d0.overrun || L.corrupt) { stopped = 1; L.count = tile_start[t]; }
            else complete0++;
        }
        if (stopped) tile_fallback(&cv, tx, ty);
    }
    tile_start[tiles] = L.count;

    /* Layer 1: texture for the tiles layer 0 delivered, as far as its own
     * bytes reach. A tile cut short keeps its flat colours. */
    int complete1 = 0;
    if ((max_layer < 0 || max_layer >= 1) && avail1 >= 5 && complete0 > 0) {
        TextureModels tm;
        models_fill((uint16_t*)&tm, sizeof(tm) / sizeof(uint16_t));
        NvdrDecoder d1;
        nvdr_dec_init(&d1, data + off1, avail1);
        int corrupt = 0;
        int lv[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
        for (int t = 0; t < complete0; t++) {
            for (size_t i = tile_start[t]; i < tile_start[t + 1] && !d1.overrun && !corrupt; i++) {
                const Leaf* f = &L.leaves[i];
                int sc = size_class(f->n);
                for (int c = 0; c < 3 && !d1.overrun && !corrupt; c++)
                    if (get_texture(&d1, &tm, sc, c, lv, f->n * f->n, &corrupt) && !d1.overrun && !corrupt)
                        apply_texture(&cv, c, f->x, f->y, f->n, lv, step[c]);
            }
            if (d1.overrun || corrupt) {
                /* Undo whatever of this tile was painted from bad bytes. */
                for (size_t i = tile_start[t]; i < tile_start[t + 1]; i++) {
                    const Leaf* f = &L.leaves[i];
                    for (int c = 0; c < 3; c++)
                        for (int j = 0; j < f->n; j++)
                            memcpy(cv.full[c] + (size_t)(f->y + j) * cv.pw + f->x,
                                   cv.flat[c] + (size_t)(f->y + j) * cv.pw + f->x, f->n);
                }
                break;
            }
            complete1++;
        }
    }

    out->pixels = (unsigned char*)malloc((size_t)h.width * h.height * 3);
    if (!out->pixels) goto done;
    out->width = h.width; out->height = h.height;
    uint8_t** planes = (max_layer == 0) ? cv.flat : cv.full;
    for (int y = 0; y < h.height; y++)
        for (int x = 0; x < h.width; x++) {
            size_t at = (size_t)y * cv.pw + x;
            ycc_to_rgb(planes[0][at], planes[1][at], planes[2][at],
                       out->pixels + ((size_t)y * h.width + x) * 3);
        }
    if (info) {
        info->tiles = tiles;
        info->tiles_complete[0] = complete0;
        info->tiles_complete[1] = complete1;
        info->layers_present = (complete0 == tiles && avail1 >= 5) ? 2 : 1;
    }
    rc = 0;

done:
    free(L.leaves);
    free(tile_start);
    canvas_free(&cv);
    return rc;
}

int nvdr_decode_file(const char* path, int max_layer,
                     NvdrImage* out, NvdrHeader* hdr, NvdrDecodeInfo* info) {
    out->pixels = NULL;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) { fclose(f); return -1; }
    uint8_t* data = (uint8_t*)malloc((size_t)size);
    if (!data) { fclose(f); return -1; }
    size_t got = fread(data, 1, (size_t)size, f);
    fclose(f);
    int rc = nvdr_decode_mem(data, got, max_layer, out, hdr, info);
    free(data);
    return rc;
}
