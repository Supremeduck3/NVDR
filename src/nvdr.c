/*
 * NVDR still images, format v10. See nvdr.h for the design; README.md
 * for the measurements behind it.
 */
/* NVDR_WASM builds the decoder for the browser (wasm/): no files, no
 * image formats, no zlib; everything else is the same code. */
#ifndef NVDR_WASM
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

#include "nvdr.h"
#include "grain.h"
#include "entropy.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef NVDR_WASM
#include <zlib.h>
#endif

/* ================================================================ image */

#ifndef NVDR_WASM
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
#endif

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
    c.rdoq = 1;
    /* The split decision is flat across 0.06..0.3; this is the middle. */
    c.lambda_k = 0.12f;
    c.max_block = NVDR_MAX_BLOCK;
    c.min_block = NVDR_MIN_BLOCK;
    c.residual = 0;
    c.deblock = 1;
    /* Measured on a 40-frame clip with a still background and a moving
     * subject: background pixels that change from one decoded frame to
     * the next fell from 5.5% to 1.3% (8.7% to 2.0% with sensor noise),
     * for 19% (26%) fewer bytes. On the panning clip, where every block
     * moves, 0.25 sits 0.1-0.2 dB above the plain quantiser curve; at 1,
     * skipped blocks carry their reference's error forward and it falls
     * 0.1 dB below. */
    c.skip_k = 0.25f;
    c.band = 8;
    c.chroma420 = NVDR_CHROMA_AUTO;
    c.grain = NVDR_GRAIN_OFF;
    c.tile_q = NULL;
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
static uint8_t scan_diag[NSIZES][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];   /* u + v */
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
                scan_diag[s][at] = (uint8_t)d;
                at++;
            }
    }
#ifndef NVDR_WASM   /* what symbols cost is for the encoder's search */
    for (int p = 1; p < (1 << NVDR_PROB_BITS); p++) {
        double p0 = (double)p / (1 << NVDR_PROB_BITS);
        bitcost[0][p] = -log2(p0);
        bitcost[1][p] = -log2(1.0 - p0);
    }
#endif
    bitcost[0][0] = bitcost[1][0] = 16.0;
    done = 1;
}

/* Encoder only: orthonormal coefficients of an n x n block. */
static void forward_dct(int s, const double* in, double* out) {
    /* Every sum runs over its terms in the same order as a plain row-times-
     * column product; the loops are only nested so the innermost one walks
     * contiguous memory, which the compiler vectorises. */
    int n = NVDR_MIN_BLOCK << s;
    const int* t = tmat[s];
    double tmp[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK], tt[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    double scale = 1.0 / (4096.0 * n);
    for (int u = 0; u < n; u++)
        for (int x = 0; x < n; x++) tt[x * n + u] = t[u * n + x];
    for (int y = 0; y < n; y++) {
        double* row = tmp + y * n;
        for (int u = 0; u < n; u++) row[u] = 0.0;
        for (int x = 0; x < n; x++) {
            double p = in[y * n + x];
            const double* col = tt + x * n;
            for (int u = 0; u < n; u++) row[u] += col[u] * p;
        }
    }
    for (int v = 0; v < n; v++) {
        double* row = out + v * n;
        for (int u = 0; u < n; u++) row[u] = 0.0;
        for (int y = 0; y < n; y++) {
            double k = t[v * n + y];
            const double* src = tmp + y * n;
            for (int u = 0; u < n; u++) row[u] += k * src[u];
        }
        for (int u = 0; u < n; u++) row[u] *= scale;
    }
}

/*
 * The decoder's inverse, and the encoder's too, so both reconstruct the
 * same pixels. Columns first with a shift of 6 and a clip to 16 bits,
 * then rows with a shift of 6 + log2(n): 4096 * n in all. The clip only
 * bites on damaged input, and it is what keeps every product inside 32
 * bits: 32767 * 90 * 32 is under 2^27.
 */
static void inverse_dct(int s, const int* in, int* out, int mu, int mv) {
    /* mu and mv bound the nonzero coefficients; every term past them is a
     * zero, so the bounds change the time and nothing else. Integer sums,
     * so the loop order, chosen for contiguous inner loops, changes
     * nothing either. */
    int n = NVDR_MIN_BLOCK << s;
    const int* t = tmat[s];
    int tmp[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    int shift2 = 6 + log2_int(n);
    for (int y = 0; y < n; y++) {
        int a[NVDR_MAX_BLOCK];
        for (int u = 0; u <= mu; u++) a[u] = 0;
        for (int v = 0; v <= mv; v++) {
            int k = t[v * n + y];
            const int* row = in + v * n;
            for (int u = 0; u <= mu; u++) a[u] += k * row[u];
        }
        for (int u = 0; u <= mu; u++) tmp[y * n + u] = clamp_coef((a[u] + 32) >> 6);
    }
    for (int y = 0; y < n; y++) {
        int a[NVDR_MAX_BLOCK];
        for (int x = 0; x < n; x++) a[x] = 0;
        for (int u = 0; u <= mu; u++) {
            int k = tmp[y * n + u];
            const int* row = t + u * n;
            for (int x = 0; x < n; x++) a[x] += k * row[x];
        }
        for (int x = 0; x < n; x++) out[y * n + x] = (a[x] + (1 << (shift2 - 1))) >> shift2;
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

/*
 * TILE STEP OFFSETS
 * -----------------
 * With NVDR_FLAG_TILEQ each tile carries, at its start in layer 0, how
 * far its step sits from the header's, in sixths of a doubling (H.264's
 * QP scale), -TQ_MAX to TQ_MAX, coded as the difference from the tile
 * before. A sequence's encoder uses it to give the parts of an intra frame
 * that the frames after it reuse a finer step (see nvdrv.c). The scale is
 * integer, identical in every decoder: 1024 * 2^(d / 6), rounded.
 */
#define TQ_MAX 12
static const int TQ_SCALE[2 * TQ_MAX + 1] = {
    256, 287, 323, 362, 406, 456, 512, 575, 645, 724, 813, 912,
    1024, 1149, 1290, 1448, 1625, 1825, 2048, 2299, 2580, 2896, 3251, 3649, 4096
};
static int tq_step(int step, int d) {
    int s = (step * TQ_SCALE[d + TQ_MAX] + 512) >> 10;
    return s < 1 ? 1 : (s > 65535 ? 65535 : s);
}

/* =============================================================== models */

typedef struct {
    uint16_t split[NSIZES];
    uint16_t split_c[NSIZES];    /* the colour tree's, in 4:2:0 */
    uint16_t dc_zero[NSIZES][3];
    uint16_t dc_sign[3];
    uint16_t dc_mag[3][MAG_UNARY];
    uint16_t tq_zero, tq_sign, tq_mag[2 * TQ_MAX];   /* the tile step offsets */
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
 * THE FLUID CONTEXT
 * -----------------
 * Every container starts its adaptive models at even odds and pays, in
 * its first symbols, to learn what the last one already knew: that big
 * leaves rarely carry texture, how often a colour needs no correction.
 * A context carries the models as one container left them into the next,
 * so a sequence's predicted frames keep learning instead of starting
 * over. The decoder has to hold the same context, so it is only for
 * containers decoded in order and in full: a sequence's frames.
 */
struct NvdrContext {
    int           valid;
    ColourModels  cm;
    TextureModels tm, tm2;
};

NvdrContext* nvdr_context_new(void) { return (NvdrContext*)calloc(1, sizeof(NvdrContext)); }
void nvdr_context_free(NvdrContext* ctx) { free(ctx); }
void nvdr_context_reset(NvdrContext* ctx) { if (ctx) ctx->valid = 0; }
void nvdr_context_copy(NvdrContext* dst, const NvdrContext* src) { *dst = *src; }
int nvdr_context_equal(const NvdrContext* a, const NvdrContext* b) {
    if (a->valid != b->valid) return 0;
    if (!a->valid) return 1;
    return !memcmp(&a->cm, &b->cm, sizeof(a->cm)) && !memcmp(&a->tm, &b->tm, sizeof(a->tm)) &&
           !memcmp(&a->tm2, &b->tm2, sizeof(a->tm2));
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

/* A sink with a coder writes the symbol, and either way counts what it
 * costs at the model's odds before they adapt. */
static void put_bit(Sink* s, uint16_t* p, int bit) {
    s->bits += bitcost[bit][*p];
    if (s->enc) nvdr_enc_bit(s->enc, p, bit);
}

static void put_direct(Sink* s, uint32_t v, int n) {
    s->bits += n;
    if (s->enc) nvdr_enc_direct(s->enc, v, n);
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

static void put_tq(Sink* s, ColourModels* m, int v) {
    put_bit(s, &m->tq_zero, v != 0);
    if (!v) return;
    put_bit(s, &m->tq_sign, v < 0);
    int r = (v < 0 ? -v : v) - 1;
    for (int i = 0; i < 2 * TQ_MAX - 1; i++) {
        put_bit(s, &m->tq_mag[i], r > i);
        if (r <= i) return;
    }
}

static int get_tq(NvdrDecoder* d, ColourModels* m) {
    if (!nvdr_dec_bit(d, &m->tq_zero)) return 0;
    int neg = nvdr_dec_bit(d, &m->tq_sign);
    int r = 0;
    for (int i = 0; i < 2 * TQ_MAX - 1; i++) {
        if (!nvdr_dec_bit(d, &m->tq_mag[i])) break;
        r = i + 1;
    }
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

/*
 * A block's AC levels over scan positions [start, end): a coded-block
 * flag, then per position a significance flag and, after a significant
 * one, a last flag, conditioned on the diagonal; magnitudes adaptive.
 * The last position of the range needs neither flag: reaching it means it
 * is the one left.
 */
static void put_texture(Sink* s, TextureModels* m, int sc, int c, const int* lv,
                        int start, int end) {
    int last = 0;
    for (int i = start; i < end; i++) if (lv[i]) last = i;
    put_bit(s, &m->cbf[sc][c], last > 0);
    if (!last) return;
    int g = 0;
    for (int i = start; i <= last; i++) {
        int a = lv[i] < 0 ? -lv[i] : lv[i];
        int pc = scan_ctx[sc][i];
        if (i < end - 1) put_bit(s, &m->sig[sc][c][pc], a != 0);
        if (!a) continue;
        if (i < end - 1) put_bit(s, &m->last[sc][c][pc], i == last);
        put_mag(s, m, c, g < 3 ? g : 3, a);
        put_direct(s, lv[i] < 0, 1);
        if (a > 1) g++;
    }
}

/* Fills lv[start, end); returns 0 when the range carries nothing. */
static int get_texture(NvdrDecoder* d, TextureModels* m, int sc, int c, int* lv,
                       int start, int end, int* corrupt) {
    memset(lv + start, 0, sizeof(int) * (size_t)(end - start));
    if (!nvdr_dec_bit(d, &m->cbf[sc][c])) return 0;
    int g = 0;
    for (int i = start; i < end; i++) {
        int pc = scan_ctx[sc][i];
        int sig = i < end - 1 ? nvdr_dec_bit(d, &m->sig[sc][c][pc]) : 1;
        if (!sig) continue;
        int last = i < end - 1 ? nvdr_dec_bit(d, &m->last[sc][c][pc]) : 1;
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

/*
 * FREQUENCY BANDS
 * ---------------
 * Texture travels in two layers: the low frequencies of every leaf, then
 * the rest. A cut file then shows the whole picture with its coarse
 * texture, not its top half sharp and its bottom half flat. The split is
 * by diagonal (u + v) relative to the leaf: a position is low when
 * u + v <= max(1, n * band / 32). band 0 puts everything in the first
 * texture layer, which is what a sequence's frames use: nobody watches a
 * video frame arrive, and the split costs bytes.
 *
 * Returns the first scan position of the high band.
 */
static int band_split(int sc, int band) {
    int n = NVDR_MIN_BLOCK << sc, count = n * n;
    if (band <= 0) return count;
    int dmax = n * band / 32;
    if (dmax < 1) dmax = 1;
    for (int i = 1; i < count; i++) if (scan_diag[sc][i] > dmax) return i;
    return count;
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
    /* How many planes, and which component the first is: 3 from Y for
     * the one tree of 4:4:4; Y alone, then Cb and Cr at half size, for
     * the two of 4:2:0. Planes index the arrays below; components index
     * the models and the steps. */
    int np, comp0;
    int fixed_pred;              /* NVDR_FLAG_RESIDUAL: every colour predicted as 128 */
    uint8_t* flat[3];
    uint8_t* full[3];
    /* The texture added so far, unclamped (to 16 bits): `full` is
     * clamp(flat + acc). The bands are added here and clamped only when
     * shown, so a high band can pull back what the low band overshot. */
    int16_t* acc[3];
} Canvas;

static int canvas_init(Canvas* cv, int w, int h, int tile, int min_block, int np, int comp0) {
    memset(cv, 0, sizeof(*cv));
    cv->w = w; cv->h = h; cv->tile = tile; cv->min_block = min_block;
    cv->np = np; cv->comp0 = comp0;
    cv->pw = (w + min_block - 1) / min_block * min_block;
    cv->ph = (h + min_block - 1) / min_block * min_block;
    for (int c = 0; c < np; c++) {
        cv->flat[c] = (uint8_t*)malloc((size_t)cv->pw * cv->ph);
        cv->full[c] = (uint8_t*)malloc((size_t)cv->pw * cv->ph);
        cv->acc[c] = (int16_t*)calloc((size_t)cv->pw * cv->ph, sizeof(int16_t));
        if (!cv->flat[c] || !cv->full[c] || !cv->acc[c]) return -1;
    }
    return 0;
}

static void canvas_free(Canvas* cv) {
    for (int c = 0; c < 3; c++) { free(cv->flat[c]); free(cv->full[c]); free(cv->acc[c]); }
}

/*
 * A leaf's predicted colour: the rounded mean of the flat colours in the
 * row just above it and the column just to its left, whichever exist on
 * the canvas; 128 at the top-left corner. Integer, and read only from
 * `flat`, so layer 0 needs nothing else to decode.
 */
static int predict(const Canvas* cv, int c, int x, int y, int n) {
    if (cv->fixed_pred) return 128;
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

/* A region set to one flat colour and no texture. */
static void paint_flat(Canvas* cv, int c, int x, int y, int w, int h, int v) {
    fill(cv->flat[c], cv->pw, x, y, w, h, v);
    fill(cv->full[c], cv->pw, x, y, w, h, v);
    for (int j = y; j < y + h; j++) memset(cv->acc[c] + (size_t)j * cv->pw + x, 0, sizeof(int16_t) * (size_t)w);
}

/*
 * A tile of layer 0 that never arrived is painted neutral grey. Painting
 * it with the colour its neighbours predict looks smoother and is not
 * safe: on a gradient it carries the top rows' colour down the picture,
 * worse than grey, so a longer prefix could show a worse picture. Grey
 * cannot be made worse by a tile arriving.
 */
static void tile_fallback(Canvas* cv, int tx, int ty) {
    if (tx >= cv->pw || ty >= cv->ph) return;
    int w = tx + cv->tile < cv->pw ? cv->tile : cv->pw - tx;
    int h = ty + cv->tile < cv->ph ? cv->tile : cv->ph - ty;
    for (int c = 0; c < cv->np; c++) {
        paint_flat(cv, c, tx, ty, w, h, 128);
    }
}

/* The pixels the texture in scan positions [start, end) adds to a leaf. */
static void texture_residual(int s, const int* lv, int start, int end, int step, int* res) {
    int n = NVDR_MIN_BLOCK << s, count = n * n;
    int coef[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    memset(coef, 0, sizeof(int) * count);
    int mu = 0, mv = 0, sh = log2_int(n);
    for (int i = start; i < end; i++) {
        if (!lv[i]) continue;
        int p = scan_pos[s][i], u = p & (n - 1), v = p >> sh;
        coef[p] = clamp_coef((long)lv[i] * step);
        if (u > mu) mu = u;
        if (v > mv) mv = v;
    }
    inverse_dct(s, coef, res, mu, mv);
}

/* Adds them to the leaf's accumulated texture, and shows flat + texture
 * clamped. */
static void add_residual(Canvas* cv, int c, int x, int y, int n, const int* res) {
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            size_t at = (size_t)(y + j) * cv->pw + x + i;
            int a = cv->acc[c][at] + res[j * n + i];
            a = a < -32768 ? -32768 : (a > 32767 ? 32767 : a);
            cv->acc[c][at] = (int16_t)a;
            cv->full[c][at] = (uint8_t)clamp_u8(cv->flat[c][at] + a);
        }
}

static void apply_texture(Canvas* cv, int c, int x, int y, int n, const int* lv,
                          int start, int end, int step) {
    int res[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    texture_residual(size_class(n), lv, start, end, step, res);
    add_residual(cv, c, x, y, n, res);
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
    h->flags = data[5];
    h->stored_bytes[0] = get_u32(data + 16);
    h->stored_bytes[1] = get_u32(data + 20);
    h->stored_bytes[2] = get_u32(data + 24);
    h->band = data[28];
    if (!h->width || !h->height || (size_t)h->width * h->height > NVDR_MAX_PIXELS) return -1;
    if (!valid_block(h->max_block) || !valid_block(h->min_block) || h->min_block > h->max_block)
        return -1;
    if (!h->q_luma || !h->q_chroma) return -1;
    if (h->flags & ~(NVDR_FLAG_RESIDUAL | NVDR_FLAG_DEBLOCK | NVDR_FLAG_CHROMA420 | NVDR_FLAG_GRAIN |
                     NVDR_FLAG_TILEQ))
        return -1;   /* a flag this decoder does not know */
    /* Grain parameters sit between the header and the first layer; byte
     * 29 says how many there are. */
    h->grain_len = data[29];
    if (h->flags & NVDR_FLAG_GRAIN) {
        if (h->grain_len != NVDR_GRAIN_SIZE || size < NVDR_HEADER_SIZE + (size_t)h->grain_len ||
            nvdr_grain_unpack(data + NVDR_HEADER_SIZE, h->grain_len, &h->grain) != 0) return -1;
    } else if (h->grain_len) return -1;
    if ((h->flags & NVDR_FLAG_CHROMA420) && h->max_block < 8) return -1;
    for (int k = 0; k < NVDR_LAYERS; k++) if (h->stored_bytes[k] > 0x7fffffffu) return -1;
    if (h->band > 32) return -1;
    return 0;
}

static uint16_t* split_model(ColourModels* m, const Canvas* cv, int n) {
    return cv->comp0 ? &m->split_c[size_class(n)] : &m->split[size_class(n)];
}

/* ============================================================= encoder */

/* One quadtree's worth of encoder state: the whole picture in 4:4:4; in
 * 4:2:0, luma, then colour at half size. */
typedef struct {
    Canvas         cv;
    double*        src[3];      /* the planes' source, padded by replication */
    double         lambda, skip_lambda;   /* see code_leaf() */
    double         lambda_base; /* lambda at the header's step, for rdoq() */
    int*           split;       /* one decision per node, per size */
    size_t         grid_base[NSIZES];
    int            grid_w[NSIZES];
    /* Every whole block's quantised texture, in scan order, for the row of
     * tiles being searched: [size][plane], block (bx, by) of the row at
     * ((by * (pw / n) + bx) * n * n). See precompute_row(). */
    int*           pre[NSIZES][3];
    /* and per block: the sum of its pixels, whether it has texture, and
     * the pixels its low and its high band add (same layout as pre) */
    double*        pre_sum[NSIZES][3];
    uint8_t*       pre_nz[NSIZES][3];
    int*           pre_lo[NSIZES][3];
    int*           pre_hi[NSIZES][3];
    int            pre_y;       /* the row's top */
} Part;

typedef struct {
    Part           part[2];
    int            nparts;
    Part*          p;           /* the tree being searched or coded */
    int            step[3];     /* per component, for the tile being coded */
    int            base_step[3];   /* the header's */
    const int8_t*  tq;          /* tile step offsets, or NULL */
    int            tiles_x;
    double         deadzone;
    int            rdoq;
    ColourModels   cm;
    TextureModels  tm;           /* the low band */
    TextureModels  tm2;          /* the high band */
    int            band_at[NSIZES];   /* first high-band scan position per size */
    double         plane_bits[3];   /* what the coded leaves cost, per component */
} Enc;

static size_t node_id(const Enc* e, int x, int y, int n) {
    int s = size_class(n);
    return e->p->grid_base[s] + (size_t)(y / n) * e->p->grid_w[s] + (size_t)(x / n);
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
/*
 * A leaf's quantised colours and textures, reconstructed into the canvas
 * as the decoder will see them. With `skip` set nothing is corrected: the
 * colour is the prediction and there is no texture. Returns the squared
 * error over the pixels inside the image.
 */
static double leaf_levels(Enc* e, int x, int y, int n, int skip,
                          int dl[3], int lv[3][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK], int* textured) {
    Part* P = e->p;
    Canvas* cv = &P->cv;
    int sc = size_class(n), count = n * n;
    double err = 0.0;
    int any = 0;
    for (int c = 0; c < cv->np; c++) {
        int step = e->step[cv->comp0 + c];
        int pred = predict(cv, c, x, y, n);
        dl[c] = 0;
        size_t b = (size_t)((y - P->pre_y) / n) * (cv->pw / n) + (size_t)(x / n);
        if (!skip) {
            double sum = P->pre_sum[sc][c][b];
            /* The orthonormal DC of (block - prediction) is n * its mean. */
            double dc = (sum / count - pred) * n;
            dl[c] = quantise(dc / step, 0.0);
        }
        int colour = clamp_u8(pred + div_round(clamp_coef((long)dl[c] * step), n));
        paint_flat(cv, c, x, y, n, n, colour);

        memset(lv[c], 0, sizeof(int) * count);
        int nonzero = 0;
        if (!skip) {
            memcpy(lv[c], P->pre[sc][c] + b * count, sizeof(int) * (size_t)count);
            nonzero = P->pre_nz[sc][c][b];
        }
        if (nonzero) {
            add_residual(cv, c, x, y, n, P->pre_lo[sc][c] + b * count);
            if (e->band_at[sc] < count) add_residual(cv, c, x, y, n, P->pre_hi[sc][c] + b * count);
            any = 1;
        }

        for (int j = 0; j < n && y + j < cv->h; j++)
            for (int i = 0; i < n && x + i < cv->w; i++) {
                size_t at = (size_t)(y + j) * cv->pw + x + i;
                double d = P->src[c][at] - cv->full[c][at];
                err += d * d;
            }
    }
    if (textured) *textured = any;
    return err;
}

/* s1 is the two texture layers' sinks, low band then high. */
static void leaf_emit(Enc* e, Sink* s0, Sink* s1, int n,
                      const int dl[3], int lv[3][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK]) {
    int sc = size_class(n), count = n * n, at = e->band_at[sc];
    const Canvas* cv = &e->p->cv;
    for (int c = 0; c < cv->np; c++) {
        int k = cv->comp0 + c;
        double before = s0->bits + s1[0].bits + s1[1].bits;
        put_dc(s0, &e->cm, sc, k, dl[c]);
        put_texture(&s1[0], &e->tm, sc, k, lv[c], 1, at);
        if (at < count) put_texture(&s1[1], &e->tm2, sc, k, lv[c], at, count);
        if (s0->enc) e->plane_bits[k] += s0->bits + s1[0].bits + s1[1].bits - before;
    }
}

/*
 * Code one leaf into both layers' sinks, reconstructing it into the canvas
 * exactly as the decoder will. Returns the squared error inside the image.
 *
 * In a residual (a sequence's predicted frame) the leaf may also be left
 * alone: no colour correction, no texture, so the decoder shows exactly
 * what the previous frame held there. That is chosen whenever it costs
 * less in error + skip_lambda * bits. Without it every leaf re-codes
 * whatever small error the reference carries, a little differently each
 * frame, and a background that does not move shimmers.
 */
static double code_leaf(Enc* e, Sink* s0, Sink* s1, int x, int y, int n, int* textured) {
    static int dl[3], lv[3][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    int skip = 0;
    double skip_lambda = e->p->skip_lambda;
    if (skip_lambda > 0.0) {
        Sink c0 = { NULL, 0 }, c1[2] = { { NULL, 0 }, { NULL, 0 } };
        double d_skip = leaf_levels(e, x, y, n, 1, dl, lv, NULL);
        leaf_emit(e, &c0, c1, n, dl, lv);
        double j_skip = d_skip + skip_lambda * (c0.bits + c1[0].bits + c1[1].bits);
        Sink k0 = { NULL, 0 }, k1[2] = { { NULL, 0 }, { NULL, 0 } };
        int tex = 0;
        double d_code = leaf_levels(e, x, y, n, 0, dl, lv, &tex);
        leaf_emit(e, &k0, k1, n, dl, lv);
        double j_code = d_code + skip_lambda * (k0.bits + k1[0].bits + k1[1].bits);
        skip = j_skip <= j_code;
        /* Coded wins: the canvas, dl and lv already hold it, and redoing
         * it would give the same (costing only bits is not coding). */
        if (!skip) {
            if (textured) *textured = tex;
            leaf_emit(e, s0, s1, n, dl, lv);
            return d_code;
        }
    }
    double err = leaf_levels(e, x, y, n, skip, dl, lv, textured);
    leaf_emit(e, s0, s1, n, dl, lv);
    return err;
}

/* A block's flat, full and accumulated texture: 12 bytes a pixel. */
static void save_block(const Canvas* cv, int x, int y, int n, uint8_t* buf) {
    for (int c = 0; c < cv->np; c++)
        for (int j = 0; j < n; j++) {
            size_t at = (size_t)(y + j) * cv->pw + x;
            memcpy(buf, cv->flat[c] + at, (size_t)n); buf += n;
            memcpy(buf, cv->full[c] + at, (size_t)n); buf += n;
            memcpy(buf, cv->acc[c] + at, sizeof(int16_t) * (size_t)n); buf += 2 * n;
        }
}

static void load_block(Canvas* cv, int x, int y, int n, const uint8_t* buf) {
    for (int c = 0; c < cv->np; c++)
        for (int j = 0; j < n; j++) {
            size_t at = (size_t)(y + j) * cv->pw + x;
            memcpy(cv->flat[c] + at, buf, (size_t)n); buf += n;
            memcpy(cv->full[c] + at, buf, (size_t)n); buf += n;
            memcpy(cv->acc[c] + at, buf, sizeof(int16_t) * (size_t)n); buf += 2 * n;
        }
}

/*
 * RATE-DISTORTION QUANTISATION
 * ----------------------------
 * Rounding each coefficient to its nearest level with a fixed dead zone
 * spends bits wherever a level is barely over a half: a 1 that costs
 * eight bits to say buys less error than eight bits are worth. rdoq()
 * decides the levels of one band of one block the way HEVC's reference
 * encoder does, against the texture models as they stand when the row is
 * prepared:
 *
 *   1. front to back, each coefficient takes whichever of its rounded
 *      level and one less (down to zero) costs less error plus lambda
 *      times the bits the models charge for it: significance, the
 *      "not last" flag, magnitude and sign;
 *   2. then the block's last level is chosen: every nonzero position is
 *      tried as the last, everything after it dropped, against dropping
 *      the whole band.
 *
 * Errors are in units of the step squared, so lambda there is the tree's
 * lambda over the step squared, the same at every step.
 */
/* The tree's lambda scaled for levels, swept from 0.4 to 1.5 on five
 * photographs (BD-rate against AV1's and x265's intra frames) and three
 * clips: 0.6 for pictures; residuals do as well at 0.6 as at 1, and 1
 * lets the regression gate's closed loop drift past its limit. The
 * first RDOQ_KEEP scan positions of a picture are left to the dead zone
 * (see the caller). */
#define RDOQ_PICTURE  0.6
#define RDOQ_RESIDUAL 0.6
#define RDOQ_KEEP     4

static double rd_bit(const uint16_t* p, int b) { return bitcost[b][*p]; }

static double rd_mag_bits(const TextureModels* m, int c, int g, int a) {
    double bits = rd_bit(&m->gt1[c][g], a > 1);
    if (a == 1) return bits;
    int r = a - 2;
    for (int i = 0; i < MAG_UNARY; i++) {
        bits += rd_bit(&m->mag[c][i], r > i);
        if (r <= i) return bits;
    }
    unsigned v = (unsigned)(r - MAG_UNARY) + 1;
    int n = 0;
    while ((v >> n) > 1) n++;
    return bits + 2 * n + 1;
}

static void rdoq(const TextureModels* m, int sc, int c, const double* v, int* lv,
                 int start, int end, double lam, int keep) {
    double cost[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];    /* coded cost of each position */
    double drop[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];    /* its error if dropped */
    int g = 0;
    for (int i = start; i < end; i++) {
        double a = fabs(v[i]);
        int pc = scan_ctx[sc][i];
        int l = (int)(a + 0.5);
        if (l > COEF_MAX) l = COEF_MAX;
        drop[i] = a * a;
        double zero_bits = i < end - 1 ? rd_bit(&m->sig[sc][c][pc], 0) : 0.0;
        double best = drop[i] + lam * zero_bits;
        int bl = 0;
        if (i < keep) {
            /* Kept as rounded with the dead zone: see the caller. */
            bl = (int)(a + 0.4);
            if (bl > COEF_MAX) bl = COEF_MAX;
            lv[i] = v[i] < 0 ? -bl : bl;
            double bits = bl ? rd_mag_bits(m, c, g < 3 ? g : 3, bl) + 1.0 +
                               (i < end - 1 ? rd_bit(&m->sig[sc][c][pc], 1) + rd_bit(&m->last[sc][c][pc], 0) : 0.0)
                             : zero_bits;
            cost[i] = (a - bl) * (a - bl) + lam * bits;
            if (bl > 1) g++;
            continue;
        }
        for (int t = l; t >= 1 && t >= l - 1; t--) {
            double bits = rd_mag_bits(m, c, g < 3 ? g : 3, t) + 1.0;
            if (i < end - 1) bits += rd_bit(&m->sig[sc][c][pc], 1) + rd_bit(&m->last[sc][c][pc], 0);
            double j = (a - t) * (a - t) + lam * bits;
            if (j < best) { best = j; bl = t; }
        }
        lv[i] = v[i] < 0 ? -bl : bl;
        cost[i] = best;
        if (bl > 1) g++;
    }
    /* The last level: coding up to k costs what positions start..k cost,
     * the flag that says k is last instead of not, and the error of
     * everything after k. */
    double tail = 0.0;
    for (int i = start; i < end; i++) tail += drop[i];
    double best = tail + lam * rd_bit(&m->cbf[sc][c], 0);
    int best_last = -1;
    double prefix = 0.0;
    int first_last = start;     /* no cut drops a kept level */
    for (int i = start; i < end && i < keep; i++) if (lv[i]) first_last = i;
    if (first_last > start || (keep > start && lv[start])) best = 1e300;
    for (int k = start; k < end; k++) {
        prefix += cost[k];
        tail -= drop[k];
        if (!lv[k] || k < first_last) continue;
        int pc = scan_ctx[sc][k];
        double flag = k < end - 1 ? rd_bit(&m->last[sc][c][pc], 1) - rd_bit(&m->last[sc][c][pc], 0) : 0.0;
        double j = prefix + tail + lam * (rd_bit(&m->cbf[sc][c], 1) + flag);
        if (j < best) { best = j; best_last = k; }
    }
    for (int i = best_last < start ? start : best_last + 1; i < end; i++) lv[i] = 0;
}

/*
 * The texture of a leaf is the DCT of its pixels minus its flat colour,
 * and a constant moves only the DC: every other row of the transform sums
 * to exactly zero (odd rows are antisymmetric, even rows fold into the
 * half-size transform, down to the 4-point one). So a block's quantised
 * texture does not depend on the colour its neighbours predict, and the
 * forward transforms, most of the encoder's time, can all be done before
 * the search, for a whole row of tiles at once and on every core, where
 * the search itself has to go tile by tile: each tile is decided against
 * the models and colours the one before it left.
 */
static void precompute_row(Enc* e, int ty) {
    /* The inverse transforms go too: the pixels a band adds are its
     * coefficients' alone, so what the search does per leaf and per size
     * is only adding them to the flat colour and measuring the error. */
    Part* P = e->p;
    Canvas* cv = &P->cv;
    P->pre_y = ty;
    for (int s = 0; s < NSIZES; s++) {
        int n = NVDR_MIN_BLOCK << s, count = n * n;
        if (!P->pre[s][0]) continue;
        int gw = cv->pw / n, gh = cv->tile / n, blocks = gw * gh;
        for (int c = 0; c < cv->np; c++) {
            int* dst = P->pre[s][c];
            const double* src = P->src[c];
            int base = e->base_step[cv->comp0 + c];
            double dz = e->deadzone;
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < blocks; b++) {
                int x = (b % gw) * n, y = ty + (b / gw) * n;
                if (y + n > cv->ph) continue;
                int step = e->tq ? tq_step(base, e->tq[(ty / cv->tile) * e->tiles_x + x / cv->tile]) : base;
                double blk[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK], co[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
                for (int j = 0; j < n; j++)
                    for (int i = 0; i < n; i++) blk[j * n + i] = src[(size_t)(y + j) * cv->pw + x + i];
                double sum = 0.0;
                for (int j = 0; j < n; j++)
                    for (int i = 0; i < n; i++) sum += blk[j * n + i];
                P->pre_sum[s][c][b] = sum;
                forward_dct(s, blk, co);
                int* q = dst + (size_t)b * count;
                int nonzero = 0;
                q[0] = 0;
                if (e->rdoq) {
                    /* Each band against its own layer's models. */
                    double v[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
                    for (int i = 1; i < count; i++) v[i] = co[scan_pos[s][i]] / step;
                    double lam = P->lambda_base / ((double)base * base) * (cv->fixed_pred ? RDOQ_RESIDUAL : RDOQ_PICTURE);
                    int at = e->band_at[s];
                    /* In a picture the lowest frequencies are left to the
                     * dead zone: they draw ramps, and dropping their level
                     * 1s turned the gate's gradient into bands (-6 dB for
                     * 18% of its bytes). */
                    int keep = cv->fixed_pred ? 0 : RDOQ_KEEP;
                    rdoq(&e->tm, s, cv->comp0 + c, v, q, 1, at < count ? at : count, lam, keep);
                    if (at < count) rdoq(&e->tm2, s, cv->comp0 + c, v, q, at, count, lam, keep);
                    for (int i = 1; i < count; i++) nonzero |= q[i];
                } else
                    for (int i = 1; i < count; i++) {
                        q[i] = quantise(co[scan_pos[s][i]] / step, dz);
                        nonzero |= q[i];
                    }
                P->pre_nz[s][c][b] = (uint8_t)(nonzero != 0);
                if (nonzero) {
                    int at = e->band_at[s];
                    texture_residual(s, q, 1, at, step, P->pre_lo[s][c] + (size_t)b * count);
                    if (at < count) texture_residual(s, q, at, count, step, P->pre_hi[s][c] + (size_t)b * count);
                }
            }
        }
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
    Part* P = e->p;
    Canvas* cv = &P->cv;
    size_t id = node_id(e, x, y, n);
    int whole = node_whole(cv, x, y, n);
    int can_split = n > cv->min_block;
    double whole_cost = 1e300, split_cost = 0.0;
    uint8_t* kept = NULL;

    if (whole) {
        Sink s0 = { NULL, 0 }, s1[2] = { { NULL, 0 }, { NULL, 0 } };
        if (can_split) put_bit(&s0, split_model(&e->cm, cv, n), 0);
        double d = code_leaf(e, &s0, s1, x, y, n, NULL);
        whole_cost = d + P->lambda * (s0.bits + s1[0].bits + s1[1].bits);
        if (!can_split) { P->split[id] = 0; return whole_cost; }
        kept = (uint8_t*)malloc((size_t)12 * n * n);
        if (kept) save_block(cv, x, y, n, kept);
    }

    Sink s0 = { NULL, 0 };
    if (whole) put_bit(&s0, split_model(&e->cm, cv, n), 1);
    split_cost = P->lambda * s0.bits;
    int h = n / 2;
    for (int k = 0; k < 4; k++) {
        int cx = x + (k & 1) * h, cy = y + (k >> 1) * h;
        if (node_exists(cv, cx, cy)) split_cost += search(e, cx, cy, h);
    }

    if (whole && kept && whole_cost <= split_cost) {
        load_block(cv, x, y, n, kept);
        free(kept);
        P->split[id] = 0;
        return whole_cost;
    }
    free(kept);
    P->split[id] = 1;
    return split_cost;
}

static void emit(Enc* e, Sink* s0, Sink* s1, int x, int y, int n, NvdrHeader* h) {
    Canvas* cv = &e->p->cv;
    size_t id = node_id(e, x, y, n);
    int whole = node_whole(cv, x, y, n);
    int split = e->p->split[id];
    if (n > cv->min_block && whole) put_bit(s0, split_model(&e->cm, cv, n), split);
    if (n > cv->min_block && (split || !whole)) {
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
    return nvdr_encode_mem_ctx(out_buf, out_len, img, cfg_in, hdr_out, NULL);
}

/* One encode in the mode cfg->chroma420 names (420 when nonzero). */
static int encode_mode(uint8_t** out_buf, size_t* out_len, const NvdrImage* img,
                       const NvdrConfig* cfg_in, NvdrHeader* hdr_out, NvdrContext* ctx) {
    *out_buf = NULL; *out_len = 0;
    NvdrConfig cfg = *cfg_in;
    if (img->width <= 0 || img->height <= 0 || img->width > 65535 || img->height > 65535 ||
        (size_t)img->width * img->height > NVDR_MAX_PIXELS) return -1;
    if (!valid_block(cfg.max_block)) cfg.max_block = NVDR_MAX_BLOCK;
    if (!valid_block(cfg.min_block) || cfg.min_block > cfg.max_block) cfg.min_block = NVDR_MIN_BLOCK;
    if (cfg.q < 1) cfg.q = 1;
    if (cfg.q > 4095) cfg.q = 4095;

    Enc e;
    memset(&e, 0, sizeof(e));
    int rc = -1;
    int8_t* tq = NULL;
    NvdrEncoder enc0, enc1, enc2;
    memset(&enc0, 0, sizeof(enc0));
    memset(&enc1, 0, sizeof(enc1));
    memset(&enc2, 0, sizeof(enc2));
    int band = cfg.band < 0 ? 0 : (cfg.band > 32 ? 32 : cfg.band);
    NvdrHeader h;
    memset(&h, 0, sizeof(h));

    /* 4:2:0 needs a colour tile of at least the smallest leaf. */
    int use420 = cfg.chroma420 && cfg.max_block >= 8 && cfg.max_block <= NVDR_MAX_BLOCK &&
                 (cfg.max_block & (cfg.max_block - 1)) == 0;
    e.nparts = use420 ? 2 : 1;
    if (canvas_init(&e.part[0].cv, img->width, img->height, cfg.max_block, cfg.min_block,
                    use420 ? 1 : 3, 0) != 0) goto done;
    if (use420 && canvas_init(&e.part[1].cv, (img->width + 1) / 2, (img->height + 1) / 2,
                              cfg.max_block / 2, NVDR_MIN_BLOCK, 2, 1) != 0) goto done;
    for (int k = 0; k < e.nparts; k++) {
        Canvas* pc = &e.part[k].cv;
        pc->fixed_pred = cfg.residual != 0;
        for (int c = 0; c < pc->np; c++) {
            e.part[k].src[c] = (double*)malloc(sizeof(double) * pc->pw * pc->ph);
            if (!e.part[k].src[c]) goto done;
        }
    }
    {
        /* YCbCr at full resolution, padded by replication; in 4:2:0 the
         * colour planes are then averaged 2x2, each sample centred between
         * the four pixels it stands for. */
        Canvas* cv = &e.part[0].cv;
        double* full[3] = { NULL, NULL, NULL };
        for (int c = 0; c < 3; c++) {
            full[c] = (!use420 || c == 0) ? e.part[0].src[c]
                    : (double*)malloc(sizeof(double) * cv->pw * cv->ph);
            if (!full[c]) goto done;
        }
        for (int y = 0; y < cv->ph; y++)
            for (int x = 0; x < cv->pw; x++) {
                int sx = x < cv->w ? x : cv->w - 1, sy = y < cv->h ? y : cv->h - 1;
                double o[3];
                rgb_to_ycc(img->pixels + ((size_t)sy * cv->w + sx) * 3, o);
                for (int c = 0; c < 3; c++) full[c][(size_t)y * cv->pw + x] = o[c];
            }
        if (use420) {
            Canvas* cc = &e.part[1].cv;
            for (int c = 1; c < 3; c++) {
                for (int y = 0; y < cc->ph; y++)
                    for (int x = 0; x < cc->pw; x++) {
                        double sum = 0;
                        for (int j = 0; j < 2; j++)
                            for (int i = 0; i < 2; i++) {
                                int fx = 2 * x + i, fy = 2 * y + j;
                                if (fx >= cv->pw) fx = cv->pw - 1;
                                if (fy >= cv->ph) fy = cv->ph - 1;
                                sum += full[c][(size_t)fy * cv->pw + fx];
                            }
                        e.part[1].src[c - 1][(size_t)y * cc->pw + x] = sum / 4;
                    }
                free(full[c]);
            }
        }
    }

    int qc = (int)(cfg.q * cfg.chroma_q + 0.5f);
    if (qc < 1) qc = 1;
    if (qc > 4095) qc = 4095;
    e.step[0] = cfg.q; e.step[1] = e.step[2] = qc;
    for (int c = 0; c < 3; c++) e.base_step[c] = e.step[c];
    e.deadzone = cfg.deadzone;
    e.rdoq = cfg.rdoq;
    double lambda = cfg.lambda_k * (double)cfg.q * cfg.q;
    for (int k = 0; k < e.nparts; k++) {
        /* A colour sample at half size stands for four pixels: its error
         * weighs four times, which is lambda a quarter. */
        double scale = e.part[k].cv.comp0 && use420 ? 0.25 : 1.0;
        e.part[k].lambda = lambda * scale;
        e.part[k].lambda_base = lambda * scale;
        e.part[k].skip_lambda = cfg.residual ? cfg.skip_k * lambda * scale : 0.0;
    }
    for (int s = 0; s < NSIZES; s++) e.band_at[s] = band_split(s, band);

    for (int k = 0; k < e.nparts; k++) {
        Part* P = &e.part[k];
        Canvas* pc = &P->cv;
        size_t nodes = 0;
        for (int s = 0; s < NSIZES; s++) {
            int n = NVDR_MIN_BLOCK << s;
            P->grid_base[s] = nodes;
            P->grid_w[s] = (pc->pw + n - 1) / n;
            nodes += (size_t)P->grid_w[s] * ((pc->ph + n - 1) / n);
        }
        P->split = (int*)calloc(nodes, sizeof(int));
        if (!P->split) goto done;
        for (int s = 0; s < NSIZES; s++) {
            int n = NVDR_MIN_BLOCK << s;
            if (n < pc->min_block || n > pc->tile) continue;
            for (int c = 0; c < pc->np; c++) {
                size_t blocks = (size_t)(pc->pw / n) * (pc->tile / n);
                if (!blocks) blocks = 1;
                P->pre[s][c] = (int*)malloc(sizeof(int) * blocks * n * n);
                P->pre_lo[s][c] = (int*)malloc(sizeof(int) * blocks * n * n);
                P->pre_hi[s][c] = (int*)malloc(sizeof(int) * blocks * n * n);
                P->pre_sum[s][c] = (double*)malloc(sizeof(double) * blocks);
                P->pre_nz[s][c] = (uint8_t*)malloc(blocks);
                if (!P->pre[s][c] || !P->pre_lo[s][c] || !P->pre_hi[s][c] || !P->pre_sum[s][c] || !P->pre_nz[s][c])
                    goto done;
            }
        }
    }

    if (ctx && ctx->valid) {
        e.cm = ctx->cm; e.tm = ctx->tm; e.tm2 = ctx->tm2;
    } else {
        models_fill((uint16_t*)&e.cm, sizeof(e.cm) / sizeof(uint16_t));
        models_fill((uint16_t*)&e.tm, sizeof(e.tm) / sizeof(uint16_t));
        models_fill((uint16_t*)&e.tm2, sizeof(e.tm2) / sizeof(uint16_t));
    }
    if (nvdr_enc_init(&enc0, 1 << 14) != 0 || nvdr_enc_init(&enc1, 1 << 16) != 0 ||
        nvdr_enc_init(&enc2, 1 << 16) != 0) goto done;
    Sink s0 = { &enc0, 0 }, s1[2] = { { &enc1, 0 }, { &enc2, 0 } };

    /* Each tile is searched against the models as they stand, then coded
     * for real, which is what adapts them for the next. In 4:2:0 a tile is
     * luma's tree, then colour's over the same area at half size. */
    Canvas* cv = &e.part[0].cv;
    e.tiles_x = (cv->pw + cv->tile - 1) / cv->tile;
    if (cfg.tile_q) {
        size_t tiles = (size_t)e.tiles_x * ((cv->ph + cv->tile - 1) / cv->tile);
        tq = (int8_t*)malloc(tiles);
        if (!tq) goto done;
        for (size_t t = 0; t < tiles; t++)
            tq[t] = (int8_t)(cfg.tile_q[t] < -TQ_MAX ? -TQ_MAX : cfg.tile_q[t] > TQ_MAX ? TQ_MAX : cfg.tile_q[t]);
        e.tq = tq;
    }
    double lambda0[2], skip0[2];
    for (int k = 0; k < e.nparts; k++) { lambda0[k] = e.part[k].lambda; skip0[k] = e.part[k].skip_lambda; }
    int tq_prev = 0;
    for (int ty = 0; ty < cv->ph; ty += cv->tile) {
        for (int k = 0; k < e.nparts; k++) {
            e.p = &e.part[k];
            if (ty / (k + 1) < e.p->cv.ph) precompute_row(&e, ty / (k + 1));
        }
        for (int tx = 0; tx < cv->pw; tx += cv->tile) {
            if (e.tq) {
                /* The tile's offset, then its trees at its step; lambda
                 * follows the step squared, as it does across q. */
                int d = e.tq[(ty / cv->tile) * e.tiles_x + tx / cv->tile];
                put_tq(&s0, &e.cm, d - tq_prev);
                tq_prev = d;
                for (int c = 0; c < 3; c++) e.step[c] = tq_step(e.base_step[c], d);
                double r = TQ_SCALE[d + TQ_MAX] / 1024.0;
                for (int k = 0; k < e.nparts; k++) {
                    e.part[k].lambda = lambda0[k] * r * r;
                    e.part[k].skip_lambda = skip0[k] * r * r;
                }
            }
            for (int k = 0; k < e.nparts; k++) {
                e.p = &e.part[k];
                int x = tx / (k + 1), y = ty / (k + 1);
                if (!node_exists(&e.p->cv, x, y)) continue;
                search(&e, x, y, e.p->cv.tile);
                emit(&e, &s0, s1, x, y, e.p->cv.tile, &h);
            }
        }
    }
    for (int c = 0; c < 3; c++) e.step[c] = e.base_step[c];
    if (nvdr_enc_finish(&enc0) != 0 || nvdr_enc_finish(&enc1) != 0 ||
        nvdr_enc_finish(&enc2) != 0) goto done;
    /* With no high band the stream holds no symbols, only the coder's
     * flush: leave it out, and the layer is empty. */
    size_t n2 = band ? enc2.count : 0;

    size_t total = NVDR_HEADER_SIZE + enc0.count + enc1.count + n2;
    uint8_t* buf = (uint8_t*)malloc(total);
    if (!buf) goto done;
    memset(buf, 0, NVDR_HEADER_SIZE);
    memcpy(buf, NVDR_MAGIC, 4);
    buf[4] = NVDR_VERSION;
    buf[5] = (uint8_t)((cfg.residual ? NVDR_FLAG_RESIDUAL : 0) | (cfg.deblock ? NVDR_FLAG_DEBLOCK : 0) |
                       (use420 ? NVDR_FLAG_CHROMA420 : 0) | (e.tq ? NVDR_FLAG_TILEQ : 0));
    put_u16(buf + 6, (uint32_t)img->width);
    put_u16(buf + 8, (uint32_t)img->height);
    buf[10] = (uint8_t)cfg.max_block;
    buf[11] = (uint8_t)cfg.min_block;
    put_u16(buf + 12, (uint32_t)e.step[0]);
    put_u16(buf + 14, (uint32_t)e.step[1]);
    put_u32(buf + 16, (uint32_t)enc0.count);
    put_u32(buf + 20, (uint32_t)enc1.count);
    put_u32(buf + 24, (uint32_t)n2);
    buf[28] = (uint8_t)band;
    memcpy(buf + NVDR_HEADER_SIZE, enc0.bytes, enc0.count);
    memcpy(buf + NVDR_HEADER_SIZE + enc0.count, enc1.bytes, enc1.count);
    if (n2) memcpy(buf + NVDR_HEADER_SIZE + enc0.count + enc1.count, enc2.bytes, n2);

    h.width = (uint16_t)img->width; h.height = (uint16_t)img->height;
    h.max_block = (uint8_t)cfg.max_block; h.min_block = (uint8_t)cfg.min_block;
    h.q_luma = (uint16_t)e.step[0]; h.q_chroma = (uint16_t)e.step[1];
    h.flags = (uint8_t)buf[5];
    h.stored_bytes[0] = (uint32_t)enc0.count;
    h.stored_bytes[1] = (uint32_t)enc1.count;
    h.stored_bytes[2] = (uint32_t)n2;
    h.band = (uint8_t)band;
    for (int c = 0; c < 3; c++) h.plane_bits[c] = e.plane_bits[c];
    if (hdr_out) *hdr_out = h;
    *out_buf = buf;
    *out_len = total;
    if (ctx) { ctx->cm = e.cm; ctx->tm = e.tm; ctx->tm2 = e.tm2; ctx->valid = 1; }
    rc = 0;

done:
    free(tq);
    nvdr_enc_free(&enc0);
    nvdr_enc_free(&enc1);
    nvdr_enc_free(&enc2);
    for (int k = 0; k < 2; k++) {
        Part* P = &e.part[k];
        for (int c = 0; c < 3; c++) free(P->src[c]);
        free(P->split);
        for (int s = 0; s < NSIZES; s++)
            for (int c = 0; c < 3; c++) {
                free(P->pre[s][c]); free(P->pre_lo[s][c]); free(P->pre_hi[s][c]);
                free(P->pre_sum[s][c]); free(P->pre_nz[s][c]);
            }
        canvas_free(&P->cv);
    }
    return rc;
}

/*
 * What halving the colour costs on its own: the mean squared error, per
 * colour sample, of Cb and Cr averaged 2x2 and scaled back up the way
 * the decoder does. A photograph's colour is smooth and it stays under 1;
 * saturated shapes with hard edges (graphics, text, the synthetic probes)
 * run past 10, and so do some strongly coloured photos.
 */
static double chroma_halving_mse(const NvdrImage* img) {
    int w = img->width, h = img->height, cw = (w + 1) / 2, ch = (h + 1) / 2;
    double* c = (double*)malloc(sizeof(double) * w * h);
    double* d = (double*)malloc(sizeof(double) * cw * ch);
    double total = 0;
    if (!c || !d) { free(c); free(d); return 1e9; }
    for (int k = 1; k < 3; k++) {
        for (size_t i = 0; i < (size_t)w * h; i++) {
            double o[3];
            rgb_to_ycc(img->pixels + i * 3, o);
            c[i] = o[k];
        }
        for (int y = 0; y < ch; y++)
            for (int x = 0; x < cw; x++) {
                double sum = 0;
                for (int j = 0; j < 2; j++)
                    for (int i = 0; i < 2; i++) {
                        int fy = 2 * y + j < h ? 2 * y + j : h - 1, fx = 2 * x + i < w ? 2 * x + i : w - 1;
                        sum += c[(size_t)fy * w + fx];
                    }
                d[(size_t)y * cw + x] = sum / 4;
            }
        for (int y = 0; y < h; y++) {
            int cy = y >> 1, oy = (y & 1) ? cy + 1 : cy - 1;
            if (oy < 0) oy = 0;
            if (oy >= ch) oy = ch - 1;
            for (int x = 0; x < w; x++) {
                int cx = x >> 1, ox = (x & 1) ? cx + 1 : cx - 1;
                if (ox < 0) ox = 0;
                if (ox >= cw) ox = cw - 1;
                double u = (9 * d[(size_t)cy * cw + cx] + 3 * d[(size_t)cy * cw + ox] +
                            3 * d[(size_t)oy * cw + cx] + d[(size_t)oy * cw + ox]) / 16;
                double e = u - c[(size_t)y * w + x];
                total += e * e;
            }
        }
    }
    free(c); free(d);
    return total / (2.0 * w * h);
}

/* The rate-distortion cost of a container: squared error of the whole
 * picture in YCbCr, colour weighed by `wc`, plus lambda per bit. */
static double container_cost(const uint8_t* buf, size_t len, const NvdrImage* img, NvdrContext* ctx,
                             double lambda, double wc) {
    NvdrImage shown;
    if (nvdr_decode_mem_ctx(buf, len, -1, &shown, NULL, NULL, ctx) != 0) return 1e300;
    double d = 0;
    for (size_t i = 0; i < (size_t)img->width * img->height; i++) {
        double a[3], b[3];
        rgb_to_ycc(img->pixels + i * 3, a);
        rgb_to_ycc(shown.pixels + i * 3, b);
        d += (a[0] - b[0]) * (a[0] - b[0]) + wc * ((a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
    }
    nvdr_image_free(&shown);
    return d + lambda * 8.0 * (double)len;
}

/* Below this, halving the colour is taken as free (see
 * chroma_halving_mse): every photograph measured, and it saves the second
 * encode. */
#define CHROMA_AUTO_FREE 2.0
/* How much colour error counts against luma's when the two modes are
 * compared. Swept over 0.25, 0.5 and 1: at 0.5 every photograph measured
 * keeps 4:2:0 and every synthetic picture (blocos, circulos, the gate's
 * sequence) goes 4:4:4; 0.25 let circulos and the sequence halve their
 * colour for 6 dB of loss, and 1 kept a strongly coloured photo whole
 * that 4:2:0 codes 30% smaller at the same luma. */
#define CHROMA_AUTO_WEIGHT 0.5

static int encode_chroma(uint8_t** out_buf, size_t* out_len, const NvdrImage* img,
                         const NvdrConfig* cfg_in, NvdrHeader* hdr_out, NvdrContext* ctx) {
    *out_buf = NULL; *out_len = 0;
    NvdrConfig cfg = *cfg_in;
    if (cfg.chroma420 != NVDR_CHROMA_AUTO) {
        cfg.chroma420 = cfg.chroma420 == NVDR_CHROMA_420;
        return encode_mode(out_buf, out_len, img, &cfg, hdr_out, ctx);
    }
    if (img->width <= 0 || img->height <= 0 || !img->pixels) return -1;
    /* A residual (a sequence's predicted frame, an album's predicted
     * photo) keeps its colour whole: halving it every frame lets colour
     * error build up along the chain of references, 1.3 dB over the
     * gate's twelve frames where the limit is 0.25. Halving a sequence's
     * colour belongs in its references too, as video codecs do it. */
    if (cfg.residual) {
        cfg.chroma420 = 0;
        return encode_mode(out_buf, out_len, img, &cfg, hdr_out, ctx);
    }
    /* Tile step offsets move the two modes' costs apart in ways that have
     * nothing to do with colour (a sequence's intra frame, refined where
     * it is reused, came out 4:2:0 on the gate's saturated drawing and
     * lost 5 dB of RGB), so the mode is chosen without them and then
     * coded with them. */
    if (cfg.tile_q && !ctx) {
        NvdrConfig plain = cfg;
        plain.tile_q = NULL;
        uint8_t* trial = NULL;
        size_t trial_len = 0;
        NvdrHeader th;
        if (encode_chroma(&trial, &trial_len, img, &plain, &th, NULL) != 0) return -1;
        free(trial);
        cfg.chroma420 = (th.flags & NVDR_FLAG_CHROMA420) ? 1 : 0;
        return encode_mode(out_buf, out_len, img, &cfg, hdr_out, ctx);
    }
    cfg.chroma420 = 1;
    if (chroma_halving_mse(img) < CHROMA_AUTO_FREE)
        return encode_mode(out_buf, out_len, img, &cfg, hdr_out, ctx);

    /* Both ways, each from the context as it was, and the cheaper kept
     * with the context it leaves. */
    NvdrContext* start = ctx ? nvdr_context_new() : NULL;
    NvdrContext* other = ctx ? nvdr_context_new() : NULL;
    NvdrContext* check = ctx ? nvdr_context_new() : NULL;
    if (ctx && (!start || !other || !check)) {
        nvdr_context_free(start); nvdr_context_free(other); nvdr_context_free(check);
        return -1;
    }
    if (ctx) { nvdr_context_copy(start, ctx); nvdr_context_copy(other, ctx); }
    uint8_t *a = NULL, *b = NULL; size_t alen = 0, blen = 0;
    NvdrHeader ha, hb;
    NvdrConfig c444 = cfg;
    c444.chroma420 = 0;
    int rc = -1;
    if (encode_mode(&a, &alen, img, &cfg, &ha, ctx) == 0 &&
        encode_mode(&b, &blen, img, &c444, &hb, other) == 0) {
        int q = cfg.q < 1 ? 1 : (cfg.q > 4095 ? 4095 : cfg.q);
        double lambda = cfg.lambda_k * (double)q * q;
        if (ctx) nvdr_context_copy(check, start);
        double ja = container_cost(a, alen, img, check, lambda, CHROMA_AUTO_WEIGHT);
        if (ctx) nvdr_context_copy(check, start);
        double jb = container_cost(b, blen, img, check, lambda, CHROMA_AUTO_WEIGHT);
        if (jb < ja) {
            free(a); a = b; alen = blen; b = NULL; ha = hb;
            if (ctx) nvdr_context_copy(ctx, other);
        }
        *out_buf = a; *out_len = alen; a = NULL;
        if (hdr_out) *hdr_out = ha;
        rc = 0;
    }
    free(a); free(b);
    nvdr_context_free(start); nvdr_context_free(other); nvdr_context_free(check);
    return rc;
}

/*
 * The whole encode: grain first (grain.c). A noisy picture is coded clean
 * and its grain described after the header, where the decoder finds it
 * before the first layer. Residuals never carry grain: a predicted frame
 * or photo would have to cancel its reference's grain and add its own.
 */
int nvdr_encode_mem_ctx(uint8_t** out_buf, size_t* out_len, const NvdrImage* img,
                        const NvdrConfig* cfg_in, NvdrHeader* hdr_out, NvdrContext* ctx) {
    *out_buf = NULL; *out_len = 0;
    tables_init();
    NvdrConfig cfg = cfg_in ? *cfg_in : nvdr_default_config();
    if (cfg.grain == NVDR_GRAIN_OFF || cfg.residual || !img->pixels)
        return encode_chroma(out_buf, out_len, img, &cfg, hdr_out, ctx);
    NvdrImage clean;
    NvdrGrain g;
    int got = nvdr_grain_estimate(img, cfg.grain == NVDR_GRAIN_ON, &clean, &g);
    if (got < 0) return -1;
    if (got == 0) return encode_chroma(out_buf, out_len, img, &cfg, hdr_out, ctx);
    uint8_t* buf; size_t len;
    NvdrHeader h;
    int rc = encode_chroma(&buf, &len, &clean, &cfg, &h, ctx);
    nvdr_image_free(&clean);
    if (rc != 0) return rc;
    uint8_t* out = (uint8_t*)malloc(len + NVDR_GRAIN_SIZE);
    if (!out) { free(buf); return -1; }
    memcpy(out, buf, NVDR_HEADER_SIZE);
    out[5] |= NVDR_FLAG_GRAIN;
    out[29] = NVDR_GRAIN_SIZE;
    nvdr_grain_pack(&g, out + NVDR_HEADER_SIZE);
    memcpy(out + NVDR_HEADER_SIZE + NVDR_GRAIN_SIZE, buf + NVDR_HEADER_SIZE, len - NVDR_HEADER_SIZE);
    free(buf);
    h.flags |= NVDR_FLAG_GRAIN;
    h.grain_len = NVDR_GRAIN_SIZE;
    h.grain = g;
    if (hdr_out) *hdr_out = h;
    *out_buf = out; *out_len = len + NVDR_GRAIN_SIZE;
    return 0;
}

#ifndef NVDR_WASM
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
#endif

/* ============================================================= decoder */

/*
 * Colour back to full size, for 4:2:0. A colour sample sits at the centre
 * of the 2x2 pixels it was averaged from, so a pixel lies a quarter of a
 * sample from the nearest one each way: bilinear weights 3/4 and 1/4,
 * 9 3 3 1 in sixteenths, with edges held. Integer, and identical in
 * nvdr.js.
 */
static void upsample_plane(const uint8_t* src, const Canvas* cc, uint8_t* dst, int dpw, int w, int h) {
    int cw = cc->w, ch = cc->h;
    for (int y = 0; y < h; y++) {
        int cy = y >> 1, oy = (y & 1) ? cy + 1 : cy - 1;
        if (oy < 0) oy = 0;
        if (oy >= ch) oy = ch - 1;
        const uint8_t* r0 = src + (size_t)cy * cc->pw;
        const uint8_t* r1 = src + (size_t)oy * cc->pw;
        uint8_t* out = dst + (size_t)y * dpw;
        for (int x = 0; x < w; x++) {
            int cx = x >> 1, ox = (x & 1) ? cx + 1 : cx - 1;
            if (ox < 0) ox = 0;
            if (ox >= cw) ox = cw - 1;
            out[x] = (uint8_t)((9 * r0[cx] + 3 * r0[ox] + 3 * r1[cx] + r1[ox] + 8) >> 4);
        }
    }
}



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
    if (n > cv->min_block) split = whole ? nvdr_dec_bit(L->d, split_model(L->cm, cv, n)) : 1;
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
    for (int c = 0; c < cv->np; c++) {
        int k = cv->comp0 + c;
        int pred = predict(cv, c, x, y, n);
        int dl = get_dc(L->d, L->cm, sc, k, &L->corrupt);
        int colour = clamp_u8(pred + div_round(clamp_coef((long)dl * L->step[k]), n));
        paint_flat(cv, c, x, y, n, n, colour);
    }
    return push_leaf(L, x, y, n);
}

/*
 * DEBLOCKING
 * ----------
 * Quantising each leaf on its own leaves a small step where two leaves
 * meet, and the eye finds a grid of small steps long before PSNR does.
 * Every leaf edge is filtered the way H.264's normal filter does it: with
 * p1 p0 | q0 q1 across the edge, a step smaller than alpha between flat
 * enough sides is taken for quantisation and pulled together by
 * ((q0 - p0) * 4 + (p1 - q1) + 4) >> 3, clipped to +-tc. A larger step is
 * taken for a real edge and left alone. The thresholds scale with the
 * quantiser, since that is what sets how big a quantisation step can be.
 *
 * It runs after decoding, on whichever layer is shown, and for a sequence
 * that output is what the next frame predicts from, so it improves the
 * reference too. Vertical edges first, then horizontal, in integers; edges
 * are 4 px apart at least, so no pixel is read by one edge and written by
 * another within a pass. The header flag turns it on.
 */
/* alpha, beta and tc are step * 20/16, 6/16 and 3/16. Swept on the six
 * samples at q 24 and 48: +0.24 and +0.28 dB on average, where a filter
 * twice as strong starts to cost PSNR. */
#define DB_ALPHA 20
#define DB_BETA   6
#define DB_TC     3
static int db_param(int step, int k16) { return (step * k16 + 8) >> 4; }

/*
 * Which edges are filtered. Only those with texture on at least one side:
 * between two flat leaves the step is their two colours, and a colour's
 * quantisation error in pixels is step / n, a fraction of a level for any
 * leaf larger than 4. That step is the picture, not the codec. Filtering
 * it anyway cost the `blocos` probe 6.7 dB, and skipping it costs the
 * photographs nothing measurable (34.40 dB against 34.41). Scaling the
 * thresholds by leaf size instead was tried and lost 0.08 dB on the
 * photographs.
 */
static int edge_filtered(int ca, int cb) { return ((ca | cb) & 0x80) != 0; }

/* The step at an edge is that of the tile its second side lies in. */
static void deblock_plane(uint8_t* p, int pw, int ph, const uint8_t* cell,
                          const uint8_t* vedge, const uint8_t* hedge, int base,
                          const int8_t* tq, int tile, int tiles_x) {
    int gw = pw / 4, gh = ph / 4;
    for (int pass = 0; pass < 2; pass++)
        for (int gy = pass; gy < gh; gy++)
            for (int gx = 1 - pass; gx < gw; gx++) {
                const uint8_t* edge = pass ? hedge : vedge;
                if (!edge[gy * gw + gx]) continue;
                int other = pass ? (gy - 1) * gw + gx : gy * gw + gx - 1;
                if (!edge_filtered(cell[gy * gw + gx], cell[other])) continue;
                int step = tq ? tq_step(base, tq[(gy * 4 / tile) * tiles_x + gx * 4 / tile]) : base;
                int alpha = db_param(step, DB_ALPHA), beta = db_param(step, DB_BETA);
                int tc = db_param(step, DB_TC);
                int along = pass ? 1 : pw, across = pass ? pw : 1;
                uint8_t* r = p + (size_t)gy * 4 * pw + gx * 4;
                for (int k = 0; k < 4; k++, r += along) {
                    int p1 = r[-2 * across], p0 = r[-across], q0 = r[0], q1 = r[across];
                    if (abs(p0 - q0) >= alpha || abs(p1 - p0) >= beta || abs(q1 - q0) >= beta) continue;
                    int d = ((q0 - p0) * 4 + (p1 - q1) + 4) >> 3;
                    d = d < -tc ? -tc : (d > tc ? tc : d);
                    r[-across] = (uint8_t)clamp_u8(p0 + d);
                    r[0] = (uint8_t)clamp_u8(q0 - d);
                }
            }
}

/* Edges on a 4-px grid: every leaf's left and top side, and every tile
 * that never arrived as one block. `cell` holds, per 4x4 cell, log2 of its
 * leaf's size in cells and 0x80 when the leaf shows texture; `textured` is
 * NULL when only colours are shown. */
static int deblock(const Canvas* cv, uint8_t** planes, const Leaf* leaves, const uint8_t* textured,
                   size_t count, int first_missing_tile, int tiles_x, int tiles, const int* step,
                   const int8_t* tq) {
    int gw = cv->pw / 4, gh = cv->ph / 4;
    uint8_t* vedge = (uint8_t*)calloc((size_t)gw * gh, 1);
    uint8_t* hedge = (uint8_t*)calloc((size_t)gw * gh, 1);
    uint8_t* cell = (uint8_t*)calloc((size_t)gw * gh, 1);
    if (!vedge || !hedge || !cell) { free(vedge); free(hedge); free(cell); return -1; }
    for (size_t i = 0; i < count; i++) {
        int x = leaves[i].x / 4, y = leaves[i].y / 4, n = leaves[i].n / 4;
        int info = log2_int(n) | (textured && textured[i] ? 0x80 : 0);
        for (int j = y; j < y + n && j < gh; j++)
            for (int k = x; k < x + n && k < gw; k++) cell[(size_t)j * gw + k] = (uint8_t)info;
        if (x > 0) for (int j = y; j < y + n && j < gh; j++) vedge[(size_t)j * gw + x] = 1;
        if (y > 0) for (int k = x; k < x + n && k < gw; k++) hedge[(size_t)y * gw + k] = 1;
    }
    for (int t = first_missing_tile; t < tiles; t++) {
        int x = (t % tiles_x) * cv->tile / 4, y = (t / tiles_x) * cv->tile / 4, n = cv->tile / 4;
        for (int j = y; j < y + n && j < gh; j++)
            for (int k = x; k < x + n && k < gw; k++) cell[(size_t)j * gw + k] = (uint8_t)log2_int(n);
        if (x > 0) for (int j = y; j < y + n && j < gh; j++) vedge[(size_t)j * gw + x] = 1;
        if (y > 0) for (int k = x; k < x + n && k < gw; k++) hedge[(size_t)y * gw + k] = 1;
    }
    for (int c = 0; c < cv->np; c++)
        deblock_plane(planes[c], cv->pw, cv->ph, cell, vedge, hedge, step[cv->comp0 + c], tq, cv->tile, tiles_x);
    free(vedge); free(hedge); free(cell);
    return 0;
}

int nvdr_decode_mem(const uint8_t* data, size_t size, int max_layer,
                    NvdrImage* out, NvdrHeader* hdr_out, NvdrDecodeInfo* info) {
    return nvdr_decode_mem_ctx(data, size, max_layer, out, hdr_out, info, NULL);
}

/* decode_once() when a tile of a layer that arrived whole stops, which
 * only damage makes it do: the decode starts over, saving tiles. */
#define DECODE_AGAIN (-2)

static int decode_once(const uint8_t* data, size_t size, int max_layer, NvdrImage* out,
                       NvdrHeader* hdr_out, NvdrDecodeInfo* info, NvdrContext* ctx, int careful);

int nvdr_decode_mem_ctx(const uint8_t* data, size_t size, int max_layer, NvdrImage* out,
                        NvdrHeader* hdr_out, NvdrDecodeInfo* info, NvdrContext* ctx) {
    /* A texture layer that arrived whole cannot stop inside a tile unless
     * the file is damaged, so the first attempt does not save every tile
     * to restore it (a tenth of the time); if one stops, the decode starts
     * over the careful way and gives what that gives. The context is only
     * written when a decode finishes, so starting over is safe. */
    int rc = decode_once(data, size, max_layer, out, hdr_out, info, ctx, 0);
    if (rc == DECODE_AGAIN) rc = decode_once(data, size, max_layer, out, hdr_out, info, ctx, 1);
    return rc;
}

static int decode_once(const uint8_t* data, size_t size, int max_layer, NvdrImage* out,
                       NvdrHeader* hdr_out, NvdrDecodeInfo* info, NvdrContext* ctx, int careful) {
    out->pixels = NULL; out->width = out->height = 0;
    tables_init();
    NvdrHeader h;
    if (read_header(data, size, &h) != 0) return -1;
    if (hdr_out) *hdr_out = h;

    size_t base = NVDR_HEADER_SIZE + (size_t)h.grain_len;
    size_t avail0 = size - base;
    if (avail0 > h.stored_bytes[0]) avail0 = h.stored_bytes[0];
    /* The range decoder primes itself with five bytes; fewer than that
     * and not one symbol of layer 0 can be read. */
    if (avail0 < 5) return -1;
    size_t off1 = base + (size_t)h.stored_bytes[0];
    size_t avail1 = size > off1 ? size - off1 : 0;
    if (avail1 > h.stored_bytes[1]) avail1 = h.stored_bytes[1];
    size_t off2 = off1 + (size_t)h.stored_bytes[1];
    size_t avail2 = size > off2 ? size - off2 : 0;
    if (avail2 > h.stored_bytes[2]) avail2 = h.stored_bytes[2];
    int band_at[NSIZES];
    for (int s = 0; s < NSIZES; s++) band_at[s] = band_split(s, h.band);

    /* One tree in 4:4:4; in 4:2:0 luma's, then colour's at half size,
     * tile by tile in the same streams. A tile is whole when both are. */
    int np = (h.flags & NVDR_FLAG_CHROMA420) ? 2 : 1;
    Canvas cvs[2];
    Layer0 L[2];
    size_t* tile_start[2] = { NULL, NULL };
    uint8_t* textured[2] = { NULL, NULL };
    int rc = -1;
    memset(cvs, 0, sizeof(cvs));
    memset(L, 0, sizeof(L));
    uint8_t* up[2] = { NULL, NULL };
    uint8_t* saved = NULL;
    int8_t* tq = NULL;
    if (canvas_init(&cvs[0], h.width, h.height, h.max_block, h.min_block, np == 2 ? 1 : 3, 0) != 0) goto done;
    if (np == 2 && canvas_init(&cvs[1], (h.width + 1) / 2, (h.height + 1) / 2, h.max_block / 2,
                               NVDR_MIN_BLOCK, 2, 1) != 0) goto done;
    Canvas* cv = &cvs[0];
    for (int k = 0; k < np; k++) cvs[k].fixed_pred = (h.flags & NVDR_FLAG_RESIDUAL) != 0;
    int tiles_x = (cv->pw + cv->tile - 1) / cv->tile, tiles_y = (cv->ph + cv->tile - 1) / cv->tile;
    int tiles = tiles_x * tiles_y;
    for (int k = 0; k < np; k++) {
        tile_start[k] = (size_t*)malloc(sizeof(size_t) * (tiles + 1));
        if (!tile_start[k]) goto done;
    }
    int step[3] = { h.q_luma, h.q_chroma, h.q_chroma };
    int tstep[3] = { step[0], step[1], step[2] };
    /* The tiles' step offsets, as layer 0 delivers them; zero where it
     * never did. */
    if ((h.flags & NVDR_FLAG_TILEQ) && !(tq = (int8_t*)calloc((size_t)tiles, 1))) goto done;

    /* Layer 0: every tile that arrives whole; the rest predicted. */
    ColourModels cm;
    TextureModels tms[2];
    if (ctx && ctx->valid) {
        cm = ctx->cm; tms[0] = ctx->tm; tms[1] = ctx->tm2;
    } else {
        models_fill((uint16_t*)&cm, sizeof(cm) / sizeof(uint16_t));
        for (int k = 0; k < 2; k++) models_fill((uint16_t*)&tms[k], sizeof(TextureModels) / sizeof(uint16_t));
    }
    NvdrDecoder d0;
    nvdr_dec_init(&d0, data + base, avail0);
    for (int k = 0; k < np; k++) { L[k].cv = &cvs[k]; L[k].d = &d0; L[k].cm = &cm; L[k].step = tstep; }
    int complete0 = 0, stopped = 0, tq_prev = 0;
    for (int t = 0; t < tiles; t++) {
        int tx = (t % tiles_x) * cv->tile, ty = (t / tiles_x) * cv->tile;
        for (int k = 0; k < np; k++) tile_start[k][t] = L[k].count;
        if (tq && !stopped) {
            int d = tq_prev + get_tq(&d0, &cm);
            if (d0.overrun || d < -TQ_MAX || d > TQ_MAX) stopped = 1;
            else {
                tq[t] = (int8_t)d; tq_prev = d;
                for (int c = 0; c < 3; c++) tstep[c] = tq_step(step[c], d);
            }
        }
        for (int k = 0; k < np && !stopped; k++) {
            int x = tx / (k + 1), y = ty / (k + 1);
            if (!node_exists(&cvs[k], x, y)) continue;
            if (read_node(&L[k], x, y, cvs[k].tile) != 0) goto done;
            if (d0.overrun || L[k].corrupt) stopped = 1;
        }
        if (stopped) {
            for (int k = 0; k < np; k++) {
                L[k].count = tile_start[k][t];
                tile_fallback(&cvs[k], tx / (k + 1), ty / (k + 1));
            }
        } else complete0++;
    }
    for (int k = 0; k < np; k++) tile_start[k][tiles] = L[k].count;

    /* Layers 1 and 2: the low and the high band of every leaf's texture,
     * each as far as its own bytes reach and no further than the layer
     * before it. A tile cut short is restored to what it showed before
     * its band arrived. */
    int complete[NVDR_LAYERS] = { complete0, 0, 0 };
    for (int k = 0; k < np; k++) {
        textured[k] = (uint8_t*)calloc(L[k].count ? L[k].count : 1, 1);
        if (!textured[k]) goto done;
    }
    saved = (uint8_t*)malloc((size_t)cv->tile * cv->tile * 9);   /* full and acc, 3 planes */
    if (!saved) goto done;
    for (int layer = 1; layer < NVDR_LAYERS; layer++) {
        const uint8_t* stream = data + (layer == 1 ? off1 : off2);
        size_t avail = layer == 1 ? avail1 : avail2;
        if ((max_layer >= 0 && max_layer < layer) || avail < 5 || complete[layer - 1] == 0) break;
        if (layer == 2 && h.band == 0) break;
        TextureModels* tm = &tms[layer - 1];
        NvdrDecoder d;
        nvdr_dec_init(&d, stream, avail);
        int corrupt = 0;
        int lv[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
        int guard = careful || avail < h.stored_bytes[layer];
        for (int t = 0; t < complete[layer - 1]; t++) {
            /* Every plane of both trees saved, luma's then colour's, in
             * one buffer: a colour tile is a quarter of a luma tile. */
            uint8_t* keep = saved;
            uint8_t was[2][1024];   /* a tile holds at most (32 / 4)^2 leaves */
            for (int k = 0; k < np && guard; k++) {
                Canvas* pc = &cvs[k];
                int tx = (t % tiles_x) * pc->tile, ty = (t / tiles_x) * pc->tile;
                if (!node_exists(pc, tx, ty)) continue;
                int tw = tx + pc->tile < pc->pw ? pc->tile : pc->pw - tx;
                int th = ty + pc->tile < pc->ph ? pc->tile : pc->ph - ty;
                for (int c = 0; c < pc->np; c++)
                    for (int j = 0; j < th; j++) {
                        size_t at = (size_t)(ty + j) * pc->pw + tx;
                        memcpy(keep, pc->full[c] + at, (size_t)tw); keep += tw;
                        memcpy(keep, pc->acc[c] + at, sizeof(int16_t) * (size_t)tw); keep += 2 * tw;
                    }
                size_t first = tile_start[k][t], nl = tile_start[k][t + 1] - first;
                for (size_t i = 0; i < nl && i < sizeof(was[k]); i++) was[k][i] = textured[k][first + i];
            }
            for (int k = 0; k < np && !d.overrun && !corrupt; k++) {
                Canvas* pc = &cvs[k];
                int tile_step[3] = { step[0], step[1], step[2] };
                if (tq) for (int c = 0; c < 3; c++) tile_step[c] = tq_step(step[c], tq[t]);
                for (size_t i = tile_start[k][t]; i < tile_start[k][t + 1] && !d.overrun && !corrupt; i++) {
                    const Leaf* f = &L[k].leaves[i];
                    int sc = size_class(f->n), count = f->n * f->n;
                    int start = layer == 1 ? 1 : band_at[sc], end = layer == 1 ? band_at[sc] : count;
                    if (start >= end) continue;
                    for (int c = 0; c < pc->np && !d.overrun && !corrupt; c++) {
                        int comp = pc->comp0 + c;
                        if (get_texture(&d, tm, sc, comp, lv, start, end, &corrupt) && !d.overrun && !corrupt) {
                            apply_texture(pc, c, f->x, f->y, f->n, lv, start, end, tile_step[comp]);
                            textured[k][i] = 1;
                        }
                    }
                }
            }
            if (d.overrun || corrupt) {
                if (!guard) { rc = DECODE_AGAIN; goto done; }
                const uint8_t* back = saved;
                for (int k = 0; k < np; k++) {
                    Canvas* pc = &cvs[k];
                    int tx = (t % tiles_x) * pc->tile, ty = (t / tiles_x) * pc->tile;
                    if (!node_exists(pc, tx, ty)) continue;
                    int tw = tx + pc->tile < pc->pw ? pc->tile : pc->pw - tx;
                    int th = ty + pc->tile < pc->ph ? pc->tile : pc->ph - ty;
                    for (int c = 0; c < pc->np; c++)
                        for (int j = 0; j < th; j++) {
                            size_t at = (size_t)(ty + j) * pc->pw + tx;
                            memcpy(pc->full[c] + at, back, (size_t)tw); back += tw;
                            memcpy(pc->acc[c] + at, back, sizeof(int16_t) * (size_t)tw); back += 2 * tw;
                        }
                    size_t first = tile_start[k][t], nl = tile_start[k][t + 1] - first;
                    for (size_t i = 0; i < nl && i < sizeof(was[k]); i++) textured[k][first + i] = was[k][i];
                }
                break;
            }
            complete[layer]++;
        }
    }

    out->pixels = (unsigned char*)malloc((size_t)h.width * h.height * 3);
    if (!out->pixels) goto done;
    out->width = h.width; out->height = h.height;
    uint8_t* planes[3];
    for (int k = 0; k < np; k++) {
        uint8_t** pl = (max_layer == 0) ? cvs[k].flat : cvs[k].full;
        if ((h.flags & NVDR_FLAG_DEBLOCK) &&
            deblock(&cvs[k], pl, L[k].leaves, max_layer == 0 ? NULL : textured[k], tile_start[k][complete0],
                    complete0, tiles_x, tiles, step, tq) != 0) {
            free(out->pixels); out->pixels = NULL; goto done;
        }
        for (int c = 0; c < cvs[k].np; c++) planes[cvs[k].comp0 + c] = pl[c];
    }
    if (np == 2) {
        /* Colour back to full size (see upsample_plane()). */
        for (int c = 0; c < 2; c++) {
            up[c] = (uint8_t*)malloc((size_t)cv->pw * cv->ph);
            if (!up[c]) { free(out->pixels); out->pixels = NULL; goto done; }
            upsample_plane(planes[1 + c], &cvs[1], up[c], cv->pw, h.width, h.height);
            planes[1 + c] = up[c];
        }
    }
    if (h.flags & NVDR_FLAG_GRAIN) {
        NvdrGrainTemplates* gt = (NvdrGrainTemplates*)malloc(sizeof(NvdrGrainTemplates));
        if (!gt) { free(out->pixels); out->pixels = NULL; goto done; }
        nvdr_grain_templates(&h.grain, gt);
        nvdr_grain_apply(&h.grain, gt, planes[0], planes[1], planes[2], cv->pw, h.width, h.height);
        free(gt);
    }
    for (int y = 0; y < h.height; y++)
        for (int x = 0; x < h.width; x++) {
            size_t at = (size_t)y * cv->pw + x;
            ycc_to_rgb(planes[0][at], planes[1][at], planes[2][at],
                       out->pixels + ((size_t)y * h.width + x) * 3);
        }
    if (info) {
        info->tiles = tiles;
        for (int k = 0; k < NVDR_LAYERS; k++) info->tiles_complete[k] = complete[k];
        info->layers_present = 1;
        if (complete0 == tiles && avail1 >= 5) info->layers_present = 2;
        if (h.band && complete[1] == tiles && avail2 >= 5) info->layers_present = 3;
    }
    if (ctx) {
        /* Only a container decoded whole leaves the models where the
         * encoder left them; anything less and the next one cannot be
         * decoded from this context. */
        int whole = complete[0] == tiles && complete[1] == tiles &&
                    (h.band == 0 || complete[2] == tiles) && (max_layer < 0 || max_layer >= NVDR_LAYERS - 1);
        if (whole) { ctx->cm = cm; ctx->tm = tms[0]; ctx->tm2 = tms[1]; ctx->valid = 1; }
        else ctx->valid = 0;
    }
    rc = 0;

done:
    free(saved);
    free(tq);
    for (int k = 0; k < 2; k++) {
        free(L[k].leaves);
        free(tile_start[k]);
        free(textured[k]);
        free(up[k]);
        canvas_free(&cvs[k]);
    }
    return rc;
}

#ifndef NVDR_WASM
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
#endif
