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
    int fixed_pred;              /* NVDR_FLAG_RESIDUAL: every colour predicted as 128 */
    uint8_t* flat[3];
    uint8_t* full[3];
    /* The texture added so far, unclamped (to 16 bits): `full` is
     * clamp(flat + acc). The bands are added here and clamped only when
     * shown, so a high band can pull back what the low band overshot. */
    int16_t* acc[3];
} Canvas;

static int canvas_init(Canvas* cv, int w, int h, int tile, int min_block) {
    memset(cv, 0, sizeof(*cv));
    cv->w = w; cv->h = h; cv->tile = tile; cv->min_block = min_block;
    cv->pw = (w + min_block - 1) / min_block * min_block;
    cv->ph = (h + min_block - 1) / min_block * min_block;
    for (int c = 0; c < 3; c++) {
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
    int w = tx + cv->tile < cv->pw ? cv->tile : cv->pw - tx;
    int h = ty + cv->tile < cv->ph ? cv->tile : cv->ph - ty;
    for (int c = 0; c < 3; c++) {
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
    if (h->flags & ~(NVDR_FLAG_RESIDUAL | NVDR_FLAG_DEBLOCK)) return -1;   /* a flag this decoder does not know */
    for (int k = 0; k < NVDR_LAYERS; k++) if (h->stored_bytes[k] > 0x7fffffffu) return -1;
    if (h->band > 32) return -1;
    return 0;
}

/* ============================================================= encoder */

typedef struct {
    Canvas         cv;
    double*        src[3];      /* YCbCr, padded by replication */
    int            step[3];
    double         deadzone, lambda;
    double         skip_lambda;  /* 0: every leaf coded; see code_leaf() */
    ColourModels   cm;
    TextureModels  tm;           /* the low band */
    TextureModels  tm2;          /* the high band */
    int            band_at[NSIZES];   /* first high-band scan position per size */
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
    double         plane_bits[3];   /* what the coded leaves cost, per plane */
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
/*
 * A leaf's quantised colours and textures, reconstructed into the canvas
 * as the decoder will see them. With `skip` set nothing is corrected: the
 * colour is the prediction and there is no texture. Returns the squared
 * error over the pixels inside the image.
 */
static double leaf_levels(Enc* e, int x, int y, int n, int skip,
                          int dl[3], int lv[3][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK], int* textured) {
    Canvas* cv = &e->cv;
    int sc = size_class(n), count = n * n;
    double err = 0.0;
    int any = 0;
    for (int c = 0; c < 3; c++) {
        int step = e->step[c];
        int pred = predict(cv, c, x, y, n);
        dl[c] = 0;
        size_t b = (size_t)((y - e->pre_y) / n) * (cv->pw / n) + (size_t)(x / n);
        if (!skip) {
            double sum = e->pre_sum[sc][c][b];
            /* The orthonormal DC of (block - prediction) is n * its mean. */
            double dc = (sum / count - pred) * n;
            dl[c] = quantise(dc / step, 0.0);
        }
        int colour = clamp_u8(pred + div_round(clamp_coef((long)dl[c] * step), n));
        paint_flat(cv, c, x, y, n, n, colour);

        memset(lv[c], 0, sizeof(int) * count);
        int nonzero = 0;
        if (!skip) {
            memcpy(lv[c], e->pre[sc][c] + b * count, sizeof(int) * (size_t)count);
            nonzero = e->pre_nz[sc][c][b];
        }
        if (nonzero) {
            add_residual(cv, c, x, y, n, e->pre_lo[sc][c] + b * count);
            if (e->band_at[sc] < count) add_residual(cv, c, x, y, n, e->pre_hi[sc][c] + b * count);
            any = 1;
        }

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

/* s1 is the two texture layers' sinks, low band then high. */
static void leaf_emit(Enc* e, Sink* s0, Sink* s1, int n,
                      const int dl[3], int lv[3][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK]) {
    int sc = size_class(n), count = n * n, at = e->band_at[sc];
    for (int c = 0; c < 3; c++) {
        double before = s0->bits + s1[0].bits + s1[1].bits;
        put_dc(s0, &e->cm, sc, c, dl[c]);
        put_texture(&s1[0], &e->tm, sc, c, lv[c], 1, at);
        if (at < count) put_texture(&s1[1], &e->tm2, sc, c, lv[c], at, count);
        if (s0->enc) e->plane_bits[c] += s0->bits + s1[0].bits + s1[1].bits - before;
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
    if (e->skip_lambda > 0.0) {
        Sink c0 = { NULL, 0 }, c1[2] = { { NULL, 0 }, { NULL, 0 } };
        double d_skip = leaf_levels(e, x, y, n, 1, dl, lv, NULL);
        leaf_emit(e, &c0, c1, n, dl, lv);
        double j_skip = d_skip + e->skip_lambda * (c0.bits + c1[0].bits + c1[1].bits);
        Sink k0 = { NULL, 0 }, k1[2] = { { NULL, 0 }, { NULL, 0 } };
        int tex = 0;
        double d_code = leaf_levels(e, x, y, n, 0, dl, lv, &tex);
        leaf_emit(e, &k0, k1, n, dl, lv);
        double j_code = d_code + e->skip_lambda * (k0.bits + k1[0].bits + k1[1].bits);
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
    for (int c = 0; c < 3; c++)
        for (int j = 0; j < n; j++) {
            size_t at = (size_t)(y + j) * cv->pw + x;
            memcpy(buf, cv->flat[c] + at, (size_t)n); buf += n;
            memcpy(buf, cv->full[c] + at, (size_t)n); buf += n;
            memcpy(buf, cv->acc[c] + at, sizeof(int16_t) * (size_t)n); buf += 2 * n;
        }
}

static void load_block(Canvas* cv, int x, int y, int n, const uint8_t* buf) {
    for (int c = 0; c < 3; c++)
        for (int j = 0; j < n; j++) {
            size_t at = (size_t)(y + j) * cv->pw + x;
            memcpy(cv->flat[c] + at, buf, (size_t)n); buf += n;
            memcpy(cv->full[c] + at, buf, (size_t)n); buf += n;
            memcpy(cv->acc[c] + at, buf, sizeof(int16_t) * (size_t)n); buf += 2 * n;
        }
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
    Canvas* cv = &e->cv;
    e->pre_y = ty;
    for (int s = 0; s < NSIZES; s++) {
        int n = NVDR_MIN_BLOCK << s, count = n * n;
        if (!e->pre[s][0]) continue;
        int gw = cv->pw / n, gh = cv->tile / n, blocks = gw * gh;
        for (int c = 0; c < 3; c++) {
            int* dst = e->pre[s][c];
            const double* src = e->src[c];
            int step = e->step[c];
            double dz = e->deadzone;
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < blocks; b++) {
                int x = (b % gw) * n, y = ty + (b / gw) * n;
                if (y + n > cv->ph) continue;
                double blk[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK], co[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
                for (int j = 0; j < n; j++)
                    for (int i = 0; i < n; i++) blk[j * n + i] = src[(size_t)(y + j) * cv->pw + x + i];
                double sum = 0.0;
                for (int j = 0; j < n; j++)
                    for (int i = 0; i < n; i++) sum += blk[j * n + i];
                e->pre_sum[s][c][b] = sum;
                forward_dct(s, blk, co);
                int* q = dst + (size_t)b * count;
                int nonzero = 0;
                q[0] = 0;
                for (int i = 1; i < count; i++) {
                    q[i] = quantise(co[scan_pos[s][i]] / step, dz);
                    nonzero |= q[i];
                }
                e->pre_nz[s][c][b] = (uint8_t)(nonzero != 0);
                if (nonzero) {
                    int at = e->band_at[s];
                    texture_residual(s, q, 1, at, step, e->pre_lo[s][c] + (size_t)b * count);
                    if (at < count) texture_residual(s, q, at, count, step, e->pre_hi[s][c] + (size_t)b * count);
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
    Canvas* cv = &e->cv;
    size_t id = node_id(e, x, y, n);
    int whole = node_whole(cv, x, y, n);
    int can_split = n > cv->min_block;
    double whole_cost = 1e300, split_cost = 0.0;
    uint8_t* kept = NULL;

    if (whole) {
        Sink s0 = { NULL, 0 }, s1[2] = { { NULL, 0 }, { NULL, 0 } };
        if (can_split) put_bit(&s0, &e->cm.split[size_class(n)], 0);
        double d = code_leaf(e, &s0, s1, x, y, n, NULL);
        whole_cost = d + e->lambda * (s0.bits + s1[0].bits + s1[1].bits);
        if (!can_split) { e->split[id] = 0; return whole_cost; }
        kept = (uint8_t*)malloc((size_t)12 * n * n);
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
    return nvdr_encode_mem_ctx(out_buf, out_len, img, cfg_in, hdr_out, NULL);
}

int nvdr_encode_mem_ctx(uint8_t** out_buf, size_t* out_len, const NvdrImage* img,
                        const NvdrConfig* cfg_in, NvdrHeader* hdr_out, NvdrContext* ctx) {
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
    NvdrEncoder enc0, enc1, enc2;
    memset(&enc0, 0, sizeof(enc0));
    memset(&enc1, 0, sizeof(enc1));
    memset(&enc2, 0, sizeof(enc2));
    int band = cfg.band < 0 ? 0 : (cfg.band > 32 ? 32 : cfg.band);
    NvdrHeader h;
    memset(&h, 0, sizeof(h));

    if (canvas_init(&e.cv, img->width, img->height, cfg.max_block, cfg.min_block) != 0) goto done;
    Canvas* cv = &e.cv;
    cv->fixed_pred = cfg.residual != 0;
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
    e.skip_lambda = cfg.residual ? cfg.skip_k * e.lambda : 0.0;
    for (int s = 0; s < NSIZES; s++) e.band_at[s] = band_split(s, band);

    size_t nodes = 0;
    for (int s = 0; s < NSIZES; s++) {
        int n = NVDR_MIN_BLOCK << s;
        e.grid_base[s] = nodes;
        e.grid_w[s] = (cv->pw + n - 1) / n;
        nodes += (size_t)e.grid_w[s] * ((cv->ph + n - 1) / n);
    }
    e.split = (int*)calloc(nodes, sizeof(int));
    if (!e.split) goto done;
    for (int s = 0; s < NSIZES; s++) {
        int n = NVDR_MIN_BLOCK << s;
        if (n < cv->min_block || n > cv->tile) continue;
        for (int c = 0; c < 3; c++) {
            size_t blocks = (size_t)(cv->pw / n) * (cv->tile / n);
            if (!blocks) blocks = 1;
            e.pre[s][c] = (int*)malloc(sizeof(int) * blocks * n * n);
            e.pre_lo[s][c] = (int*)malloc(sizeof(int) * blocks * n * n);
            e.pre_hi[s][c] = (int*)malloc(sizeof(int) * blocks * n * n);
            e.pre_sum[s][c] = (double*)malloc(sizeof(double) * blocks);
            e.pre_nz[s][c] = (uint8_t*)malloc(blocks);
            if (!e.pre[s][c] || !e.pre_lo[s][c] || !e.pre_hi[s][c] || !e.pre_sum[s][c] || !e.pre_nz[s][c])
                goto done;
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
     * for real, which is what adapts them for the next. */
    for (int ty = 0; ty < cv->ph; ty += cv->tile) {
        precompute_row(&e, ty);
        for (int tx = 0; tx < cv->pw; tx += cv->tile) {
            search(&e, tx, ty, cv->tile);
            emit(&e, &s0, s1, tx, ty, cv->tile, &h);
        }
    }
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
    buf[5] = (uint8_t)((cfg.residual ? NVDR_FLAG_RESIDUAL : 0) | (cfg.deblock ? NVDR_FLAG_DEBLOCK : 0));
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
    nvdr_enc_free(&enc0);
    nvdr_enc_free(&enc1);
    nvdr_enc_free(&enc2);
    for (int c = 0; c < 3; c++) free(e.src[c]);
    free(e.split);
    for (int s = 0; s < NSIZES; s++)
        for (int c = 0; c < 3; c++) {
            free(e.pre[s][c]); free(e.pre_lo[s][c]); free(e.pre_hi[s][c]);
            free(e.pre_sum[s][c]); free(e.pre_nz[s][c]);
        }
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

static void deblock_plane(uint8_t* p, int pw, int ph, const uint8_t* cell,
                          const uint8_t* vedge, const uint8_t* hedge, int step) {
    int gw = pw / 4, gh = ph / 4;
    for (int pass = 0; pass < 2; pass++)
        for (int gy = pass; gy < gh; gy++)
            for (int gx = 1 - pass; gx < gw; gx++) {
                const uint8_t* edge = pass ? hedge : vedge;
                if (!edge[gy * gw + gx]) continue;
                int other = pass ? (gy - 1) * gw + gx : gy * gw + gx - 1;
                if (!edge_filtered(cell[gy * gw + gx], cell[other])) continue;
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
                   size_t count, int first_missing_tile, int tiles_x, int tiles, const int* step) {
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
    for (int c = 0; c < 3; c++) deblock_plane(planes[c], cv->pw, cv->ph, cell, vedge, hedge, step[c]);
    free(vedge); free(hedge); free(cell);
    return 0;
}

int nvdr_decode_mem(const uint8_t* data, size_t size, int max_layer,
                    NvdrImage* out, NvdrHeader* hdr_out, NvdrDecodeInfo* info) {
    return nvdr_decode_mem_ctx(data, size, max_layer, out, hdr_out, info, NULL);
}

int nvdr_decode_mem_ctx(const uint8_t* data, size_t size, int max_layer, NvdrImage* out,
                        NvdrHeader* hdr_out, NvdrDecodeInfo* info, NvdrContext* ctx) {
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
    size_t off2 = off1 + (size_t)h.stored_bytes[1];
    size_t avail2 = size > off2 ? size - off2 : 0;
    if (avail2 > h.stored_bytes[2]) avail2 = h.stored_bytes[2];
    int band_at[NSIZES];
    for (int s = 0; s < NSIZES; s++) band_at[s] = band_split(s, h.band);

    Canvas cv;
    int rc = -1;
    Layer0 L;
    memset(&L, 0, sizeof(L));
    size_t* tile_start = NULL;
    uint8_t* textured = NULL;
    if (canvas_init(&cv, h.width, h.height, h.max_block, h.min_block) != 0) goto done;
    cv.fixed_pred = (h.flags & NVDR_FLAG_RESIDUAL) != 0;
    int tiles_x = (cv.pw + cv.tile - 1) / cv.tile, tiles_y = (cv.ph + cv.tile - 1) / cv.tile;
    int tiles = tiles_x * tiles_y;
    tile_start = (size_t*)malloc(sizeof(size_t) * (tiles + 1));
    if (!tile_start) goto done;
    int step[3] = { h.q_luma, h.q_chroma, h.q_chroma };

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

    /* Layers 1 and 2: the low and the high band of every leaf's texture,
     * each as far as its own bytes reach and no further than the layer
     * before it. A tile cut short is restored to what it showed before
     * its band arrived. */
    int complete[NVDR_LAYERS] = { complete0, 0, 0 };
    uint8_t* saved = NULL;
    textured = (uint8_t*)calloc(L.count ? L.count : 1, 1);
    saved = (uint8_t*)malloc((size_t)cv.tile * cv.tile * 9);   /* full and acc, 3 channels */
    if (!textured || !saved) { free(saved); goto done; }
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
        for (int t = 0; t < complete[layer - 1]; t++) {
            int tx = (t % tiles_x) * cv.tile, ty = (t / tiles_x) * cv.tile;
            int tw = tx + cv.tile < cv.pw ? cv.tile : cv.pw - tx;
            int th = ty + cv.tile < cv.ph ? cv.tile : cv.ph - ty;
            for (int c = 0; c < 3; c++)
                for (int j = 0; j < th; j++) {
                    size_t at = (size_t)(ty + j) * cv.pw + tx;
                    memcpy(saved + ((size_t)c * cv.tile + j) * cv.tile * 3, cv.full[c] + at, (size_t)tw);
                    memcpy(saved + ((size_t)c * cv.tile + j) * cv.tile * 3 + cv.tile,
                           cv.acc[c] + at, sizeof(int16_t) * (size_t)tw);
                }
            uint8_t was[1024];   /* a tile holds at most (32 / 4)^2 leaves */
            size_t first = tile_start[t], nl = tile_start[t + 1] - first;
            for (size_t i = 0; i < nl && i < sizeof(was); i++) was[i] = textured[first + i];
            for (size_t i = first; i < tile_start[t + 1] && !d.overrun && !corrupt; i++) {
                const Leaf* f = &L.leaves[i];
                int sc = size_class(f->n), count = f->n * f->n;
                int start = layer == 1 ? 1 : band_at[sc], end = layer == 1 ? band_at[sc] : count;
                if (start >= end) continue;
                for (int c = 0; c < 3 && !d.overrun && !corrupt; c++)
                    if (get_texture(&d, tm, sc, c, lv, start, end, &corrupt) && !d.overrun && !corrupt) {
                        apply_texture(&cv, c, f->x, f->y, f->n, lv, start, end, step[c]);
                        textured[i] = 1;
                    }
            }
            if (d.overrun || corrupt) {
                for (int c = 0; c < 3; c++)
                    for (int j = 0; j < th; j++) {
                        size_t at = (size_t)(ty + j) * cv.pw + tx;
                        memcpy(cv.full[c] + at, saved + ((size_t)c * cv.tile + j) * cv.tile * 3, (size_t)tw);
                        memcpy(cv.acc[c] + at, saved + ((size_t)c * cv.tile + j) * cv.tile * 3 + cv.tile,
                               sizeof(int16_t) * (size_t)tw);
                    }
                for (size_t i = 0; i < nl && i < sizeof(was); i++) textured[first + i] = was[i];
                break;
            }
            complete[layer]++;
        }
    }
    free(saved);

    out->pixels = (unsigned char*)malloc((size_t)h.width * h.height * 3);
    if (!out->pixels) goto done;
    out->width = h.width; out->height = h.height;
    uint8_t** planes = (max_layer == 0) ? cv.flat : cv.full;
    if ((h.flags & NVDR_FLAG_DEBLOCK) &&
        deblock(&cv, planes, L.leaves, max_layer == 0 ? NULL : textured, tile_start[complete0],
                complete0, tiles_x, tiles, step) != 0) {
        free(out->pixels); out->pixels = NULL; goto done;
    }
    for (int y = 0; y < h.height; y++)
        for (int x = 0; x < h.width; x++) {
            size_t at = (size_t)y * cv.pw + x;
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
    free(L.leaves);
    free(tile_start);
    free(textured);
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
