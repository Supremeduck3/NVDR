/* Part of nvdr.c, which includes it:
 * constants, the integer transforms and their tables, the colour conversion. */

/* ============================================================ constants */

#define NSIZES     4           /* 4, 8, 16, 32 */
#define POS_CTX    15
#define NVDR_MODES 43          /* intra modes, see dir_predict() */
#define MODE_INTER NVDR_MODES  /* with a base: the base picture's block */
#define MODE_SMOOTH 35          /* AV1's smooth, smooth down, smooth across */
#define MODE_FILTER 38          /* AV1's five recursive filters */
#define SIG_CTX    25          /* 5 position classes x 5 neighbourhoods */
#define GT1_CTX    10          /* 5 neighbourhoods x (a level over 1 yet or not) */
#define MAG_UNARY  14
#define EG_LIMIT   24          /* longest Exp-Golomb prefix a decoder accepts */
#define COEF_MAX   32767
#define DB_LEVEL   8           /* the deblocking filter as tuned, see deblock() */

static int size_class(int n) { return n == 4 ? 0 : n == 8 ? 1 : n == 16 ? 2 : 3; }

/*
 * LEAF SHAPES
 * -----------
 * A leaf is square, or (container v15, directional pictures) half of a
 * square node cut across or down: 8x4, 4x8, 16x8, 8x16, 32x16 or 16x32.
 * A shape indexes the scans; its class, the squares' size or a pair of
 * halves by area, indexes the models.
 */
#define NSHAPES 10
#define NMC     7
static const uint8_t SHAPE_W[NSHAPES] = { 4, 8, 16, 32, 8, 4, 16, 8, 32, 16 };
static const uint8_t SHAPE_H[NSHAPES] = { 4, 8, 16, 32, 4, 8, 8, 16, 16, 32 };
static const uint8_t SHAPE_MC[NSHAPES] = { 0, 1, 2, 3, 4, 4, 5, 5, 6, 6 };
static int shape_of(int w, int h) {
    return w == h ? size_class(w) : 4 + 2 * size_class(w < h ? w : h) + (w < h);
}
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
    c.directional = 0;
    c.base = NULL;
    c.adaptive_tx = 1;
    c.halves = 1;
    c.tx_modes = 0;
    c.fast_rank = 0;
    /* The split decision is flat across 0.06..0.3; this is the middle. */
    c.lambda_k = 0.12f;
    c.max_block = NVDR_MAX_BLOCK;
    c.min_block = NVDR_MIN_BLOCK;
    c.residual = 0;
    c.deblock = 1;
    c.restore = 1;
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
    c.slices = 0;
    c.priors = 0;
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

/*
 * TRANSFORM TYPES
 * ---------------
 * With NVDR_FLAG_TXSEL a leaf of 4, 8 or 16 pixels that is predicted
 * along a direction (or flat, in a picture with a base; never INTER)
 * carries one of TX_TYPES separable transforms, one for all its planes:
 *
 *   0  DCT down, DCT across       what every other leaf uses
 *   1  DST-VII down, DCT across   the error grows away from the row above
 *   2  DCT down, DST-VII across   ... from the column to the left
 *   3  DST-VII both ways
 *   4  none (identity) both ways  edges and text, where a DCT rings
 *   5  identity down, DCT across  stripes that run down
 *   6  DCT down, identity across  ... and across
 *
 * The DST-VII is VVC's and AV1's ADST: its first basis function starts
 * near zero at the reference and rises away from it, which is the shape
 * of what a directional prediction misses. Its integer matrices are
 * 64 * sqrt(N) times the orthonormal one, like the DCT's, each entry one
 * of the N magnitudes below (sin(pi r / (2N + 1)) scaled, rounded, then
 * nudged by one where that brought the rows nearer orthogonal) with the
 * sign the sine gives it. The identity is 64 * sqrt(N) on the diagonal
 * (181 for 128 * sqrt(2) at 8), so every type shares the DCT's shifts.
 *
 * A leaf with a type other than 0 has no DC level: nothing but a DCT has
 * a basis function that is a constant, so the colour's correction goes
 * with the rest of the texture, which is then coded from scan position 0
 * against the same models as the DCT's (a set of their own did worse:
 * see NOTEBOOK.md, "Adaptive transforms").
 */
#define TX_DCT    0
#define TX_ADST   1
#define TX_IDTX   2
#define TX_TYPES  7
#define TX_MAX_N  16            /* larger leaves are always DCT */
static const uint8_t TX_V[TX_TYPES] = { TX_DCT, TX_ADST, TX_DCT, TX_ADST, TX_IDTX, TX_IDTX, TX_DCT };
static const uint8_t TX_H[TX_TYPES] = { TX_DCT, TX_DCT, TX_ADST, TX_ADST, TX_IDTX, TX_DCT, TX_IDTX };
static const int ADST4[4] = { 29, 55, 74, 84 };
static const int ADST8[8] = { 16, 32, 46, 59, 70, 79, 85, 87 };
static const int ADST16[16] = { 8, 16, 25, 33, 41, 48, 55, 62, 68, 73, 77, 81, 84, 87, 88, 88 };

static int tmat[3][NSIZES][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];   /* [kind][size][k * n + x] */
/* The same as doubles, and transposed, for forward_tx(). */
static double tmat_d[3][NSIZES][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];    /* [k * n + x] */
static double tmat_t[3][NSIZES][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];    /* [x * n + k] */
static int scan_pos[NSHAPES][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
static uint8_t scan_ctx[NSHAPES][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
static uint8_t scan_diag[NSHAPES][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];   /* u + v */
static int16_t scan_idx[NSHAPES][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];    /* raster -> scan */
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
static void tables_build(void) {
    for (int s = 0; s < NSIZES; s++) {
        int n = NVDR_MIN_BLOCK << s, unit = NVDR_MAX_BLOCK / n;
        for (int k = 0; k < n; k++)
            for (int x = 0; x < n; x++)
                tmat[TX_DCT][s][k * n + x] = k ? cos_entry((2 * x + 1) * k * unit) : 64;
        if (n <= TX_MAX_N) {
            /* Entry (k, x) is sin(pi (2k + 1)(x + 1) / M), M = 2n + 1:
             * fold the angle into 0..pi, then onto 0..pi/2. */
            const int* mag = n == 4 ? ADST4 : n == 8 ? ADST8 : ADST16;
            int m2 = 2 * (2 * n + 1), id = n == 4 ? 128 : n == 8 ? 181 : 256;
            for (int k = 0; k < n; k++)
                for (int x = 0; x < n; x++) {
                    int a = (2 * k + 1) * (x + 1) % m2, sign = 1;
                    if (a > m2 / 2) { a -= m2 / 2; sign = -1; }
                    int r = a < m2 / 2 - a ? a : m2 / 2 - a;
                    tmat[TX_ADST][s][k * n + x] = r ? sign * mag[r - 1] : 0;
                    tmat[TX_IDTX][s][k * n + x] = k == x ? id : 0;
                }
        }
        for (int t = 0; t < 3; t++)
            for (int k = 0; k < n; k++)
                for (int x = 0; x < n; x++) {
                    tmat_d[t][s][k * n + x] = tmat[t][s][k * n + x];
                    tmat_t[t][s][x * n + k] = tmat[t][s][k * n + x];
                }
    }
    /* Diagonal scan, low frequencies first; the context of a position is
     * its diagonal, fine at the low end where most coefficients live and
     * coarser past it. A position is v * w + u, u across. */
    for (int s = 0; s < NSHAPES; s++) {
        int w = SHAPE_W[s], h = SHAPE_H[s], at = 0;
        for (int d = 0; d <= w + h - 2; d++)
            for (int v = 0; v < h; v++) {
                int u = d - v;
                if (u < 0 || u >= w) continue;
                scan_pos[s][at] = v * w + u;
                scan_idx[s][v * w + u] = (int16_t)at;
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
}

/* Several pictures may be coded at once (nvdr_batch codes one per core):
 * the tables are built by the first, and the others wait for them. */
static void tables_init(void) {
    static volatile int done = 0;
    if (done) return;
#pragma omp critical(nvdr_tables)
    if (!done) { tables_build(); done = 1; }
}

/* Encoder only: orthonormal coefficients of a block of shape `s`, w
 * across and h down, under transform type `tx` (0, the DCT, for every
 * leaf without a choice): the h-point transform down, the w-point one
 * across. */
/*
 * The seven transform types are three one-dimensional transforms, one per
 * axis, and a leaf tries them all on the same block: so the pass across
 * the rows is kept, per kind, for the last block seen, and the identity
 * is a scaling rather than a product with a matrix of zeros. Each sum
 * still runs over its terms in the order of a plain row-times-column
 * product (the loops are nested so the innermost walks contiguous memory,
 * which the compiler vectorises), and the identity's skipped terms are
 * exact zeros: the coefficients are the same to the last bit.
 */
typedef struct {
    int    s;                                   /* shape of the block held, -1 for none */
    double in[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    double rows[3][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    int    ok[3];
} RowPasses;
/* One block per plane: a leaf's types are tried plane after plane. */
#define ROW_CACHE 4
static _Thread_local RowPasses row_cache[ROW_CACHE] = { { .s = -1 }, { .s = -1 }, { .s = -1 }, { .s = -1 } };
static _Thread_local int row_next;

static void forward_tx(int s, int tx, const double* in, double* out) {
    int w = SHAPE_W[s], h = SHAPE_H[s], n = w * h, kh = TX_H[tx], kv = TX_V[tx];
    RowPasses* rc = NULL;
    for (int i = 0; i < ROW_CACHE && !rc; i++)
        if (row_cache[i].s == s && !memcmp(row_cache[i].in, in, sizeof(double) * (size_t)n)) rc = &row_cache[i];
    if (!rc) {
        rc = &row_cache[row_next];
        row_next = (row_next + 1) % ROW_CACHE;
        rc->s = s;
        memcpy(rc->in, in, sizeof(double) * (size_t)n);
        rc->ok[0] = rc->ok[1] = rc->ok[2] = 0;
    }
    double* tmp = rc->rows[kh];
    if (!rc->ok[kh]) {
        if (kh == TX_IDTX) {
            double id = tmat_d[TX_IDTX][size_class(w)][0];
            for (int i = 0; i < n; i++) tmp[i] = id * in[i];
        } else {
            const double* tt = tmat_t[kh][size_class(w)];
            for (int y = 0; y < h; y++) {
                double* row = tmp + y * w;
                for (int u = 0; u < w; u++) row[u] = 0.0;
                for (int x = 0; x < w; x++) {
                    double p = in[y * w + x];
                    const double* col = tt + x * w;
                    for (int u = 0; u < w; u++) row[u] += col[u] * p;
                }
            }
        }
        rc->ok[kh] = 1;
    }
    double scale = 1.0 / (4096.0 * sqrt((double)w * h));
    if (kv == TX_IDTX) {
        double id = tmat_d[TX_IDTX][size_class(h)][0];
        for (int i = 0; i < n; i++) out[i] = id * tmp[i] * scale;
        return;
    }
    const double* t = tmat_d[kv][size_class(h)];
    for (int v = 0; v < h; v++) {
        double* row = out + v * w;
        for (int u = 0; u < w; u++) row[u] = 0.0;
        for (int y = 0; y < h; y++) {
            double k = t[v * h + y];
            const double* src = tmp + y * w;
            for (int u = 0; u < w; u++) row[u] += k * src[u];
        }
        for (int u = 0; u < w; u++) row[u] *= scale;
    }
}

/*
 * The decoder's inverse, and the encoder's too, so both reconstruct the
 * same pixels. Columns first with a shift of 6 and a clip to 16 bits,
 * then rows with a shift of 6 + log2(n): 4096 * n in all. The clip only
 * bites on damaged input, and it is what keeps every product inside 32
 * bits: 32767 * 90 * 32 is under 2^27. A half leaf, w by h with one twice
 * the other, needs 4096 * sqrt(w h) = 4096 * sqrt(2) * min(w, h): the
 * columns are also scaled by 181 / 256, sqrt(2) / 2 to within 0.01%, and
 * the rows shifted by 6 + log2(min(w, h)).
 */
static void inverse_tx(int s, int tx, const int* in, int* out, int mu, int mv) {
    /* mu and mv bound the nonzero coefficients; every term past them is a
     * zero, so the bounds change the time and nothing else. Integer sums,
     * so the loop order, chosen for contiguous inner loops, changes
     * nothing either. The identity's 256 at 16 keeps every product inside
     * 32 bits too: 32767 * 256 * 16 is under 2^27. */
    int w = SHAPE_W[s], h = SHAPE_H[s];
    const int* t = tmat[TX_V[tx]][size_class(h)];
    const int* th = tmat[TX_H[tx]][size_class(w)];
    int tmp[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    int shift2 = 6 + log2_int(w < h ? w : h), half = w != h;
    for (int y = 0; y < h; y++) {
        int a[NVDR_MAX_BLOCK];
        for (int u = 0; u <= mu; u++) a[u] = 0;
        for (int v = 0; v <= mv; v++) {
            int k = t[v * h + y];
            const int* row = in + v * w;
            for (int u = 0; u <= mu; u++) a[u] += k * row[u];
        }
        for (int u = 0; u <= mu; u++) {
            int c = clamp_coef((a[u] + 32) >> 6);
            tmp[y * w + u] = half ? (c * 181 + 128) >> 8 : c;
        }
    }
    for (int y = 0; y < h; y++) {
        int a[NVDR_MAX_BLOCK];
        for (int x = 0; x < w; x++) a[x] = 0;
        for (int u = 0; u <= mu; u++) {
            int k = tmp[y * w + u];
            const int* row = th + u * w;
            for (int x = 0; x < w; x++) a[x] += k * row[x];
        }
        for (int x = 0; x < w; x++) out[y * w + x] = (a[x] + (1 << (shift2 - 1))) >> shift2;
    }
}


/* Round a / n to nearest, halves away from zero, n a power of two. */
static int div_round(int a, int n) {
    int k = log2_int(n);
    return a >= 0 ? (a + n / 2) >> k : -((-a + n / 2) >> k);
}
/* The pixels the DC level `v` (level times step) adds to a leaf: v over
 * sqrt(w h), as the inverse transform scales a half leaf. */
static int dc_offset(int v, int w, int h) {
    if (w == h) return div_round(v, w);
    int m = w < h ? w : h;
    return div_round(v * 181, 256 * m);
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
