/* Part of nvdr.c, which includes it:
 * the canvas, prediction (flat and directional), concealment, texture. */

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
    int dirpred;                 /* NVDR_FLAG_DIRPRED: predict from `full`, along modes */
    int inter;                   /* NVDR_FLAG_INTER: and from `base` */
    int txsel;                   /* NVDR_FLAG_TXSEL: a transform type per leaf */
    int legacy_modes;            /* a version 13 container: fifteen intra modes */
    int halves;                  /* whole nodes may be cut in two (v15, directional) */
    int slice_top;               /* the first row of the slice being coded (SLICES) */
    uint8_t* base[3];            /* with it, the base picture on this canvas */
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
    for (int c = 0; c < 3; c++) { free(cv->flat[c]); free(cv->full[c]); free(cv->acc[c]); free(cv->base[c]); }
}

/*
 * A base picture on the canvases, in integers identical in every decoder:
 * Y, Cb and Cr in 16-bit fixed point with JPEG's coefficients, the canvas
 * padded by repeating the last row and column, and in 4:2:0 the colour
 * the rounded mean of each 2x2 (repeating at the edge), as the encoder
 * averages its source.
 */
static int canvas_base(Canvas* cv, Canvas* cc, const NvdrImage* img) {
    size_t n = (size_t)cv->pw * cv->ph;
    uint8_t* full[3];
    for (int c = 0; c < 3; c++) if (!(full[c] = (uint8_t*)malloc(n))) { for (int k = 0; k < c; k++) free(full[k]); return -1; }
    for (int y = 0; y < cv->ph; y++)
        for (int x = 0; x < cv->pw; x++) {
            int sx = x < cv->w ? x : cv->w - 1, sy = y < cv->h ? y : cv->h - 1;
            const unsigned char* p = img->pixels + ((size_t)sy * img->width + sx) * 3;
            int r = p[0], g = p[1], b = p[2];
            size_t at = (size_t)y * cv->pw + x;
            full[0][at] = (uint8_t)clamp_u8((19595 * r + 38470 * g + 7471 * b + 32768) >> 16);
            full[1][at] = (uint8_t)clamp_u8(((-11059 * r - 21709 * g + 32768 * b + 32768) >> 16) + 128);
            full[2][at] = (uint8_t)clamp_u8(((32768 * r - 27439 * g - 5329 * b + 32768) >> 16) + 128);
        }
    if (!cc) {
        for (int c = 0; c < 3; c++) cv->base[c] = full[c];
        return 0;
    }
    cv->base[0] = full[0];
    for (int c = 1; c < 3; c++) {
        uint8_t* h = (uint8_t*)malloc((size_t)cc->pw * cc->ph);
        if (!h) { free(full[1]); free(full[2]); return -1; }
        for (int y = 0; y < cc->ph; y++)
            for (int x = 0; x < cc->pw; x++) {
                int sum = 0;
                for (int j = 0; j < 2; j++)
                    for (int i = 0; i < 2; i++) {
                        int fx = 2 * x + i, fy = 2 * y + j;
                        if (fx >= cv->pw) fx = cv->pw - 1;
                        if (fy >= cv->ph) fy = cv->ph - 1;
                        sum += full[c][(size_t)fy * cv->pw + fx];
                    }
                h[(size_t)y * cc->pw + x] = (uint8_t)((sum + 2) >> 2);
            }
        cc->base[c - 1] = h;
        free(full[c]);
    }
    return 0;
}

/*
 * A leaf's predicted colour: the rounded mean of the flat colours in the
 * row just above it and the column just to its left, whichever exist on
 * the canvas and, above, in its slice; 128 at the top-left corner.
 * Integer, and read only from `flat`, so layer 0 needs nothing else to
 * decode.
 */
static int predict(const Canvas* cv, int c, int x, int y, int w, int h) {
    if (cv->fixed_pred) return 128;
    const uint8_t* p = cv->dirpred ? cv->full[c] : cv->flat[c];
    int sum = 0, k = 0;
    if (y > cv->slice_top) {
        int x1 = x + w < cv->pw ? x + w : cv->pw;
        for (int i = x; i < x1; i++) sum += p[(size_t)(y - 1) * cv->pw + i];
        k += x1 - x;
    }
    if (x > 0) {
        int y1 = y + h < cv->ph ? y + h : cv->ph;
        for (int j = y; j < y1; j++) sum += p[(size_t)j * cv->pw + x - 1];
        k += y1 - y;
    }
    return k ? (sum + k / 2) / k : 128;
}

/*
 * DIRECTIONAL PREDICTION
 * ----------------------
 * With NVDR_FLAG_DIRPRED a leaf is predicted from the reconstruction
 * around it: the row above, extended right to twice the leaf by repeating
 * its last pixel, the column to the left, extended down the same way, and
 * the corner. A side the canvas does not have takes the first pixel of
 * the other; with neither everything is 128. The modes:
 *
 *   0       flat: the mean of that row and column, as without the flag
 *   1       planar, HEVC's
 *   2..18   from above, at all 17 of HEVC's angles, -32 to 32 in 32nds of
 *           a pixel per row (10, angle 0, is vertical; -32 comes from the
 *           corner, 32 from above-right)
 *   19..34  from the left, at -26 to 32 (26 is horizontal, 34 from
 *           below-left)
 *   35..37  AV1's smooth modes: the row above blended toward the bottom
 *           left pixel down the leaf and the column to the left toward
 *           the top right pixel across it, by AV1's weights; both, down
 *           only, across only
 *   38..42  AV1's recursive filters: 4x2 patches in raster order, each
 *           pixel seven taps over the corner, the four pixels above the
 *           patch and the two to its left, already predicted or not
 *
 * The angular modes are HEVC's exactly: each row steps along the
 * reference by the angle, interpolated in 32nds, and a negative angle
 * reaches round the corner by projecting the other side with HEVC's
 * inverse angles. The DC level then moves the whole prediction by a
 * constant, and the texture is the DCT of what is left. All integer.
 *
 * Containers before version 14 had 15 modes, 7 angles from above and 6
 * from the left, which read as the same angles here (OLD_MODE).
 */
static const int MODE_ANGLE[NVDR_MODES] = { 0, 0,
    -32, -26, -21, -17, -13, -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32,
    -26, -21, -17, -13, -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32 };
static int inv_angle(int a) {
    switch (a) {
    case -32: return -256;  case -26: return -315; case -21: return -390; case -17: return -482;
    case -13: return -630;  case -9:  return -910; case -5:  return -1638; case -2: return -4096;
    default:  return 0;
    }
}

/* AV1's smooth weights, by leaf size. */
static const uint8_t SM_W4[4] = { 255, 149, 85, 64 };
static const uint8_t SM_W8[8] = { 255, 197, 146, 105, 73, 50, 37, 32 };
static const uint8_t SM_W16[16] = { 255, 225, 196, 170, 145, 123, 102, 84, 68, 54, 43, 33, 26, 20, 17, 16 };
static const uint8_t SM_W32[32] = { 255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101, 92, 83, 74,
                                    66, 59, 52, 45, 39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8 };

/* AV1's filter-intra taps: per filter, per pixel of the 4x2 patch (row
 * then column), over the corner, the four above and the two to the left,
 * in sixteenths. */
static const int8_t FI_TAPS[5][8][7] = {
    { { -6, 10, 0, 0, 0, 12, 0 }, { -5, 2, 10, 0, 0, 9, 0 }, { -3, 1, 1, 10, 0, 7, 0 }, { -3, 1, 1, 2, 10, 5, 0 },
      { -4, 6, 0, 0, 0, 2, 12 }, { -3, 2, 6, 0, 0, 2, 9 }, { -3, 2, 2, 6, 0, 2, 7 }, { -3, 1, 2, 2, 6, 3, 5 } },
    { { -10, 16, 0, 0, 0, 10, 0 }, { -6, 0, 16, 0, 0, 6, 0 }, { -4, 0, 0, 16, 0, 4, 0 }, { -2, 0, 0, 0, 16, 2, 0 },
      { -10, 16, 0, 0, 0, 0, 10 }, { -6, 0, 16, 0, 0, 0, 6 }, { -4, 0, 0, 16, 0, 0, 4 }, { -2, 0, 0, 0, 16, 0, 2 } },
    { { -8, 8, 0, 0, 0, 16, 0 }, { -8, 0, 8, 0, 0, 16, 0 }, { -8, 0, 0, 8, 0, 16, 0 }, { -8, 0, 0, 0, 8, 16, 0 },
      { -4, 4, 0, 0, 0, 0, 16 }, { -4, 0, 4, 0, 0, 0, 16 }, { -4, 0, 0, 4, 0, 0, 16 }, { -4, 0, 0, 0, 4, 0, 16 } },
    { { -2, 8, 0, 0, 0, 10, 0 }, { -1, 3, 8, 0, 0, 6, 0 }, { -1, 2, 3, 8, 0, 4, 0 }, { 0, 1, 2, 3, 8, 2, 0 },
      { -1, 4, 0, 0, 0, 3, 10 }, { -1, 3, 4, 0, 0, 4, 6 }, { -1, 2, 3, 4, 0, 4, 4 }, { -1, 2, 2, 3, 4, 3, 3 } },
    { { -12, 14, 0, 0, 0, 14, 0 }, { -10, 0, 14, 0, 0, 12, 0 }, { -9, 0, 0, 14, 0, 11, 0 }, { -8, 0, 0, 0, 14, 10, 0 },
      { -10, 12, 0, 0, 0, 0, 14 }, { -9, 1, 12, 0, 0, 0, 12 }, { -8, 0, 0, 12, 0, 1, 11 }, { -7, 0, 0, 1, 12, 1, 9 } },
};

/* One angular prediction along `main`, with `side` round the corner:
 * main[0] and side[0] are the corner, then nk + nr samples each. Writes
 * P[r * nk + k] for r < nr the step along the angle, k < nk across it. */
static void angular(const int* main, const int* side, int nk, int nr, int angle, int* P) {
    int buf[4 * NVDR_MAX_BLOCK + 2], *ref = buf + 2 * NVDR_MAX_BLOCK, end = nk + nr;
    for (int k = 0; k <= end; k++) ref[k] = main[k];
    if (angle < 0) {
        int last = (nr * angle) >> 5, inv = inv_angle(angle);
        for (int k = -1; k >= last; k--) ref[k] = side[(k * inv + 128) >> 8];
    }
    for (int r = 0; r < nr; r++) {
        int pos = (r + 1) * angle, idx = pos >> 5, fact = pos & 31;
        for (int k = 0; k < nk; k++) {
            int a0 = ref[k + idx + 1], a1 = ref[k + idx + 2 <= end ? k + idx + 2 : end];
            P[r * nk + k] = fact ? ((32 - fact) * a0 + fact * a1 + 16) >> 5 : a0;
        }
    }
}

/* The position of a 4x4 cell in the order a tile's quadtree is decoded. */
static int zorder(int cx, int cy) {
    int z = 0;
    for (int b = 0; b < 4; b++) z |= ((cx >> b) & 1) << (2 * b) | ((cy >> b) & 1) << (2 * b + 1);
    return z;
}

/* Whether pixel (px, py) was decoded before the leaf at (x, y): in an
 * earlier tile, or earlier in this tile's quadtree. */
static int decoded_before(const Canvas* cv, int x, int y, int px, int py) {
    if (px < 0 || py < 0 || px >= cv->pw || py >= cv->ph) return 0;
    int t = cv->tile, lt = (y / t) * 65536 + x / t, pt = (py / t) * 65536 + px / t;
    if (pt != lt) return pt < lt;
    return zorder((px % t) / 4, (py % t) / 4) < zorder((x % t) / 4, (y % t) / 4);
}

static void dir_predict(const Canvas* cv, int c, int x, int y, int w, int h, int mode, int* P) {
    if (mode == MODE_INTER) {
        for (int j = 0; j < h; j++)
            for (int i = 0; i < w; i++) P[j * w + i] = cv->base[c][(size_t)(y + j) * cv->pw + x + i];
        return;
    }
    const uint8_t* f = cv->full[c];
    int pw = cv->pw, ne = w + h, T[2 * NVDR_MAX_BLOCK + 1] = { 0 }, L[2 * NVDR_MAX_BLOCK + 1] = { 0 };
    int top = y > cv->slice_top, left = x > 0;   /* nothing above its slice (SLICES) */
    /* T[0] and L[0] are the corner, then w + h samples: the row above and
     * on past the leaf while it has been decoded, then its last repeated;
     * the column to the left likewise down. */
    int tr = top, bl = left;
    for (int i = 0; i < ne; i++) {
        if (i >= w && tr && !(i % 4) && !decoded_before(cv, x, y, x + i, y - 1)) tr = 0;
        if (i >= h && bl && !(i % 4) && !decoded_before(cv, x, y, x - 1, y + i)) bl = 0;
        T[i + 1] = !top ? 0 : (i < w || tr) ? f[(size_t)(y - 1) * pw + x + i] : T[i];
        L[i + 1] = !left ? 0 : (i < h || bl) ? f[(size_t)(y + i) * pw + x - 1] : L[i];
    }
    if (!top && !left) { for (int i = 0; i <= ne; i++) T[i] = L[i] = 128; }
    else if (!top) { for (int i = 0; i <= ne; i++) T[i] = L[1]; L[0] = L[1]; }
    else if (!left) { for (int i = 0; i <= ne; i++) L[i] = T[1]; T[0] = T[1]; }
    else T[0] = L[0] = f[(size_t)(y - 1) * pw + x - 1];
    if (mode >= MODE_FILTER) {
        int B[NVDR_MAX_BLOCK + 2][NVDR_MAX_BLOCK + 1];
        const int8_t (*tp)[7] = FI_TAPS[mode - MODE_FILTER];
        for (int i = 0; i <= w; i++) B[0][i] = T[i];
        for (int j = 0; j <= h; j++) B[j][0] = L[j];
        for (int r = 1; r <= h; r += 2)
            for (int cc = 1; cc <= w; cc += 4) {
                int p[7] = { B[r - 1][cc - 1], B[r - 1][cc], B[r - 1][cc + 1], B[r - 1][cc + 2], B[r - 1][cc + 3],
                             B[r][cc - 1], B[r + 1][cc - 1] };
                for (int k = 0; k < 8; k++) {
                    int a = 0;
                    for (int t = 0; t < 7; t++) a += tp[k][t] * p[t];
                    a = a >= 0 ? (a + 8) >> 4 : -((-a + 8) >> 4);
                    B[r + (k >> 2)][cc + (k & 3)] = clamp_u8(a);
                }
            }
        for (int j = 0; j < h; j++)
            for (int i = 0; i < w; i++) P[j * w + i] = B[j + 1][i + 1];
        return;
    }
    if (mode >= MODE_SMOOTH) {
        const uint8_t* wv = h == 4 ? SM_W4 : h == 8 ? SM_W8 : h == 16 ? SM_W16 : SM_W32;
        const uint8_t* wh = w == 4 ? SM_W4 : w == 8 ? SM_W8 : w == 16 ? SM_W16 : SM_W32;
        int below = L[h], right = T[w], k = mode - MODE_SMOOTH;
        for (int j = 0; j < h; j++)
            for (int i = 0; i < w; i++) {
                int v = wv[j] * T[i + 1] + (256 - wv[j]) * below, hh = wh[i] * L[j + 1] + (256 - wh[i]) * right;
                P[j * w + i] = k == 0 ? (v + hh + 256) >> 9 : k == 1 ? (v + 128) >> 8 : (hh + 128) >> 8;
            }
        return;
    }
    if (w * h >= 64 && mode != 0) {
        /* HEVC's [1 2 1] over the references, corner shared. */
        int t2[2 * NVDR_MAX_BLOCK + 1], l2[2 * NVDR_MAX_BLOCK + 1];
        t2[0] = l2[0] = (T[1] + 2 * T[0] + L[1] + 2) >> 2;
        for (int i = 1; i < ne; i++) {
            t2[i] = (T[i - 1] + 2 * T[i] + T[i + 1] + 2) >> 2;
            l2[i] = (L[i - 1] + 2 * L[i] + L[i + 1] + 2) >> 2;
        }
        t2[ne] = T[ne]; l2[ne] = L[ne];
        memcpy(T, t2, sizeof(int) * (ne + 1)); memcpy(L, l2, sizeof(int) * (ne + 1));
    }
    if (mode == 1) {
        /* Planar, as VVC writes it for any shape. */
        int lw = log2_int(w), lh = log2_int(h);
        for (int j = 0; j < h; j++)
            for (int i = 0; i < w; i++) {
                int pv = ((h - 1 - j) * T[i + 1] + (j + 1) * L[h + 1]) << lw;
                int ph = ((w - 1 - i) * L[j + 1] + (i + 1) * T[w + 1]) << lh;
                P[j * w + i] = (pv + ph + w * h) >> (lw + lh + 1);
            }
        return;
    }
    int tmp[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    if (mode <= 18) { angular(T, L, w, h, MODE_ANGLE[mode], P); return; }
    /* From the left: the same along the column, then transposed. */
    angular(L, T, h, w, MODE_ANGLE[mode], tmp);
    for (int j = 0; j < h; j++)
        for (int i = 0; i < w; i++) P[j * w + i] = tmp[i * h + j];
}

/* A leaf painted with a prediction moved by `offset`, and no texture. */
static void paint_pred(Canvas* cv, int c, int x, int y, int w, int h, const int* P, int offset) {
    for (int j = 0; j < h; j++)
        for (int i = 0; i < w; i++) {
            size_t at = (size_t)(y + j) * cv->pw + x + i;
            uint8_t v = (uint8_t)clamp_u8(P[j * w + i] + offset);
            cv->flat[c][at] = cv->full[c][at] = v;
            cv->acc[c][at] = 0;
        }
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
        /* With a base, a tile that never arrived shows the base. */
        if (cv->inter)
            for (int j = ty; j < ty + h; j++)
                for (int i = tx; i < tx + w; i++) {
                    size_t at = (size_t)j * cv->pw + i;
                    cv->flat[c][at] = cv->full[c][at] = cv->base[c][at];
                }
    }
}

/*
 * CONCEALMENT
 * -----------
 * A tile whose colour never arrived, with tiles that did arrive both
 * above and below it in its column (a hole in the middle of a file with
 * slices), is filled in from them instead of left grey: each column of
 * pixels runs in a straight line from the last row above to the first
 * row below, each of the two taken as its mean over CONCEAL_R pixels
 * either side (those of tiles that arrived), since a single row's
 * texture drawn down the gap shows as stripes. A file cut short never has anything below what is
 * missing, so tile_fallback()'s guarantee stands: more bytes never show
 * a worse picture. Nothing is filled from one side alone, for the reason
 * given there. `flat` is filled from the flat rows and `full` from the
 * full ones, so each layer shown has its own. Integer, and the same in
 * nvdr.js.
 */
#define CONCEAL_R 16
/* The mean of row y around x, over the pixels whose tile in row `r` of
 * tiles arrived. */
static int conceal_edge(const Canvas* cv, const uint8_t* p, const uint8_t* whole0, int tiles_x, int r, int y, int x) {
    int sum = 0, n = 0;
    for (int i = x - CONCEAL_R; i <= x + CONCEAL_R; i++) {
        if (i < 0 || i >= cv->pw || !whole0[r * tiles_x + i / cv->tile]) continue;
        sum += p[(size_t)y * cv->pw + i];
        n++;
    }
    return (sum + n / 2) / n;
}

static void conceal(Canvas* cv, const uint8_t* whole0, int tiles_x, int tiles_y) {
    for (int tx = 0; tx < tiles_x; tx++) {
        int x0 = tx * cv->tile, x1 = x0 + cv->tile < cv->pw ? x0 + cv->tile : cv->pw;
        if (x0 >= cv->pw) continue;
        for (int r0 = 0; r0 < tiles_y; r0++) {
            if (whole0[r0 * tiles_x + tx]) continue;
            int r1 = r0;
            while (r1 < tiles_y && !whole0[r1 * tiles_x + tx]) r1++;
            int y0 = r0 * cv->tile, y1 = r1 * cv->tile;
            if (r0 > 0 && r1 < tiles_y && y1 < cv->ph) {
                int n = y1 - y0;
                for (int c = 0; c < cv->np; c++)
                    for (int k = 0; k < 2; k++) {
                        uint8_t* p = k ? cv->full[c] : cv->flat[c];
                        for (int x = x0; x < x1; x++) {
                            int a = conceal_edge(cv, p, whole0, tiles_x, r0 - 1, y0 - 1, x);
                            int b = conceal_edge(cv, p, whole0, tiles_x, r1, y1, x);
                            for (int j = 0; j < n; j++)
                                p[(size_t)(y0 + j) * cv->pw + x] =
                                    (uint8_t)((a * (n - j) + b * (j + 1) + (n + 1) / 2) / (n + 1));
                        }
                    }
            }
            r0 = r1;
        }
    }
}

/* The pixels the texture in scan positions [start, end) adds to a leaf
 * of shape s. */
static void texture_residual(int s, int tx, const int* lv, int start, int end, int step, int* res) {
    int w = SHAPE_W[s], count = w * SHAPE_H[s];
    int coef[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    memset(coef, 0, sizeof(int) * count);
    int mu = 0, mv = 0, sh = log2_int(w);
    for (int i = start; i < end; i++) {
        if (!lv[i]) continue;
        int p = scan_pos[s][i], u = p & (w - 1), v = p >> sh;
        coef[p] = clamp_coef((long)lv[i] * step);
        if (u > mu) mu = u;
        if (v > mv) mv = v;
    }
    inverse_tx(s, tx, coef, res, mu, mv);
}

/* Adds them to the leaf's accumulated texture, and shows flat + texture
 * clamped. */
static void add_residual(Canvas* cv, int c, int x, int y, int w, int h, const int* res) {
    for (int j = 0; j < h; j++)
        for (int i = 0; i < w; i++) {
            size_t at = (size_t)(y + j) * cv->pw + x + i;
            int a = cv->acc[c][at] + res[j * w + i];
            a = a < -32768 ? -32768 : (a > 32767 ? 32767 : a);
            cv->acc[c][at] = (int16_t)a;
            cv->full[c][at] = (uint8_t)clamp_u8(cv->flat[c][at] + a);
        }
}

static void apply_texture(Canvas* cv, int c, int x, int y, int s, int tx, const int* lv,
                          int start, int end, int step) {
    int res[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    texture_residual(s, tx, lv, start, end, step, res);
    add_residual(cv, c, x, y, SHAPE_W[s], SHAPE_H[s], res);
}
