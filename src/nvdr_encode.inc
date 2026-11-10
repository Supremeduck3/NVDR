/* Part of nvdr.c, which includes it:
 * the encoder: the search, slices side by side, restoration, chroma, grain. */

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

#define EST_KEEP 2   /* see try_modes() */
#define PC_SLOTS 12  /* modes a leaf's prediction cache holds: flat and DIR_TRIED tried, and room */

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
    int            band_at[NSHAPES];   /* first high-band scan position per shape */
    int            try_halves;  /* whether the search cuts nodes in two (they are flagged anyway) */
    int            tx_modes;    /* see NvdrConfig */
    int            fast_rank;   /* see NvdrConfig */
    /* While try_modes() ranks a leaf's modes and transform types,
     * leaf_levels() measures a directional leaf's error on its
     * coefficients and paints nothing; the EST_KEEP cheapest so measured
     * are then coded in full (try_modes()). */
    int            estimate;
    int            ntop;
    double         top_j[EST_KEEP];
    int            top_m[EST_KEEP], top_t[EST_KEEP];
    int            last_mode;   /* the mode code_leaf() chose last */
    /* A leaf's directional predictions while its modes and transform
     * types are tried (pred_cached()): a prediction reads only the decoded
     * pixels around the leaf, which trying the leaf does not change, and
     * computing it again for every transform type was a tenth of the
     * encoding. Keyed by the leaf; pc_part NULL when off. */
    const Part*    pc_part;
    int            pc_x, pc_y, pc_w, pc_h, pc_n;
    int            pc_mode[PC_SLOTS];
    uint8_t        pc_have[PC_SLOTS];   /* bit c: plane c is there */
    int*           pc;                  /* PC_SLOTS x 3 x NVDR_MAX_BLOCK^2 */
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
#define RDOQ_PICTURE  0.6
#define RDOQ_RESIDUAL 0.6
#define RDOQ_KEEP     4
static void rdoq(const TextureModels* m, int sc, int c, const double* v, int* lv,
                 int start, int end, double lam, int keep);

/* dir_predict(), from the leaf's cache when it is the leaf being tried. */
static void pred_cached(Enc* e, const Canvas* cv, int c, int x, int y, int w, int h, int mode, int* pr) {
    if (!e->pc || e->pc_part != e->p || e->pc_x != x || e->pc_y != y || e->pc_w != w || e->pc_h != h) {
        dir_predict(cv, c, x, y, w, h, mode, pr);
        return;
    }
    int k = 0;
    while (k < e->pc_n && e->pc_mode[k] != mode) k++;
    if (k == e->pc_n) {
        if (k == PC_SLOTS) { dir_predict(cv, c, x, y, w, h, mode, pr); return; }
        e->pc_mode[k] = mode;
        e->pc_have[k] = 0;
        e->pc_n++;
    }
    int* slot = e->pc + ((size_t)k * 3 + (size_t)c) * NVDR_MAX_BLOCK * NVDR_MAX_BLOCK;
    if (!(e->pc_have[k] >> c & 1)) {
        dir_predict(cv, c, x, y, w, h, mode, slot);
        e->pc_have[k] |= (uint8_t)(1 << c);
    }
    memcpy(pr, slot, sizeof(int) * (size_t)w * h);
}

static double leaf_levels(Enc* e, int x, int y, int w, int h, int skip, int mode, int tx,
                          int dl[3], int lv[3][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK], int* textured) {
    Part* P = e->p;
    Canvas* cv = &P->cv;
    int sh = shape_of(w, h), count = w * h, sq = w == h, n = w, sc = size_class(w);
    double err = 0.0;
    int any = 0;
    for (int c = 0; c < cv->np; c++) {
        int step = e->step[cv->comp0 + c];
        size_t b = sq ? (size_t)((y - P->pre_y) / n) * (cv->pw / n) + (size_t)(x / n) : 0;
        memset(lv[c], 0, sizeof(int) * count);
        int nonzero = 0;
        if (mode == MODE_INTER && !tx && sq) {
            /* The base under the leaf, moved by the DC; the texture of
             * the block minus the base, prepared with the row. */
            int pr[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
            dir_predict(cv, c, x, y, n, n, MODE_INTER, pr);
            double dc = P->pre_sum[sc][c][b] / count * n;
            dl[c] = quantise(dc / step, 0.0);
            paint_pred(cv, c, x, y, n, n, pr, dc_offset(clamp_coef((long)dl[c] * step), n, n));
            memcpy(lv[c], P->pre[sc][c] + b * count, sizeof(int) * (size_t)count);
            nonzero = P->pre_nz[sc][c][b];
            if (nonzero) {
                add_residual(cv, c, x, y, n, n, P->pre_lo[sc][c] + b * count);
                any = 1;
            }
        } else if (mode > 0 || cv->inter || tx || !sq) {
            /* The texture of a directional prediction depends on the
             * prediction, so it is transformed here and not beforehand;
             * so is any half leaf's, which nothing prepares. */
            int pr[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
            double blk[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK], co[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
            if (mode > 0) pred_cached(e, cv, c, x, y, w, h, mode, pr);
            else {
                /* Flat in a picture with a base: its texture is not
                 * prepared, since the row's is the block minus the base. */
                int f = predict(cv, c, x, y, w, h);
                for (int i = 0; i < count; i++) pr[i] = f;
            }
            for (int j = 0; j < h; j++)
                for (int i = 0; i < w; i++)
                    blk[j * w + i] = P->src[c][(size_t)(y + j) * cv->pw + x + i] - pr[j * w + i];
            forward_tx(sh, tx, blk, co);
            /* Without the DCT there is no DC level: the texture starts at
             * scan position 0 (see TRANSFORM TYPES). */
            int t0 = tx ? 0 : 1;
            dl[c] = tx ? 0 : quantise(co[0] / step, 0.0);
            if (!e->estimate) paint_pred(cv, c, x, y, w, h, pr, dc_offset(clamp_coef((long)dl[c] * step), w, h));
            double v[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
            for (int i = t0; i < count; i++) v[i] = co[scan_pos[sh][i]] / step;
            if (e->rdoq && !(e->estimate && e->fast_rank)) {
                double lam = P->lambda_base / ((double)e->base_step[cv->comp0 + c] * e->base_step[cv->comp0 + c]) * RDOQ_PICTURE;
                rdoq(&e->tm, sh, cv->comp0 + c, v, lv[c], t0, count, lam, RDOQ_KEEP);
            } else
                for (int i = t0; i < count; i++) lv[c][i] = quantise(v[i], e->deadzone);
            for (int i = t0; i < count; i++) nonzero |= lv[c][i];
            if (e->estimate) {
                /* The transform is orthonormal, so the error is the
                 * coefficients' quantisation error (Parseval), plus about
                 * 1/12 per pixel for rounding the reconstruction; no
                 * inverse transform. Clipping and the part of an edge leaf
                 * outside the picture are ignored, which is why the
                 * ranking is checked by coding in full. */
                double ec = (double)count / 12.0;
                if (!tx) { double d0 = co[0] - (double)dl[c] * step; ec += d0 * d0; }
                for (int i = t0; i < count; i++) { double d = co[scan_pos[sh][i]] - (double)lv[c][i] * step; ec += d * d; }
                err += ec;
                any |= nonzero;
                continue;
            }
            if (nonzero) {
                int res[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
                texture_residual(sh, tx, lv[c], t0, count, step, res);
                add_residual(cv, c, x, y, w, h, res);
                any = 1;
            }
        } else {
            int pred = predict(cv, c, x, y, n, n);
            dl[c] = 0;
            if (!skip) {
                double sum = P->pre_sum[sc][c][b];
                /* The orthonormal DC of (block - prediction) is n * its mean. */
                double dc = (sum / count - pred) * n;
                dl[c] = quantise(dc / step, 0.0);
            }
            int colour = clamp_u8(pred + dc_offset(clamp_coef((long)dl[c] * step), n, n));
            paint_flat(cv, c, x, y, n, n, colour);
            if (!skip) {
                memcpy(lv[c], P->pre[sc][c] + b * count, sizeof(int) * (size_t)count);
                nonzero = P->pre_nz[sc][c][b];
            }
            if (nonzero) {
                add_residual(cv, c, x, y, n, n, P->pre_lo[sc][c] + b * count);
                if (e->band_at[sc] < count) add_residual(cv, c, x, y, n, n, P->pre_hi[sc][c] + b * count);
                any = 1;
            }
        }

        for (int j = 0; j < h && y + j < cv->h; j++)
            for (int i = 0; i < w && x + i < cv->w; i++) {
                size_t at = (size_t)(y + j) * cv->pw + x + i;
                double d = P->src[c][at] - cv->full[c][at];
                err += d * d;
            }
    }
    if (textured) *textured = any;
    return err;
}

/* s1 is the two texture layers' sinks, low band then high. */
/* Whether a leaf carries a transform type. */
static int has_tx(const Canvas* cv, int w, int h, int mode) {
    return cv->txsel && w <= TX_MAX_N && h <= TX_MAX_N && mode != MODE_INTER;
}

static void leaf_emit(Enc* e, Sink* s0, Sink* s1, int sc, int mode, int tx,
                      const int dl[3], int lv[3][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK]) {
    int w = SHAPE_W[sc], h = SHAPE_H[sc], mc = SHAPE_MC[sc], count = w * h, at = e->band_at[sc];
    const Canvas* cv = &e->p->cv;
    if (cv->dirpred) put_mode(s0, &e->cm, mc, mode, cv->inter);
    if (has_tx(cv, w, h, mode)) put_tx(s0, &e->cm, mc, mode, tx);
    for (int c = 0; c < cv->np; c++) {
        int k = cv->comp0 + c;
        double before = s0->bits + s1[0].bits + s1[1].bits;
        if (tx) {
            /* No DC, no band split: directional pictures have one layer. */
            put_texture(&s1[0], &e->tm, sc, k, lv[c], 0, count);
            if (s0->enc) e->plane_bits[k] += s0->bits + s1[0].bits + s1[1].bits - before;
            continue;
        }
        put_dc(s0, &e->cm, mc, k, dl[c]);
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
/* How many directional modes code_leaf() tries in full after screening. */
#define DIR_TRIED 9
/* Mean squared error per sample above which a leaf of a picture with a
 * base tries the other modes besides INTER. */
#define INTRA_TRY 16.0

/* Sum of absolute 4x4 Hadamard coefficients of source minus prediction. */
static double satd_leaf(const double* src, int pw, int x, int y, int w, int h, const int* pr) {
    double acc = 0.0;
    for (int by = 0; by < h; by += 4)
        for (int bx = 0; bx < w; bx += 4) {
            double d[16];
            for (int j = 0; j < 4; j++)
                for (int i = 0; i < 4; i++)
                    d[4 * j + i] = src[(size_t)(y + by + j) * pw + x + bx + i] - pr[(by + j) * w + bx + i];
            for (int j = 0; j < 4; j++) {
                double* r = d + 4 * j, a = r[0] + r[1], b = r[0] - r[1], c = r[2] + r[3], e = r[2] - r[3];
                r[0] = a + c; r[1] = b + e; r[2] = a - c; r[3] = b - e;
            }
            for (int i = 0; i < 4; i++) {
                double a = d[i] + d[4 + i], b = d[i] - d[4 + i], c = d[8 + i] + d[12 + i], e = d[8 + i] - d[12 + i];
                acc += fabs(a + c) + fabs(b + e) + fabs(a - c) + fabs(b - e);
            }
        }
    return acc;
}

/* Keeps (mode, tx) among the EST_KEEP cheapest estimates so far. */
static void keep_estimate(Enc* e, double j, int mode, int tx) {
    int k;
    if (e->ntop < EST_KEEP) k = e->ntop++;
    else if (j >= e->top_j[EST_KEEP - 1]) return;
    else k = EST_KEEP - 1;
    e->top_j[k] = j; e->top_m[k] = mode; e->top_t[k] = tx;
    for (; k > 0 && e->top_j[k] < e->top_j[k - 1]; k--) {
        double tj = e->top_j[k]; e->top_j[k] = e->top_j[k - 1]; e->top_j[k - 1] = tj;
        int t = e->top_m[k]; e->top_m[k] = e->top_m[k - 1]; e->top_m[k - 1] = t;
        t = e->top_t[k]; e->top_t[k] = e->top_t[k - 1]; e->top_t[k - 1] = t;
    }
}

/* The other transform types for one mode, each coded in full against the
 * models; `best` is the mode's cost with the DCT. *tx becomes whichever
 * type costs least, and the cost is returned. */
static double choose_tx(Enc* e, int x, int y, int w, int h, int mode, double best, int* tx) {
    static _Thread_local int dl[3], lv[3][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    int sh = shape_of(w, h);
    *tx = TX_DCT;
    if (!has_tx(&e->p->cv, w, h, mode)) return best;
    for (int t = 1; t < TX_TYPES; t++) {
        Sink c0 = { NULL, 0 }, c1[2] = { { NULL, 0 }, { NULL, 0 } };
        double d = leaf_levels(e, x, y, w, h, 0, mode, t, dl, lv, NULL);
        leaf_emit(e, &c0, c1, sh, mode, t, dl, lv);
        double j = d + e->p->lambda * (c0.bits + c1[0].bits + c1[1].bits);
        if (e->estimate) keep_estimate(e, j, mode, t);
        if (j < best) { best = j; *tx = t; }
    }
    return best;
}

/* The modes in cand[0, nc) coded with the DCT, then the other transform
 * types tried on all of them, or with tx_modes on that many of the
 * cheapest; *mode and *tx get the winner if it beats `best`, and the cost
 * of whichever wins is returned.
 *
 * The ranking runs on estimated error (see Enc.estimate), the bits being
 * counted exactly; the EST_KEEP cheapest are then coded in full and
 * compared with `best`. On the samples, keeping 2 gives the same BD-rate
 * as coding every candidate in full in 85% of the time; keeping 1 loses
 * 0.4%. Estimating the error from the coefficients is the usual encoder
 * shortcut: Parseval's relation, the way RDOQ already weighs each level
 * (Karczewicz, Ye and Chong, "Rate distortion optimized quantization",
 * ITU-T VCEG-AH21, 2008), applied here to the choice of mode and type. */
static double try_modes(Enc* e, int x, int y, int w, int h, const int* cand, int nc, double best,
                        int* mode, int* tx) {
    static _Thread_local int dl[3], lv[3][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    int sh = shape_of(w, h);
    double jd[1 + DIR_TRIED];
    e->estimate = 1;
    e->ntop = 0;
    /* The leaf's predictions are kept until code_leaf() has coded it. */
    e->pc_part = e->p;
    e->pc_x = x; e->pc_y = y; e->pc_w = w; e->pc_h = h;
    e->pc_n = 0;
    for (int ci = 0; ci < nc; ci++) {
        Sink c0 = { NULL, 0 }, c1[2] = { { NULL, 0 }, { NULL, 0 } };
        double d = leaf_levels(e, x, y, w, h, 0, cand[ci], TX_DCT, dl, lv, NULL);
        leaf_emit(e, &c0, c1, sh, cand[ci], TX_DCT, dl, lv);
        jd[ci] = d + e->p->lambda * (c0.bits + c1[0].bits + c1[1].bits);
        keep_estimate(e, jd[ci], cand[ci], TX_DCT);
    }
    int rounds = e->tx_modes > 0 && e->tx_modes < nc ? e->tx_modes : nc;
    for (int r = 0; r < rounds; r++) {
        int bi = -1;
        for (int ci = 0; ci < nc; ci++) if (jd[ci] < 1e299 && (bi < 0 || jd[ci] < jd[bi])) bi = ci;
        if (bi < 0) break;
        int t;
        choose_tx(e, x, y, w, h, cand[bi], jd[bi], &t);
        jd[bi] = 1e300;
    }
    e->estimate = 0;
    for (int k = 0; k < e->ntop; k++) {
        Sink c0 = { NULL, 0 }, c1[2] = { { NULL, 0 }, { NULL, 0 } };
        double d = leaf_levels(e, x, y, w, h, 0, e->top_m[k], e->top_t[k], dl, lv, NULL);
        leaf_emit(e, &c0, c1, sh, e->top_m[k], e->top_t[k], dl, lv);
        double j = d + e->p->lambda * (c0.bits + c1[0].bits + c1[1].bits);
        if (j < best) { best = j; *mode = e->top_m[k]; *tx = e->top_t[k]; }
    }
    return best;
}

static double code_leaf(Enc* e, Sink* s0, Sink* s1, int x, int y, int w, int h, int* textured) {
    static _Thread_local int dl[3], lv[3][NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    int sh = shape_of(w, h);
    int skip = 0, mode = 0, tx = TX_DCT;
    double skip_lambda = e->p->skip_lambda;
    if (e->p->cv.inter) {
        /* INTER first; the other modes only where it does badly (a mean
         * squared error over INTRA_TRY), which is where something is
         * uncovered or comes in: flat and the best few directions. */
        const Canvas* cv = &e->p->cv;
        Sink c0 = { NULL, 0 }, c1[2] = { { NULL, 0 }, { NULL, 0 } };
        double d = leaf_levels(e, x, y, w, h, 0, MODE_INTER, TX_DCT, dl, lv, NULL);
        leaf_emit(e, &c0, c1, sh, MODE_INTER, TX_DCT, dl, lv);
        double best = d + e->p->lambda * (c0.bits + c1[0].bits + c1[1].bits);
        mode = MODE_INTER;
        if (d / (double)(w * h * cv->np) > INTRA_TRY) {
            int cand[1 + DIR_TRIED] = { 0 }, nc = 1;
            int pr[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
            double sc_best[DIR_TRIED];
            for (int m = 1; m < NVDR_MODES; m++) {
                dir_predict(cv, 0, x, y, w, h, m, pr);
                double sat = satd_leaf(e->p->src[0], cv->pw, x, y, w, h, pr);
                int k = nc - 1;
                if (k < DIR_TRIED) { cand[nc++] = m; sc_best[k] = sat; }
                else if (sat < sc_best[DIR_TRIED - 1]) { cand[DIR_TRIED] = m; sc_best[DIR_TRIED - 1] = sat; }
                else continue;
                for (int a = (k < DIR_TRIED ? k : DIR_TRIED - 1); a > 0 && sc_best[a] < sc_best[a - 1]; a--) {
                    double t = sc_best[a]; sc_best[a] = sc_best[a - 1]; sc_best[a - 1] = t;
                    int tm = cand[a + 1]; cand[a + 1] = cand[a]; cand[a] = tm;
                }
            }
            best = try_modes(e, x, y, w, h, cand, nc, best, &mode, &tx);
        }
    } else if (e->p->cv.dirpred) {
        /* The directional modes screened by the Hadamard sum of their
         * error on the first plane, as HEVC's reference encoder does; flat
         * and the best few tried in full, the cheapest by error plus
         * lambda times bits coded. The canvas ends up holding the last
         * tried, so the winner is reconstructed again. */
        int cand[1 + DIR_TRIED] = { 0 }, nc = 1;
        {
            const Canvas* cv = &e->p->cv;
            int pr[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
            double sc_best[DIR_TRIED];
            for (int m = 1; m < NVDR_MODES; m++) {
                dir_predict(cv, 0, x, y, w, h, m, pr);
                double sat = satd_leaf(e->p->src[0], cv->pw, x, y, w, h, pr);
                int k = nc - 1;
                if (k < DIR_TRIED) { cand[nc++] = m; sc_best[k] = sat; }
                else if (sat < sc_best[DIR_TRIED - 1]) { cand[DIR_TRIED] = m; sc_best[DIR_TRIED - 1] = sat; }
                else continue;
                /* keep the tried list sorted by score */
                for (int a = (k < DIR_TRIED ? k : DIR_TRIED - 1); a > 0 && sc_best[a] < sc_best[a - 1]; a--) {
                    double t = sc_best[a]; sc_best[a] = sc_best[a - 1]; sc_best[a - 1] = t;
                    int tm = cand[a + 1]; cand[a + 1] = cand[a]; cand[a] = tm;
                }
            }
        }
        try_modes(e, x, y, w, h, cand, nc, 1e300, &mode, &tx);
    } else if (skip_lambda > 0.0) {
        Sink c0 = { NULL, 0 }, c1[2] = { { NULL, 0 }, { NULL, 0 } };
        double d_skip = leaf_levels(e, x, y, w, h, 1, 0, TX_DCT, dl, lv, NULL);
        leaf_emit(e, &c0, c1, sh, 0, TX_DCT, dl, lv);
        double j_skip = d_skip + skip_lambda * (c0.bits + c1[0].bits + c1[1].bits);
        Sink k0 = { NULL, 0 }, k1[2] = { { NULL, 0 }, { NULL, 0 } };
        int tex = 0;
        double d_code = leaf_levels(e, x, y, w, h, 0, 0, TX_DCT, dl, lv, &tex);
        leaf_emit(e, &k0, k1, sh, 0, TX_DCT, dl, lv);
        double j_code = d_code + skip_lambda * (k0.bits + k1[0].bits + k1[1].bits);
        skip = j_skip <= j_code;
        /* Coded wins: the canvas, dl and lv already hold it, and redoing
         * it would give the same (costing only bits is not coding). */
        if (!skip) {
            if (textured) *textured = tex;
            leaf_emit(e, s0, s1, sh, 0, TX_DCT, dl, lv);
            return d_code;
        }
    }
    double err = leaf_levels(e, x, y, w, h, skip, mode, tx, dl, lv, textured);
    e->pc_part = NULL;
    leaf_emit(e, s0, s1, sh, mode, tx, dl, lv);
    e->last_mode = mode;
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

static double rd_bit(const NvdrProb* p, int b) { return bitcost[b][nvdr_prob(*p)]; }

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
    int g = 0, mc = SHAPE_MC[sc], w = SHAPE_W[sc];
    uint8_t grid[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    memset(grid, 0, (size_t)(w * SHAPE_H[sc]));
    for (int i = start; i < end; i++) {
        double a = fabs(v[i]);
        int p = scan_pos[sc][i];
        int pc = scan_ctx[sc][i], t = nb_mag(grid, w, p), sp = sig_ctx(sc, i, t);
        int l = (int)(a + 0.5);
        if (l > COEF_MAX) l = COEF_MAX;
        drop[i] = a * a;
        double zero_bits = i < end - 1 ? rd_bit(&m->sig[mc][c][sp], 0) : 0.0;
        double best = drop[i] + lam * zero_bits;
        int bl = 0;
        if (i < keep) {
            /* Kept as rounded with the dead zone: see the caller. */
            bl = (int)(a + 0.4);
            if (bl > COEF_MAX) bl = COEF_MAX;
            lv[i] = v[i] < 0 ? -bl : bl;
            grid[p] = mag3(bl);
            double bits = bl ? rd_mag_bits(m, c, gt1_ctx(t, g), bl) + 1.0 +
                               (i < end - 1 ? rd_bit(&m->sig[mc][c][sp], 1) + rd_bit(&m->last[mc][c][pc], 0) : 0.0)
                             : zero_bits;
            cost[i] = (a - bl) * (a - bl) + lam * bits;
            if (bl > 1) g++;
            continue;
        }
        for (int lvl = l; lvl >= 1 && lvl >= l - 1; lvl--) {
            double bits = rd_mag_bits(m, c, gt1_ctx(t, g), lvl) + 1.0;
            if (i < end - 1) bits += rd_bit(&m->sig[mc][c][sp], 1) + rd_bit(&m->last[mc][c][pc], 0);
            double j = (a - lvl) * (a - lvl) + lam * bits;
            if (j < best) { best = j; bl = lvl; }
        }
        lv[i] = v[i] < 0 ? -bl : bl;
        grid[p] = mag3(bl);
        cost[i] = best;
        if (bl > 1) g++;
    }
    /* The last level: coding up to k costs what positions start..k cost,
     * the flag that says k is last instead of not, and the error of
     * everything after k. */
    double tail = 0.0;
    for (int i = start; i < end; i++) tail += drop[i];
    double best = tail + lam * rd_bit(&m->cbf[mc][c], 0);
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
        double flag = k < end - 1 ? rd_bit(&m->last[mc][c][pc], 1) - rd_bit(&m->last[mc][c][pc], 0) : 0.0;
        double j = prefix + tail + lam * (rd_bit(&m->cbf[mc][c], 1) + flag);
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
                /* With a base, what mode INTER leaves: the block minus the
                 * base's; otherwise the block, whose flat colour moves only
                 * the DC. */
                const uint8_t* bp = cv->inter ? cv->base[c] : NULL;
                for (int j = 0; j < n; j++)
                    for (int i = 0; i < n; i++) {
                        size_t at = (size_t)(y + j) * cv->pw + x + i;
                        blk[j * n + i] = src[at] - (bp ? bp[at] : 0);
                    }
                double sum = 0.0;
                for (int j = 0; j < n; j++)
                    for (int i = 0; i < n; i++) sum += blk[j * n + i];
                P->pre_sum[s][c][b] = sum;
                forward_tx(s, TX_DCT, blk, co);
                int* q = dst + (size_t)b * count;
                int nonzero = 0;
                q[0] = 0;
                if (e->rdoq) {
                    /* Each band against its own layer's models. */
                    double v[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
                    for (int i = 1; i < count; i++) v[i] = co[scan_pos[s][i]] / step;
                    double lam = P->lambda_base / ((double)base * base) *
                                 (cv->fixed_pred || cv->inter ? RDOQ_RESIDUAL : RDOQ_PICTURE);
                    int at = e->band_at[s];
                    /* In a picture the lowest frequencies are left to the
                     * dead zone: they draw ramps, and dropping their level
                     * 1s turned the gate's gradient into bands (-6 dB for
                     * 18% of its bytes). */
                    int keep = cv->fixed_pred || cv->inter ? 0 : RDOQ_KEEP;
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
                    texture_residual(s, TX_DCT, q, 1, at, step, P->pre_lo[s][c] + (size_t)b * count);
                    if (at < count) texture_residual(s, TX_DCT, q, at, count, step, P->pre_hi[s][c] + (size_t)b * count);
                }
            }
        }
    }
}

/*
 * Rate-distortion search over one node. The node is coded whole, then cut
 * in two each way where it may be, then as four children, all against
 * the models as they stand, and the cheapest stays in the canvas.
 * Children see their earlier siblings' colours, as the decoder will. A
 * node that runs past the canvas has no whole option and must split; one
 * wholly past it does not exist. The decision: 0 whole, 1 split, 2 cut
 * across, 3 cut down.
 */
/*
 * Every cost here is an error plus lambda times bits, never negative, so
 * a split stops mattering as soon as its first quarters cost more than
 * the node whole: the rest are not tried. `budget` carries the same bound
 * down: the cost past which the caller's choice cannot change, so a node
 * that will not fit stops there too, and returns something no smaller
 * than its budget. Only a choice that loses is ever cut short, so the
 * decisions, and the container, are what the full search makes.
 */
static double search(Enc* e, int x, int y, int n, double budget) {
    Part* P = e->p;
    Canvas* cv = &P->cv;
    size_t id = node_id(e, x, y, n);
    int whole = node_whole(cv, x, y, n);
    int can_split = n > cv->min_block, flagged = whole && can_halve(cv, n), halves = flagged && e->try_halves;
    int sc = size_class(n), tree = cv->comp0 ? 1 : 0, choice = 1;
    double best = 1e300, split_cost = 0.0;
    uint8_t *kept = NULL, *orig = NULL;
    int stop = 0;

    if (whole) {
        if (halves) {
            orig = (uint8_t*)malloc((size_t)12 * n * n);
            if (orig) save_block(cv, x, y, n, orig);
            else halves = 0;
        }
        Sink s0 = { NULL, 0 }, s1[2] = { { NULL, 0 }, { NULL, 0 } };
        if (can_split) put_bit(&s0, split_model(&e->cm, cv, n), 0);
        if (flagged) put_bit(&s0, &e->cm.half[sc][tree], 0);
        int tex = 0;
        double d = code_leaf(e, &s0, s1, x, y, n, n, &tex);
        best = d + P->lambda * (s0.bits + s1[0].bits + s1[1].bits);
        choice = 0;
        /* In a predicted frame, a node the motion explains outright, INTER
         * with no texture, is not cut: on 9 frames of the clean pan that
         * made the encoder 22% faster for the same bytes and quality. */
        if (halves && cv->inter && e->last_mode == MODE_INTER && !tex) halves = 0;
        /* A picture without a base stops at a node that needs no texture
         * whole (as HEVC's reference encoder stops a coding unit whose
         * prediction leaves nothing to code): neither halves nor quarters
         * are tried. On the six samples, directional, that was 6% faster
         * and 0.2% smaller (progressive: 7%, the same size); in predicted
         * frames it cost 0.9 to 1.1%, and in an album's residuals 3.9%. */
        stop = !tex && !cv->inter && P->skip_lambda == 0.0;
        if (stop) halves = 0;
        if (!can_split) { P->split[id] = 0; free(orig); return best; }
        kept = (uint8_t*)malloc((size_t)12 * n * n);
        if (kept) save_block(cv, x, y, n, kept);
        for (int down = 0; halves && down < 2; down++) {
            int m = n / 2;
            load_block(cv, x, y, n, orig);
            Sink a0 = { NULL, 0 }, a1[2] = { { NULL, 0 }, { NULL, 0 } };
            put_bit(&a0, split_model(&e->cm, cv, n), 0);
            put_bit(&a0, &e->cm.half[sc][tree], 1);
            put_bit(&a0, &e->cm.half_dir[sc][tree], down);
            double dd = down ? code_leaf(e, &a0, a1, x, y, m, n, NULL) : code_leaf(e, &a0, a1, x, y, n, m, NULL);
            /* the first half alone already costs more: the second cannot help */
            if (dd + P->lambda * (a0.bits + a1[0].bits + a1[1].bits) >= best) continue;
            dd += down ? code_leaf(e, &a0, a1, x + m, y, m, n, NULL) : code_leaf(e, &a0, a1, x, y + m, n, m, NULL);
            double cost = dd + P->lambda * (a0.bits + a1[0].bits + a1[1].bits);
            if (cost < best) {
                best = cost;
                choice = 2 + down;
                if (kept) save_block(cv, x, y, n, kept);
            }
        }
        if (orig) load_block(cv, x, y, n, orig);
        free(orig);
    }

    if (whole && stop && kept) {
        load_block(cv, x, y, n, kept); free(kept); P->split[id] = choice; return best;
    }
    Sink s0 = { NULL, 0 };
    if (whole) put_bit(&s0, split_model(&e->cm, cv, n), 1);
    split_cost = P->lambda * s0.bits;
    int h = n / 2;
    double limit = budget < best ? budget : best;
    for (int k = 0; k < 4 && split_cost < limit; k++) {
        int cx = x + (k & 1) * h, cy = y + (k >> 1) * h;
        if (node_exists(cv, cx, cy)) split_cost += search(e, cx, cy, h, limit - split_cost);
    }

    if (whole && kept && best <= split_cost) {
        load_block(cv, x, y, n, kept);
        free(kept);
        P->split[id] = choice;
        return best;
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
    if (n > cv->min_block && whole) put_bit(s0, split_model(&e->cm, cv, n), split == 1);
    if (n > cv->min_block && (split == 1 || !whole)) {
        int half = n / 2;
        for (int k = 0; k < 4; k++) {
            int cx = x + (k & 1) * half, cy = y + (k >> 1) * half;
            if (node_exists(cv, cx, cy)) emit(e, s0, s1, cx, cy, half, h);
        }
        return;
    }
    if (can_halve(cv, n)) {
        int sc = size_class(n), tree = cv->comp0 ? 1 : 0;
        put_bit(s0, &e->cm.half[sc][tree], split >= 2);
        if (split >= 2) put_bit(s0, &e->cm.half_dir[sc][tree], split == 3);
    }
    int textured = 0, m = n / 2;
    if (split >= 2) {
        int w = split == 3 ? m : n, hh = split == 3 ? n : m;
        for (int k = 0; k < 2; k++) {
            textured = 0;
            code_leaf(e, s0, s1, split == 3 ? x + k * m : x, split == 3 ? y : y + k * m, w, hh, &textured);
            h->leaves[size_class(m)]++;
            h->textured += (uint32_t)textured;
        }
        return;
    }
    code_leaf(e, s0, s1, x, y, n, n, &textured);
    h->leaves[size_class(n)]++;
    h->textured += (uint32_t)textured;
}

int nvdr_encode_mem(uint8_t** out_buf, size_t* out_len, const NvdrImage* img,
                    const NvdrConfig* cfg_in, NvdrHeader* hdr_out) {
    return nvdr_encode_mem_ctx(out_buf, out_len, img, cfg_in, hdr_out, NULL);
}

static int decode_once(const uint8_t* data, size_t size, const size_t* holes, int nholes, int max_layer,
                       const NvdrImage* base, NvdrImage* out, NvdrHeader* hdr_out, NvdrDecodeInfo* info,
                       NvdrContext* ctx, int careful, NvdrRestoreFit* fit);
static int add_restoration(uint8_t** buf, size_t* len, NvdrHeader* h, const Enc* e, const NvdrImage* img,
                           const NvdrImage* base, NvdrContext* start, int restore, int choose_deblock);

/* A growing byte string: the layers' slices, end to end. */
typedef struct { uint8_t* b; size_t n, cap; } Bytes;
static int bytes_put(Bytes* o, const uint8_t* p, size_t n) {
    if (o->n + n > o->cap) {
        size_t cap = o->cap ? o->cap : 4096;
        while (cap < o->n + n) cap *= 2;
        uint8_t* g = (uint8_t*)realloc(o->b, cap);
        if (!g) return -1;
        o->b = g; o->cap = cap;
    }
    if (n) memcpy(o->b + o->n, p, n);
    o->n += n;
    return 0;
}
static int bytes_leb(Bytes* o, uint32_t v) {
    uint8_t t[5];
    int n = 0;
    do { t[n] = (uint8_t)(v & 127); v >>= 7; if (v) t[n] |= 128; n++; } while (v);
    return bytes_put(o, t, (size_t)n);
}

/* A tree's buffers for the row of tiles being searched (precompute_row()). */
static int part_scratch(Part* P) {
    Canvas* pc = &P->cv;
    for (int s = 0; s < NSIZES; s++)
        for (int c = 0; c < 3; c++)
            P->pre[s][c] = P->pre_lo[s][c] = P->pre_hi[s][c] = NULL, P->pre_sum[s][c] = NULL, P->pre_nz[s][c] = NULL;
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
                return -1;
        }
    }
    return 0;
}
static void part_scratch_free(Part* P) {
    for (int s = 0; s < NSIZES; s++)
        for (int c = 0; c < 3; c++) {
            free(P->pre[s][c]); free(P->pre_lo[s][c]); free(P->pre_hi[s][c]);
            free(P->pre_sum[s][c]); free(P->pre_nz[s][c]);
        }
}

static void precompute_row(Enc* e, int ty);
static double search(Enc* e, int x, int y, int n, double budget);
static void emit(Enc* e, Sink* s0, Sink* s1, int x, int y, int n, NvdrHeader* h);

/* The rows of tiles from ty0 up to ty1 (luma's rows), searched and coded.
 * Each tile is searched against the models as they stand, then coded for
 * real, which is what adapts them for the next. In 4:2:0 a tile is
 * luma's tree, then colour's over the same area at half size. */
static void code_rows(Enc* e, int ty0, int ty1, Sink* s0, Sink* s1, NvdrHeader* h,
                      const double* lambda0, const double* skip0) {
    Canvas* cv = &e->part[0].cv;
    int tq_prev = 0;
    for (int ty = ty0; ty < ty1; ty += cv->tile) {
        for (int k = 0; k < e->nparts; k++) {
            e->p = &e->part[k];
            if (ty / (k + 1) < e->p->cv.ph) precompute_row(e, ty / (k + 1));
        }
        for (int tx = 0; tx < cv->pw; tx += cv->tile) {
            if (e->tq) {
                /* The tile's offset, then its trees at its step; lambda
                 * follows the step squared, as it does across q. */
                int d = e->tq[(ty / cv->tile) * e->tiles_x + tx / cv->tile];
                put_tq(s0, &e->cm, d - tq_prev);
                tq_prev = d;
                for (int c = 0; c < 3; c++) e->step[c] = tq_step(e->base_step[c], d);
                double r = TQ_SCALE[d + TQ_MAX] / 1024.0;
                for (int k = 0; k < e->nparts; k++) {
                    e->part[k].lambda = lambda0[k] * r * r;
                    e->part[k].skip_lambda = skip0[k] * r * r;
                }
            }
            for (int k = 0; k < e->nparts; k++) {
                e->p = &e->part[k];
                int x = tx / (k + 1), y = ty / (k + 1);
                if (!node_exists(&e->p->cv, x, y)) continue;
                search(e, x, y, e->p->cv.tile, 1e300);
                emit(e, s0, s1, x, y, e->p->cv.tile, h);
            }
        }
    }
}

/* What coding one slice leaves (SLICES): its part of each layer, and its
 * share of the header's counts. */
typedef struct {
    NvdrEncoder enc[NVDR_LAYERS];
    NvdrHeader  h;
    double      plane_bits[3];
    int         failed;
} SliceOut;

/*
 * One slice, coded on its own from the models the picture starts with:
 * a copy of the encoder's state with its own row buffers, so slices are
 * coded side by side, one per core. They share the canvas, each writing
 * its own rows and reading nothing above them, and the tree's decisions,
 * each its own nodes. `end` takes the models the slice leaves.
 */
static void code_slice(const Enc* start, int ty0, int ty1, SliceOut* o, const double* lambda0,
                       const double* skip0, Enc* end) {
    Enc* w = (Enc*)malloc(sizeof(Enc));
    if (!w) { o->failed = 1; return; }
    *w = *start;
    w->pc = NULL;
    w->pc_part = NULL;
    for (int k = 0; k < 2; k++)
        for (int s = 0; s < NSIZES; s++)
            for (int c = 0; c < 3; c++)
                w->part[k].pre[s][c] = w->part[k].pre_lo[s][c] = w->part[k].pre_hi[s][c] = NULL,
                w->part[k].pre_sum[s][c] = NULL, w->part[k].pre_nz[s][c] = NULL;
    int bad = 0;
    for (int k = 0; k < w->nparts; k++) {
        bad |= part_scratch(&w->part[k]);
        w->part[k].cv.slice_top = ty0 / (k + 1);
    }
    if (start->pc && !(w->pc = (int*)malloc(sizeof(int) * PC_SLOTS * 3 * NVDR_MAX_BLOCK * NVDR_MAX_BLOCK))) bad = 1;
    for (int k = 0; k < NVDR_LAYERS; k++) bad |= nvdr_enc_init(&o->enc[k], k ? 1 << 14 : 1 << 12);
    if (!bad) {
        Sink s0 = { &o->enc[0], 0 }, s1[2] = { { &o->enc[1], 0 }, { &o->enc[2], 0 } };
        code_rows(w, ty0, ty1, &s0, s1, &o->h, lambda0, skip0);
        for (int k = 0; k < NVDR_LAYERS; k++) bad |= nvdr_enc_finish(&o->enc[k]);
        for (int c = 0; c < 3; c++) o->plane_bits[c] = w->plane_bits[c];
        if (end) { end->cm = w->cm; end->tm = w->tm; end->tm2 = w->tm2; }
    }
    o->failed = bad;
    for (int k = 0; k < 2; k++) part_scratch_free(&w->part[k]);
    free(w->pc);
    free(w);
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
    uint32_t* seg = NULL;
    SliceOut* so = NULL;
    int nslices_done = 0;   /* how many of `so` hold coders to free */
    Bytes lay[3];
    memset(lay, 0, sizeof(lay));
    NvdrEncoder enc0, enc1, enc2;
    memset(&enc0, 0, sizeof(enc0));
    memset(&enc1, 0, sizeof(enc1));
    memset(&enc2, 0, sizeof(enc2));
    int band = cfg.band < 0 ? 0 : (cfg.band > 32 ? 32 : cfg.band);
    int inter = cfg.base != NULL && !cfg.residual && cfg.base->width == img->width &&
                cfg.base->height == img->height;
    int dirpred = (cfg.directional || inter) && !cfg.residual;
    if (dirpred) band = 0;
    NvdrHeader h;
    memset(&h, 0, sizeof(h));
    /* Restoration is fitted by decoding what was coded, from the context
     * the container starts from. */
    int restore = cfg.restore && !cfg.residual;
    NvdrContext* start = NULL;
    if (restore && ctx) {
        if (!(start = nvdr_context_new())) return -1;
        nvdr_context_copy(start, ctx);
    }

    /* 4:2:0 needs a colour tile of at least the smallest leaf. */
    int use420 = cfg.chroma420 && cfg.max_block >= 8 && cfg.max_block <= NVDR_MAX_BLOCK &&
                 (cfg.max_block & (cfg.max_block - 1)) == 0;
    e.nparts = use420 ? 2 : 1;
    if (dirpred && !(e.pc = (int*)malloc(sizeof(int) * PC_SLOTS * 3 * NVDR_MAX_BLOCK * NVDR_MAX_BLOCK))) goto done;
    if (canvas_init(&e.part[0].cv, img->width, img->height, cfg.max_block, cfg.min_block,
                    use420 ? 1 : 3, 0) != 0) goto done;
    if (use420 && canvas_init(&e.part[1].cv, (img->width + 1) / 2, (img->height + 1) / 2,
                              cfg.max_block / 2, NVDR_MIN_BLOCK, 2, 1) != 0) goto done;
    for (int k = 0; k < e.nparts; k++) {
        Canvas* pc = &e.part[k].cv;
        pc->fixed_pred = cfg.residual != 0;
        pc->dirpred = dirpred;
        pc->inter = inter;
        pc->txsel = dirpred && cfg.adaptive_tx;
        pc->halves = dirpred;
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

    if (inter && canvas_base(&e.part[0].cv, e.nparts == 2 ? &e.part[1].cv : NULL, cfg.base) != 0) goto done;
    int qc = (int)(cfg.q * cfg.chroma_q + 0.5f);
    if (qc < 1) qc = 1;
    if (qc > 4095) qc = 4095;
    e.step[0] = cfg.q; e.step[1] = e.step[2] = qc;
    for (int c = 0; c < 3; c++) e.base_step[c] = e.step[c];
    e.deadzone = cfg.deadzone;
    e.try_halves = cfg.halves;
    e.tx_modes = cfg.tx_modes;
    e.fast_rank = cfg.fast_rank;
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
    for (int s = 0; s < NSHAPES; s++) e.band_at[s] = band_split(s, band);

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
        if (!P->split || part_scratch(P) != 0) goto done;
    }

    if (ctx && ctx->valid) {
        e.cm = ctx->cm; e.tm = ctx->tm; e.tm2 = ctx->tm2;
    } else {
        models_fill((NvdrProb*)&e.cm, sizeof(e.cm) / sizeof(NvdrProb));
        models_fill((NvdrProb*)&e.tm, sizeof(e.tm) / sizeof(NvdrProb));
        models_fill((NvdrProb*)&e.tm2, sizeof(e.tm2) / sizeof(NvdrProb));
    }
    if (nvdr_enc_init(&enc0, 1 << 14) != 0 || nvdr_enc_init(&enc1, 1 << 16) != 0 ||
        nvdr_enc_init(&enc2, 1 << 16) != 0) goto done;
    Sink s0 = { &enc0, 0 }, s1[2] = { { &enc1, 0 }, { &enc2, 0 } };

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
    /* Slices, each from the models the picture starts from, side by side;
     * a picture predicted from a base, or a residual, has none. */
    int tiles_y = (cv->ph + cv->tile - 1) / cv->tile;
    int srows = cfg.slices;
    if (srows == NVDR_SLICES_AUTO) {
        /* About sixteen slices, of 4 to 16 rows: by the picture's height,
         * never by the machine, so the file is the same everywhere. */
        srows = tiles_y / 16;
        srows = tiles_y < 8 ? 0 : srows < 4 ? 4 : srows > 16 ? 16 : srows;
    }
    srows = !inter && !cfg.residual && srows > 0 ? (srows > 255 ? 255 : srows) : 0;
    int nslices = srows ? (tiles_y + srows - 1) / srows : 1;
    const uint8_t* lp[3];
    size_t ln[3];
    Bytes table;
    memset(&table, 0, sizeof(table));
    /* Version 18: in slices, or starting from the priors, or both. */
    int v18 = srows || (cfg.priors && !inter && !cfg.residual);
    if (v18 && !(ctx && ctx->valid)) models_prior(&e.cm, &e.tm, &e.tm2, dirpred);
    if (srows) {
        if (!(seg = (uint32_t*)calloc((size_t)NVDR_LAYERS * nslices, sizeof(uint32_t))) ||
            !(so = (SliceOut*)calloc((size_t)nslices, sizeof(SliceOut)))) goto done;
        nslices_done = nslices;
        Enc* end = (Enc*)malloc(sizeof(Enc));
        if (!end) goto done;
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < nslices; i++) {
            int ty0 = i * srows * cv->tile, ty1 = ty0 + srows * cv->tile;
            code_slice(&e, ty0, ty1 < cv->ph ? ty1 : cv->ph, &so[i], lambda0, skip0, i == nslices - 1 ? end : NULL);
        }
        int bad = 0;
        for (int i = 0; i < nslices; i++) bad |= so[i].failed;
        if (!bad) { e.cm = end->cm; e.tm = end->tm; e.tm2 = end->tm2; }
        free(end);
        uint8_t r = (uint8_t)srows;
        bad |= bytes_put(&table, &r, 1);
        for (int i = 0; i < nslices && !bad; i++) {
            /* With no high band the third layer holds no symbols, only
             * the coder's flush, and is left out. */
            for (int k = 0; k < NVDR_LAYERS; k++) {
                size_t n = (k == 2 && !band) ? 0 : so[i].enc[k].count;
                bad |= bytes_put(&lay[k], so[i].enc[k].bytes, n);
                seg[k * nslices + i] = (uint32_t)n;
            }
            for (int k = 0; k < 4; k++) h.leaves[k] += so[i].h.leaves[k];
            h.textured += so[i].h.textured;
            for (int c = 0; c < 3; c++) e.plane_bits[c] += so[i].plane_bits[c];
        }
        for (int k = 0; k < NVDR_LAYERS; k++)
            for (int i = 0; i < nslices - 1; i++) bad |= bytes_leb(&table, seg[k * nslices + i]);
        if (bad) { free(table.b); goto done; }
        for (int k = 0; k < NVDR_LAYERS; k++) { lp[k] = lay[k].b; ln[k] = lay[k].n; }
    } else {
        uint8_t none = 0;
        if (v18 && bytes_put(&table, &none, 1) != 0) goto done;
        code_rows(&e, 0, cv->ph, &s0, s1, &h, lambda0, skip0);
        if (nvdr_enc_finish(&enc0) != 0 || nvdr_enc_finish(&enc1) != 0 ||
            nvdr_enc_finish(&enc2) != 0) goto done;
        /* With no high band the stream holds no symbols, only the coder's
         * flush: leave it out, and the layer is empty. */
        lp[0] = enc0.bytes; ln[0] = enc0.count;
        lp[1] = enc1.bytes; ln[1] = enc1.count;
        lp[2] = enc2.bytes; ln[2] = band ? enc2.count : 0;
    }
    for (int c = 0; c < 3; c++) e.step[c] = e.base_step[c];
    size_t n2 = ln[2];

    size_t total = NVDR_HEADER_SIZE + table.n + ln[0] + ln[1] + n2;
    uint8_t* buf = (uint8_t*)malloc(total);
    if (!buf) { free(table.b); goto done; }
    memset(buf, 0, NVDR_HEADER_SIZE);
    memcpy(buf, NVDR_MAGIC, 4);
    buf[4] = v18 ? NVDR_VERSION_SLICES : NVDR_VERSION;
    buf[5] = (uint8_t)((cfg.residual ? NVDR_FLAG_RESIDUAL : 0) | (cfg.deblock ? NVDR_FLAG_DEBLOCK : 0) |
                       (use420 ? NVDR_FLAG_CHROMA420 : 0) | (e.tq ? NVDR_FLAG_TILEQ : 0) |
                       (dirpred ? NVDR_FLAG_DIRPRED : 0) | (inter ? NVDR_FLAG_INTER : 0) |
                       (dirpred && cfg.adaptive_tx ? NVDR_FLAG_TXSEL : 0));
    put_u16(buf + 6, (uint32_t)img->width);
    put_u16(buf + 8, (uint32_t)img->height);
    buf[10] = (uint8_t)cfg.max_block;
    buf[11] = (uint8_t)cfg.min_block;
    put_u16(buf + 12, (uint32_t)e.step[0]);
    put_u16(buf + 14, (uint32_t)e.step[1]);
    put_u32(buf + 16, (uint32_t)ln[0]);
    put_u32(buf + 20, (uint32_t)ln[1]);
    put_u32(buf + 24, (uint32_t)n2);
    /* A directional picture's deblocking levels go where its band (0)
     * would; add_restoration() chooses them. */
    buf[28] = (uint8_t)(dirpred ? (cfg.deblock ? DB_LEVEL | DB_LEVEL << 4 : 0) : band);
    {
        uint8_t* at = buf + NVDR_HEADER_SIZE;
        if (table.n) memcpy(at, table.b, table.n);
        at += table.n;
        for (int k = 0; k < NVDR_LAYERS; k++) { if (ln[k]) memcpy(at, lp[k], ln[k]); at += ln[k]; }
    }

    h.width = (uint16_t)img->width; h.height = (uint16_t)img->height;
    h.max_block = (uint8_t)cfg.max_block; h.min_block = (uint8_t)cfg.min_block;
    h.q_luma = (uint16_t)e.step[0]; h.q_chroma = (uint16_t)e.step[1];
    h.flags = (uint8_t)buf[5];
    h.version = buf[4];
    h.slice_rows = (uint8_t)srows;
    h.table_len = (uint32_t)table.n;
    free(table.b);
    h.stored_bytes[0] = (uint32_t)ln[0];
    h.stored_bytes[1] = (uint32_t)ln[1];
    h.stored_bytes[2] = (uint32_t)n2;
    h.band = (uint8_t)band;
    h.deblock[0] = h.deblock[1] = dirpred && !cfg.deblock ? 0 : DB_LEVEL;
    for (int c = 0; c < 3; c++) h.plane_bits[c] = e.plane_bits[c];
    int choose_deblock = dirpred && cfg.deblock;
    if ((restore || choose_deblock) &&
        add_restoration(&buf, &total, &h, &e, img, inter ? cfg.base : NULL, start, restore, choose_deblock) != 0) {
        free(buf); goto done;
    }
    if (hdr_out) *hdr_out = h;
    *out_buf = buf;
    *out_len = total;
    if (ctx) { ctx->cm = e.cm; ctx->tm = e.tm; ctx->tm2 = e.tm2; ctx->valid = 1; }
    rc = 0;

done:
    nvdr_context_free(start);
    free(tq);
    free(seg);
    if (so) for (int i = 0; i < nslices_done; i++) for (int k = 0; k < NVDR_LAYERS; k++) nvdr_enc_free(&so[i].enc[k]);
    free(so);
    for (int k = 0; k < NVDR_LAYERS; k++) free(lay[k].b);
    free(e.pc);
    nvdr_enc_free(&enc0);
    nvdr_enc_free(&enc1);
    nvdr_enc_free(&enc2);
    for (int k = 0; k < 2; k++) {
        Part* P = &e.part[k];
        for (int c = 0; c < 3; c++) free(P->src[c]);
        free(P->split);
        part_scratch_free(P);
        canvas_free(&P->cv);
    }
    return rc;
}

/*
 * Fits restoration to the container just coded (see restore.c): decodes
 * it, handing the decoded planes and the source to the fit, and puts the
 * parameters it chose after the header, flagged, when any component
 * chose one. The container is left as it was when none did. With
 * choose_deblock the same decode chooses a directional picture's
 * deblocking levels (see deblock()), which go in the header either way.
 */
static int add_restoration(uint8_t** buf, size_t* len, NvdrHeader* h, const Enc* e, const NvdrImage* img,
                           const NvdrImage* base, NvdrContext* start, int restore, int choose_deblock) {
    NvdrRestoreFit fit;
    memset(&fit, 0, sizeof(fit));
    fit.restore = restore;
    fit.choose_deblock = choose_deblock;
    for (int k = 0; k < e->nparts; k++)
        for (int c = 0; c < e->part[k].cv.np; c++) {
            int comp = e->part[k].cv.comp0 + c;
            fit.src[comp] = e->part[k].src[c];
            fit.lambda[comp] = e->part[k].lambda_base;
        }
    /* Colour at half size is judged at full size, where it is seen, at
     * luma's lambda. */
    double* full[3] = { NULL, NULL, NULL };
    const Canvas* cv = &e->part[0].cv;
    if (e->nparts == 2) {
        for (int c = 1; c < 3; c++) {
            if (!(full[c] = (double*)malloc(sizeof(double) * cv->pw * cv->ph))) { free(full[1]); return -1; }
            fit.full_src[c] = full[c];
            fit.lambda[c] = e->part[0].lambda_base;
        }
        for (int y = 0; y < cv->h; y++)
            for (int x = 0; x < cv->w; x++) {
                double o[3];
                rgb_to_ycc(img->pixels + ((size_t)y * cv->w + x) * 3, o);
                full[1][(size_t)y * cv->pw + x] = o[1];
                full[2][(size_t)y * cv->pw + x] = o[2];
            }
        fit.full_pw = cv->pw; fit.full_w = cv->w; fit.full_h = cv->h;
    }
    NvdrImage shown;
    int rc = decode_once(*buf, *len, NULL, 0, -1, base, &shown, NULL, NULL, start, 1, &fit);
    free(shown.pixels);
    free(full[1]); free(full[2]);
    if (rc != 0 || fit.failed) { nvdr_bits_free(&fit.out); return -1; }
    if (choose_deblock) {
        (*buf)[28] = (uint8_t)(fit.deblock[0] | fit.deblock[1] << 4);
        h->deblock[0] = (uint8_t)fit.deblock[0];
        h->deblock[1] = (uint8_t)fit.deblock[1];
    }
    int used = 0;
    for (int c = 0; c < 3; c++) if (fit.bits[c] > 3) used = 1;
    if (!used || fit.out.count > 0xffff) { nvdr_bits_free(&fit.out); return 0; }
    size_t n = fit.out.count, total = *len + n;
    uint8_t* out = (uint8_t*)malloc(total);
    if (!out) { nvdr_bits_free(&fit.out); return -1; }
    memcpy(out, *buf, NVDR_HEADER_SIZE);
    memcpy(out + NVDR_HEADER_SIZE, fit.out.bytes, n);
    memcpy(out + NVDR_HEADER_SIZE + n, *buf + NVDR_HEADER_SIZE, *len - NVDR_HEADER_SIZE);
    put_u16(out + 30, (uint32_t)n);
    nvdr_bits_free(&fit.out);
    free(*buf);
    *buf = out; *len = total;
    h->restore_len = (uint16_t)n;
    return 0;
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
    if (cfg.grain == NVDR_GRAIN_OFF || cfg.residual || cfg.base || !img->pixels)
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
