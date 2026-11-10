/* Part of nvdr.c, which includes it:
 * the decoder: layer 0, the texture layers, deblocking, the picture out. */

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



typedef struct { uint16_t x, y; uint8_t w, h, tex; } Leaf;

typedef struct {
    Canvas*        cv;
    NvdrDecoder*   d;
    ColourModels*  cm;
    const int*     step;
    /* With directional prediction, each leaf's texture comes from the
     * first texture layer as soon as its colour does. */
    NvdrDecoder*   d1;
    TextureModels* tm;
    Leaf*          leaves;
    size_t         count, cap;
    int            corrupt;
} Layer0;

static int push_leaf(Layer0* L, int x, int y, int w, int h) {
    if (L->count == L->cap) {
        size_t cap = L->cap ? L->cap * 2 : 4096;
        Leaf* p = (Leaf*)realloc(L->leaves, cap * sizeof(Leaf));
        if (!p) return -1;
        L->leaves = p; L->cap = cap;
    }
    L->leaves[L->count].x = (uint16_t)x;
    L->leaves[L->count].y = (uint16_t)y;
    L->leaves[L->count].w = (uint8_t)w;
    L->leaves[L->count].h = (uint8_t)h;
    L->leaves[L->count].tex = 0;
    L->count++;
    return 0;
}

static int read_leaf(Layer0* L, int x, int y, int w, int h) {
    Canvas* cv = L->cv;
    int sh = shape_of(w, h), mc = SHAPE_MC[sh];
    int mode = 0;
    if (cv->dirpred) {
        mode = get_mode(L->d, L->cm, mc, cv->inter, cv->legacy_modes);
        if (mode < 0) { L->corrupt = 1; return 0; }
    }
    int tx = has_tx(cv, w, h, mode) ? get_tx(L->d, L->cm, mc, mode) : TX_DCT;
    if (tx < 0) { L->corrupt = 1; return 0; }
    for (int c = 0; c < cv->np; c++) {
        int k = cv->comp0 + c;
        int dl = tx ? 0 : get_dc(L->d, L->cm, mc, k, &L->corrupt);
        int offset = dc_offset(clamp_coef((long)dl * L->step[k]), w, h);
        if (mode > 0) {
            int pr[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
            dir_predict(cv, c, x, y, w, h, mode, pr);
            paint_pred(cv, c, x, y, w, h, pr, offset);
        } else
            paint_flat(cv, c, x, y, w, h, clamp_u8(predict(cv, c, x, y, w, h) + offset));
    }
    if (push_leaf(L, x, y, w, h) != 0) return -1;
    int t0 = tx ? 0 : 1;
    if (cv->dirpred) {
        /* The texture now, since the next leaf predicts from it. */
        int lv[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK], count = w * h;
        for (int c = 0; c < cv->np && !L->corrupt && !L->d1->overrun; c++)
            if (get_texture(L->d1, L->tm, sh, cv->comp0 + c, lv, t0, count, &L->corrupt) &&
                !L->corrupt && !L->d1->overrun) {
                apply_texture(cv, c, x, y, sh, tx, lv, t0, count, L->step[cv->comp0 + c]);
                L->leaves[L->count - 1].tex = 1;
            }
        /* A texture layer that runs out is damage here: the leaves after
         * would predict from a picture the encoder never saw. */
        if (L->d1->overrun) L->corrupt = 1;
    }
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
    if (can_halve(cv, n)) {
        int sc = size_class(n), t = cv->comp0 ? 1 : 0;
        if (nvdr_dec_bit(L->d, &L->cm->half[sc][t])) {
            int down = nvdr_dec_bit(L->d, &L->cm->half_dir[sc][t]), m = n / 2;
            if (L->d->overrun) return 0;
            if (down) {
                if (read_leaf(L, x, y, m, n) != 0) return -1;
                if (L->d->overrun || L->corrupt) return 0;
                return read_leaf(L, x + m, y, m, n);
            }
            if (read_leaf(L, x, y, n, m) != 0) return -1;
            if (L->d->overrun || L->corrupt) return 0;
            return read_leaf(L, x, y + m, n, m);
        }
    }
    return read_leaf(L, x, y, n, n);
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
 *
 * A directional picture says how hard (container 16): a level for luma
 * and one for colour, 0 to 15 in eighths of the thresholds below, which
 * are level 8. Its predictions and its RDOQ leave seams the tuning never
 * saw, and they differ from picture to picture: against libaom, the
 * filter on colour cost one picture 8% of its colour's bitrate and saved
 * another 0.7%, and on luma it helped one picture and hurt two. The
 * encoder decodes the picture anyway to fit restoration, and there it
 * tries every level on each tree's planes and keeps the one that leaves
 * the least squared error (deblock()).
 */
/* alpha, beta and tc are step * 20/16, 6/16 and 3/16 at level 8. Swept
 * on the six samples at q 24 and 48: +0.24 and +0.28 dB on average, where
 * a filter twice as strong starts to cost PSNR. */
#define DB_ALPHA 20
#define DB_BETA   6
#define DB_TC     3
#define DB_LEVELS 16
static int db_param(int step, int k16, int level) { return (step * k16 * level + 64) >> 7; }

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
                          const int8_t* tq, int tile, int tiles_x, int level) {
    int gw = pw / 4, gh = ph / 4;
    if (!level) return;
    for (int pass = 0; pass < 2; pass++)
        for (int gy = pass; gy < gh; gy++)
            for (int gx = 1 - pass; gx < gw; gx++) {
                const uint8_t* edge = pass ? hedge : vedge;
                if (!edge[gy * gw + gx]) continue;
                int other = pass ? (gy - 1) * gw + gx : gy * gw + gx - 1;
                if (!edge_filtered(cell[gy * gw + gx], cell[other])) continue;
                int step = tq ? tq_step(base, tq[(gy * 4 / tile) * tiles_x + gx * 4 / tile]) : base;
                int alpha = db_param(step, DB_ALPHA, level), beta = db_param(step, DB_BETA, level);
                int tc = db_param(step, DB_TC, level);
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

/* The squared error of a plane's visible part against its source. */
static double plane_sse(const uint8_t* p, const double* src, int pw, int w, int h) {
    double e = 0.0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            double d = p[(size_t)y * pw + x] - src[(size_t)y * pw + x];
            e += d * d;
        }
    return e;
}

/* Edges on a 4-px grid: every leaf's left and top side, and every tile
 * that never arrived as one block. `cell` holds, per 4x4 cell, log2 of its
 * leaf's size in cells and 0x80 when the leaf shows texture; `textured` is
 * NULL when only colours are shown. Each tree's planes (luma's, colour's)
 * are filtered at its level, or with `fit` choosing at the level that
 * leaves the least error against the source, recorded there. */
static int deblock(const Canvas* cv, uint8_t** planes, const Leaf* leaves, const uint8_t* textured,
                   size_t count, const uint8_t* whole0, int tiles_x, int tiles, const int* step,
                   const int8_t* tq, const uint8_t level[2], NvdrRestoreFit* fit) {
    int gw = cv->pw / 4, gh = cv->ph / 4;
    uint8_t* vedge = (uint8_t*)calloc((size_t)gw * gh, 1);
    uint8_t* hedge = (uint8_t*)calloc((size_t)gw * gh, 1);
    uint8_t* cell = (uint8_t*)calloc((size_t)gw * gh, 1);
    if (!vedge || !hedge || !cell) { free(vedge); free(hedge); free(cell); return -1; }
    for (size_t i = 0; i < count; i++) {
        int x = leaves[i].x / 4, y = leaves[i].y / 4, w = leaves[i].w / 4, h = leaves[i].h / 4;
        int info = log2_int(w > h ? w : h) | (textured && textured[i] ? 0x80 : 0);
        for (int j = y; j < y + h && j < gh; j++)
            for (int k = x; k < x + w && k < gw; k++) cell[(size_t)j * gw + k] = (uint8_t)info;
        if (x > 0) for (int j = y; j < y + h && j < gh; j++) vedge[(size_t)j * gw + x] = 1;
        if (y > 0) for (int k = x; k < x + w && k < gw; k++) hedge[(size_t)y * gw + k] = 1;
    }
    for (int t = 0; t < tiles; t++) {
        if (whole0[t]) continue;
        int x = (t % tiles_x) * cv->tile / 4, y = (t / tiles_x) * cv->tile / 4, n = cv->tile / 4;
        for (int j = y; j < y + n && j < gh; j++)
            for (int k = x; k < x + n && k < gw; k++) cell[(size_t)j * gw + k] = (uint8_t)log2_int(n);
        if (x > 0) for (int j = y; j < y + n && j < gh; j++) vedge[(size_t)j * gw + x] = 1;
        if (y > 0) for (int k = x; k < x + n && k < gw; k++) hedge[(size_t)y * gw + k] = 1;
    }
    int rc = 0;
    uint8_t* trial = NULL;
    for (int t = 0; t < 2; t++) {
        /* luma's plane is the first of the luma canvas; the rest are colour */
        int c0 = t && !cv->comp0 ? 1 : 0, c1 = t ? cv->np : (cv->comp0 ? 0 : 1);
        if (c0 >= c1) continue;
        int lv = level[t];
        if (fit && fit->choose_deblock) {
            size_t n = (size_t)cv->pw * cv->ph;
            if (!trial && !(trial = (uint8_t*)malloc(n))) { rc = -1; break; }
            double best = 0.0;
            for (int l = 0; l < DB_LEVELS; l++) {
                double e = 0.0;
                for (int c = c0; c < c1; c++) {
                    memcpy(trial, planes[c], n);
                    deblock_plane(trial, cv->pw, cv->ph, cell, vedge, hedge, step[cv->comp0 + c], tq, cv->tile,
                                  tiles_x, l);
                    e += plane_sse(trial, fit->src[cv->comp0 + c], cv->pw, cv->w, cv->h);
                }
                if (!l || e < best) { best = e; lv = l; }
            }
            fit->deblock[t] = lv;
        }
        for (int c = c0; c < c1; c++)
            deblock_plane(planes[c], cv->pw, cv->ph, cell, vedge, hedge, step[cv->comp0 + c], tq, cv->tile,
                          tiles_x, lv);
    }
    free(trial);
    free(vedge); free(hedge); free(cell);
    return rc;
}

size_t nvdr_container_bytes(const uint8_t* data, size_t size) {
    NvdrHeader h;
    if (read_header(data, size, &h) != 0) return 0;
    return NVDR_HEADER_SIZE + (size_t)h.grain_len + h.restore_len + h.table_len +
           (size_t)h.stored_bytes[0] + h.stored_bytes[1] + h.stored_bytes[2];
}

int nvdr_decode_mem(const uint8_t* data, size_t size, int max_layer,
                    NvdrImage* out, NvdrHeader* hdr_out, NvdrDecodeInfo* info) {
    return nvdr_decode_mem_ctx(data, size, max_layer, out, hdr_out, info, NULL);
}

/* decode_once() when a tile of a layer that arrived whole stops, which
 * only damage makes it do: the decode starts over, saving tiles. */
#define DECODE_AGAIN (-2)

int nvdr_decode_mem_ctx(const uint8_t* data, size_t size, int max_layer, NvdrImage* out,
                        NvdrHeader* hdr_out, NvdrDecodeInfo* info, NvdrContext* ctx) {
    /* A texture layer that arrived whole cannot stop inside a tile unless
     * the file is damaged, so the first attempt does not save every tile
     * to restore it (a tenth of the time); if one stops, the decode starts
     * over the careful way and gives what that gives. The context is only
     * written when a decode finishes, so starting over is safe. */
    int rc = decode_once(data, size, NULL, 0, max_layer, NULL, out, hdr_out, info, ctx, 0, NULL);
    if (rc == DECODE_AGAIN) rc = decode_once(data, size, NULL, 0, max_layer, NULL, out, hdr_out, info, ctx, 1, NULL);
    return rc;
}

int nvdr_decode_mem_lossy(const uint8_t* data, size_t size, const size_t* missing, int n,
                          int max_layer, NvdrImage* out, NvdrHeader* hdr_out, NvdrDecodeInfo* info) {
    int rc = decode_once(data, size, missing, n, max_layer, NULL, out, hdr_out, info, NULL, 0, NULL);
    if (rc == DECODE_AGAIN) rc = decode_once(data, size, missing, n, max_layer, NULL, out, hdr_out, info, NULL, 1, NULL);
    return rc;
}

int nvdr_decode_mem_base(const uint8_t* data, size_t size, const NvdrImage* base,
                         NvdrImage* out, NvdrHeader* hdr_out, NvdrDecodeInfo* info, NvdrContext* ctx) {
    int rc = decode_once(data, size, NULL, 0, -1, base, out, hdr_out, info, ctx, 0, NULL);
    if (rc == DECODE_AGAIN) rc = decode_once(data, size, NULL, 0, -1, base, out, hdr_out, info, ctx, 1, NULL);
    return rc;
}

/* How much of the `len` bytes at `off` can be read: up to the end of the
 * data or the first byte of a hole, whichever comes first. */
static size_t readable(size_t size, const size_t* holes, int nholes, size_t off, size_t len) {
    size_t lim = off + len < size ? off + len : size;
    for (int i = 0; i < nholes; i++) {
        size_t a = holes[2 * i], b = holes[2 * i + 1];
        if (a < lim && b > off) lim = a > off ? a : off;
    }
    return lim > off ? lim - off : 0;
}

static int decode_once(const uint8_t* data, size_t size, const size_t* holes, int nholes, int max_layer,
                       const NvdrImage* base_img, NvdrImage* out, NvdrHeader* hdr_out, NvdrDecodeInfo* info,
                       NvdrContext* ctx, int careful, NvdrRestoreFit* fit) {
    out->pixels = NULL; out->width = out->height = 0;
    tables_init();
    NvdrHeader h;
    if (read_header(data, size, &h) != 0) return -1;
    if (hdr_out) *hdr_out = h;
    /* Everything before the first layer has to be there. */
    size_t base = NVDR_HEADER_SIZE + (size_t)h.grain_len + h.restore_len + h.table_len;
    if (readable(size, holes, nholes, 0, base) < base) return -1;

    /* Each layer's part of each slice (one slice without them): where it
     * starts, how long it is, and how much of it can be read. */
    int ns = slice_count(&h), srows = h.slice_rows ? h.slice_rows : tiles_down(&h);
    uint32_t* seg = (uint32_t*)malloc(sizeof(uint32_t) * NVDR_LAYERS * ns);
    size_t* seg_at = (size_t*)malloc(sizeof(size_t) * NVDR_LAYERS * ns);
    size_t* seg_ok = (size_t*)malloc(sizeof(size_t) * NVDR_LAYERS * ns);
    int* got = (int*)calloc((size_t)NVDR_LAYERS * ns, sizeof(int));   /* tiles each part carried */
    if (!seg || !seg_at || !seg_ok || !got) { free(seg); free(seg_at); free(seg_ok); free(got); return -1; }
    if (h.slice_rows) slice_table(data, size, base - h.table_len, &h, seg);
    else for (int k = 0; k < NVDR_LAYERS; k++) seg[k] = h.stored_bytes[k];
    size_t avail_layer[NVDR_LAYERS] = { 0, 0, 0 };
    /* The range decoder primes itself with five bytes; fewer than that
     * and not one symbol of a part can be read. */
    int any0 = 0;
    for (size_t k = 0, at = base; k < NVDR_LAYERS; k++)
        for (int i = 0; i < ns; i++) {
            size_t j = k * ns + i;
            seg_at[j] = at;
            seg_ok[j] = readable(size, holes, nholes, at, seg[j]);
            avail_layer[k] += seg_ok[j];
            if (k == 0 && seg_ok[j] >= 5) any0 = 1;
            at += seg[j];
        }
    if (!any0) { free(seg); free(seg_at); free(seg_ok); free(got); return -1; }
    size_t avail1 = avail_layer[1], avail2 = avail_layer[2];
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
    uint8_t* whole0 = NULL;
    int8_t* tq = NULL;
    if (canvas_init(&cvs[0], h.width, h.height, h.max_block, h.min_block, np == 2 ? 1 : 3, 0) != 0) goto done;
    if (np == 2 && canvas_init(&cvs[1], (h.width + 1) / 2, (h.height + 1) / 2, h.max_block / 2,
                               NVDR_MIN_BLOCK, 2, 1) != 0) goto done;
    Canvas* cv = &cvs[0];
    int dirpred = (h.flags & NVDR_FLAG_DIRPRED) != 0;
    for (int k = 0; k < np; k++) {
        cvs[k].fixed_pred = (h.flags & NVDR_FLAG_RESIDUAL) != 0;
        cvs[k].dirpred = dirpred;
        cvs[k].inter = (h.flags & NVDR_FLAG_INTER) != 0;
        cvs[k].txsel = (h.flags & NVDR_FLAG_TXSEL) != 0;
        cvs[k].legacy_modes = h.version < 14;
        cvs[k].halves = dirpred && h.version >= 15;
    }
    /* A picture predicted from a base cannot be decoded without it. */
    if (h.flags & NVDR_FLAG_INTER) {
        if (!base_img || base_img->width != h.width || base_img->height != h.height ||
            canvas_base(&cvs[0], np == 2 ? &cvs[1] : NULL, base_img) != 0) goto done;
    }
    int tiles_x = (cv->pw + cv->tile - 1) / cv->tile, tiles_y = (cv->ph + cv->tile - 1) / cv->tile;
    int tiles = tiles_x * tiles_y;
    for (int k = 0; k < np; k++) {
        tile_start[k] = (size_t*)malloc(sizeof(size_t) * (tiles + 1));
        if (!tile_start[k]) goto done;
    }
    if (!(whole0 = (uint8_t*)calloc((size_t)tiles, 1))) goto done;
    int step[3] = { h.q_luma, h.q_chroma, h.q_chroma };
    int tstep[3] = { step[0], step[1], step[2] };
    /* The tiles' step offsets, as layer 0 delivers them; zero where it
     * never did. */
    if ((h.flags & NVDR_FLAG_TILEQ) && !(tq = (int8_t*)calloc((size_t)tiles, 1))) goto done;

    /* Layer 0: every tile that arrives whole; the rest predicted. */
    ColourModels cm, cm0;
    TextureModels tms[2], tms0[2];
    if (ctx && ctx->valid) {
        cm = ctx->cm; tms[0] = ctx->tm; tms[1] = ctx->tm2;
    } else {
        models_fill((NvdrProb*)&cm, sizeof(cm) / sizeof(NvdrProb));
        for (int k = 0; k < 2; k++) models_fill((NvdrProb*)&tms[k], sizeof(TextureModels) / sizeof(NvdrProb));
    }
    if (h.version >= 18 && !(ctx && ctx->valid)) models_prior(&cm, &tms[0], &tms[1], dirpred);
    cm0 = cm; tms0[0] = tms[0]; tms0[1] = tms[1];
    NvdrDecoder d0;
    /* A directional picture's texture comes with its colours, from the
     * slice's part of layer 1. */
    NvdrDecoder d1;
    nvdr_dec_init(&d1, data + seg_at[ns], seg_ok[ns]);
    for (int k = 0; k < np; k++) {
        L[k].cv = &cvs[k]; L[k].d = &d0; L[k].cm = &cm; L[k].step = tstep;
        L[k].d1 = &d1; L[k].tm = &tms[0];
    }
    int complete0 = 0;
    for (int sl = 0; sl < ns; sl++) {
        /* Every slice from the models the picture starts with (SLICES). */
        int t0 = sl * srows * tiles_x, t1 = (sl + 1) * srows * tiles_x;
        if (t1 > tiles) t1 = tiles;
        cm = cm0;
        int stopped = seg_ok[sl] < 5, tq_prev = 0;
        if (!stopped) nvdr_dec_init(&d0, data + seg_at[sl], seg_ok[sl]);
        if (dirpred) {
            tms[0] = tms0[0];
            nvdr_dec_init(&d1, data + seg_at[ns + sl], seg_ok[ns + sl]);
        }
        /* Damage in one slice is not damage in the next. */
        for (int k = 0; k < np; k++) { cvs[k].slice_top = t0 / tiles_x * cvs[k].tile; L[k].corrupt = 0; }
        for (int t = t0; t < t1; t++) {
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
            } else { complete0++; got[sl]++; whole0[t] = 1; }
        }
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
        if (dirpred) for (size_t i = 0; i < L[k].count; i++) textured[k][i] = L[k].leaves[i].tex;
    }
    /* With directional prediction the texture came with the colours. */
    if (dirpred) {
        complete[1] = complete0; max_layer = -1;
        for (int sl = 0; sl < ns; sl++) got[ns + sl] = got[sl];
    }
    saved = (uint8_t*)malloc((size_t)cv->tile * cv->tile * 9);   /* full and acc, 3 planes */
    if (!saved) goto done;
    for (int layer = 1; layer < NVDR_LAYERS; layer++) {
      if (dirpred || (max_layer >= 0 && max_layer < layer)) break;
      if (layer == 2 && h.band == 0) break;
      for (int sl = 0; sl < ns; sl++) {
        size_t j = (size_t)layer * ns + sl, avail = seg_ok[j];
        int t0 = sl * srows * tiles_x, before = got[(layer - 1) * ns + sl];
        if (avail < 5 || before == 0) continue;
        TextureModels* tm = &tms[layer - 1];
        *tm = tms0[layer - 1];
        NvdrDecoder d;
        nvdr_dec_init(&d, data + seg_at[j], avail);
        int corrupt = 0;
        int lv[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
        int guard = careful || avail < seg[j];
        for (int t = t0; t < t0 + before; t++) {
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
                    /* Layered pictures are not directional: leaves are square. */
                    int sc = size_class(f->w), count = f->w * f->h;
                    int start = layer == 1 ? 1 : band_at[sc], end = layer == 1 ? band_at[sc] : count;
                    if (start >= end) continue;
                    for (int c = 0; c < pc->np && !d.overrun && !corrupt; c++) {
                        int comp = pc->comp0 + c;
                        if (get_texture(&d, tm, sc, comp, lv, start, end, &corrupt) && !d.overrun && !corrupt) {
                            apply_texture(pc, c, f->x, f->y, sc, TX_DCT, lv, start, end, tile_step[comp]);
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
            got[j]++;
        }
      }
    }

    /* Tiles that never arrived, between tiles that did (CONCEALMENT). */
    if (complete0 < tiles && !(h.flags & NVDR_FLAG_INTER))
        for (int k = 0; k < np; k++) conceal(&cvs[k], whole0, tiles_x, tiles_y);

    out->pixels = (unsigned char*)malloc((size_t)h.width * h.height * 3);
    if (!out->pixels) goto done;
    out->width = h.width; out->height = h.height;
    uint8_t* planes[3];
    /* Restoration was fitted to the whole picture, and only a whole
     * picture gets it; so does the encoder's own decode, which fits it. */
    int whole_picture = complete[0] == tiles && (dirpred || (complete[1] == tiles && (h.band == 0 || complete[2] == tiles))) &&
                        (max_layer < 0 || max_layer >= NVDR_LAYERS - 1);
    NvdrRestore rs;
    NvdrRestorePlane rg[3];
    int restoring = h.restore_len && whole_picture;
    if (restoring) {
        restore_planes(&h, rg);
        if (nvdr_restore_parse(data + NVDR_HEADER_SIZE + h.grain_len, h.restore_len, rg, &rs) != 0) restoring = 0;
    }
    for (int k = 0; k < np; k++) {
        uint8_t** pl = (max_layer == 0) ? cvs[k].flat : cvs[k].full;
        if ((h.flags & NVDR_FLAG_DEBLOCK) &&
            deblock(&cvs[k], pl, L[k].leaves, max_layer == 0 ? NULL : textured[k], tile_start[k][tiles],
                    whole0, tiles_x, tiles, step, tq, h.deblock, fit) != 0) {
            free(out->pixels); out->pixels = NULL; goto done;
        }
        for (int c = 0; c < cvs[k].np; c++) {
            int comp = cvs[k].comp0 + c;
#ifndef NVDR_WASM   /* only the encoder fits; keeps the fit out of the browser's decoder */
            NvdrRestorePlane pg = { cvs[k].pw, cvs[k].ph, k ? NVDR_RESTORE_UNIT / 2 : NVDR_RESTORE_UNIT };
            if (fit && fit->restore && nvdr_restore_fit(fit, comp, pl[c], &pg, cvs[k].w, cvs[k].h) != 0) {
                free(out->pixels); out->pixels = NULL; goto done;
            }
#endif
            if (restoring && nvdr_restore_apply(&rs.comp[comp], comp, pl[c], &rg[comp]) != 0) {
                free(out->pixels); out->pixels = NULL; goto done;
            }
            planes[comp] = pl[c];
        }
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
    free(whole0);
    free(seg); free(seg_at); free(seg_ok); free(got);
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
