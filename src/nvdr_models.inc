/* Part of nvdr.c, which includes it:
 * the adaptive models, the priors, and how every symbol is coded. */

/* =============================================================== models */

typedef struct {
    NvdrProb split[NSIZES];
    NvdrProb split_c[NSIZES];    /* the colour tree's, in 4:2:0 */
    NvdrProb half[NSIZES][2];    /* a whole node cut in two, by size and tree */
    NvdrProb half_dir[NSIZES][2];   /* down, not across */
    NvdrProb dc_zero[NMC][3];
    NvdrProb dc_sign[3];
    NvdrProb dc_mag[3][MAG_UNARY];
    NvdrProb tq_zero, tq_sign, tq_mag[2 * TQ_MAX];   /* the tile step offsets */
    NvdrProb mode_flat[NMC], mode_tree[64];          /* a leaf's prediction mode */
    NvdrProb mode_inter[NMC];                        /* INTER or not, with a base */
    NvdrProb tx_dct[NMC][3];                         /* DCT or not, by shape and direction */
    NvdrProb tx_tree[3][8];                          /* which of the other four */
} ColourModels;

typedef struct {
    NvdrProb cbf[NMC][3];
    NvdrProb sig[NMC][3][SIG_CTX];
    NvdrProb last[NMC][3][POS_CTX];
    NvdrProb gt1[3][GT1_CTX];
    NvdrProb mag[3][MAG_UNARY];
} TextureModels;

static void models_fill(NvdrProb* p, size_t count) {
    for (size_t i = 0; i < count; i++) p[i] = NVDR_PROB_INIT;
}

/*
 * PRIORS (container v18)
 * ----------------------
 * A slice cannot start from what the slice before it learnt, and from
 * even odds every slice pays again for the slow start: that was most of
 * what slices cost. From version 18 a slice starts from what pictures
 * in general teach instead: priors.h holds, per context, the probability
 * it ended at averaged over a set of photographs (scripts/analysis/
 * priors.c), one table for progressive pictures and one for directional.
 * Each context takes its probability with a count of PRIOR_SEEN, so it
 * moves an eighth of the way per symbol at first, not a quarter: it is a
 * better guess than even odds and still only a guess. A file with a
 * fluid context starts from that instead, as it always did.
 *
 * A picture without slices can start from them too (NvdrConfig.priors):
 * it is then version 18 with a slice table of one byte, 0.
 */
#include "priors.h"
#define PRIOR_SEEN 12
_Static_assert(NVDR_PRIOR_COUNT == (sizeof(ColourModels) + 2 * sizeof(TextureModels)) / sizeof(NvdrProb),
               "priors.h was made for other models: run scripts/analysis/priors.c again");
static void models_prior(ColourModels* cm, TextureModels* tm, TextureModels* tm2, int directional) {
    const uint8_t* p = NVDR_PRIORS[directional ? 1 : 0];
    NvdrProb* dst[3] = { (NvdrProb*)cm, (NvdrProb*)tm, (NvdrProb*)tm2 };
    size_t n[3] = { sizeof(*cm) / sizeof(NvdrProb), sizeof(*tm) / sizeof(NvdrProb), sizeof(*tm2) / sizeof(NvdrProb) };
    for (int k = 0; k < 3; k++)
        for (size_t i = 0; i < n[k]; i++) dst[k][i] = ((NvdrProb)*p++ << 7 | 64) | (NvdrProb)PRIOR_SEEN << 16;
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
size_t nvdr_context_size(void) { return sizeof(NvdrContext); }
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
static void put_bit(Sink* s, NvdrProb* p, int bit) {
    s->bits += bitcost[bit][nvdr_prob(*p)];
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

/* A leaf's mode: "flat or not" by size, then 1..42 as six bits down a
 * tree of contexts. A version 13 container had fifteen modes, four bits
 * down the tree, which OLD_MODE renames. */
static const uint8_t OLD_MODE[15] = { 0, 1, 2, 5, 7, 10, 13, 15, 18, 21, 23, 26, 29, 31, 34 };
static void put_mode(Sink* s, ColourModels* m, int sc, int mode, int inter) {
    if (inter) {
        put_bit(s, &m->mode_inter[sc], mode != MODE_INTER);
        if (mode == MODE_INTER) return;
    }
    put_bit(s, &m->mode_flat[sc], mode != 0);
    if (!mode) return;
    int v = mode - 1, node = 1;
    for (int k = 5; k >= 0; k--) {
        int b = (v >> k) & 1;
        put_bit(s, &m->mode_tree[node], b);
        node = 2 * node + b;
    }
}

static int get_mode(NvdrDecoder* d, ColourModels* m, int sc, int inter, int legacy) {
    if (inter && !nvdr_dec_bit(d, &m->mode_inter[sc])) return MODE_INTER;
    if (!nvdr_dec_bit(d, &m->mode_flat[sc])) return 0;
    int v = 0, node = 1;
    for (int k = 0; k < (legacy ? 4 : 6); k++) {
        int b = nvdr_dec_bit(d, &m->mode_tree[node]);
        v = 2 * v + b;
        node = 2 * node + b;
    }
    if (legacy) return v < 14 ? OLD_MODE[v + 1] : -1;
    return v < NVDR_MODES - 1 ? v + 1 : -1;    /* the last 21 are damage */
}

/* Which way a mode predicts from, for the transform type's contexts:
 * flat, planar, smooth and the filters; from above; from the left. */
static int mode_dir(int mode) { return mode <= 1 || mode >= MODE_SMOOTH ? 0 : mode <= 18 ? 1 : 2; }

/* A leaf's transform type: "DCT or not" by size and direction, then the
 * other six as three bits down a tree by direction. */
static void put_tx(Sink* s, ColourModels* m, int sc, int mode, int tx) {
    int dir = mode_dir(mode);
    put_bit(s, &m->tx_dct[sc][dir], tx != TX_DCT);
    if (tx == TX_DCT) return;
    int v = tx - 1, node = 1;
    for (int k = 2; k >= 0; k--) {
        int b = (v >> k) & 1;
        put_bit(s, &m->tx_tree[dir][node], b);
        node = 2 * node + b;
    }
}

static int get_tx(NvdrDecoder* d, ColourModels* m, int sc, int mode) {
    int dir = mode_dir(mode);
    if (!nvdr_dec_bit(d, &m->tx_dct[sc][dir])) return TX_DCT;
    int v = 0, node = 1;
    for (int k = 0; k < 3; k++) {
        int b = nvdr_dec_bit(d, &m->tx_tree[dir][node]);
        v = 2 * v + b;
        node = 2 * node + b;
    }
    return v < TX_TYPES - 1 ? v + 1 : -1;
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

/*
 * TEXTURE CONTEXTS
 * ----------------
 * A coefficient is likelier to be significant, and large, when the ones
 * next to it in frequency are: the energy of a block runs in ridges and
 * falls off together. So significance is conditioned on the position's
 * class (its diagonal: 1, 2, 3-4, 5-7, 8 on) and on how much is already
 * coded around it, the levels at (u-1, v), (u, v-1), (u-1, v-1), (u-2, v)
 * and (u, v-2), each counted up to 3, the template AV1 and VVC use. All
 * five lie on earlier diagonals, so the diagonal scan has coded them. A
 * neighbour in the other band (the other layer) counts as zero, which
 * keeps the layers decodable on their own. The "more than 1" flag of a
 * magnitude is conditioned on the same sum and on whether a level over 1
 * has been seen in the band yet.
 */
/* The sum is read from a grid of the levels coded so far, each already
 * counted up to 3, laid out as the block is: positions before `start`
 * and not yet reached hold 0, which is what the scan order would give.
 * It is most of what costing a leaf's texture does, so it is a few loads
 * and no lookups. */
static int nb_mag(const uint8_t* g, int w, int p) {
    int u = p & (w - 1), v = p / w, t = 0;
    const uint8_t* r = g + p;
    if (u >= 1) { t += r[-1]; if (u >= 2) t += r[-2]; }
    if (v >= 1) { t += r[-w]; if (u >= 1) t += r[-w - 1]; if (v >= 2) t += r[-2 * w]; }
    return t;
}

static uint8_t mag3(int l) { int a = l < 0 ? -l : l; return (uint8_t)(a < 3 ? a : 3); }

static int sig_ctx(int s, int i, int t) {
    int d = scan_diag[s][i];
    int pc = d <= 1 ? 0 : d == 2 ? 1 : d <= 4 ? 2 : d <= 7 ? 3 : 4;
    int nb = (t + 1) >> 1;
    return pc * 5 + (nb < 4 ? nb : 4);
}

static int gt1_ctx(int t, int g) { return (t < 4 ? t : 4) + (g ? 5 : 0); }

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
/* `sc` is the leaf's shape; its class picks the models. */
static void put_texture(Sink* s, TextureModels* m, int sc, int c, const int* lv,
                        int start, int end) {
    int last = -1, mc = SHAPE_MC[sc];
    for (int i = start; i < end; i++) if (lv[i]) last = i;
    put_bit(s, &m->cbf[mc][c], last >= 0);
    if (last < 0) return;
    int g = 0, w = SHAPE_W[sc];
    uint8_t grid[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    memset(grid, 0, (size_t)(w * SHAPE_H[sc]));
    for (int i = start; i <= last; i++) {
        int a = lv[i] < 0 ? -lv[i] : lv[i], p = scan_pos[sc][i];
        int pc = scan_ctx[sc][i], t = nb_mag(grid, w, p);
        grid[p] = mag3(a);
        if (i < end - 1) put_bit(s, &m->sig[mc][c][sig_ctx(sc, i, t)], a != 0);
        if (!a) continue;
        if (i < end - 1) put_bit(s, &m->last[mc][c][pc], i == last);
        put_mag(s, m, c, gt1_ctx(t, g), a);
        put_direct(s, lv[i] < 0, 1);
        if (a > 1) g++;
    }
}

/* Fills lv[start, end); returns 0 when the range carries nothing. */
static int get_texture(NvdrDecoder* d, TextureModels* m, int sc, int c, int* lv,
                       int start, int end, int* corrupt) {
    int mc = SHAPE_MC[sc];
    memset(lv + start, 0, sizeof(int) * (size_t)(end - start));
    if (!nvdr_dec_bit(d, &m->cbf[mc][c])) return 0;
    int g = 0, w = SHAPE_W[sc];
    uint8_t grid[NVDR_MAX_BLOCK * NVDR_MAX_BLOCK];
    memset(grid, 0, (size_t)(w * SHAPE_H[sc]));
    for (int i = start; i < end; i++) {
        int pc = scan_ctx[sc][i], p = scan_pos[sc][i], t = nb_mag(grid, w, p);
        int sig = i < end - 1 ? nvdr_dec_bit(d, &m->sig[mc][c][sig_ctx(sc, i, t)]) : 1;
        if (!sig) continue;
        int last = i < end - 1 ? nvdr_dec_bit(d, &m->last[mc][c][pc]) : 1;
        int a;
        if (!nvdr_dec_bit(d, &m->gt1[c][gt1_ctx(t, g)])) a = 1;
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
        grid[p] = mag3(a);
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
    int n = SHAPE_W[sc] > SHAPE_H[sc] ? SHAPE_W[sc] : SHAPE_H[sc], count = SHAPE_W[sc] * SHAPE_H[sc];
    if (band <= 0) return count;
    int dmax = n * band / 32;
    if (dmax < 1) dmax = 1;
    for (int i = 1; i < count; i++) if (scan_diag[sc][i] > dmax) return i;
    return count;
}
