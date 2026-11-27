/* nvdr.c, part: the palette container (version 20). */

/*
 * PALETTE (container v20)
 * -----------------------
 * A drawing, a screenshot, an icon, a chart: a few colours, each repeated
 * over large areas, with hard edges between them. The quadtree and its
 * transforms are made for photographs, and code such a picture as if its
 * edges were texture: approximately, and at a price. A picture of at
 * most PALETTE_MAX distinct colours can instead be sent as the colours
 * themselves and, for each pixel, which of them it is: exactly, and
 * usually smaller.
 *
 * The container is the usual 32-byte header, version 20, no flags, with
 * the whole picture in layer 0 (layers 1 and 2 empty; the block sizes
 * and steps are fixed placeholders, 32, 4, 1 and 1):
 *
 *   [colours - 1, u8][colours x RGB, most frequent first][indices]
 *
 * The indices are arithmetic coded in raster order, each against its
 * left (L), top (T) and top-right (TR) neighbours, as JBIG and WebP's
 * lossless mode look at them: "same as L?", then if T differs "same as
 * T?", then if TR differs from both "same as TR?", and only then the
 * index itself, through a binary tree. Each question has eight contexts,
 * from whether L equals T, T equals TR and L equals the top-left: a flat
 * region asks one question per pixel and pays a small fraction of a bit
 * for it, an edge pays where it is. Outside the picture a neighbour is
 * the nearest one inside (the top row's T is its L, the left column's L
 * is its T, the first pixel's both are colour 0).
 *
 * The picture is decoded row by row; bytes that stop short leave the
 * rows decoded before them, and the rest in the most frequent colour,
 * likeliest the background (it measured closer than repeating the last
 * row down). The
 * encoder takes the palette when it costs less than the quadtree's
 * picture, weighing that one's error as encode_chroma() does: see
 * nvdr_encode_mem_ctx().
 */
#define NVDR_VERSION_PALETTE 20
#define PALETTE_MAX 256

typedef struct {
    NvdrProb eq_l[8], eq_t[8], eq_tr[8];
    NvdrProb tree[2 * PALETTE_MAX];
} PaletteModels;

static void palette_models_init(PaletteModels* m) {
    NvdrProb* p = (NvdrProb*)m;
    for (size_t i = 0; i < sizeof(*m) / sizeof(NvdrProb); i++) p[i] = NVDR_PROB_INIT;
}

static int palette_bits(int n) { int b = 0; while ((1 << b) < n) b++; return b; }

/* The neighbours of pixel (x, y) in `ix` (w wide) and their context. */
static inline int palette_ctx(const uint8_t* ix, int w, int x, int y, int* l, int* t, int* tr) {
    const uint8_t* row = ix + (size_t)y * w;
    const uint8_t* up = y ? row - w : NULL;
    int L, T, TL, TR;
    if (y == 0) { L = x ? row[x - 1] : 0; T = L; TL = L; TR = L; }
    else {
        T = up[x];
        L = x ? row[x - 1] : T;
        TL = x ? up[x - 1] : T;
        TR = x + 1 < w ? up[x + 1] : T;
    }
    *l = L; *t = T; *tr = TR;
    return (L == T) | (T == TR) << 1 | (L == TL) << 2;
}

/* The picture's colours, most frequent first, and each pixel's index;
 * 0 when there are more than PALETTE_MAX. */
static int palette_build(const NvdrImage* img, uint8_t pal[PALETTE_MAX][3], uint8_t* ix) {
    enum { SLOTS = 4096 };
    static const uint32_t EMPTY = 0xffffffffu;
    uint32_t key[SLOTS], count[SLOTS];
    int16_t slot_of[SLOTS];
    for (int i = 0; i < SLOTS; i++) { key[i] = EMPTY; count[i] = 0; slot_of[i] = -1; }
    int n = 0;
    uint32_t colour[PALETTE_MAX];
    size_t np = (size_t)img->width * img->height;
    for (size_t i = 0; i < np; i++) {
        const uint8_t* p = img->pixels + i * 3;
        uint32_t c = (uint32_t)p[0] << 16 | (uint32_t)p[1] << 8 | p[2];
        uint32_t s = (c * 2654435761u) >> 20;
        while (key[s] != EMPTY && key[s] != c) s = (s + 1) & (SLOTS - 1);
        if (key[s] == EMPTY) {
            if (n == PALETTE_MAX) return 0;
            key[s] = c; slot_of[s] = (int16_t)n; colour[n++] = c;
        }
        count[s]++;
    }
    /* Most frequent first: the tree learns the common indices fastest. */
    uint32_t freq[PALETTE_MAX];
    int order[PALETTE_MAX], rank[PALETTE_MAX];
    for (int s = 0; s < SLOTS; s++) if (slot_of[s] >= 0) freq[slot_of[s]] = count[s];
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 1; i < n; i++) {
        int v = order[i], j = i;
        while (j > 0 && (freq[order[j - 1]] < freq[v] ||
                         (freq[order[j - 1]] == freq[v] && colour[order[j - 1]] > colour[v]))) {
            order[j] = order[j - 1]; j--;
        }
        order[j] = v;
    }
    for (int i = 0; i < n; i++) {
        rank[order[i]] = i;
        uint32_t c = colour[order[i]];
        pal[i][0] = (uint8_t)(c >> 16); pal[i][1] = (uint8_t)(c >> 8); pal[i][2] = (uint8_t)c;
    }
    for (size_t i = 0; i < np; i++) {
        const uint8_t* p = img->pixels + i * 3;
        uint32_t c = (uint32_t)p[0] << 16 | (uint32_t)p[1] << 8 | p[2];
        uint32_t s = (c * 2654435761u) >> 20;
        while (key[s] != c) s = (s + 1) & (SLOTS - 1);
        ix[i] = (uint8_t)rank[slot_of[s]];
    }
    return n;
}

/* The palette container of `img`; -1 when it has too many colours (or
 * on no memory). */
static int palette_encode(const NvdrImage* img, uint8_t** out_buf, size_t* out_len, NvdrHeader* hdr_out) {
    *out_buf = NULL; *out_len = 0;
    int w = img->width, h = img->height;
    if (w <= 0 || h <= 0 || w > 65535 || h > 65535 || (size_t)w * h > NVDR_MAX_PIXELS) return -1;
    uint8_t pal[PALETTE_MAX][3];
    uint8_t* ix = (uint8_t*)malloc((size_t)w * h);
    if (!ix) return -1;
    int n = palette_build(img, pal, ix);
    if (!n) { free(ix); return -1; }
    PaletteModels m;
    palette_models_init(&m);
    NvdrEncoder enc;
    if (nvdr_enc_init(&enc, (size_t)w * h / 8 + 64) != 0) { free(ix); return -1; }
    int nb = palette_bits(n);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int L, T, TR, c = palette_ctx(ix, w, x, y, &L, &T, &TR), v = ix[(size_t)y * w + x];
            if (n == 1) continue;
            nvdr_enc_bit(&enc, &m.eq_l[c], v != L);
            if (v == L) continue;
            if (T != L) {
                nvdr_enc_bit(&enc, &m.eq_t[c], v != T);
                if (v == T) continue;
            }
            if (TR != L && TR != T) {
                nvdr_enc_bit(&enc, &m.eq_tr[c], v != TR);
                if (v == TR) continue;
            }
            nvdr_enc_tree(&enc, m.tree, (uint32_t)v, nb);
        }
    free(ix);
    if (nvdr_enc_finish(&enc) != 0) { nvdr_enc_free(&enc); return -1; }
    size_t body = 1 + (size_t)n * 3 + (n > 1 ? enc.count : 0);
    uint8_t* buf = (uint8_t*)malloc(NVDR_HEADER_SIZE + body);
    if (!buf) { nvdr_enc_free(&enc); return -1; }
    memset(buf, 0, NVDR_HEADER_SIZE);
    memcpy(buf, NVDR_MAGIC, 4);
    buf[4] = NVDR_VERSION_PALETTE;
    put_u16(buf + 6, (uint32_t)w); put_u16(buf + 8, (uint32_t)h);
    buf[10] = NVDR_MAX_BLOCK; buf[11] = NVDR_MIN_BLOCK;
    put_u16(buf + 12, 1); put_u16(buf + 14, 1);
    put_u32(buf + 16, (uint32_t)body);
    uint8_t* p = buf + NVDR_HEADER_SIZE;
    *p++ = (uint8_t)(n - 1);
    memcpy(p, pal, (size_t)n * 3); p += (size_t)n * 3;
    if (n > 1) memcpy(p, enc.bytes, enc.count);
    nvdr_enc_free(&enc);
    if (hdr_out) {
        memset(hdr_out, 0, sizeof(*hdr_out));
        hdr_out->version = NVDR_VERSION_PALETTE;
        hdr_out->width = (uint16_t)w; hdr_out->height = (uint16_t)h;
        hdr_out->max_block = NVDR_MAX_BLOCK; hdr_out->min_block = NVDR_MIN_BLOCK;
        hdr_out->q_luma = hdr_out->q_chroma = 1;
        hdr_out->stored_bytes[0] = (uint32_t)body;
    }
    *out_buf = buf; *out_len = NVDR_HEADER_SIZE + body;
    return 0;
}

/* The header of a palette container, which read_header() hands over. */
static int palette_header(const uint8_t* data, NvdrHeader* h) {
    memset(h, 0, sizeof(*h));
    h->version = NVDR_VERSION_PALETTE;
    h->width = (uint16_t)get_u16(data + 6);
    h->height = (uint16_t)get_u16(data + 8);
    h->max_block = data[10]; h->min_block = data[11];
    h->q_luma = (uint16_t)get_u16(data + 12); h->q_chroma = (uint16_t)get_u16(data + 14);
    h->flags = data[5];
    for (int k = 0; k < NVDR_LAYERS; k++) h->stored_bytes[k] = get_u32(data + 16 + 4 * k);
    if (!h->width || !h->height || (size_t)h->width * h->height > NVDR_MAX_PIXELS) return -1;
    /* Everything but the size and layer 0 is fixed. */
    if (h->flags || h->max_block != NVDR_MAX_BLOCK || h->min_block != NVDR_MIN_BLOCK ||
        h->q_luma != 1 || h->q_chroma != 1 || h->stored_bytes[1] || h->stored_bytes[2] ||
        data[28] || data[29] || get_u16(data + 30) || h->stored_bytes[0] < 4 ||
        h->stored_bytes[0] > 0x7fffffffu) return -1;
    return 0;
}

/* Decodes the `avail` bytes of layer 0 that can be read (of the header's
 * h->stored_bytes[0]) into `out`; -1 when not even the palette arrived. */
static int palette_decode(const uint8_t* body, size_t avail, const NvdrHeader* h, NvdrImage* out,
                          NvdrDecodeInfo* info) {
    out->pixels = NULL; out->width = out->height = 0;
    if (avail < 1) return -1;
    int n = body[0] + 1;
    size_t head = 1 + (size_t)n * 3;
    if (avail < head || h->stored_bytes[0] < head) return -1;
    int w = h->width, hh = h->height;
    uint8_t* ix = (uint8_t*)calloc((size_t)w * hh, 1);
    if (!ix) return -1;
    const uint8_t* pal = body + 1;
    int rows = hh;
    if (n > 1) {
        size_t len = h->stored_bytes[0] - head, have = avail - head < len ? avail - head : len;
        PaletteModels m;
        palette_models_init(&m);
        NvdrDecoder dec;
        nvdr_dec_init(&dec, body + head, have);
        int nb = palette_bits(n);
        rows = 0;
        for (int y = 0; y < hh; y++) {
            for (int x = 0; x < w; x++) {
                int L, T, TR, c = palette_ctx(ix, w, x, y, &L, &T, &TR), v;
                if (!nvdr_dec_bit(&dec, &m.eq_l[c])) v = L;
                else if (T != L && !nvdr_dec_bit(&dec, &m.eq_t[c])) v = T;
                else if (TR != L && TR != T && !nvdr_dec_bit(&dec, &m.eq_tr[c])) v = TR;
                else v = (int)nvdr_dec_tree(&dec, m.tree, nb);
                /* A damaged stream can name a colour there is not. */
                ix[(size_t)y * w + x] = (uint8_t)(v < n ? v : 0);
            }
            /* A row read past the bytes that arrived is not trusted. */
            if (dec.overrun) break;
            rows = y + 1;
        }
        memset(ix + (size_t)rows * w, 0, (size_t)(hh - rows) * w);
    }
    out->pixels = (unsigned char*)malloc((size_t)w * hh * 3);
    if (!out->pixels) { free(ix); return -1; }
    out->width = w; out->height = hh;
    for (size_t i = 0; i < (size_t)w * hh; i++) memcpy(out->pixels + i * 3, pal + (size_t)ix[i] * 3, 3);
    free(ix);
    if (info) {
        int whole = rows == hh;
        info->tiles = 1;
        info->layers_present = whole ? NVDR_LAYERS : 1;
        for (int k = 0; k < NVDR_LAYERS; k++) info->tiles_complete[k] = whole;
    }
    return 0;
}
