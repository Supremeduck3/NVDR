/* Part of nvdr.c, which includes it: slices and the header. */

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

/* Each component's plane, as the canvases lay them out: in 4:2:0 colour
 * sits on a half-size canvas padded to the smallest leaf, 4. */
static void restore_planes(const NvdrHeader* h, NvdrRestorePlane* g) {
    int pw = (h->width + h->min_block - 1) / h->min_block * h->min_block;
    int ph = (h->height + h->min_block - 1) / h->min_block * h->min_block;
    g[0].pw = pw; g[0].ph = ph; g[0].unit = NVDR_RESTORE_UNIT;
    for (int c = 1; c < 3; c++) {
        if (h->flags & NVDR_FLAG_CHROMA420) {
            int cw = (h->width + 1) / 2, ch = (h->height + 1) / 2;
            g[c].pw = (cw + NVDR_MIN_BLOCK - 1) / NVDR_MIN_BLOCK * NVDR_MIN_BLOCK;
            g[c].ph = (ch + NVDR_MIN_BLOCK - 1) / NVDR_MIN_BLOCK * NVDR_MIN_BLOCK;
            g[c].unit = NVDR_RESTORE_UNIT / 2;
        } else g[c] = g[0];
    }
}

/*
 * SLICES (container v17)
 * ----------------------
 * A file sent where packets are lost without being sent again (radio,
 * satellite, a datagram link) loses whatever follows the first hole in a
 * layer: a range decoder cannot pick up again in the middle of its
 * stream, and the models and predictions after the hole depend on what
 * was in it. Restart points bound that, as JPEG's restart markers and
 * H.264's slices do. Every `slice_rows` rows of tiles start a slice; each
 * layer's part of each slice is a stream of its own, coded from the
 * models as they stood at the start of the picture (the step offsets'
 * running value from 0), and a leaf's colour is predicted from nothing
 * above its slice's first row. A slice then decodes on its own: a hole
 * costs the rest of its slice's part of its layer, and for the colour
 * layer, that part of the texture layers over the same tiles.
 *
 * The file stays progressive: the slices' parts of layer 0, top to
 * bottom, then layer 1's, then layer 2's, so a prefix decodes as before.
 * After the restoration parameters comes the slice table: one byte,
 * `slice_rows` (0, no slices), then, layer by layer, the size in bytes of
 * every slice's part but the last, as LEB128; the last is what the
 * header's size for the layer leaves.
 *
 * What it costs is each part starting its models afresh and flushing its
 * coder, and the colours along a slice's top edge predicted from the left
 * alone: see NOTEBOOK.md, "Over a bad link". Progressive pictures only.
 */
static int read_leb(const uint8_t* p, size_t size, size_t* at, uint32_t* v) {
    uint32_t x = 0;
    for (int sh = 0; sh < 32; sh += 7) {
        if (*at >= size) return -1;
        uint8_t b = p[(*at)++];
        x |= (uint32_t)(b & 127) << sh;
        if (!(b & 128)) { *v = x; return 0; }
    }
    return -1;
}

static int tiles_down(const NvdrHeader* h) { return (h->height + h->max_block - 1) / h->max_block; }
static int slice_count(const NvdrHeader* h) {
    int ty = tiles_down(h);
    return h->slice_rows ? (ty + h->slice_rows - 1) / h->slice_rows : 1;
}

/* The slice table at `at` (the byte after the restoration parameters):
 * sets h->slice_rows and h->table_len, and with `seg` every layer's part
 * of every slice, [layer * slices + slice]. */
static int slice_table(const uint8_t* data, size_t size, size_t at, NvdrHeader* h, uint32_t* seg) {
    size_t first = at;
    if (at >= size) return -1;
    h->slice_rows = data[at++];
    int ns = slice_count(h);
    for (int k = 0; k < NVDR_LAYERS; k++) {
        uint32_t left = h->stored_bytes[k];
        for (int i = 0; i < ns - 1; i++) {
            uint32_t v;
            if (read_leb(data, size, &at, &v) != 0 || v > left) return -1;
            left -= v;
            if (seg) seg[k * ns + i] = v;
        }
        if (seg) seg[k * ns + ns - 1] = left;
    }
    h->table_len = (uint32_t)(at - first);
    return 0;
}

static int read_header(const uint8_t* data, size_t size, NvdrHeader* h) {
    /* Version 16 differs only in having no slice table, 15 also in having
     * no deblocking levels, 14 also in having no half nodes, 13 also in
     * having fifteen intra modes. */
    if (size < NVDR_HEADER_SIZE || memcmp(data, NVDR_MAGIC, 4) != 0 ||
        data[4] < 13 || data[4] > NVDR_VERSION_SLICES)
        return -1;
    memset(h, 0, sizeof(*h));
    h->version = data[4];
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
    h->deblock[0] = h->deblock[1] = DB_LEVEL;
    if (!h->width || !h->height || (size_t)h->width * h->height > NVDR_MAX_PIXELS) return -1;
    if (!valid_block(h->max_block) || !valid_block(h->min_block) || h->min_block > h->max_block)
        return -1;
    if (!h->q_luma || !h->q_chroma) return -1;
    if (h->flags & ~(NVDR_FLAG_RESIDUAL | NVDR_FLAG_DEBLOCK | NVDR_FLAG_CHROMA420 | NVDR_FLAG_GRAIN |
                     NVDR_FLAG_TILEQ | NVDR_FLAG_DIRPRED | NVDR_FLAG_INTER | NVDR_FLAG_TXSEL))
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
    /* Directional prediction decodes colour and texture together, so it
     * has no second texture layer and its band is 0: version 16 keeps the
     * deblocking levels in that byte instead (see deblock()), luma's in
     * the low half and colour's in the high. A residual has nothing to
     * predict from. */
    if ((h->flags & NVDR_FLAG_DIRPRED) && h->version >= 16) {
        h->deblock[0] = h->band & 15;
        h->deblock[1] = h->band >> 4;
        h->band = 0;
    }
    if (h->band > 32) return -1;
    if ((h->flags & NVDR_FLAG_DIRPRED) && (h->band != 0 || (h->flags & NVDR_FLAG_RESIDUAL))) return -1;
    if ((h->flags & NVDR_FLAG_INTER) && !(h->flags & NVDR_FLAG_DIRPRED)) return -1;
    if ((h->flags & NVDR_FLAG_TXSEL) && !(h->flags & NVDR_FLAG_DIRPRED)) return -1;
    /* Restoration parameters follow the grain's; bytes 30 and 31 say how
     * many, and none means none. (There was no flag left for them: 0x80
     * is the transform types'.) A residual is not a picture and is never
     * restored. */
    h->restore_len = (uint16_t)get_u16(data + 30);
    if (h->restore_len) {
        NvdrRestore r;
        NvdrRestorePlane g[3];
        restore_planes(h, g);
        if ((h->flags & NVDR_FLAG_RESIDUAL) ||
            size < NVDR_HEADER_SIZE + (size_t)h->grain_len + h->restore_len ||
            nvdr_restore_parse(data + NVDR_HEADER_SIZE + h->grain_len, h->restore_len, g, &r) != 0) return -1;
    }
    if (h->version >= 17) {
        if (slice_table(data, size, NVDR_HEADER_SIZE + (size_t)h->grain_len + h->restore_len, h, NULL) != 0)
            return -1;
        if (h->slice_rows && (h->flags & (NVDR_FLAG_INTER | NVDR_FLAG_RESIDUAL))) return -1;
    }
    return 0;
}

static NvdrProb* split_model(ColourModels* m, const Canvas* cv, int n) {
    return cv->comp0 ? &m->split_c[size_class(n)] : &m->split[size_class(n)];
}

/*
 * HALF NODES (container v15)
 * --------------------------
 * In a directional picture a whole node of 8 to 32 that does not split
 * in four may be cut in two instead: across, into a top and a bottom
 * leaf of n x n/2, or down, into a left and a right leaf of n/2 x n, as
 * AV1's PARTITION_HORZ and PARTITION_VERT. After the split flag comes a
 * "cut in two" flag, then "down", both by size and tree; the halves are
 * leaves, the first decoded before the second. Each is predicted along
 * its own mode over its own shape, with the row above and the column to
 * the left run on to w + h, and transformed as one block (see
 * inverse_tx()); edge cases follow from the z-order of the 4x4 cells,
 * since the halves of a node cover its quadrants in decoding order.
 */
static int can_halve(const Canvas* cv, int n) { return cv->halves && n >= 8 && n / 2 >= cv->min_block; }
