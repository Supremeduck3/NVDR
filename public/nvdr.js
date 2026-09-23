/*
 * NVDR v10 decoder — the browser half of src/nvdr.c.
 *
 * Mirrors the C decoder exactly: the same integer inverse transform, the
 * same fixed-point colour conversion, the same prediction, so the two
 * produce the same pixels. scripts/crosscheck.mjs holds them to that.
 *
 * Format, little-endian:
 *
 *   [header 32B]  "NVDR", version 11, flags, width, height, max/min block,
 *                 luma and chroma steps, the three layers' byte counts, band
 *   [layer 0]     the quadtree and every leaf's colour, arithmetic coded,
 *                 32x32 tile by tile
 *   [layer 1]     every leaf's low-frequency texture: DCT coefficients
 *   [layer 2]     every leaf's high-frequency texture
 *
 * A short buffer is a normal outcome, not an error: every tile that
 * arrived whole is decoded, a colour tile that did not is painted grey, a
 * texture tile that did not shows what the layer before gave it.
 */

const MAGIC = 0x5244564e;   // "NVDR" read as a little-endian uint32
const VERSION = 11;
const HEADER_SIZE = 32;
const MAX_PIXELS = 1 << 27; // NVDR_MAX_PIXELS
export const LAYERS = 3;
export const LAYER_NAMES = ['COR', 'COR+TEXTURA BAIXA', 'COR+TEXTURA COMPLETA'];

const NSIZES = 4, MIN_BLOCK = 4, MAX_BLOCK = 32;
const FLAG_RESIDUAL = 0x01;   // every colour predicted as 128
const FLAG_DEBLOCK = 0x02;    // leaf seams filtered after decoding
const DB_ALPHA = 20, DB_BETA = 6, DB_TC = 3;
const POS_CTX = 15, MAG_UNARY = 14, EG_LIMIT = 24, COEF_MAX = 32767;

/* --- entropy layer, mirroring src/entropy.c -------------------------- */

const PROB_BITS = 11;
export const PROB_INIT = 1 << (PROB_BITS - 1);
const MOVE_BITS = 5;
const TOP_VALUE = 1 << 24;

/*
 * The LZMA range decoder. Arithmetic is forced through >>> because the
 * coder works on unsigned 32-bit words and JavaScript's bitwise operators
 * are signed.
 */
export class ArithDecoder {
    constructor(bytes, offset, size) {
        this.bytes = bytes;
        this.pos = offset;
        this.end = offset + size;
        this.range = 0xFFFFFFFF;
        this.code = 0;
        this.overrun = false;
        // The encoder's first byte is always zero, so five are read and the
        // first is discarded.
        for (let i = 0; i < 5; i++) this.code = ((this.code << 8) | this.byte()) >>> 0;
    }

    byte() {
        if (this.pos >= this.end) { this.overrun = true; return 0; }
        return this.bytes[this.pos++];
    }

    normalize() {
        while (this.range >>> 0 < TOP_VALUE) {
            this.range = (this.range << 8) >>> 0;
            this.code = ((this.code << 8) | this.byte()) >>> 0;
        }
    }

    bit(probs, index) {
        const bound = ((this.range >>> PROB_BITS) * probs[index]) >>> 0;
        let result;
        if ((this.code >>> 0) < bound) {
            this.range = bound;
            probs[index] += ((1 << PROB_BITS) - probs[index]) >>> MOVE_BITS;
            result = 0;
        } else {
            this.code = (this.code - bound) >>> 0;
            this.range = (this.range - bound) >>> 0;
            probs[index] -= probs[index] >>> MOVE_BITS;
            result = 1;
        }
        this.normalize();
        return result;
    }

    direct(bitCount) {
        let result = 0;
        while (bitCount-- > 0) {
            this.range = this.range >>> 1;
            this.code = (this.code - this.range) >>> 0;
            const mask = (0 - (this.code >>> 31)) >>> 0;
            this.code = (this.code + (this.range & mask)) >>> 0;
            this.normalize();
            result = ((result << 1) + mask + 1) >>> 0;
        }
        return result;
    }

    tree(probs, bitCount) {
        let node = 1;
        for (let i = 0; i < bitCount; i++) node = (node << 1) | this.bit(probs, node);
        return node - (1 << bitCount);
    }
}

/* --- tables, built exactly as tables_init() builds them --------------- */

const COS_TABLE = [
    90, 90, 90, 90, 89, 88, 87, 85, 83, 82, 80, 78, 75, 73, 70, 67,
    64, 61, 57, 54, 50, 46, 43, 38, 36, 31, 25, 22, 18, 13, 9, 4, 0
];

function cosEntry(j) {
    j &= 127;
    if (j <= 32) return COS_TABLE[j];
    if (j <= 64) return -COS_TABLE[64 - j];
    if (j <= 96) return -COS_TABLE[j - 64];
    return COS_TABLE[128 - j];
}

const TMAT = [], SCAN_POS = [], SCAN_CTX = [], SCAN_DIAG = [];

/* Mirrors band_split(): the first scan position of the high band. */
function bandSplit(sc, band) {
    const n = MIN_BLOCK << sc, count = n * n;
    if (band <= 0) return count;
    const dmax = Math.max(1, Math.floor(n * band / 32));
    for (let i = 1; i < count; i++) if (SCAN_DIAG[sc][i] > dmax) return i;
    return count;
}
for (let s = 0; s < NSIZES; s++) {
    const n = MIN_BLOCK << s, unit = MAX_BLOCK / n;
    const t = new Int32Array(n * n);
    for (let k = 0; k < n; k++)
        for (let x = 0; x < n; x++)
            t[k * n + x] = k ? cosEntry((2 * x + 1) * k * unit) : 64;
    const pos = new Int32Array(n * n), ctx = new Uint8Array(n * n), diag = new Uint8Array(n * n);
    let at = 0;
    for (let d = 0; d <= 2 * (n - 1); d++)
        for (let v = 0; v < n; v++) {
            const u = d - v;
            if (u < 0 || u >= n) continue;
            pos[at] = v * n + u;
            ctx[at] = d < 8 ? d : 8 + Math.min((d - 8) >> 2, 6);
            diag[at] = d;
            at++;
        }
    TMAT.push(t); SCAN_POS.push(pos); SCAN_CTX.push(ctx); SCAN_DIAG.push(diag);
}

const sizeClass = n => (n === 4 ? 0 : n === 8 ? 1 : n === 16 ? 2 : 3);
const log2 = n => 31 - Math.clz32(n);
const clampU8 = v => (v < 0 ? 0 : v > 255 ? 255 : v);
const clampCoef = v => (v < -COEF_MAX ? -COEF_MAX : v > COEF_MAX ? COEF_MAX : v);

function divRound(a, n) {
    const k = log2(n);
    return a >= 0 ? (a + (n >> 1)) >> k : -((-a + (n >> 1)) >> k);
}

/*
 * Mirrors inverse_dct: columns >> 6 with a 16-bit clip, rows >> 6+log2 n.
 * `mu` and `mv` bound the nonzero coefficients (u <= mu, v <= mv). Every
 * term past them is a zero, so skipping them changes nothing but the
 * time: most leaves carry only a few low frequencies.
 */
const TMP = new Int32Array(MAX_BLOCK * MAX_BLOCK);
const ROW = new Int32Array(MAX_BLOCK);
function inverseDct(s, input, out, mu, mv) {
    // Integer sums, so the order, chosen for contiguous inner loops and to
    // skip zero terms, changes nothing.
    const n = MIN_BLOCK << s, t = TMAT[s], shift2 = 6 + log2(n), half = 1 << (shift2 - 1);
    for (let y = 0; y < n; y++) {
        ROW.fill(0, 0, mu + 1);
        for (let v = 0; v <= mv; v++) {
            const k = t[v * n + y], r = v * n;
            for (let u = 0; u <= mu; u++) ROW[u] += k * input[r + u];
        }
        for (let u = 0; u <= mu; u++) TMP[y * n + u] = clampCoef((ROW[u] + 32) >> 6);
    }
    for (let y = 0; y < n; y++) {
        const row = y * n;
        ROW.fill(0, 0, n);
        for (let u = 0; u <= mu; u++) {
            const k = TMP[row + u];
            if (k === 0) continue;
            const r = u * n;
            for (let x = 0; x < n; x++) ROW[x] += k * t[r + x];
        }
        for (let x = 0; x < n; x++) out[row + x] = (ROW[x] + half) >> shift2;
    }
}

/* --- models ------------------------------------------------------------ */

const probs = n => new Uint16Array(n).fill(PROB_INIT);
const grid = (outer, inner) => Array.from({ length: outer }, () => probs(inner));

function colourModels() {
    return {
        split: probs(NSIZES),
        dcZero: grid(NSIZES, 3),
        dcSign: probs(3),
        dcMag: grid(3, MAG_UNARY)
    };
}

function textureModels() {
    return {
        cbf: grid(NSIZES, 3),
        sig: Array.from({ length: NSIZES }, () => grid(3, POS_CTX)),
        last: Array.from({ length: NSIZES }, () => grid(3, POS_CTX)),
        gt1: grid(3, 4),
        mag: grid(3, MAG_UNARY)
    };
}

/* `state.corrupt` is the C side's *corrupt. */
function getEscape(d, state) {
    let n = 0;
    while (!d.direct(1)) {
        if (++n > EG_LIMIT || d.overrun) { state.corrupt = true; return 0; }
    }
    const v = n ? (((1 << n) >>> 0) | d.direct(n)) >>> 0 : 1;
    return v - 1;
}

function getDc(d, m, sc, c, state) {
    if (!d.bit(m.dcZero[sc], c)) return 0;
    const neg = d.bit(m.dcSign, c);
    let r = 0, i = 0;
    for (; i < MAG_UNARY; i++) {
        if (!d.bit(m.dcMag[c], i)) break;
        r = i + 1;
    }
    if (i === MAG_UNARY) r = MAG_UNARY + getEscape(d, state);
    if (r > COEF_MAX) { state.corrupt = true; r = COEF_MAX; }
    return neg ? -(r + 1) : r + 1;
}

/* Mirrors get_texture(): fills lv[start, end). */
function getTexture(d, m, sc, c, lv, start, end, state) {
    lv.fill(0, start, end);
    if (!d.bit(m.cbf[sc], c)) return false;
    let g = 0;
    for (let i = start; i < end; i++) {
        const pc = SCAN_CTX[sc][i];
        const sig = i < end - 1 ? d.bit(m.sig[sc][c], pc) : 1;
        if (!sig) continue;
        const last = i < end - 1 ? d.bit(m.last[sc][c], pc) : 1;
        let a;
        if (!d.bit(m.gt1[c], g < 3 ? g : 3)) a = 1;
        else {
            let r = 0, k = 0;
            for (; k < MAG_UNARY; k++) {
                if (!d.bit(m.mag[c], k)) break;
                r = k + 1;
            }
            if (k === MAG_UNARY) r = MAG_UNARY + getEscape(d, state);
            if (r > COEF_MAX) { state.corrupt = true; r = COEF_MAX; }
            a = r + 2;
            g++;
        }
        lv[i] = d.direct(1) ? -a : a;
        if (last || state.corrupt || d.overrun) break;
    }
    return true;
}

/* --- header ------------------------------------------------------------ */

const validBlock = n => n === 4 || n === 8 || n === 16 || n === 32;

/** The header, or null where read_header() refuses it. */
export function readHeader(buffer) {
    if (buffer.byteLength < HEADER_SIZE) return null;
    // An ArrayBuffer, or a view into one such as a frame inside a sequence.
    const v = ArrayBuffer.isView(buffer)
        ? new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength)
        : new DataView(buffer);
    if (v.getUint32(0, true) !== MAGIC || v.getUint8(4) !== VERSION) return null;
    const h = {
        width: v.getUint16(6, true),
        height: v.getUint16(8, true),
        maxBlock: v.getUint8(10),
        minBlock: v.getUint8(11),
        qLuma: v.getUint16(12, true),
        qChroma: v.getUint16(14, true),
        flags: v.getUint8(5),
        storedBytes: [v.getUint32(16, true), v.getUint32(20, true), v.getUint32(24, true)],
        band: v.getUint8(28)
    };
    if (!h.width || !h.height || h.width * h.height > MAX_PIXELS) return null;
    if (!validBlock(h.maxBlock) || !validBlock(h.minBlock) || h.minBlock > h.maxBlock) return null;
    if (!h.qLuma || !h.qChroma) return null;
    if (h.flags & ~(FLAG_RESIDUAL | FLAG_DEBLOCK)) return null;
    if (h.storedBytes.some(b => b > 0x7fffffff) || h.band > 32) return null;
    return h;
}

/** Bytes needed before each layer is complete. */
export function layerThresholds(header) {
    const a = HEADER_SIZE + header.storedBytes[0], b = a + header.storedBytes[1];
    return [a, b, b + header.storedBytes[2]];
}

/* --- decoder ----------------------------------------------------------- */

/**
 * Decode as far as the bytes reach. Returns null where nvdr_decode_mem
 * returns -1, otherwise
 *   { header, layersPresent, tiles, tilesComplete: [l0, l1],
 *     rgb,        // the image at maxLayer, RGB, width * height * 3
 *     flatRgb,    // layer 0 alone, only when wantFlat
 *     lowRgb }    // layers 0 and 1, what maxLayer = 1 shows, only when
 *                 // wantLow (with maxLayer 2): all three views in one pass
 */
/**
 * A fluid context: the adaptive models one container leaves behind,
 * carried into the next. Mirrors NvdrContext; an empty one behaves like
 * none. Pass the same object to decode() for containers read in order.
 */
export function newContext() { return { valid: false, cm: null, tm: null, tm2: null }; }

const RETRY = Symbol('retry');

export function decode(buffer, maxLayer = LAYERS - 1, wantFlat = false, ctx = null, wantLow = false) {
    // A texture layer that arrived whole cannot stop inside a tile unless
    // the file is damaged, so the first attempt does not save each tile to
    // restore it, a fifth of the time on a large image. If it does stop,
    // the decode starts over the careful way and gives what that gives. A
    // warm context is changed in place and cannot be started over.
    if (!(ctx && ctx.valid)) {
        const r = decodeOnce(buffer, maxLayer, wantFlat, ctx, wantLow, false);
        if (r !== RETRY) return r;
    }
    return decodeOnce(buffer, maxLayer, wantFlat, ctx, wantLow, true);
}

function decodeOnce(buffer, maxLayer, wantFlat, ctx, wantLow, careful) {
    const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    const h = readHeader(bytes);
    if (!h) return null;
    const size = bytes.length;

    let avail0 = Math.min(size - HEADER_SIZE, h.storedBytes[0]);
    if (avail0 < 5) return null;
    const off1 = HEADER_SIZE + h.storedBytes[0];
    const avail1 = Math.min(size > off1 ? size - off1 : 0, h.storedBytes[1]);
    const off2 = off1 + h.storedBytes[1];
    const avail2 = Math.min(size > off2 ? size - off2 : 0, h.storedBytes[2]);
    const bandAt = [0, 1, 2, 3].map(sc => bandSplit(sc, h.band));

    const tile = h.maxBlock, minBlock = h.minBlock;
    const pw = Math.ceil(h.width / minBlock) * minBlock;
    const ph = Math.ceil(h.height / minBlock) * minBlock;
    const flat = [0, 1, 2].map(() => new Uint8Array(pw * ph));
    const full = [0, 1, 2].map(() => new Uint8Array(pw * ph));
    // Texture added so far, unclamped to 16 bits; full = clamp(flat + acc).
    const acc = [0, 1, 2].map(() => new Int16Array(pw * ph));
    const step = [h.qLuma, h.qChroma, h.qChroma];
    const tilesX = Math.ceil(pw / tile), tilesY = Math.ceil(ph / tile), tiles = tilesX * tilesY;

    const whole = (x, y, n) => x + n <= pw && y + n <= ph;
    const exists = (x, y) => x < pw && y < ph;

    // TypedArray.fill costs a call per row, most of a 4-pixel row's time:
    // short rows are written directly.
    function fill(p, x, y, w, hh, v) {
        if (w > 16) {
            for (let j = y; j < y + hh; j++) p.fill(v, j * pw + x, j * pw + x + w);
            return;
        }
        for (let j = y; j < y + hh; j++)
            for (let at = j * pw + x, end = at + w; at < end; at++) p[at] = v;
    }

    // Mirrors paint_flat().
    function paintFlat(c, x, y, w, hh, v) {
        if (w > 16) {
            fill(flat[c], x, y, w, hh, v);
            fill(full[c], x, y, w, hh, v);
            fill(acc[c], x, y, w, hh, 0);
            return;
        }
        const f = flat[c], o = full[c], a = acc[c];
        for (let j = y; j < y + hh; j++)
            for (let at = j * pw + x, end = at + w; at < end; at++) { f[at] = v; o[at] = v; a[at] = 0; }
    }

    const fixedPred = (h.flags & FLAG_RESIDUAL) !== 0;
    function predict(c, x, y, n) {
        if (fixedPred) return 128;
        const p = flat[c];
        let sum = 0, k = 0;
        if (y > 0) {
            const x1 = Math.min(x + n, pw);
            for (let i = x; i < x1; i++) sum += p[(y - 1) * pw + i];
            k += x1 - x;
        }
        if (x > 0) {
            const y1 = Math.min(y + n, ph);
            for (let j = y; j < y1; j++) sum += p[j * pw + x - 1];
            k += y1 - y;
        }
        return k ? Math.floor((sum + (k >> 1)) / k) : 128;
    }

    function tileFallback(tx, ty) {
        const w = tx + tile < pw ? tile : pw - tx;
        const hh = ty + tile < ph ? tile : ph - ty;
        // Neutral grey, as tile_fallback() explains.
        for (let c = 0; c < 3; c++) paintFlat(c, tx, ty, w, hh, 128);
    }

    // Layer 0.
    const warm = ctx && ctx.valid;
    const cm = warm ? ctx.cm : colourModels();
    const tms = warm ? [ctx.tm, ctx.tm2] : [textureModels(), textureModels()];
    const d0 = new ArithDecoder(bytes, HEADER_SIZE, avail0);
    const s0 = { corrupt: false };
    const lx = [], ly = [], ln = [];
    const tileStart = new Int32Array(tiles + 1);

    function readNode(x, y, n) {
        let split = 0;
        if (n > minBlock) split = whole(x, y, n) ? d0.bit(cm.split, sizeClass(n)) : 1;
        if (d0.overrun || s0.corrupt) return;
        if (split) {
            const hh = n >> 1;
            for (let k = 0; k < 4; k++) {
                const cx = x + (k & 1) * hh, cy = y + (k >> 1) * hh;
                if (exists(cx, cy)) readNode(cx, cy, hh);
                if (d0.overrun || s0.corrupt) return;
            }
            return;
        }
        const sc = sizeClass(n);
        for (let c = 0; c < 3; c++) {
            const pred = predict(c, x, y, n);
            const dl = getDc(d0, cm, sc, c, s0);
            const colour = clampU8(pred + divRound(clampCoef(dl * step[c]), n));
            paintFlat(c, x, y, n, n, colour);
        }
        lx.push(x); ly.push(y); ln.push(n);
    }

    let complete0 = 0, stopped = false;
    for (let t = 0; t < tiles; t++) {
        const tx = (t % tilesX) * tile, ty = Math.floor(t / tilesX) * tile;
        tileStart[t] = lx.length;
        if (!stopped) {
            readNode(tx, ty, tile);
            if (d0.overrun || s0.corrupt) {
                stopped = true;
                lx.length = ly.length = ln.length = tileStart[t];
            } else complete0++;
        }
        if (stopped) tileFallback(tx, ty);
    }
    tileStart[tiles] = lx.length;

    // Layers 1 and 2, the low and the high texture band. Mirrors the C
    // loop: each band as far as its bytes reach and no further than the
    // layer before it; a tile cut short is restored to what it showed.
    const complete = [complete0, 0, 0];
    const texturedLeaf = new Uint8Array(lx.length);
    const lv = new Int32Array(MAX_BLOCK * MAX_BLOCK);
    const coef = new Int32Array(MAX_BLOCK * MAX_BLOCK);
    const res = new Int32Array(MAX_BLOCK * MAX_BLOCK);
    let low = null;
    const saved = [0, 1, 2].map(() => new Uint8Array(tile * tile));
    const savedAcc = [0, 1, 2].map(() => new Int16Array(tile * tile));
    for (let layer = 1; layer < LAYERS; layer++) {
        const avail = layer === 1 ? avail1 : avail2, off = layer === 1 ? off1 : off2;
        if ((maxLayer >= 0 && maxLayer < layer) || avail < 5 || complete[layer - 1] === 0) break;
        if (layer === 2 && h.band === 0) break;
        const tm = tms[layer - 1];
        const d = new ArithDecoder(bytes, off, avail);
        const guard = careful || avail < h.storedBytes[layer];
        const st = { corrupt: false };
        for (let t = 0; t < complete[layer - 1]; t++) {
            const tx = (t % tilesX) * tile, ty = Math.floor(t / tilesX) * tile;
            const tw = tx + tile < pw ? tile : pw - tx, th = ty + tile < ph ? tile : ph - ty;
            // Copied by hand: a subarray per row would be millions of
            // short-lived objects on a large image.
            if (guard) for (let c = 0; c < 3; c++) {
                const o = full[c], a = acc[c], so = saved[c], sa = savedAcc[c];
                for (let j = 0; j < th; j++)
                    for (let at = (ty + j) * pw + tx, k = j * tile, end = at + tw; at < end; at++, k++) {
                        so[k] = o[at]; sa[k] = a[at];
                    }
            }
            const first = tileStart[t], last = tileStart[t + 1];
            const was = guard ? texturedLeaf.slice(first, last) : null;
            for (let i = first; i < last && !d.overrun && !st.corrupt; i++) {
                const x = lx[i], y = ly[i], n = ln[i], sc = sizeClass(n), count = n * n;
                const start = layer === 1 ? 1 : bandAt[sc], end = layer === 1 ? bandAt[sc] : count;
                if (start >= end) continue;
                for (let c = 0; c < 3 && !d.overrun && !st.corrupt; c++) {
                    if (getTexture(d, tm, sc, c, lv, start, end, st) && !d.overrun && !st.corrupt) {
                        coef.fill(0, 0, count);
                        const pos = SCAN_POS[sc], sh = log2(n);
                        let mu = 0, mv = 0;
                        for (let k = start; k < end; k++) {
                            if (!lv[k]) continue;
                            const p = pos[k], u = p & (n - 1), v = p >> sh;
                            coef[p] = clampCoef(lv[k] * step[c]);
                            if (u > mu) mu = u;
                            if (v > mv) mv = v;
                        }
                        inverseDct(sc, coef, res, mu, mv);
                        texturedLeaf[i] = 1;
                        const o = full[c], f = flat[c], a = acc[c];
                        for (let j = 0; j < n; j++)
                            for (let ii = 0; ii < n; ii++) {
                                const at = (y + j) * pw + x + ii;
                                let v = a[at] + res[j * n + ii];
                                v = v < -32768 ? -32768 : v > 32767 ? 32767 : v;
                                a[at] = v;
                                o[at] = clampU8(f[at] + v);
                            }
                    }
                }
            }
            if (d.overrun || st.corrupt) {
                if (!guard) return RETRY;
                for (let c = 0; c < 3; c++)
                    for (let j = 0; j < th; j++) {
                        full[c].set(saved[c].subarray(j * tile, j * tile + tw), (ty + j) * pw + tx);
                        acc[c].set(savedAcc[c].subarray(j * tile, j * tile + tw), (ty + j) * pw + tx);
                    }
                texturedLeaf.set(was, first);
                break;
            }
            complete[layer]++;
        }
        // What decode(buffer, 1) would show, without a second pass.
        if (layer === 1 && wantLow) low = { planes: full.map(p => p.slice()), tex: texturedLeaf.slice() };
    }

    /* Mirrors deblock() and deblock_plane(); `tex` is null for colours only. */
    function deblock(planes, tex) {
        const gw = pw >> 2, gh = ph >> 2;
        const vedge = new Uint8Array(gw * gh), hedge = new Uint8Array(gw * gh);
        const textured = new Uint8Array(gw * gh);   // the cell's leaf shows texture
        const mark = (x, y, n, t) => {
            for (let j = y; j < y + n && j < gh; j++)
                for (let k = x; k < x + n && k < gw; k++) textured[j * gw + k] = t;
            if (x > 0) for (let j = y; j < y + n && j < gh; j++) vedge[j * gw + x] = 1;
            if (y > 0) for (let k = x; k < x + n && k < gw; k++) hedge[y * gw + k] = 1;
        };
        const kept = tileStart[complete0];
        for (let i = 0; i < kept; i++) mark(lx[i] >> 2, ly[i] >> 2, ln[i] >> 2, tex ? tex[i] : 0);
        for (let t = complete0; t < tiles; t++)
            mark(((t % tilesX) * tile) >> 2, (Math.floor(t / tilesX) * tile) >> 2, tile >> 2, 0);
        for (let c = 0; c < 3; c++) {
            const p = planes[c];
            const alpha = (step[c] * DB_ALPHA + 8) >> 4, beta = (step[c] * DB_BETA + 8) >> 4;
            const tc = (step[c] * DB_TC + 8) >> 4;
            for (let gy = 0; gy < gh; gy++)
                for (let gx = 1; gx < gw; gx++) {
                    if (!vedge[gy * gw + gx]) continue;
                    if (!textured[gy * gw + gx] && !textured[gy * gw + gx - 1]) continue;
                    for (let y = gy * 4; y < gy * 4 + 4; y++) {
                        const r = y * pw + gx * 4;
                        const p1 = p[r - 2], p0 = p[r - 1], q0 = p[r], q1 = p[r + 1];
                        if (Math.abs(p0 - q0) >= alpha || Math.abs(p1 - p0) >= beta ||
                            Math.abs(q1 - q0) >= beta) continue;
                        let d = ((q0 - p0) * 4 + (p1 - q1) + 4) >> 3;
                        d = d < -tc ? -tc : d > tc ? tc : d;
                        p[r - 1] = clampU8(p0 + d);
                        p[r] = clampU8(q0 - d);
                    }
                }
            for (let gy = 1; gy < gh; gy++)
                for (let gx = 0; gx < gw; gx++) {
                    if (!hedge[gy * gw + gx]) continue;
                    if (!textured[gy * gw + gx] && !textured[(gy - 1) * gw + gx]) continue;
                    for (let x = gx * 4; x < gx * 4 + 4; x++) {
                        const r = gy * 4 * pw + x;
                        const p1 = p[r - 2 * pw], p0 = p[r - pw], q0 = p[r], q1 = p[r + pw];
                        if (Math.abs(p0 - q0) >= alpha || Math.abs(p1 - p0) >= beta ||
                            Math.abs(q1 - q0) >= beta) continue;
                        let d = ((q0 - p0) * 4 + (p1 - q1) + 4) >> 3;
                        d = d < -tc ? -tc : d > tc ? tc : d;
                        p[r - pw] = clampU8(p0 + d);
                        p[r] = clampU8(q0 - d);
                    }
                }
        }
    }

    const toRgb = planes => {
        const out = new Uint8Array(h.width * h.height * 3);
        const [Y, Cb, Cr] = planes;
        for (let y = 0; y < h.height; y++)
            for (let x = 0; x < h.width; x++) {
                const at = y * pw + x, o = (y * h.width + x) * 3;
                const yy = Y[at], cb = Cb[at] - 128, cr = Cr[at] - 128;
                out[o] = clampU8(yy + ((91881 * cr + 32768) >> 16));
                out[o + 1] = clampU8(yy + ((-22554 * cb - 46802 * cr + 32768) >> 16));
                out[o + 2] = clampU8(yy + ((116130 * cb + 32768) >> 16));
            }
        return out;
    };

    // The filter works in place, and a second view of the same planes has
    // to start from the unfiltered ones, so filter a copy when asked for both.
    if (ctx) {
        // Only a container decoded whole leaves the models where the
        // encoder left them (mirrors nvdr_decode_mem_ctx).
        const whole = complete[0] === tiles && complete[1] === tiles &&
            (h.band === 0 || complete[2] === tiles) && (maxLayer < 0 || maxLayer >= LAYERS - 1);
        if (whole) { ctx.cm = cm; ctx.tm = tms[0]; ctx.tm2 = tms[1]; ctx.valid = true; }
        else ctx.valid = false;
    }

    const shown = (planes, tex) => {
        if (!(h.flags & FLAG_DEBLOCK)) return toRgb(planes);
        const copy = wantFlat ? planes.map(p => p.slice()) : planes;
        deblock(copy, tex);
        return toRgb(copy);
    };

    // Before rgb, which may filter `full` in place. No snapshot means the
    // low band never arrived, and then `full` is what it would show.
    const lowRgb = !wantLow ? null
        : low ? shown(low.planes, low.tex) : shown(full.map(p => p.slice()), texturedLeaf);
    return {
        header: h,
        layersPresent: h.band && complete[1] === tiles && avail2 >= 5 ? 3
                     : complete0 === tiles && avail1 >= 5 ? 2 : 1,
        tiles,
        tilesComplete: complete,
        rgb: maxLayer === 0 ? shown(flat, null) : shown(full, texturedLeaf),
        flatRgb: wantFlat ? shown(flat, null) : null,
        lowRgb
    };
}

/**
 * The picture reduced to `w` x `h` by averaging, for every output pixel,
 * all the source pixels it covers. A browser shrinking a canvas samples
 * it instead: with `image-rendering: pixelated` one pixel in every k x k,
 * and even when smoothing, a handful. On a smooth picture that passes; on
 * grain, a night sky or a textured wall, it keeps a scatter of the
 * brightest and darkest grains and the picture shows as white noise, the
 * worse the smaller it is shown. Averaging shows it as the eye would.
 */
export function shrinkRGB(rgb, width, height, w, h) {
    const out = new Uint8ClampedArray(w * h * 3);
    const bin = new Int32Array(width), binCount = new Int32Array(w);
    for (let i = 0; i < width; i++) { bin[i] = Math.floor(i * w / width); binCount[bin[i]]++; }
    const sum = new Uint32Array(w * 3);
    let y0 = 0;
    for (let y = 0; y < h; y++) {
        const y1 = Math.floor((y + 1) * height / h);
        sum.fill(0);
        for (let j = y0; j < y1; j++) {
            let k = j * width * 3;
            for (let i = 0; i < width; i++, k += 3) {
                const b = bin[i] * 3;
                sum[b] += rgb[k]; sum[b + 1] += rgb[k + 1]; sum[b + 2] += rgb[k + 2];
            }
        }
        const rows = y1 - y0;
        for (let x = 0, o = y * w * 3; x < w; x++, o += 3) {
            const n = binCount[x] * rows, half = n >> 1, b = x * 3;
            out[o] = (sum[b] + half) / n; out[o + 1] = (sum[b + 1] + half) / n; out[o + 2] = (sum[b + 2] + half) / n;
        }
        y0 = y1;
    }
    return out;
}

/**
 * Show a picture on a canvas at the size the canvas is displayed, in
 * device pixels: reduced by averaging when that is smaller than the
 * picture (see shrinkRGB), and at its own size, drawn crisp, when it is
 * shown larger. The canvas keeps the picture and is drawn again whenever
 * its displayed size changes: a canvas painted while hidden has no size
 * yet, and one on a resized page has a new one.
 */
export function paintFitted(canvas, rgb, width, height) {
    canvas._nvdrShown = { rgb, width, height };
    if (fitObserver) fitObserver.observe(canvas);
    drawFitted(canvas);
}

const fitObserver = typeof ResizeObserver === 'undefined' ? null
    : new ResizeObserver(entries => { for (const e of entries) drawFitted(e.target); });

function drawFitted(canvas) {
    const shownPicture = canvas._nvdrShown;
    if (!shownPicture) return;
    const { rgb, width, height } = shownPicture;
    const dpr = (typeof devicePixelRatio === 'number' && devicePixelRatio) || 1;
    // Not laid out yet (hidden): a small stand-in with the right shape,
    // until the observer sees the real size.
    const shown = canvas.clientWidth || Math.min(width, 256);
    const w = Math.min(width, Math.max(1, Math.ceil(shown * dpr)));
    const h = Math.max(1, Math.round(height * w / width));
    const small = w < width;
    if (canvas.width === w && canvas.height === h && canvas._nvdrDrawn === rgb) return;
    canvas.width = w; canvas.height = h;
    canvas._nvdrDrawn = rgb;
    canvas.style.imageRendering = small ? 'auto' : 'pixelated';
    showRGB(small ? shrinkRGB(rgb, width, height, w, h) : rgb, w, h, canvas.getContext('2d'));
}

/** Put an RGB buffer on a canvas. The buffer itself is left alone. */
export function showRGB(rgb, width, height, ctx) {
    const image = ctx.createImageData(width, height);
    const pixels = image.data;
    for (let i = 0, j = 0; i < width * height * 3; i += 3, j += 4) {
        pixels[j] = rgb[i]; pixels[j + 1] = rgb[i + 1]; pixels[j + 2] = rgb[i + 2];
        pixels[j + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
    return image;
}
