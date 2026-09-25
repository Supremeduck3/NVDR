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
const VERSION = 12;
const HEADER_SIZE = 32;
const MAX_PIXELS = 1 << 27; // NVDR_MAX_PIXELS
export const LAYERS = 3;
export const LAYER_NAMES = ['COR', 'COR+TEXTURA BAIXA', 'COR+TEXTURA COMPLETA'];

const NSIZES = 4, MIN_BLOCK = 4, MAX_BLOCK = 32;
const FLAG_RESIDUAL = 0x01;   // every colour predicted as 128
const FLAG_DEBLOCK = 0x02;    // leaf seams filtered after decoding
const FLAG_CHROMA420 = 0x04;  // colour in its own half-resolution tree
const FLAG_GRAIN = 0x08;      // grain parameters follow the header
const FLAG_TILEQ = 0x10;      // a step offset per tile, in layer 0
const FLAG_DIRPRED = 0x20;    // directional prediction, not progressive
const FLAG_INTER = 0x40;      // and from a base picture the caller holds

/* Mirrors TQ_SCALE and tq_step(): a tile's step from the header's and its
 * offset in sixths of a doubling. */
const TQ_MAX = 12;
const TQ_SCALE = [256, 287, 323, 362, 406, 456, 512, 575, 645, 724, 813, 912,
                  1024, 1149, 1290, 1448, 1625, 1825, 2048, 2299, 2580, 2896, 3251, 3649, 4096];
function tqStep(step, d) {
    const s = (step * TQ_SCALE[d + TQ_MAX] + 512) >> 10;
    return s < 1 ? 1 : s > 65535 ? 65535 : s;
}
const GRAIN_SIZE = 22, GRAIN_POINTS = 16, GRAIN_T = 64;
const GRAIN_KERNELS = [[1, 0, 0], [8, 1, 0], [4, 1, 0], [4, 2, 1], [2, 2, 1]];
const DB_ALPHA = 20, DB_BETA = 6, DB_TC = 3;
const POS_CTX = 15, MAG_UNARY = 14, EG_LIMIT = 24, COEF_MAX = 32767;
const SIG_CTX = 25, GT1_CTX = 10;

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
// Per scan position, the scan positions of the five neighbours nb_mag()
// reads, -1 off the block.
const SCAN_NB = [];

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
    const idx = new Int32Array(n * n);
    for (let i = 0; i < n * n; i++) idx[pos[i]] = i;
    const nb = new Int32Array(n * n * 5).fill(-1);
    const du = [1, 0, 1, 2, 0], dv = [0, 1, 1, 0, 2];
    for (let i = 0; i < n * n; i++) {
        const u = pos[i] & (n - 1), v = (pos[i] - u) / n;
        for (let k = 0; k < 5; k++) {
            const uu = u - du[k], vv = v - dv[k];
            if (uu >= 0 && vv >= 0) nb[i * 5 + k] = idx[vv * n + uu];
        }
    }
    TMAT.push(t); SCAN_POS.push(pos); SCAN_CTX.push(ctx); SCAN_DIAG.push(diag); SCAN_NB.push(nb);
}

/* Mirrors nb_mag(), sig_ctx() and gt1_ctx(): what is coded around a
 * position, neighbours outside the band counting as zero. */
function nbMag(lv, sc, i, start) {
    const nb = SCAN_NB[sc];
    let t = 0;
    for (let k = i * 5; k < i * 5 + 5; k++) {
        const j = nb[k];
        if (j < start || j >= i) continue;
        const a = lv[j] < 0 ? -lv[j] : lv[j];
        t += a < 3 ? a : 3;
    }
    return t;
}
function sigCtx(sc, i, t) {
    const d = SCAN_DIAG[sc][i];
    const pc = d <= 1 ? 0 : d === 2 ? 1 : d <= 4 ? 2 : d <= 7 ? 3 : 4;
    const nb = (t + 1) >> 1;
    return pc * 5 + (nb < 4 ? nb : 4);
}
const gt1Ctx = (t, g) => (t < 4 ? t : 4) + (g ? 5 : 0);

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
        splitC: probs(NSIZES),   // the colour tree's, in 4:2:0
        dcZero: grid(NSIZES, 3),
        dcSign: probs(3),
        dcMag: grid(3, MAG_UNARY),
        tqZero: probs(1), tqSign: probs(1), tqMag: probs(2 * TQ_MAX),
        modeFlat: probs(NSIZES), modeTree: probs(16), modeInter: probs(NSIZES)
    };
}

/* Mirrors get_mode(): INTER or not by size with a base, flat or not by
 * size, then four bits down a tree. */
function getMode(d, m, sc, inter) {
    if (inter && !d.bit(m.modeInter, sc)) return MODE_INTER;
    if (!d.bit(m.modeFlat, sc)) return 0;
    let v = 0, node = 1;
    for (let k = 0; k < 4; k++) {
        const b = d.bit(m.modeTree, node);
        v = 2 * v + b;
        node = 2 * node + b;
    }
    return v < MODES - 1 ? v + 1 : -1;    // the last two are damage
}

/* Mirrors the directional prediction in nvdr.c: MODE_ANGLE, angular(),
 * zorder(), decoded_before() and dir_predict(). */
const MODES = 15;
const MODE_INTER = MODES;
const MODE_ANGLE = [0, 0, -32, -17, -9, 0, 9, 17, 32, -17, -9, 0, 9, 17, 32];
const invAngle = a => (a === -32 ? -256 : a === -17 ? -482 : a === -9 ? -910 : 0);
const REF = new Int32Array(4 * MAX_BLOCK + 2), REF0 = 2 * MAX_BLOCK;
function angular(main, side, n, angle, P) {
    for (let k = 0; k <= 2 * n; k++) REF[REF0 + k] = main[k];
    if (angle < 0) {
        const last = (n * angle) >> 5, inv = invAngle(angle);
        for (let k = -1; k >= last; k--) REF[REF0 + k] = side[(k * inv + 128) >> 8];
    }
    for (let r = 0; r < n; r++) {
        const pos = (r + 1) * angle, idx = pos >> 5, fact = pos & 31;
        for (let k = 0; k < n; k++) {
            const a0 = REF[REF0 + k + idx + 1], a1 = REF[REF0 + (k + idx + 2 <= 2 * n ? k + idx + 2 : 2 * n)];
            P[r * n + k] = fact ? ((32 - fact) * a0 + fact * a1 + 16) >> 5 : a0;
        }
    }
}
function zorder(cx, cy) {
    let z = 0;
    for (let b = 0; b < 4; b++) z |= (((cx >> b) & 1) << (2 * b)) | (((cy >> b) & 1) << (2 * b + 1));
    return z;
}

/* Mirrors get_tq(). */
function getTq(d, m) {
    if (!d.bit(m.tqZero, 0)) return 0;
    const neg = d.bit(m.tqSign, 0);
    let r = 0;
    for (let i = 0; i < 2 * TQ_MAX - 1; i++) {
        if (!d.bit(m.tqMag, i)) break;
        r = i + 1;
    }
    return neg ? -(r + 1) : r + 1;
}

function textureModels() {
    return {
        cbf: grid(NSIZES, 3),
        sig: Array.from({ length: NSIZES }, () => grid(3, SIG_CTX)),
        last: Array.from({ length: NSIZES }, () => grid(3, POS_CTX)),
        gt1: grid(3, GT1_CTX),
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
        const pc = SCAN_CTX[sc][i], t = nbMag(lv, sc, i, start);
        const sig = i < end - 1 ? d.bit(m.sig[sc][c], sigCtx(sc, i, t)) : 1;
        if (!sig) continue;
        const last = i < end - 1 ? d.bit(m.last[sc][c], pc) : 1;
        let a;
        if (!d.bit(m.gt1[c], gt1Ctx(t, g))) a = 1;
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
        band: v.getUint8(28),
        grainLen: v.getUint8(29),
        grain: null
    };
    if (!h.width || !h.height || h.width * h.height > MAX_PIXELS) return null;
    if (!validBlock(h.maxBlock) || !validBlock(h.minBlock) || h.minBlock > h.maxBlock) return null;
    if (!h.qLuma || !h.qChroma) return null;
    if (h.flags & ~(FLAG_RESIDUAL | FLAG_DEBLOCK | FLAG_CHROMA420 | FLAG_GRAIN | FLAG_TILEQ | FLAG_DIRPRED | FLAG_INTER)) return null;
    if ((h.flags & FLAG_CHROMA420) && h.maxBlock < 8) return null;
    // Mirrors nvdr_grain_unpack(): parameters between the header and layer 0.
    if (h.flags & FLAG_GRAIN) {
        if (h.grainLen !== GRAIN_SIZE || buffer.byteLength < HEADER_SIZE + GRAIN_SIZE) return null;
        const g = i => v.getUint8(HEADER_SIZE + i);
        if (g(0) !== 1 || g(3) >= GRAIN_KERNELS.length) return null;
        h.grain = { seed: g(1) | (g(2) << 8), kernel: g(3), cb: g(4), cr: g(5),
                    sigma: Array.from({ length: GRAIN_POINTS }, (_, k) => g(6 + k)) };
    } else if (h.grainLen) return null;
    if (h.storedBytes.some(b => b > 0x7fffffff) || h.band > 32) return null;
    if ((h.flags & FLAG_DIRPRED) && (h.band !== 0 || (h.flags & FLAG_RESIDUAL))) return null;
    if ((h.flags & FLAG_INTER) && !(h.flags & FLAG_DIRPRED)) return null;
    return h;
}

/** Bytes needed before each layer is complete. */
export function layerThresholds(header) {
    const a = HEADER_SIZE + header.grainLen + header.storedBytes[0], b = a + header.storedBytes[1];
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

// A faster decode() with the same results, when one is loaded: the C
// decoder as WebAssembly (nvdr-wasm.js). Not for a fluid context, whose
// models only this decoder carries between containers.
let fastDecoder = null;
export function setFastDecoder(fn) { fastDecoder = fn; }

export function decode(buffer, maxLayer = LAYERS - 1, wantFlat = false, ctx = null, wantLow = false, base = null) {
    if (fastDecoder && !ctx) return fastDecoder(buffer, maxLayer, wantFlat, wantLow, base);
    return decodeJs(buffer, maxLayer, wantFlat, ctx, wantLow, base);
}

/* The JavaScript decoder itself, whichever decode() uses. */
/* A container with FLAG_INTER needs `base`, the picture it was predicted
 * from, as RGB of its size ({ width, height, rgb }); without it, it does
 * not decode. */
export function decodeJs(buffer, maxLayer = LAYERS - 1, wantFlat = false, ctx = null, wantLow = false, base = null) {
    // A texture layer that arrived whole cannot stop inside a tile unless
    // the file is damaged, so the first attempt does not save each tile to
    // restore it, a fifth of the time on a large image. If it does stop,
    // the decode starts over the careful way and gives what that gives. A
    // warm context is changed in place and cannot be started over.
    if (!(ctx && ctx.valid)) {
        const r = decodeOnce(buffer, maxLayer, wantFlat, ctx, wantLow, false, base);
        if (r !== RETRY) return r;
    }
    return decodeOnce(buffer, maxLayer, wantFlat, ctx, wantLow, true, base);
}

/*
 * One quadtree's canvas (mirrors Canvas and the decoder's Layer0 in C):
 * the whole picture in 4:4:4, three planes from Y; in 4:2:0, Y alone,
 * then Cb and Cr at half size. Planes index the arrays; components
 * (comp0 + plane) index the models and steps.
 */
function makePart(w, h, tile, minBlock, np, comp0, fixedPred, dirpred = false) {
    const pw = Math.ceil(w / minBlock) * minBlock, ph = Math.ceil(h / minBlock) * minBlock;
    const P = {
        w, h, pw, ph, tile, minBlock, np, comp0,
        flat: Array.from({ length: np }, () => new Uint8Array(pw * ph)),
        full: Array.from({ length: np }, () => new Uint8Array(pw * ph)),
        // Texture added so far, unclamped to 16 bits; full = clamp(flat + acc).
        acc: Array.from({ length: np }, () => new Int16Array(pw * ph)),
        lx: [], ly: [], ln: [],
        whole: (x, y, n) => x + n <= pw && y + n <= ph,
        exists: (x, y) => x < pw && y < ph
    };
    // TypedArray.fill costs a call per row, most of a 4-pixel row's time:
    // short rows are written directly.
    const fill = (p, x, y, ww, hh, v) => {
        if (ww > 16) {
            for (let j = y; j < y + hh; j++) p.fill(v, j * pw + x, j * pw + x + ww);
            return;
        }
        for (let j = y; j < y + hh; j++)
            for (let at = j * pw + x, end = at + ww; at < end; at++) p[at] = v;
    };
    // Mirrors paint_flat().
    P.paintFlat = (c, x, y, ww, hh, v) => {
        if (ww > 16) {
            fill(P.flat[c], x, y, ww, hh, v);
            fill(P.full[c], x, y, ww, hh, v);
            fill(P.acc[c], x, y, ww, hh, 0);
            return;
        }
        const f = P.flat[c], o = P.full[c], a = P.acc[c];
        for (let j = y; j < y + hh; j++)
            for (let at = j * pw + x, end = at + ww; at < end; at++) { f[at] = v; o[at] = v; a[at] = 0; }
    };
    P.dirpred = dirpred;
    P.predict = (c, x, y, n) => {
        if (fixedPred) return 128;
        const p = dirpred ? P.full[c] : P.flat[c];
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
    };
    const decodedBefore = (x, y, px, py) => {
        if (px < 0 || py < 0 || px >= pw || py >= ph) return false;
        const lt = Math.floor(y / tile) * 65536 + Math.floor(x / tile);
        const pt = Math.floor(py / tile) * 65536 + Math.floor(px / tile);
        if (pt !== lt) return pt < lt;
        return zorder(((px % tile) >> 2), ((py % tile) >> 2)) < zorder(((x % tile) >> 2), ((y % tile) >> 2));
    };
    const T = new Int32Array(2 * MAX_BLOCK + 1), L = new Int32Array(2 * MAX_BLOCK + 1);
    const T2 = new Int32Array(2 * MAX_BLOCK + 1), L2 = new Int32Array(2 * MAX_BLOCK + 1);
    const TMP = new Int32Array(MAX_BLOCK * MAX_BLOCK);
    P.dirPredict = (c, x, y, n, mode, out) => {
        if (mode === MODE_INTER) {
            const b = P.base[c];
            for (let j = 0; j < n; j++)
                for (let i = 0; i < n; i++) out[j * n + i] = b[(y + j) * pw + x + i];
            return;
        }
        const f = P.full[c], top = y > 0, left = x > 0;
        T.fill(0); L.fill(0);
        let tr = top, bl = left;
        for (let i = 0; i < 2 * n; i++) {
            if (i >= n && tr && !(i % 4) && !decodedBefore(x, y, x + i, y - 1)) tr = false;
            if (i >= n && bl && !(i % 4) && !decodedBefore(x, y, x - 1, y + i)) bl = false;
            T[i + 1] = !top ? 0 : (i < n || tr) ? f[(y - 1) * pw + x + i] : T[i];
            L[i + 1] = !left ? 0 : (i < n || bl) ? f[(y + i) * pw + x - 1] : L[i];
        }
        if (!top && !left) { for (let i = 0; i <= 2 * n; i++) T[i] = L[i] = 128; }
        else if (!top) { for (let i = 0; i <= 2 * n; i++) T[i] = L[1]; L[0] = L[1]; }
        else if (!left) { for (let i = 0; i <= 2 * n; i++) L[i] = T[1]; T[0] = T[1]; }
        else T[0] = L[0] = f[(y - 1) * pw + x - 1];
        if (n >= 8 && mode !== 0) {
            T2[0] = L2[0] = (T[1] + 2 * T[0] + L[1] + 2) >> 2;
            for (let i = 1; i < 2 * n; i++) {
                T2[i] = (T[i - 1] + 2 * T[i] + T[i + 1] + 2) >> 2;
                L2[i] = (L[i - 1] + 2 * L[i] + L[i + 1] + 2) >> 2;
            }
            T2[2 * n] = T[2 * n]; L2[2 * n] = L[2 * n];
            T.set(T2.subarray(0, 2 * n + 1)); L.set(L2.subarray(0, 2 * n + 1));
        }
        const sh = log2(n);
        if (mode === 1) {
            for (let j = 0; j < n; j++)
                for (let i = 0; i < n; i++)
                    out[j * n + i] = ((n - 1 - i) * L[j + 1] + (i + 1) * T[n + 1] +
                                      (n - 1 - j) * T[i + 1] + (j + 1) * L[n + 1] + n) >> (sh + 1);
            return;
        }
        if (mode <= 8) { angular(T, L, n, MODE_ANGLE[mode], out); return; }
        angular(L, T, n, MODE_ANGLE[mode], TMP);
        for (let j = 0; j < n; j++)
            for (let i = 0; i < n; i++) out[j * n + i] = TMP[i * n + j];
    };
    // Mirrors paint_pred().
    P.paintPred = (c, x, y, n, pr, offset) => {
        const f = P.flat[c], o = P.full[c], a = P.acc[c];
        for (let j = 0; j < n; j++)
            for (let i = 0; i < n; i++) {
                const at = (y + j) * pw + x + i, v = clampU8(pr[j * n + i] + offset);
                f[at] = v; o[at] = v; a[at] = 0;
            }
    };
    // Neutral grey, as tile_fallback() explains.
    P.tileFallback = (tx, ty) => {
        if (tx >= pw || ty >= ph) return;
        const ww = tx + tile < pw ? tile : pw - tx, hh = ty + tile < ph ? tile : ph - ty;
        for (let c = 0; c < np; c++) {
            P.paintFlat(c, tx, ty, ww, hh, 128);
            // With a base, a tile that never arrived shows the base.
            if (P.base)
                for (let j = ty; j < ty + hh; j++)
                    for (let at = j * pw + tx, end = at + ww; at < end; at++)
                        P.flat[c][at] = P.full[c][at] = P.base[c][at];
        }
    };
    return P;
}

/* Mirrors canvas_base(): the base picture on the canvases, in the same
 * integers as the C side. */
function canvasBase(P, C, rgb) {
    const n = P.pw * P.ph, full = [new Uint8Array(n), new Uint8Array(n), new Uint8Array(n)];
    for (let y = 0; y < P.ph; y++)
        for (let x = 0; x < P.pw; x++) {
            const sx = x < P.w ? x : P.w - 1, sy = y < P.h ? y : P.h - 1;
            const s = (sy * P.w + sx) * 3, r = rgb[s], g = rgb[s + 1], b = rgb[s + 2];
            const at = y * P.pw + x;
            full[0][at] = clampU8((19595 * r + 38470 * g + 7471 * b + 32768) >> 16);
            full[1][at] = clampU8(((-11059 * r - 21709 * g + 32768 * b + 32768) >> 16) + 128);
            full[2][at] = clampU8(((32768 * r - 27439 * g - 5329 * b + 32768) >> 16) + 128);
        }
    if (!C) { P.base = full; return; }
    P.base = [full[0]];
    C.base = [];
    for (let c = 1; c < 3; c++) {
        const hp = new Uint8Array(C.pw * C.ph);
        for (let y = 0; y < C.ph; y++)
            for (let x = 0; x < C.pw; x++) {
                let sum = 0;
                for (let j = 0; j < 2; j++)
                    for (let i = 0; i < 2; i++) {
                        let fx = 2 * x + i, fy = 2 * y + j;
                        if (fx >= P.pw) fx = P.pw - 1;
                        if (fy >= P.ph) fy = P.ph - 1;
                        sum += full[c][fy * P.pw + fx];
                    }
                hp[y * C.pw + x] = (sum + 2) >> 2;
            }
        C.base.push(hp);
    }
}

/* Mirrors upsample_plane(): colour back to full size, 9 3 3 1 in
 * sixteenths, edges held. */
function upsamplePlane(src, C, dst, dpw, w, h) {
    const cw = C.w, ch = C.h, cpw = C.pw;
    for (let y = 0; y < h; y++) {
        const cy = y >> 1;
        let oy = (y & 1) ? cy + 1 : cy - 1;
        if (oy < 0) oy = 0;
        if (oy >= ch) oy = ch - 1;
        const r0 = cy * cpw, r1 = oy * cpw, o = y * dpw;
        for (let x = 0; x < w; x++) {
            const cx = x >> 1;
            let ox = (x & 1) ? cx + 1 : cx - 1;
            if (ox < 0) ox = 0;
            if (ox >= cw) ox = cw - 1;
            dst[o + x] = (9 * src[r0 + cx] + 3 * src[r0 + ox] + 3 * src[r1 + cx] + src[r1 + ox] + 8) >> 4;
        }
    }
}

/* --- film grain, mirroring src/grain.c --------------------------------- */

const roundDiv = (n, d) => n >= 0 ? Math.floor((n + d / 2) / d) : -Math.floor((-n + d / 2) / d);

function grainHash(seed, bx, by, p) {
    let h = (Math.imul(seed, 0x9E3779B1) ^ Math.imul(bx, 0x85EBCA77) ^
             Math.imul(by, 0xC2B2AE3D) ^ Math.imul(p + 1, 0x27D4EB2F)) >>> 0;
    h = (h ^ (h >>> 15)) >>> 0;
    h = Math.imul(h, 0x2C1B3C6D) >>> 0;
    return (h ^ (h >>> 12)) >>> 0;
}

const grainCache = new Map();

/* Mirrors nvdr_grain_templates(): per component, 64x64 of grain with a
 * spread of 64, from AV1's LFSR through the kernel. */
function grainTemplates(g) {
    const key = `${g.seed}/${g.kernel}`;
    if (grainCache.has(key)) return grainCache.get(key);
    const k = GRAIN_KERNELS[g.kernel], R = GRAIN_T + 2, out = [];
    for (let p = 0; p < 3; p++) {
        let r = (g.seed ^ ((0x5A5A * (p + 1)) >>> 0)) & 0xffff;
        if (!r) r = 1;
        const draw = () => {
            const bit = ((r >> 0) ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
            r = ((r >> 1) | (bit << 15)) & 0xffff;
            return (r >> 5) & 2047;
        };
        const raw = new Int32Array(R * R);
        for (let i = 0; i < R * R; i++) raw[i] = draw() + draw() + draw() + draw() - 4096;
        const f = new Int32Array(GRAIN_T * GRAIN_T);
        let sum = 0;
        for (let y = 0; y < GRAIN_T; y++)
            for (let x = 0; x < GRAIN_T; x++) {
                const c = (y + 1) * R + x + 1;
                const v = k[0] * raw[c] + k[1] * (raw[c - 1] + raw[c + 1] + raw[c - R] + raw[c + R]) +
                          k[2] * (raw[c - R - 1] + raw[c - R + 1] + raw[c + R - 1] + raw[c + R + 1]);
                f[y * GRAIN_T + x] = v;
                sum += v;
            }
        const mean = Math.floor(sum / (GRAIN_T * GRAIN_T));
        let variance = 0;
        for (let i = 0; i < f.length; i++) variance += (f[i] - mean) * (f[i] - mean);
        variance = Math.floor(variance / (GRAIN_T * GRAIN_T));
        let sd = Math.floor(Math.sqrt(variance));
        while (sd * sd > variance) sd--;
        while ((sd + 1) * (sd + 1) <= variance) sd++;
        if (sd < 1) sd = 1;
        const t = new Int16Array(GRAIN_T * GRAIN_T);
        for (let i = 0; i < f.length; i++) {
            const v = Math.trunc((f[i] - mean) * 64 / sd);
            t[i] = v < -255 ? -255 : v > 255 ? 255 : v;
        }
        out.push(t);
    }
    grainCache.set(key, out);
    return out;
}

function decodeOnce(buffer, maxLayer, wantFlat, ctx, wantLow, careful, baseImg) {
    const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    const h = readHeader(bytes);
    if (!h) return null;
    const size = bytes.length;

    const base = HEADER_SIZE + h.grainLen;
    let avail0 = Math.min(size - base, h.storedBytes[0]);
    if (avail0 < 5) return null;
    const off1 = base + h.storedBytes[0];
    const avail1 = Math.min(size > off1 ? size - off1 : 0, h.storedBytes[1]);
    const off2 = off1 + h.storedBytes[1];
    const avail2 = Math.min(size > off2 ? size - off2 : 0, h.storedBytes[2]);
    const bandAt = [0, 1, 2, 3].map(sc => bandSplit(sc, h.band));

    // One tree in 4:4:4; in 4:2:0 luma's, then colour's at half size,
    // tile by tile in the same streams. A tile is whole when both are.
    const fixedPred = (h.flags & FLAG_RESIDUAL) !== 0;
    const is420 = (h.flags & FLAG_CHROMA420) !== 0;
    const dirpred = (h.flags & FLAG_DIRPRED) !== 0;
    const parts = [makePart(h.width, h.height, h.maxBlock, h.minBlock, is420 ? 1 : 3, 0, fixedPred, dirpred)];
    if (is420) parts.push(makePart((h.width + 1) >> 1, (h.height + 1) >> 1, h.maxBlock >> 1, MIN_BLOCK, 2, 1, fixedPred, dirpred));
    const inter = (h.flags & FLAG_INTER) !== 0;
    if (inter) {
        if (!baseImg || baseImg.width !== h.width || baseImg.height !== h.height) return null;
        canvasBase(parts[0], parts[1] || null, baseImg.rgb);
    }
    const main = parts[0], tile = main.tile, pw = main.pw;
    const step = [h.qLuma, h.qChroma, h.qChroma];
    const tilesX = Math.ceil(main.pw / tile), tilesY = Math.ceil(main.ph / tile), tiles = tilesX * tilesY;
    for (const P of parts) P.tileStart = new Int32Array(tiles + 1);

    // Layer 0.
    const warm = ctx && ctx.valid;
    const cm = warm ? ctx.cm : colourModels();
    if (!cm.splitC) cm.splitC = probs(NSIZES);
    if (!cm.tqZero) { cm.tqZero = probs(1); cm.tqSign = probs(1); cm.tqMag = probs(2 * TQ_MAX); }
    if (!cm.modeFlat) { cm.modeFlat = probs(NSIZES); cm.modeTree = probs(16); }
    if (!cm.modeInter) cm.modeInter = probs(NSIZES);
    // The tiles' step offsets as layer 0 delivers them, zero where it
    // never did, and the step of the tile being read.
    const tq = (h.flags & FLAG_TILEQ) ? new Int8Array(tiles) : null;
    const tstep = step.slice();
    const tms = warm ? [ctx.tm, ctx.tm2] : [textureModels(), textureModels()];
    const d0 = new ArithDecoder(bytes, base, avail0);
    const s0 = { corrupt: false };

    // Mirrors texture_residual() + add_residual(): texture levels in
    // [start, end) added to a leaf.
    const tcoef = new Int32Array(MAX_BLOCK * MAX_BLOCK), tres = new Int32Array(MAX_BLOCK * MAX_BLOCK);
    function applyTex(P, c, x, y, n, lv, start, end, stepv) {
        const sc = sizeClass(n), count = n * n, pos = SCAN_POS[sc], sh = log2(n), ppw = P.pw;
        tcoef.fill(0, 0, count);
        let mu = 0, mv = 0;
        for (let q = start; q < end; q++) {
            if (!lv[q]) continue;
            const p = pos[q], u = p & (n - 1), v = p >> sh;
            tcoef[p] = clampCoef(lv[q] * stepv);
            if (u > mu) mu = u;
            if (v > mv) mv = v;
        }
        inverseDct(sc, tcoef, tres, mu, mv);
        const o = P.full[c], f = P.flat[c], a = P.acc[c];
        for (let j = 0; j < n; j++)
            for (let ii = 0; ii < n; ii++) {
                const at = (y + j) * ppw + x + ii;
                let v = a[at] + tres[j * n + ii];
                v = v < -32768 ? -32768 : v > 32767 ? 32767 : v;
                a[at] = v;
                o[at] = clampU8(f[at] + v);
            }
    }
    // With directional prediction a leaf's texture comes from the first
    // texture layer as soon as its colour does.
    const d1 = dirpred ? new ArithDecoder(bytes, off1, avail1) : null;
    const lvd = new Int32Array(MAX_BLOCK * MAX_BLOCK), prd = new Int32Array(MAX_BLOCK * MAX_BLOCK);
    for (const P of parts) P.ltex = [];

    function readNode(P, x, y, n) {
        let split = 0;
        if (n > P.minBlock) split = P.whole(x, y, n) ? d0.bit(P.comp0 ? cm.splitC : cm.split, sizeClass(n)) : 1;
        if (d0.overrun || s0.corrupt) return;
        if (split) {
            const hh = n >> 1;
            for (let k = 0; k < 4; k++) {
                const cx = x + (k & 1) * hh, cy = y + (k >> 1) * hh;
                if (P.exists(cx, cy)) readNode(P, cx, cy, hh);
                if (d0.overrun || s0.corrupt) return;
            }
            return;
        }
        const sc = sizeClass(n);
        let mode = 0;
        if (dirpred) {
            mode = getMode(d0, cm, sc, inter);
            if (mode < 0) { s0.corrupt = true; return; }
        }
        for (let c = 0; c < P.np; c++) {
            const k = P.comp0 + c;
            const dl = getDc(d0, cm, sc, k, s0);
            const offset = divRound(clampCoef(dl * tstep[k]), n);
            if (mode > 0) {
                P.dirPredict(c, x, y, n, mode, prd);
                P.paintPred(c, x, y, n, prd, offset);
            } else P.paintFlat(c, x, y, n, n, clampU8(P.predict(c, x, y, n) + offset));
        }
        P.lx.push(x); P.ly.push(y); P.ln.push(n);
        let tex = 0;
        if (dirpred) {
            const count = n * n;
            for (let c = 0; c < P.np && !s0.corrupt && !d1.overrun; c++)
                if (getTexture(d1, tms[0], sc, P.comp0 + c, lvd, 1, count, s0) && !s0.corrupt && !d1.overrun) {
                    applyTex(P, c, x, y, n, lvd, 1, count, tstep[P.comp0 + c]);
                    tex = 1;
                }
            if (d1.overrun) s0.corrupt = true;
        }
        P.ltex.push(tex);
    }

    let complete0 = 0, stopped = false, tqPrev = 0;
    for (let t = 0; t < tiles; t++) {
        const tx = (t % tilesX) * tile, ty = Math.floor(t / tilesX) * tile;
        for (const P of parts) P.tileStart[t] = P.lx.length;
        if (tq && !stopped) {
            const dq = tqPrev + getTq(d0, cm);
            if (d0.overrun || dq < -TQ_MAX || dq > TQ_MAX) stopped = true;
            else {
                tq[t] = dq; tqPrev = dq;
                for (let c = 0; c < 3; c++) tstep[c] = tqStep(step[c], dq);
            }
        }
        for (let k = 0; k < parts.length && !stopped; k++) {
            const P = parts[k], x = tx / (k + 1), y = ty / (k + 1);
            if (!P.exists(x, y)) continue;
            readNode(P, x, y, P.tile);
            if (d0.overrun || s0.corrupt) stopped = true;
        }
        if (stopped) {
            parts.forEach((P, k) => {
                P.lx.length = P.ly.length = P.ln.length = P.ltex.length = P.tileStart[t];
                P.tileFallback(tx / (k + 1), ty / (k + 1));
            });
        } else complete0++;
    }
    for (const P of parts) P.tileStart[tiles] = P.lx.length;

    // Layers 1 and 2, the low and the high texture band. Mirrors the C
    // loop: each band as far as its bytes reach and no further than the
    // layer before it; a tile cut short is restored to what it showed.
    const complete = [complete0, 0, 0];
    if (dirpred) { complete[1] = complete0; maxLayer = LAYERS - 1; wantLow = false; }
    for (const P of parts) {
        P.textured = dirpred ? Uint8Array.from(P.ltex) : new Uint8Array(P.lx.length);
        P.saved = Array.from({ length: P.np }, () => new Uint8Array(P.tile * P.tile));
        P.savedAcc = Array.from({ length: P.np }, () => new Int16Array(P.tile * P.tile));
    }
    const lv = new Int32Array(MAX_BLOCK * MAX_BLOCK);
    const coef = new Int32Array(MAX_BLOCK * MAX_BLOCK);
    const res = new Int32Array(MAX_BLOCK * MAX_BLOCK);
    let low = null;
    const region = (P, t) => {
        const tx = (t % tilesX) * P.tile, ty = Math.floor(t / tilesX) * P.tile;
        if (!P.exists(tx, ty)) return null;
        return { tx, ty, tw: tx + P.tile < P.pw ? P.tile : P.pw - tx, th: ty + P.tile < P.ph ? P.tile : P.ph - ty };
    };
    for (let layer = 1; layer < LAYERS; layer++) {
        const avail = layer === 1 ? avail1 : avail2, off = layer === 1 ? off1 : off2;
        if (dirpred || (maxLayer >= 0 && maxLayer < layer) || avail < 5 || complete[layer - 1] === 0) break;
        if (layer === 2 && h.band === 0) break;
        const tm = tms[layer - 1];
        const d = new ArithDecoder(bytes, off, avail);
        const guard = careful || avail < h.storedBytes[layer];
        const st = { corrupt: false };
        for (let t = 0; t < complete[layer - 1]; t++) {
            // Copied by hand: a subarray per row would be millions of
            // short-lived objects on a large image.
            if (guard) for (const P of parts) {
                const r = region(P, t);
                if (!r) continue;
                for (let c = 0; c < P.np; c++) {
                    const o = P.full[c], a = P.acc[c], so = P.saved[c], sa = P.savedAcc[c];
                    for (let j = 0; j < r.th; j++)
                        for (let at = (r.ty + j) * P.pw + r.tx, k = j * P.tile, end = at + r.tw; at < end; at++, k++) {
                            so[k] = o[at]; sa[k] = a[at];
                        }
                }
                P.was = P.textured.slice(P.tileStart[t], P.tileStart[t + 1]);
            }
            const tileStep = tq ? step.map(q => tqStep(q, tq[t])) : step;
            for (let k = 0; k < parts.length && !d.overrun && !st.corrupt; k++) {
                const P = parts[k], ppw = P.pw;
                for (let i = P.tileStart[t]; i < P.tileStart[t + 1] && !d.overrun && !st.corrupt; i++) {
                    const x = P.lx[i], y = P.ly[i], n = P.ln[i], sc = sizeClass(n), count = n * n;
                    const start = layer === 1 ? 1 : bandAt[sc], end = layer === 1 ? bandAt[sc] : count;
                    if (start >= end) continue;
                    for (let c = 0; c < P.np && !d.overrun && !st.corrupt; c++) {
                        const comp = P.comp0 + c;
                        if (getTexture(d, tm, sc, comp, lv, start, end, st) && !d.overrun && !st.corrupt) {
                            coef.fill(0, 0, count);
                            const pos = SCAN_POS[sc], sh = log2(n);
                            let mu = 0, mv = 0;
                            for (let q = start; q < end; q++) {
                                if (!lv[q]) continue;
                                const p = pos[q], u = p & (n - 1), v = p >> sh;
                                coef[p] = clampCoef(lv[q] * tileStep[comp]);
                                if (u > mu) mu = u;
                                if (v > mv) mv = v;
                            }
                            inverseDct(sc, coef, res, mu, mv);
                            P.textured[i] = 1;
                            const o = P.full[c], f = P.flat[c], a = P.acc[c];
                            for (let j = 0; j < n; j++)
                                for (let ii = 0; ii < n; ii++) {
                                    const at = (y + j) * ppw + x + ii;
                                    let v = a[at] + res[j * n + ii];
                                    v = v < -32768 ? -32768 : v > 32767 ? 32767 : v;
                                    a[at] = v;
                                    o[at] = clampU8(f[at] + v);
                                }
                        }
                    }
                }
            }
            if (d.overrun || st.corrupt) {
                if (!guard) return RETRY;
                for (const P of parts) {
                    const r = region(P, t);
                    if (!r) continue;
                    for (let c = 0; c < P.np; c++)
                        for (let j = 0; j < r.th; j++) {
                            P.full[c].set(P.saved[c].subarray(j * P.tile, j * P.tile + r.tw), (r.ty + j) * P.pw + r.tx);
                            P.acc[c].set(P.savedAcc[c].subarray(j * P.tile, j * P.tile + r.tw), (r.ty + j) * P.pw + r.tx);
                        }
                    P.textured.set(P.was, P.tileStart[t]);
                }
                break;
            }
            complete[layer]++;
        }
        // What decode(buffer, 1) would show, without a second pass.
        if (layer === 1 && wantLow)
            low = { planes: parts.map(P => P.full.map(p => p.slice())), tex: parts.map(P => P.textured.slice()) };
    }

    /* Mirrors deblock() and deblock_plane(); `tex` is null for colours only. */
    function deblock(P, planes, tex) {
        const ppw = P.pw, gw = P.pw >> 2, gh = P.ph >> 2, ptile = P.tile;
        const vedge = new Uint8Array(gw * gh), hedge = new Uint8Array(gw * gh);
        const textured = new Uint8Array(gw * gh);   // the cell's leaf shows texture
        const mark = (x, y, n, t) => {
            for (let j = y; j < y + n && j < gh; j++)
                for (let k = x; k < x + n && k < gw; k++) textured[j * gw + k] = t;
            if (x > 0) for (let j = y; j < y + n && j < gh; j++) vedge[j * gw + x] = 1;
            if (y > 0) for (let k = x; k < x + n && k < gw; k++) hedge[y * gw + k] = 1;
        };
        const kept = P.tileStart[complete0];
        for (let i = 0; i < kept; i++) mark(P.lx[i] >> 2, P.ly[i] >> 2, P.ln[i] >> 2, tex ? tex[i] : 0);
        for (let t = complete0; t < tiles; t++)
            mark(((t % tilesX) * ptile) >> 2, (Math.floor(t / tilesX) * ptile) >> 2, ptile >> 2, 0);
        for (let c = 0; c < P.np; c++) {
            const p = planes[c], base = step[P.comp0 + c];
            // The step at an edge is its second side's tile's.
            let alpha = 0, beta = 0, tc = 0;
            const at = (gx, gy) => {
                const sp = tq ? tqStep(base, tq[Math.floor(gy * 4 / ptile) * tilesX + Math.floor(gx * 4 / ptile)]) : base;
                alpha = (sp * DB_ALPHA + 8) >> 4; beta = (sp * DB_BETA + 8) >> 4; tc = (sp * DB_TC + 8) >> 4;
            };
            at(0, 0);
            for (let gy = 0; gy < gh; gy++)
                for (let gx = 1; gx < gw; gx++) {
                    if (!vedge[gy * gw + gx]) continue;
                    if (!textured[gy * gw + gx] && !textured[gy * gw + gx - 1]) continue;
                    if (tq) at(gx, gy);
                    for (let y = gy * 4; y < gy * 4 + 4; y++) {
                        const r = y * ppw + gx * 4;
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
                    if (tq) at(gx, gy);
                    for (let x = gx * 4; x < gx * 4 + 4; x++) {
                        const r = gy * 4 * ppw + x;
                        const p1 = p[r - 2 * ppw], p0 = p[r - ppw], q0 = p[r], q1 = p[r + ppw];
                        if (Math.abs(p0 - q0) >= alpha || Math.abs(p1 - p0) >= beta ||
                            Math.abs(q1 - q0) >= beta) continue;
                        let d = ((q0 - p0) * 4 + (p1 - q1) + 4) >> 3;
                        d = d < -tc ? -tc : d > tc ? tc : d;
                        p[r - ppw] = clampU8(p0 + d);
                        p[r] = clampU8(q0 - d);
                    }
                }
        }
    }

    // `views` holds each part's planes; colour comes back to full size first.
    const toRgb = views => {
        const out = new Uint8Array(h.width * h.height * 3);
        let [Y, Cb, Cr] = views[0];
        if (is420) {
            Cb = new Uint8Array(pw * main.ph); Cr = new Uint8Array(pw * main.ph);
            upsamplePlane(views[1][0], parts[1], Cb, pw, h.width, h.height);
            upsamplePlane(views[1][1], parts[1], Cr, pw, h.width, h.height);
        }
        const G = h.grain, T = G && grainTemplates(G);
        for (let y = 0; y < h.height; y++) {
            let bxAt = -1, o0 = 0, o1 = 0, o2 = 0;
            for (let x = 0; x < h.width; x++) {
                const at = y * pw + x, o = (y * h.width + x) * 3;
                let yy = Y[at], cb = Cb[at] - 128, cr = Cr[at] - 128;
                if (G) {
                    // Mirrors nvdr_grain_apply().
                    if ((x >> 5) !== bxAt) {
                        bxAt = x >> 5;
                        const off = p => {
                            const hh = grainHash(G.seed, bxAt, y >> 5, p);
                            return ((hh >>> 5) & 31) * GRAIN_T + (y & 31) * GRAIN_T + (hh & 31) - (bxAt << 5);
                        };
                        o0 = off(0); o1 = off(1); o2 = off(2);
                    }
                    const idx = (yy / 17) | 0, frac = yy - idx * 17, nxt = idx + 1 < GRAIN_POINTS ? idx + 1 : GRAIN_POINTS - 1;
                    const s = G.sigma[idx] * (17 - frac) + G.sigma[nxt] * frac;
                    const lum = yy;
                    yy = clampU8(lum + roundDiv(T[0][o0 + x] * s, 64 * 8 * 17));
                    cb = clampU8(cb + 128 + roundDiv(T[1][o1 + x] * s * G.cb, 64 * 8 * 17 * 32)) - 128;
                    cr = clampU8(cr + 128 + roundDiv(T[2][o2 + x] * s * G.cr, 64 * 8 * 17 * 32)) - 128;
                }
                out[o] = clampU8(yy + ((91881 * cr + 32768) >> 16));
                out[o + 1] = clampU8(yy + ((-22554 * cb - 46802 * cr + 32768) >> 16));
                out[o + 2] = clampU8(yy + ((116130 * cb + 32768) >> 16));
            }
        }
        return out;
    };

    if (ctx) {
        // Only a container decoded whole leaves the models where the
        // encoder left them (mirrors nvdr_decode_mem_ctx).
        const whole = complete[0] === tiles && complete[1] === tiles &&
            (h.band === 0 || complete[2] === tiles) && (maxLayer < 0 || maxLayer >= LAYERS - 1);
        if (whole) { ctx.cm = cm; ctx.tm = tms[0]; ctx.tm2 = tms[1]; ctx.valid = true; }
        else ctx.valid = false;
    }

    // The filter works in place, and a second view of the same planes has
    // to start from the unfiltered ones, so filter a copy when asked for both.
    const shown = (views, texs) => {
        if (!(h.flags & FLAG_DEBLOCK)) return toRgb(views);
        const copy = wantFlat ? views.map(v => v.map(p => p.slice())) : views;
        parts.forEach((P, k) => deblock(P, copy[k], texs && texs[k]));
        return toRgb(copy);
    };

    const flats = parts.map(P => P.flat), fulls = parts.map(P => P.full), texs = parts.map(P => P.textured);
    // Before rgb, which may filter `full` in place. No snapshot means the
    // low band never arrived, and then `full` is what it would show.
    const lowRgb = !wantLow ? null
        : low ? shown(low.planes, low.tex) : shown(fulls.map(v => v.map(p => p.slice())), texs);
    return {
        header: h,
        layersPresent: h.band && complete[1] === tiles && avail2 >= 5 ? 3
                     : complete0 === tiles && avail1 >= 5 ? 2 : 1,
        tiles,
        tilesComplete: complete,
        rgb: maxLayer === 0 ? shown(flats, null) : shown(fulls, texs),
        flatRgb: wantFlat ? shown(flats, null) : null,
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
export function paintFitted(canvas, rgb, width, height, fitted = null) {
    canvas._nvdrShown = { rgb, width, height, fitted };
    if (fitObserver) fitObserver.observe(canvas);
    drawFitted(canvas);
}

/**
 * The size, in device pixels, a width x height picture is drawn at on a
 * canvas `shownWidth` CSS pixels wide: never larger than the picture.
 * A worker shrinking a picture ahead of time (see `fit` in
 * nvdr-tasks.js) uses this too, so its result is the one drawFitted
 * would have made.
 */
export function fitSize(width, height, shownWidth, dpr = 1) {
    const w = Math.min(width, Math.max(1, Math.ceil(shownWidth * dpr)));
    return { w, h: Math.max(1, Math.round(height * w / width)) };
}

/* The device-pixel width a canvas is shown at, for fitSize(). */
export function shownWidth(canvas) {
    const dpr = (typeof devicePixelRatio === 'number' && devicePixelRatio) || 1;
    return Math.ceil((canvas.clientWidth || 0) * dpr);
}

const fitObserver = typeof ResizeObserver === 'undefined' ? null
    : new ResizeObserver(entries => { for (const e of entries) drawFitted(e.target); });

function drawFitted(canvas) {
    const shownPicture = canvas._nvdrShown;
    if (!shownPicture) return;
    const { rgb, width, height, fitted } = shownPicture;
    const dpr = (typeof devicePixelRatio === 'number' && devicePixelRatio) || 1;
    // Not laid out yet (hidden): a small stand-in with the right shape,
    // until the observer sees the real size.
    const { w, h } = fitSize(width, height, canvas.clientWidth || Math.min(width, 256), dpr);
    const small = w < width;
    if (canvas.width === w && canvas.height === h && canvas._nvdrDrawn === rgb) return;
    canvas.width = w; canvas.height = h;
    canvas._nvdrDrawn = rgb;
    canvas.style.imageRendering = small ? 'auto' : 'pixelated';
    // Shrunk ahead of time for this very size, off the main thread, when
    // the caller has it; otherwise here.
    const pixels = !small ? rgb
        : fitted && fitted.w === w && fitted.h === h ? fitted.rgb
        : shrinkRGB(rgb, width, height, w, h);
    showRGB(pixels, w, h, canvas.getContext('2d'));
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
