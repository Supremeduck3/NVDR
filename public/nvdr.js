/*
 * NVDR v10 decoder — the browser half of src/nvdr.c.
 *
 * Mirrors the C decoder exactly: the same integer inverse transform, the
 * same fixed-point colour conversion, the same prediction, so the two
 * produce the same pixels. scripts/crosscheck.mjs holds them to that.
 *
 * Format, little-endian:
 *
 *   [header 32B]  "NVDR", version 10, flags, width, height, max/min block,
 *                 luma and chroma steps, the two layers' byte counts
 *   [layer 0]     the quadtree and every leaf's colour, arithmetic coded,
 *                 32x32 tile by tile
 *   [layer 1]     every leaf's texture: DCT coefficients, same order
 *
 * A short buffer is a normal outcome, not an error: every tile that
 * arrived whole is decoded, a colour tile that did not is painted grey, a texture tile that did not keeps its flat colours.
 */

const MAGIC = 0x5244564e;   // "NVDR" read as a little-endian uint32
const VERSION = 10;
const HEADER_SIZE = 32;
const MAX_PIXELS = 1 << 27; // NVDR_MAX_PIXELS
export const LAYERS = 2;
export const LAYER_NAMES = ['COR', 'COR+TEXTURA'];

const NSIZES = 4, MIN_BLOCK = 4, MAX_BLOCK = 32;
const FLAG_RESIDUAL = 0x01;   // every colour predicted as 128
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

const TMAT = [], SCAN_POS = [], SCAN_CTX = [];
for (let s = 0; s < NSIZES; s++) {
    const n = MIN_BLOCK << s, unit = MAX_BLOCK / n;
    const t = new Int32Array(n * n);
    for (let k = 0; k < n; k++)
        for (let x = 0; x < n; x++)
            t[k * n + x] = k ? cosEntry((2 * x + 1) * k * unit) : 64;
    const pos = new Int32Array(n * n), ctx = new Uint8Array(n * n);
    let at = 0;
    for (let d = 0; d <= 2 * (n - 1); d++)
        for (let v = 0; v < n; v++) {
            const u = d - v;
            if (u < 0 || u >= n) continue;
            pos[at] = v * n + u;
            ctx[at] = d < 8 ? d : 8 + Math.min((d - 8) >> 2, 6);
            at++;
        }
    TMAT.push(t); SCAN_POS.push(pos); SCAN_CTX.push(ctx);
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
function inverseDct(s, input, out, mu, mv) {
    const n = MIN_BLOCK << s, t = TMAT[s], shift2 = 6 + log2(n), half = 1 << (shift2 - 1);
    for (let y = 0; y < n; y++)
        for (let u = 0; u <= mu; u++) {
            let a = 0;
            for (let v = 0; v <= mv; v++) a += t[v * n + y] * input[v * n + u];
            TMP[y * n + u] = clampCoef((a + 32) >> 6);
        }
    for (let y = 0; y < n; y++) {
        const row = y * n;
        for (let x = 0; x < n; x++) {
            let a = 0;
            for (let u = 0; u <= mu; u++) a += t[u * n + x] * TMP[row + u];
            out[row + x] = (a + half) >> shift2;
        }
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

function getTexture(d, m, sc, c, lv, count, state) {
    lv.fill(0, 0, count);
    if (!d.bit(m.cbf[sc], c)) return false;
    let g = 0;
    for (let i = 1; i < count; i++) {
        const pc = SCAN_CTX[sc][i];
        const sig = i < count - 1 ? d.bit(m.sig[sc][c], pc) : 1;
        if (!sig) continue;
        const last = i < count - 1 ? d.bit(m.last[sc][c], pc) : 1;
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
        storedBytes: [v.getUint32(16, true), v.getUint32(20, true)]
    };
    if (!h.width || !h.height || h.width * h.height > MAX_PIXELS) return null;
    if (!validBlock(h.maxBlock) || !validBlock(h.minBlock) || h.minBlock > h.maxBlock) return null;
    if (!h.qLuma || !h.qChroma) return null;
    if (h.flags & ~FLAG_RESIDUAL) return null;
    if (h.storedBytes[0] > 0x7fffffff || h.storedBytes[1] > 0x7fffffff) return null;
    return h;
}

/** Bytes needed before each layer is complete. */
export function layerThresholds(header) {
    const a = HEADER_SIZE + header.storedBytes[0];
    return [a, a + header.storedBytes[1]];
}

/* --- decoder ----------------------------------------------------------- */

/**
 * Decode as far as the bytes reach. Returns null where nvdr_decode_mem
 * returns -1, otherwise
 *   { header, layersPresent, tiles, tilesComplete: [l0, l1],
 *     rgb,        // the image at maxLayer, RGB, width * height * 3
 *     flatRgb }   // layer 0 alone, only when wantFlat
 */
export function decode(buffer, maxLayer = LAYERS - 1, wantFlat = false) {
    const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    const h = readHeader(bytes);
    if (!h) return null;
    const size = bytes.length;

    let avail0 = Math.min(size - HEADER_SIZE, h.storedBytes[0]);
    if (avail0 < 5) return null;
    const off1 = HEADER_SIZE + h.storedBytes[0];
    const avail1 = Math.min(size > off1 ? size - off1 : 0, h.storedBytes[1]);

    const tile = h.maxBlock, minBlock = h.minBlock;
    const pw = Math.ceil(h.width / minBlock) * minBlock;
    const ph = Math.ceil(h.height / minBlock) * minBlock;
    const flat = [0, 1, 2].map(() => new Uint8Array(pw * ph));
    const full = [0, 1, 2].map(() => new Uint8Array(pw * ph));
    const step = [h.qLuma, h.qChroma, h.qChroma];
    const tilesX = Math.ceil(pw / tile), tilesY = Math.ceil(ph / tile), tiles = tilesX * tilesY;

    const whole = (x, y, n) => x + n <= pw && y + n <= ph;
    const exists = (x, y) => x < pw && y < ph;

    function fill(p, x, y, w, hh, v) {
        for (let j = y; j < y + hh; j++) p.fill(v, j * pw + x, j * pw + x + w);
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
        for (let c = 0; c < 3; c++) {
            fill(flat[c], tx, ty, w, hh, 128);
            fill(full[c], tx, ty, w, hh, 128);
        }
    }

    // Layer 0.
    const cm = colourModels();
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
            fill(flat[c], x, y, n, n, colour);
            fill(full[c], x, y, n, n, colour);
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

    // Layer 1.
    let complete1 = 0;
    if ((maxLayer < 0 || maxLayer >= 1) && avail1 >= 5 && complete0 > 0) {
        const tm = textureModels();
        const d1 = new ArithDecoder(bytes, off1, avail1);
        const s1 = { corrupt: false };
        const lv = new Int32Array(MAX_BLOCK * MAX_BLOCK);
        const coef = new Int32Array(MAX_BLOCK * MAX_BLOCK);
        const res = new Int32Array(MAX_BLOCK * MAX_BLOCK);
        for (let t = 0; t < complete0; t++) {
            for (let i = tileStart[t]; i < tileStart[t + 1] && !d1.overrun && !s1.corrupt; i++) {
                const x = lx[i], y = ly[i], n = ln[i], sc = sizeClass(n), count = n * n;
                for (let c = 0; c < 3 && !d1.overrun && !s1.corrupt; c++) {
                    if (getTexture(d1, tm, sc, c, lv, count, s1) && !d1.overrun && !s1.corrupt) {
                        coef.fill(0, 0, count);
                        const pos = SCAN_POS[sc], sh = log2(n);
                        let mu = 0, mv = 0;
                        for (let k = 1; k < count; k++) {
                            if (!lv[k]) continue;
                            const p = pos[k], u = p & (n - 1), v = p >> sh;
                            coef[p] = clampCoef(lv[k] * step[c]);
                            if (u > mu) mu = u;
                            if (v > mv) mv = v;
                        }
                        inverseDct(sc, coef, res, mu, mv);
                        const f = flat[c], o = full[c];
                        for (let j = 0; j < n; j++)
                            for (let ii = 0; ii < n; ii++) {
                                const at = (y + j) * pw + x + ii;
                                o[at] = clampU8(f[at] + res[j * n + ii]);
                            }
                    }
                }
            }
            if (d1.overrun || s1.corrupt) {
                for (let i = tileStart[t]; i < tileStart[t + 1]; i++) {
                    const x = lx[i], y = ly[i], n = ln[i];
                    for (let c = 0; c < 3; c++)
                        for (let j = 0; j < n; j++) {
                            const at = (y + j) * pw + x;
                            full[c].set(flat[c].subarray(at, at + n), at);
                        }
                }
                break;
            }
            complete1++;
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

    return {
        header: h,
        layersPresent: complete0 === tiles && avail1 >= 5 ? 2 : 1,
        tiles,
        tilesComplete: [complete0, complete1],
        rgb: toRgb(maxLayer === 0 ? flat : full),
        flatRgb: wantFlat ? toRgb(flat) : null
    };
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
