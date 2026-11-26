/*
 * NVDR in the browser: everything about a container short of decoding
 * its pixels, which is the C decoder's job, compiled to WebAssembly
 * (wasm/, nvdr-wasm.js). There used to be a second decoder here, written
 * again in JavaScript; every change to the format had to be made twice,
 * and the WebAssembly one gave the same pixels and was faster. What stays:
 *
 *   readHeader()        the header and the slice table, without decoding
 *   layerThresholds()   the bytes each layer needs, for streaming
 *   extras(), alphaOf() what rides after the layers (src/extras.h)
 *   decode()            the WebAssembly decoder, once loadWasm() (or
 *                       loadWasmBytes()) has run; it throws before that
 *   newDecodeContext()  a context for containers decoded in order
 *   ArithDecoder        the range decoder, which nvdrv.js still reads a
 *                       sequence's motion field with
 *   paintFitted() ...   putting a picture on a canvas
 *
 * The format is described in src/nvdr.h.
 */

const MAGIC = 0x5244564e;   // "NVDR" read as a little-endian uint32
export const VERSION = 16;          // what a container without slices or priors is written as
export const VERSION_SLICES = 18;   // slices, priors or both; 17 was slices from even odds
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
const FLAG_TXSEL = 0x80;      // a transform type per leaf, with FLAG_DIRPRED
const GRAIN_SIZE = 22, GRAIN_POINTS = 16, GRAIN_T = 64;
const GRAIN_KERNELS = [[1, 0, 0], [8, 1, 0], [4, 1, 0], [4, 2, 1], [2, 2, 1]];
const DB_LEVEL = 8;

const PROB_BITS = 11, STATE_BITS = 15;
export const PROB_INIT = 1 << (STATE_BITS - 1);
/* A context array of n states, as the models hold them. */
export const newProbs = n => new Uint32Array(n).fill(PROB_INIT);
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
        const s = probs[index];
        let p = s & 0xffff, n = s >>> 16;
        let q = p >>> (STATE_BITS - PROB_BITS);
        q = q < 1 ? 1 : q > 2047 ? 2047 : q;
        const bound = ((this.range >>> PROB_BITS) * q) >>> 0;
        let result;
        const sh = n < 12 ? 2 : n < 40 ? 3 : n < 96 ? 4 : 6;
        if ((this.code >>> 0) < bound) {
            this.range = bound;
            p += ((1 << STATE_BITS) - p) >>> sh;
            result = 0;
        } else {
            this.code = (this.code - bound) >>> 0;
            this.range = (this.range - bound) >>> 0;
            p -= p >>> sh;
            result = 1;
        }
        if (n < 96) n++;
        probs[index] = (p | (n << 16)) >>> 0;
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

const validBlock = n => n === 4 || n === 8 || n === 16 || n === 32;

/** The header, or null where read_header() refuses it. */
export function readHeader(buffer) {
    if (buffer.byteLength < HEADER_SIZE) return null;
    // An ArrayBuffer, or a view into one such as a frame inside a sequence.
    const v = ArrayBuffer.isView(buffer)
        ? new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength)
        : new DataView(buffer);
    // Version 16 differs only in having no slice table, 15 also in having
    // no deblocking levels, 14 also in having no half nodes, 13 also in
    // having fifteen intra modes.
    if (v.getUint32(0, true) !== MAGIC || v.getUint8(4) < 13 || v.getUint8(4) > VERSION_SLICES) return null;
    const h = {
        version: v.getUint8(4),
        width: v.getUint16(6, true),
        height: v.getUint16(8, true),
        maxBlock: v.getUint8(10),
        minBlock: v.getUint8(11),
        qLuma: v.getUint16(12, true),
        qChroma: v.getUint16(14, true),
        flags: v.getUint8(5),
        storedBytes: [v.getUint32(16, true), v.getUint32(20, true), v.getUint32(24, true)],
        band: v.getUint8(28),
        deblock: [DB_LEVEL, DB_LEVEL],
        grainLen: v.getUint8(29),
        grain: null
    };
    // Mirrors read_header(): a directional picture's band is 0, and from
    // version 16 its byte holds the deblocking levels, luma's low.
    if ((h.flags & FLAG_DIRPRED) && h.version >= 16) {
        h.deblock = [h.band & 15, h.band >> 4];
        h.band = 0;
    }
    if (!h.width || !h.height || h.width * h.height > MAX_PIXELS) return null;
    if (!validBlock(h.maxBlock) || !validBlock(h.minBlock) || h.minBlock > h.maxBlock) return null;
    if (!h.qLuma || !h.qChroma) return null;
    if (h.flags & ~(FLAG_RESIDUAL | FLAG_DEBLOCK | FLAG_CHROMA420 | FLAG_GRAIN | FLAG_TILEQ | FLAG_DIRPRED | FLAG_INTER | FLAG_TXSEL)) return null;
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
    if ((h.flags & FLAG_TXSEL) && !(h.flags & FLAG_DIRPRED)) return null;
    // Mirrors the end of read_header(): restoration parameters follow the
    // grain's, bytes 30 and 31 say how many, and a residual has none.
    h.restoreLen = v.getUint16(30, true);
    if (h.restoreLen) {
        if ((h.flags & FLAG_RESIDUAL) || buffer.byteLength < HEADER_SIZE + h.grainLen + h.restoreLen)
            return null;
        // Its contents are the decoder's to read (WebAssembly, the C code).
    }
    // Mirrors slice_table(): every layer's part of every slice, [layer * slices + slice].
    h.sliceRows = 0;
    h.tableLen = 0;
    h.slices = 1;
    h.seg = h.storedBytes.slice();
    if (h.version >= 17) {
        let at = HEADER_SIZE + h.grainLen + h.restoreLen;
        const first = at, n = buffer.byteLength;
        if (at >= n) return null;
        h.sliceRows = v.getUint8(at++);
        const ty = Math.ceil(h.height / h.maxBlock);
        const ns = h.sliceRows ? Math.ceil(ty / h.sliceRows) : 1;
        h.slices = ns;
        h.seg = new Array(LAYERS * ns);
        for (let k = 0; k < LAYERS; k++) {
            let left = h.storedBytes[k];
            for (let i = 0; i < ns - 1; i++) {
                let x = 0, ok = false;
                for (let sh = 0; sh < 32; sh += 7) {
                    if (at >= n) return null;
                    const b = v.getUint8(at++);
                    x += (b & 127) * 2 ** sh;
                    if (!(b & 128)) { ok = true; break; }
                }
                if (!ok || x > left || x > 0xffffffff) return null;
                left -= x;
                h.seg[k * ns + i] = x;
            }
            h.seg[k * ns + ns - 1] = left;
        }
        h.tableLen = at - first;
        if (h.sliceRows && (h.flags & (FLAG_INTER | FLAG_RESIDUAL))) return null;
    }
    return h;
}

/** Bytes needed before each layer is complete. */
export function layerThresholds(header) {
    const a = HEADER_SIZE + header.grainLen + header.restoreLen + (header.tableLen || 0) + header.storedBytes[0],
          b = a + header.storedBytes[1];
    return [a, b, b + header.storedBytes[2]];
}

/**
 * What rides after a container's layers (src/extras.h): chunks of a tag,
 * a length and data, which no decoder reads as picture. { ALPH, ICCP },
 * each the chunk's bytes, for the tags the file has whole.
 */
export function extras(buffer) {
    const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    const h = readHeader(bytes), out = {};
    if (!h) return out;
    for (let p = layerThresholds(h)[2]; p + 8 <= bytes.length;) {
        const n = (bytes[p + 4] | bytes[p + 5] << 8 | bytes[p + 6] << 16 | bytes[p + 7] << 24) >>> 0;
        if (n > bytes.length - p - 8) break;
        out[String.fromCharCode(bytes[p], bytes[p + 1], bytes[p + 2], bytes[p + 3])] = bytes.subarray(p + 8, p + 8 + n);
        p += 8 + n;
    }
    return out;
}

/**
 * The picture's alpha plane, width * height bytes, or null when it has
 * none (or its end has not arrived): the ALPH chunk is a grey picture of
 * the same size, whose green is the alpha.
 */
export function alphaOf(buffer) {
    const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    const chunk = extras(bytes).ALPH, h = chunk && readHeader(bytes);
    const r = chunk && decode(chunk.slice());
    if (!r || r.header.width !== h.width || r.header.height !== h.height) return null;
    const n = h.width * h.height, a = new Uint8Array(n);
    for (let i = 0; i < n; i++) a[i] = r.rgb[i * 3 + 1];
    return a;
}

/** Where each layer's part of each slice lies in the file: [{ layer, slice, at, len }]. */
export function sliceParts(header) {
    const out = [];
    let at = HEADER_SIZE + header.grainLen + header.restoreLen + (header.tableLen || 0);
    for (let k = 0; k < LAYERS; k++)
        for (let i = 0; i < header.slices; i++) {
            const len = header.seg[k * header.slices + i];
            out.push({ layer: k, slice: i, at, len });
            at += len;
        }
    return out;
}

/* --- in-loop restoration, mirroring src/restore.c ---------------------- */

/*
 * The decoder: the C one compiled to WebAssembly, installed by
 * nvdr-wasm.js's loadWasm() with the contexts it keeps.
 */
let fastDecoder = null, fastContexts = null;
/* `contexts`: { alloc() -> slot or -1, reset(slot), free(slot) }. */
export function setFastDecoder(fn, contexts = null) { fastDecoder = fn; fastContexts = contexts; }

const NOT_LOADED = 'the WebAssembly decoder is not loaded: await loadWasm() (nvdr-wasm.js) first';

/*
 * A context for a run of containers decoded in order (a sequence's frames
 * of one class, a fluid album's photos): the models one leaves behind,
 * carried into the next. It lives in the WebAssembly decoder, which keeps
 * a fixed number; give it back with freeContext() when done.
 */
export function newDecodeContext() {
    if (!fastContexts) throw new Error(NOT_LOADED);
    const slot = fastContexts.alloc();
    if (slot < 0) throw new Error('no WebAssembly decoder context free: one was not given back');
    return { slot };
}
export function resetContext(ctx) { fastContexts.reset(ctx.slot); }
export function freeContext(ctx) {
    if (ctx && ctx.slot !== undefined) { fastContexts.free(ctx.slot); ctx.slot = undefined; }
}

/*
 * The picture in `buffer`, as far as its bytes reach: { header,
 * layersPresent, tiles, tilesComplete, rgb } and, when asked, flatRgb
 * (colours only) and lowRgb (the low band). Null when there is no
 * picture yet.
 *
 *   maxLayer  0 colours only, 1 the low texture band too, 2 all of it
 *   ctx       a context from newDecodeContext(), for containers in order
 *   base      { width, height, rgb }: the picture a container with
 *             FLAG_INTER (a sequence's predicted frame) is predicted from
 *   holes     byte ranges that never arrived, as [start, end) pairs
 *             (nvdr_decode_mem_lossy())
 */
export function decode(buffer, maxLayer = LAYERS - 1, wantFlat = false, ctx = null, wantLow = false, base = null,
                       holes = null) {
    if (!fastDecoder) throw new Error(NOT_LOADED);
    return fastDecoder(buffer, maxLayer, wantFlat, wantLow, base, ctx ? ctx.slot : -1, holes);
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
export function paintFitted(canvas, rgb, width, height, fitted = null, alpha = null) {
    canvas._nvdrShown = { rgb, width, height, fitted, alpha };
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
    const { rgb, width, height, fitted, alpha } = shownPicture;
    const dpr = (typeof devicePixelRatio === 'number' && devicePixelRatio) || 1;
    // Not laid out yet (hidden): a small stand-in with the right shape,
    // until the observer sees the real size.
    const { w, h } = fitSize(width, height, canvas.clientWidth || Math.min(width, 256), dpr);
    const small = w < width;
    if (canvas.width === w && canvas.height === h && canvas._nvdrDrawn === rgb && canvas._nvdrAlpha === alpha) return;
    canvas.width = w; canvas.height = h;
    canvas._nvdrDrawn = rgb;
    canvas._nvdrAlpha = alpha;
    canvas.style.imageRendering = small ? 'auto' : 'pixelated';
    // Shrunk ahead of time for this very size, off the main thread, when
    // the caller has it; otherwise here.
    const pixels = !small ? rgb
        : fitted && fitted.w === w && fitted.h === h ? fitted.rgb
        : shrinkRGB(rgb, width, height, w, h);
    // The alpha plane at the size drawn: averaged down as the colours are.
    let a = alpha;
    if (a && small) {
        const grey = new Uint8Array(width * height * 3);
        for (let i = 0; i < width * height; i++) grey[i * 3] = a[i];
        const s = shrinkRGB(grey, width, height, w, h);
        a = new Uint8Array(w * h);
        for (let i = 0; i < w * h; i++) a[i] = s[i * 3];
    }
    showRGB(pixels, w, h, canvas.getContext('2d'), a);
}

/** Put an RGB buffer on a canvas, with its alpha plane when it has one.
 * The buffers themselves are left alone. */
export function showRGB(rgb, width, height, ctx, alpha = null) {
    const image = ctx.createImageData(width, height);
    const pixels = image.data;
    for (let i = 0, j = 0, k = 0; i < width * height * 3; i += 3, j += 4, k++) {
        pixels[j] = rgb[i]; pixels[j + 1] = rgb[i + 1]; pixels[j + 2] = rgb[i + 2];
        pixels[j + 3] = alpha ? alpha[k] : 255;
    }
    ctx.putImageData(image, 0, 0);
    return image;
}

