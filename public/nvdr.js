/*
 * NVDR container decoder — the browser half of the Progressive Residual
 * Stack.
 *
 * This mirrors nvdr.c exactly, including the part that matters most: it
 * takes whatever bytes it is given. Hand it a complete container and it
 * decodes three levels; hand it the first 5% of one and it decodes the
 * anchor and reports that the rest never arrived. A short buffer is a
 * normal outcome here, not an error path.
 *
 * Format (v3), little-endian throughout:
 *
 *   [header 72B]
 *   [level 0 : palette, split bitstream, packed anchor tokens]
 *   [level 1 : split bitstream, 3 int8 residual planes]
 *   [level 2 : split bitstream, 3 int8 residual planes]
 *
 * Each level is deflated on its own. Compressing the container as a whole
 * would be smaller and would make every prefix undecodable, which is the
 * one property this format cannot trade away — so the streams compress
 * separately and both hold at once. Inflation is why decode() is async:
 * DecompressionStream is the only inflate the platform gives us.
 *
 * Geometry is never transmitted as coordinates. Each level carries one bit
 * per visited quadtree node — 1 splits, 0 stops — and the decoder replays
 * the same subdivision the encoder used, starting from the canvas for
 * level 0 and from each previous rectangle for the levels after it.
 */

const MAGIC = 0x5244564e; // "NVDR" read as a little-endian uint32
const VERSION = 4;
const HEADER_SIZE = 72;
const COMPRESS_NONE = 0;
const COMPRESS_DEFLATE = 1;
const COMPRESS_ARITH = 2;

/* --- entropy layer, mirroring nvdr/entropy.c ------------------------- */

const PROB_BITS = 11;
const PROB_INIT = 1 << (PROB_BITS - 1);
const MOVE_BITS = 5;
const TOP_VALUE = 1 << 24;
const AREA_CTX = 16;
const MAG_CTX = 8;

function newModels() {
    return {
        split: new Uint16Array(AREA_CTX).fill(PROB_INIT),
        token: new Uint16Array(256).fill(PROB_INIT),
        sig: [new Uint16Array(3).fill(PROB_INIT), new Uint16Array(3).fill(PROB_INIT)],
        sign: [new Uint16Array(3).fill(PROB_INIT), new Uint16Array(3).fill(PROB_INIT)],
        mag: [
            [new Uint16Array(MAG_CTX).fill(PROB_INIT), new Uint16Array(MAG_CTX).fill(PROB_INIT),
             new Uint16Array(MAG_CTX).fill(PROB_INIT)],
            [new Uint16Array(MAG_CTX).fill(PROB_INIT), new Uint16Array(MAG_CTX).fill(PROB_INIT),
             new Uint16Array(MAG_CTX).fill(PROB_INIT)]
        ]
    };
}

/* Same bucketing as nvdr_area_context: both sides must index the same slot. */
function areaContext(w, h) {
    let area = w * h;
    let bucket = 0;
    while (area > 1 && bucket < AREA_CTX - 1) { area >>>= 1; bucket++; }
    return bucket;
}

/*
 * The LZMA range decoder. Arithmetic is forced through >>> because the
 * coder works on unsigned 32-bit words and JavaScript's bitwise operators
 * are signed.
 */
class ArithDecoder {
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

    residual(models, splitCtx, channel) {
        if (!this.bit(models.sig[splitCtx], channel)) return 0;
        const negative = this.bit(models.sign[splitCtx], channel);

        let remaining = 0;
        let i = 0;
        for (; i < MAG_CTX; i++) {
            if (!this.bit(models.mag[splitCtx][channel], i)) break;
            remaining = i + 1;
        }
        if (i === MAG_CTX) remaining = MAG_CTX + this.direct(7);

        const magnitude = remaining + 1;
        return negative ? -magnitude : magnitude;
    }
}

/* The mirror of replay(), reading split decisions from the arithmetic coder. */
function replayArith(dec, models, rects, x, y, w, h) {
    if (dec.overrun) return false;
    if (!dec.bit(models.split, areaContext(w, h))) return rects.push(x, y, w, h);

    const hw = w >> 1, hh = h >> 1;
    const rw = w - hw, rh = h - hh;
    return replayArith(dec, models, rects, x, y, hw, hh)
        && replayArith(dec, models, rects, x + hw, y, rw, hh)
        && replayArith(dec, models, rects, x, y + hh, hw, rh)
        && replayArith(dec, models, rects, x + hw, y + hh, rw, rh);
}
export const LEVELS = 3;
export const LEVEL_NAMES = ['ANCHOR', 'ANCHOR+R1', 'ANCHOR+R1+R2'];

class BitReader {
    constructor(bytes, offset, bitCount) {
        this.bytes = bytes;
        this.offset = offset;
        this.bitCount = bitCount;
        this.bitPos = 0;
        this.overrun = false;
    }

    next() {
        if (this.bitPos >= this.bitCount) {
            this.overrun = true;
            return 0;
        }
        const bit = (this.bytes[this.offset + (this.bitPos >> 3)] >> (this.bitPos & 7)) & 1;
        this.bitPos++;
        return bit;
    }
}

/*
 * Rectangles are kept in parallel typed arrays rather than objects: a
 * level of a 4K image can hold hundreds of thousands of them, and one
 * object per rectangle is what turns a 20ms decode into a 400ms one.
 */
class RectSet {
    constructor(capacity) {
        this.x = new Uint16Array(capacity);
        this.y = new Uint16Array(capacity);
        this.w = new Uint16Array(capacity);
        this.h = new Uint16Array(capacity);
        this.count = 0;
    }

    push(x, y, w, h) {
        if (this.count >= this.x.length) return false;
        const i = this.count++;
        this.x[i] = x; this.y[i] = y; this.w[i] = w; this.h[i] = h;
        return true;
    }
}

/* The mirror of cut_level() in nvdr.c. */
function replay(reader, rects, x, y, w, h) {
    if (reader.overrun) return false;
    if (!reader.next()) return rects.push(x, y, w, h);

    const hw = w >> 1, hh = h >> 1;
    const rw = w - hw, rh = h - hh;
    return replay(reader, rects, x, y, hw, hh)
        && replay(reader, rects, x + hw, y, rw, hh)
        && replay(reader, rects, x, y + hh, hw, rh)
        && replay(reader, rects, x + hw, y + hh, rw, rh);
}

export function readHeader(buffer) {
    if (buffer.byteLength < HEADER_SIZE) return null;
    const view = new DataView(buffer);
    if (view.getUint32(0, true) !== MAGIC) return null;
    if (view.getUint8(4) !== VERSION) return null;

    const header = {
        width: view.getUint16(6, true),
        height: view.getUint16(8, true),
        anchorBits: view.getUint8(10),
        compression: view.getUint8(5),
        step: [],
        leafCount: [],
        splitBits: [],
        rawBytes: [],
        storedBytes: []
    };
    for (let k = 0; k < LEVELS; k++) {
        header.step.push(view.getUint8(11 + k));
        header.leafCount.push(view.getUint32(16 + k * 4, true));
        header.splitBits.push(view.getUint32(28 + k * 4, true));
        header.rawBytes.push(view.getUint32(40 + k * 4, true));
        header.storedBytes.push(view.getUint32(52 + k * 4, true));
    }
    return header;
}

/*
 * How many bytes a decoder needs before each level becomes displayable.
 * These are stored bytes, so they are also what the level costs on the
 * wire — the container is not compressed again on top of itself.
 */
export function levelThresholds(header) {
    const thresholds = [];
    let total = HEADER_SIZE;
    for (let k = 0; k < LEVELS; k++) {
        total += header.storedBytes[k];
        thresholds.push(total);
    }
    return thresholds;
}

/*
 * Inflate one level. Returns null when the bytes are not all there, which
 * for a truncated container is the ordinary outcome rather than an error.
 */
async function readStream(bytes, offset, header, k) {
    const stored = header.storedBytes[k];
    if (stored === 0 || offset + stored > bytes.length) return null;

    const packed = bytes.subarray(offset, offset + stored);
    if (header.compression !== COMPRESS_DEFLATE) return packed;

    try {
        const stream = new Blob([packed]).stream()
            .pipeThrough(new DecompressionStream('deflate'));
        const raw = new Uint8Array(await new Response(stream).arrayBuffer());
        return raw.length === header.rawBytes[k] ? raw : null;
    } catch (err) {
        return null;   // a stream cut mid-block throws; that is a short file
    }
}

function clampByte(v) {
    return v < 0 ? 0 : (v > 255 ? 255 : v);
}

/**
 * Decode as far as the supplied bytes reach.
 *
 * Returns { header, levels, levelsPresent } where levels[k] holds the
 * rectangles and one RGB triple each. `levelsPresent` is how many of them
 * the bytes actually paid for — 0 means not even the anchor arrived, which
 * is the only case that yields no picture.
 */
export async function decode(buffer) {
    const header = readHeader(buffer);
    if (!header) return null;

    const bytes = new Uint8Array(buffer);
    const levels = [];
    let cursor = HEADER_SIZE;

    /* --- level 0: the contract --- */
    const anchorStream = await readStream(bytes, cursor, header, 0);
    if (!anchorStream) return { header, levels, levelsPresent: 0 };

    const arith = header.compression === COMPRESS_ARITH;
    let offset = 0;
    const paletteCount = anchorStream[offset++] + 1;   // stored biased by one
    const palette = anchorStream.subarray(offset, offset + paletteCount * 3);
    offset += paletteCount * 3;

    const anchorRects = new RectSet(header.leafCount[0]);
    const tokens = new Uint32Array(header.leafCount[0]);

    if (arith) {
        const models = newModels();
        const dec = new ArithDecoder(anchorStream, offset, anchorStream.length - offset);
        if (!replayArith(dec, models, anchorRects, 0, 0, header.width, header.height)
            || dec.overrun
            || anchorRects.count !== header.leafCount[0]) {
            return { header, levels, levelsPresent: 0 };
        }
        for (let i = 0; i < header.leafCount[0]; i++)
            tokens[i] = dec.tree(models.token, header.anchorBits);
        if (dec.overrun) return { header, levels, levelsPresent: 0 };
    } else {
        const reader = new BitReader(anchorStream, offset, header.splitBits[0]);
        if (!replay(reader, anchorRects, 0, 0, header.width, header.height)
            || reader.overrun
            || anchorRects.count !== header.leafCount[0]) {
            return { header, levels, levelsPresent: 0 };
        }
        offset += (header.splitBits[0] + 7) >> 3;
        for (let i = 0; i < header.leafCount[0]; i++) {
            const bit = i * header.anchorBits;
            let token = 0;
            for (let b = 0; b < header.anchorBits; b++) {
                const at = bit + b;
                if (anchorStream[offset + (at >> 3)] & (1 << (at & 7))) token |= (1 << b);
            }
            tokens[i] = token;
        }
    }

    const anchorRgb = new Uint8Array(anchorRects.count * 3);
    for (let i = 0; i < anchorRects.count; i++) {
        const token = tokens[i] < paletteCount ? tokens[i] : 0;
        anchorRgb[i * 3] = palette[token * 3];
        anchorRgb[i * 3 + 1] = palette[token * 3 + 1];
        anchorRgb[i * 3 + 2] = palette[token * 3 + 2];
    }
    levels.push({ rects: anchorRects, rgb: anchorRgb });
    cursor += header.storedBytes[0];

    /* --- every further level is a bonus the bytes may not have paid for --- */
    for (let k = 1; k < LEVELS; k++) {
        const stream = await readStream(bytes, cursor, header, k);
        if (!stream) break;

        const prev = levels[k - 1];
        const n = header.leafCount[k];
        const step = header.step[k];
        const rects = new RectSet(n);
        const rgb = new Uint8Array(n * 3);

        /* Expanding one previous rectangle may yield several here; they all
         * take the colour it was showing as the base of their delta, and
         * whether it subdivided is the entropy context — derived here, the
         * same way the encoder derived it, never read from the file. */
        const base = new Uint8Array(n * 3);
        const splitCtx = new Uint8Array(n);

        const models = arith ? newModels() : null;
        const dec = arith ? new ArithDecoder(stream, 0, stream.length) : null;
        const reader = arith ? null : new BitReader(stream, 0, header.splitBits[k]);

        let ok = true;
        for (let i = 0; i < prev.rects.count && ok; i++) {
            const before = rects.count;
            ok = arith
                ? replayArith(dec, models, rects, prev.rects.x[i], prev.rects.y[i],
                              prev.rects.w[i], prev.rects.h[i]) && !dec.overrun
                : replay(reader, rects, prev.rects.x[i], prev.rects.y[i],
                         prev.rects.w[i], prev.rects.h[i]) && !reader.overrun;
            const produced = rects.count - before;
            for (let j = before; j < rects.count; j++) {
                for (let c = 0; c < 3; c++) base[j * 3 + c] = prev.rgb[i * 3 + c];
                splitCtx[j] = produced > 1 ? 1 : 0;
            }
        }
        if (!ok || rects.count !== n) break;

        let residual;
        if (arith) {
            residual = new Int8Array(n * 3);
            for (let c = 0; c < 3; c++)
                for (let i = 0; i < n; i++)
                    residual[c * n + i] = dec.residual(models, splitCtx[i], c);
            if (dec.overrun) break;
        } else {
            const residualOffset = (header.splitBits[k] + 7) >> 3;
            residual = new Int8Array(
                stream.buffer, stream.byteOffset + residualOffset, n * 3);
        }

        for (let i = 0; i < n; i++) {
            for (let c = 0; c < 3; c++) {
                rgb[i * 3 + c] = clampByte(
                    base[i * 3 + c] + residual[c * n + i] * step);
            }
        }

        levels.push({ rects, rgb });
        cursor += header.storedBytes[k];
    }

    return { header, levels, levelsPresent: levels.length };
}

/**
 * Paint one level onto a canvas. Writes straight into an ImageData buffer
 * rather than issuing a fillRect per rectangle, which matters once a level
 * runs to tens of thousands of them.
 */
export function renderLevel(level, width, height, ctx) {
    const image = ctx.createImageData(width, height);
    const pixels = image.data;
    const { rects, rgb } = level;

    for (let i = 0; i < rects.count; i++) {
        const r = rgb[i * 3], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
        const x0 = rects.x[i], y0 = rects.y[i];
        const x1 = Math.min(x0 + rects.w[i], width);
        const y1 = Math.min(y0 + rects.h[i], height);
        for (let y = y0; y < y1; y++) {
            let p = (y * width + x0) * 4;
            for (let x = x0; x < x1; x++) {
                pixels[p] = r; pixels[p + 1] = g; pixels[p + 2] = b; pixels[p + 3] = 255;
                p += 4;
            }
        }
    }
    ctx.putImageData(image, 0, 0);
}
