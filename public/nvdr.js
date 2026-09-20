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
const VERSION = 6;
const HEADER_SIZE = 72;
const COMPRESS_NONE = 0;
const COMPRESS_DEFLATE = 1;
const COMPRESS_ARITH = 2;
const ORDER_AREA = 1;

/* --- entropy layer, mirroring nvdr/entropy.c ------------------------- */

const PROB_BITS = 11;
const PROB_INIT = 1 << (PROB_BITS - 1);
const MOVE_BITS = 5;
const TOP_VALUE = 1 << 24;
const AREA_CTX = 16;
const MAG_CTX = 8;
const PREV_CTX = 7;

function newModels() {
    return {
        split: new Uint16Array(AREA_CTX).fill(PROB_INIT),
        token: new Uint16Array(256).fill(PROB_INIT),
        // [split][channel][prevBucket] and [split][channel][prevBucket][magBit]
        sig: grid([2, 3], () => new Uint16Array(PREV_CTX).fill(PROB_INIT)),
        sign: grid([2], () => new Uint16Array(3).fill(PROB_INIT)),
        mag: grid([2, 3, PREV_CTX], () => new Uint16Array(MAG_CTX).fill(PROB_INIT))
    };
}

/* Nested arrays of the given shape, each leaf built by `make`. */
function grid(shape, make) {
    if (shape.length === 0) return make();
    const [head, ...rest] = shape;
    return Array.from({ length: head }, () => grid(rest, make));
}

/* Must match nvdr_prev_context: fine near zero, where the residuals are. */
function prevContext(value) {
    const magnitude = value < 0 ? -value : value;
    if (magnitude === 0) return 0;
    if (magnitude === 1) return 1;
    if (magnitude === 2) return 2;
    if (magnitude <= 4) return 3;
    if (magnitude <= 8) return 4;
    if (magnitude <= 16) return 5;
    return 6;
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

    residual(models, splitCtx, channel, prevCtx) {
        if (!this.bit(models.sig[splitCtx][channel], prevCtx)) return 0;
        const negative = this.bit(models.sign[splitCtx], channel);

        let remaining = 0;
        let i = 0;
        for (; i < MAG_CTX; i++) {
            if (!this.bit(models.mag[splitCtx][channel][prevCtx], i)) break;
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
        order: view.getUint8(14),
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
    if (stored === 0 || offset >= bytes.length) return null;

    // An arithmetic stream decodes as far as its bytes go, so a short read
    // is handed over as-is and the unit loop stops where it runs out. A
    // deflate stream has no such property, so that one is all or nothing.
    let end = offset + stored;
    if (end > bytes.length) {
        if (header.compression === COMPRESS_DEFLATE) return null;
        end = bytes.length;
    }

    const packed = bytes.subarray(offset, end);
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
    // The anchor is the contract: a partial one is no picture at all.
    if (!anchorStream || anchorStream.length < header.storedBytes[0])
        return { header, levels, levelsPresent: 0 };

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

        // The same unit order the encoder used, derived from rectangles
        // already decoded rather than read from the file.
        const order = Array.from({ length: prev.rects.count }, (_, i) => i);
        if (header.order === ORDER_AREA) {
            const area = i => prev.rects.w[i] * prev.rects.h[i];
            order.sort((a, b) => area(b) - area(a) || a - b);
        }

        const capacity = header.leafCount[k] + prev.rects.count + 1;
        const rects = new RectSet(capacity);
        const rgb = new Uint8Array(capacity * 3);

        const models = arith ? newModels() : null;
        const dec = arith ? new ArithDecoder(stream, 0, stream.length) : null;
        const reader = arith ? null : new BitReader(stream, 0, header.splitBits[k]);
        const flatResidual = arith ? null : new Int8Array(
            stream.buffer, stream.byteOffset + ((header.splitBits[k] + 7) >> 3),
            header.leafCount[k] * 3);

        const step = header.step[k];
        let processed = 0, prev0 = 0, stopped = false;

        for (let u = 0; u < order.length; u++) {
            const i = order[u];

            if (stopped) {
                // Past the end of what arrived: this unit keeps the
                // rectangle and colour it had at the level before.
                rects.push(prev.rects.x[i], prev.rects.y[i],
                           prev.rects.w[i], prev.rects.h[i]);
                const j = rects.count - 1;
                for (let c = 0; c < 3; c++) rgb[j * 3 + c] = prev.rgb[i * 3 + c];
                continue;
            }

            const before = rects.count;
            const ok = arith
                ? replayArith(dec, models, rects, prev.rects.x[i], prev.rects.y[i],
                              prev.rects.w[i], prev.rects.h[i]) && !dec.overrun
                : replay(reader, rects, prev.rects.x[i], prev.rects.y[i],
                         prev.rects.w[i], prev.rects.h[i]) && !reader.overrun;
            if (!ok) {
                // This unit did not arrive whole. Roll it back and treat
                // every remaining one as absent.
                rects.count = before;
                stopped = true;
                u--;
                continue;
            }

            const ctx = rects.count - before > 1 ? 1 : 0;
            for (let j = before; j < rects.count; j++) {
                for (let c = 0; c < 3; c++) {
                    let value;
                    if (arith) {
                        const neighbour = c > 0 ? (rgb[j * 3 + c - 1] << 24 >> 24) : prev0;
                        value = dec.residual(models, ctx, c, prevContext(neighbour));
                    } else {
                        value = flatResidual[j * 3 + c];
                    }
                    // Parked as the raw residual until the rectangle is
                    // known to have arrived whole; resolved just below.
                    rgb[j * 3 + c] = value & 0xFF;
                }
                prev0 = rgb[j * 3] << 24 >> 24;
            }

            if (arith && dec.overrun) {
                rects.count = before;
                stopped = true;
                u--;
                continue;
            }

            for (let j = before; j < rects.count; j++)
                for (let c = 0; c < 3; c++)
                    rgb[j * 3 + c] = clampByte(
                        prev.rgb[i * 3 + c] + (rgb[j * 3 + c] << 24 >> 24) * step);
            processed++;
        }

        if (processed === 0) break;

        levels.push({ rects, rgb });
        cursor += header.storedBytes[k];
        if (stopped) break;
    }

    return { header, levels, levelsPresent: levels.length };
}

/*
 * Soften the seams between rectangles.
 *
 * Every pixel is averaged with its four neighbours at `weight` each. Inside
 * a rectangle the neighbours carry the same colour, so the average returns
 * it unchanged and the pass does nothing; only the one-pixel band along a
 * seam moves. That is exactly a boundary blend without needing to know
 * where the boundaries are.
 *
 * Costs no bytes and changes no format — a choice the decoder makes. The
 * variable-width, colour-difference rule that looks like the obvious design
 * was measured first and moved PSNR by 0.03 dB; this moves it by up to
 * 0.77, because a large colour difference is usually a real edge and
 * widening the blend there smears it.
 */
export const SMOOTH_DEFAULT = 0.40;

function smooth(pixels, width, height, weight) {
    if (weight <= 0 || width < 2 || height < 2) return;
    const source = pixels.slice();
    const row = width * 4;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const p = (y * width + x) * 4;
            for (let c = 0; c < 3; c++) {
                let acc = source[p + c], total = 1;
                if (x > 0)          { acc += weight * source[p - 4 + c];   total += weight; }
                if (x < width - 1)  { acc += weight * source[p + 4 + c];   total += weight; }
                if (y > 0)          { acc += weight * source[p - row + c]; total += weight; }
                if (y < height - 1) { acc += weight * source[p + row + c]; total += weight; }
                pixels[p + c] = Math.round(acc / total);
            }
        }
    }
}

/**
 * Paint one level onto a canvas. Writes straight into an ImageData buffer
 * rather than issuing a fillRect per rectangle, which matters once a level
 * runs to tens of thousands of them.
 */
export function renderLevel(level, width, height, ctx, weight = SMOOTH_DEFAULT) {
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
    smooth(pixels, width, height, weight);
    ctx.putImageData(image, 0, 0);
}
