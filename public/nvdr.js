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
 * Format (v2), little-endian throughout:
 *
 *   [header 56B]
 *   [level 0 : palette, split bitstream, packed anchor tokens]
 *   [level 1 : split bitstream, 3 int8 residual planes]
 *   [level 2 : split bitstream, 3 int8 residual planes]
 *
 * Geometry is never transmitted as coordinates. Each level carries one bit
 * per visited quadtree node — 1 splits, 0 stops — and the decoder replays
 * the same subdivision the encoder used, starting from the canvas for
 * level 0 and from each previous rectangle for the levels after it.
 */

const MAGIC = 0x5244564e; // "NVDR" read as a little-endian uint32
const VERSION = 2;
const HEADER_SIZE = 56;
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
        step: [],
        leafCount: [],
        splitBits: [],
        streamBytes: []
    };
    for (let k = 0; k < LEVELS; k++) {
        header.step.push(view.getUint8(11 + k));
        header.leafCount.push(view.getUint32(16 + k * 4, true));
        header.splitBits.push(view.getUint32(28 + k * 4, true));
        header.streamBytes.push(view.getUint32(40 + k * 4, true));
    }
    return header;
}

/* How many bytes a decoder needs before each level becomes displayable. */
export function levelThresholds(header) {
    const thresholds = [];
    let total = HEADER_SIZE;
    for (let k = 0; k < LEVELS; k++) {
        total += header.streamBytes[k];
        thresholds.push(total);
    }
    return thresholds;
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
export function decode(buffer) {
    const header = readHeader(buffer);
    if (!header) return null;

    const bytes = new Uint8Array(buffer);
    const levels = [];
    let available = buffer.byteLength - HEADER_SIZE;
    let cursor = HEADER_SIZE;

    /* --- level 0: the contract --- */
    if (available < header.streamBytes[0]) {
        return { header, levels, levelsPresent: 0 };
    }

    let offset = cursor;
    const paletteCount = bytes[offset++];
    const palette = bytes.subarray(offset, offset + paletteCount * 3);
    offset += paletteCount * 3;

    const anchorRects = new RectSet(header.leafCount[0]);
    const anchorReader = new BitReader(bytes, offset, header.splitBits[0]);
    if (!replay(anchorReader, anchorRects, 0, 0, header.width, header.height)
        || anchorReader.overrun
        || anchorRects.count !== header.leafCount[0]) {
        return { header, levels, levelsPresent: 0 };
    }
    offset += (header.splitBits[0] + 7) >> 3;

    const anchorRgb = new Uint8Array(anchorRects.count * 3);
    for (let i = 0; i < anchorRects.count; i++) {
        const bit = i * header.anchorBits;
        let token = 0;
        for (let b = 0; b < header.anchorBits; b++) {
            const at = bit + b;
            if (bytes[offset + (at >> 3)] & (1 << (at & 7))) token |= (1 << b);
        }
        if (token >= paletteCount) token = 0;
        anchorRgb[i * 3] = palette[token * 3];
        anchorRgb[i * 3 + 1] = palette[token * 3 + 1];
        anchorRgb[i * 3 + 2] = palette[token * 3 + 2];
    }
    levels.push({ rects: anchorRects, rgb: anchorRgb });

    cursor += header.streamBytes[0];
    available -= header.streamBytes[0];

    /* --- every further level is a bonus the bytes may not have paid for --- */
    for (let k = 1; k < LEVELS; k++) {
        if (header.streamBytes[k] === 0 || available < header.streamBytes[k]) break;

        const prev = levels[k - 1];
        const rects = new RectSet(header.leafCount[k]);
        const reader = new BitReader(bytes, cursor, header.splitBits[k]);
        const rgb = new Uint8Array(header.leafCount[k] * 3);

        /* Expanding one previous rectangle may yield several here; they all
         * take the colour it was showing as the base of their delta. */
        const residualOffset = cursor + ((header.splitBits[k] + 7) >> 3);
        const residual = new Int8Array(
            buffer, residualOffset, header.leafCount[k] * 3);
        const n = header.leafCount[k];
        const step = header.step[k];

        let ok = true;
        for (let i = 0; i < prev.rects.count && ok; i++) {
            const before = rects.count;
            ok = replay(reader, rects,
                        prev.rects.x[i], prev.rects.y[i],
                        prev.rects.w[i], prev.rects.h[i]) && !reader.overrun;
            for (let j = before; j < rects.count; j++) {
                for (let c = 0; c < 3; c++) {
                    rgb[j * 3 + c] = clampByte(
                        prev.rgb[i * 3 + c] + residual[c * n + j] * step);
                }
            }
        }
        if (!ok || rects.count !== header.leafCount[k]) break;

        levels.push({ rects, rgb });
        cursor += header.streamBytes[k];
        available -= header.streamBytes[k];
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
