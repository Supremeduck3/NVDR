/*
 * NVDRV in the browser — the sequence decoder, mirroring src/nvdrv.c.
 *
 * Every frame is an ordinary NVDR container, decoded by nvdr.js exactly as
 * a still is. What this adds is the loop around it: the reference each
 * predicted frame is added to, built from the previous frame by a global
 * translation or a field of per-block vectors, and the state carried from
 * one frame to the next.
 *
 * That state has to match the C decoder's byte for byte, not just look
 * right. A predicted frame is a residual on top of it, so one pixel off at
 * frame 1 is a different reference at frame 2 and the error compounds
 * down the clip. scripts/crosscheck_seq.mjs checks every frame.
 */
import { decode, showRGB, ArithDecoder, PROB_INIT } from './nvdr.js';

const MAGIC = 0x5644564e;      // "NVDV" read as a little-endian uint32
const VERSION = 5;
const HEADER_SIZE = 24;
const FRAME_HEADER = 12;
const MAX_PIXELS = 1 << 27;    // NVDR_MAX_PIXELS

export const INTRA = 0;
export const PRED = 1;

function clamp255(v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

export function readSequenceHeader(buffer) {
    if (buffer.byteLength < HEADER_SIZE) return null;
    const view = new DataView(buffer);
    if (view.getUint32(0, true) !== MAGIC || view.getUint8(4) !== VERSION) return null;
    const info = {
        width: view.getUint16(6, true),
        height: view.getUint16(8, true),
        frameCount: view.getUint32(10, true),
        fps: view.getUint8(14),
        gop: view.getUint8(15)
    };
    // Every frame buffer is sized from this before a frame is read.
    if (!info.width || !info.height || info.width * info.height > MAX_PIXELS) return null;
    return info;
}

/*
 * The motion field, mirroring pack_field/unpack_field in nvdrv.c: each
 * vector predicted by the median of its left, top and top-right
 * neighbours, then a match bit or the difference, arithmetic coded.
 */
const MV_MAG_CTX = 6;

function median3(a, b, c) {
    if (a > b) { const t = a; a = b; b = t; }
    return c < a ? a : (c > b ? b : c);
}

function mvComponent(dec, m, c, canBeZero) {
    if (canBeZero && !dec.bit(m.zero, c)) return 0;
    const negative = dec.bit(m.sign, c);
    let remaining = 0, i = 0;
    for (; i < MV_MAG_CTX; i++) {
        if (!dec.bit(m.mag[c], i)) break;
        remaining = i + 1;
    }
    if (i === MV_MAG_CTX) remaining = MV_MAG_CTX + dec.direct(8);
    return negative ? -(remaining + 1) : remaining + 1;
}

/* Returns false where unpack_field returns -1: a vector outside int8. */
function unpackField(bytes, offset, len, nbx, nby, gdx, gdy, vx, vy) {
    const m = {
        same: new Uint16Array(3).fill(PROB_INIT),
        zero: new Uint16Array(2).fill(PROB_INIT),
        sign: new Uint16Array(2).fill(PROB_INIT),
        mag: [new Uint16Array(MV_MAG_CTX).fill(PROB_INIT),
              new Uint16Array(MV_MAG_CTX).fill(PROB_INIT)]
    };
    const nb = nbx * nby;
    const same = new Uint8Array(nb);
    const dec = new ArithDecoder(bytes, offset, len);
    for (let b = 0; b < nb; b++) {
        const x = b % nbx, y = (b - x) / nbx;
        let px, py;
        if (y === 0) {
            px = x > 0 ? vx[b - 1] : gdx;
            py = x > 0 ? vy[b - 1] : gdy;
        } else {
            const t = b - nbx;
            const lx = x > 0 ? vx[b - 1] : gdx, ly = x > 0 ? vy[b - 1] : gdy;
            const rx = x + 1 < nbx ? vx[t + 1] : gdx, ry = x + 1 < nbx ? vy[t + 1] : gdy;
            px = median3(lx, vx[t], rx);
            py = median3(ly, vy[t], ry);
        }
        const ctx = (x > 0 && same[b - 1] ? 1 : 0) + (y > 0 && same[b - nbx] ? 1 : 0);
        same[b] = dec.bit(m.same, ctx) ? 0 : 1;
        let ex = 0, ey = 0;
        if (!same[b]) {
            ex = mvComponent(dec, m, 0, true);
            ey = mvComponent(dec, m, 1, ex !== 0);
        }
        const nx = px + ex, ny = py + ey;
        if (nx < -128 || nx > 127 || ny < -128 || ny > 127) return false;
        vx[b] = nx; vy[b] = ny;
    }
    return true;
}

/* The reference translated by one vector, edges held. Mirrors shift_into. */
function shiftInto(src, dst, width, height, dx, dy) {
    for (let y = 0; y < height; y++) {
        let sy = y + dy;
        if (sy < 0) sy = 0; else if (sy >= height) sy = height - 1;
        const srow = sy * width * 3;
        let o = y * width * 3;
        for (let x = 0; x < width; x++, o += 3) {
            let sx = x + dx;
            if (sx < 0) sx = 0; else if (sx >= width) sx = width - 1;
            const s = srow + sx * 3;
            dst[o] = src[s]; dst[o + 1] = src[s + 1]; dst[o + 2] = src[s + 2];
        }
    }
}

/*
 * The reference assembled block by block, each block from its own
 * quarter-pixel vector, bilinear between whole pixels, edges held.
 * Mirrors qsample() and block_predict() exactly: weights (4 - a) and a on
 * each axis, rounded with + 8 >> 4.
 */
function blockPredict(src, dst, width, height, block, vx, vy) {
    // Block by block: a whole-pixel vector is a copy, and a block whose
    // source rectangle lies inside the frame needs no edge clamping. Every
    // pixel still comes out exactly as qsample() makes it.
    const nbx = Math.ceil(width / block), nby = Math.ceil(height / block);
    const stride = width * 3;
    for (let by = 0; by < nby; by++) {
        const y0 = by * block, y1 = Math.min(y0 + block, height);
        for (let bx = 0; bx < nbx; bx++) {
            const x0 = bx * block, x1 = Math.min(x0 + block, width);
            const b = by * nbx + bx, fx = vx[b], fy = vy[b];
            const ox = fx >> 2, oy = fy >> 2, ax = fx & 3, ay = fy & 3;
            const inside = x0 + ox >= 0 && x1 + ox < width && y0 + oy >= 0 && y1 + oy < height;
            if (inside && !ax && !ay) {
                for (let y = y0; y < y1; y++) {
                    const s0 = (y + oy) * stride + (x0 + ox) * 3;
                    dst.set(src.subarray(s0, s0 + (x1 - x0) * 3), y * stride + x0 * 3);
                }
                continue;
            }
            const w00 = (4 - ax) * (4 - ay), w01 = ax * (4 - ay), w10 = (4 - ax) * ay, w11 = ax * ay;
            for (let y = y0; y < y1; y++) {
                let iy = y + oy, iy1 = iy + 1;
                if (!inside) {
                    iy = iy < 0 ? 0 : iy >= height ? height - 1 : iy;
                    iy1 = iy1 < 0 ? 0 : iy1 >= height ? height - 1 : iy1;
                }
                const r0 = iy * stride, r1 = iy1 * stride;
                let o = y * stride + x0 * 3;
                for (let x = x0; x < x1; x++, o += 3) {
                    let ix = x + ox, ix1 = ix + 1;
                    if (!inside) {
                        ix = ix < 0 ? 0 : ix >= width ? width - 1 : ix;
                        ix1 = ix1 < 0 ? 0 : ix1 >= width ? width - 1 : ix1;
                    }
                    const a0 = r0 + ix * 3, a1 = r0 + ix1 * 3, b0 = r1 + ix * 3, b1 = r1 + ix1 * 3;
                    dst[o] = (w00 * src[a0] + w01 * src[a1] + w10 * src[b0] + w11 * src[b1] + 8) >> 4;
                    dst[o + 1] = (w00 * src[a0 + 1] + w01 * src[a1 + 1] + w10 * src[b0 + 1] + w11 * src[b1 + 1] + 8) >> 4;
                    dst[o + 2] = (w00 * src[a0 + 2] + w01 * src[a1 + 2] + w10 * src[b0 + 2] + w11 * src[b1 + 2] + 8) >> 4;
                }
            }
        }
    }
}

/*
 * Decode one frame's container at full quality into `out`. Returns false
 * where reconstruct() in nvdrv.c returns -1: the container did not decode,
 * or is not the size of the sequence.
 */
function reconstruct(bytes, out, width, height) {
    const result = decode(bytes);
    if (!result) return false;
    if (result.header.width !== width || result.header.height !== height) return false;
    out.set(result.rgb);
    return true;
}

/**
 * Plays a sequence frame by frame. `next()` resolves to
 *   { kind, partial, pixels }   — a frame; `pixels` is the decoder state
 *   null                        — the stream ended
 * and throws on a container that does not make sense, where the C decoder
 * returns -1.
 *
 * `pixels` is the live state, overwritten by the next frame: copy it if
 * it has to outlive the call.
 */
export class SequenceDecoder {
    constructor(buffer) {
        this.info = readSequenceHeader(buffer);
        if (!this.info) throw new Error('not an NVDRV v5 file');
        this.bytes = new Uint8Array(buffer);
        this.pos = HEADER_SIZE;
        const n = this.info.width * this.info.height * 3;
        this.state = new Uint8Array(n);
        this.scratch = new Uint8Array(n);
        this.error = new Uint8Array(n);
        this.index = 0;
    }

    async next() {
        const { width, height } = this.info;
        const bytes = this.bytes, size = bytes.length;
        if (this.pos + FRAME_HEADER > size) return null;

        const view = new DataView(bytes.buffer, bytes.byteOffset + this.pos, FRAME_HEADER);
        const kind = view.getUint8(0);
        const block = view.getUint8(1);
        const dx = view.getInt16(2, true);
        const dy = view.getInt16(4, true);
        let len = view.getUint32(6, true);
        if (kind !== INTRA && kind !== PRED) throw new Error('bad frame type');
        if (kind === INTRA && block) throw new Error('intra frame with a motion field');
        this.pos += FRAME_HEADER;

        let ref = this.state;
        if (kind === PRED && block) {
            // The field has to arrive whole: without it there is no
            // reference to add the residual to, so a cut inside it ends
            // the stream.
            if (this.pos + 4 > size || len < 4) return null;
            const fieldLen = new DataView(bytes.buffer, bytes.byteOffset + this.pos, 4)
                .getUint32(0, true);
            if (fieldLen > len - 4 || this.pos + 4 + fieldLen > size) return null;
            const nbx = Math.ceil(width / block), nby = Math.ceil(height / block);
            const vx = new Int8Array(nbx * nby), vy = new Int8Array(nbx * nby);
            // The field is in quarter pixels, the global vector in whole ones.
            if (!unpackField(bytes, this.pos + 4, fieldLen, nbx, nby, dx * 4, dy * 4, vx, vy))
                throw new Error('motion field is damaged');
            blockPredict(this.state, this.scratch, width, height, block, vx, vy);
            ref = this.scratch;
            this.pos += 4 + fieldLen;
            len -= 4 + fieldLen;
        } else if (kind === PRED && (dx || dy)) {
            shiftInto(this.state, this.scratch, width, height, dx, dy);
            ref = this.scratch;
        }

        // A cut file ends mid-frame. The still decoder reads as far as the
        // bytes reach, so the last frame shows at the quality that arrived.
        const have = size - this.pos;
        let partial = false;
        if (len > have) { len = have; partial = true; }
        if (len === 0) return null;

        const body = bytes.subarray(this.pos, this.pos + len);
        if (kind === INTRA) {
            // A frame cut before its colour layer ends the stream.
            if (!reconstruct(body, this.state, width, height)) {
                if (partial) return null;
                throw new Error('frame does not decode');
            }
        } else {
            if (!reconstruct(body, this.error, width, height)) {
                if (partial) return null;
                throw new Error('frame does not decode');
            }
            const err = this.error, st = this.state;
            for (let i = 0; i < st.length; i++)
                st[i] = clamp255(err[i] - 128 + ref[i]);
        }

        this.pos += len;
        const index = this.index++;
        return { index, kind, partial, pixels: this.state };
    }
}

export { showRGB };
