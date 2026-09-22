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
const VERSION = 6;
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
 * quarter-pixel vector. Mirrors qsample() and block_predict() in nvdrv.c
 * exactly: H.264's 6-tap half pixels (B across, H down, J the centre from
 * the unrounded horizontal sums) and quarter pixels as the rounded mean of
 * the two nearest samples, all over the reference with its edges held.
 *
 * The C side builds the half-pixel planes once per frame. Here each block
 * computes only the samples its vector's phase needs, over its own
 * region, which comes to the same numbers for a fraction of the work: a
 * whole-pixel vector is a copy, and most phases need one plane of three.
 */
// Which samples each phase ((fy & 3) << 2 | (fx & 3)) reads.
const USE_F = [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
const USE_B = [0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1];
const USE_H = [0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1];
const USE_J = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0];
const MAXB = 128 + 1;
const PF = new Uint8Array(MAXB * MAXB * 3), PB = new Uint8Array(MAXB * MAXB * 3);
const PH = new Uint8Array(MAXB * MAXB * 3), PJ = new Uint8Array(MAXB * MAXB * 3);
const B1 = new Int32Array((MAXB + 5) * MAXB * 3);
const XS = new Int32Array(MAXB + 8), YS = new Int32Array(MAXB + 8);
const SG = new Int32Array((MAXB + 6) * (MAXB + 6) * 3);

const clip8 = v => (v < 0 ? 0 : v > 255 ? 255 : v);

function blockPredict(src, dst, width, height, block, vx, vy) {
    const nbx = Math.ceil(width / block), nby = Math.ceil(height / block);
    const stride = width * 3;
    for (let by = 0; by < nby; by++) {
        const y0 = by * block, bh = Math.min(block, height - y0);
        for (let bx = 0; bx < nbx; bx++) {
            const x0 = bx * block, bw = Math.min(block, width - x0);
            const b = by * nbx + bx, fx = vx[b], fy = vy[b];
            const X0 = x0 + (fx >> 2), Y0 = y0 + (fy >> 2);
            const phase = ((fy & 3) << 2) | (fx & 3);
            // Clamped source columns X0-2 .. X0+bw+3 and rows Y0-2 .. Y0+bh+3.
            for (let i = 0; i < bw + 6; i++) {
                const X = X0 - 2 + i;
                XS[i] = (X < 0 ? 0 : X >= width ? width - 1 : X) * 3;
            }
            for (let j = 0; j < bh + 6; j++) {
                const Y = Y0 - 2 + j;
                YS[j] = (Y < 0 ? 0 : Y >= height ? height - 1 : Y) * stride;
            }
            const ls = bw + 1;                    // local stride, in samples
            if (phase === 0) {
                for (let j = 0; j < bh; j++) {
                    const row = YS[j + 2];
                    let o = (y0 + j) * stride + x0 * 3;
                    for (let i = 0; i < bw; i++, o += 3) {
                        const s0 = row + XS[i + 2];
                        dst[o] = src[s0]; dst[o + 1] = src[s0 + 1]; dst[o + 2] = src[s0 + 2];
                    }
                }
                continue;
            }
            // The source region the filters reach, gathered once, edges
            // held: rows Y0-2 .. Y0+bh+3, columns X0-2 .. X0+bw+3.
            const gs = (bw + 6) * 3;
            for (let j = 0; j < bh + 6; j++) {
                const row = YS[j];
                let g = j * gs;
                for (let i = 0; i < bw + 6; i++, g += 3) {
                    const s0 = row + XS[i];
                    SG[g] = src[s0]; SG[g + 1] = src[s0 + 1]; SG[g + 2] = src[s0 + 2];
                }
            }
            const n3 = (bw + 1) * 3, ls3 = ls * 3;
            if (USE_F[phase])
                for (let j = 0; j <= bh; j++) {
                    const g = (j + 2) * gs + 6, l = j * ls3;
                    for (let k = 0; k < n3; k++) PF[l + k] = SG[g + k];
                }
            if (USE_J[phase] || USE_B[phase]) {
                // Unrounded horizontal sums; all bh+6 rows only for J.
                const j0 = USE_J[phase] ? 0 : 2, j1 = USE_J[phase] ? bh + 6 : bh + 3;
                for (let j = j0; j < j1; j++) {
                    const g = j * gs, l = j * ls3;
                    for (let k = 0; k < n3; k++) {
                        const q = g + k;
                        B1[l + k] = SG[q] - 5 * SG[q + 3] + 20 * SG[q + 6] + 20 * SG[q + 9] -
                                    5 * SG[q + 12] + SG[q + 15];
                    }
                }
                if (USE_B[phase])
                    for (let j = 0; j <= bh; j++) {
                        const l = j * ls3, m = (j + 2) * ls3;
                        for (let k = 0; k < n3; k++) PB[l + k] = clip8((B1[m + k] + 16) >> 5);
                    }
                if (USE_J[phase])
                    for (let j = 0; j <= bh; j++) {
                        const l = j * ls3, m = (j + 2) * ls3;
                        for (let k = 0; k < n3; k++) {
                            const q = m + k;
                            PJ[l + k] = clip8((B1[q - 2 * ls3] - 5 * B1[q - ls3] + 20 * B1[q] +
                                20 * B1[q + ls3] - 5 * B1[q + 2 * ls3] + B1[q + 3 * ls3] + 512) >> 10);
                        }
                    }
            }
            if (USE_H[phase])
                for (let j = 0; j <= bh; j++) {
                    const g = j * gs + 6, l = j * ls3;
                    for (let k = 0; k < n3; k++) {
                        const q = g + k;
                        PH[l + k] = clip8((SG[q] - 5 * SG[q + gs] + 20 * SG[q + 2 * gs] +
                            20 * SG[q + 3 * gs] - 5 * SG[q + 4 * gs] + SG[q + 5 * gs] + 16) >> 5);
                    }
                }
            // Each phase is one sample or the mean of two: pick the planes
            // and offsets once per block, not per pixel.
            const R = 3, D = ls * 3;
            let A, oa, Bp = null, ob = 0;
            switch (phase) {
            case 1:  A = PF; oa = 0; Bp = PB; ob = 0; break;
            case 2:  A = PB; oa = 0; break;
            case 3:  A = PB; oa = 0; Bp = PF; ob = R; break;
            case 4:  A = PF; oa = 0; Bp = PH; ob = 0; break;
            case 5:  A = PB; oa = 0; Bp = PH; ob = 0; break;
            case 6:  A = PB; oa = 0; Bp = PJ; ob = 0; break;
            case 7:  A = PB; oa = 0; Bp = PH; ob = R; break;
            case 8:  A = PH; oa = 0; break;
            case 9:  A = PH; oa = 0; Bp = PJ; ob = 0; break;
            case 10: A = PJ; oa = 0; break;
            case 11: A = PJ; oa = 0; Bp = PH; ob = R; break;
            case 12: A = PH; oa = 0; Bp = PF; ob = D; break;
            case 13: A = PH; oa = 0; Bp = PB; ob = D; break;
            case 14: A = PJ; oa = 0; Bp = PB; ob = D; break;
            default: A = PH; oa = R; Bp = PB; ob = D;
            }
            for (let j = 0; j < bh; j++) {
                const o = (y0 + j) * stride + x0 * 3, l = j * ls * 3;
                const n = bw * 3;
                if (Bp) for (let k = 0; k < n; k++) dst[o + k] = (A[l + k + oa] + Bp[l + k + ob] + 1) >> 1;
                else for (let k = 0; k < n; k++) dst[o + k] = A[l + k + oa];
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
        if (!this.info) throw new Error('not an NVDRV v6 file');
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
