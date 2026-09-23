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
const VERSION = 8;
const HEADER_SIZE = 24;
const FRAME_HEADER = 20;
const MAX_PIXELS = 1 << 27;    // NVDR_MAX_PIXELS
const MV_MAX = 384;            // NVDRV_MV_MAX, quarter pixels
const MAX_DPB = 15 + 3;        // NVDRV_MAX_DPB
const MV_ESC_ALBUM = 8, MV_ESC_SEQ = 10;

export const INTRA = 0;
export const PRED = 1;
export const BI = 2;
const MODE_BI = 0, MODE_FWD = 1, MODE_BWD = 2;

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
        gop: view.getUint8(15),
        bframes: view.getUint8(16)
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

function mvComponent(dec, m, c, canBeZero, esc) {
    if (canBeZero && !dec.bit(m.zero, c)) return 0;
    const negative = dec.bit(m.sign, c);
    let remaining = 0, i = 0;
    for (; i < MV_MAG_CTX; i++) {
        if (!dec.bit(m.mag[c], i)) break;
        remaining = i + 1;
    }
    if (i === MV_MAG_CTX) remaining = MV_MAG_CTX + dec.direct(esc);
    return negative ? -(remaining + 1) : remaining + 1;
}

function mvModels() {
    return {
        same: new Uint16Array(3).fill(PROB_INIT),
        zero: new Uint16Array(2).fill(PROB_INIT),
        sign: new Uint16Array(2).fill(PROB_INIT),
        mag: [new Uint16Array(MV_MAG_CTX).fill(PROB_INIT),
              new Uint16Array(MV_MAG_CTX).fill(PROB_INIT)]
    };
}

/* mv_predict(): the model's vector at the block plus the median of how
 * far the left, top and top-right neighbours stray from the model (zero
 * off the frame), the left one alone on the top row. */
function mvPredict(vx, vy, mx, my, nbx, b, out) {
    const x = b % nbx, y = (b - x) / nbx;
    if (y === 0) {
        out[0] = mx[b] + (x > 0 ? vx[b - 1] - mx[b - 1] : 0);
        out[1] = my[b] + (x > 0 ? vy[b - 1] - my[b - 1] : 0);
        return;
    }
    const t = b - nbx;
    const lx = x > 0 ? vx[b - 1] - mx[b - 1] : 0, ly = x > 0 ? vy[b - 1] - my[b - 1] : 0;
    const rx = x + 1 < nbx ? vx[t + 1] - mx[t + 1] : 0, ry = x + 1 < nbx ? vy[t + 1] - my[t + 1] : 0;
    out[0] = mx[b] + median3(lx, vx[t] - mx[t], rx);
    out[1] = my[b] + median3(ly, vy[t] - my[t], ry);
}

const PRED2 = new Int32Array(2);

/* dec_vector(): false for a vector outside -lim-1 .. lim, which is damage. */
function decVector(dec, m, same, vx, vy, mx, my, nbx, b, esc, lim) {
    mvPredict(vx, vy, mx, my, nbx, b, PRED2);
    const x = b % nbx;
    const ctx = (x > 0 && same[b - 1] ? 1 : 0) + (b >= nbx && same[b - nbx] ? 1 : 0);
    same[b] = dec.bit(m.same, ctx) ? 0 : 1;
    let ex = 0, ey = 0;
    if (!same[b]) {
        ex = mvComponent(dec, m, 0, true, esc);
        ey = mvComponent(dec, m, 1, ex !== 0, esc);
    }
    const nx = PRED2[0] + ex, ny = PRED2[1] + ey;
    if (nx < -lim - 1 || nx > lim || ny < -lim - 1 || ny > lim) return false;
    vx[b] = nx; vy[b] = ny;
    return true;
}

function inheritVector(same, vx, vy, mx, my, nbx, b) {
    mvPredict(vx, vy, mx, my, nbx, b, PRED2);
    vx[b] = PRED2[0]; vy[b] = PRED2[1];
    same[b] = 1;
}

/* Returns false where unpack_field returns -1. */
function unpackField(bytes, offset, len, nbx, nby, mx, my, vx, vy, esc, lim) {
    const m = mvModels();
    const nb = nbx * nby;
    const same = new Uint8Array(nb);
    const dec = new ArithDecoder(bytes, offset, len);
    for (let b = 0; b < nb; b++)
        if (!decVector(dec, m, same, vx, vy, mx, my, nbx, b, esc, lim)) return false;
    return true;
}

/* model_get(): an affine model's six int16s. */
const MODEL_BYTES = 12;
const FRAME_MODEL0 = 1, FRAME_MODEL1 = 2;
function modelGet(bytes, at) {
    const v = new DataView(bytes.buffer, bytes.byteOffset + at, MODEL_BYTES);
    return [0, 2, 4, 6, 8, 10].map(o => v.getInt16(o, true));
}

/* model_fill(): the model's vector at every block's centre. */
function modelFill([a0, a1, a2, b0, b1, b2], width, height, block, mx, my) {
    const nbx = Math.ceil(width / block), nby = Math.ceil(height / block);
    const clamp = t => (t < -MV_MAX ? -MV_MAX : t > MV_MAX ? MV_MAX : t);
    for (let by = 0; by < nby; by++)
        for (let bx = 0; bx < nbx; bx++) {
            const cx = bx * block + (block >> 1), cy = by * block + (block >> 1), b = by * nbx + bx;
            // Exact in doubles, and floor() is the C side's 64-bit >> 16.
            mx[b] = clamp(a0 + Math.floor((a1 * cx + a2 * cy + 32768) / 65536));
            my[b] = clamp(b0 + Math.floor((b1 * cx + b2 * cy + 32768) / 65536));
        }
}

function modeCtx(mode, nbx, b, which) {
    const x = b % nbx;
    let n = 0;
    if (x > 0) n += which ? (mode[b - 1] === MODE_BWD ? 1 : 0) : (mode[b - 1] !== MODE_BI ? 1 : 0);
    if (b >= nbx) n += which ? (mode[b - nbx] === MODE_BWD ? 1 : 0) : (mode[b - nbx] !== MODE_BI ? 1 : 0);
    return n;
}

/* unpack_field_bi(): per block the mode, then each used list's vector. */
function unpackFieldBi(bytes, offset, len, nbx, nby, m0x, m0y, m1x, m1y, mode, v0x, v0y, v1x, v1y) {
    const l0 = mvModels(), l1 = mvModels();
    const notBi = new Uint16Array(3).fill(PROB_INIT), bwd = new Uint16Array(3).fill(PROB_INIT);
    const nb = nbx * nby;
    const same0 = new Uint8Array(nb), same1 = new Uint8Array(nb);
    const dec = new ArithDecoder(bytes, offset, len);
    for (let b = 0; b < nb; b++) {
        mode[b] = MODE_BI;
        if (dec.bit(notBi, modeCtx(mode, nbx, b, 0)))
            mode[b] = dec.bit(bwd, modeCtx(mode, nbx, b, 1)) ? MODE_BWD : MODE_FWD;
        if (mode[b] !== MODE_BWD) {
            if (!decVector(dec, l0, same0, v0x, v0y, m0x, m0y, nbx, b, MV_ESC_SEQ, MV_MAX)) return false;
        } else inheritVector(same0, v0x, v0y, m0x, m0y, nbx, b);
        if (mode[b] !== MODE_FWD) {
            if (!decVector(dec, l1, same1, v1x, v1y, m1x, m1y, nbx, b, MV_ESC_SEQ, MV_MAX)) return false;
        } else inheritVector(same1, v1x, v1y, m1x, m1y, nbx, b);
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

/* Blocks whose `mode` is `skip` are left alone (a B frame's blocks that do
 * not read this reference). */
function blockPredict(src, dst, width, height, block, vx, vy, mode = null, skip = -1) {
    const nbx = Math.ceil(width / block), nby = Math.ceil(height / block);
    const stride = width * 3;
    for (let by = 0; by < nby; by++) {
        const y0 = by * block, bh = Math.min(block, height - y0);
        for (let bx = 0; bx < nbx; bx++) {
            const x0 = bx * block, bw = Math.min(block, width - x0);
            const b = by * nbx + bx, fx = vx[b], fy = vy[b];
            if (mode && mode[b] === skip) continue;
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
 * One image predicted from another of the same size, outside a sequence:
 * an album's photo. Mirrors nvdrv_predict_decode(). `ref` is the RGB image
 * before; returns the new RGB image, or null where the C side returns -1.
 */
export function predictDecode(ref, width, height, data) {
    if (data.length < 9) return null;
    const v = new DataView(data.buffer, data.byteOffset, data.length);
    const block = v.getUint8(0);
    if (block < 4 || block > 128) return null;
    const dx = v.getInt16(1, true), dy = v.getInt16(3, true);
    const fieldLen = v.getUint32(5, true);
    if (fieldLen > data.length - 9) return null;
    const nbx = Math.ceil(width / block), nby = Math.ceil(height / block);
    const vx = new Int16Array(nbx * nby), vy = new Int16Array(nbx * nby);
    // An album's field is predicted from one translation and has no model.
    const tx = new Int16Array(nbx * nby).fill(dx * 4), ty = new Int16Array(nbx * nby).fill(dy * 4);
    if (!unpackField(data, 9, fieldLen, nbx, nby, tx, ty, vx, vy, MV_ESC_ALBUM, 127)) return null;
    const pred = new Uint8Array(width * height * 3), err = new Uint8Array(width * height * 3);
    blockPredict(ref, pred, width, height, block, vx, vy);
    if (!reconstruct(data.subarray(9 + fieldLen), err, width, height)) return null;
    for (let i = 0; i < pred.length; i++) pred[i] = clamp255(err[i] - 128 + pred[i]);
    return pred;
}

/*
 * find_refs(): among the held frames, the nearest before `display` and
 * the nearest after it, as indices, -1 where there is none.
 */
function findRefs(dpb, display) {
    let before = -1, after = -1;
    for (let i = 0; i < dpb.length; i++) {
        const d = dpb[i].display;
        if (d < display && (before < 0 || d > dpb[before].display)) before = i;
        if (d > display && (after < 0 || d < dpb[after].display)) after = i;
    }
    return [before, after];
}

/**
 * Plays a sequence frame by frame, in display order. `next()` resolves to
 *   { index, kind, partial, pixels }   — a frame; `index` is its display
 *                                        number, `pixels` its RGB
 *   null                               — the stream ended
 * and throws on a container that does not make sense, where the C decoder
 * returns -1. Frames are stored in coding order (an anchor, then the B
 * frames before it), so a frame is shown once every frame before it has
 * been; mirrors nvdrv_decode_next().
 */
export class SequenceDecoder {
    constructor(buffer) {
        this.info = readSequenceHeader(buffer);
        if (!this.info) throw new Error('not an NVDRV v8 file');
        this.bytes = new Uint8Array(buffer);
        this.pos = HEADER_SIZE;
        const n = this.info.width * this.info.height * 3;
        this.pred = new Uint8Array(n);
        this.pred1 = new Uint8Array(n);
        this.error = new Uint8Array(n);
        this.dpb = [];
        this.nextOut = 0;
        this.ended = false;
    }

    async next() {
        for (;;) {
            const i = this.dpb.findIndex(s => s.display === this.nextOut);
            if (i >= 0) {
                const s = this.dpb[i];
                const shown = this.nextOut++;
                // Every frame still to come predicts from this one or later.
                this.dpb = this.dpb.filter(x => x.display >= shown);
                return { index: s.display, kind: s.kind, partial: s.partial, pixels: s.pixels };
            }
            if (!this.ended) {
                if (!this.decodeOne()) this.ended = true;
                continue;
            }
            // A cut file ended with frames missing: skip to the next one
            // that did arrive.
            let next = -1;
            for (const s of this.dpb) if (s.display > this.nextOut && (next < 0 || s.display < next)) next = s.display;
            if (next < 0) return null;
            this.nextOut = next;
        }
    }

    /* decode_one(): true when a frame was decoded, false at the end of the
     * stream; throws on damage. */
    decodeOne() {
        const { width, height } = this.info;
        const bytes = this.bytes, size = bytes.length;
        if (this.pos + FRAME_HEADER > size) return false;

        const view = new DataView(bytes.buffer, bytes.byteOffset + this.pos, FRAME_HEADER);
        const kind = view.getUint8(0);
        const block = view.getUint8(1);
        const dx = view.getInt16(2, true);
        const dy = view.getInt16(4, true);
        let len = view.getUint32(6, true);
        const dx1 = view.getInt16(10, true);
        const dy1 = view.getInt16(12, true);
        const display = view.getUint32(14, true);
        const flags = view.getUint8(18);
        if (kind !== INTRA && kind !== PRED && kind !== BI) throw new Error('bad frame type');
        if (kind === INTRA && block) throw new Error('intra frame with a motion field');
        if ((flags & ~(FRAME_MODEL0 | FRAME_MODEL1)) || view.getUint8(19)) throw new Error('bad frame flags');
        if (flags && (!block || (kind !== BI && (flags & FRAME_MODEL1)))) throw new Error('bad frame flags');
        if (block && (block < 4 || block > 128)) throw new Error('bad block size');
        if (kind === BI && !block) throw new Error('B frame without a motion field');
        if (display > 0x3fffffff) throw new Error('bad display number');
        if (display < this.nextOut || this.dpb.length >= MAX_DPB ||
            this.dpb.some(s => s.display === display)) throw new Error('frame out of order');
        const [before, after] = findRefs(this.dpb, display);
        if (kind !== INTRA && before < 0) throw new Error('predicted frame without a reference');
        if (kind === BI && after < 0) throw new Error('B frame without a reference after it');
        this.pos += FRAME_HEADER;

        let ref = kind === INTRA ? null : this.dpb[before].pixels;
        if (block) {
            // The field has to arrive whole: without it there is no
            // reference to add the residual to, so a cut inside it ends
            // the stream.
            if (this.pos + 4 > size || len < 4) return false;
            const fieldLen = new DataView(bytes.buffer, bytes.byteOffset + this.pos, 4)
                .getUint32(0, true);
            if (fieldLen > len - 4 || this.pos + 4 + fieldLen > size) return false;
            const nbx = Math.ceil(width / block), nby = Math.ceil(height / block), nb = nbx * nby;
            const v0x = new Int16Array(nb), v0y = new Int16Array(nb);
            const m0x = new Int16Array(nb), m0y = new Int16Array(nb);
            // The field starts with the affine model of each flagged list;
            // the others are their global vector. Then the vectors, in
            // quarter pixels.
            const fp = this.pos + 4;
            const head = MODEL_BYTES * ((flags & FRAME_MODEL0 ? 1 : 0) + (flags & FRAME_MODEL1 ? 1 : 0));
            if (fieldLen < head) throw new Error('motion field is damaged');
            let mp = fp;
            const model0 = flags & FRAME_MODEL0 ? modelGet(bytes, mp) : [dx * 4, 0, 0, dy * 4, 0, 0];
            if (flags & FRAME_MODEL0) mp += MODEL_BYTES;
            const model1 = flags & FRAME_MODEL1 ? modelGet(bytes, mp) : [dx1 * 4, 0, 0, dy1 * 4, 0, 0];
            modelFill(model0, width, height, block, m0x, m0y);
            if (kind === PRED) {
                if (!unpackField(bytes, fp + head, fieldLen - head, nbx, nby, m0x, m0y, v0x, v0y,
                                 MV_ESC_SEQ, MV_MAX))
                    throw new Error('motion field is damaged');
                blockPredict(ref, this.pred, width, height, block, v0x, v0y);
            } else {
                const v1x = new Int16Array(nb), v1y = new Int16Array(nb), mode = new Uint8Array(nb);
                const m1x = new Int16Array(nb), m1y = new Int16Array(nb);
                modelFill(model1, width, height, block, m1x, m1y);
                if (!unpackFieldBi(bytes, fp + head, fieldLen - head, nbx, nby, m0x, m0y, m1x, m1y,
                                   mode, v0x, v0y, v1x, v1y))
                    throw new Error('motion field is damaged');
                const P0 = this.pred, P1 = this.pred1;
                blockPredict(ref, P0, width, height, block, v0x, v0y, mode, MODE_BWD);
                blockPredict(this.dpb[after].pixels, P1, width, height, block, v1x, v1y, mode, MODE_FWD);
                // Backward blocks take the second prediction, the mean
                // blocks the rounded mean of both; forward ones are in P0.
                const stride = width * 3;
                for (let b = 0; b < nb; b++) {
                    if (mode[b] === MODE_FWD) continue;
                    const x0 = (b % nbx) * block, y0 = Math.floor(b / nbx) * block;
                    const n = Math.min(block, width - x0) * 3, h = Math.min(block, height - y0);
                    for (let j = 0; j < h; j++) {
                        const o = (y0 + j) * stride + x0 * 3;
                        if (mode[b] === MODE_BWD) P0.set(P1.subarray(o, o + n), o);
                        else for (let k = o; k < o + n; k++) P0[k] = (P0[k] + P1[k] + 1) >> 1;
                    }
                }
            }
            ref = this.pred;
            this.pos += 4 + fieldLen;
            len -= 4 + fieldLen;
        } else if (kind === PRED && (dx || dy)) {
            shiftInto(ref, this.pred, width, height, dx, dy);
            ref = this.pred;
        }

        // A cut file ends mid-frame. The still decoder reads as far as the
        // bytes reach, so the last frame shows at the quality that arrived.
        const have = size - this.pos;
        let partial = false;
        if (len > have) { len = have; partial = true; }
        if (len === 0) return false;

        const body = bytes.subarray(this.pos, this.pos + len);
        const pixels = new Uint8Array(width * height * 3);
        // A frame cut before its colour layer ends the stream.
        if (!reconstruct(body, kind === INTRA ? pixels : this.error, width, height)) {
            if (partial) return false;
            throw new Error('frame does not decode');
        }
        if (kind !== INTRA) {
            const err = this.error;
            for (let i = 0; i < pixels.length; i++) pixels[i] = clamp255(err[i] - 128 + ref[i]);
        }
        this.dpb.push({ display, kind, partial, pixels });
        this.pos += len;
        return true;
    }
}

export { showRGB };
