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
import { decode, paintLevel, showRGB, SMOOTH_DEFAULT } from './nvdr.js';

const MAGIC = 0x5644564e;      // "NVDV" read as a little-endian uint32
const VERSION = 2;
const HEADER_SIZE = 24;
const FRAME_HEADER = 12;
const MAX_PIXELS = 1 << 27;    // NVDR_MAX_PIXELS

export const INTRA = 0;
export const PRED = 1;

function clamp255(v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }
function int8(v) { return (v << 24) >> 24; }

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

/* The zlib stream the C side writes with compress2. */
async function inflate(bytes) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
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

/* The reference assembled block by block. Mirrors block_predict. */
function blockPredict(src, dst, width, height, block, vx, vy) {
    const nbx = Math.ceil(width / block);
    for (let y = 0; y < height; y++) {
        const rowBase = Math.floor(y / block) * nbx;
        let o = y * width * 3;
        for (let x = 0; x < width; x++, o += 3) {
            const b = rowBase + Math.floor(x / block);
            let sx = x + vx[b], sy = y + vy[b];
            if (sx < 0) sx = 0; else if (sx >= width) sx = width - 1;
            if (sy < 0) sy = 0; else if (sy >= height) sy = height - 1;
            const s = (sy * width + sx) * 3;
            dst[o] = src[s]; dst[o + 1] = src[s + 1]; dst[o + 2] = src[s + 2];
        }
    }
}

/*
 * Decode one frame's container and paint it flat into `out`, cleared
 * first. Returns false where reconstruct() in nvdrv.c returns -1: the
 * container did not decode, or is not the size of the sequence.
 */
async function reconstruct(bytes, out, width, height) {
    // decode() wants an ArrayBuffer of its own.
    const copy = bytes.slice().buffer;
    const result = await decode(copy);
    if (!result || result.levelsPresent === 0) return false;
    if (result.header.width !== width || result.header.height !== height) return false;
    out.fill(0);
    for (let k = 0; k < result.levelsPresent; k++)
        paintLevel(result.levels[k], width, height, out, 3);
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
        if (!this.info) throw new Error('not an NVDRV v2 file');
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
            const nb = Math.ceil(width / block) * Math.ceil(height / block);
            let raw;
            try {
                raw = await inflate(bytes.subarray(this.pos + 4, this.pos + 4 + fieldLen));
            } catch (e) {
                throw new Error('motion field does not inflate');
            }
            if (raw.length !== nb * 2) throw new Error('motion field has the wrong size');
            const vx = new Int8Array(nb), vy = new Int8Array(nb);
            for (let i = 0; i < nb; i++) {
                // (int8_t)(gdx + (int8_t)raw[i]), wrap included.
                vx[i] = int8(dx + int8(raw[i]));
                vy[i] = int8(dy + int8(raw[nb + i]));
            }
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
            if (!await reconstruct(body, this.state, width, height))
                throw new Error('frame does not decode');
        } else {
            if (!await reconstruct(body, this.error, width, height))
                throw new Error('frame does not decode');
            const err = this.error, st = this.state;
            for (let i = 0; i < st.length; i++)
                st[i] = clamp255(err[i] - 128 + ref[i]);
        }

        this.pos += len;
        const index = this.index++;
        return { index, kind, partial, pixels: this.state };
    }
}

export { showRGB, SMOOTH_DEFAULT };
