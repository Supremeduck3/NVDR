/*
 * NVDRV in the browser: a sequence, decoded by src/nvdrv.c compiled to
 * WebAssembly (nvdr-wasm.js), the code the command-line decoder runs, so
 * every frame is the C decoder's to the bit.
 *
 * What stays here is reading the file without decoding it: the header,
 * where each frame starts and ends (for the player's cut), and the sound
 * track. scripts/crosscheck_seq.mjs holds the decoder to the C one.
 */
import { wasmSequence, wasmPredictDecode } from './nvdr-wasm.js';
import { showRGB } from './nvdr.js';

const MAGIC = 0x5644564e;      // "NVDV" read as a little-endian uint32
const VERSION = 22;
const HEADER_SIZE = 24;
const MAX_PIXELS = 1 << 27;    // NVDR_MAX_PIXELS

export const INTRA = 0;
export const PRED = 1;
export const BI = 2;
export const PRED2 = 3;   // a P frame from two anchors (MORE REFERENCES in nvdrv.c)

export function readSequenceHeader(buffer) {
    if (buffer.byteLength < HEADER_SIZE) return null;
    const view = new DataView(buffer);
    if (view.getUint32(0, true) !== MAGIC || view.getUint8(4) < 20 || view.getUint8(4) > VERSION) return null;
    const info = {
        version: view.getUint8(4),
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

/* Mirrors COMPACT FRAMES in nvdrv.c: get_varint(), unzigzag() and
 * expand_container(). A varint is { v, n }: n bytes used, 0 when the bytes
 * ran out first, -1 when longer than five. */
function getVarint(bytes, pos, end) {
    let r = 0;
    for (let i = 0; i < 5; i++) {
        if (pos + i >= end) return { v: 0, n: 0 };
        const b = bytes[pos + i];
        r += (b & 0x7f) * 2 ** (7 * i);
        if (!(b & 0x80)) return r > 0xffffffff ? { v: 0, n: -1 } : { v: r, n: i + 1 };
    }
    return { v: 0, n: -1 };
}
const unzigzag = u => (u % 2 ? -((u - 1) / 2) - 1 : u / 2);
// The HDR_ bits of a frame header: what differs from the last frame of
// its class (COMPACT FRAMES in nvdrv.c).
const HDR_BLOCK = 1, HDR_GLOBAL = 2, HDR_FLAGS = 4, HDR_BLOCKS = 8, HDR_QL = 16, HDR_QC = 32, HDR_BAND = 64;

/*
 * Where each frame is, without decoding any: { kind, display, start, end }
 * in file order, read from the compact frame headers as decode_one() in nvdrv.c reads
 * them. A frame cut short ends at the end of the bytes; the walk stops at
 * a header that did not arrive whole or makes no sense.
 */
/*
 * A sequence's sound track (SOUND in nvdrv.h): { mime, bytes }, a file
 * the browser's <audio> plays, carried whole after the last frame. Null
 * when there is none, or the file was cut before its end.
 */
const SOUND_MARK = 0xff;
export function soundTrack(buffer) {
    const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    const frames = frameIndex(bytes);
    let at = frames.length ? frames[frames.length - 1].end : HEADER_SIZE;
    if (at + 6 > bytes.length || bytes[at] !== SOUND_MARK ||
        String.fromCharCode(...bytes.subarray(at + 1, at + 5)) !== 'NVAU') return null;
    const n = bytes[at + 5];
    at += 6;
    if (at + n + 4 > bytes.length) return null;
    const mime = String.fromCharCode(...bytes.subarray(at, at + n));
    at += n;
    const len = (bytes[at] | bytes[at + 1] << 8 | bytes[at + 2] << 16 | bytes[at + 3] << 24) >>> 0;
    at += 4;
    if (!/^audio\/[a-z0-9.+-]+$/.test(mime) || at + len > bytes.length) return null;
    return { mime, bytes: bytes.subarray(at, at + len) };
}

export function frameIndex(buffer) {
    const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer), size = bytes.length;
    const out = [];
    let pos = HEADER_SIZE, lastDisplay = 0;
    while (pos < size) {
        let at = pos;
        const a = bytes[at++], kind = a & 3, changed = (a >> 6) & 1;
        if (a >> 7) break;
        let r = getVarint(bytes, at, size);
        if (r.n <= 0) break;
        at += r.n;
        const display = lastDisplay + unzigzag(r.v);
        lastDisplay = display;
        let which = 0;
        if (changed) { if (at >= size) break; which = bytes[at++]; }
        if (which & HDR_BLOCK) at++;
        if (which & HDR_GLOBAL)
            for (let k = 0; k < (kind === BI || kind === PRED2 ? 4 : 2); k++) {
                r = getVarint(bytes, at, size);
                if (r.n <= 0) return out;
                at += r.n;
            }
        if (which & HDR_FLAGS) at++;
        if (which & HDR_BLOCKS) at += 2;
        for (const bit of [HDR_QL, HDR_QC]) {
            if (!(which & bit)) continue;
            r = getVarint(bytes, at, size);
            if (r.n <= 0) return out;
            at += r.n;
        }
        if (which & HDR_BAND) at += 2;
        r = getVarint(bytes, at, size);
        if (r.n <= 0) break;
        at += r.n;
        out.push({ kind, display, start: pos, end: at + r.v });
        pos = at + r.v;
    }
    return out;
}

/**
 * Plays a sequence frame by frame, in display order. `next()` resolves to
 *   { index, kind, partial, pixels }   — a frame; `index` is its display
 *                                        number, `pixels` its RGB
 *   null                               — the stream ended
 * and throws on a container that does not make sense, where the C decoder
 * returns -1. `buffer` is the file or any prefix of it: a frame cut short
 * shows at the quality its bytes paid for, frames never delivered are
 * skipped. close() gives the decoder's memory back; it also goes when
 * the decoder is collected.
 */
const sequenceRegistry = new FinalizationRegistry(seq => seq.close());

export class SequenceDecoder {
    constructor(buffer) {
        this.info = readSequenceHeader(buffer);
        if (!this.info) throw new Error('not an NVDRV file of format 20 to 22');
        this.seq = wasmSequence(new Uint8Array(buffer));
        if (!this.seq) throw new Error('not an NVDRV file of format 20 to 22');
        sequenceRegistry.register(this, this.seq, this);
    }

    close() {
        this.seq.close();
        sequenceRegistry.unregister(this);
    }

    async next() { return this.seq.next(); }
}

/**
 * One image predicted from another of the same size, outside a sequence:
 * an album's photo (nvdrv_predict_decode()). `ref` is the RGB image
 * before; returns the new RGB image, or null where the C side returns -1.
 */
export function predictDecode(ref, width, height, data) {
    return wasmPredictDecode(ref, width, height, data);
}

export { showRGB };
