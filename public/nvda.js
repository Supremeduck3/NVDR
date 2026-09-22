/*
 * NVDA albums in the browser: many still images in one file. An image is
 * either coded on its own, with the models the one before it left behind,
 * or predicted from the image before it. Mirrors src/nvda.c.
 */
import { decode, newContext } from './nvdr.js';
import { predictDecode } from './nvdrv.js';

const MAGIC = 0x4144564e;   // "NVDA" read as a little-endian uint32
const VERSION = 2;
const HEADER_SIZE = 16;
const FLAG_FLUID = 0x01;
const FLAG_PREDICT = 0x02;
export const KIND_INTRA = 0, KIND_PRED = 1;
const MAX_IMAGES = 4096;

export function isAlbum(buffer) {
    return buffer.byteLength >= HEADER_SIZE &&
        new DataView(buffer).getUint32(0, true) === MAGIC;
}

/**
 * Walks an album. `next()` returns { index, name, bytes, partial, result }
 * with `result` what nvdr.js decode() gives, null at the end, and throws on
 * damage, where nvda_next() returns -1.
 */
export class AlbumReader {
    constructor(buffer) {
        const v = new DataView(buffer);
        if (buffer.byteLength < HEADER_SIZE || v.getUint32(0, true) !== MAGIC ||
            v.getUint8(4) !== VERSION || (v.getUint8(5) & ~(FLAG_FLUID | FLAG_PREDICT)))
            throw new Error('not an NVDA album');
        this.bytes = new Uint8Array(buffer);
        this.view = v;
        this.count = v.getUint32(6, true);
        if (this.count > MAX_IMAGES) throw new Error('not an NVDA album');
        this.fluid = (v.getUint8(5) & FLAG_FLUID) !== 0;
        this.ctx = this.fluid ? newContext() : null;
        this.pos = HEADER_SIZE;
        this.index = 0;
        this.prev = null;   // { rgb, width, height }: what a kind-1 image predicts from
    }

    /**
     * The next image: { index, name, kind, bytes, partial, width, height, rgb },
     * null at the end, and throws on damage (nvda_next() returning -1).
     */
    next() {
        const { bytes, view } = this, size = bytes.length;
        if (this.index >= this.count || this.pos + 2 > size) return null;
        const nl = view.getUint16(this.pos, true);
        if (nl > 255) throw new Error('album is damaged');
        if (this.pos + 2 + nl + 5 > size) return null;
        const name = new TextDecoder().decode(bytes.subarray(this.pos + 2, this.pos + 2 + nl));
        const kind = bytes[this.pos + 2 + nl];
        if (kind !== KIND_INTRA && kind !== KIND_PRED) throw new Error('album is damaged');
        let len = view.getUint32(this.pos + 2 + nl + 1, true);
        const at = this.pos + 2 + nl + 5;
        let partial = false;
        if (len > size - at) { len = size - at; partial = true; }
        const payload = bytes.subarray(at, at + len);
        let rgb = null, width = 0, height = 0;
        if (kind === KIND_INTRA) {
            const result = decode(payload, 2, false, this.ctx);
            if (result) { rgb = result.rgb; width = result.header.width; height = result.header.height; }
        } else {
            if (!this.prev) throw new Error('album is damaged');
            ({ width, height } = this.prev);
            rgb = predictDecode(this.prev.rgb, width, height, payload);
        }
        if (!rgb) {
            if (partial) return null;
            throw new Error('album is damaged');
        }
        this.prev = { rgb, width, height };
        const index = this.index;
        this.pos = at + len;
        this.index = partial ? this.count : this.index + 1;
        return { index, name, kind, bytes: len, partial, width, height, rgb };
    }
}
