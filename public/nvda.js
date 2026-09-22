/*
 * NVDA albums in the browser: many still images in one file, each decoded
 * with the models the one before it left behind. Mirrors src/nvda.c.
 */
import { decode, newContext } from './nvdr.js';

const MAGIC = 0x4144564e;   // "NVDA" read as a little-endian uint32
const VERSION = 1;
const HEADER_SIZE = 16;
const FLAG_FLUID = 0x01;
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
            v.getUint8(4) !== VERSION || (v.getUint8(5) & ~FLAG_FLUID))
            throw new Error('not an NVDA album');
        this.bytes = new Uint8Array(buffer);
        this.view = v;
        this.count = v.getUint32(6, true);
        if (this.count > MAX_IMAGES) throw new Error('not an NVDA album');
        this.fluid = (v.getUint8(5) & FLAG_FLUID) !== 0;
        this.ctx = this.fluid ? newContext() : null;
        this.pos = HEADER_SIZE;
        this.index = 0;
    }

    next(wantFlat = false) {
        const { bytes, view } = this, size = bytes.length;
        if (this.index >= this.count || this.pos + 2 > size) return null;
        const nl = view.getUint16(this.pos, true);
        if (nl > 255) throw new Error('album is damaged');
        if (this.pos + 2 + nl + 4 > size) return null;
        const name = new TextDecoder().decode(bytes.subarray(this.pos + 2, this.pos + 2 + nl));
        let len = view.getUint32(this.pos + 2 + nl, true);
        const at = this.pos + 2 + nl + 4;
        let partial = false;
        if (len > size - at) { len = size - at; partial = true; }
        const result = decode(bytes.subarray(at, at + len), 2, wantFlat, this.ctx);
        if (!result) {
            if (partial) return null;
            throw new Error('album is damaged');
        }
        const index = this.index;
        this.pos = at + len;
        this.index = partial ? this.count : this.index + 1;
        return { index, name, bytes: len, partial, result };
    }
}
