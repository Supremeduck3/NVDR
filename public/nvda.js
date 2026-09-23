/*
 * NVDA albums in the browser. Mirrors src/nvda.c: an index up front, and
 * every photo either coded on its own or predicted from an earlier photo
 * within the album's window. Any photo can be decoded on its own, pulling
 * in only its chain of references; openAlbumImage() below fetches exactly
 * those bytes from a server with range requests.
 */
import { decode, newContext } from './nvdr.js';
import { predictDecode } from './nvdrv.js';

const MAGIC = 0x4144564e;   // "NVDA" read as a little-endian uint32
const VERSION = 3;
const HEADER_SIZE = 16;
const ENTRY_SIZE = 16;
const FLAG_FLUID = 0x01;
const FLAG_PREDICT = 0x02;
const MAX_IMAGES = 4096;
const MAX_WINDOW = 32;
const MAX_PIXELS = 1 << 27;
export const KIND_INTRA = 0, KIND_PRED = 1;

export function isAlbum(buffer) {
    return buffer.byteLength >= 4 && new DataView(buffer).getUint32(0, true) === MAGIC;
}

/* Header and index; throws where nvda_open() returns -1. */
function parseIndex(bytes) {
    const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (bytes.length < HEADER_SIZE || v.getUint32(0, true) !== MAGIC || v.getUint8(4) !== VERSION ||
        (v.getUint8(5) & ~(FLAG_FLUID | FLAG_PREDICT)))
        throw new Error('not an NVDA album');
    const flags = v.getUint8(5), count = v.getUint32(6, true), window = v.getUint8(10);
    if (count > MAX_IMAGES || window < 1 || window > MAX_WINDOW) throw new Error('not an NVDA album');
    const table = HEADER_SIZE + count * ENTRY_SIZE;
    if (bytes.length < table) throw new Error('album index incomplete');
    const index = [];
    for (let i = 0; i < count; i++) {
        const e = HEADER_SIZE + i * ENTRY_SIZE;
        const x = {
            offset: v.getUint32(e, true), length: v.getUint32(e + 4, true),
            kind: v.getUint8(e + 8), ref: v.getUint8(e + 9),
            width: v.getUint16(e + 10, true), height: v.getUint16(e + 12, true),
            nameLen: v.getUint16(e + 14, true)
        };
        let ok = x.nameLen <= 255 && x.offset >= table + x.nameLen && x.width > 0 && x.height > 0 &&
                 x.width * x.height <= MAX_PIXELS;
        if (x.kind === KIND_INTRA) ok = ok && x.ref === 0;
        else if (x.kind === KIND_PRED) {
            ok = ok && x.ref >= 1 && x.ref <= window && x.ref <= i;
            if (ok) { const y = index[i - x.ref]; ok = y.width === x.width && y.height === x.height; }
        } else ok = false;
        if (!ok) throw new Error('album index is damaged');
        index.push(x);
    }
    return { flags, count, window, table, index };
}

/**
 * Reads an album from the bytes it has (the whole file, or with sparse
 * byte ranges filled in: see openAlbumImage). decode(i) returns
 * { width, height, rgb, partial }, null when the bytes are not there yet,
 * and throws on damage, as nvda_decode() returns 1, 0 and -1.
 */
export class AlbumReader {
    constructor(buffer) {
        this.bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
        Object.assign(this, parseIndex(this.bytes));
        this.fluid = (this.flags & FLAG_FLUID) !== 0;
        this.ctx = this.fluid ? newContext() : null;
        this.nextFluid = 0;
        this.ring = new Array(this.window).fill(null);   // { i, rgb, width, height }
    }

    name(i) {
        const x = this.index[i], at = x.offset - x.nameLen;
        if (at + x.nameLen > this.bytes.length) return '';
        return new TextDecoder().decode(this.bytes.subarray(at, at + x.nameLen));
    }

    /* The images photo i needs, itself last: its chain of references. */
    chain(i) {
        const out = [i];
        while (this.index[i].kind === KIND_PRED) { i -= this.index[i].ref; out.unshift(i); }
        return out;
    }

    decode(i) {
        if (i < 0 || i >= this.count) throw new Error('no such image');
        const slot = this.ring[i % this.window];
        if (slot && slot.i === i) return { width: slot.width, height: slot.height, rgb: slot.rgb.slice(), partial: false };
        const x = this.index[i];
        if (this.fluid && i !== this.nextFluid) throw new Error('a fluid album reads in order');
        if (x.offset >= this.bytes.length) return null;
        let len = x.length, cut = false;
        if (len > this.bytes.length - x.offset) { len = this.bytes.length - x.offset; cut = true; }
        const payload = this.bytes.subarray(x.offset, x.offset + len);
        let rgb;
        if (x.kind === KIND_INTRA) {
            const r = decode(payload, 2, false, this.ctx);
            if (!r) { if (cut) return null; throw new Error('album is damaged'); }
            if (r.header.width !== x.width || r.header.height !== x.height) throw new Error('album is damaged');
            rgb = r.rgb;
        } else {
            const ref = this.decode(i - x.ref);
            if (!ref) return null;
            if (ref.partial) return null;
            rgb = predictDecode(ref.rgb, x.width, x.height, payload);
            if (!rgb) { if (cut) return null; throw new Error('album is damaged'); }
        }
        if (this.fluid) this.nextFluid = i + 1;
        if (!cut) this.ring[i % this.window] = { i, rgb, width: x.width, height: x.height };
        return { width: x.width, height: x.height, rgb: cut ? rgb : rgb.slice(), partial: cut };
    }
}

/*
 * What this page already fetched from each album: the requests, by URL and
 * byte range. Several <nvdr-img> of one album share the header, the index
 * and the photos their chains have in common, so a page showing the whole
 * album downloads about the album, not each chain again.
 */
const fetchedRanges = new Map();

function cachedRange(url, start, end) {
    let ranges = fetchedRanges.get(url);
    if (!ranges) fetchedRanges.set(url, ranges = new Map());
    const key = `${start}-${end}`;
    let p = ranges.get(key);
    if (!p) {
        p = (async () => {
            const res = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` } });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return { buf: new Uint8Array(await res.arrayBuffer()), whole: res.status === 200 };
        })();
        p.catch(() => ranges.delete(key));   // a failed request may be retried
        ranges.set(key, p);
    }
    return p.then(r => ({ ...r, fresh: !p.counted && (p.counted = true) }));
}

/**
 * One photo of an album on a server, fetching only what it needs: the
 * header, the index, then the byte ranges of the photo and its chain of
 * references. A server that ignores ranges sends the whole file, which
 * works the same way, only heavier. `which` is a 0-based index or a name.
 * Returns { reader, index, image, fetched } where `fetched` counts the
 * bytes this call downloaded; ranges another photo of the page already
 * fetched are reused and not counted.
 */
export async function openAlbumImage(url, which) {
    let fetched = 0;
    const get = async (start, end) => {
        const r = await cachedRange(url, start, end);
        if (r.fresh) fetched += r.buf.length;
        return r;
    };
    // Header first, then the index it announces.
    let head = await get(0, HEADER_SIZE);
    let file = null;
    if (head.whole) file = head.buf;
    const count = new DataView(head.buf.buffer, head.buf.byteOffset).getUint32(6, true);
    const table = HEADER_SIZE + Math.min(count, MAX_IMAGES) * ENTRY_SIZE;
    if (!file) {
        const t = await get(0, table);
        if (t.whole) file = t.buf;
        else head = t;
    }
    let bytes;
    let reader;
    if (file) {
        bytes = file;
        reader = new AlbumReader(bytes);
    } else {
        const { index } = parseIndex(head.buf);
        const end = Math.max(...index.map(x => x.offset + x.length));
        bytes = new Uint8Array(end);
        bytes.set(head.buf, 0);
        reader = new AlbumReader(bytes);
    }
    const i = typeof which === 'number' ? which
        : reader.index.findIndex((_, k) => reader.name(k) === which);
    if (i < 0 || i >= reader.count) throw new Error('no such image in album');
    if (!file) {
        if (reader.fluid) {
            // Every photo depends on all the ones before: fetch them all.
            const x = reader.index[i];
            const r = await get(reader.table, x.offset + x.length);
            bytes.set(r.buf, reader.table);
            for (let k = 0; k <= i; k++) reader.decode(k);
        } else {
            for (const k of reader.chain(i)) {
                const x = reader.index[k];
                const r = await get(x.offset - x.nameLen, x.offset + x.length);
                bytes.set(r.buf, x.offset - x.nameLen);
            }
        }
    } else if (reader.fluid) {
        for (let k = 0; k < i; k++) reader.decode(k);
    }
    return { reader, index: i, image: reader.decode(i), fetched };
}
