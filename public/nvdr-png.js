/*
 * NVDR files dressed as PNGs, as tools/nvdr_png.c makes them: an ordinary
 * PNG whose picture is a preview any viewer shows, and whose private
 * ancillary chunk `nvDr` carries the NVDR file (image, album or video)
 * untouched. The chunk comes right after IHDR, so its data starts at byte
 * 41 and a page streaming the PNG can decode the NVDR inside as it
 * arrives (nvdr-img.js).
 */

const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const NVDR_CHUNK = [0x6e, 0x76, 0x44, 0x72];   // "nvDr"
const IEND = [0x49, 0x45, 0x4e, 0x44];
export const NVDR_AT = 41;                     // where the chunk's data starts

const u8 = b => b instanceof Uint8Array ? b : new Uint8Array(b);
const at = (b, i, what) => what.every((v, k) => b[i + k] === v);
const be32 = (b, i) => ((b[i] << 24) | (b[i + 1] << 16) | (b[i + 2] << 8) | b[i + 3]) >>> 0;

export const isPng = b => { b = u8(b); return b.length >= 8 && at(b, 0, SIGNATURE); };

/* What a file is by its first four bytes: 'NVDR' an image, 'NVDA' an
 * album, 'NVDV' a video, or null. */
export function kindOf(b) {
    b = u8(b);
    if (b.length < 4) return null;
    const m = String.fromCharCode(b[0], b[1], b[2], b[3]);
    return ['NVDR', 'NVDA', 'NVDV'].includes(m) ? m : null;
}

let table = null;
function crc32(b, from, to) {
    if (!table) {
        table = new Uint32Array(256);
        for (let n = 0; n < 256; n++) {
            let c = n;
            for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
            table[n] = c >>> 0;
        }
    }
    let c = 0xffffffff;
    for (let i = from; i < to; i++) c = table[(c ^ b[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
}

/* The NVDR file inside a whole PNG, as a view of the same bytes; null
 * when `bytes` is not a PNG, carries none, or its chunk is damaged. */
export function nvdrFromPng(bytes) {
    const b = u8(bytes);
    if (!isPng(b)) return null;
    for (let i = 8; i + 12 <= b.length;) {
        const n = be32(b, i);
        if (n > b.length - i - 12) return null;
        if (at(b, i + 4, NVDR_CHUNK))
            return crc32(b, i + 4, i + 8 + n) === be32(b, i + 8 + n) ? b.subarray(i + 8, i + 8 + n) : null;
        if (at(b, i + 4, IEND)) return null;
        i += 12 + n;
    }
    return null;
}

/* For a PNG still arriving: once its first 41 bytes are in, where the
 * NVDR inside starts and how long it is ({ start, length }); null when
 * they say it is not laid out the way nvdr_png lays it out, undefined
 * while too little has arrived to tell. */
export function nvdrInPngPrefix(bytes, have) {
    const b = u8(bytes);
    if (have < NVDR_AT) return have >= 8 && !at(b, 0, SIGNATURE) ? null : undefined;
    if (!at(b, 0, SIGNATURE) || !at(b, 37, NVDR_CHUNK)) return null;
    return { start: NVDR_AT, length: be32(b, 33) };
}

/* Puts an NVDR file inside a PNG (`preview`, any valid one, such as a
 * canvas's toBlob('image/png')), right after its IHDR. */
export function wrapPng(nvdr, preview) {
    const p = u8(preview), d = u8(nvdr);
    if (!isPng(p) || p.length < 33) throw new Error('the preview is not a PNG');
    const out = new Uint8Array(p.length + 12 + d.length);
    out.set(p.subarray(0, 33), 0);                 // signature and IHDR
    const v = new DataView(out.buffer);
    v.setUint32(33, d.length);
    out.set(NVDR_CHUNK, 37);
    out.set(d, NVDR_AT);
    v.setUint32(NVDR_AT + d.length, crc32(out, 37, NVDR_AT + d.length));
    out.set(p.subarray(33), NVDR_AT + d.length + 4);
    return out;
}

/*
 * A PNG carrying `bytes` (an NVDR image, album or video), for a page to
 * offer as a download. The page's server makes it with nvdr_png, whose
 * preview is a 256-colour palette; served as plain files (no /png), the
 * preview is `canvas` shrunk to 320 pixels on its longer side, in full
 * colour, which costs more bytes.
 */
export async function pngFor(bytes, canvas) {
    const kind = kindOf(bytes);
    const ext = { NVDR: '.nvdr', NVDA: '.nvda', NVDV: '.nvdrv' }[kind];
    if (!ext) throw new Error('not an NVDR file');
    try {
        const res = await fetch('/png', { method: 'POST', headers: { 'x-file-ext': ext }, body: bytes });
        if (res.ok) {
            const png = new Uint8Array(await res.arrayBuffer());
            if (nvdrFromPng(png)) return png;
        }
    } catch { /* no server: made here */ }
    const side = 320, s = Math.min(1, side / Math.max(canvas.width, canvas.height));
    const small = document.createElement('canvas');
    small.width = Math.max(1, Math.round(canvas.width * s));
    small.height = Math.max(1, Math.round(canvas.height * s));
    const g = small.getContext('2d');
    g.imageSmoothingQuality = 'high';
    g.drawImage(canvas, 0, 0, small.width, small.height);
    const blob = await new Promise(r => small.toBlob(r, 'image/png'));
    return wrapPng(bytes, await blob.arrayBuffer());
}

/* Hands `bytes` to the browser as a download named `name`. */
export function download(bytes, name, type = 'application/octet-stream') {
    const url = URL.createObjectURL(new Blob([bytes], { type }));
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
}
