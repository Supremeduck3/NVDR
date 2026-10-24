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

/* ---------------------------------------------- previews, as nvdr_png's */

const PREVIEW_MAX = 640, PREVIEW_MIN = 128, REFINE = 5, DITHER = 75;

/* `rgb` (3 bytes a pixel, w x h) averaged down so the longer side is
 * `side`: shrink() in nvdr_png.c. */
function shrinkTo(rgb, w, h, side) {
    const big = Math.max(w, h);
    if (big <= side) return { rgb, w, h };
    const W = Math.max(1, Math.floor(w * side / big)), H = Math.max(1, Math.floor(h * side / big));
    const o = new Uint8Array(W * H * 3);
    for (let y = 0; y < H; y++) {
        const y0 = Math.floor(y * h / H), y1 = Math.floor((y + 1) * h / H);
        for (let x = 0; x < W; x++) {
            const x0 = Math.floor(x * w / W), x1 = Math.floor((x + 1) * w / W), n = (x1 - x0) * (y1 - y0);
            for (let c = 0; c < 3; c++) {
                let s = 0;
                for (let j = y0; j < y1; j++) for (let i = x0; i < x1; i++) s += rgb[(j * w + i) * 3 + c];
                o[(y * W + x) * 3 + c] = Math.floor((s + (n >> 1)) / n);
            }
        }
    }
    return { rgb: o, w: W, h: H };
}

/* Median cut on a 15-bit histogram, then k-means: quantise_palette(). */
function palette(rgb, colours) {
    const hist = new Float64Array(32768), sum = new Float64Array(32768 * 3);
    for (let i = 0; i < rgb.length; i += 3) {
        const k = (rgb[i] >> 3) << 10 | (rgb[i + 1] >> 3) << 5 | rgb[i + 2] >> 3;
        hist[k]++; sum[k * 3] += rgb[i]; sum[k * 3 + 1] += rgb[i + 1]; sum[k * 3 + 2] += rgb[i + 2];
    }
    const fit = b => {
        const lo = [31, 31, 31], hi = [0, 0, 0];
        let n = 0;
        for (let r = b.lo[0]; r <= b.hi[0]; r++) for (let g = b.lo[1]; g <= b.hi[1]; g++) for (let u = b.lo[2]; u <= b.hi[2]; u++) {
            const h = hist[r << 10 | g << 5 | u];
            if (!h) continue;
            n += h;
            [r, g, u].forEach((v, c) => { if (v < lo[c]) lo[c] = v; if (v > hi[c]) hi[c] = v; });
        }
        b.count = n;
        if (n) { b.lo = lo; b.hi = hi; }
        return b;
    };
    const boxes = [fit({ lo: [0, 0, 0], hi: [31, 31, 31], count: 0 })];
    while (boxes.length < colours) {
        let pick = -1, axis = 0, worth = 0;
        boxes.forEach((b, i) => {
            let a = 0;
            for (let c = 1; c < 3; c++) if (b.hi[c] - b.lo[c] > b.hi[a] - b.lo[a]) a = c;
            const w = b.count * (b.hi[a] - b.lo[a]);
            if (w > worth) { worth = w; pick = i; axis = a; }
        });
        if (pick < 0) break;
        const b = boxes[pick], half = Math.floor(b.count / 2);
        let acc = 0, cut = b.lo[axis];
        for (let v = b.lo[axis]; v < b.hi[axis]; v++) {
            const slab = fit({ lo: b.lo.slice(), hi: b.hi.slice(), count: 0 });
            slab.lo[axis] = slab.hi[axis] = v;
            acc += fit(slab).count;
            cut = v;
            if (acc >= half) break;
        }
        const upper = { lo: b.lo.slice(), hi: b.hi.slice(), count: 0 };
        b.hi = b.hi.slice(); b.hi[axis] = cut; upper.lo[axis] = cut + 1;
        fit(b); boxes.push(fit(upper));
    }
    const pal = boxes.map(b => {
        const s = [0, 0, 0];
        let c = 0;
        for (let r = b.lo[0]; r <= b.hi[0]; r++) for (let g = b.lo[1]; g <= b.hi[1]; g++) for (let u = b.lo[2]; u <= b.hi[2]; u++) {
            const k = r << 10 | g << 5 | u;
            c += hist[k]; s[0] += sum[k * 3]; s[1] += sum[k * 3 + 1]; s[2] += sum[k * 3 + 2];
        }
        return s.map(v => c ? Math.round(v / c) : 0);
    });
    for (let it = 0; it < REFINE; it++) {
        const acc = pal.map(() => [0, 0, 0, 0]);
        for (let k = 0; k < 32768; k++) {
            if (!hist[k]) continue;
            const v = [sum[k * 3] / hist[k], sum[k * 3 + 1] / hist[k], sum[k * 3 + 2] / hist[k]];
            let best = 0, bd = Infinity;
            pal.forEach((p, i) => { const d = (v[0] - p[0]) ** 2 + (v[1] - p[1]) ** 2 + (v[2] - p[2]) ** 2; if (d < bd) { bd = d; best = i; } });
            const a = acc[best];
            a[0] += sum[k * 3]; a[1] += sum[k * 3 + 1]; a[2] += sum[k * 3 + 2]; a[3] += hist[k];
        }
        acc.forEach((a, i) => { if (a[3]) pal[i] = [a[0], a[1], a[2]].map(v => Math.round(v / a[3])); });
    }
    return pal;
}

/* Palette indices, Floyd-Steinberg dithered at `dither` percent,
 * serpentine: map_pixels(). */
function indices(rgb, w, h, pal, dither) {
    const memo = new Int16Array(1 << 18).fill(-1), out = new Uint8Array(w * h);
    const nearest = (r, g, b) => {
        const k = (r >> 2) << 12 | (g >> 2) << 6 | b >> 2;
        if (memo[k] >= 0) return memo[k];
        let best = 0, bd = Infinity;
        pal.forEach((p, i) => { const d = (r - p[0]) ** 2 + (g - p[1]) ** 2 + (b - p[2]) ** 2; if (d < bd) { bd = d; best = i; } });
        return (memo[k] = best);
    };
    let cur = new Float32Array((w + 2) * 3), next = new Float32Array((w + 2) * 3);
    for (let y = 0; y < h; y++) {
        next.fill(0);
        const rev = y & 1;
        for (let i = 0; i < w; i++) {
            const x = rev ? w - 1 - i : i, dx = rev ? -1 : 1, v = [0, 0, 0];
            for (let c = 0; c < 3; c++) {
                const t = rgb[(y * w + x) * 3 + c] + cur[(x + 1) * 3 + c];
                v[c] = t < 0 ? 0 : t > 255 ? 255 : Math.round(t);
            }
            const k = nearest(v[0], v[1], v[2]);
            out[y * w + x] = k;
            for (let c = 0; c < 3; c++) {
                const e = (v[c] - pal[k][c]) * dither / 100;
                cur[(x + 1 + dx) * 3 + c] += e * 7 / 16;
                next[(x + 1 - dx) * 3 + c] += e * 3 / 16;
                next[(x + 1) * 3 + c] += e * 5 / 16;
                next[(x + 1 + dx) * 3 + c] += e / 16;
            }
        }
        [cur, next] = [next, cur];
    }
    return out;
}

function chunk(type, data) {
    const out = new Uint8Array(12 + data.length), v = new DataView(out.buffer);
    v.setUint32(0, data.length);
    for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
    out.set(data, 8);
    v.setUint32(8 + data.length, crc32(out, 4, 8 + data.length));
    return out;
}

/* A palette PNG of `rgb` at `side`, deflated by the browser (zlib's
 * format is what IDAT holds). */
async function palettePng(rgb, w, h, side, dither) {
    const s = shrinkTo(rgb, w, h, side), pal = palette(s.rgb, 256), idx = indices(s.rgb, s.w, s.h, pal, dither);
    const raw = new Uint8Array((s.w + 1) * s.h);
    for (let y = 0; y < s.h; y++) raw.set(idx.subarray(y * s.w, (y + 1) * s.w), y * (s.w + 1) + 1);
    const idat = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new CompressionStream('deflate'))).arrayBuffer());
    const ihdr = new Uint8Array(13), v = new DataView(ihdr.buffer);
    v.setUint32(0, s.w); v.setUint32(4, s.h); ihdr[8] = 8; ihdr[9] = 3;
    const parts = [new Uint8Array(SIGNATURE), chunk('IHDR', ihdr), chunk('PLTE', Uint8Array.from(pal.flat())),
                   chunk('IDAT', idat), chunk('IEND', new Uint8Array(0))];
    const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let at = 0;
    for (const p of parts) { out.set(p, at); at += p.length; }
    return { png: out, idat: idat.length };
}

/*
 * A PNG carrying `bytes` (an NVDR image, album or video), for a page to
 * offer as a download, with `rgb` (w x h, the decoded picture) as its
 * preview. The page's server makes it with nvdr_png when it has /png;
 * served as plain files, it is made here the same way. By default
 * (opts.preview 'auto') the preview is the largest, to 8 pixels and at
 * most 640, whose picture data costs no more than the NVDR file; a
 * number fixes its longer side instead (0 whole). opts.dither is the
 * percent of colour error diffused, 75 by default.
 */
export async function pngFor(bytes, rgb, w, h, opts = {}) {
    const kind = kindOf(bytes);
    const ext = { NVDR: '.nvdr', NVDA: '.nvda', NVDV: '.nvdrv' }[kind];
    if (!ext) throw new Error('not an NVDR file');
    const fixed = opts.preview !== undefined && opts.preview !== 'auto' ? Number(opts.preview) : null;
    const dither = opts.dither ?? DITHER;
    try {
        const headers = { 'x-file-ext': ext, 'x-dither': String(dither) };
        if (fixed !== null) headers['x-preview'] = String(fixed);
        const res = await fetch('/png', { method: 'POST', headers, body: bytes });
        if (res.ok) {
            const png = new Uint8Array(await res.arrayBuffer());
            if (nvdrFromPng(png)) return png;
        }
    } catch { /* no server: made here */ }
    if (fixed !== null) return wrapPng(bytes, (await palettePng(rgb, w, h, fixed || Math.max(w, h), dither)).png);
    const budget = bytes.byteLength;
    let hi = Math.min(PREVIEW_MAX, Math.max(w, h)), best = await palettePng(rgb, w, h, hi, dither);
    if (best.idat > budget && hi > PREVIEW_MIN) {
        let lo = PREVIEW_MIN;
        best = await palettePng(rgb, w, h, lo, dither);
        while (best.idat <= budget && hi - lo > 8) {
            const mid = (lo + hi) >> 1, t = await palettePng(rgb, w, h, mid, dither);
            if (t.idat <= budget) { best = t; lo = mid; } else hi = mid;
        }
    }
    return wrapPng(bytes, best.png);
}

/* The page's PNG options, from its first `.png-preview` and `.png-dither`
 * selects (the page keeps every copy of them in step). */
export function pngOptions() {
    const p = document.querySelector('.png-preview'), d = document.querySelector('.png-dither');
    return { preview: p ? p.value : 'auto', dither: d ? Number(d.value) : DITHER };
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
