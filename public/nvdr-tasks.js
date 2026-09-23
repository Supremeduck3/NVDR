/*
 * The decoding work a page hands off: what nvdr-worker.js runs off the
 * main thread, and what nvdr-decoder.js runs in place when a module
 * worker cannot start. Every call takes a plain object and returns
 * { result, transfer }, where `transfer` lists the pixel buffers the
 * result owns, so a worker can hand them over without copying.
 *
 * State lives here, per worker: containers put once and decoded many
 * times (a slider cutting the same file at every position), open album
 * readers (which hold the photos their chains need), and nvda.js's cache
 * of byte ranges already fetched from each album URL.
 */
import { decode, shrinkRGB, fitSize } from './nvdr.js';
import { AlbumReader, openAlbumImage } from './nvda.js';
import { loadWasm, wasmActive } from './nvdr-wasm.js';

// The C decoder as WebAssembly, when it loads: from then on decode()
// (here and in nvda.js) uses it, twice as fast on a large photo and
// several times on a small one. Until it has loaded, or where it cannot,
// the JavaScript decoder answers.
const wasm = loadWasm();

const stored = new Map();   // key -> Uint8Array
const albums = new Map();   // key -> AlbumReader

function bytesOf(m) {
    if (m.bytes) return m.bytes instanceof Uint8Array ? m.bytes : new Uint8Array(m.bytes);
    const b = stored.get(m.key);
    if (!b) throw new Error(`nothing stored as ${m.key}`);
    return m.length === undefined ? b : b.subarray(0, m.length);
}

/*
 * `fit`, when a task gives it, is the device-pixel width the picture will
 * be shown at: the result then also carries it shrunk to that size
 * ({ rgb, w, h }, see paintFitted), so the page does not average tens of
 * megapixels on its own thread. Null when the picture is not larger.
 */
function fitted(rgb, width, height, fit) {
    if (!rgb || !fit || fit >= width) return null;
    const { w, h } = fitSize(width, height, fit);
    return { rgb: shrinkRGB(rgb, width, height, w, h), w, h };
}

const buffers = (...arrays) => arrays.filter(Boolean).map(a => a.buffer);

function image(img, fit) {
    if (!img) return { result: null, transfer: [] };
    img.fitted = fitted(img.rgb, img.width, img.height, fit);
    return { result: img, transfer: buffers(img.rgb, img.fitted && img.fitted.rgb) };
}

const ops = {
    /* Keep a container (or an album) to decode later by key. */
    put(m) {
        stored.set(m.key, m.bytes instanceof Uint8Array ? m.bytes : new Uint8Array(m.bytes));
        albums.delete(m.key);
        return { result: true, transfer: [] };
    },

    drop(m) {
        stored.delete(m.key);
        albums.delete(m.key);
        return { result: true, transfer: [] };
    },

    /* decode() of nvdr.js, on `bytes` or on the first `length` bytes of a
     * stored container. */
    decode(m) {
        const r = decode(bytesOf(m), m.maxLayer ?? 2, !!m.wantFlat, null, !!m.wantLow);
        if (!r) return { result: null, transfer: [] };
        const { width, height } = r.header;
        const result = {
            engine: wasmActive() ? 'WebAssembly' : 'JavaScript',
            header: r.header, layersPresent: r.layersPresent, tiles: r.tiles,
            tilesComplete: r.tilesComplete, rgb: r.rgb, flatRgb: r.flatRgb, lowRgb: r.lowRgb,
            fitted: fitted(r.rgb, width, height, m.fit),
            flatFitted: fitted(r.flatRgb, width, height, m.fit),
            lowFitted: fitted(r.lowRgb, width, height, m.fit)
        };
        const transfer = buffers(r.rgb, r.flatRgb, r.lowRgb, result.fitted && result.fitted.rgb,
                                 result.flatFitted && result.flatFitted.rgb,
                                 result.lowFitted && result.lowFitted.rgb);
        return { result, transfer };
    },

    /* Open a stored album: its index, for the page to lay out. */
    albumOpen(m) {
        const reader = new AlbumReader(bytesOf(m));
        albums.set(m.key, reader);
        const names = reader.index.map((_, i) => reader.name(i));
        return { result: { count: reader.count, window: reader.window, index: reader.index, names },
                 transfer: [] };
    },

    /* One photo of an opened album: { width, height, rgb, partial }, null
     * when its bytes are not there; throws on damage. */
    albumDecode(m) {
        const reader = albums.get(m.key);
        if (!reader) throw new Error(`no album open as ${m.key}`);
        return image(reader.decode(m.index), m.fit);
    },

    /* One photo of an album on a server, by range requests (see
     * openAlbumImage). `url` must be absolute: a worker has no page to
     * resolve it against. */
    async albumImage(m) {
        const got = await openAlbumImage(m.url, m.which);
        const { result, transfer } = image(got.image, m.fit);
        return { result: { image: result, index: got.index, fetched: got.fetched }, transfer };
    }
};

export async function runTask(m) {
    const op = ops[m.op];
    if (!op) throw new Error(`unknown task ${m.op}`);
    await wasm;
    return op(m);
}
