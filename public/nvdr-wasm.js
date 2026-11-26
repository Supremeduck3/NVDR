/*
 * The C decoder, compiled to WebAssembly (wasm/, `make wasm`), behind the
 * decoder behind nvdr.js's decode(), and the only one: it is the code the
 * command-line decoder runs, so its pixels are the C decoder's
 * (scripts/crosscheck.mjs holds the two builds to the same bytes).
 *
 *   import { loadWasm } from './nvdr-wasm.js';
 *   await loadWasm();          // from here on nvdr.js's decode() works
 *
 * Contexts for containers decoded in order (a sequence's frames, a fluid
 * album's photos) live here too, in a pool of 64.
 */
import { readHeader, setFastDecoder } from './nvdr.js';

let exports_ = null, loading = null;

/* Whether decode() is running the WebAssembly decoder. */
export const wasmActive = () => exports_ !== null;

export function loadWasm(url = new URL('./nvdr.wasm', import.meta.url)) {
    if (!loading) {
        loading = (async () => {
            if (typeof WebAssembly === 'undefined') return null;
            const res = await fetch(url);
            if (!res.ok) return null;
            const { instance } = await WebAssembly.instantiate(await res.arrayBuffer(), {});
            exports_ = instance.exports;
            setFastDecoder(decodeWasm, wasmContexts);
            return exports_;
        })().catch(() => null);
    }
    return loading;
}

/* Loads from bytes already in hand (Node, tests). */
export async function loadWasmBytes(bytes) {
    const { instance } = await WebAssembly.instantiate(bytes, {});
    exports_ = instance.exports;
    setFastDecoder(decodeWasm, wasmContexts);
    return exports_;
}

/* decode() of nvdr.js. Each view is a decode of its own (the C decoder
 * shows one layer per call); layer 0 and 1 are the cheap ones. */
/* The pool of contexts wasm/api.c keeps, for newDecodeContext(). */
const wasmContexts = {
    alloc: () => exports_.nvdr_wasm_ctx_alloc(),
    reset: k => exports_.nvdr_wasm_ctx_reset(k),
    free: k => exports_.nvdr_wasm_ctx_free(k)
};

export function decodeWasm(buffer, maxLayer = 2, wantFlat = false, wantLow = false, base = null, slot = -1,
                           holes = null) {
    const w = exports_;
    const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    const header = readHeader(bytes);
    if (!header) return null;
    const at = w.nvdr_wasm_input(bytes.length);
    new Uint8Array(w.memory.buffer, at, bytes.length).set(bytes);
    // A predicted container decodes against its base, at full quality.
    if (base && (header.flags & 0x40)) {
        const bp = w.nvdr_wasm_base(base.width, base.height);
        new Uint8Array(w.memory.buffer, bp, base.rgb.length).set(base.rgb);
    }
    // Byte ranges that never arrived: as [start, end) pairs, or flat.
    if (holes && holes.length) {
        const flat = Array.isArray(holes[0]) ? holes.flat() : Array.from(holes);
        const hp = w.nvdr_wasm_holes(flat.length >> 1);
        new Uint32Array(w.memory.buffer, hp, flat.length).set(flat);
    }
    const view = layer => {
        const r = w.nvdr_wasm_decode(bytes.length, layer);
        if (!r) { w.nvdr_wasm_release(); return null; }
        // Views of memory are made after the call: a decode can grow it.
        const m = new Int32Array(w.memory.buffer, r, 8);
        const out = {
            rgb: new Uint8Array(w.memory.buffer, m[0] >>> 0, m[1] * m[2] * 3).slice(),
            tiles: m[3], layersPresent: m[4], tilesComplete: [m[5], m[6], m[7]]
        };
        w.nvdr_wasm_release();
        return out;
    };
    // A context moves with one decode: the main view's.
    if (slot >= 0) w.nvdr_wasm_use_ctx(slot);
    const main = view(maxLayer);
    if (!main) return null;
    return {
        header,
        layersPresent: main.layersPresent,
        tiles: main.tiles,
        tilesComplete: main.tilesComplete,
        rgb: main.rgb,
        flatRgb: wantFlat ? view(0).rgb : null,
        lowRgb: wantLow ? view(1).rgb : null
    };
}

const NOT_LOADED = 'the WebAssembly decoder (nvdr.wasm) is not loaded: await loadWasm() first';

/* Copies `bytes` into the decoder's heap: the pointer, for a call that
 * takes it (or for nvdr_wasm_free()). */
function put(w, bytes) {
    const at = w.nvdr_wasm_alloc(bytes.length);
    if (!at) throw new Error('out of WebAssembly memory');
    new Uint8Array(w.memory.buffer, at, bytes.length).set(bytes);
    return at;
}

/*
 * A sequence decoder of src/nvdrv.c, the command-line decoder's, on
 * `bytes` (a whole .nvdrv or any prefix of one): { width, height, next(),
 * close() }. next() is the next frame in display order, { index, kind,
 * partial, pixels } with pixels a copy of its own, or null at the end;
 * it throws where the C decoder returns -1. Null when the header makes
 * no sense. close() gives the decoder's memory back.
 */
export function wasmSequence(bytes) {
    const w = exports_;
    if (!w) throw new Error(NOT_LOADED);
    const s = w.nvdr_wasm_seq_open(put(w, bytes), bytes.length);
    if (!s) return null;
    const info = new Int32Array(w.memory.buffer, s, 9);
    let open = true;
    return {
        width: info[5], height: info[6],
        next() {
            if (!open) throw new Error('sequence decoder closed');
            // The view after the call: a frame can grow the memory.
            const r = w.nvdr_wasm_seq_next(s);
            const m = new Int32Array(w.memory.buffer, r, 9);
            if (m[0] < 0) throw new Error('the sequence is damaged');
            if (m[0] === 0) return null;
            const pixels = new Uint8Array(w.memory.buffer, m[4] >>> 0, m[5] * m[6] * 3).slice();
            return { index: m[1], kind: m[2], partial: !!m[3], pixels };
        },
        close() { if (open) { open = false; w.nvdr_wasm_seq_close(s); } }
    };
}

/*
 * One picture predicted from another of its size, an album's photo
 * (nvdrv_predict_decode() in src/nvdrv.c): the new RGB picture, or null
 * where the C side returns -1.
 */
export function wasmPredictDecode(ref, width, height, data) {
    const w = exports_;
    if (!w) throw new Error(NOT_LOADED);
    const r = put(w, ref), d = put(w, data);
    const o = w.nvdr_wasm_predict(r, width, height, d, data.length);
    w.nvdr_wasm_free(d); w.nvdr_wasm_free(r);
    if (!o) return null;
    const out = new Uint8Array(w.memory.buffer, o, width * height * 3).slice();
    w.nvdr_wasm_free(o);
    return out;
}
