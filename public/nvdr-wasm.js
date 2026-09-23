/*
 * The C decoder, compiled to WebAssembly (wasm/, `make wasm`), behind the
 * same decode() as nvdr.js. It is the code the command-line decoder runs,
 * so its pixels are the C decoder's, which nvdr.js is held to anyway
 * (scripts/crosscheck_wasm.mjs checks all three agree).
 *
 *   import { loadWasm } from './nvdr-wasm.js';
 *   await loadWasm();          // from here on nvdr.js's decode() uses it
 *
 * A container decoded with a fluid context (a sequence, a fluid album)
 * still goes through nvdr.js, which keeps the context's models; so does
 * everything when WebAssembly is not available.
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
            setFastDecoder(decodeWasm);
            return exports_;
        })().catch(() => null);
    }
    return loading;
}

/* Loads from bytes already in hand (Node, tests). */
export async function loadWasmBytes(bytes) {
    const { instance } = await WebAssembly.instantiate(bytes, {});
    exports_ = instance.exports;
    setFastDecoder(decodeWasm);
    return exports_;
}

/* decode() of nvdr.js, without a fluid context. Each view is a decode of
 * its own (the C decoder shows one layer per call); layer 0 and 1 are
 * the cheap ones. */
export function decodeWasm(buffer, maxLayer = 2, wantFlat = false, wantLow = false) {
    const w = exports_;
    const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    const header = readHeader(bytes);
    if (!header) return null;
    const at = w.nvdr_wasm_input(bytes.length);
    new Uint8Array(w.memory.buffer, at, bytes.length).set(bytes);
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
