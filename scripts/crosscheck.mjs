/*
 * The C and JS decoders have to agree on every byte, or the viewer shows
 * something the container does not contain. This decodes the same
 * container both ways — each layer, cuts that land inside either layer,
 * and copies with a few bits flipped, whole length, which is where the JS
 * decoder starts over (see decode() in nvdr.js), and copies with holes
 * (packets that never arrived, filled with noise the decoders must not
 * read: nvdr_decode --missing, decode()'s `holes`) — and reports the
 * first pixel where they differ.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { decodeJs as decode, readHeader, sliceParts } from '../public/nvdr.js';
import { loadWasmBytes, decodeWasm } from '../public/nvdr-wasm.js';

// The WebAssembly build of the C decoder is held to the same pixels, when
// it has been built (make wasm).
const wasmPath = new URL('../public/nvdr.wasm', import.meta.url).pathname;
let wasm = false;
try { await loadWasmBytes(readFileSync(wasmPath)); wasm = true; } catch { }

const [, , container] = process.argv;
if (!container) {
    console.error('usage: node scripts/crosscheck.mjs <file.nvdr>');
    process.exit(2);
}

function readPPM(path) {
    const buf = readFileSync(path);
    // P6\n<w> <h>\n255\n
    let at = 0;
    const fields = [];
    while (fields.length < 4) {
        while (buf[at] === 0x20 || buf[at] === 0x0a || buf[at] === 0x09) at++;
        const start = at;
        while (at < buf.length && buf[at] !== 0x20 && buf[at] !== 0x0a && buf[at] !== 0x09) at++;
        fields.push(buf.toString('ascii', start, at));
    }
    at++;
    return { width: +fields[1], height: +fields[2], pixels: buf.subarray(at) };
}

const decoder = new URL('../nvdr_decode', import.meta.url).pathname;
let failures = 0;

function check(label, file, layer, holes = null) {
    const bytes = readFileSync(file);
    let cDecoded = true;
    const missing = holes ? ['--missing', holes.map(([a, b]) => `${a}-${b}`).join(',')] : [];
    try {
        execFileSync(decoder, [file, '/tmp/cc.ppm', '--layer', String(layer), ...missing], { stdio: 'ignore' });
    } catch {
        cDecoded = false;
    }
    const result = decode(new Uint8Array(bytes), layer, false, null, false, null, holes);
    const jsDecoded = !!result;
    if (!cDecoded || !jsDecoded) {
        if (cDecoded !== jsDecoded) {
            console.log(`${label}: C ${cDecoded ? 'decoded' : 'refused'}, ` +
                        `JS ${jsDecoded ? 'decoded' : 'refused'}`);
            failures++;
        } else {
            console.log(`${label}: both refuse (no colour layer yet)`);
        }
        return;
    }
    const ref = readPPM('/tmp/cc.ppm');
    if (wasm && !holes) {
        const w = decodeWasm(new Uint8Array(bytes), layer);
        for (let i = 0; i < ref.pixels.length; i++)
            if (!w || w.rgb[i] !== ref.pixels[i]) {
                console.log(`${label}: WebAssembly differs from C at pixel ${Math.floor(i / 3)}`);
                failures++;
                return;
            }
    }
    const got = result.rgb;
    for (let i = 0; i < got.length; i++)
        if (got[i] !== ref.pixels[i]) {
            console.log(`${label}: differ at pixel ${Math.floor(i / 3)} channel ${i % 3}: ` +
                        `js ${got[i]} vs c ${ref.pixels[i]}`);
            failures++;
            return;
        }
    console.log(`${label}: identical${wasm && !holes ? ' (C, JS, WASM)' : ' (C, JS)'} (${result.header.width}x${result.header.height}, ` +
                `tiles ${result.tilesComplete.join('/')} of ${result.tiles})`);
}

const size = readFileSync(container).length;
for (const layer of [0, 1, 2]) check(`whole layer=${layer}`, container, layer);
for (const pct of [3, 8, 17, 34, 52, 71, 88, 96]) {
    const cut = `/tmp/cc_${pct}.nvdr`;
    execFileSync('bash', ['-c', `head -c ${Math.floor(size * pct / 100)} ${container} > ${cut}`]);
    check(`cut ${pct}%`, cut, 2);
    check(`cut ${pct}% layer=1`, cut, 1);
}
// Damage at full length: a deterministic handful of flipped bits past
// the header, so the layers read to their end and stop mid-tile.
const whole = readFileSync(container);
let seed = 12345;
const rnd = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296;
for (let k = 0; k < 12; k++) {
    const b = Buffer.from(whole);
    for (let j = 0, flips = 1 + Math.floor(rnd() * 4); j < flips; j++)
        b[32 + Math.floor(rnd() * (b.length - 32))] ^= 1 << Math.floor(rnd() * 8);
    const damaged = `/tmp/cc_damaged_${k}.nvdr`;
    writeFileSync(damaged, b);
    check(`damaged ${k}`, damaged, 2);
}
// Holes: one to three lost packets of 1200 bytes past what precedes the
// first layer, their bytes replaced with noise.
const hdr = readHeader(new Uint8Array(whole));
const first = hdr ? sliceParts(hdr)[0].at : 32;
for (let k = 0; k < 8 && whole.length > first + 64; k++) {
    const b = Buffer.from(whole), holes = [];
    for (let j = 0, n = 1 + Math.floor(rnd() * 3); j < n; j++) {
        const a = first + Math.floor(rnd() * (b.length - first));
        holes.push([a, Math.min(b.length, a + 1200)]);
    }
    holes.sort((x, y) => x[0] - y[0]);
    for (const [a, e] of holes) for (let i = a; i < e; i++) b[i] = Math.floor(rnd() * 256);
    const holed = `/tmp/cc_holes_${k}.nvdr`;
    writeFileSync(holed, b);
    const label = `holes ${k} (${holes.map(([a, e]) => `${a}-${e}`).join(',')})`;
    check(label, holed, 2, holes);
    // What lies in a hole must make no difference: noise against zeros.
    for (const [a, e] of holes) b.fill(0, a, e);
    writeFileSync(holed + '0', b);
    const js = decode(new Uint8Array(b), 2, false, null, false, null, holes);
    const ref = readPPM('/tmp/cc.ppm');
    if (js && js.rgb.some((v, i) => v !== ref.pixels[i])) {
        console.log(`${label}: the bytes in a hole change the picture`);
        failures++;
    }
}
process.exit(failures ? 1 : 0);
