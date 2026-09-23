/*
 * The C and JS decoders have to agree on every byte, or the viewer shows
 * something the container does not contain. This decodes the same
 * container both ways — each layer, cuts that land inside either layer,
 * and copies with a few bits flipped, whole length, which is where the JS
 * decoder starts over (see decode() in nvdr.js) — and reports the first
 * pixel where they differ.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { decodeJs as decode } from '../public/nvdr.js';
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

function check(label, file, layer) {
    const bytes = readFileSync(file);
    let cDecoded = true;
    try {
        execFileSync(decoder, [file, '/tmp/cc.ppm', '--layer', String(layer)], { stdio: 'ignore' });
    } catch {
        cDecoded = false;
    }
    const result = decode(new Uint8Array(bytes), layer);
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
    if (wasm) {
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
    console.log(`${label}: identical${wasm ? ' (C, JS, WASM)' : ''} (${result.header.width}x${result.header.height}, ` +
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
process.exit(failures ? 1 : 0);
