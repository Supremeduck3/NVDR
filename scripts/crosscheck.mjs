/*
 * The C and JS decoders have to agree on every byte, or the viewer shows
 * something the container does not contain. This decodes the same
 * container both ways — each layer, and cuts that land inside either
 * layer — and reports the first pixel where they differ.
 */
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { decode } from '../public/nvdr.js';

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
    const got = result.rgb;
    for (let i = 0; i < got.length; i++)
        if (got[i] !== ref.pixels[i]) {
            console.log(`${label}: differ at pixel ${Math.floor(i / 3)} channel ${i % 3}: ` +
                        `js ${got[i]} vs c ${ref.pixels[i]}`);
            failures++;
            return;
        }
    console.log(`${label}: identical (${result.header.width}x${result.header.height}, ` +
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
process.exit(failures ? 1 : 0);
