/*
 * The C and JS decoders have to agree on every byte, or the viewer shows
 * something the format did not encode. This decodes the same container
 * both ways — at every level, and at truncations that cut a level open —
 * and reports the first pixel where they differ.
 */
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { decode, renderLevel, SMOOTH_DEFAULT } from '../public/nvdr.js';

const [, , container] = process.argv;
if (!container) {
    console.error('usage: node scripts/crosscheck.mjs <file.nvdr>');
    process.exit(2);
}

function fakeCtx(w, h) {
    let out = null;
    return {
        createImageData: () => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h }),
        putImageData: img => { out = img; },
        get result() { return out; }
    };
}

function readPPM(path) {
    const buf = readFileSync(path);
    // P6\n<w> <h>\n255\n
    let at = 0, fields = [];
    while (fields.length < 4) {
        while (buf[at] === 0x20 || buf[at] === 0x0a || buf[at] === 0x09) at++;
        let start = at;
        while (at < buf.length && buf[at] !== 0x20 && buf[at] !== 0x0a && buf[at] !== 0x09) at++;
        fields.push(buf.toString('ascii', start, at));
    }
    at++;
    return { width: +fields[1], height: +fields[2], pixels: buf.subarray(at) };
}

let failures = 0;
async function check(label, file, level, smooth) {
    const bytes = readFileSync(file);
    const args = [file, '/tmp/cc.ppm', '--smooth', String(smooth)];
    if (level !== null) args.push('--level', String(level));
    execFileSync(new URL('../nvdr/nvdr_decode', import.meta.url).pathname, args, { stdio: 'ignore' });
    const ref = readPPM('/tmp/cc.ppm');

    const result = await decode(bytes.buffer.slice(bytes.byteOffset,
                                                   bytes.byteOffset + bytes.length));
    if (!result || result.levelsPresent === 0) {
        console.log(`${label}: JS decoded nothing, C did`); failures++; return;
    }
    const k = level === null ? result.levelsPresent - 1
                             : Math.min(level, result.levelsPresent - 1);
    const { width, height } = result.header;
    const ctx = fakeCtx(width, height);
    renderLevel(result.levels[k], width, height, ctx, smooth);
    const got = ctx.result.data;

    for (let p = 0; p < width * height; p++)
        for (let c = 0; c < 3; c++)
            if (got[p * 4 + c] !== ref.pixels[p * 3 + c]) {
                console.log(`${label}: differ at pixel ${p} channel ${c}: ` +
                            `js ${got[p * 4 + c]} vs c ${ref.pixels[p * 3 + c]}`);
                failures++;
                return;
            }
    console.log(`${label}: identical (${width}x${height}, level ${k})`);
}

const size = readFileSync(container).length;
for (const smooth of [0, SMOOTH_DEFAULT]) {
    for (const level of [0, 1, 2, null])
        await check(`whole smooth=${smooth} level=${level}`, container, level, smooth);
    for (const pct of [8, 17, 34, 52, 71, 88, 96]) {
        const cut = `/tmp/cc_${pct}.nvdr`;
        execFileSync('bash', ['-c',
            `head -c ${Math.floor(size * pct / 100)} ${container} > ${cut}`]);
        await check(`cut ${pct}% smooth=${smooth}`, cut, null, smooth);
    }
}
process.exit(failures ? 1 : 0);
