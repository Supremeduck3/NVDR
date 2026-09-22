/*
 * The C and JS sequence decoders have to hold the same state after every
 * frame, byte for byte. A predicted frame is a residual on top of that
 * state, so a one-pixel disagreement at frame 1 is a different reference
 * at frame 2 and the error compounds down the clip — a player that drifts
 * from the file while every individual frame looks plausible.
 *
 * This decodes a sequence both ways, whole and cut at several points, and
 * reports the first frame and pixel where they part.
 *
 *     node scripts/crosscheck_seq.mjs file.nvdrv
 */
import { readFileSync, mkdtempSync, rmSync, writeFileSync, readdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { SequenceDecoder } from '../public/nvdrv.js';

const [, , container] = process.argv;
if (!container) {
    console.error('usage: node scripts/crosscheck_seq.mjs <file.nvdrv>');
    process.exit(2);
}
const decoder = new URL('../nvdrv_decode', import.meta.url).pathname;

function readPPM(path) {
    const buf = readFileSync(path);
    let at = 0; const f = [];
    while (f.length < 4) {
        while (buf[at] === 0x20 || buf[at] === 0x0a || buf[at] === 0x09) at++;
        const s = at;
        while (at < buf.length && buf[at] !== 0x20 && buf[at] !== 0x0a && buf[at] !== 0x09) at++;
        f.push(buf.toString('ascii', s, at));
    }
    return buf.subarray(at + 1);
}

let failures = 0;

async function check(label, bytes) {
    const dir = mkdtempSync(join(tmpdir(), 'nvdrv-cc-'));
    try {
        const file = join(dir, 'in.nvdrv');
        writeFileSync(file, bytes);
        let cFailed = false;
        try {
            execFileSync(decoder, [file, '--out', dir, '--smooth', '0', '--ppm'], { stdio: 'pipe' });
        } catch { cFailed = true; }
        const cFrames = readdirSync(dir).filter(n => n.endsWith('.ppm')).sort();

        let js, jsError = null, n = 0;
        try {
            js = new SequenceDecoder(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
        } catch (e) { jsError = e; }

        if (!jsError) {
            for (;;) {
                let f;
                try { f = await js.next(); } catch (e) { jsError = e; break; }
                if (!f) break;
                if (n >= cFrames.length) {
                    console.log(`${label}: JS produced frame ${n}, C stopped at ${cFrames.length}`);
                    failures++; return;
                }
                const ref = readPPM(join(dir, cFrames[n]));
                for (let i = 0; i < ref.length; i++) {
                    if (f.pixels[i] !== ref[i]) {
                        console.log(`${label}: frame ${n} differs at byte ${i}: ` +
                                    `js ${f.pixels[i]} vs c ${ref[i]}`);
                        failures++; return;
                    }
                }
                n++;
            }
        }
        if (n !== cFrames.length) {
            console.log(`${label}: JS decoded ${n} frames, C decoded ${cFrames.length}` +
                        (jsError ? ` (JS: ${jsError.message})` : ''));
            failures++; return;
        }
        console.log(`${label}: identical, ${n} frames` + (cFailed || jsError ? ' (both stopped on bad data)' : ''));
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
}

const whole = readFileSync(container);
await check('whole', whole);
for (const pct of [3, 11, 27, 44, 61, 78, 93]) {
    const cut = whole.subarray(0, Math.floor(whole.length * pct / 100));
    await check(`cut ${pct}%`, cut);
}
process.exit(failures ? 1 : 0);
