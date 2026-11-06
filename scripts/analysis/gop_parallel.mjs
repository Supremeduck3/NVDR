// node scripts/analysis/gop_parallel.mjs <frames-dir> [--jobs N] [--chunk 48] [--limit N] [-- encoder args]
//
// What coding a video's GOPs side by side would buy, measured before the
// encoder is changed for it. The frames are coded once whole, as
// nvdrv_encode does now, and once cut into pieces of `--chunk` frames (the
// GOP, 48 by default), up to `--jobs` pieces at a time, each its own
// nvdrv_encode with the machine's cores shared out through
// OMP_NUM_THREADS. Reports wall time, bytes and PSNR-Y for both.
//
// The pieces are not what a chunked encoder would write. The groups are
// open: the B frames before an intra frame predict from it, and the
// temporal filter and look-ahead of an intra frame read the frames around
// it. A piece coded alone has neither, so it ends on P frames and filters
// its first frame from one side. What that costs in bytes and dB is the
// other half of the measurement: it is the most a chunked encoder that
// gets these right can save over the pieces.
//
// PSNR-Y is on BT.601 luma from 8-bit RGB, the frames decoded by
// nvdrv_decode against the source frames, in name order.
import { execFile } from 'node:child_process';
import { mkdtempSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, symlinkSync, copyFileSync } from 'node:fs';
import { cpus, tmpdir } from 'node:os';
import { join, dirname, resolve, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const argv = process.argv.slice(2);
const dash = argv.indexOf('--');
const encArgs = dash >= 0 ? argv.slice(dash + 1) : [];
const own = dash >= 0 ? argv.slice(0, dash) : argv;
const opt = (name, def) => { const i = own.indexOf(name); return i >= 0 ? Number(own[i + 1]) : def; };
const src = own[0] && resolve(own[0]);
if (!src) {
    console.error('usage: node scripts/analysis/gop_parallel.mjs <frames-dir> [--jobs N] [--chunk 48] [--limit N] [-- encoder args]');
    process.exit(2);
}
const cores = cpus().length;
const jobs = opt('--jobs', cores), chunk = opt('--chunk', 48), limit = opt('--limit', 0);
const exe = (name) => join(root, process.platform === 'win32' ? `${name}.exe` : name);

const IMG = new Set(['.png', '.jpg', '.jpeg', '.ppm', '.bmp', '.tga']);
let frames = readdirSync(src).filter(n => IMG.has(extname(n).toLowerCase())).sort();
if (limit > 0) frames = frames.slice(0, limit);
if (!frames.length) { console.error(`no frames in ${src}`); process.exit(1); }

const work = mkdtempSync(join(tmpdir(), 'gop_parallel_'));

function run(cmd, args, env) {
    return new Promise((ok, fail) => {
        execFile(cmd, args, { env: { ...process.env, ...env }, maxBuffer: 64 << 20 }, (err, stdout, stderr) => {
            if (err) fail(new Error(`${cmd} ${args.join(' ')}: ${(stderr || err.message).trim()}`));
            else ok(stdout);
        });
    });
}

/* A directory of the given frames, linked rather than copied where the
 * system allows it. */
function frameDir(name, list) {
    const dir = join(work, name);
    mkdirSync(dir);
    for (const f of list) {
        try { symlinkSync(join(src, f), join(dir, f)); } catch { copyFileSync(join(src, f), join(dir, f)); }
    }
    return dir;
}

function readPPM(file) {
    const b = readFileSync(file);
    let at = 0;
    const token = () => {
        for (;;) {
            while (b[at] === 0x20 || b[at] === 0x0a || b[at] === 0x0d || b[at] === 0x09) at++;
            if (b[at] !== 0x23) break;
            while (b[at] !== 0x0a) at++;
        }
        const s = at;
        while (b[at] > 0x20) at++;
        return b.toString('latin1', s, at);
    };
    if (token() !== 'P6') throw new Error(`${file}: not a binary PPM`);
    const w = Number(token()), h = Number(token());
    token(); at++;
    return { w, h, rgb: b.subarray(at, at + w * h * 3) };
}

/* Luma squared error of every decoded frame against its source. The
 * sources go through the same reader, which is why they must be PPM. */
function lumaSse(decodedDir, sources) {
    const out = readdirSync(decodedDir).filter(n => n.endsWith('.ppm')).sort();
    if (out.length !== sources.length) throw new Error(`${decodedDir}: ${out.length} frames for ${sources.length}`);
    return out.map((n, i) => {
        const a = readPPM(join(decodedDir, n)), b = readPPM(join(src, sources[i]));
        let sse = 0;
        for (let p = 0; p < a.w * a.h * 3; p += 3) {
            const ya = 0.299 * a.rgb[p] + 0.587 * a.rgb[p + 1] + 0.114 * a.rgb[p + 2];
            const yb = 0.299 * b.rgb[p] + 0.587 * b.rgb[p + 1] + 0.114 * b.rgb[p + 2];
            sse += (ya - yb) * (ya - yb);
        }
        return { sse, px: a.w * a.h };
    });
}

const psnr = (mse) => 10 * Math.log10(255 * 255 / mse);
function summary(errs) {
    const mean = errs.reduce((s, e) => s + psnr(e.sse / e.px), 0) / errs.length;
    const pooled = psnr(errs.reduce((s, e) => s + e.sse, 0) / errs.reduce((s, e) => s + e.px, 0));
    return { mean, pooled };
}

async function encodeAndScore(name, list, threads) {
    const dir = frameDir(name, list), out = join(work, `${name}.nvdrv`), dec = join(work, `${name}_dec`);
    await run(exe('nvdrv_encode'), [dir, out, ...encArgs], { OMP_NUM_THREADS: String(threads) });
    mkdirSync(dec);
    await run(exe('nvdrv_decode'), [out, '--out', dec, '--ppm']);
    return { bytes: statSync(out).size, errs: lumaSse(dec, list) };
}

if (frames.some(f => extname(f).toLowerCase() !== '.ppm')) {
    console.error('the source frames must be PPM (scripts/analysis/mkclips.mjs writes them)');
    process.exit(2);
}

try {
    console.log(`${frames.length} frames, ${cores} cores; pieces of ${chunk}, ${jobs} at a time` +
                (encArgs.length ? `; encoder: ${encArgs.join(' ')}` : ''));

    let t = Date.now();
    const whole = await encodeAndScore('whole', frames, cores);
    const wholeTime = (Date.now() - t) / 1000;

    const pieces = [];
    for (let i = 0; i < frames.length; i += chunk) pieces.push(frames.slice(i, i + chunk));
    const threads = Math.max(1, Math.floor(cores / Math.min(jobs, pieces.length)));
    t = Date.now();
    const results = new Array(pieces.length);
    let next = 0;
    await Promise.all(Array.from({ length: Math.min(jobs, pieces.length) }, async () => {
        while (next < pieces.length) {
            const k = next++;
            results[k] = await encodeAndScore(`piece${String(k).padStart(3, '0')}`, pieces[k], threads);
        }
    }));
    const piecesTime = (Date.now() - t) / 1000;

    const a = summary(whole.errs);
    const bytes = results.reduce((s, r) => s + r.bytes, 0);
    const b = summary(results.flatMap(r => r.errs));
    const row = (label, secs, n, s) =>
        console.log(`  ${label.padEnd(30)} ${secs.toFixed(1).padStart(7)} s ${String(n).padStart(10)} B` +
                    `   PSNR-Y ${s.mean.toFixed(2)} dB (clip ${s.pooled.toFixed(2)})`);
    console.log('                            tempo        bytes');
    row('inteiro', wholeTime, whole.bytes, a);
    row(`${pieces.length} pedacos, ${threads} thread(s) cada`, piecesTime, bytes, b);
    console.log(`\n  ${(wholeTime / piecesTime).toFixed(2)}x mais rapido, ` +
                `${(100 * (bytes / whole.bytes - 1)).toFixed(1)}% de bytes, ` +
                `${(b.mean - a.mean).toFixed(2)} dB`);
} finally {
    rmSync(work, { recursive: true, force: true });
}
