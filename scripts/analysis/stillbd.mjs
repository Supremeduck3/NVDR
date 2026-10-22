// node scripts/analysis/stillbd.mjs <out.json> [--bin DIR] [--qs 12,16,20,28,40] [--images a.jpg,b.jpg] [-- encoder args...]
// node scripts/analysis/stillbd.mjs --compare <base.json> <test.json>
//
// The still-picture counterpart of seqbd.mjs: each image (the samples by
// default) encoded at several q with nvdr_encode and decoded, PSNR-Y
// (BT.601 luma of the RGB) and PSNR-RGB kept per image with the bytes and
// the encoding time; --compare gives each image's BD-rate on both and the
// time ratio.
import { execFileSync } from 'node:child_process';
import { readdirSync, statSync, writeFileSync, readFileSync, mkdtempSync, rmSync } from 'node:fs';
import { join, dirname, basename } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { readPPM } from './addnoise.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const args = process.argv.slice(2);

function fit(xs, ys) {
    const m = 4, A = Array.from({ length: m }, () => new Array(m + 1).fill(0));
    for (let k = 0; k < xs.length; k++) {
        const p = [1, xs[k], xs[k] ** 2, xs[k] ** 3];
        for (let i = 0; i < m; i++) { for (let j = 0; j < m; j++) A[i][j] += p[i] * p[j]; A[i][m] += p[i] * ys[k]; }
    }
    for (let i = 0; i < m; i++) {
        let pv = i;
        for (let r = i + 1; r < m; r++) if (Math.abs(A[r][i]) > Math.abs(A[pv][i])) pv = r;
        [A[i], A[pv]] = [A[pv], A[i]];
        for (let r = 0; r < m; r++) { if (r === i) continue; const f = A[r][i] / A[i][i]; for (let c = i; c <= m; c++) A[r][c] -= f * A[i][c]; }
    }
    return A.map((r, i) => r[m] / r[i]);
}
const integral = (c, a, b) => { const F = x => c[0] * x + c[1] * x * x / 2 + c[2] * x ** 3 / 3 + c[3] * x ** 4 / 4; return F(b) - F(a); };
function bd(a, b, key) {
    const lo = Math.max(Math.min(...a.map(p => p[key])), Math.min(...b.map(p => p[key])));
    const hi = Math.min(Math.max(...a.map(p => p[key])), Math.max(...b.map(p => p[key])));
    const ca = fit(a.map(p => p[key]), a.map(p => Math.log(p.bytes))), cb = fit(b.map(p => p[key]), b.map(p => Math.log(p.bytes)));
    return (Math.exp((integral(cb, lo, hi) - integral(ca, lo, hi)) / (hi - lo)) - 1) * 100;
}

if (args[0] === '--compare') {
    const a = JSON.parse(readFileSync(args[1])), b = JSON.parse(readFileSync(args[2]));
    let sy = 0, sr = 0, ta = 0, tb = 0, n = 0;
    for (const k of Object.keys(a)) {
        if (!b[k]) continue;
        const y = bd(a[k], b[k], 'y'), r = bd(a[k], b[k], 'rgb');
        const t1 = a[k].reduce((s, p) => s + p.sec, 0), t2 = b[k].reduce((s, p) => s + p.sec, 0);
        console.log(`${k.padEnd(28)} Y ${y.toFixed(1).padStart(6)}%  RGB ${r.toFixed(1).padStart(6)}%  time x${(t2 / t1).toFixed(2)}`);
        sy += y; sr += r; ta += t1; tb += t2; n++;
    }
    console.log(`${'mean'.padEnd(28)} Y ${(sy / n).toFixed(1).padStart(6)}%  RGB ${(sr / n).toFixed(1).padStart(6)}%  time x${(tb / ta).toFixed(2)}`);
    process.exit(0);
}

const out = args[0];
const end = args.includes('--') ? args.indexOf('--') : args.length;
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 && i < end ? args[i + 1] : d; };
const bin = opt('--bin', root), qs = opt('--qs', '12,16,20,28,40').split(',').map(Number);
const extra = args.slice(end + 1);
const images = opt('--images', '') ? opt('--images', '').split(',')
    : readdirSync(join(root, 'samples')).filter(f => /\.(jpg|png|ppm)$/i.test(f)).map(f => join(root, 'samples', f));
const tmp = mkdtempSync(join(tmpdir(), 'stillbd-'));
const psnr = m => 10 * Math.log10(65025 / m);
const res = {};
for (const img of images) {
    /* The encoder and the score read the same pixels: the image as PPM. */
    const ref = join(tmp, 'ref.ppm');
    execFileSync('ffmpeg', ['-loglevel', 'error', '-y', '-i', img, ref]);
    const s = readPPM(ref).rgb;
    res[basename(img)] = qs.map(q => {
        const f = join(tmp, 'v.nvdr'), d = join(tmp, 'v.ppm'), t0 = process.hrtime.bigint();
        execFileSync(join(bin, 'nvdr_encode'), [ref, f, '--q', String(q), '--quiet', ...extra]);
        const sec = Number(process.hrtime.bigint() - t0) / 1e9;
        execFileSync(join(bin, 'nvdr_decode'), [f, d]);
        const r = readPPM(d).rgb;
        let ey = 0, er = 0;
        for (let i = 0; i < s.length; i += 3) ey += (.299 * (s[i] - r[i]) + .587 * (s[i + 1] - r[i + 1]) + .114 * (s[i + 2] - r[i + 2])) ** 2;
        for (let i = 0; i < s.length; i++) er += (s[i] - r[i]) ** 2;
        return { q, bytes: statSync(f).size, y: psnr(ey / (s.length / 3)), rgb: psnr(er / s.length), sec };
    });
    process.stderr.write(`${basename(img)} done\n`);
}
rmSync(tmp, { recursive: true, force: true });
writeFileSync(out, JSON.stringify(res));
