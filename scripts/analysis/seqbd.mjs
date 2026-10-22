// node scripts/analysis/seqbd.mjs <clipdir> <out.json> [--bin DIR] [--frames N] [--qs 14,20,28,40] [-- encoder args...]
// node scripts/analysis/seqbd.mjs --compare <base.json> <test.json>
//
// The measurement behind every "BD-rate over N frames" in the README's
// sequence sections: encode a clip at several q with nvdrv_encode, decode
// it, score PSNR-Y (BT.601 luma of the RGB) and PSNR-RGB over all frames,
// and keep bytes and scores in a JSON; --compare gives the BD-rate of one
// such run against another, on both.
import { execFileSync } from 'node:child_process';
import { readdirSync, statSync, writeFileSync, readFileSync, rmSync, mkdirSync, mkdtempSync } from 'node:fs';
import { join, dirname } from 'node:path';
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
    console.log(`Y ${bd(a, b, 'y').toFixed(1)}%  RGB ${bd(a, b, 'rgb').toFixed(1)}%`);
    process.exit(0);
}

const [dir, out] = args;
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 && i < args.indexOf('--') >>> 0 ? args[i + 1] : d; };
const bin = opt('--bin', root), frames = Number(opt('--frames', 0)), qs = opt('--qs', '12,17,24,34,48').split(',').map(Number);
const extra = args.includes('--') ? args.slice(args.indexOf('--') + 1) : [];
const names = readdirSync(dir).filter(f => /\.(ppm|png)$/.test(f)).sort();
const src = (frames ? names.slice(0, frames) : names).map(f => readPPM(join(dir, f)).rgb);
const tmp = mkdtempSync(join(tmpdir(), 'seqbd-'));
const psnr = m => 10 * Math.log10(65025 / m);
const pts = [];
for (const q of qs) {
    const f = join(tmp, 'v.nvdrv'), dec = join(tmp, 'dec');
    execFileSync(join(bin, 'nvdrv_encode'), [dir, f, '--q', String(q), ...(frames ? ['--limit', String(frames)] : []), ...extra], { stdio: 'ignore' });
    rmSync(dec, { recursive: true, force: true }); mkdirSync(dec);
    execFileSync(join(bin, 'nvdrv_decode'), [f, '--out', dec, '--ppm'], { stdio: 'ignore' });
    let ey = 0, er = 0;
    const got = readdirSync(dec).sort();
    got.forEach((n, k) => {
        const d = readPPM(join(dec, n)).rgb, s = src[k];
        let e = 0, r = 0;
        for (let i = 0; i < s.length; i += 3) e += (.299 * (s[i] - d[i]) + .587 * (s[i + 1] - d[i + 1]) + .114 * (s[i + 2] - d[i + 2])) ** 2;
        for (let i = 0; i < s.length; i++) r += (s[i] - d[i]) ** 2;
        ey += e / (s.length / 3); er += r / s.length;
    });
    pts.push({ q, bytes: statSync(f).size, y: psnr(ey / got.length), rgb: psnr(er / got.length) });
}
rmSync(tmp, { recursive: true, force: true });
writeFileSync(out, JSON.stringify(pts));
for (const p of pts) console.log(`q${p.q}\t${p.bytes}\tY ${p.y.toFixed(2)}  RGB ${p.rgb.toFixed(2)}`);
