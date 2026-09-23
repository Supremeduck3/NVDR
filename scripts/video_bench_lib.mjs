/*
 * What the video benchmarks share: frames in and scored, NVDRV's own
 * curve, I420 both ways, and the Bjontegaard delta rate.
 */
import { execFileSync } from 'node:child_process';
import { readdirSync, statSync, mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { readPPM } from './analysis/addnoise.mjs';

const ROOT = new URL('..', import.meta.url).pathname;

export function loadFrames(dir) {
    return readdirSync(dir).filter(f => f.endsWith('.ppm')).sort().map(f => readPPM(join(dir, f)));
}

export const psnr = mse => mse > 0 ? 10 * Math.log10(255 * 255 / mse) : 99;
export function score(src, got) {
    let sy = 0, se = 0;
    const n = src.width * src.height;
    for (let i = 0; i < n; i++) {
        const a = src.rgb, b = got;
        const ya = 0.299 * a[3 * i] + 0.587 * a[3 * i + 1] + 0.114 * a[3 * i + 2];
        const yb = 0.299 * b[3 * i] + 0.587 * b[3 * i + 1] + 0.114 * b[3 * i + 2];
        sy += (ya - yb) ** 2;
        for (let c = 0; c < 3; c++) se += (a[3 * i + c] - b[3 * i + c]) ** 2;
    }
    return { y: sy / n, rgb: se / (3 * n) };
}
/* A stream's PSNR is that of the mean squared error over all frames. */
export function streamScore(frames, decoded) {
    let y = 0, rgb = 0;
    decoded.forEach((d, k) => { const s = score(frames[k], d); y += s.y; rgb += s.rgb; });
    return { psnrY: psnr(y / decoded.length), psnrRGB: psnr(rgb / decoded.length) };
}

/* ------------------------------------------------------------ NVDR */

export function runNvdr(dir, frames, fps, qs = [10, 14, 20, 28, 40, 56]) {
    const seconds = frames.length / fps;
    const tmp = mkdtempSync(join(tmpdir(), 'nvdrv-'));
    const points = [];
    for (const q of qs) {
        const file = join(tmp, 'v.nvdrv'), out = join(tmp, 'dec');
        execFileSync(join(ROOT, 'nvdrv_encode'), [dir, file, '--q', String(q), '--fps', String(fps)], { stdio: 'ignore' });
        rmSync(out, { recursive: true, force: true });
        execFileSync('mkdir', ['-p', out]);
        execFileSync(join(ROOT, 'nvdrv_decode'), [file, '--out', out, '--ppm'], { stdio: 'ignore' });
        const decoded = readdirSync(out).sort().map(f => readPPM(join(out, f)).rgb);
        const bytes = statSync(file).size;
        points.push({ setting: q, bytes, kbps: bytes * 8 / seconds / 1000, ...streamScore(frames, decoded) });
    }
    rmSync(tmp, { recursive: true, force: true });
    return points;
}

/*
 * The browser's own RGB to YUV and back (a VideoFrame made from RGBA,
 * drawn to a canvas) loses more than the codecs do: VP8 at 9.7 Mbit/s came
 * back at 36 dB. So frames go in as I420 made here (BT.601, full range,
 * colour averaged 2x2) and come out as their planes: luma is scored on
 * the Y plane itself, and RGB after scaling colour back up here the way
 * NVDR's decoder does.
 */
export function toI420(frame) {
    const { width: w, height: h, rgb } = frame;
    const cw = w >> 1, ch = h >> 1;
    const out = new Uint8Array(w * h + 2 * cw * ch);
    const Y = new Float64Array(w * h), U = new Float64Array(w * h), V = new Float64Array(w * h);
    for (let i = 0; i < w * h; i++) {
        const r = rgb[3 * i], g = rgb[3 * i + 1], b = rgb[3 * i + 2];
        Y[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        U[i] = -0.168736 * r - 0.331264 * g + 0.5 * b + 128;
        V[i] = 0.5 * r - 0.418688 * g - 0.081312 * b + 128;
        out[i] = Math.max(0, Math.min(255, Math.round(Y[i])));
    }
    for (let y = 0; y < ch; y++)
        for (let x = 0; x < cw; x++) {
            const a = 2 * y * w + 2 * x, k = y * cw + x;
            out[w * h + k] = Math.round((U[a] + U[a + 1] + U[a + w] + U[a + w + 1]) / 4);
            out[w * h + cw * ch + k] = Math.round((V[a] + V[a + 1] + V[a + w] + V[a + w + 1]) / 4);
        }
    return out;
}

export function fromI420(buf, w, h) {
    const cw = w >> 1, ch = h >> 1, rgb = new Uint8Array(w * h * 3);
    const up = (plane, x, y) => {
        const cx = x >> 1, cy = y >> 1;
        const ox = Math.min(cw - 1, Math.max(0, (x & 1) ? cx + 1 : cx - 1)), oy = Math.min(ch - 1, Math.max(0, (y & 1) ? cy + 1 : cy - 1));
        return (9 * plane[cy * cw + cx] + 3 * plane[cy * cw + ox] + 3 * plane[oy * cw + cx] + plane[oy * cw + ox]) / 16;
    };
    const U = buf.subarray(w * h, w * h + cw * ch), V = buf.subarray(w * h + cw * ch);
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
            const i = y * w + x, yy = buf[i], cb = up(U, x, y) - 128, cr = up(V, x, y) - 128;
            rgb[3 * i] = Math.max(0, Math.min(255, Math.round(yy + 1.402 * cr)));
            rgb[3 * i + 1] = Math.max(0, Math.min(255, Math.round(yy - 0.344136 * cb - 0.714136 * cr)));
            rgb[3 * i + 2] = Math.max(0, Math.min(255, Math.round(yy + 1.772 * cb)));
        }
    return rgb;
}

/* A decoded I420 stream against the source frames: luma on the Y plane
 * itself, RGB after scaling colour back up the way NVDR's decoder does. */
export function scoreI420(frames, planes) {
    const { width, height } = frames[0];
    let sy = 0, srgb = 0;
    planes.forEach((buf, k) => {
        const src = frames[k], n = width * height;
        for (let i = 0; i < n; i++) {
            const ya = 0.299 * src.rgb[3 * i] + 0.587 * src.rgb[3 * i + 1] + 0.114 * src.rgb[3 * i + 2];
            sy += (ya - buf[i]) ** 2;
        }
        srgb += score(src, fromI420(buf, width, height)).rgb;
    });
    return { psnrY: psnr(sy / (planes.length * width * height)), psnrRGB: psnr(srgb / planes.length) };
}

/* ------------------------------------------------------------ BD rate */

function cubicFit(xs, ys) {
    const m = 4, A = Array.from({ length: m }, () => new Array(m + 1).fill(0));
    for (let k = 0; k < xs.length; k++) {
        const p = [1, xs[k], xs[k] ** 2, xs[k] ** 3];
        for (let i = 0; i < m; i++) { for (let j = 0; j < m; j++) A[i][j] += p[i] * p[j]; A[i][m] += p[i] * ys[k]; }
    }
    for (let i = 0; i < m; i++) {
        let piv = i;
        for (let r = i + 1; r < m; r++) if (Math.abs(A[r][i]) > Math.abs(A[piv][i])) piv = r;
        [A[i], A[piv]] = [A[piv], A[i]];
        for (let r = 0; r < m; r++) { if (r === i) continue; const f = A[r][i] / A[i][i]; for (let c = i; c <= m; c++) A[r][c] -= f * A[i][c]; }
    }
    return A.map((row, i) => row[m] / row[i]);
}
const integral = (c, lo, hi) => { const F = x => c[0] * x + c[1] * x * x / 2 + c[2] * x ** 3 / 3 + c[3] * x ** 4 / 4; return F(hi) - F(lo); };
export function bdRate(ref, test, key) {
    const r = ref.slice().sort((a, b) => a[key] - b[key]), t = test.slice().sort((a, b) => a[key] - b[key]);
    if (r.length < 4 || t.length < 4) return null;
    const lo = Math.max(r[0][key], t[0][key]), hi = Math.min(r[r.length - 1][key], t[t.length - 1][key]);
    if (hi - lo < 0.3) return null;
    const cr = cubicFit(r.map(p => p[key]), r.map(p => Math.log(p.bytes)));
    const ct = cubicFit(t.map(p => p[key]), t.map(p => Math.log(p.bytes)));
    return (Math.exp((integral(ct, lo, hi) - integral(cr, lo, hi)) / (hi - lo)) - 1) * 100;
}

export const fmtPoint = p =>
    `${p.kbps.toFixed(0).padStart(6)} kbit/s  Y ${p.psnrY.toFixed(2)}  RGB ${p.psnrRGB.toFixed(2)}`;
const pct = v => v === null ? 'n/a' : (v >= 0 ? '+' : '') + v.toFixed(1) + '%';
export function fmtBd(label, ref, nvdr) {
    return `  NVDR vs ${label}: ${pct(bdRate(ref, nvdr, 'psnrY'))} on PSNR-Y, ${pct(bdRate(ref, nvdr, 'psnrRGB'))} on PSNR-RGB`;
}
