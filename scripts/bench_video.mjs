/*
 * NVDR's sequences against the video codecs a browser has: VP8, VP9 and
 * AV1 through WebCodecs, on the same frames, at equal PSNR.
 *
 *   node scripts/bench_video.mjs <frames-dir> [--fps 24]
 *
 * The frames are PPMs (scripts/analysis/frames.c makes test clips; the
 * server extracts real ones with ffmpeg). Every codec is run at several
 * rates, every decoded frame is scored against its source (PSNR on luma
 * and on RGB), and the curves are compared by Bjontegaard delta rate on
 * PSNR-Y: how much bigger (+) or smaller (-) NVDR's stream is. WebCodecs'
 * encoders are set up for real time, so an offline VP9 or AV1 encode with
 * more effort would do better than these; they are what a browser can
 * make, which is the comparison a page would face. Frames go in and come
 * out as I420 so the scores measure the codecs, not the browser's colour
 * conversion. RATES=a,b,... runs only the browser codecs at those kbit/s.
 */
import { execFileSync } from 'node:child_process';
import { createServer } from 'node:http';
import { readdirSync, statSync, mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { readPPM } from './analysis/addnoise.mjs';

const ROOT = new URL('..', import.meta.url).pathname;
const args = process.argv.slice(2);
const dir = args.find(a => !a.startsWith('--'));
const fps = Number(args[args.indexOf('--fps') + 1]) || 24;
if (!dir) { console.error('usage: node scripts/bench_video.mjs <frames-dir> [--fps 24]'); process.exit(2); }

const names = readdirSync(dir).filter(f => f.endsWith('.ppm')).sort();
const frames = names.map(f => readPPM(join(dir, f)));
const { width, height } = frames[0];
const seconds = frames.length / fps;

const psnr = mse => mse > 0 ? 10 * Math.log10(255 * 255 / mse) : 99;
function score(src, got) {
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
function streamScore(decoded) {
    let y = 0, rgb = 0;
    decoded.forEach((d, k) => { const s = score(frames[k], d); y += s.y; rgb += s.rgb; });
    return { psnrY: psnr(y / decoded.length), psnrRGB: psnr(rgb / decoded.length) };
}

/* ------------------------------------------------------------ NVDR */

function runNvdr() {
    const tmp = mkdtempSync(join(tmpdir(), 'nvdrv-'));
    const points = [];
    for (const q of [10, 14, 20, 28, 40, 56]) {
        const file = join(tmp, 'v.nvdrv'), out = join(tmp, 'dec');
        execFileSync(join(ROOT, 'nvdrv_encode'), [dir, file, '--q', String(q), '--fps', String(fps)], { stdio: 'ignore' });
        rmSync(out, { recursive: true, force: true });
        execFileSync('mkdir', ['-p', out]);
        execFileSync(join(ROOT, 'nvdrv_decode'), [file, '--out', out, '--ppm'], { stdio: 'ignore' });
        const decoded = readdirSync(out).sort().map(f => readPPM(join(out, f)).rgb);
        const bytes = statSync(file).size;
        points.push({ setting: q, bytes, kbps: bytes * 8 / seconds / 1000, ...streamScore(decoded) });
    }
    rmSync(tmp, { recursive: true, force: true });
    return points;
}

/* ------------------------------------------------------- WebCodecs */

async function playwright() {
    try { return await import('playwright'); } catch {}
    const global = execFileSync('npm', ['root', '-g']).toString().trim();
    return import(join(global, 'playwright', 'index.mjs'));
}

/*
 * The browser's own RGB to YUV and back (a VideoFrame made from RGBA,
 * drawn to a canvas) loses more than the codecs do: VP8 at 9.7 Mbit/s came
 * back at 36 dB. So frames go in as I420 made here (BT.601, full range,
 * colour averaged 2x2) and come out as their planes: luma is scored on
 * the Y plane itself, and RGB after scaling colour back up here the way
 * NVDR's decoder does.
 */
function toI420(frame) {
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

function fromI420(buf, w, h) {
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

async function runBrowser(page, codec, kbps) {
    await page.evaluate(({ codec, width, height, fps, kbps }) => {
        window.__chunks = []; window.__decoded = []; window.__bytes = 0; window.__err = null;
        window.__enc = new VideoEncoder({
            output: (chunk) => { const b = new Uint8Array(chunk.byteLength); chunk.copyTo(b); window.__bytes += b.length;
                                 window.__chunks.push(new EncodedVideoChunk({ type: chunk.type, timestamp: chunk.timestamp, data: b })); },
            error: e => { window.__err = e.message; }
        });
        window.__enc.configure({ codec, width, height, bitrate: kbps * 1000, framerate: fps, latencyMode: 'quality', bitrateMode: 'variable' });
    }, { codec, width, height, fps, kbps });
    for (let k = 0; k < frames.length; k++) {
        const i420 = Buffer.from(toI420(frames[k]));
        await page.evaluate(({ b64, k, width, height, fps }) => {
            const data = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
            const f = new VideoFrame(data, { format: 'I420', codedWidth: width, codedHeight: height,
                                             timestamp: Math.round(k * 1e6 / fps),
                                             colorSpace: { matrix: 'bt470bg', primaries: 'bt470bg', transfer: 'bt709', fullRange: true } });
            window.__enc.encode(f, { keyFrame: k % 48 === 0 });
            f.close();
        }, { b64: i420.toString('base64'), k, width, height, fps });
    }
    const info = await page.evaluate(async ({ codec, width, height }) => {
        await window.__enc.flush();
        const pending = [];
        const dec = new VideoDecoder({
            // copyTo() only converts to RGB, so take the decoder's own layout
            // and repack it as tight I420 (NV12 interleaves the chroma).
            output: f => { const raw = new Uint8Array(f.allocationSize()), fmt = f.format;
                           pending.push(f.copyTo(raw).then(layout => {
                               const w = f.visibleRect.width, h = f.visibleRect.height, cw = (w + 1) >> 1, ch = (h + 1) >> 1;
                               const b = new Uint8Array(w * h + 2 * cw * ch);
                               for (let y = 0; y < h; y++) b.set(raw.subarray(layout[0].offset + y * layout[0].stride, layout[0].offset + y * layout[0].stride + w), y * w);
                               const U = w * h, V = U + cw * ch;
                               for (let y = 0; y < ch; y++)
                                   for (let x = 0; x < cw; x++) {
                                       if (fmt === 'NV12') {
                                           const o = layout[1].offset + y * layout[1].stride + 2 * x;
                                           b[U + y * cw + x] = raw[o]; b[V + y * cw + x] = raw[o + 1];
                                       } else {
                                           b[U + y * cw + x] = raw[layout[1].offset + y * layout[1].stride + x];
                                           b[V + y * cw + x] = raw[layout[2].offset + y * layout[2].stride + x];
                                       }
                                   }
                               window.__formats = (window.__formats || new Set()).add(fmt);
                               window.__decoded.push(b); f.close(); })); },
            error: e => { window.__err = e.message; }
        });
        dec.configure({ codec, codedWidth: width, codedHeight: height });
        for (const c of window.__chunks) dec.decode(c);
        await dec.flush();
        await Promise.all(pending);
        return { bytes: window.__bytes, frames: window.__decoded.length, err: window.__err || null };
    }, { codec, width, height });
    if (info.err) throw new Error(info.err);
    let sy = 0, srgb = 0;
    for (let k = 0; k < info.frames; k++) {
        const b64 = await page.evaluate(k => {
            const a = window.__decoded[k]; let s = '';
            for (let i = 0; i < a.length; i += 0x8000) s += String.fromCharCode(...a.subarray(i, i + 0x8000));
            return btoa(s);
        }, k);
        const buf = new Uint8Array(Buffer.from(b64, 'base64'));
        const src = frames[k], n = width * height;
        for (let i = 0; i < n; i++) {
            const ya = 0.299 * src.rgb[3 * i] + 0.587 * src.rgb[3 * i + 1] + 0.114 * src.rgb[3 * i + 2];
            sy += (ya - buf[i]) ** 2;
        }
        srgb += score(src, fromI420(buf, width, height)).rgb;
    }
    return { bytes: info.bytes, kbps: info.bytes * 8 / seconds / 1000,
             psnrY: psnr(sy / (info.frames * width * height)), psnrRGB: psnr(srgb / info.frames) };
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
function bdRate(ref, test, key) {
    const r = ref.slice().sort((a, b) => a[key] - b[key]), t = test.slice().sort((a, b) => a[key] - b[key]);
    if (r.length < 4 || t.length < 4) return null;
    const lo = Math.max(r[0][key], t[0][key]), hi = Math.min(r[r.length - 1][key], t[t.length - 1][key]);
    if (hi - lo < 0.3) return null;
    const cr = cubicFit(r.map(p => p[key]), r.map(p => Math.log(p.bytes)));
    const ct = cubicFit(t.map(p => p[key]), t.map(p => Math.log(p.bytes)));
    return (Math.exp((integral(ct, lo, hi) - integral(cr, lo, hi)) / (hi - lo)) - 1) * 100;
}

/* ------------------------------------------------------------- main */

const server = createServer((req, res) => { res.writeHead(200, { 'Content-Type': 'text/html' }); res.end('<!doctype html><title>bench</title>'); });
await new Promise(r => server.listen(0, '127.0.0.1', r));
const { chromium } = await playwright();
const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${server.address().port}/`);   // WebCodecs needs a secure context

const fmt = p => `${p.kbps.toFixed(0).padStart(6)} kbit/s  Y ${p.psnrY.toFixed(2)}  RGB ${p.psnrRGB.toFixed(2)}`;
console.log(`${frames.length} frames ${width}x${height} at ${fps} fps\n`);
const nvdr = process.env.RATES ? [] : runNvdr();
console.log('NVDR');
nvdr.forEach(p => console.log(`  q ${String(p.setting).padStart(2)}  ${fmt(p)}`));
const rates = process.env.RATES ? process.env.RATES.split(',').map(Number) : nvdr.map(p => Math.max(30, Math.round(p.kbps)));
for (const [label, codec] of [['VP8', 'vp8'], ['VP9', 'vp09.00.10.08'], ['AV1', 'av01.0.04M.08']]) {
    const pts = [];
    for (const kbps of rates) {
        try { pts.push(await runBrowser(page, codec, kbps)); } catch (e) { console.log(`  ${label} ${kbps}: ${e.message}`); }
    }
    console.log(label);
    pts.forEach(p => console.log(`  ${fmt(p)}`));
    const bd = bdRate(pts, nvdr, 'psnrY'), bdr = bdRate(pts, nvdr, 'psnrRGB');
    console.log(`  NVDR vs ${label}: ${bd === null ? 'n/a' : (bd >= 0 ? '+' : '') + bd.toFixed(1) + '%'} on PSNR-Y, ` +
                `${bdr === null ? 'n/a' : (bdr >= 0 ? '+' : '') + bdr.toFixed(1) + '%'} on PSNR-RGB`);
}
await browser.close();
server.close();
