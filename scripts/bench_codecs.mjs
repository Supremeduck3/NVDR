/*
 * NVDR against what a site would serve instead, at equal quality: the
 * browser's own JPEG and WebP encoders (Chromium's canvas.toBlob), and the
 * reference encoders of the newer formats when they are installed:
 *
 *   avif   avifenc (libavif, aom), 4:2:0, speed 4
 *   jxl    cjxl (libjxl), effort 7, over its butteraugli distance
 *   heic   heif-enc (libheif, x265): HEVC intra, as phones store photos
 *
 *   node scripts/bench_codecs.mjs [images...]      (or: make bench)
 *   CODECS=avif,jxl node scripts/bench_codecs.mjs  (a subset)
 *   make bench-photos                               (large camera photos,
 *                                                    scripts/fetch_photos.py)
 *
 * Every codec gets the same pixels: each image is read once the way the
 * encoder reads it (output/convert) and handed to the browser as a
 * lossless PNG. Each codec is run over a range of its quality knob; the
 * decoded pictures are scored against the source here, by one
 * implementation of each metric:
 *
 *   PSNR-Y   luma only, the usual yardstick. Blind to colour, so it
 *            favours codecs that store colour at half resolution
 *            (JPEG and WebP here: 4:2:0).
 *   SSIM-Y   structure on luma, 11x11 Gaussian windows (sigma 1.5).
 *   PSNR-RGB all three channels, colour error included.
 *
 * and the curves compared by Bjontegaard delta rate: how much larger (+)
 * or smaller (-) NVDR's file is than the other codec's over the quality
 * range both cover. SSIM enters the fit as -10 log10(1 - SSIM).
 *
 * With no images given it runs the samples, plus a copy of
 * montanha_pessoas with high-ISO sensor noise added
 * (scripts/analysis/addnoise.mjs). Results go to output/bench/: the
 * curves in results.json, the BD-rate table in results.md.
 */
import { execFileSync, execSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync, statSync, readdirSync } from 'node:fs';
import { basename, extname, join } from 'node:path';
import { readPPM, writePPM, addNoise } from './analysis/addnoise.mjs';

const ROOT = new URL('..', import.meta.url).pathname;
const OUT = join(ROOT, 'output', 'bench');
const SRC = join(OUT, 'src');
mkdirSync(SRC, { recursive: true });

const NVDR_Q = [6, 8, 12, 16, 24, 32, 48, 64, 96];
// Extra encoder options for every NVDR run, e.g. NVDR_ARGS="--chroma 444".
const NVDR_ARGS = (process.env.NVDR_ARGS || '').split(/\s+/).filter(Boolean);
const BROWSER_Q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];
const AVIF_YUV = process.env.AVIF_YUV || '420';

/* The command-line codecs: a quality knob swept over its useful range,
 * encode `png` to `file`, decode `file` to a PNG. Each covers roughly
 * the same range of PSNR as NVDR_Q, from about 30 to 45 dB on luma. */
const CLI = {
    avif: {
        tool: 'avifenc', ext: 'avif', settings: [10, 18, 26, 34, 42, 50, 58, 66, 74, 82],
        encode: (png, file, q) => ['avifenc', ['-j', 'all', '-s', '4', '-y', AVIF_YUV, '-q', String(q), png, file]],
        decode: (file, out) => ['avifdec', ['-j', 'all', file, out]],
    },
    jxl: {
        tool: 'cjxl', ext: 'jxl', settings: [0.4, 0.6, 0.9, 1.3, 1.9, 2.8, 4, 6, 9],
        encode: (png, file, d) => ['cjxl', [png, file, '-d', String(d), '-e', '7']],
        decode: (file, out) => ['djxl', [file, out]],
    },
    heic: {
        tool: 'heif-enc', ext: 'heic', settings: [10, 18, 26, 34, 42, 50, 58, 66, 74],
        encode: (png, file, q) => ['heif-enc', ['-e', 'x265', '-q', String(q), '--no-alpha', '-o', file, png]],
        decode: (file, out) => ['heif-convert', [file, out]],
    },
};
const has = tool => { try { execSync(`command -v ${tool}`, { stdio: 'ignore' }); return true; } catch { return false; } };
const CODECS = (process.env.CODECS || ['jpeg', 'webp', ...Object.keys(CLI).filter(c => has(CLI[c].tool))].join(','))
    .split(',').filter(Boolean);
const NAMES = { jpeg: 'JPEG', webp: 'WebP', avif: 'AVIF', jxl: 'JPEG XL', heic: 'HEIC' };

async function playwright() {
    try { return await import('playwright'); } catch {}
    const global = execSync('npm root -g').toString().trim();
    return import(join(global, 'playwright', 'index.mjs'));
}

/* ------------------------------------------------------------ metrics */

function lumaOf(rgb, n) {
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) y[i] = 0.299 * rgb[3 * i] + 0.587 * rgb[3 * i + 1] + 0.114 * rgb[3 * i + 2];
    return y;
}

const psnr = mse => mse > 0 ? 10 * Math.log10(255 * 255 / mse) : 99;

function blur(src, w, h) {
    const k = [], r = 5;
    let sum = 0;
    for (let i = -r; i <= r; i++) { k.push(Math.exp(-(i * i) / (2 * 1.5 * 1.5))); sum += k[k.length - 1]; }
    for (let i = 0; i < k.length; i++) k[i] /= sum;
    const tmp = new Float64Array(w * h), out = new Float64Array(w * h);
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
            let a = 0;
            for (let i = -r; i <= r; i++) { const xx = Math.min(w - 1, Math.max(0, x + i)); a += k[i + r] * src[y * w + xx]; }
            tmp[y * w + x] = a;
        }
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
            let a = 0;
            for (let i = -r; i <= r; i++) { const yy = Math.min(h - 1, Math.max(0, y + i)); a += k[i + r] * tmp[yy * w + x]; }
            out[y * w + x] = a;
        }
    return out;
}

function ssim(a, b, w, h) {
    const n = w * h, aa = new Float64Array(n), bb = new Float64Array(n), ab = new Float64Array(n);
    for (let i = 0; i < n; i++) { aa[i] = a[i] * a[i]; bb[i] = b[i] * b[i]; ab[i] = a[i] * b[i]; }
    const ma = blur(a, w, h), mb = blur(b, w, h), saa = blur(aa, w, h), sbb = blur(bb, w, h), sab = blur(ab, w, h);
    const c1 = (0.01 * 255) ** 2, c2 = (0.03 * 255) ** 2;
    let total = 0, count = 0;
    // Windows that fit inside the picture only.
    for (let y = 5; y < h - 5; y++)
        for (let x = 5; x < w - 5; x++) {
            const i = y * w + x;
            const va = saa[i] - ma[i] * ma[i], vb = sbb[i] - mb[i] * mb[i], cov = sab[i] - ma[i] * mb[i];
            total += ((2 * ma[i] * mb[i] + c1) * (2 * cov + c2)) /
                     ((ma[i] * ma[i] + mb[i] * mb[i] + c1) * (va + vb + c2));
            count++;
        }
    return count ? total / count : 1;
}

function score(src, got) {
    const n = src.width * src.height;
    let se = 0;
    for (let i = 0; i < 3 * n; i++) { const d = src.rgb[i] - got[i]; se += d * d; }
    const ya = src.luma || (src.luma = lumaOf(src.rgb, n)), yb = lumaOf(got, n);
    let sy = 0;
    for (let i = 0; i < n; i++) { const d = ya[i] - yb[i]; sy += d * d; }
    return { psnrY: psnr(sy / n), psnrRGB: psnr(se / (3 * n)), ssimY: ssim(ya, yb, src.width, src.height) };
}

/* --------------------------------------------------------- BD rate */

// Least-squares cubic through (x, y), as coefficients [c0..c3].
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
        for (let r = 0; r < m; r++) {
            if (r === i) continue;
            const f = A[r][i] / A[i][i];
            for (let c = i; c <= m; c++) A[r][c] -= f * A[i][c];
        }
    }
    return A.map((row, i) => row[m] / row[i]);
}

const integral = (c, lo, hi) => {
    const F = x => c[0] * x + c[1] * x * x / 2 + c[2] * x ** 3 / 3 + c[3] * x ** 4 / 4;
    return F(hi) - F(lo);
};

/* Bjontegaard delta rate of `test` against `ref`, in %: points are
 * { bytes, q } where q is the quality metric in dB. Null when the curves
 * do not overlap enough to compare. */
function bdRate(ref, test) {
    const clean = pts => pts.filter(p => isFinite(p.q) && p.q < 90).sort((a, b) => a.q - b.q);
    const r = clean(ref), t = clean(test);
    if (r.length < 4 || t.length < 4) return null;
    const lo = Math.max(r[0].q, t[0].q), hi = Math.min(r[r.length - 1].q, t[t.length - 1].q);
    if (hi - lo < 0.5) return null;
    const cr = cubicFit(r.map(p => p.q), r.map(p => Math.log(p.bytes)));
    const ct = cubicFit(t.map(p => p.q), t.map(p => Math.log(p.bytes)));
    const diff = (integral(ct, lo, hi) - integral(cr, lo, hi)) / (hi - lo);
    return (Math.exp(diff) - 1) * 100;
}

/* ------------------------------------------------------------ codecs */

/* JPEG sources are cropped off their 8x8 (and 16x16 chroma) grid, so the
 * browser's JPEG does not get to line its blocks up with the ones the
 * source was built from: re-encoding a JPEG on its own grid near its own
 * quality comes back almost exact, and scored 14 dB above its curve. */
function prepare(path) {
    const name = basename(path, extname(path)).normalize('NFD').replace(/[^\x20-\x7e]/g, '');
    const ppm = join(SRC, `${name}.ppm`), png = join(SRC, `${name}.png`);
    const crop = /\.jpe?g$/i.test(path) ? ['3', '5'] : [];
    execFileSync(join(ROOT, 'output', 'convert'), [path, ppm, ...crop]);
    execFileSync(join(ROOT, 'output', 'convert'), [path, png, ...crop]);
    return { name, ppm, png, ...readPPM(ppm) };
}

function runNvdr(img) {
    const points = [];
    const file = join(OUT, 'tmp.nvdr'), dec = join(OUT, 'tmp.ppm');
    let planes = null;
    for (const q of NVDR_Q) {
        const report = execFileSync(join(ROOT, 'nvdr_encode'), [img.ppm, file, '--q', String(q), ...NVDR_ARGS]).toString();
        execFileSync(join(ROOT, 'nvdr_decode'), [file, dec]);
        const bytes = statSync(file).size;
        const m = /planes Y\/Cb\/Cr: ([\d.]+)%\/([\d.]+)%\/([\d.]+)%/.exec(report);
        if (q === 24 && m) planes = { y: +m[1], cb: +m[2], cr: +m[3] };
        points.push({ setting: q, bytes, ...score(img, readPPM(dec).rgb) });
    }
    return { points, planes };
}

function runCli(img, codec) {
    const c = CLI[codec], file = join(OUT, `tmp.${c.ext}`), dec = join(OUT, 'tmp_dec.png'), ppm = join(OUT, 'tmp_dec.ppm');
    return c.settings.map(q => {
        const [enc, encArgs] = c.encode(img.png, file, q);
        execFileSync(enc, encArgs, { stdio: 'ignore' });
        const [d, decArgs] = c.decode(file, dec);
        execFileSync(d, decArgs, { stdio: 'ignore' });
        execFileSync(join(ROOT, 'output', 'convert'), [dec, ppm]);
        const got = readPPM(ppm);
        if (got.width !== img.width || got.height !== img.height)
            throw new Error(`${codec} ${img.name}: decoded ${got.width}x${got.height}`);
        return { setting: q, bytes: statSync(file).size, ...score(img, got.rgb) };
    });
}

async function runBrowser(page, img, type) {
    const b64 = readFileSync(img.png).toString('base64');
    const source = await page.evaluate(async ({ b64 }) => {
        const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const opts = { colorSpaceConversion: 'none', premultiplyAlpha: 'none' };
        const bmp = await createImageBitmap(new Blob([bytes], { type: 'image/png' }), opts);
        const canvas = new OffscreenCanvas(bmp.width, bmp.height);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(bmp, 0, 0);
        self.benchCanvas = canvas;
        self.toB64 = (rgba) => {
            const rgb = new Uint8Array(rgba.length / 4 * 3);
            for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) { rgb[j] = rgba[i]; rgb[j + 1] = rgba[i + 1]; rgb[j + 2] = rgba[i + 2]; }
            let s = '';
            for (let i = 0; i < rgb.length; i += 0x8000) s += String.fromCharCode(...rgb.subarray(i, i + 0x8000));
            return btoa(s);
        };
        return self.toB64(ctx.getImageData(0, 0, bmp.width, bmp.height).data);
    }, { b64 });
    const seen = Buffer.from(source, 'base64');
    if (!seen.equals(Buffer.from(img.rgb))) throw new Error(`${img.name}: the browser does not see the source's pixels`);
    const out = [];
    for (const q of BROWSER_Q) {
        const r = await page.evaluate(async ({ type, q }) => {
            const opts = { colorSpaceConversion: 'none', premultiplyAlpha: 'none' };
            const blob = await self.benchCanvas.convertToBlob({ type, quality: q });
            if (blob.type !== type) return { error: `${type} not supported` };
            const back = await createImageBitmap(blob, opts);
            const c2 = new OffscreenCanvas(back.width, back.height), x2 = c2.getContext('2d');
            x2.drawImage(back, 0, 0);
            return { bytes: blob.size, rgb: self.toB64(x2.getImageData(0, 0, back.width, back.height).data) };
        }, { type, q });
        if (r.error) throw new Error(r.error);
        out.push({ setting: q, bytes: r.bytes, ...score(img, new Uint8Array(Buffer.from(r.rgb, 'base64'))) });
    }
    return out;
}

/* ------------------------------------------------------------- main */

let inputs = process.argv.slice(2);
if (!inputs.length) {
    inputs = readdirSync(join(ROOT, 'samples')).filter(f => /\.(jpe?g|png)$/i.test(f)).map(f => join(ROOT, 'samples', f));
    execFileSync(join(ROOT, 'output', 'convert'), [join(ROOT, 'samples', 'montanha_pessoas.jpg'), join(SRC, 'clean.ppm'), '3', '5']);
    const noisy = join(SRC, 'montanha_ruido.ppm');
    writePPM(noisy, addNoise(readPPM(join(SRC, 'clean.ppm')), 1, 7));
    inputs.push(noisy);
}

const browserCodecs = { jpeg: 'image/jpeg', webp: 'image/webp' };
let browser = null, page = null;
if (CODECS.some(c => browserCodecs[c])) {
    const { chromium } = await playwright();
    browser = await chromium.launch();
    page = await browser.newPage();
}
const all = [];
const fmt = (v, d = 1) => v === null ? 'n/a' : `${v >= 0 ? '+' : ''}${v.toFixed(d)}%`;
const dbs = s => -10 * Math.log10(Math.max(1e-10, 1 - s));
const metrics = [p => p.psnrY, p => dbs(p.ssimY), p => p.psnrRGB];

console.log(`NVDR contra ${CODECS.map(c => NAMES[c]).join(', ')}: BD-rate PSNR-Y / SSIM-Y / PSNR-RGB`);
for (const path of inputs) {
    const img = prepare(path);
    const t0 = Date.now();
    const nvdr = runNvdr(img);
    const curves = { nvdr: nvdr.points }, vs = {};
    for (const codec of CODECS) {
        curves[codec] = browserCodecs[codec] ? await runBrowser(page, img, browserCodecs[codec]) : runCli(img, codec);
        vs[codec] = metrics.map(m => bdRate(curves[codec].map(p => ({ bytes: p.bytes, q: m(p) })),
                                            nvdr.points.map(p => ({ bytes: p.bytes, q: m(p) }))));
    }
    all.push({ name: img.name, width: img.width, height: img.height, planes: nvdr.planes, vs, curves });
    const pl = nvdr.planes ? `  bits Y/Cb/Cr ${nvdr.planes.y.toFixed(0)}/${nvdr.planes.cb.toFixed(0)}/${nvdr.planes.cr.toFixed(0)}` : '';
    console.log(`${img.name} ${img.width}x${img.height} (${((Date.now() - t0) / 1000).toFixed(0)} s)${pl}`);
    for (const codec of CODECS)
        console.log(`  vs ${NAMES[codec].padEnd(8)} ${vs[codec].map(v => fmt(v).padStart(7)).join(' ')}`);
}
if (browser) await browser.close();

const mean = (codec, i) => {
    const v = all.map(r => r.vs[codec][i]).filter(x => x !== null);
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
};
console.log(`média de ${all.length}`);
for (const codec of CODECS)
    console.log(`  vs ${NAMES[codec].padEnd(8)} ${[0, 1, 2].map(i => fmt(mean(codec, i)).padStart(7)).join(' ')}`);
console.log('\n(+ = NVDR precisa de mais bytes para a mesma qualidade; - = menos)');

// The same table as Markdown, one column per codec and metric.
const head = CODECS.flatMap(c => [`${NAMES[c]} PSNR-Y`, 'SSIM-Y', 'PSNR-RGB']);
const md = [`| image | size | ${head.join(' | ')} |`, `|---|---|${head.map(() => '---:').join('|')}|`];
for (const r of all)
    md.push(`| ${r.name} | ${r.width}x${r.height} | ${CODECS.flatMap(c => r.vs[c].map(v => fmt(v))).join(' | ')} |`);
md.push(`| **mean** | | ${CODECS.flatMap(c => [0, 1, 2].map(i => `**${fmt(mean(c, i))}**`)).join(' | ')} |`);
writeFileSync(join(OUT, 'results.md'), md.join('\n') + '\n');
writeFileSync(join(OUT, 'results.json'), JSON.stringify(all, null, 1));
console.log(`curvas em ${join('output', 'bench', 'results.json')}, tabela em ${join('output', 'bench', 'results.md')}`);
