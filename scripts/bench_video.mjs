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
import { join } from 'node:path';
import { loadFrames, runNvdr, toI420, scoreI420, fmtPoint, fmtBd } from './video_bench_lib.mjs';
const args = process.argv.slice(2);
const dir = args.find(a => !a.startsWith('--'));
const fps = Number(args[args.indexOf('--fps') + 1]) || 24;
if (!dir) { console.error('usage: node scripts/bench_video.mjs <frames-dir> [--fps 24]'); process.exit(2); }

const frames = loadFrames(dir);
const { width, height } = frames[0];
const seconds = frames.length / fps;

/* ------------------------------------------------------- WebCodecs */

async function playwright() {
    try { return await import('playwright'); } catch {}
    const global = execFileSync('npm', ['root', '-g']).toString().trim();
    return import(join(global, 'playwright', 'index.mjs'));
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
    const planes = [];
    for (let k = 0; k < info.frames; k++) {
        const b64 = await page.evaluate(k => {
            const a = window.__decoded[k]; let s = '';
            for (let i = 0; i < a.length; i += 0x8000) s += String.fromCharCode(...a.subarray(i, i + 0x8000));
            return btoa(s);
        }, k);
        planes.push(new Uint8Array(Buffer.from(b64, 'base64')));
    }
    return { bytes: info.bytes, kbps: info.bytes * 8 / seconds / 1000, ...scoreI420(frames, planes) };
}

/* ------------------------------------------------------------- main */

const server = createServer((req, res) => { res.writeHead(200, { 'Content-Type': 'text/html' }); res.end('<!doctype html><title>bench</title>'); });
await new Promise(r => server.listen(0, '127.0.0.1', r));
const { chromium } = await playwright();
const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${server.address().port}/`);   // WebCodecs needs a secure context

console.log(`${frames.length} frames ${width}x${height} at ${fps} fps\n`);
const nvdr = process.env.RATES ? [] : runNvdr(dir, frames, fps);
console.log('NVDR');
nvdr.forEach(p => console.log(`  q ${String(p.setting).padStart(2)}  ${fmtPoint(p)}`));
const rates = process.env.RATES ? process.env.RATES.split(',').map(Number) : nvdr.map(p => Math.max(30, Math.round(p.kbps)));
for (const [label, codec] of [['VP8', 'vp8'], ['VP9', 'vp09.00.10.08'], ['AV1', 'av01.0.04M.08']]) {
    const pts = [];
    for (const kbps of rates) {
        try { pts.push(await runBrowser(page, codec, kbps)); } catch (e) { console.log(`  ${label} ${kbps}: ${e.message}`); }
    }
    console.log(label);
    pts.forEach(p => console.log(`  ${fmtPoint(p)}`));
    console.log(fmtBd(label, pts, nvdr));
}
await browser.close();
server.close();
