/*
 * NVDR's sequences against the best encoders there are, run offline at
 * their slow settings: libaom's AV1, x265's HEVC, x264's H.264 and
 * libvpx's VP9, through ffmpeg. bench_video.mjs measures what a browser
 * can encode in real time; this is the comparison with the state of the
 * art, which is the harder one.
 *
 *   node scripts/bench_video_offline.mjs <frames-dir> [--fps 24] [--ffmpeg PATH]
 *                                        [--only av1,hevc,h264,vp9]
 *
 * Frames go in as I420 made here (BT.601, full range, colour averaged
 * 2x2) and come back as I420, scored the same way as in bench_video.mjs,
 * so the numbers are the codecs' alone. Every encoder gets the same GOP as
 * NVDRV (one intra frame every 48) and is tuned for PSNR where it has
 * such a setting (x264 and x265 otherwise trade PSNR for psychovisual
 * detail). Sizes are the bitstream: IVF's 32-byte header and 12 bytes a
 * frame are taken off. An ffmpeg with these encoders comes with
 * `pip install imageio-ffmpeg`; the path defaults to that one's.
 */
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync, readFileSync, statSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { loadFrames, runNvdr, toI420, scoreI420, fmtPoint, fmtBd } from './video_bench_lib.mjs';

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const dir = args.find((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--')));
const fps = Number(opt('--fps', 24));
if (!dir) {
    console.error('usage: node scripts/bench_video_offline.mjs <frames-dir> [--fps 24] [--ffmpeg PATH] [--only av1,hevc,h264,vp9]');
    process.exit(2);
}

function findFfmpeg() {
    const given = opt('--ffmpeg', process.env.NVDR_FFMPEG);
    if (given) return given;
    try {
        return execFileSync('python3', ['-c', 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'])
            .toString().trim();
    } catch { return 'ffmpeg'; }
}
const FFMPEG = findFfmpeg();

const frames = loadFrames(dir);
const { width, height } = frames[0];
const seconds = frames.length / fps;
const gop = 48;

/* Slow presets, the same GOP, PSNR tuning where there is one. */
const CODECS = {
    av1: { label: 'AV1 (libaom, cpu-used 3)', ext: 'ivf', settings: [30, 36, 42, 48, 54, 60],
           args: c => ['-c:v', 'libaom-av1', '-crf', String(c), '-b:v', '0', '-cpu-used', '3',
                       '-row-mt', '1', '-g', String(gop), '-keyint_min', String(gop), '-f', 'ivf'] },
    hevc: { label: 'HEVC (x265, slow)', ext: 'hevc', settings: [22, 26, 30, 34, 38, 42],
            args: c => ['-c:v', 'libx265', '-preset', 'slow', '-tune', 'psnr', '-crf', String(c),
                        '-x265-params', `keyint=${gop}:min-keyint=${gop}:log-level=error`, '-f', 'hevc'] },
    h264: { label: 'H.264 (x264, veryslow)', ext: '264', settings: [22, 26, 30, 34, 38, 42],
            args: c => ['-c:v', 'libx264', '-preset', 'veryslow', '-tune', 'psnr', '-crf', String(c),
                        '-g', String(gop), '-keyint_min', String(gop), '-f', 'h264'] },
    vp9: { label: 'VP9 (libvpx, good, cpu-used 1)', ext: 'ivf', settings: [30, 36, 42, 48, 54, 60],
           args: c => ['-c:v', 'libvpx-vp9', '-deadline', 'good', '-cpu-used', '1', '-crf', String(c),
                       '-b:v', '0', '-row-mt', '1', '-g', String(gop), '-f', 'ivf'] },
};
const only = opt('--only', 'av1,hevc,h264,vp9').split(',');

const tmp = mkdtempSync(join(tmpdir(), 'nvdrv-offline-'));
const input = join(tmp, 'in.yuv');
writeFileSync(input, Buffer.concat(frames.map(f => Buffer.from(toI420(f)))));
const frameBytes = width * height + 2 * (width >> 1) * (height >> 1);

function runOffline(codec, setting) {
    const out = join(tmp, `out.${codec.ext}`), back = join(tmp, 'back.yuv');
    rmSync(out, { force: true }); rmSync(back, { force: true });
    execFileSync(FFMPEG, ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
                          '-s', `${width}x${height}`, '-r', String(fps), '-i', input, ...codec.args(setting), out],
                 { stdio: ['ignore', 'ignore', 'inherit'] });
    execFileSync(FFMPEG, ['-hide_banner', '-loglevel', 'error', '-y', '-i', out,
                          '-f', 'rawvideo', '-pix_fmt', 'yuv420p', back], { stdio: ['ignore', 'ignore', 'inherit'] });
    const raw = readFileSync(back);
    const planes = [];
    for (let k = 0; k < frames.length && (k + 1) * frameBytes <= raw.length; k++)
        planes.push(new Uint8Array(raw.buffer, raw.byteOffset + k * frameBytes, frameBytes));
    if (planes.length !== frames.length) throw new Error(`decoded ${planes.length} of ${frames.length} frames`);
    let bytes = statSync(out).size;
    if (codec.ext === 'ivf') bytes -= 32 + 12 * frames.length;
    return { setting, bytes, kbps: bytes * 8 / seconds / 1000, ...scoreI420(frames, planes) };
}

if (!existsSync(FFMPEG) && FFMPEG !== 'ffmpeg') { console.error(`no ffmpeg at ${FFMPEG}`); process.exit(1); }
console.log(`${frames.length} frames ${width}x${height} at ${fps} fps, ${FFMPEG}\n`);
const nvdr = runNvdr(dir, frames, fps);
console.log('NVDR');
nvdr.forEach(p => console.log(`  q ${String(p.setting).padStart(2)}  ${fmtPoint(p)}`));
for (const name of only) {
    const codec = CODECS[name];
    if (!codec) { console.log(`unknown codec ${name}`); continue; }
    const pts = [];
    for (const c of codec.settings) {
        try { pts.push(runOffline(codec, c)); } catch (e) { console.log(`  ${name} ${c}: ${e.message}`); }
    }
    console.log(codec.label);
    pts.forEach(p => console.log(`  crf ${String(p.setting).padStart(2)}  ${fmtPoint(p)}`));
    console.log(fmtBd(name.toUpperCase(), pts, nvdr));
}
rmSync(tmp, { recursive: true, force: true });
