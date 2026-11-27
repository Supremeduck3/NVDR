// node scripts/analysis/stillref.mjs <av1|hevc> <out.json> [--images a.jpg,b.jpg] [--crfs 14,22,30,38,46] [--444]
//
// The reference encoders on still pictures, in the JSON stillbd.mjs
// writes, so that `stillbd.mjs --compare ref.json nvdr.json` gives NVDR's
// BD-rate against them over the same pictures and the same scoring:
//
//   av1    libaom through ffmpeg, as a still picture (-still-picture 1),
//          cpu-used 3, PSNR tuning, constant quality over --crfs
//   hevc   x265 through ffmpeg, every frame intra (keyint 1), preset
//          slow, PSNR tuning
//
// Each gets the picture as 8-bit 4:2:0 (4:4:4 with --444; AV1's profile
// 1, HEVC's range extensions), BT.601 at full range, padded to even
// sizes for 4:2:0 and cropped back, and is decoded back to RGB the same
// way. 4:2:0 halves the colour, which PSNR-Y does not see and PSNR-RGB
// does: NVDR chooses per picture (4:4:4 on a strongly coloured one), so
// both are worth running; PSNR-Y (BT.601 luma of the RGB) and
// PSNR-RGB are scored against the source as stillbd.mjs scores NVDR, so
// the conversion's own error counts against the reference encoder, as
// NVDR's colour error counts against it. Sizes are the bitstream alone:
// IVF's 32-byte header and 12 bytes per frame are taken off for AV1.
import { execFileSync } from 'node:child_process';
import { readdirSync, statSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { join, dirname, basename } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { readPPM } from './addnoise.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const args = process.argv.slice(2);
const [codec, out] = args;
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
if (!['av1', 'hevc'].includes(codec) || !out) {
    console.error('usage: node scripts/analysis/stillref.mjs <av1|hevc> <out.json> [--images a,b] [--crfs 14,22,30,38,46]');
    process.exit(2);
}
const ffmpeg = process.env.NVDR_FFMPEG || 'ffmpeg';
const yuv = args.includes('--444') ? 'yuv444p' : 'yuv420p';
const crfs = opt('--crfs', codec === 'av1' ? '14,22,30,38,46' : '18,23,28,33,38').split(',').map(Number);
const images = opt('--images', '') ? opt('--images', '').split(',')
    : readdirSync(join(root, 'samples')).filter(f => /\.(jpg|png|ppm)$/i.test(f)).map(f => join(root, 'samples', f));
const tmp = mkdtempSync(join(tmpdir(), 'stillref-'));
const psnr = m => 10 * Math.log10(65025 / m);
const TO_YUV = 'scale=out_range=full:out_color_matrix=bt601:flags=accurate_rnd+full_chroma_int';
const TO_RGB = 'scale=in_range=full:in_color_matrix=bt601:flags=accurate_rnd+full_chroma_int';

function encode(ref, f, crf) {
    const pad = yuv === 'yuv420p' ? 'pad=ceil(iw/2)*2:ceil(ih/2)*2,' : '';
    const common = ['-loglevel', 'error', '-y', '-i', ref, '-vf', pad + TO_YUV, '-pix_fmt', yuv, '-color_range', 'pc', '-frames:v', '1'];
    if (codec === 'av1')
        execFileSync(ffmpeg, [...common, '-c:v', 'libaom-av1', '-still-picture', '1', '-cpu-used', '3', '-crf', String(crf),
                              '-b:v', '0', '-aom-params', 'tune=psnr', '-f', 'ivf', f]);
    else
        execFileSync(ffmpeg, [...common, '-c:v', 'libx265', '-preset', 'slow', '-tune', 'psnr', '-crf', String(crf),
                              '-x265-params', 'keyint=1:log-level=error', '-f', 'hevc', f]);
    return statSync(f).size - (codec === 'av1' ? 32 + 12 : 0);
}

const res = {};
for (const img of images) {
    const ref = join(tmp, 'ref.ppm');
    execFileSync(ffmpeg, ['-loglevel', 'error', '-y', '-i', img, ref]);
    const { rgb: s, width, height } = readPPM(ref);
    res[basename(img)] = crfs.map(crf => {
        const f = join(tmp, codec === 'av1' ? 'v.ivf' : 'v.hevc'), d = join(tmp, 'v.ppm');
        const t0 = process.hrtime.bigint();
        const bytes = encode(ref, f, crf);
        const sec = Number(process.hrtime.bigint() - t0) / 1e9;
        execFileSync(ffmpeg, ['-loglevel', 'error', '-y', '-color_range', 'pc', '-i', f, '-vf',
                              `${TO_RGB},crop=${width}:${height}:0:0`, '-pix_fmt', 'rgb24', d]);
        const r = readPPM(d).rgb;
        let ey = 0, er = 0;
        for (let i = 0; i < s.length; i += 3) ey += (.299 * (s[i] - r[i]) + .587 * (s[i + 1] - r[i + 1]) + .114 * (s[i + 2] - r[i + 2])) ** 2;
        for (let i = 0; i < s.length; i++) er += (s[i] - r[i]) ** 2;
        return { q: crf, bytes, y: psnr(ey / (s.length / 3)), rgb: psnr(er / s.length), sec };
    });
    process.stderr.write(`${basename(img)} done\n`);
}
rmSync(tmp, { recursive: true, force: true });
writeFileSync(out, JSON.stringify(res));
