// node scripts/analysis/mkclips.mjs <outdir> [frames]
//
// The three test sequences the sequence measurements use, rebuilt from a
// sample photo so anyone can reproduce them (they are not committed, and
// a scratch directory does not survive). Each is 960x540, cropped from
// samples/montanha_pessoas.jpg scaled up to 1920 wide, with bilinear
// resampling so the motion is subpixel:
//
//   clean   a camera panning and zooming in, and one patch of the photo
//           moving across it against the camera
//   noisy   the same with sensor-like noise, new in every frame
//   multi   a camera nearly still and three patches moving their own
//           ways, with light noise
//
// Synthetic, so optimistic in the ways frames.c lists: no light changes,
// no content entering that the source does not have, noise that is white.
// Needs ffmpeg to read the JPEG.
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { writePPM } from './addnoise.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const [out, nArg] = process.argv.slice(2);
if (!out) { console.error('usage: node mkclips.mjs <outdir> [frames]'); process.exit(2); }
const N = Number(nArg || 48), W = 960, H = 540;

const SW = 1920;
const raw = execFileSync('ffmpeg', ['-loglevel', 'error', '-i', join(root, 'samples', 'montanha_pessoas.jpg'),
    '-vf', `scale=${SW}:-2:flags=lanczos`, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], { maxBuffer: 1 << 28 });
const SH = raw.length / (SW * 3);

// Deterministic noise (xorshift + Box-Muller), so a clip is the same every time.
let seed = 0x9e3779b9;
const rnd = () => { seed ^= seed << 13; seed >>>= 0; seed ^= seed >>> 17; seed ^= seed << 5; seed >>>= 0; return seed / 4294967296; };
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());

function sample(x, y, c) {
    x = Math.min(Math.max(x, 0), SW - 1.001); y = Math.min(Math.max(y, 0), SH - 1.001);
    const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
    const at = (xx, yy) => raw[(yy * SW + xx) * 3 + c];
    return (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
}

// camera(k): top-left and scale of the window; patches(k): [{sx, sy, dx, dy, w, h}]
// copy the source rectangle at (sx, sy) to frame position (dx, dy).
function clip(name, camera, patches, sigma) {
    const dir = join(out, name);
    mkdirSync(dir, { recursive: true });
    for (let k = 0; k < N; k++) {
        const { x: cx, y: cy, s } = camera(k), rgb = new Uint8Array(W * H * 3);
        for (let y = 0; y < H; y++)
            for (let x = 0; x < W; x++)
                for (let c = 0; c < 3; c++) rgb[(y * W + x) * 3 + c] = sample(cx + x * s, cy + y * s, c);
        for (const p of patches(k))
            for (let y = 0; y < p.h; y++)
                for (let x = 0; x < p.w; x++) {
                    const X = Math.round(p.dx) + x, Y = Math.round(p.dy) + y;
                    if (X < 0 || Y < 0 || X >= W || Y >= H) continue;
                    for (let c = 0; c < 3; c++) rgb[(Y * W + X) * 3 + c] = sample(p.sx + x + (p.dx % 1), p.sy + y + (p.dy % 1), c);
                }
        if (sigma > 0) for (let i = 0; i < rgb.length; i++) rgb[i] = Math.min(255, Math.max(0, Math.round(rgb[i] + sigma * gauss())));
        writePPM(join(dir, `f${String(k).padStart(3, '0')}.ppm`), { width: W, height: H, rgb });
    }
    console.log(`${name}: ${N} frames -> ${dir}`);
}

// Pan right and down at 1.7 and 0.6 px a frame, zooming in 0.3% a frame.
const pan = k => { const s = 1.4 * Math.pow(0.997, k); return { x: 200 + 1.7 * k + (1.4 - s) * W / 2, y: 120 + 0.6 * k + (1.4 - s) * H / 2, s }; };
const one = k => [{ sx: 60, sy: 40, w: 200, h: 140, dx: 700 - 6.3 * k, dy: 300 - 2.1 * k }];
const still = k => ({ x: 400 + 0.25 * k, y: 300, s: 1.0 });
const three = k => [
    { sx: 1500, sy: 100, w: 160, h: 120, dx: 50 + 4.7 * k, dy: 60 + 1.3 * k },
    { sx: 100, sy: 900, w: 140, h: 140, dx: 780 - 3.1 * k, dy: 100 + 2.6 * k },
    { sx: 900, sy: 950, w: 220, h: 100, dx: 300 + 1.2 * k, dy: 420 - 3.4 * k },
];

clip('clean', pan, one, 0);
clip('noisy', pan, one, 4.0);
clip('multi', still, three, 2.5);
