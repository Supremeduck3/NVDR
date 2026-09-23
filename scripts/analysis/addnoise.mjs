/*
 * node scripts/analysis/addnoise.mjs <in.ppm> <out.ppm> [strength] [seed]
 *
 * A clean photo made to look shot at high ISO: sensor noise whose spread
 * grows with the light (shot noise, sigma^2 = a * signal + b), mostly
 * shared by the three channels, some in colour, lightly correlated
 * between neighbours as a demosaiced sensor's is. The codec benchmark
 * and the grain work use it because it keeps the clean picture: what a
 * grain-aware codec should give back is known exactly.
 */
import { readFileSync, writeFileSync } from 'node:fs';

export function readPPM(path) {
    const buf = readFileSync(path);
    let at = 0;
    const fields = [];
    while (fields.length < 4) {
        while (buf[at] === 0x20 || buf[at] === 0x0a || buf[at] === 0x0d || buf[at] === 0x09) at++;
        const start = at;
        while (at < buf.length && ![0x20, 0x0a, 0x0d, 0x09].includes(buf[at])) at++;
        fields.push(buf.toString('ascii', start, at));
    }
    at++;
    const width = +fields[1], height = +fields[2];
    return { width, height, rgb: new Uint8Array(buf.subarray(at, at + width * height * 3)) };
}

export function writePPM(path, { width, height, rgb }) {
    writeFileSync(path, Buffer.concat([Buffer.from(`P6\n${width} ${height}\n255\n`), Buffer.from(rgb)]));
}

export function addNoise({ width, height, rgb }, strength = 1, seed = 1) {
    let s = seed >>> 0 || 1;
    const uniform = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) + 0.5) / 4294967296;
    const gauss = () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
    const n = width * height;
    // White noise, then a light 3-tap blur each way: grain a little over a
    // pixel wide, as demosaicing leaves it.
    const field = () => {
        const f = new Float32Array(n);
        for (let i = 0; i < n; i++) f[i] = gauss();
        const g = new Float32Array(n);
        for (let y = 0; y < height; y++)
            for (let x = 0; x < width; x++) {
                const i = y * width + x;
                g[i] = 0.25 * f[x > 0 ? i - 1 : i] + 0.5 * f[i] + 0.25 * f[x < width - 1 ? i + 1 : i];
            }
        for (let y = 0; y < height; y++)
            for (let x = 0; x < width; x++) {
                const i = y * width + x;
                f[i] = 0.25 * g[y > 0 ? i - width : i] + 0.5 * g[i] + 0.25 * g[y < height - 1 ? i + width : i];
            }
        // The blur takes the spread from 1 to 0.375; give it back.
        for (let i = 0; i < n; i++) f[i] /= 0.375;
        return f;
    };
    const luma = field(), cr = field(), cb = field();
    const out = new Uint8Array(rgb.length);
    for (let i = 0; i < n; i++) {
        const r = rgb[3 * i], g = rgb[3 * i + 1], b = rgb[3 * i + 2];
        const signal = 0.299 * r + 0.587 * g + 0.114 * b;
        const sigma = strength * Math.sqrt(0.18 * signal + 9);   // ~3 in the dark, ~7.5 in the light
        const l = luma[i] * sigma, u = cb[i] * sigma * 0.45, v = cr[i] * sigma * 0.45;
        const c = [r + l + 1.402 * v, g + l - 0.344 * u - 0.714 * v, b + l + 1.772 * u];
        for (let k = 0; k < 3; k++) out[3 * i + k] = Math.max(0, Math.min(255, Math.round(c[k])));
    }
    return { width, height, rgb: out };
}

if (import.meta.url === `file://${process.argv[1]}`) {
    const [, , input, output, strength = '1', seed = '1'] = process.argv;
    if (!input || !output) {
        console.error('usage: node scripts/analysis/addnoise.mjs <in.ppm> <out.ppm> [strength] [seed]');
        process.exit(2);
    }
    writePPM(output, addNoise(readPPM(input), Number(strength), Number(seed)));
}
