// node scripts/analysis/badnet.mjs <image> [--q 24] [--scenario name,...] [--kbps N --rtt MS --loss P --burst L]
//                                  [--runs 50] [--seed S] [--json out.json] [--svg out.svg]
//                                  [--formats jpeg-prog,avif,...] [--slices 2] [--protect 0.5,0.25,0.1]
//
// Sending a photo over a bad link: what the receiver sees, and when.
//
// Each format is encoded to the same final quality (PSNR-Y of the NVDR
// file at --q, progressive), then cut at growing lengths and decoded,
// which gives its quality for every number of bytes received; a model of
// the link turns bytes into seconds. Two transports:
//
//   with retransmission (TCP-like): packets go out at the link's rate,
//     a lost one is sent again once its loss is noticed, a round trip
//     later, and the receiver can use only the bytes up to the first one
//     still missing;
//   without (UDP-like): what arrived before the first lost packet is all
//     there is, over --runs draws of the losses.
//
// Then NVDR alone, without retransmission but made for it: every packet
// numbered (public/nvdr-fec.js), the receiver decoding the file with its
// holes (nvdr.js's `holes`), with and without slices (--slices rows of
// tiles per slice, default 2) and parity (--protect, the share of parity
// for the colour, the low band and the high band; default 0.5,0.25,0.1),
// against the same parity spread evenly. All of it sent at the link's
// rate, so it is complete when the last packet lands.
//
// Losses come in bursts (Gilbert-Elliott: --loss the mean rate, --burst
// the mean run of lost packets). No slow start, no congestion control,
// fixed 1200-byte packets: the link's shape, not a TCP stack.
//
// Formats: NVDR progressive and directional, JPEG baseline and
// progressive (ImageMagick), WebP, AVIF and JPEG XL (ffmpeg). A cut file
// counts as a picture only if its decoder makes one of it; with the tools
// here only JPEG and NVDR do (JPEG XL can decode progressively, but not
// through ffmpeg). Needs ffmpeg and ImageMagick's convert.
import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, readFileSync, existsSync, rmSync, statSync } from 'node:fs';
import { join, dirname, basename } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { readPPM } from './addnoise.mjs';
import { decodeJs } from '../../public/nvdr.js';
import { packetize, reassemble } from '../../public/nvdr-fec.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const image = args[0];
if (!image || image.startsWith('--')) {
    console.error('usage: node scripts/analysis/badnet.mjs <image> [--q 24] [--scenario sat,2g,3g] ' +
                  '[--kbps N --rtt MS --loss P --burst L] [--runs 50] [--json out.json] [--svg out.svg]');
    process.exit(2);
}
const MTU = 1200;
const SCENARIOS = {
    sat: { name: 'satélite / rádio', kbps: 16, rtt: 1200, loss: 0.05, burst: 3 },
    '2g': { name: '2G ruim', kbps: 40, rtt: 800, loss: 0.03, burst: 2 },
    '3g': { name: '3G instável', kbps: 300, rtt: 300, loss: 0.02, burst: 2 },
};
const scenarios = opt('--kbps') ? [{ name: 'personalizado', kbps: +opt('--kbps'), rtt: +opt('--rtt', 500),
                                     loss: +opt('--loss', 0.02), burst: +opt('--burst', 1) }]
                                : opt('--scenario', 'sat,2g,3g').split(',').map(k => SCENARIOS[k]);
const RUNS = +opt('--runs', 50);
const USEFUL = 25;   /* dB PSNR-Y: a picture someone can make out */
const tmp = mkdtempSync(join(tmpdir(), 'badnet-'));
const run = (cmd, a) => execFileSync(cmd, a, { stdio: ['ignore', 'pipe', 'pipe'] });

/* ------------------------------------------------------------ quality */

const refPath = join(tmp, 'ref.ppm');
run('ffmpeg', ['-loglevel', 'error', '-y', '-i', image, refPath]);
const REF = readPPM(refPath);
function psnrY(img) {
    if (!img || img.width !== REF.width || img.height !== REF.height) return null;
    const a = REF.rgb, b = img.rgb;
    let e = 0;
    for (let i = 0; i < a.length; i += 3)
        e += (.299 * (a[i] - b[i]) + .587 * (a[i + 1] - b[i + 1]) + .114 * (a[i + 2] - b[i + 2])) ** 2;
    return 10 * Math.log10(65025 / (e / (a.length / 3)));
}

/* -------------------------------------------------------------- formats */

// Each: encode(param, out) and decode(file) -> image or null, the param's
// range and which way quality grows.
const ppm = join(tmp, 'dec.ppm');
function decodeWith(cmd, argsOf) {
    return (file) => {
        if (existsSync(ppm)) rmSync(ppm);
        try { run(cmd, argsOf(file)); } catch { /* a cut file may fail, or warn and still write */ }
        if (!existsSync(ppm) || statSync(ppm).size === 0) return null;
        try { return readPPM(ppm); } catch { return null; }
    };
}
const nvdrDecode = decodeWith(join(root, 'nvdr_decode'), f => [f, ppm]);
const magickDecode = decodeWith('convert', f => [f, ppm]);
const ffDecode = decodeWith('ffmpeg', f => ['-loglevel', 'quiet', '-y', '-i', f, '-frames:v', '1', ppm]);
const FORMATS = [
    { key: 'nvdr-prog', name: 'NVDR progressivo', ext: 'nvdr', lo: 4, hi: 200, up: false, int: true,
      encode: (q, o) => run(join(root, 'nvdr_encode'), [refPath, o, '--q', String(q), '--quiet']), decode: nvdrDecode },
    { key: 'nvdr-dir', name: 'NVDR direcional', ext: 'nvdr', lo: 4, hi: 200, up: false, int: true,
      encode: (q, o) => run(join(root, 'nvdr_encode'), [refPath, o, '--q', String(q), '--directional', '--fast', '--quiet']),
      decode: nvdrDecode },
    { key: 'jpeg', name: 'JPEG', ext: 'jpg', lo: 5, hi: 100, up: true, int: true,
      encode: (q, o) => run('convert', [refPath, '-quality', String(q), o]), decode: magickDecode },
    { key: 'jpeg-prog', name: 'JPEG progressivo', ext: 'jpg', lo: 5, hi: 100, up: true, int: true,
      encode: (q, o) => run('convert', [refPath, '-quality', String(q), '-interlace', 'Plane', o]), decode: magickDecode },
    { key: 'webp', name: 'WebP', ext: 'webp', lo: 1, hi: 100, up: true, int: true,
      encode: (q, o) => run('ffmpeg', ['-loglevel', 'error', '-y', '-i', refPath, '-c:v', 'libwebp', '-quality', String(q), o]),
      decode: ffDecode },
    { key: 'avif', name: 'AVIF', ext: 'avif', lo: 0, hi: 63, up: false, int: true,
      encode: (q, o) => run('ffmpeg', ['-loglevel', 'error', '-y', '-i', refPath, '-c:v', 'libaom-av1', '-still-picture', '1',
                                       '-cpu-used', '4', '-crf', String(q), o]), decode: ffDecode },
    { key: 'jxl', name: 'JPEG XL', ext: 'jxl', lo: 0.1, hi: 15, up: false, int: false,
      encode: (q, o) => run('ffmpeg', ['-loglevel', 'error', '-y', '-i', refPath, '-c:v', 'libjxl', '-distance', q.toFixed(3), o]),
      decode: ffDecode },
];

/* The file of format f closest to quality `target` without falling under
 * it by more than a hair, by bisection on its quality knob. */
function encodeAt(f, target) {
    const out = join(tmp, `${f.key}.${f.ext}`);
    const at = (p) => { f.encode(p, out); return { p, q: psnrY(f.decode(out)), bytes: statSync(out).size }; };
    let lo = f.lo, hi = f.hi, best = null;
    for (let it = 0; it < (f.int ? 9 : 12); it++) {
        const mid = f.int ? Math.round((lo + hi) / 2) : (lo + hi) / 2;
        const r = at(mid);
        // `good`: at least the target; then try for fewer bytes.
        const good = r.q !== null && r.q >= target - 0.05;
        if (good && (!best || r.bytes < best.bytes)) best = r;
        if (good === f.up) hi = mid; else lo = mid;   // up: quality grows with the knob
        if (f.int && hi - lo <= 1) break;
    }
    if (!best) best = at(f.up ? f.hi : f.lo);
    f.encode(best.p, out);
    return { ...best, path: out };
}

/* Quality against bytes received: the file cut at geometric lengths. */
function curve(f, file) {
    const bytes = readFileSync(file), n = bytes.length, pts = [];
    const cut = join(tmp, `cut.${f.ext}`);
    for (let k = 48; ; k = Math.ceil(k * 1.18)) {
        const len = Math.min(k, n);
        writeFileSync(cut, bytes.subarray(0, len));
        pts.push({ bytes: len, q: psnrY(f.decode(cut)) });
        if (len === n) break;
    }
    return pts;
}
const qualityAt = (pts, bytes) => { let q = null; for (const p of pts) if (p.bytes <= bytes) q = p.q ?? q; return q; };

/* -------------------------------------------------------------- the link */

function rng(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }
/* Gilbert-Elliott: mean loss `loss`, mean burst `burst` packets. */
function losses(sc, rand) {
    const pBG = 1 / Math.max(1, sc.burst), pGB = sc.loss * pBG / Math.max(1e-9, 1 - sc.loss);
    let bad = false;
    return () => { bad = bad ? rand() >= pBG : rand() < pGB; return bad; };
}
/* With retransmission: when each packet of a `size`-byte file has
 * arrived, in seconds; the bytes up to time t are the packets before the
 * first one still missing. */
function arrivals(size, sc, rand) {
    const n = Math.ceil(size / MTU), tp = MTU * 8 / (sc.kbps * 1000), half = sc.rtt / 2000;
    const lost = losses(sc, rand), done = new Array(n).fill(Infinity), retry = [];
    let t = 0, next = 0;
    while (next < n || retry.length) {
        // A retransmission once its loss is known, else the next new packet.
        retry.sort((a, b) => a.due - b.due);
        let k;
        if (retry.length && (retry[0].due <= t || next >= n)) { const r = retry.shift(); t = Math.max(t, r.due); k = r.k; }
        else k = next++;
        t += tp;
        if (lost()) retry.push({ k, due: t + sc.rtt / 1000 });
        else done[k] = t + half;
    }
    return done;
}
function timeForBytes(done, bytes) {
    const n = Math.ceil(bytes / MTU);
    let t = 0;
    for (let i = 0; i < n; i++) t = Math.max(t, done[i]);
    return t;
}
/* Without: the bytes before the first lost packet. */
function udpPrefix(size, sc, rand) {
    const n = Math.ceil(size / MTU), lost = losses(sc, rand);
    for (let i = 0; i < n; i++) if (lost()) return i * MTU;
    return size;
}

/* ------------------------------------------------------------------ run */

const q = +opt('--q', 24);
/* --formats limits the comparison (NVDR progressive is always in it). */
const only = opt('--formats', '') ? opt('--formats').split(',') : null;
const fmts = FORMATS.filter(f => !only || f.key === 'nvdr-prog' || only.includes(f.key)).map(f => ({ ...f }));
console.error(`${basename(image)}: ${REF.width}x${REF.height}`);
const base = encodeAt.call(null, { ...fmts[0], lo: q, hi: q }, 0);
const target = base.q;
console.error(`alvo: ${target.toFixed(2)} dB PSNR-Y (NVDR progressivo, q ${q}, ${base.bytes} B)`);
const results = [];
for (const f of fmts) {
    let e;
    try { e = f.key === 'nvdr-prog' ? base : encodeAt(f, target); }
    catch (err) { console.error(`${f.name}: não codificou (${err.message.split('\n')[0]})`); continue; }
    const pts = curve(f, e.path);
    results.push({ key: f.key, name: f.name, bytes: e.bytes, q: e.q, pts });
    console.error(`${f.name.padEnd(18)} ${String(e.bytes).padStart(7)} B  ${e.q.toFixed(2)} dB`);
}

const fmtT = t => !isFinite(t) ? '—' : t < 10 ? `${t.toFixed(1)} s` : `${Math.round(t)} s`;
const fmtQ = v => v === null ? 'nada' : `${v.toFixed(1)} dB`;
const report = [];
for (const sc of scenarios) {
    console.log(`\n== ${sc.name}: ${sc.kbps} kbit/s, RTT ${sc.rtt} ms, perda ${(sc.loss * 100).toFixed(0)}% ` +
                `(rajadas de ~${sc.burst}), pacotes de ${MTU} B`);
    console.log('com reenvio (TCP):');
    console.log('  formato              bytes   1ª imagem   útil (25 dB)   a 3 dB do fim   completa    em 5 s     em 15 s');
    const deadlines = [5, 15];
    const scr = { scenario: sc, tcp: [], udp: [] };
    for (const r of results) {
        // The median over the runs, of each figure.
        const firstT = [], usefulT = [], goodT = [], fullT = [], atD = deadlines.map(() => []);
        const firstBytes = (r.pts.find(p => p.q !== null) || {}).bytes;
        // A first picture can be a grey frame with a strip at the top (a
        // cut baseline JPEG): "useful" is the first at 25 dB or more.
        const usefulBytes = (r.pts.find(p => p.q !== null && p.q >= USEFUL) || {}).bytes;
        const goodBytes = (r.pts.find(p => p.q !== null && p.q >= r.q - 3) || {}).bytes;
        for (let s = 0; s < RUNS; s++) {
            const done = arrivals(r.bytes, sc, rng(+opt('--seed', 1) * 7919 + s + 1));
            firstT.push(firstBytes ? timeForBytes(done, firstBytes) : Infinity);
            usefulT.push(usefulBytes ? timeForBytes(done, usefulBytes) : Infinity);
            goodT.push(goodBytes ? timeForBytes(done, goodBytes) : Infinity);
            fullT.push(timeForBytes(done, r.bytes));
            deadlines.forEach((d, i) => {
                let have = 0;
                for (let k = 0; k < done.length && done[k] <= d; k++) have = Math.min(r.bytes, (k + 1) * MTU);
                atD[i].push(qualityAt(r.pts, have));
            });
        }
        const med = a => [...a].sort((x, y) => x - y)[a.length >> 1];
        const medQ = a => { const v = a.map(x => x === null ? -1 : x).sort((x, y) => x - y)[a.length >> 1]; return v < 0 ? null : v; };
        const row = { key: r.key, bytes: r.bytes, first: med(firstT), useful: med(usefulT), good: med(goodT), full: med(fullT),
                      at: deadlines.map((d, i) => medQ(atD[i])) };
        scr.tcp.push(row);
        console.log(`  ${r.name.padEnd(18)} ${String(r.bytes).padStart(7)}   ${fmtT(row.first).padStart(9)}   ${fmtT(row.useful).padStart(12)}   ` +
                    `${fmtT(row.good).padStart(13)}   ${fmtT(row.full).padStart(8)}   ${fmtQ(row.at[0]).padStart(8)}   ${fmtQ(row.at[1]).padStart(8)}`);
    }
    console.log(`sem reenvio (UDP), ${RUNS} sorteios das perdas:`);
    console.log('  formato              chega algo   qualidade (mediana)   chega inteiro');
    for (const r of results) {
        let any = 0, whole = 0;
        const qs = [];
        for (let s = 0; s < RUNS; s++) {
            const have = udpPrefix(r.bytes, sc, rng(+opt('--seed', 1) * 104729 + s + 1));
            const v = qualityAt(r.pts, have);
            if (v !== null) any++;
            if (have >= r.bytes) whole++;
            qs.push(v === null ? -1 : v);
        }
        qs.sort((x, y) => x - y);
        const m = qs[RUNS >> 1];
        scr.udp.push({ key: r.key, any: any / RUNS, whole: whole / RUNS, median: m < 0 ? null : m });
        console.log(`  ${r.name.padEnd(18)} ${String(Math.round(100 * any / RUNS) + '%').padStart(10)}   ` +
                    `${fmtQ(m < 0 ? null : m).padStart(19)}   ${String(Math.round(100 * whole / RUNS) + '%').padStart(13)}`);
    }
    report.push(scr);
}

/* ----------------------------------------- NVDR made for a lossy link */

const SLICES = +opt('--slices', 2);
const PROTECT = opt('--protect', '0.5,0.25,0.1').split(',').map(Number);
const sliced = join(tmp, 'sliced.nvdr');
run(join(root, 'nvdr_encode'), [refPath, sliced, '--q', String(q), '--slices', String(SLICES), '--quiet']);
const plainFile = new Uint8Array(readFileSync(base.path)), slicedFile = new Uint8Array(readFileSync(sliced));
const jsImage = r => r && { width: r.header.width, height: r.header.height, rgb: r.rgb };
const parityOf = pk => pk.reduce((n, p) => n + p.length, 0);
const uep = packetize(slicedFile, { mtu: MTU, protect: PROTECT });
/* The same bytes of parity spread evenly, found by bisection. */
let lo = 0, hi = 2;
for (let i = 0; i < 30; i++) {
    const mid = (lo + hi) / 2;
    if (parityOf(packetize(slicedFile, { mtu: MTU, protect: [mid, mid, mid] })) > parityOf(uep)) hi = mid; else lo = mid;
}
const flat = lo;
const LOSSY = [
    { name: 'sem fatias', file: plainFile, pk: packetize(plainFile, { mtu: MTU, protect: [0] }) },
    { name: `fatias de ${SLICES}`, file: slicedFile, pk: packetize(slicedFile, { mtu: MTU, protect: [0] }) },
    { name: `fatias + FEC igual`, file: slicedFile, pk: packetize(slicedFile, { mtu: MTU, protect: [flat, flat, flat] }) },
    { name: `fatias + FEC desigual`, file: slicedFile, pk: uep },
];
const fullQ = LOSSY.map(v => psnrY(jsImage(decodeJs(v.file, 2))));
for (const [i, v] of LOSSY.entries())
    console.error(`${v.name.padEnd(22)} ${v.file.length} B em ${v.pk.length} pacotes (${parityOf(v.pk)} B), ${fullQ[i].toFixed(2)} dB`);
for (const [si, sc] of scenarios.entries()) {
    console.log(`\n== ${sc.name}, NVDR sem reenvio, ${RUNS} sorteios: buracos decodificados, fatias, FEC ` +
                `(desigual ${PROTECT.join('/')}, igual ${flat.toFixed(2)})`);
    console.log('  variante                  bytes   chega algo   mediana   pior 10%   inteiro   completo em');
    report[si].lossy = [];
    for (const v of LOSSY) {
        const qs = [];
        let whole = 0;
        const sent = parityOf(v.pk);
        for (let s = 0; s < RUNS; s++) {
            const lost = losses(sc, rng(+opt('--seed', 1) * 104729 + s + 1));
            const got = v.pk.filter(() => !lost());
            const r = reassemble(got);
            const img = r ? decodeJs(r.data, 2, false, null, false, null, r.holes) : null;
            const y = img ? psnrY(jsImage(img)) : null;
            qs.push(y === null ? -1 : y);
            if (r && !r.holes.length) whole++;
        }
        qs.sort((a, b) => a - b);
        const any = qs.filter(x => x >= 0).length, med = qs[RUNS >> 1], p10 = qs[Math.floor(RUNS / 10)];
        const t = v.pk.length * MTU * 8 / (sc.kbps * 1000) + sc.rtt / 2000;
        report[si].lossy.push({ name: v.name, bytes: sent, any: any / RUNS, median: med, p10, whole: whole / RUNS, time: t });
        console.log(`  ${v.name.padEnd(22)} ${String(sent).padStart(8)}   ${String(Math.round(100 * any / RUNS) + '%').padStart(10)}   ` +
                    `${fmtQ(med < 0 ? null : med).padStart(7)}   ${fmtQ(p10 < 0 ? null : p10).padStart(8)}   ` +
                    `${String(Math.round(100 * whole / RUNS) + '%').padStart(7)}   ${fmtT(t).padStart(11)}`);
    }
}

if (opt('--json')) writeFileSync(opt('--json'), JSON.stringify({ image: basename(image), target, formats: results, report }, null, 1));

/* Quality against time on the first scenario, one run, as an SVG. */
if (opt('--svg')) {
    const sc = scenarios[0], W = 760, H = 380, L = 56, B = 44, R = 150, T = 30;
    const curves = results.map(r => {
        const done = arrivals(r.bytes, sc, rng(7919 * +opt('--seed', 1) + 1));
        return { r, pts: r.pts.filter(p => p.q !== null).map(p => ({ t: timeForBytes(done, p.bytes), q: p.q })) };
    });
    const tmax = Math.max(...curves.map(c => Math.max(...c.pts.map(p => p.t)))) * 1.05;
    const qmin = 10, qmax = Math.ceil(Math.max(...results.map(r => r.q)) + 1);
    const X = t => L + (W - L - R) * t / tmax, Y = v => T + (H - T - B) * (1 - (v - qmin) / (qmax - qmin));
    const colours = ['#4b9fff', '#3fb950', '#f85149', '#d29922', '#a371f7', '#db61a2', '#8b949e'];
    let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" font-family="sans-serif" font-size="12">` +
              `<rect width="${W}" height="${H}" fill="#0d1117"/>` +
              `<text x="${L}" y="18" fill="#e6edf3">Qualidade ao longo do tempo: ${sc.name} (${sc.kbps} kbit/s, perda ${sc.loss * 100}%)</text>`;
    for (let v = qmin; v <= qmax; v += 5) svg += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#30363d"/>` +
                                                 `<text x="${L - 6}" y="${Y(v) + 4}" fill="#8b949e" text-anchor="end">${v} dB</text>`;
    const step = tmax > 60 ? 20 : tmax > 20 ? 5 : 2;
    for (let t = 0; t <= tmax; t += step) svg += `<text x="${X(t)}" y="${H - B + 18}" fill="#8b949e" text-anchor="middle">${t} s</text>`;
    curves.forEach((c, i) => {
        const col = colours[i % colours.length];
        if (!c.pts.length) return;
        let d = '', prev = null;
        for (const p of c.pts) {
            const q2 = Math.max(qmin, p.q);
            // Each curve rises from the floor where its first picture
            // arrives, so a format that shows nothing until the end is a
            // step there, not a line hidden at the top.
            d += prev === null ? `M${X(p.t)},${Y(qmin)}V${Y(q2)}` : `H${X(p.t)}V${Y(q2)}`;
            prev = q2;
        }
        d += `H${X(tmax)}`;
        svg += `<path d="${d}" fill="none" stroke="${col}" stroke-width="2"/>` +
               `<text x="${W - R + 8}" y="${T + 16 * i + 4}" fill="${col}">${c.r.name}</text>`;
    });
    writeFileSync(opt('--svg'), svg + '</svg>');
}
rmSync(tmp, { recursive: true, force: true });
