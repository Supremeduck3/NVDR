/*
 * C against JS on an album: every image, whole and with the file cut at
 * several points. The fluid context makes each image depend on the models
 * the one before left, so one wrong model shows up in every image after.
 */
import { readFileSync, mkdtempSync, rmSync, readdirSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { AlbumReader } from '../public/nvda.js';

const [, , album] = process.argv;
if (!album) { console.error('usage: node scripts/crosscheck_album.mjs <file.nvda>'); process.exit(2); }
const tool = new URL('../nvdr_album', import.meta.url).pathname;

function cDecode(file) {
    const dir = mkdtempSync(join(tmpdir(), 'nvda-'));
    try { execFileSync(tool, ['unpack', file, dir], { stdio: 'pipe' }); } catch { }
    const names = readdirSync(dir).sort();
    const out = names.map(n => readFileSync(join(dir, n)));
    rmSync(dir, { recursive: true });
    return out;
}

// The C tool writes PNG with filter 0 on every row, so its pixels are the
// inflated IDAT with one byte skipped per row.
import { inflateSync } from 'node:zlib';
function pngPixels(buf) {
    let p = 8, w = 0, h = 0; const idat = [];
    while (p < buf.length) {
        const len = buf.readUInt32BE(p), type = buf.toString('ascii', p + 4, p + 8);
        if (type === 'IHDR') { w = buf.readUInt32BE(p + 8); h = buf.readUInt32BE(p + 12); }
        if (type === 'IDAT') idat.push(buf.subarray(p + 8, p + 8 + len));
        p += 12 + len;
    }
    const raw = inflateSync(Buffer.concat(idat)), out = new Uint8Array(w * h * 3);
    for (let y = 0; y < h; y++) out.set(raw.subarray(y * (w * 3 + 1) + 1, (y + 1) * (w * 3 + 1)), y * w * 3);
    return out;
}

let failures = 0;
function jsDecodeAll(file) {
    const bytes = readFileSync(file);
    const out = [];
    try {
        const r = new AlbumReader(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
        for (let i = 0; i < r.count; i++) {
            const e = r.decode(i);
            if (!e) break;
            out.push(e.rgb);
            if (e.partial) break;
        }
    } catch (err) { /* both sides stop on damage */ }
    return out;
}

function compare(label, c, js) {
    if (c.length !== js.length) {
        console.log(`${label}: C ${c.length} images, JS ${js.length}`); failures++; return;
    }
    for (let i = 0; i < c.length; i++)
        for (let k = 0; k < c[i].length; k++)
            if (c[i][k] !== js[i][k]) {
                console.log(`${label}: image ${i} differs at byte ${k}: js ${js[i][k]} vs c ${c[i][k]}`);
                failures++; return;
            }
    console.log(`${label}: identical, ${c.length} images`);
}

function check(label, file) { compare(label, cDecode(file).map(pngPixels), jsDecodeAll(file)); }

const size = readFileSync(album).length;
check('whole', album);
// Random access: the last photo alone, from a fresh reader on each side.
{
    const bytes = readFileSync(album);
    const r = new AlbumReader(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
    if (!r.fluid && r.count) {
        const last = r.count - 1;
        const dir = mkdtempSync(join(tmpdir(), 'nvda-'));
        execFileSync(tool, ['unpack', album, dir, '--only', String(last + 1)], { stdio: 'pipe' });
        const c = readdirSync(dir).map(n => pngPixels(readFileSync(join(dir, n))));
        rmSync(dir, { recursive: true });
        compare(`photo ${last + 1} alone (chain ${r.chain(last).map(k => k + 1).join('>')})`, c,
                [new AlbumReader(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length)).decode(last).rgb]);
    }
}
for (const pct of [5, 21, 38, 55, 72, 90]) {
    const cut = join(tmpdir(), `nvda_cut_${pct}.nvda`);
    writeFileSync(cut, readFileSync(album).subarray(0, Math.floor(size * pct / 100)));
    check(`cut ${pct}%`, cut);
}
process.exit(failures ? 1 : 0);
