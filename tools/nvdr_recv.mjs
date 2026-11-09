#!/usr/bin/env node
// nvdr_recv — the other end of tools/nvdr_send.mjs.
//
//   node tools/nvdr_recv.mjs <out-prefix> [port] [options]
//
//   --wait MS   a file is taken as over when nothing of it has arrived for
//               this long (default 2000), or as soon as it is whole
//   --once      leave after the first file
//
// Listens for packets, fills in what the parity allows, and for every file
// writes <out-prefix><id>.nvdr (the bytes that never came as zeros) and
// <out-prefix><id>.png, the picture decoded around the holes. Files are
// told apart by sender and by the number in their packets.
import dgram from 'node:dgram';
import { writeFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';
import { reassemble, packetId } from '../public/nvdr-fec.js';
import { decodeJs } from '../public/nvdr.js';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const pos = args.filter((a, i) => !a.startsWith('--') && args[i - 1] !== '--wait');
if (!pos.length) {
    console.error('usage: node tools/nvdr_recv.mjs <out-prefix> [port] [--wait MS] [--once]');
    process.exit(2);
}
const [prefix, port = '5004'] = pos;
const wait = +opt('--wait', 2000), once = args.includes('--once');

const CRC = new Uint32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
function png(rgb, w, h) {
    const chunk = (type, data) => {
        const b = Buffer.alloc(12 + data.length);
        b.writeUInt32BE(data.length, 0); b.write(type, 4, 'ascii'); data.copy(b, 8);
        let c = 0xffffffff;
        for (let i = 4; i < 8 + data.length; i++) c = CRC[(c ^ b[i]) & 255] ^ (c >>> 8);
        b.writeUInt32BE((c ^ 0xffffffff) >>> 0, 8 + data.length);
        return b;
    };
    const head = Buffer.alloc(13);
    head.writeUInt32BE(w, 0); head.writeUInt32BE(h, 4); head[8] = 8; head[9] = 2;
    const raw = Buffer.alloc(h * (1 + w * 3));
    for (let y = 0; y < h; y++) Buffer.from(rgb.buffer, rgb.byteOffset + y * w * 3, w * 3).copy(raw, y * (1 + w * 3) + 1);
    return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', head),
                          chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

const files = new Map();   // sender + id -> { packets, timer, t0 }
const sock = dgram.createSocket('udp4');
function finish(key) {
    const f = files.get(key);
    if (!f) return;
    clearTimeout(f.timer);
    clearTimeout(f.check);
    files.delete(key);
    const r = reassemble(f.packets);
    if (!r) return;
    const name = `${prefix}${r.id}`;
    writeFileSync(`${name}.nvdr`, r.data);
    const missing = r.holes.reduce((n, [a, b]) => n + b - a, 0);
    const img = decodeJs(r.data, 2, false, null, false, null, r.holes);
    let shown = 'no picture: the header or all of the colour is missing';
    if (img) {
        writeFileSync(`${name}.png`, png(img.rgb, img.header.width, img.header.height));
        shown = `${name}.png, tiles ${img.tilesComplete.join('/')} of ${img.tiles}`;
    }
    console.log(`file ${r.id} from ${f.from}: ${f.packets.length} packets in ${((Date.now() - f.t0) / 1000).toFixed(1)} s, ` +
                `${r.recovered} filled in by parity, ` +
                (missing ? `${missing} of ${r.size} bytes missing (--missing ${r.holes.map(([a, b]) => `${a}-${b}`).join(',')})`
                         : `whole (${r.size} bytes)`) + `; ${shown}`);
    if (once) sock.close();
}
sock.on('message', (msg, from) => {
    const id = packetId(msg);
    if (id < 0) return;
    const key = `${from.address}:${from.port}:${id}`;
    let f = files.get(key);
    if (!f) files.set(key, f = { packets: [], t0: Date.now(), from: `${from.address}:${from.port}` });
    f.packets.push(new Uint8Array(msg));
    clearTimeout(f.timer);
    clearTimeout(f.check);
    // Whole already? Asked at the first pause, since it means solving for
    // the parity: over as soon as it is, else when nothing more comes.
    f.check = setTimeout(() => {
        const r = reassemble(f.packets);
        if (r && !r.holes.length) finish(key);
    }, 60);
    f.timer = setTimeout(() => finish(key), wait);
});
sock.bind(+port, () => console.log(`listening on UDP ${port}; files go to ${prefix}<id>.nvdr and .png`));
