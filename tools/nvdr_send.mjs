#!/usr/bin/env node
// nvdr_send — an NVDR file over UDP, as packets that survive losing some.
//
//   node tools/nvdr_send.mjs <file.nvdr> <host> [port] [options]
//
//   --kbps N         pace the packets at N kbit/s (default 512; 0 sends at once)
//   --protect a,b,c  parity for the colour, the low band and the high band,
//                    as a share of each one's packets (default 0.5,0.25,0.1)
//   --mtu N          bytes per packet, header included (default 1200)
//   --id N           the file's number on the link (default: random)
//   --drop P         for testing: lose packets at mean rate P before sending,
//   --burst L        in runs of mean length L (default 1)
//
// The receiver is tools/nvdr_recv.mjs. The packets and their parity are
// public/nvdr-fec.js's; encode the file with --slices so that a hole the
// parity cannot fill costs one slice and not the rest of a layer.
import dgram from 'node:dgram';
import { readFileSync } from 'node:fs';
import { packetize } from '../public/nvdr-fec.js';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const flags = new Set(['--kbps', '--protect', '--mtu', '--id', '--drop', '--burst']);
const pos = args.filter((a, i) => !flags.has(a) && !flags.has(args[i - 1]));
if (pos.length < 2) {
    console.error('usage: node tools/nvdr_send.mjs <file.nvdr> <host> [port] [--kbps N] [--protect a,b,c] ' +
                  '[--mtu N] [--id N] [--drop P --burst L]');
    process.exit(2);
}
const [path, host, port = '5004'] = pos;
const file = new Uint8Array(readFileSync(path));
const mtu = +opt('--mtu', 1200), kbps = +opt('--kbps', 512);
const id = opt('--id') !== undefined ? +opt('--id') & 0xffff : Math.floor(Math.random() * 65536);
const packets = packetize(file, { mtu, protect: opt('--protect', '0.5,0.25,0.1').split(',').map(Number), id });

// Gilbert-Elliott, as scripts/analysis/badnet.mjs draws its losses.
const loss = +opt('--drop', 0), burst = Math.max(1, +opt('--burst', 1));
const pBG = 1 / burst, pGB = loss * pBG / Math.max(1e-9, 1 - loss);
let bad = false;
const lost = () => (bad = bad ? Math.random() >= pBG : Math.random() < pGB);

const sock = dgram.createSocket('udp4');
const gap = kbps > 0 ? mtu * 8 / kbps : 0;   // ms between packets
let sent = 0, dropped = 0, bytes = 0;
const t0 = Date.now();
for (const [i, p] of packets.entries()) {
    if (gap) await new Promise(r => setTimeout(r, Math.max(0, t0 + i * gap - Date.now())));
    if (loss && lost()) { dropped++; continue; }
    await new Promise((res, rej) => sock.send(p, +port, host, e => (e ? rej(e) : res())));
    sent++; bytes += p.length;
}
sock.close();
console.log(`${path}: ${file.length} bytes as ${packets.length} packets (file ${id}) to ${host}:${port}, ` +
            `${sent} sent (${bytes} bytes)${dropped ? `, ${dropped} dropped on purpose` : ''}, ` +
            `${((Date.now() - t0) / 1000).toFixed(1)} s`);
