/*
 * nvdr-fec.js — an NVDR file as datagrams that survive losing some.
 *
 * Over a link that loses packets and cannot ask for them again (radio,
 * satellite, a broadcast, a datagram channel whose round trip is too
 * long to wait for), what arrives is the file with holes. The decoder
 * already makes the most of holes (decode()'s `holes` in nvdr.js, and
 * slices, SLICES in nvdr.c, keep a hole to the rest of one slice); this
 * adds redundancy so that most holes can be filled in before decoding.
 *
 * THE CODE
 * --------
 * A Reed-Solomon erasure code over GF(2^8), applied across packets, as
 * Rizzo describes it for network protocols ("Effective erasure codes for
 * reliable computer communication protocols", 1997): k packets of data
 * and m of parity form a block, and any k of the k + m that arrive give
 * back all k. The code is systematic, so data packets are the file's own
 * bytes and a receiver uses them as they come; the parity is a Cauchy
 * matrix (Blömer, Kalfane, Karp, Karpinski, Luby and Zuckerman, "An
 * XOR-based erasure-resilient coding scheme", 1995), every square
 * submatrix of which is invertible, so whichever k packets arrive, the
 * system solves. Bursts of loss are no harder than scattered loss: an
 * erasure code counts packets, not where they were.
 *
 * UNEQUAL PROTECTION
 * ------------------
 * The bytes of an NVDR file are not worth the same. Without the header
 * and the colour layer there is no picture at all; without the high
 * texture band there is a slightly softer one. So each layer is coded in
 * blocks of its own with its own share of parity, more for what comes
 * first, as Priority Encoding Transmission does (Albanese, Blömer,
 * Edmonds, Luby and Sudan, 1996). The header, slice table and colour
 * layer are a few kilobytes: protecting them twice over costs little.
 *
 * A PACKET
 * --------
 *   byte  0      0xD7, this format
 *   byte  1      the block's number, in the file
 *   bytes 2, 3   k and m: the block's data and parity packets
 *   byte  4      this packet's index in the block, data first
 *   bytes 5-8    where the block's bytes start in the file (LE)
 *   bytes 9-12   how many bytes the block holds
 *   bytes 13-16  the file's length
 *   bytes 17-18  the block's packet payload: every data packet's but the
 *                last's, and every parity packet's
 *   bytes 19-20  the file's number, the sender's choice: what tells one
 *                file's packets from the next's on the same link
 *   then the payload: a data packet's share of the block (the last one
 *   short), or a parity packet's, as long as the longest data packet
 *
 * Packets are sent block by block, each block's data then its parity, in
 * the file's order, so what arrives first is still the start of the
 * progressive file.
 */
import { readHeader, layerThresholds } from './nvdr.js';

const MAGIC = 0xd7;
export const PACKET_HEADER = 21;

/* GF(2^8) with the polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11d). */
const EXP = new Uint8Array(512), LOG = new Uint8Array(256);
for (let i = 0, x = 1; i < 255; i++) {
    EXP[i] = x; LOG[x] = i;
    x <<= 1;
    if (x & 0x100) x ^= 0x11d;
}
for (let i = 255; i < 512; i++) EXP[i] = EXP[i - 255];
const mul = (a, b) => (a && b ? EXP[LOG[a] + LOG[b]] : 0);
const inv = a => EXP[255 - LOG[a]];
/* dst ^= c * src, over `n` bytes. */
function mulAdd(dst, src, c, n) {
    if (!c) return;
    const lc = LOG[c];
    for (let i = 0; i < n; i++) { const s = src[i]; if (s) dst[i] ^= EXP[lc + LOG[s]]; }
}
/* The parity matrix: row i, column j, 1 / (x_i + y_j), x_i = i and
 * y_j = m + j, all distinct while k + m <= 256. */
const cauchy = (i, j, m) => inv(i ^ (m + j));

/* The protection classes of a file: [start, end) of what precedes layer 1
 * (header, tables, colour), of layer 1 and of layer 2. Anything that is
 * not an NVDR still is one class. */
export function classesOf(file) {
    const h = readHeader(file);
    if (!h) return [[0, file.length]];
    const [a, b, c] = layerThresholds(h);
    return [[0, Math.min(a, file.length)], [a, Math.min(b, file.length)], [b, Math.min(c, file.length)]]
        .filter(([s, e]) => e > s);
}

/*
 * The file as packets of at most `mtu` bytes, in the order to send them.
 * `protect[c]` is the share of parity for class c (the colour, the low
 * band, the high band): 0.5 sends one parity packet for every two of
 * data. A class's block holds at most as many packets as k + m <= 255
 * allows.
 */
export function packetize(file, { mtu = 1200, protect = [0.5, 0.25, 0.1], id = 0 } = {}) {
    const pay = mtu - PACKET_HEADER, out = [];
    let blockNo = 0;
    classesOf(file).forEach(([start, end], c) => {
        const r = protect[Math.min(c, protect.length - 1)] || 0;
        const kmax = Math.max(1, Math.floor(255 / (1 + r)));
        const total = Math.ceil((end - start) / pay);
        for (let first = 0; first < total; first += kmax) {
            const k = Math.min(kmax, total - first), m = r > 0 ? Math.min(255 - k, Math.ceil(k * r)) : 0;
            const bs = start + first * pay, be = Math.min(end, bs + k * pay);
            if (blockNo > 255) throw new Error('a file of more than 256 blocks');
            const head = (idx, len) => {
                const p = new Uint8Array(PACKET_HEADER + len), v = new DataView(p.buffer);
                p[0] = MAGIC; p[1] = blockNo; p[2] = k; p[3] = m; p[4] = idx;
                v.setUint32(5, bs, true); v.setUint32(9, be - bs, true); v.setUint32(13, file.length, true);
                v.setUint16(17, pay, true);
                v.setUint16(19, id, true);
                return p;
            };
            const data = [];
            for (let j = 0; j < k; j++) {
                const s = bs + j * pay, e = Math.min(be, s + pay), p = head(j, e - s);
                p.set(file.subarray(s, e), PACKET_HEADER);
                data.push(p.subarray(PACKET_HEADER));
                out.push(p);
            }
            const plen = Math.min(pay, be - bs);
            for (let i = 0; i < m; i++) {
                const p = head(k + i, plen), par = p.subarray(PACKET_HEADER);
                for (let j = 0; j < k; j++) mulAdd(par, data[j], cauchy(i, j, m), data[j].length);
                out.push(p);
            }
            blockNo++;
        }
    });
    return out;
}

/* Solves A x = b over GF(2^8) for the rows of b (Gauss-Jordan); A is n x n
 * and invertible, b is n rows of `len` bytes. Changes both. */
function solve(A, b, len) {
    const n = A.length;
    for (let c = 0; c < n; c++) {
        let p = c;
        while (!A[p][c]) p++;
        [A[c], A[p]] = [A[p], A[c]]; [b[c], b[p]] = [b[p], b[c]];
        const ic = inv(A[c][c]);
        for (let j = 0; j < n; j++) A[c][j] = mul(A[c][j], ic);
        for (let i = 0; i < len; i++) b[c][i] = mul(b[c][i], ic);
        for (let r = 0; r < n; r++) {
            const f = A[r][c];
            if (r === c || !f) continue;
            for (let j = 0; j < n; j++) A[r][j] ^= mul(f, A[c][j]);
            mulAdd(b[r], b[c], f, len);
        }
    }
    return b;
}

/*
 * What a receiver makes of the packets that arrived, in any order: the
 * file as far as it is known, every block that got k of its packets
 * filled in whole, and the byte ranges still missing, as decode() takes
 * them. Null when nothing arrived. Packets of another file (another
 * number than the first packet's, or than `id` when given) are left out.
 */
export const packetId = p => (p.length >= PACKET_HEADER && p[0] === MAGIC ? p[19] | p[20] << 8 : -1);
export function reassemble(packets, id = -1) {
    const blocks = new Map();
    let size = -1;
    for (const p of packets) {
        if (p.length < PACKET_HEADER || p[0] !== MAGIC) continue;
        const v = new DataView(p.buffer, p.byteOffset, p.byteLength);
        const no = p[1], k = p[2], m = p[3], idx = p[4];
        if (!k || idx >= k + m) continue;
        if (id < 0) id = packetId(p);
        if (packetId(p) !== id) continue;
        size = v.getUint32(13, true);
        let b = blocks.get(no);
        if (!b) blocks.set(no, b = { k, m, at: v.getUint32(5, true), len: v.getUint32(9, true),
                                     pay: v.getUint16(17, true), got: new Map() });
        b.got.set(idx, p.subarray(PACKET_HEADER));
    }
    if (size < 0) return null;
    const data = new Uint8Array(size), covered = [];
    let recovered = 0;
    for (const b of blocks.values()) {
        if (b.at + b.len > size || !b.pay || b.len > b.k * b.pay) continue;
        const pay = b.pay, lenOf = j => Math.max(0, Math.min(pay, b.len - j * pay));
        const missing = [];
        for (let j = 0; j < b.k; j++) if (!b.got.has(j)) missing.push(j);
        const parity = [...b.got.keys()].filter(i => i >= b.k).slice(0, missing.length);
        if (missing.length && parity.length === missing.length) {
            // Each parity row, less what the data at hand contributes, is
            // the missing data through the Cauchy submatrix.
            const rhs = parity.map(i => {
                const r = new Uint8Array(pay);
                r.set(b.got.get(i).subarray(0, pay));
                for (let j = 0; j < b.k; j++) if (b.got.has(j)) mulAdd(r, b.got.get(j), cauchy(i - b.k, j, b.m), lenOf(j));
                return r;
            });
            const A = parity.map(i => missing.map(j => cauchy(i - b.k, j, b.m)));
            solve(A, rhs, pay);
            missing.forEach((j, t) => b.got.set(j, rhs[t].subarray(0, lenOf(j))));
            recovered += missing.length;
        }
        for (let j = 0; j < b.k; j++) {
            const d = b.got.get(j);
            if (!d) continue;
            const s = b.at + j * pay, n = Math.min(lenOf(j), d.length);
            data.set(d.subarray(0, n), s);
            covered.push([s, s + n]);
        }
    }
    covered.sort((x, y) => x[0] - y[0]);
    const holes = [];
    let at = 0;
    for (const [s, e] of covered) {
        if (s > at) holes.push([at, s]);
        at = Math.max(at, e);
    }
    if (at < size) holes.push([at, size]);
    return { data, holes, size, recovered, id };
}
