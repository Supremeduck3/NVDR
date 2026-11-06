/*
 * worker.js — codes stretches of a video for another machine's server.js.
 *
 *   node worker.js [port]        (default 3001, or NVDR_WORKER_PORT)
 *
 * A video longer than one GOP is coded in stretches, one per GOP, which
 * join into the file a single pass writes, to the byte (STRETCHES in
 * src/nvdrv.c). The machine serving the page codes some and hands the
 * rest to the machines named in its NVDR_WORKERS, each running this; see
 * "Vários computadores" in GUIA.md.
 *
 *   GET  /info     { format, cores, busy }: the sequence format this
 *                  machine's nvdrv_encode writes, and how many stretches
 *                  it codes at once (one per core).
 *   POST /stretch  the frames a stretch reads, framed as an album upload
 *                  is (u32 count, then per file: u16 name length, name,
 *                  u32 size, bytes; little-endian), with X-From, X-To and
 *                  X-Args (the encoder's options, as JSON). The answer is
 *                  the stretch's file, with nvdrv_encode's last line in
 *                  X-Stretch and its report in X-Report.
 *
 * It answers only machines on the local network (private and link-local
 * addresses, and this machine) and has no password: anyone on that
 * network can have it code. The options it passes on are a fixed list of
 * numeric ones, and the frames reach the disk under sanitised names in a
 * folder of its own.
 */
const http = require('http');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFile } = require('child_process');

const PORT = Number(process.argv[2] || process.env.NVDR_WORKER_PORT || 3001);
const CORES = os.cpus().length;
const MAX_BYTES = 1024 * 1024 * 1024;
const MAX_FRAMES = 256;
let busy = 0;

/* The encoder's options a stretch may carry: each takes one number, but
 * for the two flags. */
const NUMERIC = new Set(['--gop', '--search', '--block', '--intra-thresh', '--fps', '--q', '--pred-q',
                         '--skip-k', '--mv-lambda', '--bframes', '--b-q-step', '--lookahead', '--tpl',
                         '--refs', '--tf', '--tf-strength', '--tf-levels']);
const FLAGS = new Set(['--fast', '--no-restore']);

function findEncoder() {
    for (const name of ['nvdrv_encode.exe', 'nvdrv_encode']) {
        const p = path.join(__dirname, name);
        if (fs.existsSync(p)) return p;
    }
    return null;
}

/* Private, link-local and loopback addresses, IPv4 (also mapped into
 * IPv6) and IPv6. */
function isLocalNetwork(address) {
    const a = String(address || '').replace(/^::ffff:/i, '');
    const v4 = /^(\d+)\.(\d+)\.(\d+)\.(\d+)$/.exec(a);
    if (v4) {
        const [x, y] = [Number(v4[1]), Number(v4[2])];
        return x === 10 || x === 127 || (x === 192 && y === 168) || (x === 172 && y >= 16 && y <= 31) ||
               (x === 169 && y === 254);
    }
    const l = a.toLowerCase();
    return l === '::1' || /^f[cd][0-9a-f]{2}:/.test(l) || /^fe[89ab][0-9a-f]:/.test(l);
}

function safeArgs(raw) {
    let list;
    try { list = JSON.parse(raw || '[]'); } catch { return null; }
    if (!Array.isArray(list)) return null;
    const out = [];
    for (let i = 0; i < list.length; i++) {
        const a = String(list[i]);
        if (FLAGS.has(a)) out.push(a);
        else if (NUMERIC.has(a) && /^-?\d{1,6}(\.\d{1,6})?$/.test(String(list[i + 1]))) out.push(a, String(list[++i]));
        else return null;
    }
    return out;
}

function unpack(body) {
    if (body.length < 4) return null;
    const count = body.readUInt32LE(0);
    if (!count || count > MAX_FRAMES) return null;
    const files = [];
    let at = 4;
    for (let i = 0; i < count; i++) {
        if (at + 2 > body.length) return null;
        const nl = body.readUInt16LE(at); at += 2;
        if (nl > 255 || at + nl + 4 > body.length) return null;
        const name = body.toString('utf8', at, at + nl); at += nl;
        const size = body.readUInt32LE(at); at += 4;
        if (at + size > body.length) return null;
        if (!/^[A-Za-z0-9_-]{1,64}\.(png|ppm|jpe?g)$/i.test(name)) return null;
        files.push({ name, data: body.subarray(at, at + size) });
        at += size;
    }
    return files;
}

function answer(res, code, text) {
    if (res.writableEnded) return;
    res.writeHead(code, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end(text);
}

function handleStretch(req, res, encoder) {
    const from = Number(req.headers['x-from']), to = Number(req.headers['x-to']);
    const args = safeArgs(req.headers['x-args']);
    if (!Number.isInteger(from) || !Number.isInteger(to) || from < 0 || to < 0 || !args)
        return answer(res, 400, 'bad stretch');
    if (busy >= CORES) return answer(res, 503, 'busy');
    busy++;
    let done = false;
    const finish = () => { if (!done) { done = true; busy--; } };
    res.on('close', finish);

    const chunks = [];
    let received = 0, aborted = false;
    req.on('data', (c) => {
        if (aborted) return;
        received += c.length;
        if (received > MAX_BYTES) { aborted = true; answer(res, 413, 'too large'); req.destroy(); finish(); return; }
        chunks.push(c);
    });
    req.on('end', () => {
        if (aborted) return;
        const files = unpack(Buffer.concat(chunks));
        if (!files) { finish(); return answer(res, 400, 'bad frames'); }
        const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'nvdr_worker_'));
        const cleanup = () => fs.rm(dir, { recursive: true, force: true }, () => {});
        const frames = path.join(dir, 'frames');
        fs.mkdirSync(frames);
        for (const f of files) fs.writeFileSync(path.join(frames, f.name), f.data);
        const out = path.join(dir, 'stretch.nvdrv');
        console.log(`Stretch ${from}-${to || 'end'} for ${req.socket.remoteAddress}: ${files.length} frames...`);
        // One core a stretch: the coordinator sends as many as there are.
        execFile(encoder, [frames, out, ...args, '--from', String(from), '--to', String(to)],
                 { timeout: 30 * 60 * 1000, maxBuffer: 16 * 1024 * 1024, env: { ...process.env, OMP_NUM_THREADS: '1' } },
                 (err, stdout, stderr) => {
            if (err) { finish(); cleanup(); return answer(res, 500, (stderr || err.message).trim()); }
            const lines = stdout.trim().split('\n');
            fs.readFile(out, (err2, data) => {
                cleanup();
                if (err2) { finish(); return answer(res, 500, 'no output'); }
                res.writeHead(200, {
                    'Content-Type': 'application/octet-stream',
                    'Content-Length': data.length,
                    'X-Stretch': lines[lines.length - 1],
                    'X-Report': Buffer.from(stdout, 'utf8').toString('base64')
                });
                res.end(data);
                finish();
                console.log(`Stretch ${from}-${to || 'end'}: ${data.length} bytes.`);
            });
        });
    });
}

const encoder = findEncoder();
if (!encoder) {
    console.error(`nvdrv_encode not found in ${__dirname}. Build it with \`make\`.`);
    process.exit(1);
}
execFile(encoder, ['--version'], (err, stdout) => {
    const m = /format (\d+)/.exec(stdout || '');
    if (err || !m) { console.error('nvdrv_encode --version failed: rebuild it with `make`.'); process.exit(1); }
    const format = Number(m[1]);
    http.createServer((req, res) => {
        if (!isLocalNetwork(req.socket.remoteAddress)) return answer(res, 403, 'local network only');
        if (req.method === 'GET' && req.url === '/info') {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            return res.end(JSON.stringify({ format, cores: CORES, busy }));
        }
        if (req.method === 'POST' && req.url === '/stretch') return handleStretch(req, res, encoder);
        answer(res, 404, 'not found');
    }).listen(PORT, () => {
        console.log(`NVDR worker: format ${format}, ${CORES} cores, port ${PORT} (local network only)`);
    });
});
