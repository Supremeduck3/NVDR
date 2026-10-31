/*
 * cluster.js — a video's stretches coded on several machines.
 *
 * nvdrv_encode codes a video longer than one GOP in stretches, one per
 * GOP, on every core of this machine, and joins them into the file one
 * pass writes (STRETCHES in src/nvdrv.c). With NVDR_WORKERS naming other
 * machines that run worker.js,
 *
 *   NVDR_WORKERS=192.168.0.20,192.168.0.21:3005 node server.js
 *
 * server.js hands the stretches out here instead: each machine takes as
 * many at once as it has cores, this one included, and the files are
 * joined the way nvdrv_encode joins them.
 *
 * A stretch codes the intra frame it starts with without writing it,
 * because the stretch before writes it; the two must have coded the same
 * picture, or the frames after it would predict from one the player never
 * sees. Each stretch reports a check of both of its end frames, and every
 * join is compared. Two machines whose compilers or libraries round
 * differently can disagree there: then every stretch coded elsewhere is
 * coded again here. A machine that fails, or does not answer, has its
 * stretches coded here too.
 */
const http = require('http');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFile } = require('child_process');

const WORKERS = (process.env.NVDR_WORKERS || '').split(',').map(s => s.trim()).filter(Boolean)
    .map(s => (/:\d+$/.test(s) ? s : `${s}:3001`));
const STRETCH_TIMEOUT = 30 * 60 * 1000;

function run(cmd, args, env) {
    return new Promise((ok, fail) => {
        execFile(cmd, args, { timeout: STRETCH_TIMEOUT, maxBuffer: 16 * 1024 * 1024, env: { ...process.env, ...env } },
                 (err, stdout, stderr) => (err ? fail(new Error((stderr || err.message).trim())) : ok(stdout)));
    });
}

function request(address, method, url, headers, body, timeout) {
    return new Promise((ok, fail) => {
        const [host, port] = address.split(/:(?=\d+$)/);
        const req = http.request({ host, port: Number(port), method, path: url, headers, timeout }, (res) => {
            const chunks = [];
            res.on('data', (c) => chunks.push(c));
            res.on('end', () => ok({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks) }));
            res.on('error', fail);
        });
        req.on('timeout', () => req.destroy(new Error(`${address}: no answer`)));
        req.on('error', fail);
        req.end(body);
    });
}

/* nvdrv_encode's last line for a stretch: the first and last display
 * numbers it wrote, how many frames, and the checks of the intra frames
 * it starts and ends with. */
function parseStretch(line) {
    const m = /^trecho (\d+) (\d+) (\d+) ([0-9a-f]{8}) ([0-9a-f]{8})$/.exec(String(line || '').trim());
    return m && { first: Number(m[1]), last: Number(m[2]), frames: Number(m[3]), start: m[4], end: m[5] };
}

function frameNames(dir) {
    return fs.readdirSync(dir).filter(n => /\.(png|ppm|jpe?g)$/i.test(n)).sort();
}

async function codeHere(encoder, framesDir, names, st, args, threads) {
    const dir = fs.mkdtempSync(path.join(path.dirname(framesDir), 'stretch_'));
    try {
        const sub = path.join(dir, 'frames');
        fs.mkdirSync(sub);
        for (let d = st.first; d <= st.last; d++) {
            const from = path.join(framesDir, names[d]), to = path.join(sub, names[d]);
            try { fs.linkSync(from, to); } catch { fs.copyFileSync(from, to); }
        }
        const out = path.join(dir, 'stretch.nvdrv');
        const stdout = await run(encoder, [sub, out, ...args, '--from', String(st.from), '--to', String(st.to)],
                                 { OMP_NUM_THREADS: String(threads) });
        const lines = stdout.trim().split('\n');
        const info = parseStretch(lines[lines.length - 1]);
        if (!info) throw new Error('nvdrv_encode did not report the stretch');
        return { ...info, data: fs.readFileSync(out), where: 'este computador' };
    } finally {
        fs.rmSync(dir, { recursive: true, force: true });
    }
}

async function codeThere(address, framesDir, names, st, args) {
    const parts = [Buffer.alloc(4)];
    parts[0].writeUInt32LE(st.last - st.first + 1);
    for (let d = st.first; d <= st.last; d++) {
        const name = Buffer.from(names[d], 'utf8'), data = fs.readFileSync(path.join(framesDir, names[d]));
        const head = Buffer.alloc(6 + name.length);
        head.writeUInt16LE(name.length, 0);
        name.copy(head, 2);
        head.writeUInt32LE(data.length, 2 + name.length);
        parts.push(head, data);
    }
    const body = Buffer.concat(parts);
    const r = await request(address, 'POST', '/stretch', {
        'Content-Type': 'application/octet-stream', 'Content-Length': body.length,
        'X-From': String(st.from), 'X-To': String(st.to), 'X-Args': JSON.stringify(args)
    }, body, STRETCH_TIMEOUT);
    if (r.status !== 200) throw new Error(`${address}: ${r.status} ${r.body.toString('utf8').slice(0, 200)}`);
    const info = parseStretch(r.headers['x-stretch']);
    if (!info) throw new Error(`${address}: no stretch line`);
    return { ...info, data: r.body, where: address };
}

/* The workers that answer and write the same format as this machine. */
async function usableWorkers(format) {
    const found = await Promise.all(WORKERS.map(async (address) => {
        try {
            const r = await request(address, 'GET', '/info', {}, undefined, 3000);
            const info = JSON.parse(r.body.toString('utf8'));
            if (r.status === 200 && info.format === format && info.cores > 0) return { address, cores: info.cores };
            console.warn(`Cluster: ${address} writes format ${info.format}, this machine ${format}: not used.`);
        } catch (e) {
            console.warn(`Cluster: ${address} does not answer (${e.message}): not used.`);
        }
        return null;
    }));
    return found.filter(Boolean);
}

/* The stretches into one file, as nvdrv_encode's join() does: the first
 * whole, the others after their 24-byte header with their first display
 * number rewritten as the change from the stretch before; then the
 * frame count. */
function join(results) {
    const varint = (v) => { const b = []; while (v >= 0x80) { b.push((v & 0x7f) | 0x80); v >>>= 7; } b.push(v); return Buffer.from(b); };
    const zigzag = (v) => (v >= 0 ? 2 * v : -2 * v - 1);
    const out = [results[0].data];
    let frames = results[0].frames, last = results[0].last;
    for (let k = 1; k < results.length; k++) {
        const r = results[k];
        if (!r.frames) continue;
        const d = r.data;
        let old = 0;
        while (d[25 + old] & 0x80) old++;
        old++;
        out.push(d.subarray(24, 25), varint(zigzag(r.first - last)), d.subarray(25 + old));
        frames += r.frames;
        last = r.last;
    }
    const file = Buffer.concat(out);
    file.writeUInt32LE(frames, 10);
    return file;
}

/*
 * Codes frames in `framesDir` into `outPath` with nvdrv_encode's `args`,
 * sharing the stretches with the workers. Resolves to the report, or to
 * null when there is nothing to share (no worker answers, or one stretch):
 * then nvdrv_encode codes it alone, on this machine's cores.
 */
async function encodeShared(encoder, framesDir, outPath, args) {
    if (!WORKERS.length) return null;
    const version = await run(encoder, ['--version']);
    const format = Number((/format (\d+)/.exec(version) || [])[1]);
    const plan = (await run(encoder, [framesDir, '--plan', ...args])).trim().split('\n')
        .map(l => /^trecho (\d+) (\d+) (\d+) (\d+)$/.exec(l)).filter(Boolean)
        .map(m => ({ from: Number(m[1]), to: Number(m[2]), first: Number(m[3]), last: Number(m[4]) }));
    if (plan.length < 2) return null;
    const workers = await usableWorkers(format);
    if (!workers.length) return null;
    const names = frameNames(framesDir);
    const t0 = Date.now();

    // Every machine takes the next stretch as soon as one of its cores is
    // free; a stretch a worker could not code goes back for this machine.
    const results = new Array(plan.length), queue = plan.map((_, k) => k), retry = [];
    // One lane a core, taken in turn from each machine, so that a video
    // of few stretches still spreads over all of them.
    const here = os.cpus().length;
    const machines = [{ address: null, cores: here }, ...workers];
    const lanes = [];
    for (let i = 0; i < Math.max(...machines.map(m => m.cores)); i++)
        for (const m of machines) if (i < m.cores) lanes.push(m.address);
    await Promise.all(lanes.map(async (address) => {
        while (queue.length) {
            const k = queue.shift();
            if (!address) { results[k] = await codeHere(encoder, framesDir, names, plan[k], args, 1); continue; }
            try {
                results[k] = await codeThere(address, framesDir, names, plan[k], args);
            } catch (e) {
                console.warn(`Cluster: stretch ${k} on ${address} failed (${e.message}); coding it here.`);
                retry.push(k);
                return;
            }
        }
    }));
    await Promise.all(retry.map(async (k) => { results[k] = await codeHere(encoder, framesDir, names, plan[k], args, 1); }));

    // Every join: the intra frame a stretch starts with is the one the
    // stretch before ended with.
    const mismatched = () => results.some((r, k) => k > 0 && r.start !== results[k - 1].end);
    if (mismatched()) {
        console.warn('Cluster: the machines coded a shared intra frame differently; coding their stretches here.');
        await Promise.all(results.map(async (r, k) => {
            if (r.where !== 'este computador') results[k] = await codeHere(encoder, framesDir, names, plan[k], args, 1);
        }));
        if (mismatched()) throw new Error('stretches coded on this machine do not join');
    }

    const file = join(results);
    fs.writeFileSync(outPath, file);
    const secs = ((Date.now() - t0) / 1000).toFixed(1);
    const lines = [`${names.length} quadros em ${plan.length} trechos, ${secs} s, ${file.length} bytes`,
                   `  máquinas: este computador (${here} núcleos), ` +
                   workers.map(w => `${w.address} (${w.cores})`).join(', ')];
    results.forEach((r, k) => lines.push(`  trecho ${k} (quadros ${plan[k].from}–${plan[k].to || names.length - 1}): ` +
                                          `${r.where}, ${r.frames} quadros, ${r.data.length} bytes`));
    return lines.join('\n');
}

module.exports = { encodeShared, WORKERS };
