const http = require('http');
const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');
const crypto = require('crypto');

const PORT = 3000;
const MAX_UPLOAD_BYTES = 64 * 1024 * 1024;

// Whitelist of accepted file extensions. Any other value is rejected
// before reaching the file system — protects shell, sanitize path.
const ALLOWED_EXTS = new Set(['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif', '.tga']);

// Video goes through ffmpeg, which is looked up on PATH unless these say
// otherwise. It runs on the machine serving the page, not in the browser.
const VIDEO_EXTS = new Set(['.mp4', '.webm', '.mov', '.mkv', '.avi', '.m4v']);
const MAX_VIDEO_BYTES = 512 * 1024 * 1024;
const FFMPEG = process.env.NVDR_FFMPEG || 'ffmpeg';
const FFPROBE = process.env.NVDR_FFPROBE || 'ffprobe';

const PUBLIC_DIR = path.join(__dirname, 'public');
const STATIC_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8'
};

// Ensure output dir exists
const outDir = path.join(__dirname, 'output');
if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
}

// B1: extension passed by the client is validated against an allow-list
// to keep it from being interpreted as a shell fragment. Returns null
// on anything other than a known image extension.
function sanitizeExtension(raw, allowed = ALLOWED_EXTS) {
    if (typeof raw !== 'string') return null;
    const ext = raw.trim().toLowerCase();
    if (!allowed.has(ext)) return null;
    if (ext.includes('/') || ext.includes('\\') || ext.includes('\0')) return null;
    return ext;
}

// B2: best-effort cleanup helper. Each cleanup is fire-and-forget unlink;
// failures are logged but do not block the call site.
function safeUnlink(p) {
    fs.unlink(p, (err) => {
        if (err && err.code !== 'ENOENT') {
            console.warn(`Could not delete ${p}: ${err.message}`);
        }
    });
}

// Across platforms the built binaries differ by extension:
//   Windows (gcc/MinGW + `make`):  name.exe
//   Linux/macOS (gcc):              name
// Pick whichever exists. Returning null lets the caller answer with a
// clear 500 rather than an opaque ENOENT.
function findExecutable(dir, name) {
    for (const candidate of [path.join(dir, `${name}.exe`), path.join(dir, name)]) {
        try {
            if (fs.existsSync(candidate)) return candidate;
        } catch (e) { /* permission errors fall through */ }
    }
    return null;
}

/*
 * Stream an upload to disk, then hand the path to `onReady`.
 *
 * The pipe at the bottom is load-bearing: without it the request body is
 * never consumed, 'finish' never fires, and the connection hangs until the
 * client times out.
 */
function receiveUpload(req, res, onReady,
                       { exts = ALLOWED_EXTS, limit = MAX_UPLOAD_BYTES } = {}) {
    const fileExtension = sanitizeExtension(req.headers['x-file-ext'], exts);
    if (!fileExtension) {
        res.writeHead(400, { 'Content-Type': 'text/plain' });
        res.end('Invalid or missing x-file-ext header');
        return;
    }

    const fileId = crypto.randomUUID();
    const inputPath = path.join(outDir, `temp_${fileId}${fileExtension}`);
    const writeStream = fs.createWriteStream(inputPath);
    let aborted = false;
    let received = 0;

    writeStream.on('error', (err) => {
        aborted = true;
        console.error('Failed to write upload file:', err);
        if (!res.writableEnded) {
            res.writeHead(500);
            res.end('Upload write failed');
        }
        safeUnlink(inputPath);
    });

    req.on('aborted', () => {
        aborted = true;
        writeStream.destroy();
        safeUnlink(inputPath);
    });

    // Guard against unbounded uploads filling the disk.
    req.on('data', (chunk) => {
        received += chunk.length;
        if (received > limit && !aborted) {
            aborted = true;
            writeStream.destroy();
            if (!res.writableEnded) {
                res.writeHead(413, { 'Content-Type': 'text/plain' });
                res.end(`Upload too large (limit ${limit} bytes)`);
            }
            req.destroy();
            safeUnlink(inputPath);
        }
    });

    writeStream.on('finish', () => {
        if (aborted || res.writableEnded) {
            safeUnlink(inputPath);
            return;
        }
        onReady(inputPath, fileId);
    });

    req.pipe(writeStream);
}

/*
 * Run a converter and return its output file as an octet-stream. Every
 * intermediate file is scheduled for deletion whichever way the call
 * goes.
 */
function runConverter(res, { exePath, args, resultPath, artifacts, label }) {
    const cleanupAll = () => setTimeout(() => artifacts.forEach(safeUnlink), 5000);

    execFile(exePath, args, (error, stdout, stderr) => {
        if (error) {
            console.error(`${label} failed:`, error);
            console.error('Stderr:', stderr);
            if (!res.writableEnded) {
                res.writeHead(500);
                res.end(`Conversion failed: ${stderr || error.message}`);
            }
            cleanupAll();
            return;
        }

        fs.readFile(resultPath, (err, data) => {
            if (err) {
                console.error(`Failed to read ${label} output:`, err);
                if (!res.writableEnded) {
                    res.writeHead(500);
                    res.end(`${label} produced no output`);
                }
                cleanupAll();
                return;
            }
            // No Content-Encoding here on purpose: the container already
            // carries each level deflated on its own, and wrapping the whole
            // response in one more stream would undo the property that makes
            // a prefix decodable.
            res.writeHead(200, {
                'Content-Type': 'application/octet-stream',
                'Content-Length': data.length,
                'X-Encoder-Report': Buffer.from(stdout || '', 'utf8').toString('base64')
            });
            res.end(data);
            console.log(`${label} completed. Sent ${data.length} bytes.`);
            cleanupAll();
        });
    });
}

function serveStatic(req, res) {
    const requested = req.url === '/' ? '/index.html' : req.url.split('?')[0];
    const resolved = path.join(PUBLIC_DIR, path.normalize(requested));

    // path.normalize collapses "..", but the containment check is what
    // actually keeps a crafted URL inside public/.
    if (!resolved.startsWith(PUBLIC_DIR + path.sep)) {
        res.writeHead(403);
        res.end('Forbidden');
        return;
    }

    const type = STATIC_TYPES[path.extname(resolved).toLowerCase()];
    if (!type) {
        res.writeHead(404);
        res.end('Not found');
        return;
    }

    fs.readFile(resolved, (err, data) => {
        if (err) {
            res.writeHead(404);
            res.end('Not found');
            return;
        }
        res.writeHead(200, { 'Content-Type': type });
        res.end(data);
    });
}

/* An integer header inside [lo, hi], or the default. Headers are untrusted
 * and these end up in argv. */
function intHeader(req, name, def, lo, hi) {
    const raw = req.headers[name];
    if (!/^\d{1,5}$/.test(raw || '')) return def;
    const v = Number(raw);
    return v < lo ? lo : (v > hi ? hi : v);
}

/*
 * The source's frame rate, so the player runs at the speed it was shot.
 * ffprobe prints it as a fraction — 24000/1001 for film on video. Not every
 * ffmpeg install ships ffprobe, so failing that, ffmpeg's own description
 * of the input ("25 fps") is read instead, and 24 is the last resort.
 */
function probeFps(input, done) {
    const accept = (v) => (v >= 1 && v <= 255 ? Math.round(v) : null);
    execFile(FFPROBE, ['-v', 'error', '-select_streams', 'v:0',
                       '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', input],
             { timeout: 30000 }, (err, stdout) => {
        const m = /^(\d+)(?:\/(\d+))?/.exec((stdout || '').trim());
        const fps = !err && m ? accept(Number(m[1]) / Number(m[2] || 1)) : null;
        if (fps) return done(fps);
        // Without an output, ffmpeg describes the input and exits nonzero.
        execFile(FFMPEG, ['-hide_banner', '-i', input], { timeout: 30000 }, (_e, _o, stderr) => {
            const f = /(\d+(?:\.\d+)?) fps/.exec(stderr || '');
            done((f && accept(Number(f[1]))) || 24);
        });
    });
}

/*
 * Video in, sequence out: ffmpeg cuts the upload into frames, nvdrv_encode
 * codes them, and the container goes back to the page.
 *
 * Frames are taken as the source has them. Asking ffmpeg for a frame rate
 * makes it duplicate or drop frames to hit it, and a duplicated frame is a
 * perfect prediction — it would make the codec look better than it is.
 * Duration and width are capped because encoding is the slow direction and
 * the frames sit on disk as PNG while it runs.
 */
function handleVideo(req, res) {
    const seconds = intHeader(req, 'x-seconds', 5, 1, 30);
    const maxWidth = intHeader(req, 'x-max-width', 1280, 64, 3840);

    receiveUpload(req, res, (inputPath, fileId) => {
        const outPath = path.join(outDir, `temp_${fileId}.nvdrv`);
        const framesDir = fs.mkdtempSync(path.join(outDir, `frames_${fileId}_`));
        const cleanup = () => setTimeout(() => {
            safeUnlink(inputPath);
            safeUnlink(outPath);
            fs.rm(framesDir, { recursive: true, force: true }, () => {});
        }, 5000);
        const fail = (code, text) => {
            console.error(`Video: ${text}`);
            if (!res.writableEnded) {
                res.writeHead(code, { 'Content-Type': 'text/plain; charset=utf-8' });
                res.end(text);
            }
            cleanup();
        };

        const encoder = findExecutable(__dirname, 'nvdrv_encode');
        if (!encoder) return fail(500, `nvdrv_encode não encontrado em ${__dirname}. Rode \`make\`.`);

        probeFps(inputPath, (fps) => {
            const args = ['-hide_banner', '-loglevel', 'error', '-i', inputPath,
                          '-map', '0:v:0', '-t', String(seconds), '-frames:v', '900',
                          '-vf', `scale='min(${maxWidth},iw)':-2`,
                          '-f', 'image2', path.join(framesDir, 'f%04d.png')];
            console.log(`Video: extracting up to ${seconds}s at <=${maxWidth}px, ${fps} fps...`);
            execFile(FFMPEG, args, { timeout: 300000 }, (err, _out, stderr) => {
                if (err && err.code === 'ENOENT') {
                    return fail(501, 'ffmpeg não encontrado no PATH do servidor. Instale o ffmpeg ' +
                                     'ou aponte NVDR_FFMPEG para o executável.');
                }
                if (err) return fail(422, `ffmpeg não conseguiu ler o vídeo: ${(stderr || err.message).trim()}`);

                const frames = fs.readdirSync(framesDir).filter(n => n.endsWith('.png')).length;
                if (!frames) return fail(422, 'ffmpeg não extraiu nenhum quadro.');
                console.log(`Video: ${frames} frames, encoding...`);

                execFile(encoder, [framesDir, outPath, '--fps', String(fps)],
                         { timeout: 900000, maxBuffer: 8 * 1024 * 1024 }, (err2, stdout, stderr2) => {
                    if (err2) return fail(500, `nvdrv_encode falhou: ${(stderr2 || err2.message).trim()}`);
                    fs.readFile(outPath, (err3, data) => {
                        if (err3) return fail(500, 'nvdrv_encode não produziu saída');
                        // The per-frame report runs a line per frame; keep
                        // its head and its summary so the header stays small.
                        let lines = (stdout || '').trim().split('\n');
                        if (lines.length > 60)
                            lines = [...lines.slice(0, 42), `  … ${lines.length - 50} quadros …`, ...lines.slice(-8)];
                        res.writeHead(200, {
                            'Content-Type': 'application/octet-stream',
                            'Content-Length': data.length,
                            'X-Encoder-Report': Buffer.from(lines.join('\n'), 'utf8').toString('base64'),
                            'X-Frames': String(frames),
                            'X-Fps': String(fps)
                        });
                        res.end(data);
                        console.log(`Video: sent ${data.length} bytes for ${frames} frames.`);
                        cleanup();
                    });
                });
            });
        });
    }, { exts: VIDEO_EXTS, limit: MAX_VIDEO_BYTES });
}

const server = http.createServer((req, res) => {
    if (req.method === 'GET') {
        serveStatic(req, res);
        return;
    }

    if (req.method !== 'POST') {
        res.writeHead(404);
        res.end('Not found');
        return;
    }

    if (req.url === '/video') {
        handleVideo(req, res);
        return;
    }

    if (req.url === '/nvdr') {
        const quality = req.headers['x-q'];

        receiveUpload(req, res, (inputPath, fileId) => {
            const outNvdr = path.join(outDir, `temp_${fileId}.nvdr`);
            const artifacts = [inputPath, outNvdr];

            const exePath = findExecutable(__dirname, 'nvdr_encode');
            if (!exePath) {
                console.error('PRS encode failed: nvdr_encode not found');
                if (!res.writableEnded) {
                    res.writeHead(500);
                    res.end(`nvdr_encode not found in ${__dirname}. Build it with \`make\`.`);
                }
                setTimeout(() => artifacts.forEach(safeUnlink), 5000);
                return;
            }

            const args = [inputPath, outNvdr];
            // Only forward options that match the encoder's own grammar —
            // a header is untrusted input, and these reach argv directly.
            if (/^[1-9]\d{0,3}$/.test(quality || '')) args.push('--q', quality);

            console.log(`PRS encode for ${inputPath}...`);
            runConverter(res, {
                exePath, args, resultPath: outNvdr, artifacts, label: 'PRS'
            });
        });
        return;
    }

    res.writeHead(404);
    res.end('Not found');
});

server.listen(PORT, () => {
    console.log(`\n==========================================`);
    console.log(`🚀 NVDR server running!`);
    console.log(`➡️  http://localhost:${PORT}/`);
    console.log(`==========================================\n`);
});
