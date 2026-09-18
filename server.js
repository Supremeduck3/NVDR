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
function sanitizeExtension(raw) {
    if (typeof raw !== 'string') return null;
    const ext = raw.trim().toLowerCase();
    if (!ALLOWED_EXTS.has(ext)) return null;
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
function receiveUpload(req, res, onReady) {
    const fileExtension = sanitizeExtension(req.headers['x-file-ext']);
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
        if (received > MAX_UPLOAD_BYTES && !aborted) {
            aborted = true;
            writeStream.destroy();
            if (!res.writableEnded) {
                res.writeHead(413, { 'Content-Type': 'text/plain' });
                res.end(`Upload too large (limit ${MAX_UPLOAD_BYTES} bytes)`);
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

    // --- SVBC pipeline (src/, image_to_svg) ---
    if (req.url === '/convert') {
        const minTile = req.headers['x-min-tile'] || '-1';
        const homoThresh = req.headers['x-homo-thresh'] || '-1.0';
        const logoMode = req.headers['x-logo-mode'] === '1';

        receiveUpload(req, res, (inputPath, fileId) => {
            const outBase = path.join(outDir, `temp_${fileId}.svg`);
            const outSvbc = path.join(outDir, `temp_${fileId}.svbc`);
            const outSvbcz = path.join(outDir, `temp_${fileId}.svbcz`);
            const artifacts = [inputPath, outBase, outSvbc, outSvbcz];

            const exePath = findExecutable(__dirname, 'image_to_svg');
            if (!exePath) {
                console.error('Conversion failed: image_to_svg not found');
                if (!res.writableEnded) {
                    res.writeHead(500);
                    res.end(`image_to_svg not found in ${__dirname}. Build it with \`make\`.`);
                }
                setTimeout(() => artifacts.forEach(safeUnlink), 5000);
                return;
            }

            console.log(`SVBC conversion for ${inputPath} ` +
                        `(Tile: ${minTile}, Thresh: ${homoThresh}, Logo: ${logoMode})...`);

            const args = [inputPath, outBase, minTile, homoThresh, '--format', 'svbc'];
            if (logoMode) args.push('--logo');

            runConverter(res, {
                exePath, args, resultPath: outSvbc, artifacts, label: 'SVBC'
            });
        });
        return;
    }

    // --- PRS pipeline (nvdr/, nvdr_encode) ---
    if (req.url === '/nvdr') {
        const anchorBits = req.headers['x-anchor-bits'];
        const tolerance = req.headers['x-tolerance'];

        receiveUpload(req, res, (inputPath, fileId) => {
            const outNvdr = path.join(outDir, `temp_${fileId}.nvdr`);
            const artifacts = [inputPath, outNvdr];

            const exeDir = path.join(__dirname, 'nvdr');
            const exePath = findExecutable(exeDir, 'nvdr_encode');
            if (!exePath) {
                console.error('PRS encode failed: nvdr_encode not found');
                if (!res.writableEnded) {
                    res.writeHead(500);
                    res.end(`nvdr_encode not found in ${exeDir}. Build it with \`cd nvdr && make\`.`);
                }
                setTimeout(() => artifacts.forEach(safeUnlink), 5000);
                return;
            }

            const args = [inputPath, outNvdr];
            // Only forward options that match the encoder's own grammar —
            // a header is untrusted input, and these reach argv directly.
            if (/^[1-8]$/.test(anchorBits || '')) args.push('--anchor-bits', anchorBits);
            if (/^[\d.]+,[\d.]+,[\d.]+$/.test(tolerance || '')) args.push('--tolerance', tolerance);

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
    console.log(`➡️  SVBC studio:  http://localhost:${PORT}/`);
    console.log(`➡️  PRS viewer:   http://localhost:${PORT}/nvdr.html`);
    console.log(`==========================================\n`);
});
