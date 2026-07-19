const http = require('http');
const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');
const crypto = require('crypto');

const PORT = 3000;

// Whitelist of accepted file extensions. Any other value is rejected
// before reaching the file system — protects shell, sanitize path.
const ALLOWED_EXTS = new Set(['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif', '.tga']);

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

const server = http.createServer((req, res) => {
    if (req.method === 'GET' && (req.url === '/' || req.url === '/index.html')) {
        fs.readFile(path.join(__dirname, 'public', 'index.html'), (err, data) => {
            if (err) {
                res.writeHead(500);
                res.end("Error loading HTML");
            } else {
                res.writeHead(200, { 'Content-Type': 'text/html' });
                res.end(data);
            }
        });
    } else if (req.method === 'POST' && req.url === '/convert') {
        const fileExtension = sanitizeExtension(req.headers['x-file-ext']);
        if (!fileExtension) {
            res.writeHead(400, { 'Content-Type': 'text/plain' });
            res.end("Invalid or missing x-file-ext header");
            return;
        }
        const minTile = req.headers['x-min-tile'] || '-1';
        const homoThresh = req.headers['x-homo-thresh'] || '-1.0';
        const logoMode = req.headers['x-logo-mode'] === '1';

        const fileId = crypto.randomUUID();
        const inputPath = path.join(outDir, `temp_${fileId}${fileExtension}`);
        const outBase = path.join(outDir, `temp_${fileId}.svg`);
        const outSvbc = path.join(outDir, `temp_${fileId}.svbc`);
        const outSvbcz = path.join(outDir, `temp_${fileId}.svbcz`);

        // B2: best-effort cleanup helper. Each cleanup is fire-and-forget
        // unlink; failures are logged but do not block the call site.
        const safeUnlink = (p) => {
            fs.unlink(p, (err) => {
                if (err && err.code !== 'ENOENT') {
                    console.warn(`Could not delete ${p}: ${err.message}`);
                }
            });
        };

        const writeStream = fs.createWriteStream(inputPath);
        let aborted = false;

        writeStream.on('error', (err) => {
            aborted = true;
            console.error("Failed to write upload file:", err);
            res.writeHead(500);
            res.end("Upload write failed");
            safeUnlink(inputPath);
        });

        req.on('aborted', () => {
            aborted = true;
            safeUnlink(inputPath);
        });

        writeStream.on('finish', () => {
            if (aborted || res.writableEnded) {
                safeUnlink(inputPath);
                return;
            }
            console.log(`Starting conversion for ${inputPath} (Tile: ${minTile}, Thresh: ${homoThresh}, Logo: ${logoMode})...`);

            // B1: argv array passed to execFile — args are NOT parsed by
            // a shell, so header values can't inject commands. The exe
            // path is local and resolved via path.join.
            const exePath = path.join(__dirname, 'image_to_svg.exe');
            const args = [
                inputPath,
                outBase,
                minTile,
                homoThresh,
                '--format', 'svbc',
            ];
            if (logoMode) args.push('--logo');

            // B2: cleanup-on-error helper. Schedules deletion of any
            // intermediate files regardless of which step bailed.
            const cleanupAll = () => {
                setTimeout(() => {
                    [inputPath, outBase, outSvbc, outSvbcz].forEach(safeUnlink);
                }, 5000);
            };

            execFile(exePath, args, (error, stdout, stderr) => {
                if (error) {
                    console.error("Conversion failed:", error);
                    console.error("Stderr:", stderr);
                    if (!res.writableEnded) {
                        res.writeHead(500);
                        res.end(`Conversion failed: ${stderr || error.message}`);
                    }
                    cleanupAll();
                    return;
                }

                fs.readFile(outSvbc, (err, svbcData) => {
                    if (err) {
                        console.error("Failed to read SVBC file:", err);
                        if (!res.writableEnded) {
                            res.writeHead(500);
                            res.end("SVBC generation failed");
                        }
                        cleanupAll();
                        return;
                    }
                    res.writeHead(200, {
                        'Content-Type': 'application/octet-stream',
                        'Content-Length': svbcData.length
                    });
                    res.end(svbcData);
                    console.log(`Conversion completed. Sent ${svbcData.length} bytes.`);

                    // B2: deferred cleanup of every file we created,
                    // including the SVG (which may or may not exist).
                    setTimeout(() => {
                        [inputPath, outBase, outSvbc, outSvbcz].forEach(safeUnlink);
                    }, 5000);
                });
            });
        });
    } else {
        res.writeHead(404);
        res.end("Not found");
    }
});

server.listen(PORT, () => {
    console.log(`\n==========================================`);
    console.log(`🚀 SVBC Studio server running!`);
    console.log(`➡️  Access at: http://localhost:${PORT}/`);
    console.log(`==========================================\n`);
});
