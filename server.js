const http = require('http');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const crypto = require('crypto');

const PORT = 3000;

// Ensure output dir exists
const outDir = path.join(__dirname, 'output');
if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir);
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
        const fileExtension = req.headers['x-file-ext'] || '.jpg';
        const minTile = req.headers['x-min-tile'] || '-1';
        const homoThresh = req.headers['x-homo-thresh'] || '-1.0';
        const logoMode = req.headers['x-logo-mode'] === '1';
        
        const fileId = crypto.randomUUID();
        const inputPath = path.join(outDir, `temp_${fileId}${fileExtension}`);
        const outBase = path.join(outDir, `temp_${fileId}.svg`);
        const outSvbc = path.join(outDir, `temp_${fileId}.svbc`);
        const outSvbcz = path.join(outDir, `temp_${fileId}.svbcz`);

        const writeStream = fs.createWriteStream(inputPath);
        req.pipe(writeStream);

        writeStream.on('error', (err) => {
            console.error("Failed to write upload file:", err);
            res.writeHead(500);
            res.end("Upload write failed");
        });

        writeStream.on('finish', () => {
            console.log(`Starting conversion for ${inputPath} (Tile: ${minTile}, Thresh: ${homoThresh}, Logo: ${logoMode})...`);
            let command = `.\\image_to_svg.exe "${inputPath}" "${outBase}" ${minTile} ${homoThresh} --format svbc`;
            if (logoMode) command += ' --logo';
            
            exec(command, (error, stdout, stderr) => {
                if (error) {
                    console.error("Conversion failed:", error);
                    console.error("Stderr:", stderr);
                    res.writeHead(500);
                    res.end(`Conversion failed: ${stderr || error.message}`);
                    return;
                }
                
                fs.readFile(outSvbc, (err, svbcData) => {
                    if (err) {
                        console.error("Failed to read SVBC file:", err);
                        res.writeHead(500);
                        res.end("SVBC generation failed");
                        return;
                    }
                    res.writeHead(200, { 
                        'Content-Type': 'application/octet-stream',
                        'Content-Length': svbcData.length
                    });
                    res.end(svbcData);
                    console.log(`Conversion completed. Sent ${svbcData.length} bytes.`);

                    // Cleanup files
                    setTimeout(() => {
                        [inputPath, outSvbc, outSvbcz].forEach(p => {
                            fs.unlink(p, () => {});
                        });
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
