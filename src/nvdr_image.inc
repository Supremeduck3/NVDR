/* Part of nvdr.c, which includes it:
 * image files in and out: stb_image, EXIF orientation, PPM and PNG. */

/* ================================================================ image */

#ifndef NVDR_WASM
/*
 * A camera stores a photo the way its sensor lies and says in EXIF which
 * way is up; stb_image decodes the pixels and does not read that. The
 * orientation of a JPEG, 1 to 8 as EXIF numbers them (1: as stored), from
 * tag 0x0112 of the first directory in its APP1 segment.
 */
int nvdr_exif_orientation(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 1;
    uint8_t b[65536 + 4];
    int found = 1;
    if (fread(b, 1, 2, f) == 2 && b[0] == 0xff && b[1] == 0xd8) {
        for (;;) {
            if (fread(b, 1, 4, f) != 4 || b[0] != 0xff) break;
            size_t len = ((size_t)b[2] << 8 | b[3]);
            if (len < 2 || b[1] == 0xda) break;   /* the scan: no APP1 before it */
            len -= 2;
            if (b[1] != 0xe1) { if (fseek(f, (long)len, SEEK_CUR) != 0) break; continue; }
            if (fread(b, 1, len, f) != len) break;
            if (len < 14 || memcmp(b, "Exif\0\0", 6) != 0) continue;   /* XMP is an APP1 too */
            const uint8_t* t = b + 6;
            size_t n = len - 6;
            int le = t[0] == 'I' && t[1] == 'I';
            if (!le && !(t[0] == 'M' && t[1] == 'M')) break;
            #define EX16(p) (le ? (unsigned)((p)[0] | (p)[1] << 8) : (unsigned)((p)[0] << 8 | (p)[1]))
            #define EX32(p) (le ? (size_t)EX16(p) | (size_t)EX16((p) + 2) << 16 : (size_t)EX16(p) << 16 | EX16((p) + 2))
            size_t ifd = EX32(t + 4);
            if (ifd + 2 > n) break;
            unsigned count = EX16(t + ifd);
            for (unsigned i = 0; i < count && ifd + 2 + (size_t)(i + 1) * 12 <= n; i++) {
                const uint8_t* e = t + ifd + 2 + (size_t)i * 12;
                if (EX16(e) == 0x0112 && EX16(e + 2) == 3) {
                    unsigned v = EX16(e + 8);
                    if (v >= 1 && v <= 8) found = (int)v;
                }
            }
            #undef EX16
            #undef EX32
            break;
        }
    }
    fclose(f);
    return found;
}

/* The picture turned the way orientation `o` says it is seen. */
int nvdr_image_orient(NvdrImage* img, int o) {
    if (o <= 1) return 0;
    int w = img->width, h = img->height, W = o >= 5 ? h : w, H = o >= 5 ? w : h;
    unsigned char* out = (unsigned char*)malloc((size_t)W * H * 3);
    if (!out) return -1;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int sx, sy;
            switch (o) {
                case 2: sx = w - 1 - x; sy = y; break;           /* mirrored */
                case 3: sx = w - 1 - x; sy = h - 1 - y; break;   /* upside down */
                case 4: sx = x; sy = h - 1 - y; break;
                case 5: sx = y; sy = x; break;
                case 6: sx = y; sy = h - 1 - x; break;           /* held upright, the usual one */
                case 7: sx = w - 1 - y; sy = h - 1 - x; break;
                default: sx = w - 1 - y; sy = x; break;          /* 8 */
            }
            memcpy(out + ((size_t)y * W + x) * 3, img->pixels + ((size_t)sy * w + sx) * 3, 3);
        }
    free(img->pixels);
    img->pixels = out; img->width = W; img->height = H;
    return 0;
}

int nvdr_image_load(NvdrImage* img, const char* path) {
    int channels;
    img->pixels = stbi_load(path, &img->width, &img->height, &channels, 3);
    if (!img->pixels) return -1;
    /* As a viewer shows it: the way up its EXIF says. */
    if (nvdr_image_orient(img, nvdr_exif_orientation(path)) != 0) { free(img->pixels); img->pixels = NULL; return -1; }
    return 0;
}

int nvdr_image_write_ppm(const NvdrImage* img, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P6\n%d %d\n255\n", img->width, img->height);
    size_t n = (size_t)img->width * img->height * 3;
    size_t written = fwrite(img->pixels, 1, n, f);
    fclose(f);
    return written == n ? 0 : -1;
}

/*
 * PNG writer. PPM is trivial to emit but opens in almost nothing, which
 * makes eyeballing a decode harder than it needs to be. zlib is already
 * linked for the container, so a real PNG costs one deflate and three
 * CRCs.
 */
static void png_chunk(FILE* f, const char* type, const uint8_t* data, size_t len) {
    uint8_t size[4] = {
        (uint8_t)(len >> 24), (uint8_t)(len >> 16), (uint8_t)(len >> 8), (uint8_t)len
    };
    fwrite(size, 1, 4, f);
    fwrite(type, 1, 4, f);
    if (len) fwrite(data, 1, len, f);

    uLong crc = crc32(0, (const Bytef*)type, 4);
    if (len) crc = crc32(crc, (const Bytef*)data, (uInt)len);
    uint8_t crc_bytes[4] = {
        (uint8_t)(crc >> 24), (uint8_t)(crc >> 16), (uint8_t)(crc >> 8), (uint8_t)crc
    };
    fwrite(crc_bytes, 1, 4, f);
}

int nvdr_image_write_png(const NvdrImage* img, const char* path) {
    /* Every scanline is prefixed with filter type 0 — no prediction. A
     * filter would shrink the file; nothing here depends on its size. */
    size_t stride = (size_t)img->width * 3;
    size_t raw_size = (stride + 1) * (size_t)img->height;
    uint8_t* raw = (uint8_t*)malloc(raw_size);
    if (!raw) return -1;
    for (int y = 0; y < img->height; y++) {
        raw[(stride + 1) * (size_t)y] = 0;
        memcpy(raw + (stride + 1) * (size_t)y + 1, img->pixels + stride * (size_t)y, stride);
    }

    uLongf packed_size = compressBound((uLong)raw_size);
    uint8_t* packed = (uint8_t*)malloc(packed_size);
    if (!packed || compress2(packed, &packed_size, raw, (uLong)raw_size, 6) != Z_OK) {
        free(raw); free(packed);
        return -1;
    }
    free(raw);

    FILE* f = fopen(path, "wb");
    if (!f) { free(packed); return -1; }

    static const uint8_t signature[8] = { 137, 'P', 'N', 'G', 13, 10, 26, 10 };
    fwrite(signature, 1, 8, f);

    uint8_t ihdr[13];
    uint32_t w = (uint32_t)img->width, h = (uint32_t)img->height;
    ihdr[0] = (uint8_t)(w >> 24); ihdr[1] = (uint8_t)(w >> 16);
    ihdr[2] = (uint8_t)(w >> 8);  ihdr[3] = (uint8_t)w;
    ihdr[4] = (uint8_t)(h >> 24); ihdr[5] = (uint8_t)(h >> 16);
    ihdr[6] = (uint8_t)(h >> 8);  ihdr[7] = (uint8_t)h;
    ihdr[8] = 8;    /* bit depth */
    ihdr[9] = 2;    /* truecolour */
    ihdr[10] = ihdr[11] = ihdr[12] = 0;
    png_chunk(f, "IHDR", ihdr, sizeof(ihdr));
    png_chunk(f, "IDAT", packed, packed_size);
    png_chunk(f, "IEND", NULL, 0);

    fclose(f);
    free(packed);
    return 0;
}

int nvdr_image_write(const NvdrImage* img, const char* path) {
    size_t len = strlen(path);
    if (len >= 4 && strcmp(path + len - 4, ".png") == 0)
        return nvdr_image_write_png(img, path);
    return nvdr_image_write_ppm(img, path);
}
#endif

void nvdr_image_free(NvdrImage* img) {
    free(img->pixels);
    img->pixels = NULL;
    img->width = img->height = 0;
}

double nvdr_psnr(const NvdrImage* a, const NvdrImage* b) {
    if (a->width != b->width || a->height != b->height) return -1.0;
    size_t n = (size_t)a->width * a->height * 3;
    double mse = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a->pixels[i] - (double)b->pixels[i];
        mse += d * d;
    }
    mse /= (double)n;
    return mse <= 0.0 ? 99.0 : 10.0 * log10(255.0 * 255.0 / mse);
}
