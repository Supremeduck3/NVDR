/*
 * nvdr_png — an NVDR file dressed as a PNG, and back.
 *
 *   nvdr_png <in.nvdr|in.nvda|in.nvdrv> <out.png> [--preview N]
 *   nvdr_png --extract <in.png> <out>
 *
 * The PNG is an ordinary one: any viewer, browser or phone shows its
 * picture, a preview decoded from the file (the image, an album's first
 * photo, a video's first frame). The preview is as large as it can be, up
 * to 640 pixels on its longer side, while costing no more than the NVDR
 * file itself, so the PNG is at most twice the file: 256 colours by median
 * cut refined by k-means, Floyd-Steinberg dithered at 75% so smooth skin
 * and sky show no bands. On the six samples at q 24 that is previews of
 * 128 to 280 pixels and PNGs 2.02 times the NVDR, where a fixed 320
 * pixels without dithering made them 2.65 times and showed bands. The NVDR file
 * rides along untouched in a private ancillary chunk, `nvDr`, which PNG
 * readers skip:
 *
 *   n  lower case: ancillary, a reader may ignore it
 *   v  lower case: private, not a registered chunk
 *   D  upper case: as the PNG specification requires of the third letter
 *   r  lower case: safe to copy, an editor that keeps the picture may keep it
 *
 * The chunk comes straight after IHDR, so its data always starts at byte
 * 41 of the PNG (signature 8, IHDR 25, the chunk's length and type 8): a
 * viewer streaming the PNG can start decoding the NVDR inside it as soon
 * as it arrives, as nvdr-img.js does. What it holds names itself by its
 * own magic (NVDR, NVDA or NVDV). public/nvdr-png.js reads and writes the
 * same layout.
 *
 * Services that recompress what they are sent (most social networks and
 * chat apps) drop ancillary chunks, and what arrives is the preview alone.
 * E-mail, file hosting, and anything that passes the file as a file keep
 * it whole.
 */
#include "nvdr.h"
#include "nvda.h"
#include "nvdrv.h"
#include "pngwrap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t* read_all(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    size_t cap = 1 << 16, n = 0;
    uint8_t* buf = malloc(cap);
    for (size_t got; buf && (got = fread(buf + n, 1, cap - n, f)) > 0;) {
        n += got;
        if (n == cap) {
            uint8_t* grown = realloc(buf, cap *= 2);
            if (!grown) free(buf);
            buf = grown;
        }
    }
    fclose(f);
    *len = n;
    return buf;
}

/* The picture to show: the image, an album's first photo, a video's
 * first frame. */
static int preview_of(const char* path, const uint8_t* data, size_t len, NvdrImage* out) {
    if (len >= 4 && !memcmp(data, NVDR_MAGIC, 4)) {
        NvdrHeader hdr;
        NvdrDecodeInfo info;
        return nvdr_decode_mem(data, len, -1, out, &hdr, &info);
    }
    if (len >= 4 && !memcmp(data, NVDA_MAGIC, 4)) {
        NvdaReader r;
        if (nvda_open(&r, data, len) != 0) return -1;
        int partial = 0, ok = nvda_decode(&r, 0, out, &partial);
        nvda_close(&r);
        return ok == 1 ? 0 : -1;
    }
    if (len >= 4 && !memcmp(data, NVDRV_MAGIC, 4)) {
        NvdrvDecoder* dec;
        NvdrvInfo info;
        if (nvdrv_decode_open(&dec, path, &info) != 0) return -1;
        out->width = info.width;
        out->height = info.height;
        out->pixels = malloc((size_t)info.width * info.height * 3);
        int kind, partial, ok = out->pixels ? nvdrv_decode_next(dec, out, &kind, &partial) : -1;
        nvdrv_decode_close(dec);
        if (ok == 1) return 0;
        free(out->pixels);
        return -1;
    }
    return -1;
}

static int wrap(const char* in_path, const char* out_path, const NvdrPngOptions* o) {
    size_t len;
    uint8_t* data = read_all(in_path, &len);
    if (!data) { fprintf(stderr, "cannot read '%s'\n", in_path); return 1; }
    NvdrImage full;
    if (preview_of(in_path, data, len, &full) != 0) {
        fprintf(stderr, "'%s' is not an NVDR image, album or video this build can decode\n", in_path);
        free(data);
        return 1;
    }
    NvdrPngResult r;
    int ok = nvdr_png_write(out_path, data, len, &full, o, &r) == 0;
    nvdr_image_free(&full);
    if (ok) {
        printf("%s -> %s  %zu bytes: %zu NVDR + %zu preview %dx%d",
               in_path, out_path, r.bytes, len, r.bytes - len, r.width, r.height);
        if (r.dithered) printf(", dithered");
        if (r.colours) printf(", %d colours", r.colours);
        printf("\n");
    } else fprintf(stderr, "cannot write '%s'\n", out_path);
    free(data);
    return ok ? 0 : 1;
}

static int extract(const char* in_path, const char* out_path) {
    size_t len, at, n;
    uint8_t* png = read_all(in_path, &len);
    int found = png ? nvdr_png_find(png, len, &at, &n) : -1;
    if (found <= 0) {
        fprintf(stderr, found < 0 ? "'%s' is not a PNG, or its nvDr chunk is damaged\n"
                                  : "'%s' carries no NVDR file (no nvDr chunk)\n", in_path);
        free(png);
        return 1;
    }
    FILE* f = fopen(out_path, "wb");
    int ok = f && fwrite(png + at, 1, n, f) == n;
    if (f) ok = fclose(f) == 0 && ok;
    if (ok) printf("%s -> %s  %zu bytes\n", in_path, out_path, n);
    else fprintf(stderr, "cannot write '%s'\n", out_path);
    free(png);
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc >= 4 && !strcmp(argv[1], "--extract")) return extract(argv[2], argv[3]);
    NvdrPngOptions o = nvdr_png_options();
    if (argc < 3 || argv[1][0] == '-' || argv[2][0] == '-') {
        fprintf(stderr,
            "usage: %s <in.nvdr|in.nvda|in.nvdrv> <out.png> [options]\n"
            "       %s --extract <in.png> <out>\n"
            "  --preview N  the preview's longer side at most, in pixels (default %d, 0 whole)\n"
            "  --budget F   shrink the preview until it costs at most F times the NVDR\n"
            "               file (default %.1f; 0 keeps --preview's size)\n"
            "  --colours N  the preview's palette, 2..256 (default %d), 0 for full colour\n"
            "  --dither N   percent of the colour error diffused (default %d, 0 none)\n"
            "  --refine N   k-means rounds on the palette (default %d)\n",
            argv[0], argv[0], o.side, o.budget, o.colours, o.dither, o.refine);
        return 2;
    }
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--preview") && i + 1 < argc) o.side = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "--colours") || !strcmp(argv[i], "--colors")) && i + 1 < argc) o.colours = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--refine") && i + 1 < argc) o.refine = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dither") && i + 1 < argc) o.dither = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--budget") && i + 1 < argc) o.budget = atof(argv[++i]);
    }
    return wrap(argv[1], argv[2], &o);
}
