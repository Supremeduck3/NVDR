/*
 * fuzz — hand the decoders containers that lie, and see what breaks.
 *
 * Every test before this one fed the decoders either valid files or valid
 * files cut short. A file that arrives over a network can also arrive
 * damaged, and a file that comes from somewhere untrusted can be damaged
 * on purpose. A decoder that trusts a length, a count or a bit it read
 * from the stream is one malformed byte from reading or writing outside
 * its buffers.
 *
 * This mutates real containers — flipping bits, overwriting bytes,
 * truncating, splicing header fields to extreme values — and decodes each
 * result from memory, still and sequence alike. Built with AddressSanitizer
 * and UndefinedBehaviorSanitizer, any out-of-bounds access, overflow or
 * leak stops it with the input that caused it written to disk.
 *
 * The random source is seeded, so a failure is reproducible from its
 * iteration number.
 *
 *     make fuzz && ./fuzz_nvdr samples/*.jpg --iters 20000
 */
#include "nvdr.h"
#include "nvdrv.h"
#include "nvda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 16);
}

static void mutate(uint8_t* d, size_t* len, size_t header) {
    size_t n = *len;
    if (n == 0) return;
    int kind = rnd() % 6;
    int count = 1 + rnd() % 8;
    for (int i = 0; i < count; i++) {
        size_t at = rnd() % n;
        switch (kind) {
        case 0: d[at] ^= (uint8_t)(1u << (rnd() % 8)); break;         /* bit flip */
        case 1: d[at] = (uint8_t)rnd(); break;                         /* byte */
        case 2: if (header) d[rnd() % header] = (uint8_t)rnd(); break; /* header */
        case 3: {                                                      /* extreme field */
            static const uint8_t extremes[] = { 0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF };
            if (header) d[rnd() % header] = extremes[rnd() % 6];
            break;
        }
        case 4: if (n > 1) { size_t cut = rnd() % n; *len = cut ? cut : 1; n = *len; } break;
        case 5: {                                                      /* copy a run */
            size_t from = rnd() % n, to = rnd() % n, k = 1 + rnd() % 64;
            if (from + k <= n && to + k <= n) memmove(d + to, d + from, k);
            break;
        }
        }
    }
}

static int decode_still(const uint8_t* d, size_t n) {
    /* Both layer caps, because layer 0 alone stops before the texture
     * decoder and the full decode goes through it. */
    int ok = 0;
    for (int layer = 0; layer < NVDR_LAYERS; layer++) {
        NvdrImage img;
        if (nvdr_decode_mem(d, n, layer, &img, NULL, NULL) != 0) continue;
        nvdr_image_free(&img);
        ok = 1;
    }
    return ok;
}

static int decode_sequence(const char* path) {
    NvdrvDecoder* dec;
    NvdrvInfo info;
    if (nvdrv_decode_open(&dec, path, &info) != 0) return 0;
    NvdrImage f;
    f.width = info.width; f.height = info.height;
    f.pixels = (unsigned char*)malloc((size_t)f.width * f.height * 3 + 1);
    int frames = 0, kind, partial;
    while (f.pixels && frames < 64 && nvdrv_decode_next(dec, &f, &kind, &partial) == 1) frames++;
    free(f.pixels);
    nvdrv_decode_close(dec);
    return frames;
}

/* An album decoded end to end, the fluid context carried through it. */
static int decode_album(const uint8_t* d, size_t n) {
    NvdaReader r;
    if (nvda_open(&r, d, n) != 0) return 0;
    int k = 0, rc, partial;
    char name[256];
    NvdrImage img;
    while (k < 16 && (rc = nvda_next(&r, name, sizeof name, &img, NULL, &partial)) == 1) {
        nvdr_image_free(&img);
        k++;
    }
    nvda_close(&r);
    return k;
}

static uint8_t* read_all(const char* path, size_t* n) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* d = (uint8_t*)malloc((size_t)s);
    *n = d ? fread(d, 1, (size_t)s, f) : 0;
    fclose(f);
    return d;
}

int main(int argc, char** argv) {
    long iters = 5000;
    const char* seq = NULL;
    int nseeds = 0;
    uint8_t* seeds[64];
    size_t seed_len[64];
    NvdrConfig cfg = nvdr_default_config();
    const char* album_paths[3];
    int nalbum = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) { iters = atol(argv[++i]); continue; }
        if (!strcmp(argv[i], "--seq") && i + 1 < argc) { seq = argv[++i]; continue; }
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) { rng_state ^= (uint64_t)atoll(argv[++i]) * 0x100000001B3ull; continue; }
        if (nseeds >= 64) continue;
        /* An image becomes seed containers by encoding it every way that
         * changes what the decoder walks: the default, a fine quantiser
         * (large coefficients, escapes), a coarse one (mostly flat
         * leaves), and a tree held to 8..16 px. A path no seed reaches is
         * a path the fuzzer never attacks. */
        NvdrImage img;
        if (nvdr_image_load(&img, argv[i]) != 0) { fprintf(stderr, "skip %s\n", argv[i]); continue; }
        if (nalbum < 3) album_paths[nalbum++] = argv[i];
        for (int variant = 0; variant < 4 && nseeds < 64; variant++) {
            NvdrConfig c = cfg;
            if (variant == 1) c.q = 1;
            if (variant == 2) c.q = 200;
            if (variant == 3) { c.min_block = 8; c.max_block = 16; c.band = 0; }
            if (variant == 2) c.band = 32;
            if (nvdr_encode_mem(&seeds[nseeds], &seed_len[nseeds], &img, &c, NULL) == 0) nseeds++;
        }
        nvdr_image_free(&img);
    }

    /* The first images as an album, the fluid context and prediction on,
     * with the last of them repeated so the seed holds a predicted entry
     * as well as coded ones. The repeat shares the pixels it copies. */
    uint8_t* album = NULL;
    size_t album_len = 0;
    if (nalbum >= 2) {
        NvdrImage imgs[4];
        const char* names[4];
        int ok = 1;
        for (int k = 0; k < nalbum; k++) { ok &= nvdr_image_load(&imgs[k], album_paths[k]) == 0; names[k] = album_paths[k]; }
        imgs[nalbum] = imgs[nalbum - 1];
        names[nalbum] = album_paths[nalbum - 1];
        NvdrConfig c = cfg;
        c.q = 60;   /* small seeds mutate into more distinct inputs */
        if (ok && nvda_write("fuzz_album_seed.nvda", nalbum + 1, names, imgs, &c, 1, 1, NULL, NULL, NULL) == 0)
            album = read_all("fuzz_album_seed.nvda", &album_len);
        remove("fuzz_album_seed.nvda");
        for (int k = 0; k < nalbum; k++) nvdr_image_free(&imgs[k]);
    }

    size_t seq_len = 0;
    uint8_t* seq_data = seq ? read_all(seq, &seq_len) : NULL;
    if (!nseeds && !seq_data) {
        fprintf(stderr, "usage: %s <images...> [--seq file.nvdrv] [--iters N] [--seed S]\n", argv[0]);
        return 2;
    }

    long decoded = 0, rejected = 0, seq_runs = 0, album_runs = 0;
    for (long it = 0; it < iters; it++) {
        if (album && rnd() % 4 == 0) {
            uint8_t* buf = (uint8_t*)malloc(album_len);
            memcpy(buf, album, album_len);
            size_t len = album_len;
            mutate(buf, &len, NVDA_HEADER_SIZE + 64);
            FILE* f = fopen("fuzz_last_input.bin", "wb");
            if (f) { fwrite(buf, 1, len, f); fclose(f); }
            album_runs++;
            if (decode_album(buf, len) > 0) decoded++; else rejected++;
            free(buf);
            continue;
        }
        int do_seq = seq_data && (!nseeds || (rnd() % 3 == 0));
        const uint8_t* src = do_seq ? seq_data : seeds[rnd() % nseeds];
        size_t n = do_seq ? seq_len : seed_len[src == seeds[0] ? 0 : 0];
        if (!do_seq) {
            int s = (int)(rnd() % nseeds);
            src = seeds[s]; n = seed_len[s];
        }
        uint8_t* buf = (uint8_t*)malloc(n);
        memcpy(buf, src, n);
        size_t len = n;
        mutate(buf, &len, do_seq ? NVDRV_HEADER_SIZE + NVDRV_FRAME_HEADER : NVDR_HEADER_SIZE);

        /* The input is on disk before it is decoded, so whatever crashes
         * leaves its cause behind. */
        FILE* f = fopen("fuzz_last_input.bin", "wb");
        if (f) { fwrite(buf, 1, len, f); fclose(f); }

        if (do_seq) { seq_runs++; if (decode_sequence("fuzz_last_input.bin") > 0) decoded++; else rejected++; }
        else if (decode_still(buf, len)) decoded++; else rejected++;
        free(buf);

        if ((it + 1) % 1000 == 0)
            fprintf(stderr, "  %ld iteracoes, %ld decodificados, %ld recusados, %ld sequencias, %ld albuns\n",
                    it + 1, decoded, rejected, seq_runs, album_runs);
    }
    remove("fuzz_last_input.bin");
    printf("%ld iteracoes sem falha de memoria (%ld decodificados, %ld recusados)\n",
           iters, decoded, rejected);
    for (int i = 0; i < nseeds; i++) free(seeds[i]);
    free(seq_data);
    free(album);
    return 0;
}
