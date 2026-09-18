/*
 * nvdr_encode — build a PRS container and report what each level costs
 * and buys.
 *
 * The report is the point. The spec makes testable claims about the
 * residual stack (§1.3): that the anchor carries the bulk of the
 * information and that the residuals correcting it are far lower entropy.
 * Every run prints the numbers that confirm or refute them on real data,
 * rather than leaving them assumed.
 */
#include "nvdr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char* argv0) {
    fprintf(stderr,
        "usage: %s <image> <out.nvdr> [options]\n"
        "  --anchor-bits N    anchor palette is 2^N entries (default 4)\n"
        "  --tolerance A,B,C  per-level tolerance, coarse to fine\n"
        "                     (default 0.090,0.040,0.018)\n"
        "  --step B,C         residual quantisation step for levels 1 and 2\n"
        "                     (default 16,4)\n"
        "  --min-tile N       smallest tile edge (default 2)\n"
        "  --max-depth N      deepest subdivision (default 12)\n",
        argv0);
}

static int parse_floats(const char* text, float* out, int expected) {
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "%s", text);
    int n = 0;
    for (char* tok = strtok(buffer, ","); tok && n < expected; tok = strtok(NULL, ","))
        out[n++] = (float)atof(tok);
    return n == expected ? 0 : -1;
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 2; }

    const char* in_path  = argv[1];
    const char* out_path = argv[2];
    NvdrConfig cfg = nvdr_default_config();

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--anchor-bits") && i + 1 < argc) {
            cfg.anchor_bits = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--tolerance") && i + 1 < argc) {
            if (parse_floats(argv[++i], cfg.tolerance, NVDR_LEVELS) != 0) {
                fprintf(stderr, "--tolerance needs %d comma-separated values\n", NVDR_LEVELS);
                return 2;
            }
        } else if (!strcmp(argv[i], "--step") && i + 1 < argc) {
            float steps[NVDR_LEVELS - 1];
            if (parse_floats(argv[++i], steps, NVDR_LEVELS - 1) != 0) {
                fprintf(stderr, "--step needs %d comma-separated values\n", NVDR_LEVELS - 1);
                return 2;
            }
            for (int k = 1; k < NVDR_LEVELS; k++) cfg.step[k] = (int)steps[k - 1];
        } else if (!strcmp(argv[i], "--min-tile") && i + 1 < argc) {
            cfg.min_tile = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--max-depth") && i + 1 < argc) {
            cfg.max_depth = atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (cfg.anchor_bits < 1 || cfg.anchor_bits > 8) {
        fprintf(stderr, "anchor-bits must be between 1 and 8\n");
        return 2;
    }
    for (int k = 1; k < NVDR_LEVELS; k++) {
        if (cfg.step[k] < 1 || cfg.step[k] > 64) {
            fprintf(stderr, "residual steps must be between 1 and 64\n");
            return 2;
        }
        if (cfg.tolerance[k] > cfg.tolerance[k - 1]) {
            fprintf(stderr, "tolerances must decrease from coarse to fine\n");
            return 2;
        }
    }

    NvdrImage source;
    if (nvdr_image_load(&source, in_path) != 0) {
        fprintf(stderr, "cannot read image '%s'\n", in_path);
        return 1;
    }
    if (source.width > 65535 || source.height > 65535) {
        fprintf(stderr, "image exceeds the 65535px the container addresses\n");
        nvdr_image_free(&source);
        return 1;
    }

    NvdrHeader hdr;
    if (nvdr_encode_file(out_path, &source, &cfg, &hdr) != 0) {
        fprintf(stderr, "encode failed\n");
        nvdr_image_free(&source);
        return 1;
    }

    /* Read the container back and measure each level the way a consumer
     * would see it, rather than trusting the encoder's own state. */
    NvdrPyramid pyr;
    NvdrHeader read_hdr;
    if (nvdr_decode_file(out_path, &pyr, &read_hdr) != 0) {
        fprintf(stderr, "wrote a container that does not read back\n");
        nvdr_image_free(&source);
        return 1;
    }

    NvdrImage canvas;
    canvas.width = source.width;
    canvas.height = source.height;
    canvas.pixels = (unsigned char*)calloc((size_t)source.width * source.height * 3, 1);
    if (!canvas.pixels) {
        nvdr_pyramid_free(&pyr);
        nvdr_image_free(&source);
        return 1;
    }

    printf("%s  %dx%d\n", in_path, source.width, source.height);
    printf("  level      rects        raw     stored  cumulative     PSNR\n");
    size_t cumulative = NVDR_HEADER_SIZE;
    static const char* names[NVDR_LEVELS] = { "ANCHOR", "R1", "R2" };
    for (int k = 0; k < pyr.levels_present; k++) {
        nvdr_render_level(&pyr.level[k], &canvas);
        cumulative += hdr.stored_bytes[k];
        printf("  %-8s %8u %10u %10u  %10zu   %6.2f dB\n",
               names[k], pyr.level[k].count, hdr.raw_bytes[k],
               hdr.stored_bytes[k], cumulative, nvdr_psnr(&source, &canvas));
    }

    free(canvas.pixels);
    nvdr_pyramid_free(&pyr);
    nvdr_image_free(&source);
    return 0;
}
