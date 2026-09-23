/*
 * convert <in> <out.png|out.ppm> [x0 y0]: an image read the way the
 * encoder reads it (stb_image), written losslessly, optionally without
 * its first x0 columns and y0 rows. The codec benchmark gives every codec
 * the same pixels this way: a browser's JPEG decoder and stb's can differ,
 * a PNG cannot. The crop is for sources that are themselves JPEGs: off
 * their 8x8 grid, re-encoding as JPEG no longer lines up with the blocks
 * the source was made of, which would flatter it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nvdr.h"

int main(int argc, char** argv) {
    if (argc != 3 && argc != 5) {
        fprintf(stderr, "usage: %s <in> <out.png|out.ppm> [x0 y0]\n", argv[0]);
        return 2;
    }
    NvdrImage img;
    if (nvdr_image_load(&img, argv[1]) != 0) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    int x0 = argc == 5 ? atoi(argv[3]) : 0, y0 = argc == 5 ? atoi(argv[4]) : 0;
    if (x0 < 0 || y0 < 0 || x0 >= img.width || y0 >= img.height) {
        fprintf(stderr, "crop outside the image\n");
        nvdr_image_free(&img);
        return 1;
    }
    int w = img.width - x0, h = img.height - y0;
    for (int y = 0; y < h; y++)
        memmove(img.pixels + (size_t)y * w * 3, img.pixels + ((size_t)(y + y0) * img.width + x0) * 3, (size_t)w * 3);
    img.width = w; img.height = h;
    int rc = nvdr_image_write(&img, argv[2]);
    nvdr_image_free(&img);
    return rc == 0 ? 0 : 1;
}
