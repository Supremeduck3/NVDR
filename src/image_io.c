#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "image_io.h"

int image_load(Image* img, const char* path) {
    // Força 3 canais (RGB) independente do formato original
    img->data = stbi_load(path, &img->width, &img->height, &img->channels, 3);
    if (!img->data) {
        return -1;
    }
    img->channels = 3;
    return 0;
}

void image_free(Image* img) {
    if (img->data) {
        stbi_image_free(img->data);
        img->data = NULL;
    }
    img->width    = 0;
    img->height   = 0;
    img->channels = 0;
}

unsigned char* image_pixel(const Image* img, int x, int y) {
    return img->data + (y * img->width + x) * img->channels;
}
