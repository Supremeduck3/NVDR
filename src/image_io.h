#ifndef IMAGE_IO_H
#define IMAGE_IO_H

typedef struct {
    unsigned char* data;  // pixels RGB, row-major
    int width;
    int height;
    int channels;         // sempre 3 (RGB) após load
} Image;

// Carrega imagem de disco (PNG, JPG, BMP, etc.)
// Retorna 0 em sucesso, -1 em erro
int  image_load(Image* img, const char* path);

// Libera memória da imagem
void image_free(Image* img);

// Retorna ponteiro para o pixel (x, y) — sem bounds check
unsigned char* image_pixel(const Image* img, int x, int y);

#endif /* IMAGE_IO_H */
