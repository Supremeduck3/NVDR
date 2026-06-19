#ifndef COLOR_H
#define COLOR_H

#include "image_io.h"

// Calcula cor média de uma região (x, y, w, h) da imagem
void color_avg(const Image* img, int x, int y, int w, int h,
               unsigned char* r, unsigned char* g, unsigned char* b);

// Calcula homogeneidade da região (0.0 = uniforme, 1.0 = caótico)
// Versão perceptual BT.601 (pesa mais verde, depois vermelho, depois azul)
float color_homogeneity(const Image* img, int x, int y, int w, int h,
                        unsigned char avg_r, unsigned char avg_g, unsigned char avg_b);

// Formata cor como string hex CSS: "#rrggbb" ou "#rgb" (forma curta quando possível)
// buf deve ter pelo menos 8 bytes
void color_to_hex(unsigned char r, unsigned char g, unsigned char b, char* buf);

// Calcula gradiente vertical: média da metade superior vs inferior
// Retorna 1 se diferença > threshold, 0 caso contrário
int color_gradient_v(const Image* img, int x, int y, int w, int h,
                     unsigned char* top_r, unsigned char* top_g, unsigned char* top_b,
                     unsigned char* bot_r, unsigned char* bot_g, unsigned char* bot_b,
                     int threshold);
// Calcula gradiente horizontal: média da metade esquerda vs direita
// Retorna 1 se diferença > threshold, 0 caso contrário
int color_gradient_h(const Image* img, int x, int y, int w, int h,
                     unsigned char* left_r, unsigned char* left_g, unsigned char* left_b,
                     unsigned char* right_r, unsigned char* right_g, unsigned char* right_b,
                     int threshold);

// Edge Snapping: If region has a high-contrast edge, returns the dominant
// side's color instead of the blurred average. Returns 1 if snapped, 0 if not.
int color_edge_snap(const Image* img, int x, int y, int w, int h,
                    unsigned char* r, unsigned char* g, unsigned char* b,
                    int edge_threshold);

// Conversão RGB → YCbCr aproximada (BT.601)
// Usada internamente para homogeneidade perceptual e no SVBC writer
void rgb_to_ycbcr(unsigned char r, unsigned char g, unsigned char b,
                  int* Y, int* Cb, int* Cr);

#endif /* COLOR_H */
