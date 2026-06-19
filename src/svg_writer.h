#ifndef SVG_WRITER_H
#define SVG_WRITER_H

#include <stdio.h>

typedef struct {
    FILE* file;
    int   width;
    int   height;
} SVGWriter;

// Abre arquivo SVG e escreve header
int  svg_open(SVGWriter* w, const char* path, int width, int height);

// Fecha tag </svg> e o arquivo
void svg_close(SVGWriter* w);

// Seção <defs> para filtros e gradientes
void svg_begin_defs(SVGWriter* w);
void svg_end_defs(SVGWriter* w);

// Filtro de blur para suavizar bordas
void svg_blur_filter(SVGWriter* w, const char* id, float stddev);

// Gradiente linear vertical (top → bottom)
void svg_linear_gradient_v(SVGWriter* w, int id,
                           const char* top_color, const char* bot_color);

// Abre grupo com filtro opcional (filter_id pode ser NULL)
void svg_begin_group(SVGWriter* w, const char* id, const char* filter_id);
void svg_end_group(SVGWriter* w);

// Define primitive for codebook in <defs>
void svg_def_rect(SVGWriter* w, const char* id, int width, int height, const char* fill);

// Escreve <rect> com cor sólida (com overlap de +1px)
void svg_rect(SVGWriter* w, int x, int y, int width, int height, const char* fill);

// Use a codebook primitive by ID
void svg_use(SVGWriter* w, const char* ref_id, int x, int y);

#endif /* SVG_WRITER_H */
