#include "svg_writer.h"

int svg_open(SVGWriter* w, const char* path, int width, int height) {
    w->file = fopen(path, "w");
    if (!w->file) return -1;
    setvbuf(w->file, NULL, _IOFBF, 65536);  // 64KB buffer — reduz syscalls no hot path
    w->width  = width;
    w->height = height;

    /* shape-rendering="geometricPrecision" enables sub-pixel anti-aliasing
     * on curved and diagonal edges. The previous "crispEdges" value
     * explicitly disabled this, making every rect boundary look jagged. */
    fprintf(w->file,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" "
        "width=\"%d\" height=\"%d\" "
        "viewBox=\"0 0 %d %d\" "
        "shape-rendering=\"geometricPrecision\">\n",
        width, height, width, height);

    return 0;
}

void svg_close(SVGWriter* w) {
    if (w->file) {
        fprintf(w->file, "</svg>\n");
        fclose(w->file);
        w->file = NULL;
    }
}

void svg_begin_defs(SVGWriter* w) {
    fprintf(w->file, "<defs>\n");
}

void svg_end_defs(SVGWriter* w) {
    fprintf(w->file, "</defs>\n");
}

void svg_blur_filter(SVGWriter* w, const char* id, float stddev) {
    fprintf(w->file,
        "<filter id=\"%s\"><feGaussianBlur stdDeviation=\"%.1f\"/></filter>\n",
        id, stddev);
}

void svg_linear_gradient_v(SVGWriter* w, int id,
                           const char* top_color, const char* bot_color) {
    fprintf(w->file,
        "<linearGradient id=\"g%d\" x1=\"0\" y1=\"0\" x2=\"0\" y2=\"1\">"
        "<stop offset=\"0%%\" stop-color=\"%s\"/>"
        "<stop offset=\"100%%\" stop-color=\"%s\"/>"
        "</linearGradient>\n",
        id, top_color, bot_color);
}

void svg_begin_group(SVGWriter* w, const char* id, const char* filter_id) {
    if (filter_id) {
        fprintf(w->file, "<g id=\"%s\" filter=\"url(#%s)\">\n", id, filter_id);
    } else {
        fprintf(w->file, "<g id=\"%s\">\n", id);
    }
}

void svg_end_group(SVGWriter* w) {
    fprintf(w->file, "</g>\n");
}

void svg_rect(SVGWriter* w, int x, int y, int width, int height, const char* fill) {
    // Snap-to-Grid: exact integer coords, crispEdges prevents sub-pixel gaps
    fprintf(w->file,
        "<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" fill=\"%s\"/>\n",
        x, y, width, height, fill);
}

void svg_def_rect(SVGWriter* w, const char* id, int width, int height, const char* fill) {
    if (fill) {
        fprintf(w->file, "<rect id=\"%s\" width=\"%d\" height=\"%d\" fill=\"%s\"/>\n", 
                id, width, height, fill);
    } else {
        fprintf(w->file, "<rect id=\"%s\" width=\"%d\" height=\"%d\"/>\n", 
                id, width, height);
    }
}

void svg_use(SVGWriter* w, const char* ref_id, int x, int y) {
    fprintf(w->file, "<use href=\"#%s\" x=\"%d\" y=\"%d\"/>\n", ref_id, x, y);
}
