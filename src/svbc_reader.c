#include "svbc_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int svbc_read(SVBCFile* out, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    out->palette = NULL;
    out->nodes = NULL;

    /* Lê header */
    if (fread(&out->header, sizeof(SVBC_Header), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    /* Valida magic */
    if (memcmp(out->header.magic, "SVBC", 4) != 0) {
        fclose(f);
        return -1;
    }

    /* Valida versão (suporta v0.3 intercalado e v0.4 planar) */
    if (out->header.version != 3 && out->header.version != 4) {
        fclose(f);
        return -1;
    }

    /* Allocation protection */
    if (out->header.codebook_count > 0) {
        out->palette = (SVBC_Color*)malloc((size_t)out->header.codebook_count * sizeof(SVBC_Color));
        if (!out->palette) {
            fclose(f);
            return -1;
        }
        if (fread(out->palette, sizeof(SVBC_Color), out->header.codebook_count, f) != out->header.codebook_count) {
            free(out->palette);
            out->palette = NULL;
            fclose(f);
            return -1;
        }
    }

    if (out->header.node_count == 0) {
        fclose(f);
        return 0;
    }

    out->nodes = (SVBC_Node*)malloc((size_t)out->header.node_count * sizeof(SVBC_Node));
    if (!out->nodes) {
        if (out->palette) { free(out->palette); out->palette = NULL; }
        fclose(f);
        return -1;
    }

    size_t read;
    if (out->header.version >= 4) {
        /* Payload planar: reconstrói os registros a partir dos planos. */
        uint32_t n = out->header.node_count;
        uint16_t* plane16 = (uint16_t*)malloc((size_t)n * sizeof(uint16_t));
        uint8_t*  plane8  = (uint8_t*)malloc((size_t)n);
        read = 0;
        if (plane16 && plane8) {
            int ok = 1;
            #define SVBC_READ_PLANE16(field)                                 \
                do {                                                         \
                    if (ok && fread(plane16, sizeof(uint16_t), n, f) != n) { \
                        ok = 0;                                              \
                    } else if (ok) {                                         \
                        for (uint32_t k = 0; k < n; k++)                     \
                            out->nodes[k].field = plane16[k];                \
                    }                                                        \
                } while (0)

            SVBC_READ_PLANE16(x);
            SVBC_READ_PLANE16(y);
            SVBC_READ_PLANE16(w);
            SVBC_READ_PLANE16(h);
            SVBC_READ_PLANE16(token_id);
            #undef SVBC_READ_PLANE16

            if (ok && fread(plane8, 1, n, f) != n) ok = 0;
            if (ok) {
                for (uint32_t k = 0; k < n; k++) out->nodes[k].layer_id = plane8[k];
                read = n;
            }
        }
        free(plane16);
        free(plane8);
    } else {
        /* v0.3: registros intercalados de 11 bytes. */
        read = fread(out->nodes, sizeof(SVBC_Node), out->header.node_count, f);
    }
    fclose(f);

    if (read != out->header.node_count) {
        if (out->palette) { free(out->palette); out->palette = NULL; }
        free(out->nodes);
        out->nodes = NULL;
        return -1;
    }

    return 0;
}

void svbc_free(SVBCFile* f) {
    if (f->palette) {
        free(f->palette);
        f->palette = NULL;
    }
    if (f->nodes) {
        free(f->nodes);
        f->nodes = NULL;
    }
    memset(&f->header, 0, sizeof(SVBC_Header));
}
