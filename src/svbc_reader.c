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

    /* Valida versão (suporta v0.3) */
    if (out->header.version != 3) {
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

    /* Lê todos os nós de uma vez */
    size_t read = fread(out->nodes, sizeof(SVBC_Node), out->header.node_count, f);
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
