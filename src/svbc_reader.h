#ifndef SVBC_READER_H
#define SVBC_READER_H

#include "svbc_format.h"

typedef struct {
    SVBC_Header header;
    SVBC_Color* palette; /* array de colors do codebook */
    SVBC_Node*  nodes;   /* array alocado pelo reader, liberado por svbc_free() */
} SVBCFile;

/* Lê arquivo .svbc completo. Retorna 0 em sucesso, -1 em erro. */
int  svbc_read(SVBCFile* out, const char* path);

/* Libera memória alocada pelo reader. */
void svbc_free(SVBCFile* f);

#endif /* SVBC_READER_H */
