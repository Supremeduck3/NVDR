#ifndef SVBC_WRITER_H
#define SVBC_WRITER_H

#include "svbc_format.h"
#include "quadtree.h"

/*
 * Escreve header + todos os nós folha da quadtree no caminho especificado.
 * layer_id é classificado por posição Y (nativo da quadtree).
 * Retorna 0 em sucesso, -1 em erro.
 */
int svbc_write(const char* path,
               const QuadTree* qt,
               int img_width, int img_height);

#endif /* SVBC_WRITER_H */

