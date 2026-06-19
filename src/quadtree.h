#ifndef QUADTREE_H
#define QUADTREE_H

#include "image_io.h"

typedef struct {
    int   min_tile_size;    // menor região possível em px (default: 4)
    int   max_depth;        // profundidade máxima da quadtree (default: 8)
    float homo_threshold;   // tolerância de homogeneidade 0.0–1.0 (default: 0.04)
    int   semantic_layers;  // agrupar em layers nomeadas (default: 1)
    
    // PSQ Multipliers
    float psq_sky_mult;
    float psq_midground_mult;
} SVGConfig;

typedef struct {
    int x, y, w, h;                     // bounds da região
    int depth;
    int is_leaf;
    int layer_id;                        // semantic layer identity
    unsigned char avg_r, avg_g, avg_b;   // cor média
    float homogeneity;                   // 0.0 = uniforme, 1.0 = caótico
    int children[4];                     // índices no pool (-1 = sem filho)
} QuadNode;

typedef struct {
    QuadNode* nodes;   // pool pré-alocado
    int count;
    int capacity;
} QuadTree;

// Aloca pool com capacidade para max_nodes nós
int  quadtree_init(QuadTree* qt, int max_nodes);

// Libera o pool de nós
void quadtree_free(QuadTree* qt);

// Forward declaration (full definition in sat.h)
struct SAT_tag;
typedef struct SAT_tag SAT;

// Constrói a quadtree a partir da imagem, usando SAT para queries O(1)
// Retorna índice do nó raiz (sempre 0), ou -1 em erro
int  quadtree_build(QuadTree* qt, const Image* img, const SAT* sat,
                    const SVGConfig* cfg);

#endif /* QUADTREE_H */
