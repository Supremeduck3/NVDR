#ifndef SVBC_FORMAT_H
#define SVBC_FORMAT_H

#include <stdint.h>

/*
 * SVBC v0.4 — SVG Binary Compression
 * Formato binário nativo do pipeline NVDR.
 *
 * Layout do arquivo (v0.4):
 *   [SVBC_Header      — 20 bytes]
 *   [SVBC_Color × codebook_count — 3 bytes cada]
 *   [plano x      — uint16 × node_count]
 *   [plano y      — uint16 × node_count]
 *   [plano w      — uint16 × node_count]
 *   [plano h      — uint16 × node_count]
 *   [plano token  — uint16 × node_count]
 *   [plano layer  — uint8  × node_count]
 *
 * v0.3 → v0.4 (mesmo tamanho não comprimido, ~2x menor depois do gzip):
 *   - Nós gravados em planos separados (structure-of-arrays) em vez de
 *     registros de 11 bytes intercalados. Cada plano vira uma sequência
 *     quase monotônica, que o deflate modela muito melhor do que campos
 *     alternando entre coordenada, cor e flag.
 *   - Nós ordenados em varredura (y, depois x) em vez de por área
 *     decrescente. A ordem por área destruía a localidade espacial e
 *     junto com ela a redundância que o compressor exploraria.
 *   Medido em samples/montanha_pessoas.jpg: 111 KB → 57 KB gzipado,
 *   sem perder um único bit de informação.
 *
 * SVBC_Node continua sendo a forma em memória (11 bytes); em disco os
 * campos são desmembrados nos planos acima.
 */

#pragma pack(push, 1)
typedef struct {
    uint8_t  magic[4];        // "SVBC"
    uint8_t  version;         // 3
    uint8_t  _pad[1];         // reservado
    uint16_t codebook_count;  // entradas no codebook (0–65535)
    uint16_t img_width;
    uint16_t img_height;
    uint32_t node_count;
    uint32_t _reserved;       // reservado para flags futuras
} SVBC_Header;                // 20 bytes
#pragma pack(pop)

_Static_assert(sizeof(SVBC_Header) == 20, "SVBC_Header size mismatch");

#pragma pack(push, 1)
typedef struct {
    uint8_t r, g, b;
} SVBC_Color;                 // 3 bytes
#pragma pack(pop)

// Nó (Node) - Upgrade v0.3
#pragma pack(push, 1)
typedef struct {
    uint16_t x, y, w, h;    // 8 bytes
    uint16_t token_id;      // 2 bytes — índice no codebook (0-65535)
    uint8_t  layer_id;      // 1 byte — see encoding below
} SVBC_Node;                // 11 bytes
#pragma pack(pop)

_Static_assert(sizeof(SVBC_Node) == 11, "SVBC_Node size mismatch");

/*
 * layer_id encoding (1 byte):
 *   bits [0:5] = layer index (0-63), e.g. 0=sky, 1=midground, 2=foreground
 *   bit  [6]   = BLEND_RIGHT: gradient blend with right neighbor
 *   bit  [7]   = BLEND_BELOW: gradient blend with neighbor below
 *
 * Backward compatible: readers that ignore high bits get correct layer_id.
 */
#define SVBC_BLEND_RIGHT  0x40
#define SVBC_BLEND_BELOW  0x80
#define SVBC_LAYER_MASK   0x3F

#endif /* SVBC_FORMAT_H */
