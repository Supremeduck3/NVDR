# SVBC — SVG Binary Compression
## Especificação v0.3 (Fluid Codebook) — implementação atual

> **Status:** Esta spec descreve a v0.3 implementada em `src/svbc_format.h`.
> A evolução v0.2 → v0.3 é apenas a elevação de `token_id` para `uint16_t`
> (65536 cores em vez de 256). Compatibilidade preservada na prática:
> o formato tem versionamento no header (`version=3`), readers que só
> entendem `codebook_count<=255` continuam funcionando enquanto a
> maioria dos codebooks couberem em uma única byte.

## Resumo das mudanças v0.2 → v0.3

|                  | v0.2            | v0.3 (atual)                             |
|------------------|-----------------|------------------------------------------|
| Bytes por nó     | 10              | 11                                       |
| `token_id`       | uint8 (256)     | uint16 (65536)                           |
| Codebook max     | 256 entradas    | 65535 entradas (16-bit field)            |
| Header           | 20 bytes        | 20 bytes (sem mudança)                  |
| Restante         | idem            | idem                                     |

Economia estimada em 36.000 nós com codebook cheio: **73 KB antes de qualquer compressão** (vs 72 KB da v0.2 — +1 byte/nó é compensado pela redução de quantização nos nós que precisariam de fallback).

> Decisão: BANDING em fotos naturais com 256 cores é ruim o suficiente
> para exigir N > 256. v0.3 com uint16 cobre praticamente qualquer
> imagem em prática. O custo (+1 byte/nó ~36 KB em 36.000 nós) é
> aceitável dado que isso elimina o passo de fallback de quantização.

---

## Resumo das mudanças v0.1 → v0.2 (original)

|                  | v0.1 | v0.2 |
|------------------|------|------|
| Bytes por nó     | 12   | 10   |
| Cor no nó        | RGB cru (3 bytes) | token_id (1 byte) |
| Codebook         | nenhum | até 256 entradas RGB |
| PSQ              | não   | sim (threshold por layer) |
| PRS              | não   | sim (ordenação por área) |
| Header           | 16 bytes | 20 bytes |

Economia estimada em 36.000 nós: **72 KB antes de qualquer compressão**.

---

## Resumo das mudanças v0.1 → v0.2

| | v0.1 | v0.2 |
|---|---|---|
| Bytes por nó | 12 | 10 |
| Cor no nó | RGB cru (3 bytes) | token_id (1 byte) |
| Codebook | nenhum | até 256 entradas RGB |
| PSQ | não | sim (threshold por layer) |
| PRS | não | sim (ordenação por área) |
| Header | 16 bytes | 20 bytes |

Economia estimada em 36.000 nós: **72 KB antes de qualquer compressão**.

---

## Formato do arquivo `.svbc` v0.2

```
[SVBC_Header   — 20 bytes         ]
[SVBC_Color × codebook_count — 3 bytes cada]
[SVBC_Node  × node_count     — 10 bytes cada]
```

---

## Structs C

### Header (20 bytes)

```c
#pragma pack(push, 1)
typedef struct {
    uint8_t  magic[4];        // "SVBC"
    uint8_t  version;         // 2
    uint8_t  _pad[1];         // reservado
    uint16_t codebook_count;  // entradas no codebook (0–256)
    uint16_t img_width;
    uint16_t img_height;
    uint32_t node_count;
    uint32_t _reserved;       // reservado para flags futuras
} SVBC_Header;                // 20 bytes
#pragma pack(pop)

_Static_assert(sizeof(SVBC_Header) == 20, "SVBC_Header size mismatch");
```

---

### Codebook Entry (3 bytes)

```c
#pragma pack(push, 1)
typedef struct {
    uint8_t r, g, b;
} SVBC_Color;                 // 3 bytes
#pragma pack(pop)
```

Posição no arquivo: imediatamente após o header.
Tamanho total da seção: `codebook_count × 3` bytes.

---

### Node (10 bytes) — ⚠️ ordem importa para alinhamento

```c
#pragma pack(push, 1)
typedef struct {
    uint16_t x, y, w, h;   // 8 bytes — todos uint16_t juntos, alinhados em 2
    uint8_t  layer_id;      // 1 byte
    uint8_t  token_id;      // 1 byte — índice no codebook
} SVBC_Node;                // 10 bytes
#pragma pack(pop)

_Static_assert(sizeof(SVBC_Node) == 10, "SVBC_Node size mismatch");
```

> **Por que essa ordem?** Todos os `uint16_t` vêm primeiro, garantindo que as leituras de 2 bytes ocorram em offsets pares (0, 2, 4, 6). Os dois `uint8_t` vêm no final em offsets 8 e 9. Isso evita undefined behavior em ARM e leituras desalinhadas em x86 mesmo sem `#pragma pack`.

---

## Mudança 1 — Fluid Codebook

### O que é
Em vez de armazenar RGB cru em cada nó (3 bytes), construímos uma paleta de até 256 cores e cada nó armazena apenas o índice (1 byte).

### Construção da paleta (em `svbc_writer.c`)

```c
// 1. Coleta todas as cores únicas dos nós folha
// 2. Se unique_count <= max_colors: paleta exata, sem perda
// 3. Se unique_count > max_colors: quantização por k-means ou
//    median cut (parâmetro configurável)
// 4. Cada nó recebe o token_id da cor mais próxima na paleta
```

### Parâmetro configurável — não fixar em 256

A proposta original fixava em 256 cores. **Isso é agressivo demais para fotos naturais** — uma montanha com gradiente de céu terá banding visível com 256 cores.

```c
typedef struct {
    // ... campos anteriores ...
    int   codebook_size;    // default: 256 para logos/ícones
                            //          1024 para fotos (usa uint16_t token_id)
    float codebook_quality; // 0.0 = velocidade máxima, 1.0 = qualidade máxima
} SVGConfig;
```

> **Decisão de design em aberto:** usar `uint8_t token_id` (máx 256 cores, 1 byte) ou `uint16_t token_id` (máx 65536 cores, 2 bytes)? A v0.2 usa `uint8_t` para manter 10 bytes/nó. Para fotos naturais, uma v0.3 pode elevar para `uint16_t` e 11 bytes/nó se o banding for inaceitável nos testes.

---

## Mudança 2 — PSQ (Perceptual Semantic Quantization)

Threshold de homogeneidade variável por layer. Regiões distantes (céu) aceitam mais imprecisão de cor; regiões próximas (foreground) exigem mais fidelidade.

```c
// Em quadtree.c — substituir threshold fixo por threshold por layer
static float psq_threshold(float base_threshold, int layer_id) {
    switch (layer_id) {
        case LAYER_SKY:        return base_threshold * 4.0f;  // 4× mais tolerante
        case LAYER_MIDGROUND:  return base_threshold * 2.0f;  // 2× mais tolerante
        case LAYER_FOREGROUND: return base_threshold * 1.0f;  // sem alteração
        default:               return base_threshold;
    }
}
```

Os multiplicadores `4.0` e `2.0` são o ponto de partida — precisam de calibração em imagens reais. Adicionar ao `SVGConfig`:

```c
float psq_sky_mult;        // default: 4.0
float psq_midground_mult;  // default: 2.0
```

---

## Mudança 3 — Coalescing por layer (threshold concreto)

A proposta original dizia "flexibilidade perceptiva maior" sem número. Aqui está o número:

```c
// Em optimizer.c
static float coalesce_threshold_for_layer(int layer_id, float base) {
    switch (layer_id) {
        case LAYER_SKY:        return base * 3.0f;
        case LAYER_MIDGROUND:  return base * 1.5f;
        case LAYER_FOREGROUND: return base * 1.0f;
        default:               return base;
    }
}
```

Esses valores (`3.0`, `1.5`) são estimativas iniciais — calibrar comparando LPIPS antes/depois na imagem de montanha.

---

## Mudança 4 — PRS (Progressive Residual Sorting)

Ordenar os nós no arquivo do maior para o menor por área (`w × h`) antes de gravar.

```c
// Em svbc_writer.c, antes do fwrite dos nós:
static int cmp_node_area_desc(const void* a, const void* b) {
    const SVBC_Node* na = a;
    const SVBC_Node* nb = b;
    int area_a = na->w * na->h;
    int area_b = nb->w * nb->h;
    return area_b - area_a;  // decrescente
}
qsort(nodes, node_count, sizeof(SVBC_Node), cmp_node_area_desc);
```

**Efeito no browser:** o Canvas desenha primeiro os blocos grandes (estrutura da imagem aparece imediatamente) e depois os detalhes finos. Perceived performance equivalente ao progressive JPEG — sem custo de bytes.

**Render JS atualizado para ler codebook:**

```javascript
const res  = await fetch("imagem.svbc");
const buf  = await res.arrayBuffer();
const view = new DataView(buf);

// Header
const codebookCount = view.getUint16(6,  true);
const nodeCount     = view.getUint32(12, true);

// Lê codebook
const HEADER_SIZE = 20;
const palette = [];
for (let i = 0; i < codebookCount; i++) {
    const off = HEADER_SIZE + i * 3;
    palette.push([
        view.getUint8(off),
        view.getUint8(off + 1),
        view.getUint8(off + 2)
    ]);
}

// Lê e renderiza nós (maior → menor por PRS)
const canvas = document.getElementById("c");
const ctx    = canvas.getContext("2d");
const NODES_OFFSET = HEADER_SIZE + codebookCount * 3;
const NODE_SIZE    = 10;

for (let i = 0; i < nodeCount; i++) {
    const off      = NODES_OFFSET + i * NODE_SIZE;
    const x        = view.getUint16(off,     true);
    const y        = view.getUint16(off + 2, true);
    const w        = view.getUint16(off + 4, true);
    const h        = view.getUint16(off + 6, true);
    // layer_id = view.getUint8(off + 8) — ignorado no render básico
    const token_id = view.getUint8(off + 9);
    const [r, g, b] = palette[token_id];
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(x, y, w, h);
}
```

---

## Problemas identificados na proposta original

| # | Problema | Status |
|---|---|---|
| 1 | `uint8_t w, h` estoura em >255px | Corrigido na v0.1, mantido na v0.2 |
| 2 | Sem header | Corrigido na v0.1, expandido na v0.2 |
| 3 | `SVBC_Node` com bytes misturados quebraria alinhamento em ARM | **Corrigido na v0.2** — todos `uint16_t` antes dos `uint8_t` |
| 4 | `_Static_assert` não mencionado na proposta de upgrade | **Corrigido na v0.2** — asserts atualizados |
| 5 | 256 cores fixas inadequado para fotos naturais | **Documentado como decisão em aberto** — v0.2 usa 256 + parâmetro configurável |
| 6 | Threshold de coalescing sem valor concreto | **Corrigido** — valores iniciais definidos com nota de calibração |

---

## Projeção de tamanho (imagem da montanha, ~36.000 nós)

| Formato | Tamanho estimado |
|---|---|
| SVG texto | ~1.3 MB |
| SVGZ (gzip do SVG) | ~260 KB |
| SVBC v0.1 (sem codebook) | ~300 KB |
| SVBC v0.2 (com codebook + PSQ) | ~90–120 KB |
| SVBC v0.2 + gzip | ~50–70 KB |

---

## O que muda no código

```
src/
├── svbc_format.h      [MODIFY] structs atualizadas + static_asserts
├── svbc_writer.c      [MODIFY] codebook builder + PRS sort + PSQ threshold
├── svbc_reader.c      [MODIFY] lê seção de codebook antes dos nós
├── optimizer.c        [MODIFY] coalesce threshold por layer
├── quadtree.c         [MODIFY] psq_threshold() por layer
└── main.c             [MODIFY] passa SemanticMap para optimizer + novos params
demo.html              [MODIFY] JS lê codebook antes de renderizar
```

---

## Decisões em aberto

1. **256 vs 1024 cores:** testar banding em foto de montanha com 256 cores antes de decidir se v0.3 eleva para `uint16_t token_id`. Se o banding for visível, eleva — o custo é +1 byte/nó (~36 KB em 36.000 nós).

2. **Multiplicadores PSQ e coalescing:** os valores `4.0×`, `2.0×`, `3.0×`, `1.5×` são ponto de partida. Calibrar comparando MAE antes/depois em pelo menos 3 imagens de tipos diferentes (foto, logo, ilustração).

3. **Quantização do codebook:** k-means é mais preciso, median cut é mais rápido. Para MVP, median cut é suficiente.

---

*SVBC v0.2 incorpora o Fluid Codebook do NVDR v0.14 aplicado a imagens estáticas. A struct `SVBC_Node` com `token_id` é diretamente análoga ao Dimensional Token do NVDR — a diferença é que o NVDR resolve o token em VRAM via codebook distribuído, e o SVBC resolve localmente via paleta embutida no arquivo.*
