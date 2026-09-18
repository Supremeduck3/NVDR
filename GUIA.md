# Guia de teste

## Por que existem dois pipelines

O repositório carrega **dois codecs independentes**, e é isso que confunde
na hora de testar. Eles não compartilham código, nem formato, nem Makefile:

| | pipeline antigo | pipeline novo |
|---|---|---|
| código | `src/` | `nvdr/` |
| formato | `.svbc` | `.nvdr` |
| binário | `image_to_svg` | `nvdr/nvdr_encode`, `nvdr/nvdr_decode` |
| build | `make` na raiz | `make` dentro de `nvdr/` |
| página | `localhost:3000/` | `localhost:3000/nvdr.html` |

O pipeline novo é o que implementa a Progressive Residual Stack do spec.
O antigo continua no repositório porque ainda funciona e serve de
comparação — nada depende dele.

---

## Build

Os dois, uma vez cada:

```bash
make               # image_to_svg   (pipeline antigo)
cd nvdr && make    # nvdr_encode, nvdr_decode
cd ..
```

Se algum falhar, é dependência faltando: ambos precisam de `gcc` e `zlib`
(`libz-dev` no Debian/Ubuntu). O antigo também usa OpenMP.

---

## O jeito mais rápido de ver funcionando

```bash
node server.js
```

Abre `http://localhost:3000/nvdr.html`, arrasta uma imagem e pronto:

- as três camadas lado a lado, com bytes acumulados e número de retângulos
- um slider que corta o arquivo para simular download interrompido
- o relatório do encoder embaixo

É o caminho mais curto para *olhar* o resultado. O CLI abaixo é para
*medir*.

---

## CLI

### Codificar

```bash
./nvdr/nvdr_encode samples/montanha_pessoas.jpg /tmp/m.nvdr
```

```
samples/montanha_pessoas.jpg  768x512
  level      rects        raw     stored  cumulative     PSNR
  ANCHOR       5941       4011       3123        3195    20.19 dB
  R1          33592     106128      28919       32114    25.11 dB
  R2          52216     163951      53925       86039    26.26 dB
```

Como ler cada coluna:

- **rects** — quantos retângulos esse nível desenha. Cresce a cada nível
  porque o refinamento subdivide.
- **raw** — o tamanho do stream antes da codificação de entropia.
- **stored** — o tamanho real no arquivo. É também o que trafegaria na
  rede: o container não é comprimido de novo por cima.
- **cumulative** — soma de `stored` até aqui, mais o header. É quantos
  bytes um decoder precisa ter recebido para exibir esse nível.
- **PSNR** — qualidade contra a imagem original, medida **decodificando o
  arquivo de volta**, não a partir do estado interno do encoder.

Compare `cumulative` do último nível com o tamanho da imagem de origem —
é a taxa de compressão real.

### Decodificar

```bash
./nvdr/nvdr_decode /tmp/m.nvdr /tmp/saida.png
```

Termine o nome em `.png` e sai PNG; qualquer outra extensão sai PPM.

Para ver um nível específico:

```bash
./nvdr/nvdr_decode /tmp/m.nvdr /tmp/anchor.png --level 0
./nvdr/nvdr_decode /tmp/m.nvdr /tmp/meio.png   --level 1
```

Para medir a qualidade junto:

```bash
./nvdr/nvdr_decode /tmp/m.nvdr /tmp/saida.png --compare samples/montanha_pessoas.jpg
```

### Testar a garantia de truncagem

Esta é a propriedade central do formato: cortar o arquivo em qualquer
ponto depois do anchor ainda produz imagem.

```bash
head -c 4000 /tmp/m.nvdr > /tmp/cortado.nvdr
./nvdr/nvdr_decode /tmp/cortado.nvdr /tmp/cortado.png
```

```
/tmp/cortado.nvdr  768x512  5941 rects  rendered at ANCHOR  (file carries only ANCHOR)
```

Varrendo vários cortes de uma vez:

```bash
SZ=$(stat -c%s /tmp/m.nvdr)
for pct in 100 60 40 39 3; do
  head -c $((SZ*pct/100)) /tmp/m.nvdr > /tmp/t.nvdr
  printf "%3d%% -> " $pct
  ./nvdr/nvdr_decode /tmp/t.nvdr /tmp/t.png --compare samples/montanha_pessoas.jpg
done
```

O único corte que não produz imagem é abaixo do stream do anchor — nesse
caso o decoder diz isso e sai com erro, que é o contrato, não uma falha.

---

## Parâmetros que valem mexer

```bash
./nvdr/nvdr_encode entrada.jpg saida.nvdr \
    --tolerance 0.090,0.040,0.018 \
    --step 16,4 \
    --anchor-bits 4 \
    --codec arith
```

- **`--tolerance A,B,C`** — quão uniforme uma região precisa ser para
  parar de subdividir, por nível, do grosso para o fino. Valores maiores
  no primeiro número deixam o anchor menor e mais grosseiro. Têm que ser
  decrescentes.
- **`--step B,C`** — o passo de quantização do resíduo nos níveis 1 e 2.
  Menor = mais fiel e mais pesado.
- **`--anchor-bits N`** — a paleta do anchor tem `2^N` cores. O default 4
  é o "int4" do spec. `6` costuma dar um anchor melhor a custo total
  quase neutro e vale testar; `8` custa mais bytes e quase não melhora em
  cima de 6.
- **`--weber F`** — quanto mais a tolerância aperta nas sombras. O default
  64 equaliza a alocação de detalhe entre regiões escuras e claras; um
  valor enorme (`1e9`) volta à métrica de diferença absoluta antiga.
- **`--codec arith|deflate`** — `arith` é o default. `deflate` existe só
  para comparação; os dois decodificam para imagens idênticas.

Comparando os dois codecs na mesma imagem:

```bash
./nvdr/nvdr_encode entrada.jpg /tmp/a.nvdr --codec arith   | tail -1
./nvdr/nvdr_encode entrada.jpg /tmp/d.nvdr --codec deflate | tail -1
```

---

## Coisas que confundem

**O `.nvdr` às vezes é maior que o JPEG de origem.** É esperado em
imagens pequenas e detalhadas. O formato não ganha de JPEG em
rate-distortion para foto — o que ele entrega é que os primeiros
poucos KB já são uma imagem inteira. Se o objetivo for arquivo menor
com qualidade igual, JPEG ainda vence.

**PSNR baixo (18-26 dB) não é bug.** É o custo de representar a imagem
com retângulos de cor chapada. JPEG a q75 fica em 32-38 dB. O número
está lá justamente para não esconder isso.

**PSNR não enxerga tudo.** Ele mede erro absoluto, então é cego para onde
o erro está. Uma mudança que tira detalhe do céu e dá para as sombras
melhora a imagem e não move o PSNR — foi exatamente o caso do `--weber`.
Quando avaliar uma mudança de alocação, olhe a imagem, não só o número.

**O servidor não testa o codec novo em `/`.** A página raiz é o pipeline
antigo, que devolve `.svbc`. O codec novo está em `/nvdr.html`.

**`scripts/verify.py` não existe nesta branch.** O gate de regressão e
determinismo ficou em `claude/fix-core-pipeline` e nunca foi mergeado —
o PR #6 levou só o primeiro commit daquela branch. Se quiser o gate,
falta mergear `aad53da`.
