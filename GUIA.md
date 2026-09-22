# Guia de teste

## Onde está cada coisa

```
src/        o codec: nvdr.c e entropy.c, mais nvdrv.c para sequências
tools/      quatro CLIs: nvdr_encode/decode (imagem) e nvdrv_encode/decode
            (sequência)
public/     o decoder do navegador (nvdr.js) e a página que o usa
scripts/    o gate de regressão, o crosscheck C×JS, e análises
samples/    as imagens em que todo número do README foi medido
vendor/     stb_image.h, o único código de terceiros
reference/  dois módulos guardados de um pipeline removido; não compilam
            no build
```

Havia um segundo codec no repositório (`src/` antigo, formato `.svbc`,
binário `image_to_svg`). Ele foi removido: com a taxa casada o NVDR saía
menor e 2,2 a 6,1 dB melhor nas seis imagens de `samples/`, e ainda tem
truncagem progressiva, que o `.svbc` não tinha. Está tudo no histórico do
git se precisar — veja `reference/README.md`.

---

## Build

```bash
make
```

Precisa de `gcc` e `zlib` (`libz-dev` no Debian/Ubuntu). Saem quatro
binários na raiz: `nvdr_encode`, `nvdr_decode`, `nvdrv_encode` e
`nvdrv_decode`.

OpenMP é usado se estiver disponível (o Makefile detecta sozinho) e
paraleliza o ajuste de rampa. Sem ele compila igual, só mais devagar. A
saída é **idêntica bit a bit** nos dois casos — conferido contra hash
gravado e em 70 execuções.

Para rodar o gate de regressão:

```bash
make check
```

Ele checa cinco coisas em cada imagem de `samples/`:

1. **Determinismo** — codifica duas vezes, os dois arquivos têm que sair
   byte a byte iguais. Sem isso, qualquer medição A/B está medindo ruído.
2. **Qualidade** — decodifica de volta e compara contra
   `scripts/baseline.json`, **por imagem**. Piso global não serve: as
   amostras vão de 21 a 30 dB, então uma queda que importa em uma some no
   meio das outras.
3. **Tamanho** — bytes contra a mesma baseline, para uma mudança que
   compra qualidade com bytes aparecer como o que é.
4. **Truncagem** — corta em 9 pontos; todo corte acima do anchor tem que
   decodificar, e a qualidade não pode cair conforme os bytes aumentam.
5. **C contra JS** — os dois decoders têm que dar os mesmos pixels, em
   todos os níveis e em arquivo cortado.

E mais uma, sobre uma sequência de 12 quadros que ele também gera na hora:
**deriva**. O encoder tem que predizer da própria reconstrução, nunca do
quadro fonte, porque é só isso que o decoder tem. Errar isso é a forma
clássica de um codec derivar. Conferi que o gate pega: trocando o laço de
reconstrução pelo quadro fonte, o arquivo fica **78,7% menor** — um teste
que só olhasse tamanho chamaria de melhoria — e custa **8,90 dB** até o
quadro 12.

Ele também **gera três imagens sintéticas** na hora (`blocos`, `circulos`,
`degrade`) e passa elas pelos mesmos testes. Elas existem porque o
`samples/` é só foto, e foto escondeu uma regressão que custou 5,5 dB numa
imagem de formas chapadas — árvore que satura reage à tolerância no
sentido oposto de árvore que não satura.

Quando uma mudança for melhoria de verdade e não regressão, você regrava a
baseline:

```bash
python3 scripts/verify.py --update
```

Conferi que ele pega: reintroduzindo a tolerância velha, ele reprova 7
amostras e sai com código 1, com `circulos` acusando -4,60 dB.

---

## O jeito mais rápido de ver funcionando

```bash
node server.js
```

Abre `http://localhost:3000`. A página tem duas abas, **Imagem** e **Vídeo**.

### Vídeo

Arrasta um MP4, WebM, MOV, MKV ou AVI. O servidor usa o **ffmpeg da sua
máquina** para extrair os quadros, codifica com `nvdrv_encode`, e o player
toca no navegador. Um `.nvdrv` já codificado também pode ser arrastado —
esse toca direto, sem passar pelo servidor.

- Precisa de `ffmpeg` no PATH. Se estiver em outro lugar:
  `NVDR_FFMPEG=/caminho/ffmpeg node server.js`. Sem ele, a página avisa.
- O fps é lido do próprio arquivo (com `ffprobe` se houver, senão pela
  descrição que o `ffmpeg` dá da entrada), e os quadros são extraídos
  **sem reamostrar** — reamostrar duplica quadros, e quadro duplicado é
  predição perfeita, o que faria o codec parecer melhor do que é.
- **Duração** (3 a 20 s) e **largura máxima** limitam o trabalho: codificar
  é a direção lenta, e os quadros ficam em PNG no disco enquanto isso.
  Com os padrões (5 s, 1280 px), um clipe de 960×540 levou 14 s ida e volta.

O player mostra, por quadro, quanto custou **decodificar** e **exibir**
contra o orçamento do fps do vídeo, e quantas vezes travou. A faixa embaixo
é o arquivo inteiro: cada barra é um quadro, larga pelos bytes que custou,
alta se for intra. Clicar ou arrastar corta o arquivo ali — um prefixo do
arquivo é um prefixo do filme, e o quadro onde o corte cai ainda aparece.

### Imagem

Arrasta uma imagem e pronto:

- as três camadas lado a lado, com bytes acumulados e número de retângulos
- um slider que corta o arquivo para simular download interrompido
- o relatório do encoder embaixo

É o caminho mais curto para *olhar* o resultado. O CLI abaixo é para
*medir*.

---

## CLI

### Codificar

```bash
./nvdr_encode samples/montanha_pessoas.jpg /tmp/m.nvdr
```

```
samples/montanha_pessoas.jpg  768x512
  level      rects        raw     stored  cumulative     PSNR
  ANCHOR       9364       6292       4470        4542    19.55 dB
  R1          37663     269528      12538       17080    24.30 dB
  R2          53707     383331      34685       51765    26.30 dB
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
./nvdr_decode /tmp/m.nvdr /tmp/saida.png
```

Termine o nome em `.png` e sai PNG; qualquer outra extensão sai PPM.

Para ver um nível específico:

```bash
./nvdr_decode /tmp/m.nvdr /tmp/anchor.png --level 0
./nvdr_decode /tmp/m.nvdr /tmp/meio.png   --level 1
```

O decoder suaviza as emendas entre retângulos por padrão. `--smooth 0`
desliga e mostra os retângulos duros — útil para ver o que o formato
realmente gravou:

```bash
./nvdr_decode /tmp/m.nvdr /tmp/duro.png --smooth 0
```

Para medir a qualidade junto:

```bash
./nvdr_decode /tmp/m.nvdr /tmp/saida.png --compare samples/montanha_pessoas.jpg
```

### Testar a garantia de truncagem

Esta é a propriedade central do formato: cortar o arquivo em qualquer
ponto depois do anchor ainda produz imagem.

```bash
head -c 4600 /tmp/m.nvdr > /tmp/cortado.nvdr
./nvdr_decode /tmp/cortado.nvdr /tmp/cortado.png
```

```
/tmp/cortado.nvdr  768x512  9364 rects  rendered at ANCHOR  (file carries only ANCHOR)
```

O anchor inteiro cabe em ~4,5 KB. Com 12000 bytes já entra parte do R1:

```
/tmp/cortado.nvdr  768x512  28705 rects  rendered at ANCHOR+R1  (file carries only ANCHOR+R1)
```

A qualidade sobe continuamente com os bytes, sem degraus — um nível
parcial é aproveitado até onde chegou:

```
 10% -> 19.73 dB      50% -> 25.80 dB
 20% -> 23.30 dB      75% -> 26.42 dB
 30% -> 24.71 dB     100% -> 27.00 dB
```

Varrendo vários cortes de uma vez:

```bash
SZ=$(stat -c%s /tmp/m.nvdr)
for pct in 100 60 40 39 3; do
  head -c $((SZ*pct/100)) /tmp/m.nvdr > /tmp/t.nvdr
  printf "%3d%% -> " $pct
  ./nvdr_decode /tmp/t.nvdr /tmp/t.png --compare samples/montanha_pessoas.jpg
done
```

O único corte que não produz imagem é abaixo do stream do anchor — nesse
caso o decoder diz isso e sai com erro, que é o contrato, não uma falha.

---

## Parâmetros que valem mexer

```bash
./nvdr_encode entrada.jpg saida.nvdr \
    --tolerance 0.090,0.040,0.018 \
    --step 16,4 \
    --anchor-bits 4 \
    --codec arith
```

- **`--tolerance A,B,C`** — quão uniforme uma região precisa ser para
  parar de subdividir, por nível, do grosso para o fino. Valores maiores
  no primeiro número deixam o anchor menor e mais grosseiro. Têm que ser
  decrescentes. Default `0.090,0.040,0.018`.

  **Esse parâmetro não significa a mesma coisa em imagens diferentes**, e
  isso confunde muito na hora de testar. Em foto com grão ou ruído (céu
  estrelado, folhagem) a árvore nunca satura — entre 0,060 e 0,040 um céu
  estrelado vai de 4.507 para 103.891 retângulos e continua crescendo até
  0,010. Ali a tolerância é um dial de taxa suave e qualquer valor dá um
  resultado razoável. Em imagem simples (formas chapadas, desenho, texto) a
  árvore satura: `blocos` dá 16 retângulos em toda a faixa, de 0,200 a
  0,010. Ali não é dial, é penhasco — abaixo da saturação não compra nada,
  e acima destrói as bordas, que é onde a imagem está (1% dos pixels
  carregam metade do erro quadrático). Afrouxar custou 5,5 dB para economizar
  7% dos bytes numa dessas.

  Ou seja: não calibre tolerância só em foto.
- **`--step B,C`** — o passo de quantização do resíduo nos níveis 1 e 2.
  Menor = mais fiel e mais pesado.
- **`--anchor-bits N`** — a paleta do anchor tem `2^N` cores. O default 4
  é o "int4" do spec. `6` costuma dar um anchor melhor a custo total
  quase neutro e vale testar; `8` custa mais bytes e quase não melhora em
  cima de 6.
- **`--weber F`** — quanto mais a tolerância aperta nas sombras. O default
  64 equaliza a alocação de detalhe entre regiões escuras e claras; um
  valor enorme (`1e9`) volta à métrica de diferença absoluta antiga.
- **`--texture F`** — gasta menos resolução em textura fina (grama
  distante, folhagem) e mais em estrutura (rostos, bordas). É **controle
  de taxa**: desligado por padrão, porque também limita a qualidade
  máxima. Abaixo de ~metade da taxa padrão vale +3 a +4 dB contra
  simplesmente afrouxar a tolerância; acima disso é pior. Comece em `1`.
- **`--chroma N`** — quanto mais grosso o croma é quantizado que a luma.
  Default 2. Corta 28-44% dos bytes contra RGB por -0,13 a -0,32 dB, e a
  maior parte disso vem da transformada em si, não da subamostragem.
  Acima de 2 começa a aparecer desvio de cor. `0` volta para RGB.
- **`--order area|dfs`** — `area` (default) manda os retângulos maiores
  primeiro, então um stream cortado cobre a tela inteira grosseiramente em
  vez de um canto em detalhe. Custa 0,18% em bytes e vale até +1,67 dB no
  primeiro décimo. `dfs` mantém a ordem da árvore.
- **`--gradient F`** — quanto um retângulo pode carregar uma **rampa de
  cor** num eixo em vez de ser chapado. É decisão por retângulo: o encoder
  compara custo (um flag, um bit de eixo, três inclinações) contra o erro
  que a rampa tira, e `F` é o preço do bit. Por isso **nunca perde para o
  chapado** — quando não compensa, o retângulo fica chapado. Default 600;
  `0` desliga. No default ganha em todas as imagens medidas: +0,23 a
  +2,43 dB por 2 a 33% mais bytes, inclusive nas sintéticas. Valores
  menores (ex. `280`) gastam mais e rendem mais. A rampa rende em proporção
  à **área** do retângulo, então com tolerância apertada os retângulos são
  pequenos e cada rampa explica menos — foi por isso que o 280 original,
  calibrado junto com tolerância frouxa, não sobreviveu à volta dela.
- **`--gradient-step N`** — o passo de quantização das inclinações. Default
  16, que é grosso de propósito: a inclinação é codificada quase em unário,
  então um passo fino encarece justamente as rampas grandes, que são as que
  valem a pena.
- **`--codec arith|deflate`** — `arith` é o default. `deflate` existe só
  para comparação e **não carrega rampas**, então para comparar os dois
  codecs de verdade use `--gradient 0` nos dois. Com rampas desligadas em
  ambos, `arith` corta 18 a 29% dos bytes.

Comparando os dois codecs na mesma imagem:

```bash
./nvdr_encode entrada.jpg /tmp/a.nvdr --codec arith   | tail -1
./nvdr_encode entrada.jpg /tmp/d.nvdr --codec deflate | tail -1
```

---

## Vídeo

```bash
./nvdrv_encode pasta_de_quadros/ saida.nvdrv
./nvdrv_decode saida.nvdrv --out reproduzido/ --compare pasta_de_quadros/
```

Os quadros são lidos em ordem de nome (`f000.png`, `f001.png`, …), em PNG,
JPG, BMP ou PPM. O encoder imprime por quadro se ele saiu INTRA ou predito,
quanto custou e qual vetor de movimento achou — se um quadro predito fica
do tamanho de um intra, a predição não está funcionando e isso tem que
aparecer, não ser diluído numa taxa média.

Parâmetros:

- **`--gop N`** — força um quadro intra a cada N (default 48, dois segundos
  a 24 fps). Serve para entrar no meio do stream e para um erro não
  contaminar o resto do filme. `0` deixa só o quadro 0 intra: menor e sem
  busca.
- **`--search N`** — raio da busca de movimento global, em pixels (default
  12). `0` fixa a referência no lugar, o que medido **não presta**: com um
  pan de 3 px/quadro a referência co-localizada vale 7%, e com um vetor
  global vale 61%.
- **`--intra-thresh F`** — erro médio absoluto acima do qual o quadro vai
  intra mesmo fora do GOP (default 24). É assim que corte de cena é
  detectado, não declarado.

Contra codificar cada quadro sozinho, nas sequências sintéticas: **−66,6%
a −75,1% com qualidade igual ou melhor**.

A truncagem agora vale nos **dois eixos** — um prefixo do arquivo é um
prefixo do filme, e o quadro em que o corte cai ainda aparece, na qualidade
que os bytes dele pagaram.

---

## Conferindo que C e JS decodificam igual

Para vídeo o equivalente é `scripts/crosscheck_seq.mjs`: compara o
**estado** dos dois decoders depois de cada quadro, não a imagem na tela.
Um quadro predito é um resíduo em cima desse estado, então um pixel de
diferença no quadro 1 vira outra referência no quadro 2 e o erro se
acumula — o player sairia do arquivo com cada quadro parecendo normal.

```bash
node scripts/crosscheck_seq.mjs /tmp/saida.nvdrv
```

O gate (`make check`) roda os dois.

O decoder em C e o do navegador têm que produzir **os mesmos bytes** — se
divergirem, a página mostra algo que o formato não gravou. O script
compara os dois em todos os níveis e em vários cortes, com e sem
suavização:

```bash
node scripts/crosscheck.mjs /tmp/m.nvdr
```

```
whole smooth=0 level=0: identical (768x512, level 0)
...
cut 96% smooth=0.4: identical (768x512, level 2)
```

Sai com código 1 e aponta o primeiro pixel divergente se algo quebrar.
Vale rodar depois de mexer em `src/entropy.c`, `src/nvdr.c` ou
`public/nvdr.js` — as três coisas que precisam concordar.

---

## Coisas que confundem

**O `.nvdr` às vezes é maior que o JPEG de origem.** É esperado em
imagens pequenas e detalhadas. O formato não ganha de JPEG em
rate-distortion para foto — o que ele entrega é que os primeiros
poucos KB já são uma imagem inteira. Se o objetivo for arquivo menor
com qualidade igual, JPEG ainda vence.

**Um retângulo não é necessariamente uma cor chapada.** Desde a v8 ele
pode carregar uma rampa num eixo, e o encoder escolhe por retângulo. Isso
tirou o teto do formato: com tudo chapado o montanha_pessoas empacava em
~26 dB por mais bytes que você jogasse nele.

**Um corte no meio de uma unidade não perde mais o que já chegou.** Desde
a v9 a cor de cada retângulo vai intercalada com a geometria dele, então
uma unidade cortada pela metade entrega tudo que chegou e o resto cai para
a cor do nível anterior. Não muda o tamanho do arquivo em nada. O ganho é
pequeno e concentrado em cortes bem no começo (+0,88 dB no macarrão a 12%),
zero no resto.

**Céu estrelado / imagem com grão parece o melhor caso e é o pior.** Ela
comprime bem e pontua bem porque a árvore nunca satura — dá para pedir
qualquer taxa e ela entrega. Mas o anchor sai com **1 retângulo** e o R1
com 13, então o arquivo não tem estado intermediário nenhum: truncar dá
21,59 dB em 1% do arquivo e 21,75 dB em 25%, parado ao longo de um quarto
dos bytes. Uma foto normal sobe 18,40 → 19,85 → 22,45 → 25,46 dB na mesma
faixa. Bons números de taxa/qualidade ali escondem que a escada
progressiva, que é o ponto do formato, não existe naquela imagem.

**PSNR baixo (18-26 dB) não é bug.** É o custo de representar a imagem
com retângulos de cor chapada. JPEG a q75 fica em 32-38 dB. O número
está lá justamente para não esconder isso.

**PSNR não enxerga tudo.** Ele mede erro absoluto, então é cego para onde
o erro está. Uma mudança que tira detalhe do céu e dá para as sombras
melhora a imagem e não move o PSNR — foi exatamente o caso do `--weber`.
Quando avaliar uma mudança de alocação, olhe a imagem, não só o número.

