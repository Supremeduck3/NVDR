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
4. **Truncagem** — corta em 9 pontos; todo corte que já tem o começo da
   camada de cor tem que decodificar, e a qualidade não pode cair conforme
   os bytes aumentam.
5. **C contra JS** — os dois decoders têm que dar os mesmos pixels, nas
   duas camadas e em arquivo cortado.

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

- as duas camadas lado a lado (só cor, e cor + textura), com bytes acumulados
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
samples/montanha_pessoas.jpg  768x512  q 24/24  blocks 4..32
  layer       stored  cumulative     PSNR
  COR            4726        4758    20.84 dB
  TEX-BAIXA     13318       18076    25.57 dB
  TEX-ALTA      33934       52012    33.41 dB
```

Desde o formato v10, a imagem é uma **quadtree de blocos quadrados** de 4 a
32 px, e cada bloco carrega duas coisas: uma **cor** (prevista pelos
vizinhos, mais uma pequena correção) e, se compensar, a **textura** do
bloco, como coeficientes de DCT do tamanho do bloco. Um bloco só com cor é
exatamente um retângulo chapado do formato antigo.

Como ler:

- **COR** — a primeira camada: a árvore e a cor de cada bloco. Sozinha já
  é uma imagem inteira de blocos chapados.
- **TEX-BAIXA** — a segunda camada: as frequências baixas da textura de
  **todos** os blocos. Com ela a imagem inteira já tem a textura grossa.
- **TEX-ALTA** — a terceira: o detalhe fino.
- **stored** — o tamanho real da camada no arquivo, que é também o que
  trafegaria na rede.
- **cumulative** — soma até aqui, mais o header: quantos bytes um decoder
  precisa ter recebido para exibir essa camada inteira.
- **PSNR** — medido **decodificando o arquivo de volta**, não a partir do
  estado interno do encoder.
- **leaves** — quantos blocos de cada tamanho a árvore escolheu, e quantos
  levam textura. Blocos grandes onde a imagem é lisa, pequenos onde tem
  detalhe.

### Decodificar

```bash
./nvdr_decode /tmp/m.nvdr /tmp/saida.png
```

Termine o nome em `.png` e sai PNG; qualquer outra extensão sai PPM.

Só a camada de cor, ou cor + textura baixa:

```bash
./nvdr_decode /tmp/m.nvdr /tmp/cor.png   --layer 0
./nvdr_decode /tmp/m.nvdr /tmp/baixa.png --layer 1
```

Para medir a qualidade junto:

```bash
./nvdr_decode /tmp/m.nvdr /tmp/saida.png --compare samples/montanha_pessoas.jpg
```

### Testar a garantia de truncagem

Cortar o arquivo em qualquer ponto depois do começo da camada de cor ainda
produz imagem. Cada camada é enviada em blocos de 32×32, de cima para
baixo, e a textura vem em duas: primeiro a **grossa da imagem inteira**,
depois o detalhe. Um bloco de cor que não chegou fica **cinza**, e um
bloco de textura que não chegou mostra o que a camada anterior deu a ele.

```bash
SZ=$(stat -c%s /tmp/m.nvdr)
for pct in 3 10 30 50 75 100; do
  head -c $((SZ*pct/100)) /tmp/m.nvdr > /tmp/t.nvdr
  printf "%3d%% -> " $pct
  ./nvdr_decode /tmp/t.nvdr /tmp/t.png --compare samples/montanha_pessoas.jpg
done
```

No montanha_pessoas: 30% do arquivo dá 24,6 dB, 50% dá 26,5 dB, 75% dá
28,8 dB e 100% dá 33,4 dB. A qualidade nunca cai com mais bytes.

O único corte que não produz imagem é antes dos primeiros bytes da camada
de cor. Nesse caso o decoder diz isso e sai com erro, que é o contrato,
não uma falha.

---

## Parâmetros que valem mexer

```bash
./nvdr_encode entrada.jpg saida.nvdr --q 24
```

- **`--q N`** — o passo de quantização, **o** botão de qualidade. Menor é
  melhor e maior. O default 24 foi escolhido para as amostras saírem do
  tamanho que o formato antigo dava, ou menores, com cerca de +9 dB. No
  montanha_pessoas: `--q 12` dá 88 KB a 39,1 dB, `--q 24` dá 53 KB a
  33,3 dB, `--q 48` dá 23 KB a 28,6 dB.
- **`--chroma-q F`** — o passo do croma relativo ao da luma (default 1,0,
  que mediu melhor que 1,5 e 2,5).
- **`--deadzone F`** — de 0 a 0,5: quanto um coeficiente pequeno tende a
  virar zero. Default 0,1, o melhor medido.
- **`--lambda F`** — o peso dos bits contra o erro na decisão de dividir um
  bloco. Quase não muda nada entre 0,06 e 0,3; default 0,12.
- **`--max-block N` / `--min-block N`** — o maior e o menor bloco (4 a 32).
  32 ganhou de 16 por até 0,5 dB.
- **`--band N`** — onde a textura se divide entre as camadas baixa e alta
  (default 8; `0` deixa tudo numa camada só). Com 8, metade do arquivo dá
  em média +2,4 dB contra `0`, e o arquivo sai ~1% menor.
- **`--no-deblock`** — desliga o filtro que suaviza as emendas entre blocos.
- **`--quiet`** — só grava o arquivo, sem a tabela por camada (que decodifica
  o arquivo três vezes para medir o PSNR).

---

## Álbum: várias imagens num arquivo só

```bash
./nvdr_album pack /tmp/album.nvda samples/*.jpg
./nvdr_album unpack /tmp/album.nvda /tmp/saida --compare samples
```

As imagens vão em ordem num `.nvda`. Cada uma é codificada de um de dois
jeitos, e o encoder escolhe:

- **prevista de uma foto anterior**: de **qualquer uma** das últimas 8
  (`--window N`, até 32), desde que do mesmo tamanho. O encoder procura as
  mais parecidas por miniatura e tenta as duas melhores. Usa o mesmo
  mecanismo dos quadros de vídeo: movimento em quarto de pixel, skip e só
  o que mudou. Serve para rajada, mesmo cenário, capturas de tela, e
  também para um ensaio que vai e volta entre dois assuntos.
- **sozinha**, quando nenhuma anterior ajuda.

A comparação é feita **com a mesma qualidade**: a versão prevista é
refinada até ficar tão boa quanto a sozinha, e só ganha se continuar
menor. O `pack` mostra, por imagem, o tipo escolhido (`prevista #k` diz de
qual foto), quanto ela custou e quanto custaria sozinha.

Para tirar **uma foto só** do álbum (decodifica só ela e as de que depende):

```bash
./nvdr_album unpack /tmp/album.nvda /tmp/saida --only 3
```

Na página, solte **várias imagens de uma vez** na aba Imagem: elas viram
um álbum, mostrado em grade com o tamanho de cada uma. Um `.nvda` pronto
abre direto.

**Quanto economiza:**

| álbum | economia |
|---|---|
| fotos sem relação | 0% (nunca fica maior) |
| fotos do mesmo cenário, 0,4 s entre elas | ~39% |
| rajada, 0,12 s entre elas | ~58% |
| dois cenários intercalados | ~66% (com `--window 1` seria 0%) |

Álbuns de fotos grandes demoram: cada foto é codificada sozinha e,
quando parece com uma anterior, também prevista dela, para comparar. Fotos
cujas miniaturas diferem mais que `--distance` (10 por padrão) nem são
tentadas. Numa máquina de 4 núcleos, três fotos de 13,5 Mpx levam ~20 s
sem relação entre si e ~47 s quando uma repete a outra.

`--fluid` liga o **codebook fluido** (as probabilidades do codificador
aritmético passam de uma foto para a próxima): mais ~0,7%, mas aí cada foto
depende de todas as anteriores e o álbum só pode ser lido em ordem. Por
isso vem desligado.

### Usando num site

O arquivo continua com as propriedades quando vai para um site. Coloque
o `nvdr-img.js` (e `nvdr.js`, `nvda.js`, `nvdrv.js`) junto do site e use o
elemento no lugar do `<img>`:

```html
<script type="module" src="nvdr-img.js"></script>
<nvdr-img src="foto.nvdr" alt="Montanha"></nvdr-img>
<nvdr-img src="galeria.nvda#3" alt="Terceira foto do álbum"></nvdr-img>
<nvdr-img src="galeria.nvda#praia.jpg"></nvdr-img>
```

- **Uma imagem `.nvdr`** aparece enquanto baixa e melhora a cada byte.
- **Uma foto de álbum** baixa só o índice, ela e as fotos de que ela
  depende (requisições HTTP com `Range`), não o álbum inteiro. Várias fotos
  do mesmo álbum na página compartilham o que já baixaram: uma grade com o
  álbum todo baixa o tamanho do álbum, uma vez.
- Qualquer servidor estático serve (nginx, Apache, CDN, S3: todos aceitam
  `Range`). Se o servidor não aceitar, o álbum vem inteiro e funciona igual.

Para ver funcionando:

```bash
make demo          # gera public/demo/montanha.nvdr e public/demo/galeria.nvda
node server.js     # abra http://localhost:3000/galeria.html
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
- **`--q N`** — o passo de quantização dos quadros intra (default 24), e
  **`--pred-q N`** o dos quadros preditos (default 1,2× o `--q`).
- **`--block N`** — movimento por bloco de NxN, com vetores em **quarto de
  pixel** (default: 16 a partir de 0,2 Mpx, 8 abaixo; `0` usa um vetor só
  para o quadro todo).
- **`--mv-lambda N`** — quanto um bit de vetor pesa contra o erro do bloco
  (default 16).
- **`--skip-k F`** — quão facilmente um quadro predito deixa um bloco
  **exatamente como estava** no quadro anterior, em vez de recodificar a
  pequena diferença (default 0,25; `0` corrige todos). É o que tira a
  cintilação do fundo parado: sem isso, 5,5% dos pixels de um fundo
  parado mudavam a cada quadro, e com o padrão são 1,3%, com 19% menos
  bytes. Valores maiores economizam mais, mas deixam o erro do quadro
  anterior se arrastar; acima de ~0,5 perde para simplesmente aumentar o
  `--q`.

No clipe de referência de 960×540, `--q 20` dá **622 kbit/s a 36,8 dB**.
O VP8 faz 605 kbit/s a 37,2 dB. A interpolação de quarto de pixel usa o
filtro de 6 taps do H.264, que no laço fechado vale ~1 dB sobre a
bilinear.

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
divergirem, a página mostra algo que o formato não gravou. Por isso tudo
que o decoder calcula é inteiro: a DCT inversa é a aproximação inteira do
HEVC, a conversão de cor é em ponto fixo. O script compara os dois nas duas
camadas e em vários cortes:

```bash
node scripts/crosscheck.mjs /tmp/m.nvdr
```

```
whole layer=0: identical (768x512, tiles 384/0 of 384)
whole layer=1: identical (768x512, tiles 384/384 of 384)
cut 3%: identical (768x512, tiles 182/0 of 384)
...
```

Sai com código 1 e aponta o primeiro pixel divergente se algo quebrar.
Vale rodar depois de mexer em `src/entropy.c`, `src/nvdr.c` ou
`public/nvdr.js` — as três coisas que precisam concordar.

---

## Coisas que confundem

**Arquivos antigos (`.nvdr` v9 e v10, `.nvdrv` antes da versão 6) não abrem mais.** O formato mudou de
retângulos chapados para quadtree + DCT; recodifique a imagem. O codec
antigo está em `reference/nvdr_v9`, só para reproduzir medições antigas.

**Com qualidade baixa aparecem blocos.** Com `--q` alto, os blocos grandes
de 32×32 ficam chapados e as emendas aparecem. É o artefato típico de DCT.
O filtro de deblocking suaviza as emendas que têm textura de um dos lados;
entre dois blocos chapados a emenda é a própria imagem e fica.

**PSNR não enxerga tudo.** Ele mede erro absoluto, então é cego para onde
o erro está. Quando avaliar uma mudança, olhe a imagem, não só o número.

**Uma foto granulada (céu noturno, parede texturizada) aparecia cheia de
pontos brancos na terceira camada ou na grade do álbum.** Os pixels
estavam certos; o navegador é que, ao reduzir o canvas, pegava um pixel a
cada 10×10 e jogava o resto fora, e sobravam grãos soltos. Agora a página
reduz a imagem pela média de todos os pixels, como o olho faria. Se ainda
aparecer, compare com `./nvdr_decode arquivo.nvdr saida.png` aberto num
visualizador de imagens: se lá estiver limpo, o problema é de exibição.
