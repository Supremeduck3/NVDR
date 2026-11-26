# O projeto NVDR, parte por parte

Um guia para entender o que existe aqui, como as peças se encaixam e onde
cada uma está. Para o passo a passo de uso, veja o `GUIA.md`; para as
medições e o porquê de cada decisão, o `NOTEBOOK.md`.

---

## 1. Em uma frase

O NVDR é um **codec**: um programa que transforma fotos e vídeos em
arquivos pequenos e depois os transforma de volta em imagem. Foi escrito
do zero, em C, e tem decodificadores que rodam no navegador e mostram
exatamente os mesmos pixels que o programa em C.

Ele faz três tipos de arquivo:

| Arquivo | O que guarda | Parecido com |
|---|---|---|
| `.nvdr` | uma foto | JPEG, WebP, AVIF |
| `.nvda` | um álbum de fotos | — |
| `.nvdrv` | um vídeo (com som opcional) | MP4 com H.264, HEVC ou AV1 |

E qualquer um deles pode ir **dentro de um PNG comum**, para ser mandado
por onde só passa imagem.

---

## 2. O mapa geral

```
                 ┌──────────────────────────────────────────────┐
  foto / vídeo → │  CODIFICADOR (C)            src/             │ → arquivo .nvdr / .nvda / .nvdrv
                 │  ferramentas de linha de comando   tools/    │
                 └──────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼─────────────────────────────┐
          ▼                           ▼                             ▼
  DECODIFICADOR C            DECODIFICADOR NO NAVEGADOR       ENVIO POR REDE RUIM
  (nvdr_decode etc.)         WebAssembly (o mesmo C)          pacotes + paridade
                             e JavaScript (cópia à mão)       (nvdr-fec.js, UDP)
                                      │
                                      ▼
                             PÁGINA E SERVIDOR
                             public/index.html, server.js

  Em volta de tudo: TESTES (scripts/verify.py, crosschecks, fuzzer, CI)
                    e MEDIÇÕES (scripts/analysis/, NOTEBOOK.md)
```

---

## 3. Conceitos que aparecem em todo lugar

Não precisa decorar; volte aqui quando um termo aparecer.

- **Pixel e canais.** Cada ponto da imagem tem três números. O codec não
  trabalha em RGB, e sim em **YCbCr**: Y é o brilho (o olho é sensível a
  ele) e Cb/Cr são a cor (o olho é menos sensível, então muitas vezes vão
  em metade da resolução, o "4:2:0").
- **Bloco ou folha.** A imagem é cortada em quadrados de 32×32 pixels
  (os *tiles*), e cada um é subdividido numa árvore (*quadtree*) até
  pedaços de 4×4. Áreas lisas ficam em pedaços grandes; detalhes, em
  pedaços pequenos. Cada pedaço final é uma **folha**.
- **Predição.** Em vez de guardar a folha, o codec *adivinha* como ela é a
  partir do que já foi decodificado (os vizinhos, ou o quadro anterior do
  vídeo) e guarda só **a diferença** entre o palpite e a verdade.
- **Transformada.** A diferença é convertida em "frequências" (DCT, a
  mesma ideia do JPEG). Na maioria das imagens quase toda a energia fica
  nas frequências baixas, e o resto vira zero.
- **Quantização (`q`).** Os números da transformada são arredondados num
  passo `q`. É aqui que a qualidade se perde e o tamanho cai. `q` maior,
  arquivo menor e mais borrado.
- **Codificação de entropia.** O último passo: os números viram bits com
  um **codificador aritmético** que aprende, enquanto codifica, quais
  valores são mais prováveis e gasta menos bits com eles. Esse
  aprendizado são os **modelos** (ou contextos).
- **PSNR.** Uma medida de qualidade em decibéis: quanto mais alto, mais
  parecido com o original. 30 dB é aceitável, 40 dB é quase
  indistinguível.
- **BD-rate.** A forma justa de comparar dois codecs: quanto um precisa
  de bytes a mais (ou a menos) para a mesma qualidade, na média de várias
  qualidades. "−10%" quer dizer 10% menor.

---

## 4. As peças, uma por uma

### 4.1 A foto (`.nvdr`)

**Onde:** `src/nvdr.c`, que é dividido em partes (`src/nvdr_*.inc`), e
`src/nvdr.h`, que descreve o formato.

**Como funciona, em ordem:**

1. **Carregar a imagem** (`nvdr_image.inc`): lê JPEG, PNG etc., e gira a
   foto conforme o EXIF (foto de celular em pé).
2. **Converter para YCbCr**, e decidir se a cor vai em metade da resolução.
3. **Montar a árvore** de cada tile (`nvdr_encode.inc`): o codificador
   experimenta dividir ou não cada bloco e fica com o que custar menos em
   *erro + bits*.
4. **Prever cada folha** (`nvdr_planes.inc`), de um de dois jeitos:
   - **Progressivo:** só pela cor média dos vizinhos. O arquivo vem em
     três camadas: (0) cores, (1) detalhe grosso, (2) detalhe fino.
     Qualquer pedaço do começo do arquivo já é uma imagem inteira, que vai
     ganhando nitidez conforme o resto chega.
   - **Direcional:** cada folha é prevista dos pixels vizinhos ao longo de
     uma de 43 direções (como no HEVC e no AV1). Arquivo bem menor, mas só
     aparece inteiro.
5. **Transformar e quantizar** o que a predição errou (`nvdr_transform.inc`),
   escolhendo o arredondamento de cada coeficiente pelo custo (RDOQ).
6. **Codificar tudo** com o codificador aritmético (`src/entropy.c`) e os
   modelos (`nvdr_models.inc`).
7. **Pós-processamento guardado em poucos bytes:** filtro de bordas entre
   blocos (*deblocking*), filtro de restauração (`src/restore.c`) e, se
   pedido, a "receita" do grão do sensor (`src/grain.c`), que é retirado
   antes e recriado na hora de mostrar.

**Recursos acrescentados por último:**

| Recurso | Para que serve | Onde |
|---|---|---|
| Fatias (`--slices`) | codificar uma faixa por núcleo do processador, e isolar estragos de rede | `nvdr_encode.inc`, `nvdr_header.inc` |
| Modelos iniciais (versão 18) | o codificador começa já "sabendo" como fotos costumam ser | `src/priors.h`, `nvdr_models.inc` |
| Transparência e perfil de cor | guardados depois das camadas, sem quebrar decodificadores antigos | `src/extras.c` |
| Decodificar com buracos | mostrar o que chegou quando faltam pedaços do arquivo | `nvdr_decode.inc` |
| Ocultação | faixa perdida vira um degradê em vez de cinza | `nvdr_planes.inc` |

**O arquivo por dentro:** cabeçalho de 32 bytes → parâmetros do grão e do
filtro → tabela das fatias → camada 0 → camada 1 → camada 2 → extras
(transparência, perfil de cor).

### 4.2 O álbum (`.nvda`)

**Onde:** `src/nvda.c`, `src/nvda.h`, ferramenta `tools/nvdr_album.c`.

Várias fotos num arquivo só. Fotos parecidas (uma rajada) são previstas
da foto anterior, como quadros de vídeo, e os modelos aprendidos numa foto
continuam na seguinte. Tem um índice, então dá para abrir uma foto do meio
sem decodificar as outras.

### 4.3 O vídeo (`.nvdrv`)

**Onde:** `src/nvdrv.c` (o maior arquivo do projeto), `src/nvdrv.h`,
`src/mc.c`; ferramentas `tools/nvdrv_encode.c` e `tools/nvdrv_decode.c`.

Cada quadro do vídeo é, no fundo, uma foto NVDR. O que muda é a predição:

- **Quadro intra:** uma foto comum, sem depender de nada. Aparece a cada
  48 quadros (o *grupo de quadros*, ou GOP) e é onde dá para começar a
  assistir ou cortar.
- **Quadro P:** previsto do quadro anterior (ou dos dois anteriores, o
  `PRED2`), movendo blocos conforme um **campo de movimento**: um vetor
  por bloco dizendo "este pedaço veio de lá".
- **Quadro B:** previsto do quadro de antes *e* do de depois, que por
  isso é codificado antes. É o tipo mais barato.

**O que vai em cada quadro:** um cabeçalho curto (3 a 30 bytes), o campo
de movimento (~40% dos quadros previstos) e a imagem corrigida (~60%).

**Ferramentas em volta do vídeo:**

| Recurso | O que faz |
|---|---|
| Filtro temporal | tira ruído das âncoras antes de prever delas |
| Olhar adiante | decide a qualidade de cada quadro pelo quanto ele será reaproveitado |
| Vários núcleos e computadores | cada grupo de quadros é codificado à parte (`cluster.js`, `worker.js`) |
| Taxa alvo (`--kbps`) | acha a qualidade que dá a taxa pedida |
| Som (`--audio`) | a trilha vai inteira depois do último quadro |

### 4.4 O PNG disfarçado

**Onde:** `src/pngwrap.c`, `tools/nvdr_png.c`, `public/nvdr-png.js`.

O arquivo NVDR vai dentro de um PNG comum, num pedaço que os
visualizadores ignoram, e o PNG mostra uma prévia. Qualquer programa
abre como imagem; o visualizador NVDR acha o arquivo dentro e mostra na
qualidade cheia. Serviços que recomprimem imagens (redes sociais) jogam
o pedaço fora e só fica a prévia.

### 4.5 Envio por rede ruim

**Onde:** `public/nvdr-fec.js`, `tools/nvdr_send.mjs`, `tools/nvdr_recv.mjs`,
simulador `scripts/analysis/badnet.mjs`.

Para rádio, satélite ou qualquer canal que perde pacotes sem reenviar:

1. **Fatias:** um pacote perdido estraga só uma faixa da imagem.
2. **Paridade (FEC):** pacotes extras de Reed-Solomon permitem
   reconstruir os perdidos; a camada de cor, que é pequena e essencial,
   recebe mais proteção.
3. **Buracos:** o que não deu para recuperar é decodificado em volta.
4. **Ocultação:** faixa sem cor entre duas que chegaram vira um degradê.

### 4.6 Os decodificadores no navegador

**Onde:** `public/`.

| Arquivo | Papel |
|---|---|
| `nvdr.wasm` + `nvdr-wasm.js` | o decodificador C compilado para o navegador (rápido) |
| `nvdr.js`, `nvdrv.js`, `nvda.js` | o decodificador reescrito à mão em JavaScript (reserva e conferência) |
| `nvdr-priors.js` | a mesma tabela de modelos iniciais do C |
| `nvdr-tasks.js`, `nvdr-worker.js`, `nvdr-decoder.js` | decodificam fora da página (*worker*), para ela não travar |
| `nvdr-img.js` | o elemento `<nvdr-img>`, para usar o formato num site |

O WebAssembly é o mesmo código C, então na prática há **dois**
decodificadores: o C e a cópia em JavaScript. Toda mudança de formato
precisa ser feita nos dois. Unificar, deixando só o WebAssembly, é uma
decisão pendente (seção 7).

### 4.7 A página e o servidor

**Onde:** `public/index.html`, `server.js`.

`node server.js` abre `http://localhost:3000`. A página tem duas abas,
**Imagem** e **Vídeo**: o usuário solta um arquivo, escolhe as opções, o
servidor roda o codificador C e a página mostra o resultado decodificado
no navegador, com as camadas, um controle para cortar o arquivo e simular
download interrompido, e o player de vídeo. Também codifica uma pasta
inteira. Por segurança, só atende a própria máquina; `--lan` abre para a
rede.

### 4.8 As ferramentas de linha de comando

**Onde:** `tools/`. Cada uma mostra suas opções quando roda sem argumentos.

| Ferramenta | Faz |
|---|---|
| `nvdr_encode` / `nvdr_decode` | foto |
| `nvdr_batch` | todas as fotos de uma pasta, uma por núcleo |
| `nvdr_album` | álbum |
| `nvdr_png` | põe e tira o NVDR de dentro de um PNG |
| `nvdrv_encode` / `nvdrv_decode` | vídeo |
| `nvdr_send.mjs` / `nvdr_recv.mjs` | envio e recepção por UDP |

Também há a biblioteca `libnvdr.a` (`make install`), para outro programa
usar o codec diretamente.

### 4.9 Os testes (a garantia)

**Onde:** `scripts/`, `.github/workflows/ci.yml`.

| Teste | Garante |
|---|---|
| `scripts/verify.py` (`make check`) | o codificador é determinístico, a qualidade e o tamanho não pioraram, arquivos cortados ainda decodificam |
| `scripts/crosscheck*.mjs` | C, JavaScript e WebAssembly dão os mesmos pixels, inclusive com cortes, danos e buracos |
| `scripts/fuzz.c` (`make fuzz`) | arquivos corrompidos de propósito não quebram o decodificador |
| CI no GitHub | roda os três acima a cada push |

### 4.10 As medições

**Onde:** `scripts/analysis/` e `NOTEBOOK.md`.

Cada decisão do codec foi medida antes de entrar, e o caderno registra
também o que **não** deu certo. Os scripts principais:

| Script | Mede |
|---|---|
| `stillbd.mjs` | BD-rate de fotos entre duas versões |
| `seqbd.mjs` | o mesmo para vídeo |
| `mkclips.mjs` | recria os clipes de teste a partir de uma foto |
| `bench_codecs.mjs`, `bench_video_offline.mjs` | comparação com JPEG, WebP, AVIF, HEVC, AV1 |
| `badnet.mjs` | simulação de rede ruim |
| `priors.c` | treina os modelos iniciais |

---

## 5. Um exemplo do começo ao fim

Uma foto de 768×512 pixels, modo direcional:

1. `nvdr_encode foto.jpg foto.nvdr --directional --slices auto`
2. A foto é lida, girada se o EXIF pedir, convertida para YCbCr.
3. É cortada em 4 fatias, uma por núcleo.
4. Em cada fatia, cada tile de 32×32 vira uma árvore de folhas; cada
   folha é prevista dos vizinhos, a diferença é transformada, arredondada
   e codificada. Cerca de 3 segundos ao todo.
5. Sai um arquivo de ~41 KB (o JPEG da mesma qualidade teria ~98 KB).
6. No navegador, o WebAssembly decodifica em ~40 ms, refazendo as mesmas
   previsões na mesma ordem e somando as diferenças guardadas.

---

## 6. Onde estamos

BD-rate em PSNR, negativo = NVDR menor:

| Comparação | Resultado |
|---|---|
| Foto contra JPEG | −37% |
| Foto contra WebP | −20% |
| Foto direcional contra AV1 | +6,5% (medição antiga) |
| Vídeo contra H.264 (x264) | −40% a −43% |
| Vídeo contra HEVC (x265) | −13% a −17% |
| Vídeo contra AV1 (libaom) | +54% a +56% |

O ponto mais fraco é o **vídeo contra o AV1**. Já foi medido e descartado:
codificar os vetores de outro jeito, blocos maiores e filtros de
interpolação por bloco. Ainda não medido: movimento deformado, máscaras
compostas, mais referências e a distribuição de qualidade entre quadros.

---

## 7. Decisões em aberto

1. **Unificar os decodificadores do navegador.** Deixar só o
   WebAssembly e apagar a cópia em JavaScript (~2 000 linhas). Cada
   mudança de formato passaria a ser feita uma vez só. Perde-se a
   conferência independente C contra JS.
2. **Ganhos medidos e guardados para a próxima mudança de formato**
   (melhor aplicar depois da unificação):
   - modelos iniciais para vídeo: ~1%;
   - modelos iniciais para o campo de movimento: ~0,3%;
   - peso maior para os modelos iniciais das fotos: ~0,2%.
3. **Base de testes maior** (fotos e vídeos reais) e **licença**:
   adiados.

---

## 8. Como acompanhar o trabalho

- **O que mudou:** `git log --oneline`. Cada commit explica o que fez e,
  quando é medição, o resultado.
- **Por que mudou:** `NOTEBOOK.md`, em ordem cronológica; o título de cada
  seção diz o tema.
- **Se ainda funciona:** a aba Actions do GitHub, ou `make check`.
- **Para ver com os próprios olhos:** `node server.js` e
  `http://localhost:3000`.
