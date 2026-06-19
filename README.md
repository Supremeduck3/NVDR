# Image to SVG Converter (iasvg)

Este projeto converte imagens raster (PNG/JPG) em vetores (SVG/SVBC) utilizando uma segmentação otimizada via *quadtree*.

## 🚀 Como Rodar o Servidor Web (Interface Gráfica)

O projeto possui um servidor Node.js que serve uma página web para você testar as conversões enviando arquivos.

1. Instale o [Node.js](https://nodejs.org/) (se não tiver).
2. Abra o terminal na raiz desta pasta (`iasvg`).
3. Inicie o servidor:
   ```bash
   node server.js
   ```
4. Abra o seu navegador e acesse: `http://localhost:3000`

---

## 💻 Como Rodar via Linha de Comando (CLI)

Se você quiser rodar o programa C diretamente no terminal, use o executável gerado:

**Uso Básico:**
```bash
.\image_to_svg.exe <imagem_de_entrada> <caminho_de_saida> [min_tile] [homo_thresh] [--format svg|svbc|all]
```

**Exemplo Prático:**
```bash
.\image_to_svg.exe "samples\montanha_pessoas.jpg" "output\resultado" --format svg
```
*Isso vai ler a imagem e gerar o arquivo `resultado.svg` dentro da pasta `output`.*

---

## 🔨 Como Compilar o Código C (Apenas se alterar o código)

Se você editar os arquivos `.c` na pasta `src/`, precisará compilar novamente. Basta usar o comando `make` (ou `mingw32-make` no Windows) na raiz do projeto:

```bash
make
```
