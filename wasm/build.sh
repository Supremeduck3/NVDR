#!/bin/sh
# Builds public/nvdr.wasm, the C decoder for the browser, with clang and
# wasm-ld alone (no Emscripten, no wasi-libc): see wasm/libc.c.
set -e
cd "$(dirname "$0")/.."
OUT=${OUT:-public/nvdr.wasm}
TMP=$(mktemp -d)
FLAGS="--target=wasm32 -O3 -msimd128 -mbulk-memory -ffreestanding -nostdlib -DNVDR_WASM
       -Iwasm/include -Isrc -Wall -Wno-unused-parameter -Wno-unknown-pragmas
       -ffunction-sections -fdata-sections"
for f in src/nvdr.c src/entropy.c src/grain.c wasm/libc.c wasm/api.c; do
    ${CLANG:-clang} $FLAGS -c "$f" -o "$TMP/$(basename "$f" .c).o"
done
# Only the API is exported; the linker drops the encoder and everything
# only it calls. No imports may remain.
${WASM_LD:-wasm-ld} --no-entry --gc-sections --export-dynamic -o "$OUT" "$TMP"/*.o
rm -rf "$TMP"
echo "$OUT: $(wc -c < "$OUT") bytes"
