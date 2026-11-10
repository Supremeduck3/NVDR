# NVDR — still images (format v10) and sequences.
#
# Deliberately dependency-free: C11, libm, zlib and stb_image. Nothing
# here needs a GPU, and nothing here should start needing one.

CC      = gcc
# OpenMP parallelises the sequence encoder's block motion search, which is
# per-block and shares nothing. Without it the pragmas are ignored and the
# build still works.
OPENMP  = $(shell $(CC) -fopenmp -E - < /dev/null > /dev/null 2>&1 && echo -fopenmp)
CFLAGS  = -std=c11 -O3 -Wall -Wextra -Wno-unused-parameter -Isrc -Ivendor $(OPENMP)
LDLIBS  = -lm -lz

CODEC   = src/nvdr.c src/entropy.c src/grain.c src/restore.c
HEADERS = src/nvdr.h src/entropy.h src/grain.h src/restore.h src/priors.h
SEQ     = $(CODEC) src/nvdrv.c src/mc.c
SEQ_H   = $(HEADERS) src/nvdrv.h src/mc.h
TOOLS   = nvdr_encode nvdr_decode nvdrv_encode nvdrv_decode nvdr_album nvdr_png nvdr_batch

all: $(TOOLS)

nvdr_encode: tools/nvdr_encode.c $(CODEC) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ tools/nvdr_encode.c $(CODEC) $(LDLIBS)

nvdr_decode: tools/nvdr_decode.c $(CODEC) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ tools/nvdr_decode.c $(CODEC) $(LDLIBS)

nvdr_album: tools/nvdr_album.c src/nvda.c src/nvda.h $(SEQ) $(SEQ_H)
	$(CC) $(CFLAGS) -o $@ tools/nvdr_album.c src/nvda.c $(SEQ) $(LDLIBS)

nvdr_batch: tools/nvdr_batch.c src/pngwrap.c src/pngwrap.h $(CODEC) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ tools/nvdr_batch.c src/pngwrap.c $(CODEC) $(LDLIBS)

nvdr_png: tools/nvdr_png.c src/pngwrap.c src/pngwrap.h src/nvda.c src/nvda.h $(SEQ) $(SEQ_H)
	$(CC) $(CFLAGS) -o $@ tools/nvdr_png.c src/pngwrap.c src/nvda.c $(SEQ) $(LDLIBS)

nvdrv_encode: tools/nvdrv_encode.c $(SEQ) $(SEQ_H)
	$(CC) $(CFLAGS) -o $@ tools/nvdrv_encode.c $(SEQ) $(LDLIBS)

nvdrv_decode: tools/nvdrv_decode.c $(SEQ) $(SEQ_H)
	$(CC) $(CFLAGS) -o $@ tools/nvdrv_decode.c $(SEQ) $(LDLIBS)

# The codec as a library, for a program that calls it instead of running
# the tools: libnvdr.a, and with `make install` the headers, the tools and
# a pkg-config file under PREFIX. Link with what `pkg-config --libs
# --static nvdr` says: -lnvdr -lm -lz, and OpenMP when built with it.
# LIB_VERSION is the still container's version and the sequence format's.
LIB_VERSION = 18.21
LIB_SRC = $(SEQ) src/nvda.c src/pngwrap.c
LIB_OBJ = $(LIB_SRC:src/%.c=build/%.o)
LIB_H   = src/nvdr.h src/nvdrv.h src/nvda.h src/pngwrap.h

build/%.o: src/%.c $(SEQ_H) src/nvda.h src/pngwrap.h
	@mkdir -p build
	$(CC) $(CFLAGS) -fPIC -c -o $@ $<

libnvdr.a: $(LIB_OBJ)
	$(AR) rcs $@ $^

PREFIX ?= /usr/local
install: all libnvdr.a
	install -d $(DESTDIR)$(PREFIX)/bin $(DESTDIR)$(PREFIX)/lib/pkgconfig $(DESTDIR)$(PREFIX)/include/nvdr
	install -m 755 $(TOOLS) $(DESTDIR)$(PREFIX)/bin
	install -m 644 libnvdr.a $(DESTDIR)$(PREFIX)/lib
	install -m 644 $(LIB_H) $(DESTDIR)$(PREFIX)/include/nvdr
	printf 'prefix=%s\nlibdir=$${prefix}/lib\nincludedir=$${prefix}/include/nvdr\n\nName: nvdr\nDescription: NVDR still image and sequence codec\nVersion: %s\nLibs: -L$${libdir} -lnvdr\nLibs.private: -lm -lz %s\nCflags: -I$${includedir}\n' \
	    '$(PREFIX)' '$(LIB_VERSION)' '$(OPENMP)' > $(DESTDIR)$(PREFIX)/lib/pkgconfig/nvdr.pc

# The regression gate: encodes every sample, checks quality against a
# floor, checks the encoder is deterministic, and checks the C and JS
# decoders agree byte for byte.
check: $(TOOLS)
	python3 scripts/verify.py

# Mutates real containers and decodes them under AddressSanitizer and
# UndefinedBehaviorSanitizer. Any out-of-bounds access stops it with the
# offending input left in fuzz_last_input.bin.
fuzz_nvdr: scripts/fuzz.c src/nvda.c src/nvda.h $(SEQ) $(SEQ_H)
	$(CC) -std=c11 -g -O1 -fsanitize=address,undefined -fno-sanitize-recover=all \
	    -Isrc -Ivendor $(OPENMP) -o $@ scripts/fuzz.c src/nvda.c $(SEQ) $(LDLIBS)

fuzz: fuzz_nvdr

output/convert: scripts/analysis/convert.c $(CODEC) $(HEADERS)
	mkdir -p output
	$(CC) $(CFLAGS) -o $@ scripts/analysis/convert.c $(CODEC) $(LDLIBS)

# NVDR against the browser's JPEG and WebP, and AVIF, JPEG XL and HEIC
# when avifenc, cjxl and heif-enc are installed, at equal quality (see
# scripts/bench_codecs.mjs); writes output/bench/.
bench: nvdr_encode nvdr_decode output/convert
	node scripts/bench_codecs.mjs

# The same on five large camera photos (2.5 to 21 MP), fetched and
# developed from raw by scripts/fetch_photos.py (needs rawpy).
bench-photos: nvdr_encode nvdr_decode output/convert
	python3 scripts/fetch_photos.py
	node scripts/bench_codecs.mjs samples/photos/*.png

# The decoder for the browser, as WebAssembly: clang and wasm-ld only.
wasm: public/nvdr.wasm

public/nvdr.wasm: $(CODEC) $(HEADERS) src/mc.c src/mc.h wasm/api.c wasm/libc.c wasm/include/*.h wasm/build.sh
	./wasm/build.sh

# The example files public/galeria.html shows: one image, and an album
# whose photos are a pan over one sample, interleaved with other samples.
demo: nvdr_encode nvdr_album
	mkdir -p public/demo output/demo_frames
	$(CC) -std=c11 -O2 -Isrc -Ivendor -o output/frames scripts/analysis/frames.c $(CODEC) $(LDLIBS)
	./output/frames samples/montanha_pessoas.jpg output/demo_frames pan 6
	./nvdr_encode samples/montanha_pessoas.jpg public/demo/montanha.nvdr > /dev/null
	./nvdr_album pack public/demo/galeria.nvda output/demo_frames/f000.ppm samples/OIP-1304511485.jpg \
	    output/demo_frames/f001.ppm output/demo_frames/f002.ppm samples/OIP-4140498144.jpg \
	    output/demo_frames/f003.ppm output/demo_frames/f004.ppm output/demo_frames/f005.ppm

clean:
	rm -f $(TOOLS) fuzz_nvdr libnvdr.a $(LIB_OBJ)

.PHONY: all check clean fuzz demo bench bench-photos install
