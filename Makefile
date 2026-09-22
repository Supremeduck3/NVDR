# NVDR — still images (format v10) and sequences.
#
# Deliberately dependency-free: C11, libm, zlib and stb_image. Nothing
# here needs a GPU, and nothing here should start needing one.

CC      = gcc
# OpenMP parallelises the sequence encoder's block motion search, which is
# per-block and shares nothing. Without it the pragmas are ignored and the
# build still works.
OPENMP  = $(shell $(CC) -fopenmp -E - < /dev/null > /dev/null 2>&1 && echo -fopenmp)
CFLAGS  = -std=c11 -O2 -Wall -Wextra -Wno-unused-parameter -Isrc -Ivendor $(OPENMP)
LDLIBS  = -lm -lz

CODEC   = src/nvdr.c src/entropy.c
HEADERS = src/nvdr.h src/entropy.h
SEQ     = $(CODEC) src/nvdrv.c
SEQ_H   = $(HEADERS) src/nvdrv.h
TOOLS   = nvdr_encode nvdr_decode nvdrv_encode nvdrv_decode nvdr_album

all: $(TOOLS)

nvdr_encode: tools/nvdr_encode.c $(CODEC) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ tools/nvdr_encode.c $(CODEC) $(LDLIBS)

nvdr_decode: tools/nvdr_decode.c $(CODEC) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ tools/nvdr_decode.c $(CODEC) $(LDLIBS)

nvdr_album: tools/nvdr_album.c src/nvda.c src/nvda.h $(CODEC) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ tools/nvdr_album.c src/nvda.c $(CODEC) $(LDLIBS)

nvdrv_encode: tools/nvdrv_encode.c $(SEQ) $(SEQ_H)
	$(CC) $(CFLAGS) -o $@ tools/nvdrv_encode.c $(SEQ) $(LDLIBS)

nvdrv_decode: tools/nvdrv_decode.c $(SEQ) $(SEQ_H)
	$(CC) $(CFLAGS) -o $@ tools/nvdrv_decode.c $(SEQ) $(LDLIBS)

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

clean:
	rm -f $(TOOLS) fuzz_nvdr

.PHONY: all check clean fuzz
