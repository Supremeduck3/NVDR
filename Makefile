# NVDR — Progressive Residual Stack, still-image implementation.
#
# Deliberately dependency-free: C11, libm, zlib and stb_image. Nothing
# here needs a GPU, and nothing here should start needing one.

CC      = gcc
CFLAGS  = -std=c11 -O2 -Wall -Wextra -Wno-unused-parameter -Isrc -Ivendor
LDLIBS  = -lm -lz

CODEC   = src/nvdr.c src/entropy.c
HEADERS = src/nvdr.h src/entropy.h
SEQ     = $(CODEC) src/nvdrv.c
SEQ_H   = $(HEADERS) src/nvdrv.h
TOOLS   = nvdr_encode nvdr_decode nvdrv_encode nvdrv_decode

all: $(TOOLS)

nvdr_encode: tools/nvdr_encode.c $(CODEC) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ tools/nvdr_encode.c $(CODEC) $(LDLIBS)

nvdr_decode: tools/nvdr_decode.c $(CODEC) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ tools/nvdr_decode.c $(CODEC) $(LDLIBS)

nvdrv_encode: tools/nvdrv_encode.c $(SEQ) $(SEQ_H)
	$(CC) $(CFLAGS) -o $@ tools/nvdrv_encode.c $(SEQ) $(LDLIBS)

nvdrv_decode: tools/nvdrv_decode.c $(SEQ) $(SEQ_H)
	$(CC) $(CFLAGS) -o $@ tools/nvdrv_decode.c $(SEQ) $(LDLIBS)

# The regression gate: encodes every sample, checks quality against a
# floor, checks the encoder is deterministic, and checks the C and JS
# decoders agree byte for byte.
check: $(TOOLS)
	python3 scripts/verify.py

clean:
	rm -f $(TOOLS)

.PHONY: all check clean
