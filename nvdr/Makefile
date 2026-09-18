# NVDR — Progressive Residual Stack, still-image implementation.
#
# Deliberately dependency-free: C11, libm, and stb_image. Nothing here
# needs a GPU, and nothing here should start needing one.

CC      = gcc
CFLAGS  = -std=c11 -O2 -Wall -Wextra -Wno-unused-parameter -I. -I../vendor
LDLIBS  = -lm -lz

TOOLS   = nvdr_encode nvdr_decode

all: $(TOOLS)

nvdr_encode: tools/nvdr_encode.c nvdr.c entropy.c nvdr.h entropy.h
	$(CC) $(CFLAGS) -o $@ tools/nvdr_encode.c nvdr.c entropy.c $(LDLIBS)

nvdr_decode: tools/nvdr_decode.c nvdr.c entropy.c nvdr.h entropy.h
	$(CC) $(CFLAGS) -o $@ tools/nvdr_decode.c nvdr.c entropy.c $(LDLIBS)

clean:
	rm -f $(TOOLS)

.PHONY: all clean
