CC = gcc
CFLAGS = -std=c11 -O2 -Wall -Wextra -Ivendor -Isrc -fopenmp
LDLIBS = -lm -lz
TARGET = image_to_svg
CHECKER = svbc_check
SRCS = src/main.c src/image_io.c src/quadtree.c src/color.c src/svg_writer.c src/optimizer.c src/svbc_writer.c src/svbc_reader.c src/contour.c src/sat.c src/color_hash.c src/codebook_db.c

all: $(TARGET) $(CHECKER)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LDLIBS) -fopenmp

# Independent verifier: decodes a container through the public reader and
# checks it against the source image. No OpenMP, no zlib — it must stay
# buildable on its own so a consumer can vet a file without the encoder.
$(CHECKER): tools/svbc_check.c src/svbc_reader.c
	$(CC) -std=c11 -O2 -Wall -Wextra -Ivendor -Isrc -o $(CHECKER) tools/svbc_check.c src/svbc_reader.c -lm

# Regression gate: determinism, structure and size across samples/.
verify: all
	python3 scripts/verify.py

clean:
	rm -f $(TARGET) $(CHECKER)

.PHONY: all clean verify
