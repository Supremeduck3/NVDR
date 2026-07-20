CC = gcc
CFLAGS = -std=c11 -O2 -Wall -Wextra -Ivendor -Isrc -fopenmp
TARGET = image_to_svg
SRCS = src/main.c src/image_io.c src/quadtree.c src/color.c src/svg_writer.c src/optimizer.c src/svbc_writer.c src/svbc_reader.c src/contour.c src/sat.c src/color_hash.c src/codebook_db.c

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) -lm -fopenmp

clean:
	rm -f $(TARGET)

.PHONY: clean
