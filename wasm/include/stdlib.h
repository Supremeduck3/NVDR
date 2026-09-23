/* The C library the decoder needs, for wasm/libc.c (NVDR_WASM builds). */
#ifndef NVDR_WASM_STDLIB_H
#define NVDR_WASM_STDLIB_H
#include <stddef.h>
void* malloc(size_t n);
void* calloc(size_t n, size_t size);
void* realloc(void* p, size_t n);
void  free(void* p);
int   abs(int v);
long  labs(long v);
/* Declared for the encoder's sake; the decoder never calls them, and the
 * linker drops what calls them. */
void  qsort(void* base, size_t n, size_t size, int (*cmp)(const void*, const void*));
char* getenv(const char* name);
double atof(const char* s);
int   atoi(const char* s);
#endif
