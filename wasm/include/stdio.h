#ifndef NVDR_WASM_STDIO_H
#define NVDR_WASM_STDIO_H
/* No files in the browser: only for the encoders' diagnostics and output,
 * which the linker drops. */
#include <stddef.h>
typedef struct FILE FILE;
extern FILE* stderr;
#define SEEK_SET 0
int fprintf(FILE* f, const char* fmt, ...);
FILE* fopen(const char* path, const char* mode);
int fclose(FILE* f);
int fseek(FILE* f, long off, int whence);
size_t fwrite(const void* p, size_t size, size_t n, FILE* f);
#endif
