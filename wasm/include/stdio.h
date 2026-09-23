#ifndef NVDR_WASM_STDIO_H
#define NVDR_WASM_STDIO_H
/* No files in the browser: only for the encoder's diagnostics, which the
 * linker drops. */
typedef struct FILE FILE;
extern FILE* stderr;
int fprintf(FILE* f, const char* fmt, ...);
#endif
