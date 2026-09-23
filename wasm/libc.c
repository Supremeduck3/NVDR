/*
 * The little C library the NVDR decoder needs in the browser: memory and
 * nothing else. No wasi-libc, no Emscripten: clang and wasm-ld build it.
 *
 * Memory is a stack. malloc takes from the top; freeing or growing the
 * block on top gives it back or grows it in place, anything else waits.
 * A decode frees everything it allocates before it returns, and the page
 * rewinds the stack between decodes (nvdr_wasm_rewind in api.c), so
 * nothing lingers.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

extern unsigned char __heap_base;
static uintptr_t top = 0;       /* first free byte */
static uintptr_t last = 0;      /* the block on top, or 0 */

typedef struct { size_t size; size_t pad[16 / sizeof(size_t) - 1]; } Head;   /* 16 bytes, keeps blocks aligned */

static int reserve(uintptr_t end) {
    size_t have = __builtin_wasm_memory_size(0) * 65536;
    if (end <= have) return 1;
    size_t pages = (end - have + 65535) / 65536;
    return __builtin_wasm_memory_grow(0, pages) != (size_t)-1;
}

uintptr_t nvdr_heap_top(void) { if (!top) top = ((uintptr_t)&__heap_base + 15) & ~(uintptr_t)15; return top; }
void nvdr_heap_rewind(uintptr_t to) { top = to; last = 0; }

void* malloc(size_t n) {
    uintptr_t at = nvdr_heap_top();
    n = (n + 15) & ~(size_t)15;
    if (!reserve(at + sizeof(Head) + n)) return NULL;
    ((Head*)at)->size = n;
    last = at;
    top = at + sizeof(Head) + n;
    return (void*)(at + sizeof(Head));
}

void free(void* p) {
    if (!p) return;
    uintptr_t at = (uintptr_t)p - sizeof(Head);
    if (at == last) { top = at; last = 0; }
}

void* calloc(size_t n, size_t size) {
    size_t total = n * size;
    if (size && total / size != n) return NULL;
    void* p = malloc(total);
    if (p) memset(p, 0, total);
    return p;
}

void* realloc(void* p, size_t n) {
    if (!p) return malloc(n);
    uintptr_t at = (uintptr_t)p - sizeof(Head);
    size_t old = ((Head*)at)->size;
    if (at == last) {
        size_t m = (n + 15) & ~(size_t)15;
        if (!reserve(at + sizeof(Head) + m)) return NULL;
        ((Head*)at)->size = m;
        top = at + sizeof(Head) + m;
        return p;
    }
    void* q = malloc(n);
    if (q) memcpy(q, p, old < n ? old : n);
    return q;
}

void* memcpy(void* d, const void* s, size_t n) { __builtin_memcpy(d, s, n); return d; }
void* memmove(void* d, const void* s, size_t n) { __builtin_memmove(d, s, n); return d; }
void* memset(void* d, int c, size_t n) { __builtin_memset(d, c, n); return d; }
int memcmp(const void* a, const void* b, size_t n) {
    const unsigned char *x = a, *y = b;
    for (size_t i = 0; i < n; i++) if (x[i] != y[i]) return x[i] < y[i] ? -1 : 1;
    return 0;
}
size_t strlen(const char* s) { size_t n = 0; while (s[n]) n++; return n; }
int strcmp(const char* a, const char* b) { while (*a && *a == *b) { a++; b++; } return (unsigned char)*a - (unsigned char)*b; }
int abs(int v) { return v < 0 ? -v : v; }
long labs(long v) { return v < 0 ? -v : v; }
