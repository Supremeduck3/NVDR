/*
 * The little C library the NVDR decoder needs in the browser: memory and
 * nothing else. No wasi-libc, no Emscripten: clang and wasm-ld build it.
 *
 * Memory is a heap of chunks with boundary tags, the free ones kept in
 * bins by size and merged with their free neighbours as they are given
 * back, the way dlmalloc does it, much simplified. Allocations outlive a
 * decode: a sequence's decoder (nvdrv.c) keeps its frames from one call
 * to the next and frees them in any order.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

extern unsigned char __heap_base;

/* A chunk starts 8 bytes before a 16-byte boundary, so its payload, after
 * the two words of header, is aligned; sizes are multiples of 16. The
 * size of the chunk before is only meaningful when that one is free. */
typedef struct Chunk {
    size_t prev_size;
    size_t head;             /* size | INUSE | PINUSE */
    struct Chunk* next;      /* free chunks: their bin's list */
    struct Chunk* prev;
} Chunk;
#define INUSE  ((size_t)1)
#define PINUSE ((size_t)2)
#define HEAD   (2 * sizeof(size_t))
#define MIN_CHUNK ((sizeof(Chunk) + 15) & ~(size_t)15)
#define SIZE(c) ((c)->head & ~(size_t)15)
#define AT(c, off) ((Chunk*)((uintptr_t)(c) + (off)))

/* Exact bins of 16 bytes below 1 KB, then four per power of two. */
#define NBINS 152
static Chunk* bins[NBINS];
static uint32_t binmap[(NBINS + 31) / 32];
/* Everything from `top` on is free; the chunk before it is always in use. */
static Chunk* top;

static int bin_of(size_t size) {
    if (size < 1024) return (int)(size >> 4);
    int b = 31 - __builtin_clz((uint32_t)size);
    return 64 + (b - 10) * 4 + (int)((size >> (b - 2)) & 3);
}

static void bin_insert(Chunk* c) {
    int i = bin_of(SIZE(c));
    c->prev = NULL;
    c->next = bins[i];
    if (bins[i]) bins[i]->prev = c;
    bins[i] = c;
    binmap[i >> 5] |= 1u << (i & 31);
}

static void bin_unlink(Chunk* c) {
    int i = bin_of(SIZE(c));
    if (c->prev) c->prev->next = c->next; else bins[i] = c->next;
    if (c->next) c->next->prev = c->prev;
    if (!bins[i]) binmap[i >> 5] &= ~(1u << (i & 31));
}

static int reserve(uintptr_t end) {
    size_t have = __builtin_wasm_memory_size(0) * 65536;
    if (end <= have) return 1;
    size_t pages = (end - have + 65535) / 65536;
    return __builtin_wasm_memory_grow(0, pages) != (size_t)-1;
}

static Chunk* top_chunk(void) {
    if (!top) top = (Chunk*)((((uintptr_t)&__heap_base + HEAD + 15) & ~(uintptr_t)15) - HEAD);
    return top;
}

/* The first free chunk of at least `need` bytes, out of its bin. */
static Chunk* take_free(size_t need) {
    int i = bin_of(need);
    for (Chunk* c = bins[i]; c; c = c->next)
        if (SIZE(c) >= need) { bin_unlink(c); return c; }
    /* Every chunk of a later bin is big enough. */
    for (int w = (i + 1) >> 5, first = 1; w < (int)(sizeof binmap / sizeof *binmap); w++, first = 0) {
        uint32_t m = binmap[w];
        if (first) m &= (i + 1) & 31 ? ~0u << ((i + 1) & 31) : ~0u;
        if (!m) continue;
        Chunk* c = bins[(w << 5) + __builtin_ctz(m)];
        bin_unlink(c);
        return c;
    }
    return NULL;
}

void free(void* p);

void* malloc(size_t n) {
    if (n > ((size_t)1 << 30)) return NULL;
    size_t need = (n + HEAD + 15) & ~(size_t)15;
    if (need < MIN_CHUNK) need = MIN_CHUNK;
    Chunk* c = take_free(need);
    if (c) {
        size_t size = SIZE(c);
        if (size - need >= MIN_CHUNK) {
            /* The rest is a chunk of its own, free; the one after it
             * already knows its neighbour is free. */
            Chunk* r = AT(c, need);
            r->head = (size - need) | PINUSE;
            AT(r, size - need)->prev_size = size - need;
            bin_insert(r);
            size = need;
        } else {
            AT(c, size)->head |= PINUSE;
        }
        c->head = size | INUSE | PINUSE;
        return (char*)c + HEAD;
    }
    c = top_chunk();
    /* Room for the chunk and the header of whatever comes after it. */
    if (!reserve((uintptr_t)c + need + HEAD)) return NULL;
    c->head = need | INUSE | PINUSE;
    top = AT(c, need);
    return (char*)c + HEAD;
}

void free(void* p) {
    if (!p) return;
    Chunk* c = (Chunk*)((char*)p - HEAD);
    size_t size = SIZE(c);
    Chunk* next = AT(c, size);
    if (!(c->head & PINUSE)) {
        size_t ps = c->prev_size;
        c = (Chunk*)((uintptr_t)c - ps);
        bin_unlink(c);
        size += ps;
    }
    if (next == top) { top = c; return; }
    if (!(next->head & INUSE)) {
        size_t ns = SIZE(next);
        bin_unlink(next);
        size += ns;
        next = AT(c, size);
        if (next == top) { top = c; return; }
    }
    c->head = size | PINUSE;
    next->prev_size = size;
    next->head &= ~PINUSE;
    bin_insert(c);
}

void* calloc(size_t n, size_t size) {
    size_t total = n * size;
    if (size && total / size != n) return NULL;
    void* p = malloc(total);
    if (p) memset(p, 0, total);
    return p;
}

/* Gives back the end of an in-use chunk past `keep` bytes, when that is a
 * chunk's worth: made a chunk of its own, then freed like any other. */
static void trim(Chunk* c, size_t keep) {
    size_t size = SIZE(c);
    if (size - keep < MIN_CHUNK) return;
    Chunk* r = AT(c, keep);
    r->head = (size - keep) | INUSE | PINUSE;
    c->head = keep | (c->head & 15);
    free((char*)r + HEAD);
}

void* realloc(void* p, size_t n) {
    if (!p) return malloc(n);
    if (n > ((size_t)1 << 30)) return NULL;
    Chunk* c = (Chunk*)((char*)p - HEAD);
    size_t size = SIZE(c);
    size_t need = (n + HEAD + 15) & ~(size_t)15;
    if (need < MIN_CHUNK) need = MIN_CHUNK;
    if (need <= size) { trim(c, need); return p; }
    Chunk* next = AT(c, size);
    if (next == top) {
        if (!reserve((uintptr_t)c + need + HEAD)) return NULL;
        c->head = need | (c->head & 15);
        top = AT(c, need);
        return p;
    }
    if (!(next->head & INUSE) && size + SIZE(next) >= need) {
        size_t total = size + SIZE(next);
        bin_unlink(next);
        c->head = total | (c->head & 15);
        AT(c, total)->head |= PINUSE;
        trim(c, need);
        return p;
    }
    void* q = malloc(n);
    if (!q) return NULL;
    memcpy(q, p, size - HEAD);
    free(p);
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
