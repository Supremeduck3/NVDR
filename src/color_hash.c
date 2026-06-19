#include "color_hash.h"
#include <stdlib.h>
#include <string.h>

// Knuth multiplicative hash
static inline uint32_t hash_rgb(uint32_t key, int mask) {
    return (key * 2654435761u) & (uint32_t)mask;
}

static inline uint32_t make_key(unsigned char r, unsigned char g, unsigned char b) {
    return ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
}

static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

void cht_init(ColorHashTable* t, int initial_capacity) {
    t->capacity = next_pow2(initial_capacity < 16 ? 16 : initial_capacity);
    t->count = 0;
    t->buckets = (ColorHashEntry*)calloc((size_t)t->capacity, sizeof(ColorHashEntry));
}

void cht_free(ColorHashTable* t) {
    if (t->buckets) {
        free(t->buckets);
        t->buckets = NULL;
    }
    t->capacity = 0;
    t->count = 0;
}

// Rehash to double capacity
static void cht_rehash(ColorHashTable* t) {
    int old_cap = t->capacity;
    ColorHashEntry* old = t->buckets;

    t->capacity = old_cap * 2;
    t->buckets = (ColorHashEntry*)calloc((size_t)t->capacity, sizeof(ColorHashEntry));
    t->count = 0;

    int mask = t->capacity - 1;
    for (int i = 0; i < old_cap; i++) {
        if (!old[i].occupied) continue;
        uint32_t h = hash_rgb(old[i].key, mask);
        while (t->buckets[h].occupied) {
            h = (h + 1) & (uint32_t)mask;
        }
        t->buckets[h] = old[i];
        t->count++;
    }
    free(old);
}

int cht_find_or_insert(ColorHashTable* t, unsigned char r, unsigned char g, unsigned char b,
                       int new_index, int* out_index) {
    // Rehash at 70% load
    if (t->count * 10 >= t->capacity * 7) {
        cht_rehash(t);
    }

    uint32_t key = make_key(r, g, b);
    int mask = t->capacity - 1;
    uint32_t h = hash_rgb(key, mask);

    while (t->buckets[h].occupied) {
        if (t->buckets[h].key == key) {
            *out_index = t->buckets[h].index;
            return 1; // found
        }
        h = (h + 1) & (uint32_t)mask;
    }

    // Not found — insert
    t->buckets[h].key = key;
    t->buckets[h].index = new_index;
    t->buckets[h].occupied = 1;
    t->count++;
    *out_index = new_index;
    return 0;
}
