#ifndef COLOR_HASH_H
#define COLOR_HASH_H

#include <stdint.h>

/*
 * color_hash.h — Open-addressing hash table for RGB color deduplication
 *
 * Replaces O(N) linear scans in optimizer_apply_ilut() and svbc_writer.c
 * with O(1) amortized lookups.
 */

typedef struct {
    uint32_t key;       // (r << 16) | (g << 8) | b
    int      index;     // index in the external ColorWeight array
    uint8_t  occupied;  // 0 = empty, 1 = used
} ColorHashEntry;

typedef struct {
    ColorHashEntry* buckets;
    int capacity;       // always a power of 2
    int count;
} ColorHashTable;

// Initialize hash table. initial_capacity will be rounded up to next power of 2.
void cht_init(ColorHashTable* t, int initial_capacity);

// Free hash table memory.
void cht_free(ColorHashTable* t);

// Look up (r,g,b) in the table.
// If found: returns 1, sets *out_index to the stored index.
// If not found: inserts with index=new_index, returns 0.
// Automatically rehashes at 70% load.
int cht_find_or_insert(ColorHashTable* t, unsigned char r, unsigned char g, unsigned char b,
                       int new_index, int* out_index);

#endif /* COLOR_HASH_H */
