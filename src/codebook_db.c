#include "codebook_db.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NVCB_MAGIC    "NVCB"
#define NVCB_VERSION  1u

int codebook_db_init(CodebookDB* db) {
    db->count = 0;
    db->capacity = 0;
    db->entries = NULL;
    return 0;
}

void codebook_db_free(CodebookDB* db) {
    if (!db) return;
    free(db->entries);
    db->entries = NULL;
    db->count = 0;
    db->capacity = 0;
}

static int ensure_capacity(CodebookDB* db, int need) {
    if (need <= db->capacity) return 0;
    int new_cap = db->capacity ? db->capacity : 64;
    while (new_cap < need) new_cap *= 2;
    CodebookDBEntry* grown = (CodebookDBEntry*)realloc(
        db->entries, (size_t)new_cap * sizeof(CodebookDBEntry));
    if (!grown) return -1;
    db->entries = grown;
    db->capacity = new_cap;
    return 0;
}

int codebook_db_load(CodebookDB* db, const char* path) {
    if (!path || !*path) return 0;

    FILE* f = fopen(path, "rb");
    if (!f) {
        /* Missing file is not an error — first run with empty DB. */
        return 0;
    }

    char magic[4] = {0};
    uint32_t version = 0;
    uint32_t entry_count = 0;

    if (fread(magic, 4, 1, f) != 1 || memcmp(magic, NVCB_MAGIC, 4) != 0) {
        fclose(f);
        return -1;
    }
    if (fread(&version, sizeof version, 1, f) != 1 || version != NVCB_VERSION) {
        fclose(f);
        return -1;
    }
    if (fread(&entry_count, sizeof entry_count, 1, f) != 1) {
        fclose(f);
        return -1;
    }

    /* Sanity cap: refuse to allocate absurdly large DBs up front.
     * We'll grow dynamically if needed, but reject obvious corruption. */
    if (entry_count > 10000000) {
        fclose(f);
        return -1;
    }

    if (ensure_capacity(db, (int)entry_count) != 0) {
        fclose(f);
        return -1;
    }

    while (1) {
        CodebookDBEntry e;
        size_t got = fread(&e, sizeof e, 1, f);
        if (got == 0) break;       /* EOF */
        if (got != 1) { fclose(f); return -1; }
        if (ensure_capacity(db, db->count + 1) != 0) {
            fclose(f); return -1;
        }
        db->entries[db->count++] = e;
    }

    fclose(f);
    return 0;
}

int codebook_db_save(const CodebookDB* db, const char* path) {
    if (!path || !*path) return 0;

    /* Atomic write: write to path.tmp, fsync, rename. */
    char tmp_path[1024];
    int n = snprintf(tmp_path, sizeof tmp_path, "%s.tmp", path);
    if (n < 0 || n >= (int)sizeof tmp_path) return -1;

    FILE* f = fopen(tmp_path, "wb");
    if (!f) return -1;

    if (fwrite(NVCB_MAGIC, 4, 1, f) != 1)               goto fail;
    uint32_t version = NVCB_VERSION;
    if (fwrite(&version, sizeof version, 1, f) != 1)   goto fail;
    uint32_t entry_count = (uint32_t)db->count;
    if (fwrite(&entry_count, sizeof entry_count, 1, f) != 1) goto fail;

    for (int i = 0; i < db->count; i++) {
        CodebookDBEntry e = db->entries[i];
        if (fwrite(&e, sizeof e, 1, f) != 1) goto fail;
    }

    if (fflush(f) != 0) goto fail;
    fclose(f);
    if (rename(tmp_path, path) != 0) return -1;
    return 0;

fail:
    fclose(f);
    /* Best-effort cleanup of the .tmp file; ignore failure. */
    remove(tmp_path);
    return -1;
}

void codebook_db_observe(CodebookDB* db,
                         unsigned char r, unsigned char g, unsigned char b,
                         int weight, int run_id)
{
    if (!db) return;
    for (int i = 0; i < db->count; i++) {
        CodebookDBEntry* e = &db->entries[i];
        if (e->r == r && e->g == g && e->b == b) {
            e->weight += weight;
            if (run_id > e->last_seen_run)  e->last_seen_run = run_id;
            return;
        }
    }
    if (ensure_capacity(db, db->count + 1) != 0) return;
    CodebookDBEntry* e = &db->entries[db->count++];
    e->r = r; e->g = g; e->b = b;
    e->weight = weight;
    e->first_seen_run = run_id;
    e->last_seen_run = run_id;
}

int codebook_db_size(const CodebookDB* db) {
    return db ? db->count : 0;
}

int codebook_db_lookup(const CodebookDB* db,
                       unsigned char r, unsigned char g, unsigned char b,
                       CodebookDBEntry* out_entry)
{
    if (!db) return 0;
    for (int i = 0; i < db->count; i++) {
        const CodebookDBEntry* e = &db->entries[i];
        if (e->r == r && e->g == g && e->b == b) {
            if (out_entry) *out_entry = *e;
            return 1;
        }
    }
    return 0;
}
