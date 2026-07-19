#ifndef CODEBOOK_DB_H
#define CODEBOOK_DB_H

#include <stdint.h>
#include <stddef.h>

/*
 * codebook_db.h — Persisted cross-file codebook (Fluid Mechanism A bootstrap)
 *
 * Persists color palette entries discovered across runs of
 * optimizer_apply_ilut() so that future image conversions can re-use
 * the same color IDs without re-discovering them.
 *
 * File format (binary little-endian, append-only except on rotation):
 *   [4 bytes : "NVCB" magic]
 *   [4 bytes : format version (uint32 LE, currently 1)]
 *   [4 bytes : entry_count (uint32 LE; informational only)]
 *   repeat until EOF, each 16 bytes:
 *     [3 bytes: r, g, b                              ]
 *     [4 bytes: weight (int32 LE, accumulated area)   ]
 *     [4 bytes: first_seen_run (int32 LE, 1-based)    ]
 *     [4 bytes: last_seen_run  (int32 LE, 1-based)    ]
 *     [1 byte : padding/reserved (0x00)               ]
 *
 * Compact (16-byte records). Loads/saves are O(N) over records.
 * No external dependencies. No locking — call sites run single-threaded
 * around the load/save boundary.
 *
 * The codebook is intentionally NOT used to bias the iLUT selection
 * algorithm — it only seeds pre-existing colors so they get a head
 * start in the Pareto loop. Selection still honors current-run weights.
 */

typedef struct {
    unsigned char r, g, b;
    int32_t       weight;
    int32_t       first_seen_run;
    int32_t       last_seen_run;
} CodebookDBEntry;

typedef struct {
    CodebookDBEntry* entries;
    int              count;
    int              capacity;
} CodebookDB;

/* Initialize an empty in-memory DB. Returns 0 on success. */
int codebook_db_init(CodebookDB* db);

/* Free DB-owned memory. Safe to call on uninitialized struct. */
void codebook_db_free(CodebookDB* db);

/* Load a codebook from `path`. Append-mode is N/A — the file is
 * re-read in full. Returns 0 on success (including when path is NULL
 * or empty, in which case the DB stays empty). Returns -1 on read
 * error or unrecognized magic.
 * Missing file is NOT an error — that means "first run." */
int codebook_db_load(CodebookDB* db, const char* path);

/* Persist the current DB to `path` atomically (write tmp + rename).
 * Existing files are overwritten. Returns 0 on success, -1 on error.
 * NULL or empty path returns 0 (no-op). */
int codebook_db_save(const CodebookDB* db, const char* path);

/* Append (or update) a single (r, g, b, weight) observation from
 * `run_id`. If the color is already in the DB, weight is added.
 * If the color is new, a new entry is appended. */
void codebook_db_observe(CodebookDB* db,
                          unsigned char r, unsigned char g, unsigned char b,
                          int weight, int run_id);

/* Number of unique colors currently stored. */
int codebook_db_size(const CodebookDB* db);

/* Lookup helper (returns 1 + sets *out_entry on hit, 0 on miss). */
int codebook_db_lookup(const CodebookDB* db,
                        unsigned char r, unsigned char g, unsigned char b,
                        CodebookDBEntry* out_entry);

#endif /* CODEBOOK_DB_H */
