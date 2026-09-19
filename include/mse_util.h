/* mse_util.h — shared low-level building blocks.
 *
 * These exist so the rest of the port can use flat, cache-friendly,
 * manually-managed memory (the whole point of doing this in C) instead
 * of reaching for a general-purpose dict/list at every turn the way the
 * Python original does. Three pieces:
 *
 *   i32vec     — growable int32_t array (stands in for Python's list/
 *                array('i') used throughout graph.py/tokenizer.py).
 *   Arena      — bump allocator for strings/objects that live for the
 *                lifetime of a model (vocab strings, etc.) — one big
 *                block, no per-string free(), no fragmentation.
 *   StrMap     — open-addressing string -> int32 map (token_to_id).
 *   PairMap    — open-addressing (int32,int32) -> int32 map with
 *                Python-Counter-compatible insertion-order tracking,
 *                used by the BPE merge loop's "most frequent pair"
 *                selection (see tokenizer.c).
 */
#ifndef MSE_UTIL_H
#define MSE_UTIL_H

#include <stdint.h>
#include <stddef.h>

/* ----------------------------------------------------------------- i32vec */
typedef struct {
    int32_t *data;
    size_t   len;
    size_t   cap;
} i32vec;

void i32vec_init(i32vec *v);
void i32vec_free(i32vec *v);
void i32vec_push(i32vec *v, int32_t x);
void i32vec_reserve(i32vec *v, size_t n);

/* ----------------------------------------------------------------- Arena */
#define ARENA_BLOCK_SIZE (1 << 20)  /* 1 MiB blocks */

typedef struct ArenaBlock {
    struct ArenaBlock *next;
    size_t used;
    size_t cap;
    char   data[];
} ArenaBlock;

typedef struct {
    ArenaBlock *head;
} Arena;

void  arena_init(Arena *a);
void  arena_free(Arena *a);
void *arena_alloc(Arena *a, size_t n);
/* Copies [s, s+len) into the arena and NUL-terminates it. */
char *arena_strndup(Arena *a, const char *s, size_t len);

/* ----------------------------------------------------------------- StrMap
 * Open-addressing hash map: interned string -> int32 id. Used for
 * BPETokenizer.token_to_id. Keys are owned by the caller (typically an
 * Arena) — StrMap only stores pointers + lengths.
 */
typedef struct {
    const char *key;
    int32_t     key_len;
    int32_t     value;
    uint8_t     occupied;
} StrMapSlot;

typedef struct {
    StrMapSlot *slots;
    size_t      cap;    /* power of 2 */
    size_t      count;
} StrMap;

void    strmap_init(StrMap *m);
void    strmap_free(StrMap *m);
/* Returns 1 and sets *out if found, else 0. */
int     strmap_get(const StrMap *m, const char *key, int32_t key_len, int32_t *out);
/* Inserts or overwrites. */
void    strmap_put(StrMap *m, const char *key, int32_t key_len, int32_t value);

/* ----------------------------------------------------------------- PairMap
 * (int32 a, int32 b) -> { count, seq }, with soft-delete + revival
 * semantics matching CPython's Counter dict: deleting a key and later
 * re-inserting it gives it a NEW, later position in "most recently
 * (re)inserted" order. seq is a monotonically increasing counter used
 * to replicate Counter.most_common(1)'s tie-break (ties go to the
 * least-recently-(re)inserted pair, matching CPython's max() semantics
 * over dict iteration order for n=1).
 *
 * KNOWN LIMITATION: this reproduces the *tie-break rule*, not bit-for-
 * bit CPython dict slot ordering — the two agree on which pair the
 * algorithm treats as "first" among equal counts as long as pair
 * identities and revival events line up, which they do here since we
 * drive PairMap through the exact same add/subtract/delete/revive
 * sequence tokenizer.c performs. Real corpora essentially never hit an
 * exact top-pair-frequency tie (same caveat the Python module itself
 * documents); this only matters for byte-for-byte reproducibility on
 * adversarial/synthetic ties.
 */
typedef struct {
    int64_t key;      /* packed (a<<32)|b, b as uint32 */
    int32_t count;
    int64_t seq;
    uint8_t occupied;  /* slot has ever held this key (probe-chain integrity) */
    uint8_t active;    /* key currently "present" (count > 0, not deleted) */
} PairMapSlot;

typedef struct {
    PairMapSlot *slots;
    size_t       cap;     /* power of 2 */
    size_t       occupied_count; /* occupied (incl. tombstoned), for resize */
    int64_t      next_seq;
} PairMap;

void  pairmap_init(PairMap *m);
void  pairmap_free(PairMap *m);
/* Adds delta to (a,b)'s count. If the key was absent/inactive, it is
 * (re)created with count=delta and a fresh seq (matching Python's
 * `Counter[pair] += delta` on a missing/deleted key). If the resulting
 * count <= 0, the key becomes inactive (matching `del pair_counts[pair]`
 * when count drops to <= 0) — its count is NOT preserved across the
 * inactive period.
 */
void  pairmap_add(PairMap *m, int32_t a, int32_t b, int32_t delta);
/* Finds the active entry with the highest count, tie-broken by lowest
 * seq (== earliest among currently-active). Returns 0 if none active. */
int   pairmap_max(const PairMap *m, int32_t *a, int32_t *b, int32_t *count);

#endif /* MSE_UTIL_H */
