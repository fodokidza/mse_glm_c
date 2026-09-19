#include "mse_util.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* ============================================================ i32vec */
void i32vec_init(i32vec *v) { v->data = NULL; v->len = 0; v->cap = 0; }

void i32vec_free(i32vec *v) {
    free(v->data);
    v->data = NULL; v->len = v->cap = 0;
}

void i32vec_reserve(i32vec *v, size_t n) {
    if (n <= v->cap) return;
    size_t newcap = v->cap ? v->cap * 2 : 8;
    if (newcap < n) newcap = n;
    v->data = (int32_t *)realloc(v->data, newcap * sizeof(int32_t));
    v->cap = newcap;
}

void i32vec_push(i32vec *v, int32_t x) {
    if (v->len == v->cap) i32vec_reserve(v, v->len + 1);
    v->data[v->len++] = x;
}

/* ============================================================ Arena */
void arena_init(Arena *a) { a->head = NULL; }

void arena_free(Arena *a) {
    ArenaBlock *b = a->head;
    while (b) { ArenaBlock *next = b->next; free(b); b = next; }
    a->head = NULL;
}

static ArenaBlock *arena_new_block(size_t min_size) {
    size_t cap = ARENA_BLOCK_SIZE;
    if (cap < min_size) cap = min_size;
    ArenaBlock *b = (ArenaBlock *)malloc(sizeof(ArenaBlock) + cap);
    b->next = NULL;
    b->used = 0;
    b->cap = cap;
    return b;
}

void *arena_alloc(Arena *a, size_t n) {
    /* 8-byte align */
    n = (n + 7u) & ~(size_t)7u;
    if (!a->head || a->head->used + n > a->head->cap) {
        ArenaBlock *b = arena_new_block(n);
        b->next = a->head;
        a->head = b;
    }
    void *p = a->head->data + a->head->used;
    a->head->used += n;
    return p;
}

char *arena_strndup(Arena *a, const char *s, size_t len) {
    char *p = (char *)arena_alloc(a, len + 1);
    memcpy(p, s, len);
    p[len] = '\0';
    return p;
}

/* ============================================================ StrMap */
static uint64_t fnv1a(const char *s, int32_t len) {
    uint64_t h = 1469598103934665603ULL;
    for (int32_t i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

void strmap_init(StrMap *m) {
    m->cap = 16;
    m->count = 0;
    m->slots = (StrMapSlot *)calloc(m->cap, sizeof(StrMapSlot));
}

void strmap_free(StrMap *m) {
    free(m->slots);
    m->slots = NULL; m->cap = 0; m->count = 0;
}

static int str_eq(const char *a, int32_t alen, const char *b, int32_t blen) {
    return alen == blen && memcmp(a, b, (size_t)alen) == 0;
}

static void strmap_grow(StrMap *m);

int strmap_get(const StrMap *m, const char *key, int32_t key_len, int32_t *out) {
    if (m->cap == 0) return 0;
    uint64_t h = fnv1a(key, key_len);
    size_t mask = m->cap - 1;
    size_t i = (size_t)h & mask;
    for (size_t probe = 0; probe < m->cap; probe++) {
        const StrMapSlot *s = &m->slots[i];
        if (!s->occupied) return 0;
        if (str_eq(s->key, s->key_len, key, key_len)) { *out = s->value; return 1; }
        i = (i + 1) & mask;
    }
    return 0;
}

void strmap_put(StrMap *m, const char *key, int32_t key_len, int32_t value) {
    if (m->count * 4 >= m->cap * 3) strmap_grow(m); /* load factor 0.75 */
    uint64_t h = fnv1a(key, key_len);
    size_t mask = m->cap - 1;
    size_t i = (size_t)h & mask;
    for (;;) {
        StrMapSlot *s = &m->slots[i];
        if (!s->occupied) {
            s->occupied = 1; s->key = key; s->key_len = key_len; s->value = value;
            m->count++;
            return;
        }
        if (str_eq(s->key, s->key_len, key, key_len)) { s->value = value; return; }
        i = (i + 1) & mask;
    }
}

static void strmap_grow(StrMap *m) {
    StrMapSlot *old = m->slots;
    size_t oldcap = m->cap;
    m->cap *= 2;
    m->slots = (StrMapSlot *)calloc(m->cap, sizeof(StrMapSlot));
    m->count = 0;
    for (size_t i = 0; i < oldcap; i++) {
        if (old[i].occupied) strmap_put(m, old[i].key, old[i].key_len, old[i].value);
    }
    free(old);
}

/* ============================================================ PairMap */
static uint64_t pair_hash(int64_t key) {
    uint64_t h = (uint64_t)key;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

static int64_t pack(int32_t a, int32_t b) {
    return ((int64_t)(uint32_t)a << 32) | (uint32_t)b;
}

void pairmap_init(PairMap *m) {
    m->cap = 16;
    m->occupied_count = 0;
    m->next_seq = 1;
    m->slots = (PairMapSlot *)calloc(m->cap, sizeof(PairMapSlot));
}

void pairmap_free(PairMap *m) {
    free(m->slots);
    m->slots = NULL; m->cap = 0; m->occupied_count = 0;
}

static PairMapSlot *pairmap_find_slot(PairMap *m, int64_t key) {
    size_t mask = m->cap - 1;
    size_t i = (size_t)(pair_hash(key) & mask);
    for (size_t probe = 0; probe < m->cap; probe++) {
        PairMapSlot *s = &m->slots[i];
        if (!s->occupied || s->key == key) return s;
        i = (i + 1) & mask;
    }
    return NULL; /* unreachable if load factor kept < 1 */
}

static void pairmap_grow(PairMap *m) {
    PairMapSlot *old = m->slots;
    size_t oldcap = m->cap;
    m->cap *= 2;
    m->slots = (PairMapSlot *)calloc(m->cap, sizeof(PairMapSlot));
    m->occupied_count = 0;
    for (size_t i = 0; i < oldcap; i++) {
        if (old[i].occupied) {
            PairMapSlot *dst = pairmap_find_slot(m, old[i].key);
            *dst = old[i];
            m->occupied_count++;
        }
    }
    free(old);
}

void pairmap_add(PairMap *m, int32_t a, int32_t b, int32_t delta) {
    if (m->occupied_count * 4 >= m->cap * 3) pairmap_grow(m);
    int64_t key = pack(a, b);
    PairMapSlot *s = pairmap_find_slot(m, key);
    if (!s->occupied) {
        s->occupied = 1; s->key = key; s->count = 0; s->active = 0;
        m->occupied_count++;
    }
    if (!s->active) {
        /* Revival (or first creation): matches Python `Counter[pair] += delta`
         * on a missing key — starts from 0, gets a brand-new insertion
         * position. */
        s->count = 0;
        s->seq = m->next_seq++;
        s->active = 1;
    }
    s->count += delta;
    if (s->count <= 0) {
        s->active = 0; /* matches `del pair_counts[pair]` */
    }
}

int pairmap_max(const PairMap *m, int32_t *a, int32_t *b, int32_t *count) {
    int found = 0;
    int32_t best_count = 0;
    int64_t best_seq = 0;
    int64_t best_key = 0;
    for (size_t i = 0; i < m->cap; i++) {
        const PairMapSlot *s = &m->slots[i];
        if (!s->occupied || !s->active) continue;
        if (!found || s->count > best_count ||
            (s->count == best_count && s->seq < best_seq)) {
            found = 1;
            best_count = s->count;
            best_seq = s->seq;
            best_key = s->key;
        }
    }
    if (!found) return 0;
    *a = (int32_t)((uint64_t)best_key >> 32);
    *b = (int32_t)((uint64_t)best_key & 0xffffffffu);
    *count = best_count;
    return 1;
}
