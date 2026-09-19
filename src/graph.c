/* graph.c — C port of graph.py.
 *
 * Python builds these with dict/set/defaultdict, relying on their
 * insertion-order and membership semantics to get deterministic,
 * order-preserving dedup. This port replaces every one of those with
 * a single reusable trick: treat any fixed-width tuple (a pair, a
 * triple) or even a whole variable-length token sequence as a raw
 * byte string and dedupe/group it through the same generic StrMap
 * (bytes -> first-seen index) already written for the tokenizer's
 * vocab — one hash map implementation serving every "have I seen this
 * tuple before" question in the codebase instead of a different
 * dict/set type per call site.
 */
#include "mse_graph.h"
#include "mse_format.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ----------------------------------------------------- generic helpers */
/* Sort helper: stable sort of `n` items by an int32 primary key, tie-
 * broken by original index (== stability), via an explicit (key,
 * origidx) pair array + qsort. `reorder` receives the resulting
 * permutation (old index for each new position). */
typedef struct { int32_t key; int32_t idx; } SortPair;
static int sortpair_cmp(const void *pa, const void *pb) {
    const SortPair *a = (const SortPair *)pa, *b = (const SortPair *)pb;
    if (a->key != b->key) return (a->key < b->key) ? -1 : 1;
    return (a->idx < b->idx) ? -1 : (a->idx > b->idx ? 1 : 0);
}
static int32_t *stable_sort_perm(const int32_t *keys, int32_t n) {
    size_t cnt = (n > 0) ? (size_t)n : 1u;
    SortPair *sp = (SortPair *)malloc(sizeof(SortPair) * cnt);
    for (int32_t i = 0; i < n; i++) { sp[i].key = keys[i]; sp[i].idx = i; }
    if (n > 0) qsort(sp, (size_t)n, sizeof(SortPair), sortpair_cmp);
    int32_t *perm = (int32_t *)malloc(sizeof(int32_t) * cnt);
    for (int32_t i = 0; i < n; i++) perm[i] = sp[i].idx;
    free(sp);
    return perm;
}

static int64_t pack2(int32_t a, int32_t b) { return ((int64_t)(uint32_t)a << 32) | (uint32_t)b; }

/* ============================================================ EdgeMatrix */
void em_init(EdgeMatrix *m) { memset(m, 0, sizeof(*m)); }
void em_free(EdgeMatrix *m) { free(m->src); free(m->dst); free(m->count); free(m->index); memset(m, 0, sizeof(*m)); }

void em_build(EdgeMatrix *m, const i32vec *sequences, int32_t n_seq, int32_t vocab_size) {
    StrMap seen; strmap_init(&seen);
    Arena arena; arena_init(&arena);
    i32vec oa, ob, oc; i32vec_init(&oa); i32vec_init(&ob); i32vec_init(&oc);

    for (int32_t s = 0; s < n_seq; s++) {
        const i32vec *seq = &sequences[s];
        for (size_t i = 0; i + 1 < seq->len; i++) {
            int32_t a = seq->data[i], b = seq->data[i + 1];
            int64_t key = pack2(a, b);
            int32_t idx;
            if (strmap_get(&seen, (const char *)&key, (int32_t)sizeof(key), &idx)) {
                oc.data[idx]++;
            } else {
                idx = (int32_t)oa.len;
                i32vec_push(&oa, a); i32vec_push(&ob, b); i32vec_push(&oc, 1);
                char *owned = arena_strndup(&arena, (const char *)&key, (int32_t)sizeof(key));
                strmap_put(&seen, owned, (int32_t)sizeof(key), idx);
            }
        }
    }
    strmap_free(&seen); arena_free(&arena);

    em_build_from_pairs(m, &oa, &ob, &oc, vocab_size);
    i32vec_free(&oa); i32vec_free(&ob); i32vec_free(&oc);
}

/* Builds an EdgeMatrix from already-deduplicated (src,dst,count)
 * arrays -- the CSR-indexing tail of em_build(), split out for the
 * same reason as bm_build_from_triples: model.c's incremental merge
 * needs it too, on a (src,dst) union with counts already summed. */
void em_build_from_pairs(EdgeMatrix *m, const i32vec *src, const i32vec *dst, const i32vec *count, int32_t vocab_size) {
    em_free(m);
    m->vocab_size = vocab_size;

    int32_t n = (int32_t)src->len;
    int32_t *perm = stable_sort_perm(src->data, n);
    m->src = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n ? n : 1));
    m->dst = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n ? n : 1));
    m->count = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n ? n : 1));
    for (int32_t i = 0; i < n; i++) {
        int32_t p = perm[i];
        m->src[i] = src->data[p]; m->dst[i] = dst->data[p]; m->count[i] = count->data[p];
    }
    m->n = n;
    free(perm);

    m->index = (int32_t *)calloc((size_t)vocab_size + 1, sizeof(int32_t));
    for (int32_t i = 0; i < n; i++) m->index[m->src[i] + 1]++;
    for (int32_t i = 1; i <= vocab_size; i++) m->index[i] += m->index[i - 1];
}

void em_successors(const EdgeMatrix *m, int32_t token, i32vec *out) {
    if (token < 0 || token + 1 >= m->vocab_size + 1 || !m->index) return;
    int32_t start = m->index[token], end = m->index[token + 1];
    for (int32_t i = start; i < end; i++) i32vec_push(out, m->dst[i]);
}

int32_t em_frequency(const EdgeMatrix *m, int32_t token, int32_t candidate) {
    if (token < 0 || token + 1 >= m->vocab_size + 1 || !m->index) return 0;
    int32_t start = m->index[token], end = m->index[token + 1];
    for (int32_t i = start; i < end; i++) if (m->dst[i] == candidate) return m->count[i];
    return 0;
}

int em_save(const EdgeMatrix *m, const char *path) {
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    MseEdgeHeader h = { MSE_EDGE_MAGIC, 1, m->vocab_size, m->n, m->vocab_size + 1 };
    fwrite(&h, sizeof(h), 1, f);
    fwrite(m->src, sizeof(int32_t), (size_t)m->n, f);
    fwrite(m->dst, sizeof(int32_t), (size_t)m->n, f);
    fwrite(m->count, sizeof(int32_t), (size_t)m->n, f);
    fwrite(m->index, sizeof(int32_t), (size_t)h.index_len, f);
    fclose(f); return 0;
}

int em_load(EdgeMatrix *m, const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    MseEdgeHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1 || h.magic != MSE_EDGE_MAGIC) { fclose(f); return -1; }
    em_init(m); /* NOT em_free(m): `m` may be freshly declared/uninitialized
                 * (Python's from_dict() is a classmethod returning a new
                 * object; this is the instance-method equivalent, so it
                 * must not assume `m` already holds valid pointers). */
    m->vocab_size = h.vocab_size; m->n = h.n_edges;
    m->src = (int32_t *)malloc(sizeof(int32_t) * (size_t)(m->n ? m->n : 1));
    m->dst = (int32_t *)malloc(sizeof(int32_t) * (size_t)(m->n ? m->n : 1));
    m->count = (int32_t *)malloc(sizeof(int32_t) * (size_t)(m->n ? m->n : 1));
    m->index = (int32_t *)malloc(sizeof(int32_t) * (size_t)h.index_len);
    int ok = 1;
    if (m->n > 0) {
        ok &= fread(m->src, sizeof(int32_t), (size_t)m->n, f) == (size_t)m->n;
        ok &= fread(m->dst, sizeof(int32_t), (size_t)m->n, f) == (size_t)m->n;
        ok &= fread(m->count, sizeof(int32_t), (size_t)m->n, f) == (size_t)m->n;
    }
    ok &= fread(m->index, sizeof(int32_t), (size_t)h.index_len, f) == (size_t)h.index_len;
    fclose(f);
    return ok ? 0 : -1;
}

/* ========================================================== BridgeMatrix
 * OrderedGroups: pair-key -> member index list, preserving the order
 * in which each key was FIRST seen (matches Python's
 * defaultdict(list) + dict-iteration-order semantics used for
 * groups_by_st / groups_by_sb).
 */
typedef struct { int64_t key; i32vec members; } OGroup;
typedef struct {
    StrMap key_to_group;  /* packed key bytes -> index into groups[] */
    Arena  arena;
    OGroup *groups;
    int32_t n, cap;
} OrderedGroups;

static void ogroups_init(OrderedGroups *g) {
    strmap_init(&g->key_to_group); arena_init(&g->arena);
    g->groups = NULL; g->n = 0; g->cap = 0;
}
static void ogroups_free(OrderedGroups *g) {
    for (int32_t i = 0; i < g->n; i++) i32vec_free(&g->groups[i].members);
    free(g->groups);
    strmap_free(&g->key_to_group); arena_free(&g->arena);
}
static void ogroups_add(OrderedGroups *g, int64_t key, int32_t val) {
    int32_t idx;
    if (!strmap_get(&g->key_to_group, (const char *)&key, (int32_t)sizeof(key), &idx)) {
        if (g->n == g->cap) { g->cap = g->cap ? g->cap * 2 : 16; g->groups = (OGroup *)realloc(g->groups, sizeof(OGroup) * g->cap); }
        idx = g->n++;
        g->groups[idx].key = key;
        i32vec_init(&g->groups[idx].members);
        char *owned = arena_strndup(&g->arena, (const char *)&key, (int32_t)sizeof(key));
        strmap_put(&g->key_to_group, owned, (int32_t)sizeof(key), idx);
    }
    i32vec_push(&g->groups[idx].members, val);
}

void bm_init(BridgeMatrix *m) { memset(m, 0, sizeof(*m)); }
void bm_free(BridgeMatrix *m) {
    free(m->source); free(m->target); free(m->bridge); free(m->cluster_id); free(m->index);
    free(m->tindex_offsets); free(m->tindex_values);
    free(m->cluster_members_flat); free(m->cluster_members_offset);
    memset(m, 0, sizeof(*m));
}

void bm_build(BridgeMatrix *m, const i32vec *sequences, int32_t n_seq, int32_t vocab_size) {
    /* dedup triples (source,target,bridge), first-seen order */
    StrMap seen; strmap_init(&seen);
    Arena arena; arena_init(&arena);
    i32vec os, ot, ob; i32vec_init(&os); i32vec_init(&ot); i32vec_init(&ob);
    for (int32_t s = 0; s < n_seq; s++) {
        const i32vec *seq = &sequences[s];
        for (size_t i = 0; i + 2 < seq->len; i++) {
            int32_t source = seq->data[i], bridge = seq->data[i + 1], target = seq->data[i + 2];
            struct { int32_t s, t, b; } k = { source, target, bridge };
            int32_t idx;
            if (!strmap_get(&seen, (const char *)&k, (int32_t)sizeof(k), &idx)) {
                idx = (int32_t)os.len;
                i32vec_push(&os, source); i32vec_push(&ot, target); i32vec_push(&ob, bridge);
                char *owned = arena_strndup(&arena, (const char *)&k, (int32_t)sizeof(k));
                strmap_put(&seen, owned, (int32_t)sizeof(k), idx);
            }
        }
    }
    strmap_free(&seen);
    arena_free(&arena);

    bm_build_from_triples(m, &os, &ot, &ob, vocab_size);
    i32vec_free(&os); i32vec_free(&ot); i32vec_free(&ob);
}

/* Builds a BridgeMatrix from an ALREADY-DEDUPED (but not yet sorted or
 * clustered) triple list -- the part of bm_build() that has nothing to
 * do with where the triples came from. Split out so model.c's
 * incremental-merge path (union old-matrix triples + new-sequence
 * triples, both already distinct, then re-cluster from scratch over
 * the union) can reuse the exact same sort/cluster/index/t_index logic
 * instead of a second, riskier copy of it. */
void bm_build_from_triples(BridgeMatrix *m, const i32vec *src, const i32vec *tgt, const i32vec *brg, int32_t vocab_size) {
    bm_free(m);
    m->vocab_size = vocab_size;

    int32_t n = (int32_t)src->len;
    int32_t *perm = stable_sort_perm(src->data, n); /* stable sort by source */
    int32_t *tsource = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n ? n : 1));
    int32_t *ttarget = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n ? n : 1));
    int32_t *tbridge = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n ? n : 1));
    for (int32_t i = 0; i < n; i++) {
        int32_t p = perm[i];
        tsource[i] = src->data[p]; ttarget[i] = tgt->data[p]; tbridge[i] = brg->data[p];
    }
    free(perm);

    int32_t *cluster_id = (int32_t *)calloc((size_t)(n > 0 ? n : 1), sizeof(int32_t));

    /* group by (source,target) and (source,bridge), in the order the
     * now-sorted `triples` array is scanned (matches Python's
     * `for idx,(s,t,b) in enumerate(triples): ...`). */
    OrderedGroups gst, gsb; ogroups_init(&gst); ogroups_init(&gsb);
    for (int32_t i = 0; i < n; i++) {
        ogroups_add(&gst, pack2(tsource[i], ttarget[i]), i);
        ogroups_add(&gsb, pack2(tsource[i], tbridge[i]), i);
    }

    int32_t next_cluster = 1;
    for (int32_t g = 0; g < gst.n; g++) {
        i32vec *mem = &gst.groups[g].members;
        if (mem->len > 1) {
            for (size_t k = 0; k < mem->len; k++) cluster_id[mem->data[k]] = next_cluster;
            next_cluster++;
        }
    }
    for (int32_t g = 0; g < gsb.n; g++) {
        i32vec *mem = &gsb.groups[g].members;
        if (mem->len > 1) {
            int all_zero = 1;
            for (size_t k = 0; k < mem->len; k++) if (cluster_id[mem->data[k]] != 0) { all_zero = 0; break; }
            if (all_zero) {
                for (size_t k = 0; k < mem->len; k++) cluster_id[mem->data[k]] = next_cluster;
                next_cluster++;
            }
        }
    }
    ogroups_free(&gst); ogroups_free(&gsb);

    m->source = tsource; m->target = ttarget; m->bridge = tbridge; m->cluster_id = cluster_id;
    m->n = n;
    m->max_cluster_id = next_cluster - 1;

    m->index = (int32_t *)calloc((size_t)vocab_size + 1, sizeof(int32_t));
    for (int32_t i = 0; i < n; i++) m->index[m->source[i] + 1]++;
    for (int32_t i = 1; i <= vocab_size; i++) m->index[i] += m->index[i - 1];

    /* t_index: token -> sorted distinct nonzero cluster ids (as bridge or target) */
    i32vec *sets = (i32vec *)calloc((size_t)vocab_size, sizeof(i32vec));
    for (int32_t i = 0; i < vocab_size; i++) i32vec_init(&sets[i]);
    for (int32_t i = 0; i < n; i++) {
        int32_t c = m->cluster_id[i];
        if (c == 0) continue;
        int32_t b = m->bridge[i], t = m->target[i];
        i32vec *sb = &sets[b];
        int found = 0; for (size_t k = 0; k < sb->len; k++) if (sb->data[k] == c) { found = 1; break; }
        if (!found) i32vec_push(sb, c);
        i32vec *st = &sets[t];
        found = 0; for (size_t k = 0; k < st->len; k++) if (st->data[k] == c) { found = 1; break; }
        if (!found) i32vec_push(st, c);
    }
    m->tindex_offsets = (int32_t *)calloc((size_t)vocab_size + 1, sizeof(int32_t));
    int32_t total = 0;
    for (int32_t i = 0; i < vocab_size; i++) {
        /* insertion sort: these per-token cluster-id lists are small */
        int32_t *d = sets[i].data; size_t ln = sets[i].len;
        for (size_t a = 1; a < ln; a++) { int32_t v = d[a]; size_t b = a; while (b > 0 && d[b-1] > v) { d[b] = d[b-1]; b--; } d[b] = v; }
        total += (int32_t)sets[i].len;
    }
    m->tindex_values = (int32_t *)malloc(sizeof(int32_t) * (size_t)(total ? total : 1));
    int32_t off = 0;
    for (int32_t i = 0; i < vocab_size; i++) {
        m->tindex_offsets[i] = off;
        if (sets[i].len) memcpy(m->tindex_values + off, sets[i].data, sizeof(int32_t) * sets[i].len);
        off += (int32_t)sets[i].len;
        i32vec_free(&sets[i]);
    }
    m->tindex_offsets[vocab_size] = off;
    free(sets);
}

void bm_triples_from_source(const BridgeMatrix *m, int32_t source, i32vec *out_target, i32vec *out_bridge, i32vec *out_cluster) {
    if (source < 0 || source + 1 >= m->vocab_size + 1 || !m->index) return;
    int32_t start = m->index[source], end = m->index[source + 1];
    for (int32_t i = start; i < end; i++) {
        i32vec_push(out_target, m->target[i]);
        i32vec_push(out_bridge, m->bridge[i]);
        i32vec_push(out_cluster, m->cluster_id[i]);
    }
}

static void bm_ensure_cluster_index(BridgeMatrix *m) {
    if (m->cluster_members_offset) return;
    int32_t maxc = m->max_cluster_id;
    i32vec *lists = (i32vec *)calloc((size_t)(maxc + 1), sizeof(i32vec));
    for (int32_t i = 0; i <= maxc; i++) i32vec_init(&lists[i]);
    for (int32_t i = 0; i < m->n; i++) {
        int32_t c = m->cluster_id[i];
        if (c) i32vec_push(&lists[c], i);
    }
    m->cluster_members_offset = (int32_t *)malloc(sizeof(int32_t) * (size_t)(maxc + 2));
    int32_t total = 0;
    for (int32_t c = 0; c <= maxc; c++) total += (int32_t)lists[c].len;
    m->cluster_members_flat = (int32_t *)malloc(sizeof(int32_t) * (size_t)(total ? total : 1));
    int32_t off = 0;
    for (int32_t c = 0; c <= maxc; c++) {
        m->cluster_members_offset[c] = off;
        if (lists[c].len) memcpy(m->cluster_members_flat + off, lists[c].data, sizeof(int32_t) * lists[c].len);
        off += (int32_t)lists[c].len;
        i32vec_free(&lists[c]);
    }
    m->cluster_members_offset[maxc + 1] = off;
    free(lists);
}

BmAxis bm_cluster_axis(BridgeMatrix *m, int32_t cluster_id, i32vec *out_source, i32vec *out_target, i32vec *out_bridge) {
    bm_ensure_cluster_index(m);
    if (cluster_id < 0 || cluster_id > m->max_cluster_id) return BM_AXIS_NONE;
    int32_t start = m->cluster_members_offset[cluster_id], end = m->cluster_members_offset[cluster_id + 1];
    int32_t cnt = end - start;
    for (int32_t i = start; i < end; i++) {
        int32_t ti = m->cluster_members_flat[i];
        i32vec_push(out_source, m->source[ti]);
        i32vec_push(out_target, m->target[ti]);
        i32vec_push(out_bridge, m->bridge[ti]);
    }
    if (cnt < 2) return BM_AXIS_NONE;
    int32_t t0 = out_target->data[0], b0 = out_bridge->data[0];
    int all_t = 1, all_b = 1;
    for (int32_t i = 0; i < cnt; i++) {
        if (out_target->data[i] != t0) all_t = 0;
        if (out_bridge->data[i] != b0) all_b = 0;
    }
    if (all_t) return BM_AXIS_BRIDGE;   /* source+target fixed, bridge varies */
    if (all_b) return BM_AXIS_TARGET;   /* source+bridge fixed, target varies */
    return BM_AXIS_NONE;
}

void bm_tindex_get(const BridgeMatrix *m, int32_t token, i32vec *out) {
    if (token < 0 || token >= m->vocab_size || !m->tindex_offsets) return;
    int32_t start = m->tindex_offsets[token], end = m->tindex_offsets[token + 1];
    for (int32_t i = start; i < end; i++) i32vec_push(out, m->tindex_values[i]);
}

BmAxis bm_cluster_axis_of(BridgeMatrix *m, int32_t cluster_id) {
    bm_ensure_cluster_index(m);
    if (cluster_id < 0 || cluster_id > m->max_cluster_id) return BM_AXIS_NONE;
    int32_t start = m->cluster_members_offset[cluster_id], end = m->cluster_members_offset[cluster_id + 1];
    if (end - start < 2) return BM_AXIS_NONE;
    int32_t t0i = m->cluster_members_flat[start];
    int32_t t0 = m->target[t0i], b0 = m->bridge[t0i];
    int all_t = 1, all_b = 1;
    for (int32_t i = start; i < end; i++) {
        int32_t ti = m->cluster_members_flat[i];
        if (m->target[ti] != t0) all_t = 0;
        if (m->bridge[ti] != b0) all_b = 0;
    }
    if (all_t) return BM_AXIS_BRIDGE;
    if (all_b) return BM_AXIS_TARGET;
    return BM_AXIS_NONE;
}

int bm_has_triple(const BridgeMatrix *m, int32_t source, int32_t bridge_tok, int32_t target) {
    if (source < 0 || source + 1 >= m->vocab_size + 1 || !m->index) return 0;
    int32_t start = m->index[source], end = m->index[source + 1];
    for (int32_t i = start; i < end; i++)
        if (m->bridge[i] == bridge_tok && m->target[i] == target) return 1;
    return 0;
}

int bm_save(const BridgeMatrix *m, const char *path) {
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    int32_t tindex_len = m->tindex_offsets ? m->tindex_offsets[m->vocab_size] : 0;
    MseBridgeHeader h = { MSE_BRIDGE_MAGIC, 1, m->vocab_size, m->n, m->vocab_size + 1, tindex_len };
    fwrite(&h, sizeof(h), 1, f);
    fwrite(m->source, sizeof(int32_t), (size_t)m->n, f);
    fwrite(m->target, sizeof(int32_t), (size_t)m->n, f);
    fwrite(m->bridge, sizeof(int32_t), (size_t)m->n, f);
    fwrite(m->cluster_id, sizeof(int32_t), (size_t)m->n, f);
    fwrite(m->index, sizeof(int32_t), (size_t)h.index_len, f);
    fwrite(m->tindex_offsets, sizeof(int32_t), (size_t)m->vocab_size + 1, f);
    fwrite(m->tindex_values, sizeof(int32_t), (size_t)tindex_len, f);
    fwrite(&m->max_cluster_id, sizeof(int32_t), 1, f);
    fclose(f); return 0;
}

int bm_load(BridgeMatrix *m, const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    MseBridgeHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1 || h.magic != MSE_BRIDGE_MAGIC) { fclose(f); return -1; }
    bm_init(m); /* see em_load's comment: must not assume `m` is valid */
    m->vocab_size = h.vocab_size; m->n = h.n_triples;
    size_t nn = (size_t)(m->n ? m->n : 1);
    m->source = (int32_t *)malloc(sizeof(int32_t) * nn);
    m->target = (int32_t *)malloc(sizeof(int32_t) * nn);
    m->bridge = (int32_t *)malloc(sizeof(int32_t) * nn);
    m->cluster_id = (int32_t *)malloc(sizeof(int32_t) * nn);
    m->index = (int32_t *)malloc(sizeof(int32_t) * (size_t)h.index_len);
    m->tindex_offsets = (int32_t *)malloc(sizeof(int32_t) * (size_t)(m->vocab_size + 1));
    m->tindex_values = (int32_t *)malloc(sizeof(int32_t) * (size_t)(h.n_tindex_entries ? h.n_tindex_entries : 1));
    int ok = 1;
    if (m->n > 0) {
        ok &= fread(m->source, sizeof(int32_t), (size_t)m->n, f) == (size_t)m->n;
        ok &= fread(m->target, sizeof(int32_t), (size_t)m->n, f) == (size_t)m->n;
        ok &= fread(m->bridge, sizeof(int32_t), (size_t)m->n, f) == (size_t)m->n;
        ok &= fread(m->cluster_id, sizeof(int32_t), (size_t)m->n, f) == (size_t)m->n;
    }
    ok &= fread(m->index, sizeof(int32_t), (size_t)h.index_len, f) == (size_t)h.index_len;
    ok &= fread(m->tindex_offsets, sizeof(int32_t), (size_t)(m->vocab_size + 1), f) == (size_t)(m->vocab_size + 1);
    if (h.n_tindex_entries > 0)
        ok &= fread(m->tindex_values, sizeof(int32_t), (size_t)h.n_tindex_entries, f) == (size_t)h.n_tindex_entries;
    ok &= fread(&m->max_cluster_id, sizeof(int32_t), 1, f) == 1;
    fclose(f);
    return ok ? 0 : -1;
}

/* ===================================================== RelationshipMatrix */
void rm_init(RelationshipMatrix *m) { memset(m, 0, sizeof(*m)); }
void rm_free(RelationshipMatrix *m) {
    free(m->r_triple); free(m->r_rel); free(m->index); free(m->rel_count);
    free(m->by_triple_rel); free(m->by_triple_offset);
    memset(m, 0, sizeof(*m));
}

void rm_build(RelationshipMatrix *m, const i32vec *sequences, int32_t n_seq, const BridgeMatrix *bridge) {
    rm_free(m);

    /* triple (source,target,bridge) -> triple_id, from the ALREADY
     * built (deduped/sorted/clustered) BridgeMatrix rows. */
    StrMap triple_to_id; strmap_init(&triple_to_id);
    Arena tarena; arena_init(&tarena);
    for (int32_t i = 0; i < bridge->n; i++) {
        struct { int32_t s, t, b; } k = { bridge->source[i], bridge->target[i], bridge->bridge[i] };
        char *owned = arena_strndup(&tarena, (const char *)&k, (int32_t)sizeof(k));
        strmap_put(&triple_to_id, owned, (int32_t)sizeof(k), i);
    }

    /* dedup sequences by exact content, first-seen order */
    StrMap seq_seen; strmap_init(&seq_seen);
    Arena seq_arena; arena_init(&seq_arena);
    i32vec *unique_seqs = NULL; int32_t n_unique = 0, cap_unique = 0;
    i32vec counts; i32vec_init(&counts);

    for (int32_t s = 0; s < n_seq; s++) {
        const i32vec *seq = &sequences[s];
        int32_t keylen = (int32_t)(seq->len * sizeof(int32_t));
        int32_t rel_id;
        if (strmap_get(&seq_seen, (const char *)seq->data, keylen, &rel_id)) {
            counts.data[rel_id]++;
        } else {
            rel_id = n_unique;
            if (n_unique == cap_unique) { cap_unique = cap_unique ? cap_unique * 2 : 16; unique_seqs = (i32vec *)realloc(unique_seqs, sizeof(i32vec) * cap_unique); }
            i32vec_init(&unique_seqs[n_unique]);
            for (size_t i = 0; i < seq->len; i++) i32vec_push(&unique_seqs[n_unique], seq->data[i]);
            n_unique++;
            i32vec_push(&counts, 1);
            char *owned = arena_strndup(&seq_arena, (const char *)seq->data, keylen);
            strmap_put(&seq_seen, owned, keylen, rel_id);
        }
    }

    i32vec rrows_triple, rrows_rel;
    i32vec_init(&rrows_triple); i32vec_init(&rrows_rel);
    for (int32_t rel_id = 0; rel_id < n_unique; rel_id++) {
        i32vec *seq = &unique_seqs[rel_id];
        for (size_t i = 0; i + 2 < seq->len; i++) {
            struct { int32_t s, t, b; } k = { seq->data[i], seq->data[i + 2], seq->data[i + 1] };
            int32_t triple_id;
            if (strmap_get(&triple_to_id, (const char *)&k, (int32_t)sizeof(k), &triple_id)) {
                i32vec_push(&rrows_triple, triple_id);
                i32vec_push(&rrows_rel, rel_id);
            }
        }
    }
    /* stable sort rows by rel_id */
    int32_t n_rows = (int32_t)rrows_rel.len;
    int32_t *perm = stable_sort_perm(rrows_rel.data, n_rows);
    m->r_triple = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n_rows ? n_rows : 1));
    m->r_rel = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n_rows ? n_rows : 1));
    for (int32_t i = 0; i < n_rows; i++) {
        int32_t p = perm[i];
        m->r_triple[i] = rrows_triple.data[p];
        m->r_rel[i] = rrows_rel.data[p];
    }
    free(perm);
    m->n_rows = n_rows;
    m->n_rels = n_unique;
    m->rel_count = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n_unique ? n_unique : 1));
    if (n_unique) memcpy(m->rel_count, counts.data, sizeof(int32_t) * (size_t)n_unique);

    m->index = (int32_t *)calloc((size_t)n_unique + 1, sizeof(int32_t));
    for (int32_t i = 0; i < n_rows; i++) m->index[m->r_rel[i] + 1]++;
    for (int32_t i = 1; i <= n_unique; i++) m->index[i] += m->index[i - 1];

    for (int32_t i = 0; i < n_unique; i++) i32vec_free(&unique_seqs[i]);
    free(unique_seqs);
    i32vec_free(&counts);
    i32vec_free(&rrows_triple); i32vec_free(&rrows_rel);
    strmap_free(&seq_seen); arena_free(&seq_arena);
    strmap_free(&triple_to_id); arena_free(&tarena);
}

int32_t rm_count(const RelationshipMatrix *m, int32_t rel_id) {
    if (rel_id < 0 || rel_id >= m->n_rels) return 0;
    return m->rel_count[rel_id];
}

void rm_triples_for_relationship(const RelationshipMatrix *m, int32_t rel_id, i32vec *out) {
    if (rel_id < 0 || rel_id + 1 >= m->n_rels + 1 || !m->index) return;
    int32_t start = m->index[rel_id], end = m->index[rel_id + 1];
    for (int32_t i = start; i < end; i++) i32vec_push(out, m->r_triple[i]);
}

static void rm_ensure_triple_index(RelationshipMatrix *m) {
    if (m->by_triple_offset) return;
    int32_t max_triple = -1;
    for (int32_t i = 0; i < m->n_rows; i++) if (m->r_triple[i] > max_triple) max_triple = m->r_triple[i];
    int32_t n_triples = max_triple + 1;
    m->by_triple_n = n_triples;
    m->by_triple_offset = (int32_t *)calloc((size_t)n_triples + 1, sizeof(int32_t));
    for (int32_t i = 0; i < m->n_rows; i++) m->by_triple_offset[m->r_triple[i] + 1]++;
    for (int32_t i = 1; i <= n_triples; i++) m->by_triple_offset[i] += m->by_triple_offset[i - 1];

    m->by_triple_rel = (int32_t *)malloc(sizeof(int32_t) * (size_t)(m->n_rows ? m->n_rows : 1));
    int32_t *cursor = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n_triples + 1));
    memcpy(cursor, m->by_triple_offset, sizeof(int32_t) * (size_t)(n_triples + 1));
    /* stable, sorted-by-triple-id fill (Python sorts row indices by triple_id) */
    int32_t *perm = stable_sort_perm(m->r_triple, m->n_rows);
    for (int32_t i = 0; i < m->n_rows; i++) {
        int32_t row = perm[i];
        int32_t tid = m->r_triple[row];
        m->by_triple_rel[cursor[tid]++] = m->r_rel[row];
    }
    free(perm);
    free(cursor);
}

void rm_relationships_for_triple(RelationshipMatrix *m, int32_t triple_id, i32vec *out) {
    rm_ensure_triple_index(m);
    if (triple_id < 0 || triple_id + 1 >= m->by_triple_n + 1) return;
    int32_t start = m->by_triple_offset[triple_id], end = m->by_triple_offset[triple_id + 1];
    for (int32_t i = start; i < end; i++) i32vec_push(out, m->by_triple_rel[i]);
}

int rm_save(const RelationshipMatrix *m, const char *path) {
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    MseRelHeader h = { MSE_REL_MAGIC, 1, m->n_rels, m->n_rows, m->n_rels + 1 };
    fwrite(&h, sizeof(h), 1, f);
    fwrite(m->r_triple, sizeof(int32_t), (size_t)m->n_rows, f);
    fwrite(m->r_rel, sizeof(int32_t), (size_t)m->n_rows, f);
    fwrite(m->index, sizeof(int32_t), (size_t)h.index_len, f);
    fwrite(m->rel_count, sizeof(int32_t), (size_t)m->n_rels, f);
    fclose(f); return 0;
}

int rm_load(RelationshipMatrix *m, const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    MseRelHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1 || h.magic != MSE_REL_MAGIC) { fclose(f); return -1; }
    rm_init(m); /* see em_load's comment: must not assume `m` is valid */
    m->n_rels = h.n_rels; m->n_rows = h.n_rows;
    m->r_triple = (int32_t *)malloc(sizeof(int32_t) * (size_t)(m->n_rows ? m->n_rows : 1));
    m->r_rel = (int32_t *)malloc(sizeof(int32_t) * (size_t)(m->n_rows ? m->n_rows : 1));
    m->index = (int32_t *)malloc(sizeof(int32_t) * (size_t)h.index_len);
    m->rel_count = (int32_t *)malloc(sizeof(int32_t) * (size_t)(m->n_rels ? m->n_rels : 1));
    int ok = 1;
    if (m->n_rows > 0) {
        ok &= fread(m->r_triple, sizeof(int32_t), (size_t)m->n_rows, f) == (size_t)m->n_rows;
        ok &= fread(m->r_rel, sizeof(int32_t), (size_t)m->n_rows, f) == (size_t)m->n_rows;
    }
    ok &= fread(m->index, sizeof(int32_t), (size_t)h.index_len, f) == (size_t)h.index_len;
    if (m->n_rels > 0)
        ok &= fread(m->rel_count, sizeof(int32_t), (size_t)m->n_rels, f) == (size_t)m->n_rels;
    fclose(f);
    return ok ? 0 : -1;
}
