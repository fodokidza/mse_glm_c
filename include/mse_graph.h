/* mse_graph.h — Edge Matrix, Bridge Matrix, Relationship Matrix.
 * C port of graph.py. Already array-backed/CSR in the Python original
 * (array('i')); this port keeps that shape but replaces every Python
 * dict/set/defaultdict used during build() with an open-addressing
 * hash map or a plain sort, and drops the JSON to_dict()/from_dict()
 * pair in favor of the flat binary format in mse_format.h.
 */
#ifndef MSE_GRAPH_H
#define MSE_GRAPH_H

#include "mse_util.h"

/* ------------------------------------------------------------ EdgeMatrix */
typedef struct {
    int32_t *src, *dst, *count;   /* deduplicated (src,dst) pairs, count = bigram frequency */
    int32_t  n;                   /* number of unique pairs */
    int32_t *index;                /* CSR offsets, size vocab_size+1 */
    int32_t  vocab_size;
} EdgeMatrix;

/* em_init/em_build/em_load all populate a struct assumed to be
 * zero-initialized garbage-free memory (either freshly declared, or
 * already em_free()'d by the caller) — none of them free prior
 * contents first. Call em_free() yourself before reusing a struct
 * that already holds a built/loaded matrix, exactly as you would
 * before any other re-init in C. */
void em_init(EdgeMatrix *m);
void em_free(EdgeMatrix *m);
/* sequences: array of `n_seq` token-id sequences (i32vec each). */
void em_build(EdgeMatrix *m, const i32vec *sequences, int32_t n_seq, int32_t vocab_size);
/* Same construction from already-deduplicated (src,dst,count) arrays
 * -- see graph.c's comment. Used directly by model.c's incremental merge. */
void em_build_from_pairs(EdgeMatrix *m, const i32vec *src, const i32vec *dst, const i32vec *count, int32_t vocab_size);
/* Writes successors of `token` into `out` (caller-inited i32vec), sorted ascending by dst. */
void em_successors(const EdgeMatrix *m, int32_t token, i32vec *out);
int32_t em_frequency(const EdgeMatrix *m, int32_t token, int32_t candidate);
int em_save(const EdgeMatrix *m, const char *path);
int em_load(EdgeMatrix *m, const char *path);

/* ---------------------------------------------------------- BridgeMatrix */
typedef struct {
    int32_t *source, *target, *bridge, *cluster_id; /* deduplicated triples */
    int32_t  n;
    int32_t *index;         /* CSR offsets on `source`, size vocab_size+1 */
    int32_t  vocab_size;

    /* token -> sorted distinct nonzero cluster ids it participates in
     * (as either bridge or target) — flattened CSR alongside a map
     * from token to its row, since tokens are a dense 0..vocab_size-1
     * range. */
    int32_t *tindex_offsets; /* size vocab_size+1 */
    int32_t *tindex_values;  /* size tindex_offsets[vocab_size] */

    /* lazily built: cluster_id -> member triple indices */
    int32_t *cluster_members_flat;   /* NULL until built */
    int32_t *cluster_members_offset; /* size max_cluster_id+2, NULL until built */
    int32_t  max_cluster_id;
} BridgeMatrix;

typedef enum { BM_AXIS_NONE = 0, BM_AXIS_BRIDGE, BM_AXIS_TARGET } BmAxis;

void bm_init(BridgeMatrix *m);
void bm_free(BridgeMatrix *m);
void bm_build(BridgeMatrix *m, const i32vec *sequences, int32_t n_seq, int32_t vocab_size);
/* Same construction (sort by source, cluster, CSR, t_index) but from
 * an already-deduplicated triple list rather than raw sequences --
 * see graph.c's comment. Used directly by model.c's incremental merge. */
void bm_build_from_triples(BridgeMatrix *m, const i32vec *src, const i32vec *tgt, const i32vec *brg, int32_t vocab_size);
/* triples_from_source: appends (target,bridge,cluster) triples for `source` to out arrays (parallel, same length). */
void bm_triples_from_source(const BridgeMatrix *m, int32_t source, i32vec *out_target, i32vec *out_bridge, i32vec *out_cluster);
/* cluster_axis: returns axis and fills out_source/out_target/out_bridge (parallel) with member triples. */
BmAxis bm_cluster_axis(BridgeMatrix *m, int32_t cluster_id, i32vec *out_source, i32vec *out_target, i32vec *out_bridge);
/* Cheap axis-only query -- same result as bm_cluster_axis's return
 * value but without populating member-triple output vectors. Used on
 * hot paths (importance.c's per-triple _trigger_for_triple scan, IVM's
 * important_member_tokens()) that only need the axis, not the member
 * list itself. */
BmAxis bm_cluster_axis_of(BridgeMatrix *m, int32_t cluster_id);
/* Exact-triple membership test: was (source,bridge_tok,target) ever a
 * literal trained Bridge Matrix row? O(row length for `source`), not
 * O(n). Used by IVM's V8 (triple witness) and mirrors the same check
 * Strict Mode's Stage 2 already performs via _exact_triple_rels. */
int bm_has_triple(const BridgeMatrix *m, int32_t source, int32_t bridge_tok, int32_t target);
void bm_tindex_get(const BridgeMatrix *m, int32_t token, i32vec *out /* sorted cluster ids */);
int bm_save(const BridgeMatrix *m, const char *path);
int bm_load(BridgeMatrix *m, const char *path);

/* ----------------------------------------------------- RelationshipMatrix */
typedef struct {
    int32_t *r_triple, *r_rel;  /* rows, sorted by rel */
    int32_t  n_rows;
    int32_t *index;             /* CSR offsets on rel, size n_rels+1 */
    int32_t  n_rels;
    int32_t *rel_count;         /* size n_rels: literal training occurrences per unique sentence */

    /* lazily built secondary CSR keyed on triple_id */
    int32_t *by_triple_rel;     /* NULL until built */
    int32_t *by_triple_offset;  /* NULL until built, size n_triples+1 */
    int32_t  by_triple_n;
} RelationshipMatrix;

void rm_init(RelationshipMatrix *m);
void rm_free(RelationshipMatrix *m);
void rm_build(RelationshipMatrix *m, const i32vec *sequences, int32_t n_seq, const BridgeMatrix *bridge);
int32_t rm_count(const RelationshipMatrix *m, int32_t rel_id);
void rm_triples_for_relationship(const RelationshipMatrix *m, int32_t rel_id, i32vec *out);
void rm_relationships_for_triple(RelationshipMatrix *m, int32_t triple_id, i32vec *out);
int rm_save(const RelationshipMatrix *m, const char *path);
int rm_load(RelationshipMatrix *m, const char *path);

#endif /* MSE_GRAPH_H */
