/* mse_ctm.h — C port of ctm.py: the shared token->relationships index
 * (token_to_relationships) and ContextTriggerMatrix.
 *
 * token_to_relationships is stored here as CSR with each token's
 * relationship-id list kept SORTED and deduplicated — the Python
 * original uses a `set` per token; a sorted array gets the same
 * membership/intersection semantics via binary search / merge-walk
 * instead of hashing, and it's what ivm.c's heavy "|rels(t) & rels(c)|"
 * traffic (V1/V3/V4/V7/V9) is built on.
 */
#ifndef MSE_CTM_H
#define MSE_CTM_H

#include "mse_graph.h"
#include "mse_util.h"

typedef struct {
    int32_t *offsets; /* size vocab_size+1 */
    int32_t *values;  /* sorted, deduplicated relationship ids, per token */
    int32_t  vocab_size;
} TokenRels;

/* RESERVED = {PAD, UNK, BOS} (see mse_config.h's mse_is_reserved) —
 * tokens in RESERVED never get an entry here, matching ctm.py. */
void token_rels_build(TokenRels *tr, RelationshipMatrix *rels, const BridgeMatrix *bridges, int32_t vocab_size);
void token_rels_free(TokenRels *tr);
/* Row for `token`: *out_n = length, returns NULL (and *out_n = 0) if
 * the token has no relationships (matches Python's `.get(t)` -> None/empty). */
const int32_t *token_rels_row(const TokenRels *tr, int32_t token, int32_t *out_n);
int32_t token_rels_intersect_count(const TokenRels *tr, int32_t a, int32_t b);
int     token_rels_intersect_any(const TokenRels *tr, int32_t a, int32_t b);
/* Writes the sorted intersection of a's and b's rows into `out`. */
void    token_rels_intersect(const TokenRels *tr, int32_t a, int32_t b, i32vec *out);
/* True iff any element of `set_sorted` (length n, sorted ascending)
 * appears in token `t`'s row. */
int     token_rels_row_intersects(const TokenRels *tr, int32_t t, const int32_t *set_sorted, int32_t n);

/* ------------------------------------------------------- ContextTriggerMatrix */
typedef struct {
    int32_t cluster_id;
    BmAxis  axis;
    int32_t *members;      /* sorted distinct member token ids */
    int32_t  n_members;
    int32_t *sig_offsets;  /* size n_members+1 */
    int32_t *sig_trigger;  /* flat, per member: trigger token ids */
    int32_t *sig_support;  /* flat, per member: support counts, parallel to sig_trigger */
} CtmClusterSig;

typedef struct {
    CtmClusterSig *sigs;
    int32_t n_sigs;
    int32_t *cluster_to_idx; /* size max_cluster_id+1, -1 if no signature; NULL if max_cluster_id<0 */
    int32_t max_cluster_id;
} ContextTriggerMatrix;

void ctm_build(ContextTriggerMatrix *ctm, BridgeMatrix *bridges, RelationshipMatrix *rels,
                const TokenRels *token_rels, int32_t min_support);
void ctm_free(ContextTriggerMatrix *ctm);

/* {member: score} for cluster_id, appended as parallel arrays. Empty
 * (0 length) if cluster_id has no signature. */
void ctm_score_members(const ContextTriggerMatrix *ctm, int32_t cluster_id,
                        const int32_t *context_tokens, int32_t n_context,
                        i32vec *out_members, i32vec *out_scores);
/* Highest-scoring member(s). Returns 0 (nothing written) if no
 * signature or every score is 0. `out_top` receives the tied top
 * members (sorted ascending); returns the max score via *out_score. */
int ctm_select(const ContextTriggerMatrix *ctm, int32_t cluster_id,
                const int32_t *context_tokens, int32_t n_context,
                i32vec *out_top, int32_t *out_score);
/* Resolves a tie among `candidates` via a shared cluster. Returns the
 * winning token id, or -1 if no resolution (caller falls back to its
 * own tie-break). */
int32_t ctm_resolve_tie(const ContextTriggerMatrix *ctm, BridgeMatrix *bridges,
                         const int32_t *candidates, int32_t n_candidates,
                         const int32_t *context_tokens, int32_t n_context);

#endif /* MSE_CTM_H */
