/* mse_ivm.h — C port of ivm.py: the 9-layer (V1-V9) Open Mode scoring
 * engine, plus the legacy resolve_tie() used by Strict Mode's optional
 * importance_votes add-on.
 *
 * Design departures from a literal translation:
 *   - "adjacent_pairs()" is never materialized as its own set: V6 and
 *     the bigram-frequency tie-break both query the EdgeMatrix
 *     directly (em_frequency/em_successors) — it already *is* exactly
 *     that deduplicated directed-bigram index, so building a second
 *     copy of the same information as a hash set would be pure
 *     duplication.
 *   - "literal_triples()" is never materialized either: V8's exact-
 *     triple check queries the BridgeMatrix's own CSR (bm_has_triple)
 *     instead of a separately-built hash set — same data, one fewer
 *     copy, one fewer thing that could drift out of sync.
 *   - co_occurrence_index() (the "gate") is a real structure here
 *     (CoOccurIndex, CSR of (token -> sorted (other,count) rows)),
 *     because V1/V3/V4/V9 genuinely need the aggregated per-pair
 *     shared-relationship COUNT, which the two structures above can't
 *     supply. It intentionally does NOT include Python's adjacency
 *     backfill (shared=0 placeholder entries) — that backfill exists
 *     only so a merged Python iteration loop still "sees" adjacency-
 *     only candidates; this port computes V6 from a separate,
 *     independent EdgeMatrix walk instead of piggybacking on the
 *     gate's iteration, so the backfill has nothing left to do here.
 *   - The sparse per-token score CACHE (build_cache/enable_cache in
 *     the Python original) is a pure, explicitly opt-in performance
 *     optimization that the Python docstring itself guarantees
 *     produces "identical numeric output" to the live path — it is
 *     deferred to a later pass; this port's ivm_score_candidates() IS
 *     the (semantically authoritative) live path.
 *   - The full audit trace (9 separate per-layer dicts, "knows",
 *     "influence", "important_tokens") is deferred too — only the
 *     final per-candidate scores (what select() actually decides on)
 *     are computed. Every vote layer's *arithmetic* is still ported
 *     exactly; only the extra bookkeeping for human inspection is not
 *     yet exposed.
 */
#ifndef MSE_IVM_H
#define MSE_IVM_H

#include "mse_graph.h"
#include "mse_ctm.h"
#include "mse_util.h"

typedef struct {
    int32_t *offsets;      /* size vocab_size+1 */
    int32_t *other_token;  /* flat, sorted ascending within each row */
    int32_t *count;        /* flat, parallel to other_token: shared relationship count (> 0) */
    int32_t  vocab_size;
} CoOccurIndex;

typedef struct {
    StrMap key_to_idx;     /* packed (source,bridge) bytes -> row index */
    Arena  arena;
    int32_t *offsets;      /* size n+1 */
    int32_t *rel_ids;      /* flat, sorted ascending per row */
    int32_t  n;
} BigramRelsIndex;

typedef struct {
    TokenRels     token_rels;      /* owned */
    uint8_t      *important;       /* owned, size vocab_size */
    CoOccurIndex  co_occurring;    /* owned */
    BigramRelsIndex bigram_rels;   /* owned */
    const EdgeMatrix   *edges;     /* borrowed */
    BridgeMatrix       *bridges;   /* borrowed */
    int32_t vocab_size;

    double important_weight, influence_weight, context_weight, context_influence_weight,
           bigram_witness_weight, adjacency_weight, prev_current_weight, triple_weight,
           whole_context_weight;
} ImportanceVoteMatrix;

/* current/previous use -1 as the "None" sentinel throughout (valid
 * token ids are always >= 0). */
#define MSE_NONE (-1)

void ivm_build(ImportanceVoteMatrix *ivm, const EdgeMatrix *edges, BridgeMatrix *bridges,
                RelationshipMatrix *rels, int32_t vocab_size);
void ivm_free(ImportanceVoteMatrix *ivm);

/* Legacy tie-break (resolve_tie): Σ influence(t) per important t that
 * knows a candidate. Returns the winner, or MSE_NONE if no resolution
 * (context empty, no important token knows anything, or a residual
 * tie) -- same "fall back to caller's tie-break" contract as Python. */
int32_t ivm_resolve_tie(const ImportanceVoteMatrix *ivm,
                         const int32_t *candidates, int32_t n_candidates,
                         const int32_t *context_tokens, int32_t n_context);

/* Primary Open Mode scoring: writes each candidate's V1..V9 summed
 * score into out_scores (caller-allocated, size n_candidates, parallel
 * to `candidates`). */
void ivm_score_candidates(const ImportanceVoteMatrix *ivm,
                           const int32_t *candidates, int32_t n_candidates,
                           const int32_t *context_tokens, int32_t n_context,
                           int32_t current, int32_t previous,
                           double *out_scores);

/* select(): score_candidates() + the 3-stage tie-break cascade
 * (bigram frequency -> global frequency -> lowest token id). Returns
 * MSE_NONE only when n_candidates == 0. */
int32_t ivm_select(const ImportanceVoteMatrix *ivm,
                    const int32_t *candidates, int32_t n_candidates,
                    const int32_t *context_tokens, int32_t n_context,
                    int32_t current, int32_t previous);

#endif /* MSE_IVM_H */
