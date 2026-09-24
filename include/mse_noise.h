/* mse_noise.h — C port of noise.py: the three-stage noise-cancellation
 * scoring that feeds ivm.c's V10 ("noise vote"), the tenth and last of
 * Open Mode's weighted-voting layers (see mse_ivm.h).
 *
 * For one candidate token C, against the whole corpus's relationship
 * structure:
 *   stage1 = n_rels - |relationships C itself sits in|
 *   stage2 = n_rels - |relationships that KNOW C, directly or through
 *            any token that has ever shared a sentence with C|
 *   stage3 = |every distinct token in the corpus| - |every token that
 *            has ever shared a sentence with C|
 *   average = vote_weight * (stage1 + stage2 + stage3) / 3
 *
 * Each stage answers "does this voter already know the candidate?"
 * and counts a "no" as evidence worth paying attention to — a common
 * word naturally lands near 0 on every stage since it already
 * co-occurs with almost everything; no stopword list needed.
 *
 * KEY SIMPLIFICATION (matches the Python original's CURRENT,
 * optimized noise.py — not the literal per-(anchor, candidate) formula
 * an earlier revision used): none of the three stages above depend on
 * which anchor/context token is asking. A candidate's average is a
 * property of the candidate ALONE, so it is computed once per
 * candidate and cached forever after, instead of once per (anchor,
 * candidate) PAIR — the earlier formula's per-pair vocabulary-sized
 * set construction is what made it O(vocab^2) in practice; this port
 * only ever implements the fast, anchor-independent form.
 *
 * ivm.c's V10 then sums this average over the context: for a
 * candidate C, V10 = IVM_NOISE_WEIGHT * average(C) * (how many context
 * tokens have C in their co-occurring row) — exactly IVM_CONTEXT_WEIGHT's
 * raw count (V3), reused rather than recomputed (see ivm.c).
 */
#ifndef MSE_NOISE_H
#define MSE_NOISE_H

#include "mse_graph.h"
#include "mse_ctm.h"

typedef struct {
    const RelationshipMatrix *rels;   /* borrowed */
    const TokenRels *token_rels;      /* borrowed -- reused, not recomputed */
    int32_t vocab_size;
    int32_t n_rels;
    double  vote_weight;

    /* reverse index: relationship_id -> sorted, deduplicated, non-
     * STRUCTURAL (see mse_config.h's mse_is_structural) token ids it
     * contains -- built once from mse_sequence_for_relationship(),
     * reused by every candidate's stage1/stage2/stage3 computation. */
    int32_t *rt_offsets;   /* size n_rels+1 */
    int32_t *rt_tokens;    /* flat */
    int32_t  vocab_all_size; /* |union of every rt_tokens row| -- stage 3's pool size */

    /* lazy per-candidate cache: NULL/0 until first queried, cached
     * forever after (mirrors noise.py's _stage_cache/_average_cache). */
    double  *avg_cache;     /* size vocab_size; meaningful only where avg_ok[c] */
    uint8_t *avg_ok;        /* size vocab_size; 1 once avg_cache[c] has been computed */
} NoiseIndex;

/* Builds the reverse index (relationship -> token set) and allocates
 * the (empty) per-candidate cache. Cheap: no per-candidate work
 * happens here — see noise_candidate_average()'s lazy-fill contract.
 * `bridges` is only read during this call (to reconstruct each
 * relationship's literal sequence via mse_sequence_for_relationship());
 * `rels`/`token_rels` must outlive `ni` (borrowed, not copied). */
void noise_index_build(NoiseIndex *ni, RelationshipMatrix *rels, const BridgeMatrix *bridges,
                        const TokenRels *token_rels, int32_t vocab_size, double vote_weight);
void noise_index_free(NoiseIndex *ni);

/* Anchor-independent averaged noise-cancellation score for one
 * candidate token. Returns 0 (and leaves *out unset) for a candidate
 * that can't be scored — PAD/UNK/BOS/EOS (mse_is_structural; WORD_BOUND
 * is deliberately still scorable, same as every other vote layer), or
 * a token that appears in no relationship at all — matching noise.py's
 * "None means zero, not an error" contract. Returns 1 and sets *out
 * otherwise. First call for a given candidate does O(|reach(c)|) work
 * (bounded by that token's own co-occurrence neighborhood, not the
 * whole vocabulary); every later call for the same candidate is a
 * cache hit. */
int noise_candidate_average(NoiseIndex *ni, int32_t cand, double *out);

#endif /* MSE_NOISE_H */
