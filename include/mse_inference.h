/* mse_inference.h — C port of inference.py.
 *
 * active_rels (a set of relationship ids the generation path has been
 * consistent with so far) is a real set in Python; here it's a plain
 * deduplicated i32vec (RelSet). Real per-step relationship-id sets are
 * small (a handful of matching training sentences), so the O(n*m)
 * linear-scan intersection this uses is simpler to get right than
 * sorted-array merge machinery and costs nothing measurable at this
 * scale — a legitimate "don't over-engineer it" call, not an
 * oversight; see relset_intersect in inference.c if this ever needs
 * to scale to much larger active_rels sets.
 *
 * The full per-step trace dict (candidates list, per-layer vote
 * breakdowns) is deferred just as ivm.c's is — only what generate()
 * itself actually needs to keep going (chosen token, new active_rels,
 * a stage/rule label for auditability) is threaded through StepTrace.
 */
#ifndef MSE_INFERENCE_H
#define MSE_INFERENCE_H

#include "mse_graph.h"
#include "mse_ctm.h"
#include "mse_ivm.h"
#include "mse_util.h"

typedef i32vec RelSet; /* deduplicated, unordered */

typedef enum { IE_MODE_STRICT = 0, IE_MODE_OPEN = 1 } IeMode;

typedef struct {
    const EdgeMatrix   *edges;
    BridgeMatrix       *bridges;
    RelationshipMatrix *rels;
    IeMode  mode;
    int32_t *vocab;   /* Open Mode candidate pool (borrowed, not owned) */
    int32_t  n_vocab;
} InferenceEngine;

typedef struct {
    int32_t chosen;
    int32_t stage;
    const char *rule;   /* static string literal */
    RelSet  active_rels; /* owned by caller after the call; free with i32vec_free */
} StepTrace;

void ie_init(InferenceEngine *ie, const EdgeMatrix *edges, BridgeMatrix *bridges,
             RelationshipMatrix *rels, IeMode mode, int32_t *vocab, int32_t n_vocab);

/* previous/current/target use MSE_NONE (-1) as the "None" sentinel,
 * same convention as ivm.h. active_rels_in may be NULL/empty (start of
 * generation). ctm/ivm are optional (pass NULL to skip). context_tokens
 * should be deduplicated (a set, not a raw token sequence) — same
 * expectation ivm.c's functions have, since that's what every real
 * caller (ie_generate below) actually passes. */
int32_t ie_step(InferenceEngine *ie, int32_t previous, int32_t current,
                 const RelSet *active_rels_in,
                 const int32_t *context_tokens, int32_t n_context,
                 ContextTriggerMatrix *ctm, ImportanceVoteMatrix *ivm,
                 StepTrace *out_trace);

/* Generates up to max_tokens ids, appended to `ids` (caller-inited
 * i32vec, pre-loaded with the prompt). ctm/ivm optional (NULL to
 * skip). Returns 0 normally; the sequence includes every generated
 * token but NOT a trailing EOS (matches generate()'s `break` before
 * appending). */
void ie_generate(InferenceEngine *ie, i32vec *ids, int32_t max_tokens,
                  ContextTriggerMatrix *ctm, ImportanceVoteMatrix *ivm);

#endif /* MSE_INFERENCE_H */
