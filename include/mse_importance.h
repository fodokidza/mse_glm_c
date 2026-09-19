/* mse_importance.h — C port of the core, non-analysis parts of
 * importance.py: sequence reconstruction and per-triple trigger
 * lookup. These two are load-bearing (ctm.c and ivm.c both call
 * them); importance.py's read-only reporting functions
 * (important_tokens_in_sequence, trigger_matrix, expected_importance)
 * are analysis-layer, not part of the generation/scoring path, and
 * are deferred to a later phase — see the chat thread's phase plan.
 */
#ifndef MSE_IMPORTANCE_H
#define MSE_IMPORTANCE_H

#include "mse_graph.h"
#include "mse_util.h"

/* Reconstructs the literal, ordered training sentence for one
 * relationship_id by chaining its triples (triple k's target is
 * triple k+1's one new token). Appends token ids to `out` (caller-
 * inited i32vec). Returns 0 on success, -1 if the chain doesn't hold
 * (Relationship/Bridge matrices out of sync — should never happen on
 * a model this codebase built itself). Appends nothing if rel_id has
 * no triples (e.g. a <3-token training sentence). */
int mse_sequence_for_relationship(RelationshipMatrix *rels, const BridgeMatrix *bridges,
                                   int32_t rel_id, i32vec *out);

/* (axis, trigger_a, trigger_b, important_token) for one clustered
 * triple, or BM_AXIS_NONE if the triple isn't clustered (cluster_id
 * == 0). For axis==BM_AXIS_BRIDGE: trigger=(source,target), important
 * token=bridge. For axis==BM_AXIS_TARGET: trigger=(source,bridge),
 * important token=target. */
BmAxis mse_trigger_for_triple(BridgeMatrix *bridges, int32_t triple_id,
                               int32_t *trigger_a, int32_t *trigger_b, int32_t *important_token);

#endif /* MSE_IMPORTANCE_H */
