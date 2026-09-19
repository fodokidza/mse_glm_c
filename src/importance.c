#include "mse_importance.h"

int mse_sequence_for_relationship(RelationshipMatrix *rels, const BridgeMatrix *bridges,
                                   int32_t rel_id, i32vec *out) {
    i32vec triple_ids; i32vec_init(&triple_ids);
    rm_triples_for_relationship(rels, rel_id, &triple_ids);
    if (triple_ids.len == 0) { i32vec_free(&triple_ids); return 0; }

    int32_t t0 = triple_ids.data[0];
    i32vec_push(out, bridges->source[t0]);
    i32vec_push(out, bridges->bridge[t0]);
    i32vec_push(out, bridges->target[t0]);
    for (size_t k = 1; k < triple_ids.len; k++) {
        int32_t tid = triple_ids.data[k];
        int32_t prev2 = out->data[out->len - 2], prev1 = out->data[out->len - 1];
        if (bridges->source[tid] != prev2 || bridges->bridge[tid] != prev1) {
            i32vec_free(&triple_ids);
            return -1; /* Relationship/Bridge matrices out of sync */
        }
        i32vec_push(out, bridges->target[tid]);
    }
    i32vec_free(&triple_ids);
    return 0;
}

BmAxis mse_trigger_for_triple(BridgeMatrix *bridges, int32_t triple_id,
                               int32_t *trigger_a, int32_t *trigger_b, int32_t *important_token) {
    int32_t cid = bridges->cluster_id[triple_id];
    if (cid == 0) return BM_AXIS_NONE;
    BmAxis axis = bm_cluster_axis_of(bridges, cid);
    int32_t s = bridges->source[triple_id], t = bridges->target[triple_id], br = bridges->bridge[triple_id];
    if (axis == BM_AXIS_BRIDGE) {
        *trigger_a = s; *trigger_b = t; *important_token = br;
    } else { /* BM_AXIS_TARGET (or NONE, in which case caller shouldn't have reached here since cid != 0 always has a determinable axis for a real cluster) */
        *trigger_a = s; *trigger_b = br; *important_token = t;
    }
    return axis;
}
