#include "mse_ctm.h"
#include "mse_importance.h"
#include "mse_config.h"
#include <stdlib.h>
#include <string.h>

/* ==================================================== TokenRels (CSR) */
void token_rels_build(TokenRels *tr, RelationshipMatrix *rels, const BridgeMatrix *bridges, int32_t vocab_size) {
    tr->vocab_size = vocab_size;
    i32vec *buckets = (i32vec *)calloc((size_t)vocab_size, sizeof(i32vec));
    for (int32_t i = 0; i < vocab_size; i++) i32vec_init(&buckets[i]);

    StrMap seen; strmap_init(&seen);
    Arena arena; arena_init(&arena);
    i32vec rel_ids; i32vec_init(&rel_ids);

    for (int32_t tid = 0; tid < bridges->n; tid++) {
        rel_ids.len = 0;
        rm_relationships_for_triple((RelationshipMatrix *)rels, tid, &rel_ids);
        if (rel_ids.len == 0) continue;
        int32_t toks[3] = { bridges->source[tid], bridges->bridge[tid], bridges->target[tid] };
        for (int r = 0; r < 3; r++) {
            int32_t tok = toks[r];
            for (size_t i = 0; i < rel_ids.len; i++) {
                int32_t rid = rel_ids.data[i];
                struct { int32_t t, r; } key = { tok, rid };
                int32_t idx;
                if (!strmap_get(&seen, (const char *)&key, (int32_t)sizeof(key), &idx)) {
                    i32vec_push(&buckets[tok], rid);
                    char *owned = arena_strndup(&arena, (const char *)&key, (int32_t)sizeof(key));
                    strmap_put(&seen, owned, (int32_t)sizeof(key), 1);
                }
            }
        }
    }
    i32vec_free(&rel_ids);
    strmap_free(&seen); arena_free(&arena);

    tr->offsets = (int32_t *)calloc((size_t)vocab_size + 1, sizeof(int32_t));
    int32_t total = 0;
    for (int32_t t = 0; t < vocab_size; t++) {
        /* sort each (typically small) bucket ascending -- insertion sort */
        int32_t *d = buckets[t].data; size_t ln = buckets[t].len;
        for (size_t a = 1; a < ln; a++) { int32_t v = d[a]; size_t b = a; while (b > 0 && d[b-1] > v) { d[b] = d[b-1]; b--; } d[b] = v; }
        total += (int32_t)buckets[t].len;
    }
    tr->values = (int32_t *)malloc(sizeof(int32_t) * (size_t)(total ? total : 1));
    int32_t off = 0;
    for (int32_t t = 0; t < vocab_size; t++) {
        tr->offsets[t] = off;
        if (buckets[t].len) memcpy(tr->values + off, buckets[t].data, sizeof(int32_t) * buckets[t].len);
        off += (int32_t)buckets[t].len;
        i32vec_free(&buckets[t]);
    }
    tr->offsets[vocab_size] = off;
    free(buckets);
}

void token_rels_free(TokenRels *tr) {
    free(tr->offsets); free(tr->values);
    tr->offsets = NULL; tr->values = NULL; tr->vocab_size = 0;
}

const int32_t *token_rels_row(const TokenRels *tr, int32_t token, int32_t *out_n) {
    if (token < 0 || token >= tr->vocab_size || !tr->offsets) { *out_n = 0; return NULL; }
    int32_t start = tr->offsets[token], end = tr->offsets[token + 1];
    *out_n = end - start;
    return (*out_n > 0) ? (tr->values + start) : NULL;
}

int32_t token_rels_intersect_count(const TokenRels *tr, int32_t a, int32_t b) {
    int32_t na, nb;
    const int32_t *ra = token_rels_row(tr, a, &na);
    const int32_t *rb = token_rels_row(tr, b, &nb);
    if (!ra || !rb) return 0;
    int32_t i = 0, j = 0, count = 0;
    while (i < na && j < nb) {
        if (ra[i] == rb[j]) { count++; i++; j++; }
        else if (ra[i] < rb[j]) i++;
        else j++;
    }
    return count;
}

int token_rels_intersect_any(const TokenRels *tr, int32_t a, int32_t b) {
    int32_t na, nb;
    const int32_t *ra = token_rels_row(tr, a, &na);
    const int32_t *rb = token_rels_row(tr, b, &nb);
    if (!ra || !rb) return 0;
    int32_t i = 0, j = 0;
    while (i < na && j < nb) {
        if (ra[i] == rb[j]) return 1;
        else if (ra[i] < rb[j]) i++;
        else j++;
    }
    return 0;
}

void token_rels_intersect(const TokenRels *tr, int32_t a, int32_t b, i32vec *out) {
    int32_t na, nb;
    const int32_t *ra = token_rels_row(tr, a, &na);
    const int32_t *rb = token_rels_row(tr, b, &nb);
    if (!ra || !rb) return;
    int32_t i = 0, j = 0;
    while (i < na && j < nb) {
        if (ra[i] == rb[j]) { i32vec_push(out, ra[i]); i++; j++; }
        else if (ra[i] < rb[j]) i++;
        else j++;
    }
}

int token_rels_row_intersects(const TokenRels *tr, int32_t t, const int32_t *set_sorted, int32_t n) {
    int32_t nt;
    const int32_t *rt = token_rels_row(tr, t, &nt);
    if (!rt || n == 0) return 0;
    int32_t i = 0, j = 0;
    while (i < nt && j < n) {
        if (rt[i] == set_sorted[j]) return 1;
        else if (rt[i] < set_sorted[j]) i++;
        else j++;
    }
    return 0;
}

/* ============================================================ CTM */
static void build_member_signature(BridgeMatrix *bridges, RelationshipMatrix *rels, const TokenRels *token_rels,
                                    int32_t member, i32vec *out_triggers, i32vec *out_counts) {
    int32_t n_rels;
    const int32_t *rel_ids = token_rels_row(token_rels, member, &n_rels);
    if (!rel_ids) return;

    StrMap seen; strmap_init(&seen);
    Arena arena; arena_init(&arena);

    for (int32_t i = 0; i < n_rels; i++) {
        i32vec seq; i32vec_init(&seq);
        mse_sequence_for_relationship(rels, bridges, rel_ids[i], &seq);
        for (size_t k = 0; k < seq.len; k++) {
            int32_t t = seq.data[k];
            if (t == member || mse_is_reserved(t)) continue;
            int dup = 0;
            for (size_t j = 0; j < k; j++) if (seq.data[j] == t) { dup = 1; break; }
            if (dup) continue;
            int32_t key = t, idx;
            if (strmap_get(&seen, (const char *)&key, (int32_t)sizeof(key), &idx)) {
                out_counts->data[idx]++;
            } else {
                idx = (int32_t)out_triggers->len;
                i32vec_push(out_triggers, t);
                i32vec_push(out_counts, 1);
                char *owned = arena_strndup(&arena, (const char *)&key, (int32_t)sizeof(key));
                strmap_put(&seen, owned, (int32_t)sizeof(key), idx);
            }
        }
        i32vec_free(&seq);
    }
    strmap_free(&seen); arena_free(&arena);
}

void ctm_build(ContextTriggerMatrix *ctm, BridgeMatrix *bridges, RelationshipMatrix *rels,
                const TokenRels *token_rels, int32_t min_support) {
    memset(ctm, 0, sizeof(*ctm));
    ctm->max_cluster_id = bridges->max_cluster_id;
    if (ctm->max_cluster_id < 0) ctm->max_cluster_id = 0;
    ctm->cluster_to_idx = (int32_t *)malloc(sizeof(int32_t) * (size_t)(ctm->max_cluster_id + 1));
    for (int32_t i = 0; i <= ctm->max_cluster_id; i++) ctm->cluster_to_idx[i] = -1;

    int32_t cap = 16;
    ctm->sigs = (CtmClusterSig *)malloc(sizeof(CtmClusterSig) * (size_t)cap);
    ctm->n_sigs = 0;

    i32vec src, tgt, brg;
    for (int32_t cid = 1; cid <= ctm->max_cluster_id; cid++) {
        i32vec_init(&src); i32vec_init(&tgt); i32vec_init(&brg);
        BmAxis axis = bm_cluster_axis(bridges, cid, &src, &tgt, &brg);
        if (axis == BM_AXIS_NONE) { i32vec_free(&src); i32vec_free(&tgt); i32vec_free(&brg); continue; }

        /* distinct sorted members: bridge values (bridge-axis) or
         * target values (target-axis) across the cluster's rows */
        i32vec members_raw; i32vec_init(&members_raw);
        for (size_t i = 0; i < src.len; i++) i32vec_push(&members_raw, axis == BM_AXIS_BRIDGE ? brg.data[i] : tgt.data[i]);
        i32vec_free(&src); i32vec_free(&tgt); i32vec_free(&brg);
        /* sort + dedup */
        int32_t *d = members_raw.data; size_t ln = members_raw.len;
        for (size_t a = 1; a < ln; a++) { int32_t v = d[a]; size_t b = a; while (b > 0 && d[b-1] > v) { d[b] = d[b-1]; b--; } d[b] = v; }
        int32_t n_members = 0;
        for (size_t i = 0; i < ln; i++) if (i == 0 || d[i] != d[i - 1]) d[n_members++] = d[i];

        if (ctm->n_sigs == cap) { cap *= 2; ctm->sigs = (CtmClusterSig *)realloc(ctm->sigs, sizeof(CtmClusterSig) * (size_t)cap); }
        CtmClusterSig *sig = &ctm->sigs[ctm->n_sigs];
        sig->cluster_id = cid; sig->axis = axis;
        sig->n_members = n_members;
        sig->members = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n_members ? n_members : 1));
        memcpy(sig->members, d, sizeof(int32_t) * (size_t)n_members);
        i32vec_free(&members_raw);

        sig->sig_offsets = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n_members + 1));
        i32vec all_trig, all_supp; i32vec_init(&all_trig); i32vec_init(&all_supp);
        for (int32_t mi = 0; mi < n_members; mi++) {
            sig->sig_offsets[mi] = (int32_t)all_trig.len;
            i32vec triggers, counts; i32vec_init(&triggers); i32vec_init(&counts);
            build_member_signature(bridges, rels, token_rels, sig->members[mi], &triggers, &counts);
            for (size_t k = 0; k < triggers.len; k++) {
                if (counts.data[k] >= min_support) {
                    i32vec_push(&all_trig, triggers.data[k]);
                    i32vec_push(&all_supp, counts.data[k]);
                }
            }
            i32vec_free(&triggers); i32vec_free(&counts);
        }
        sig->sig_offsets[n_members] = (int32_t)all_trig.len;
        sig->sig_trigger = (int32_t *)malloc(sizeof(int32_t) * (size_t)(all_trig.len ? all_trig.len : 1));
        sig->sig_support = (int32_t *)malloc(sizeof(int32_t) * (size_t)(all_supp.len ? all_supp.len : 1));
        if (all_trig.len) memcpy(sig->sig_trigger, all_trig.data, sizeof(int32_t) * all_trig.len);
        if (all_supp.len) memcpy(sig->sig_support, all_supp.data, sizeof(int32_t) * all_supp.len);
        i32vec_free(&all_trig); i32vec_free(&all_supp);

        ctm->cluster_to_idx[cid] = ctm->n_sigs;
        ctm->n_sigs++;
    }
}

void ctm_free(ContextTriggerMatrix *ctm) {
    for (int32_t i = 0; i < ctm->n_sigs; i++) {
        free(ctm->sigs[i].members);
        free(ctm->sigs[i].sig_offsets);
        free(ctm->sigs[i].sig_trigger);
        free(ctm->sigs[i].sig_support);
    }
    free(ctm->sigs);
    free(ctm->cluster_to_idx);
    memset(ctm, 0, sizeof(*ctm));
}

void ctm_score_members(const ContextTriggerMatrix *ctm, int32_t cluster_id,
                        const int32_t *context_tokens, int32_t n_context,
                        i32vec *out_members, i32vec *out_scores) {
    if (cluster_id < 0 || cluster_id > ctm->max_cluster_id) return;
    int32_t idx = ctm->cluster_to_idx[cluster_id];
    if (idx < 0) return;
    const CtmClusterSig *sig = &ctm->sigs[idx];
    for (int32_t mi = 0; mi < sig->n_members; mi++) {
        int32_t score = 0;
        int32_t start = sig->sig_offsets[mi], end = sig->sig_offsets[mi + 1];
        for (int32_t k = start; k < end; k++) {
            int32_t trig = sig->sig_trigger[k];
            for (int32_t c = 0; c < n_context; c++) {
                if (context_tokens[c] == trig) { score += sig->sig_support[k]; break; }
            }
        }
        i32vec_push(out_members, sig->members[mi]);
        i32vec_push(out_scores, score);
    }
}

int ctm_select(const ContextTriggerMatrix *ctm, int32_t cluster_id,
                const int32_t *context_tokens, int32_t n_context,
                i32vec *out_top, int32_t *out_score) {
    i32vec members, scores; i32vec_init(&members); i32vec_init(&scores);
    ctm_score_members(ctm, cluster_id, context_tokens, n_context, &members, &scores);
    if (members.len == 0) { i32vec_free(&members); i32vec_free(&scores); return 0; }
    int32_t max_score = scores.data[0];
    for (size_t i = 1; i < scores.len; i++) if (scores.data[i] > max_score) max_score = scores.data[i];
    if (max_score <= 0) { i32vec_free(&members); i32vec_free(&scores); return 0; }
    for (size_t i = 0; i < members.len; i++) if (scores.data[i] == max_score) i32vec_push(out_top, members.data[i]);
    /* sort ascending */
    int32_t *d = out_top->data; size_t ln = out_top->len;
    for (size_t a = 1; a < ln; a++) { int32_t v = d[a]; size_t b = a; while (b > 0 && d[b-1] > v) { d[b] = d[b-1]; b--; } d[b] = v; }
    *out_score = max_score;
    i32vec_free(&members); i32vec_free(&scores);
    return 1;
}

int32_t ctm_resolve_tie(const ContextTriggerMatrix *ctm, BridgeMatrix *bridges,
                         const int32_t *candidates, int32_t n_candidates,
                         const int32_t *context_tokens, int32_t n_context) {
    if (n_context == 0 || n_candidates == 0) return -1;

    /* sorted candidates */
    i32vec cand; i32vec_init(&cand);
    for (int32_t i = 0; i < n_candidates; i++) i32vec_push(&cand, candidates[i]);
    int32_t *cd = cand.data; size_t cln = cand.len;
    for (size_t a = 1; a < cln; a++) { int32_t v = cd[a]; size_t b = a; while (b > 0 && cd[b-1] > v) { cd[b] = cd[b-1]; b--; } cd[b] = v; }

    /* per-candidate sorted cluster-id sets, intersected */
    i32vec shared; i32vec_init(&shared);
    int have_shared = 0;
    for (size_t i = 0; i < cand.len; i++) {
        i32vec cs; i32vec_init(&cs);
        bm_tindex_get(bridges, cand.data[i], &cs); /* already sorted ascending */
        if (cs.len == 0) { i32vec_free(&cs); i32vec_free(&cand); i32vec_free(&shared); return -1; }
        if (!have_shared) {
            for (size_t k = 0; k < cs.len; k++) i32vec_push(&shared, cs.data[k]);
            have_shared = 1;
        } else {
            i32vec next; i32vec_init(&next);
            size_t a = 0, b = 0;
            while (a < shared.len && b < cs.len) {
                if (shared.data[a] == cs.data[b]) { i32vec_push(&next, shared.data[a]); a++; b++; }
                else if (shared.data[a] < cs.data[b]) a++;
                else b++;
            }
            i32vec_free(&shared); shared = next;
        }
        i32vec_free(&cs);
    }
    i32vec_free(&cand);
    if (shared.len == 0) { i32vec_free(&shared); return -1; }

    int32_t result = -1;
    for (size_t i = 0; i < shared.len && result == -1; i++) {
        int32_t cid = shared.data[i];
        i32vec top; i32vec_init(&top);
        int32_t score;
        if (ctm_select(ctm, cid, context_tokens, n_context, &top, &score)) {
            i32vec relevant; i32vec_init(&relevant);
            for (size_t t = 0; t < top.len; t++)
                for (int32_t c = 0; c < n_candidates; c++)
                    if (top.data[t] == candidates[c]) { i32vec_push(&relevant, top.data[t]); break; }
            if (relevant.len == 1) result = relevant.data[0];
            i32vec_free(&relevant);
        }
        i32vec_free(&top);
    }
    i32vec_free(&shared);
    return result;
}
