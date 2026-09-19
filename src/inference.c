#include "mse_inference.h"
#include "mse_importance.h"
#include "mse_config.h"
#include <stdlib.h>
#include <string.h>

/* ============================================================ RelSet */
static void relset_add_unique(RelSet *s, int32_t v) {
    for (size_t i = 0; i < s->len; i++) if (s->data[i] == v) return;
    i32vec_push(s, v);
}
static void relset_extend_unique(RelSet *dst, const i32vec *src) {
    for (size_t i = 0; i < src->len; i++) relset_add_unique(dst, src->data[i]);
}
static void relset_intersect(const RelSet *a, const RelSet *b, RelSet *out) {
    for (size_t i = 0; i < a->len; i++)
        for (size_t j = 0; j < b->len; j++)
            if (a->data[i] == b->data[j]) { relset_add_unique(out, a->data[i]); break; }
}
static void relset_copy(const RelSet *src, RelSet *dst) {
    i32vec_init(dst);
    if (src) for (size_t i = 0; i < src->len; i++) i32vec_push(dst, src->data[i]);
}
/* narrow(): dst gets a FRESH set (caller must not have pre-inited it
 * with content); matches Python's `_narrow` which always returns a
 * (possibly aliased) set rather than mutating in place. */
static void narrow(const RelSet *active, const RelSet *newset, RelSet *out) {
    i32vec_init(out);
    if (active->len == 0) { for (size_t i = 0; i < newset->len; i++) i32vec_push(out, newset->data[i]); return; }
    RelSet inter; i32vec_init(&inter);
    relset_intersect(active, newset, &inter);
    if (inter.len) { *out = inter; }
    else { i32vec_free(&inter); for (size_t i = 0; i < newset->len; i++) i32vec_push(out, newset->data[i]); }
}

/* ======================================================== signal helpers */
static void bridge_rels_from_source(BridgeMatrix *bridges, RelationshipMatrix *rels,
                                     int32_t current, int32_t candidate, RelSet *out) {
    if (current < 0 || current + 1 >= bridges->vocab_size + 1 || !bridges->index) return;
    int32_t start = bridges->index[current], end = bridges->index[current + 1];
    i32vec tmp; i32vec_init(&tmp);
    for (int32_t i = start; i < end; i++) {
        if (bridges->bridge[i] == candidate) {
            tmp.len = 0;
            rm_relationships_for_triple(rels, i, &tmp);
            relset_extend_unique(out, &tmp);
        }
    }
    i32vec_free(&tmp);
}

static void exact_triple_rels(BridgeMatrix *bridges, RelationshipMatrix *rels,
                               int32_t previous, int32_t current, int32_t target, RelSet *out) {
    if (previous == MSE_NONE) return;
    if (previous < 0 || previous + 1 >= bridges->vocab_size + 1 || !bridges->index) return;
    int32_t start = bridges->index[previous], end = bridges->index[previous + 1];
    i32vec tmp; i32vec_init(&tmp);
    for (int32_t i = start; i < end; i++) {
        if (bridges->bridge[i] == current && (target == MSE_NONE || bridges->target[i] == target)) {
            tmp.len = 0;
            rm_relationships_for_triple(rels, i, &tmp);
            relset_extend_unique(out, &tmp);
        }
    }
    i32vec_free(&tmp);
}

static int32_t bigram_tie_break(const EdgeMatrix *edges, int32_t current, const int32_t *tied, int32_t n) {
    i32vec t; i32vec_init(&t);
    for (int32_t i = 0; i < n; i++) i32vec_push(&t, tied[i]);
    int32_t *d = t.data; size_t ln = t.len;
    for (size_t a = 1; a < ln; a++) { int32_t v = d[a]; size_t b = a; while (b > 0 && d[b-1] > v) { d[b] = d[b-1]; b--; } d[b] = v; }
    int32_t max_freq = -1;
    for (size_t i = 0; i < t.len; i++) { int32_t f = em_frequency(edges, current, t.data[i]); if (f > max_freq) max_freq = f; }
    int32_t winner = t.len ? t.data[0] : MSE_NONE;
    for (size_t i = 0; i < t.len; i++) if (em_frequency(edges, current, t.data[i]) == max_freq) { winner = t.data[i]; break; }
    i32vec_free(&t);
    return winner;
}

static int32_t find_index(const i32vec *arr, int32_t v) {
    for (size_t i = 0; i < arr->len; i++) if (arr->data[i] == v) return (int32_t)i;
    return -1;
}

static void sort_i32(i32vec *v) {
    int32_t *d = v->data; size_t ln = v->len;
    for (size_t a = 1; a < ln; a++) { int32_t x = d[a]; size_t b = a; while (b > 0 && d[b-1] > x) { d[b] = d[b-1]; b--; } d[b] = x; }
}

/* ============================================================== API */
void ie_init(InferenceEngine *ie, const EdgeMatrix *edges, BridgeMatrix *bridges,
             RelationshipMatrix *rels, IeMode mode, int32_t *vocab, int32_t n_vocab) {
    ie->edges = edges; ie->bridges = bridges; ie->rels = rels;
    ie->mode = mode; ie->vocab = vocab; ie->n_vocab = n_vocab;
}

static int32_t open_mode_step(InferenceEngine *ie, int32_t previous, int32_t current,
                               const int32_t *candidates, int32_t n_candidates,
                               const int32_t *context_tokens, int32_t n_context,
                               ImportanceVoteMatrix *ivm, const RelSet *active_rels, StepTrace *out_trace) {
    (void)ie;
    if (ivm) {
        int32_t winner = ivm_select(ivm, candidates, n_candidates, context_tokens, n_context, current, previous);
        if (winner != MSE_NONE) {
            out_trace->chosen = winner; out_trace->stage = 1; out_trace->rule = "ctm_weighted_vote";
            relset_copy(active_rels, &out_trace->active_rels);
            return winner;
        }
    }
    int32_t winner = candidates[0];
    out_trace->chosen = winner; out_trace->stage = 1; out_trace->rule = "ctm_unavailable_deterministic_fallback";
    relset_copy(active_rels, &out_trace->active_rels);
    return winner;
}

int32_t ie_step(InferenceEngine *ie, int32_t previous, int32_t current,
                 const RelSet *active_rels_in,
                 const int32_t *context_tokens, int32_t n_context,
                 ContextTriggerMatrix *ctm, ImportanceVoteMatrix *ivm,
                 StepTrace *out_trace) {
    RelSet active_rels; relset_copy(active_rels_in, &active_rels);

    if (ie->mode == IE_MODE_OPEN) {
        if (ie->n_vocab == 0) {
            out_trace->chosen = TOK_EOS; out_trace->stage = 4; out_trace->rule = "termination_empty_vocabulary";
            relset_copy(&active_rels, &out_trace->active_rels);
            i32vec_free(&active_rels);
            return TOK_EOS;
        }
        int32_t winner = open_mode_step(ie, previous, current, ie->vocab, ie->n_vocab,
                                         context_tokens, n_context, ivm, &active_rels, out_trace);
        i32vec_free(&active_rels);
        return winner;
    }

    /* ---- STRICT MODE ---- */
    i32vec succs; i32vec_init(&succs);
    em_successors(ie->edges, current, &succs);
    sort_i32(&succs);
    if (succs.len == 0) {
        out_trace->chosen = TOK_EOS; out_trace->stage = 4; out_trace->rule = "termination_no_successors";
        relset_copy(&active_rels, &out_trace->active_rels);
        i32vec_free(&active_rels); i32vec_free(&succs);
        return TOK_EOS;
    }

    /* Stage 1 */
    i32vec s1_cand; i32vec_init(&s1_cand);
    RelSet *s1_rels = (RelSet *)malloc(sizeof(RelSet) * succs.len);
    int32_t n_s1 = 0;
    for (size_t i = 0; i < succs.len; i++) {
        int32_t C = succs.data[i];
        RelSet rels; i32vec_init(&rels);
        bridge_rels_from_source(ie->bridges, ie->rels, current, C, &rels);
        if (active_rels.len) {
            RelSet matched; i32vec_init(&matched);
            relset_intersect(&rels, &active_rels, &matched);
            i32vec_free(&rels);
            if (matched.len) { i32vec_push(&s1_cand, C); s1_rels[n_s1++] = matched; }
            else i32vec_free(&matched);
        } else {
            if (rels.len) { i32vec_push(&s1_cand, C); s1_rels[n_s1++] = rels; }
            else i32vec_free(&rels);
        }
    }
    if (n_s1 == 0) {
        for (size_t i = 0; i < succs.len; i++) {
            i32vec_push(&s1_cand, succs.data[i]);
            RelSet e; i32vec_init(&e);
            s1_rels[n_s1++] = e;
        }
    }

    int32_t token = MSE_NONE;
    const char *rule = NULL;
    int stage = 1;
    RelSet new_rels; int have_new_rels = 0;

    if (n_s1 == 1) {
        token = s1_cand.data[0];
        narrow(&active_rels, &s1_rels[0], &new_rels); have_new_rels = 1;
        rule = "bridge_lineage_unique";
        goto emit;
    }

    if (ivm) {
        int32_t winner = ivm_resolve_tie(ivm, s1_cand.data, n_s1, context_tokens, n_context);
        rule = "importance_vote_resolved";
        if (winner == MSE_NONE && ctm) { winner = ctm_resolve_tie(ctm, ie->bridges, s1_cand.data, n_s1, context_tokens, n_context); rule = "context_trigger_resolved"; }
        if (winner == MSE_NONE) { winner = bigram_tie_break(ie->edges, current, s1_cand.data, n_s1); rule = "importance_vote_bigram_frequency"; }
        token = winner;
        int32_t idx = find_index(&s1_cand, token);
        narrow(&active_rels, &s1_rels[idx], &new_rels); have_new_rels = 1;
        goto emit;
    }

    if (previous == MSE_NONE) {
        int32_t winner = ctm ? ctm_resolve_tie(ctm, ie->bridges, s1_cand.data, n_s1, context_tokens, n_context) : MSE_NONE;
        if (winner != MSE_NONE) { token = winner; rule = "context_trigger_resolved"; }
        else { token = bigram_tie_break(ie->edges, current, s1_cand.data, n_s1); rule = "no_previous_bigram_frequency"; }
        int32_t idx = find_index(&s1_cand, token);
        narrow(&active_rels, &s1_rels[idx], &new_rels); have_new_rels = 1;
        goto emit;
    }

    /* ---- Stage 2 ---- */
    {
        RelSet prev_curr_rels; i32vec_init(&prev_curr_rels);
        exact_triple_rels(ie->bridges, ie->rels, previous, current, MSE_NONE, &prev_curr_rels);
        if (active_rels.len && prev_curr_rels.len) {
            RelSet narrowed; i32vec_init(&narrowed);
            relset_intersect(&active_rels, &prev_curr_rels, &narrowed);
            if (narrowed.len) { i32vec_free(&active_rels); active_rels = narrowed; }
            else i32vec_free(&narrowed);
        }
        i32vec_free(&prev_curr_rels);

        i32vec s2_cand; i32vec_init(&s2_cand);
        RelSet *s2_rels = (RelSet *)malloc(sizeof(RelSet) * (size_t)(n_s1 ? n_s1 : 1));
        int32_t n_s2 = 0;
        for (int32_t i = 0; i < n_s1; i++) {
            int32_t C = s1_cand.data[i];
            RelSet rels; i32vec_init(&rels);
            exact_triple_rels(ie->bridges, ie->rels, previous, current, C, &rels);
            if (active_rels.len) {
                RelSet matched; i32vec_init(&matched);
                relset_intersect(&rels, &active_rels, &matched);
                i32vec_free(&rels);
                if (matched.len) { i32vec_push(&s2_cand, C); s2_rels[n_s2++] = matched; }
                else i32vec_free(&matched);
            } else {
                if (rels.len) { i32vec_push(&s2_cand, C); s2_rels[n_s2++] = rels; }
                else i32vec_free(&rels);
            }
        }

        if (n_s2 > 0) {
            if (n_s2 == 1) {
                token = s2_cand.data[0];
                narrow(&active_rels, &s2_rels[0], &new_rels); have_new_rels = 1;
                rule = "exact_match_unique"; stage = 2;
            } else {
                int32_t winner = ctm ? ctm_resolve_tie(ctm, ie->bridges, s2_cand.data, n_s2, context_tokens, n_context) : MSE_NONE;
                if (winner != MSE_NONE) { token = winner; rule = "context_trigger_resolved"; }
                else { token = bigram_tie_break(ie->edges, current, s2_cand.data, n_s2); rule = "bigram_frequency"; }
                int32_t idx = find_index(&s2_cand, token);
                narrow(&active_rels, &s2_rels[idx], &new_rels); have_new_rels = 1;
                stage = 2;
            }
        } else {
            int32_t winner = ctm ? ctm_resolve_tie(ctm, ie->bridges, s1_cand.data, n_s1, context_tokens, n_context) : MSE_NONE;
            if (winner != MSE_NONE) { token = winner; rule = "context_trigger_resolved"; }
            else { token = bigram_tie_break(ie->edges, current, s1_cand.data, n_s1); rule = "s2_empty_bigram_frequency"; }
            int32_t idx = find_index(&s1_cand, token);
            narrow(&active_rels, &s1_rels[idx], &new_rels); have_new_rels = 1;
            stage = 1;
        }

        for (int32_t i = 0; i < n_s2; i++) i32vec_free(&s2_rels[i]);
        free(s2_rels);
        i32vec_free(&s2_cand);
    }

emit:
    out_trace->chosen = token; out_trace->stage = stage; out_trace->rule = rule;
    if (have_new_rels) out_trace->active_rels = new_rels; else relset_copy(&active_rels, &out_trace->active_rels);

    for (int32_t i = 0; i < n_s1; i++) i32vec_free(&s1_rels[i]);
    free(s1_rels);
    i32vec_free(&s1_cand);
    i32vec_free(&succs);
    i32vec_free(&active_rels);
    return token;
}

void ie_generate(InferenceEngine *ie, i32vec *ids, int32_t max_tokens,
                  ContextTriggerMatrix *ctm, ImportanceVoteMatrix *ivm) {
    if (ie->mode == IE_MODE_STRICT) {
        for (int32_t i = 1; i + 1 < (int32_t)ids->len; i++) {
            int32_t curr = ids->data[i], nxt = ids->data[i + 1];
            i32vec succ; i32vec_init(&succ);
            em_successors(ie->edges, curr, &succ);
            int found = 0;
            for (size_t k = 0; k < succ.len; k++) if (succ.data[k] == nxt) { found = 1; break; }
            i32vec_free(&succ);
            if (!found) return; /* illegal prompt bigram: stop, matching Python's early return */
        }
    }

    RelSet active_rels; i32vec_init(&active_rels);
    for (int32_t i = 0; i + 2 < (int32_t)ids->len; i++) {
        RelSet rels; i32vec_init(&rels);
        exact_triple_rels(ie->bridges, ie->rels, ids->data[i], ids->data[i + 1], ids->data[i + 2], &rels);
        if (rels.len) {
            RelSet nr; narrow(&active_rels, &rels, &nr);
            i32vec_free(&active_rels); active_rels = nr;
        } else {
            i32vec_free(&active_rels); i32vec_init(&active_rels);
        }
        i32vec_free(&rels);
    }

    int32_t previous = (ids->len >= 2) ? ids->data[ids->len - 2] : MSE_NONE;
    int32_t current = ids->data[ids->len - 1];

    for (int32_t step_i = 0; step_i < max_tokens; step_i++) {
        /* context_tokens = set(ids) */
        uint8_t *seen = (uint8_t *)calloc((size_t)ie->bridges->vocab_size, 1);
        int32_t vs = ie->bridges->vocab_size;
        i32vec ctx; i32vec_init(&ctx);
        for (size_t i = 0; i < ids->len; i++) {
            int32_t t = ids->data[i];
            if (t < 0 || t >= vs || seen[t]) continue;
            seen[t] = 1;
            i32vec_push(&ctx, t);
        }
        free(seen);

        StepTrace trace;
        int32_t token = ie_step(ie, previous, current, &active_rels, ctx.data, (int32_t)ctx.len, ctm, ivm, &trace);
        i32vec_free(&ctx);

        if (token == TOK_EOS) { i32vec_free(&trace.active_rels); break; }
        i32vec_push(ids, token);
        /* active_rels = set(trace.active_rels) or active_rels -- keep the
         * old set if the new one is empty (falsy), matching Python's `or`. */
        if (trace.active_rels.len) { i32vec_free(&active_rels); active_rels = trace.active_rels; }
        else i32vec_free(&trace.active_rels);
        previous = current; current = token;
    }
    i32vec_free(&active_rels);
}
