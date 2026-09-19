#include "mse_ivm.h"
#include "mse_importance.h"
#include "mse_config.h"
#include <stdlib.h>
#include <string.h>

/* ===================================================== important[] */
static void build_important(uint8_t *important, BridgeMatrix *bridges, int32_t vocab_size) {
    memset(important, 0, (size_t)vocab_size);
    for (int32_t tid = 0; tid < bridges->n; tid++) {
        int32_t a, b, tok;
        BmAxis axis = mse_trigger_for_triple(bridges, tid, &a, &b, &tok);
        if (axis == BM_AXIS_NONE) continue;
        if (tok >= 0 && tok < vocab_size && !mse_is_reserved(tok)) important[tok] = 1;
    }
}

/* ================================================== BigramRelsIndex */
static int64_t pack2(int32_t a, int32_t b) { return ((int64_t)(uint32_t)a << 32) | (uint32_t)b; }

typedef struct { int64_t key; i32vec rels; } BrGroup;

static void bigram_rels_build(BigramRelsIndex *idx, BridgeMatrix *bridges, RelationshipMatrix *rels) {
    strmap_init(&idx->key_to_idx);
    arena_init(&idx->arena);

    BrGroup *groups = NULL; int32_t n_groups = 0, cap = 0;
    i32vec tmp; i32vec_init(&tmp);

    for (int32_t tid = 0; tid < bridges->n; tid++) {
        tmp.len = 0;
        rm_relationships_for_triple(rels, tid, &tmp);
        if (tmp.len == 0) continue;
        int64_t key = pack2(bridges->source[tid], bridges->bridge[tid]);
        int32_t gi;
        if (!strmap_get(&idx->key_to_idx, (const char *)&key, (int32_t)sizeof(key), &gi)) {
            if (n_groups == cap) { cap = cap ? cap * 2 : 16; groups = (BrGroup *)realloc(groups, sizeof(BrGroup) * (size_t)cap); }
            gi = n_groups++;
            groups[gi].key = key;
            i32vec_init(&groups[gi].rels);
            char *owned = arena_strndup(&idx->arena, (const char *)&key, (int32_t)sizeof(key));
            strmap_put(&idx->key_to_idx, owned, (int32_t)sizeof(key), gi);
        }
        for (size_t k = 0; k < tmp.len; k++) i32vec_push(&groups[gi].rels, tmp.data[k]);
    }
    i32vec_free(&tmp);

    idx->n = n_groups;
    idx->offsets = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n_groups + 1));
    int32_t total = 0;
    for (int32_t g = 0; g < n_groups; g++) {
        /* sort + dedup this group's rel ids */
        int32_t *d = groups[g].rels.data; size_t ln = groups[g].rels.len;
        for (size_t a = 1; a < ln; a++) { int32_t v = d[a]; size_t b = a; while (b > 0 && d[b-1] > v) { d[b] = d[b-1]; b--; } d[b] = v; }
        int32_t uniq = 0;
        for (size_t i = 0; i < ln; i++) if (i == 0 || d[i] != d[i-1]) d[uniq++] = d[i];
        groups[g].rels.len = (size_t)uniq;
        total += uniq;
    }
    idx->rel_ids = (int32_t *)malloc(sizeof(int32_t) * (size_t)(total ? total : 1));
    int32_t off = 0;
    /* re-map group index -> offset row, but key_to_idx already points at
     * `groups[gi]`'s original index, which we keep stable (groups[] is
     * never reordered), so idx->offsets is simply indexed by that same gi. */
    for (int32_t g = 0; g < n_groups; g++) {
        idx->offsets[g] = off;
        if (groups[g].rels.len) memcpy(idx->rel_ids + off, groups[g].rels.data, sizeof(int32_t) * groups[g].rels.len);
        off += (int32_t)groups[g].rels.len;
        i32vec_free(&groups[g].rels);
    }
    idx->offsets[n_groups] = off;
    free(groups);
}

static const int32_t *bigram_rels_lookup(const BigramRelsIndex *idx, int32_t source, int32_t bridge_tok, int32_t *out_n) {
    int64_t key = pack2(source, bridge_tok);
    int32_t gi;
    if (!strmap_get(&idx->key_to_idx, (const char *)&key, (int32_t)sizeof(key), &gi)) { *out_n = 0; return NULL; }
    int32_t start = idx->offsets[gi], end = idx->offsets[gi + 1];
    *out_n = end - start;
    return (*out_n > 0) ? (idx->rel_ids + start) : NULL;
}

static void bigram_rels_free(BigramRelsIndex *idx) {
    strmap_free(&idx->key_to_idx);
    arena_free(&idx->arena);
    free(idx->offsets); free(idx->rel_ids);
    memset(idx, 0, sizeof(*idx));
}

/* ====================================================== CoOccurIndex */
typedef struct { int64_t k; int32_t idx; } KI;
static int ki_cmp(const void *x, const void *y) {
    int64_t a = ((const KI *)x)->k, b = ((const KI *)y)->k;
    return (a < b) ? -1 : (a > b) ? 1 : 0;
}

static void co_occur_build(CoOccurIndex *co, const TokenRels *token_rels, int32_t n_rels, int32_t vocab_size) {
    co->vocab_size = vocab_size;
    i32vec *rel_buckets = (i32vec *)calloc((size_t)(n_rels ? n_rels : 1), sizeof(i32vec));
    for (int32_t r = 0; r < n_rels; r++) i32vec_init(&rel_buckets[r]);
    for (int32_t t = 0; t < vocab_size; t++) {
        if (mse_is_reserved(t)) continue;
        int32_t n; const int32_t *row = token_rels_row(token_rels, t, &n);
        for (int32_t i = 0; i < n; i++) {
            int32_t r = row[i];
            if (r >= 0 && r < n_rels) i32vec_push(&rel_buckets[r], t);
        }
    }

    StrMap seen; strmap_init(&seen);
    Arena arena; arena_init(&arena);
    i32vec pa, pb, pc; i32vec_init(&pa); i32vec_init(&pb); i32vec_init(&pc);

    for (int32_t r = 0; r < n_rels; r++) {
        i32vec *toks = &rel_buckets[r];
        for (size_t i = 0; i < toks->len; i++) {
            for (size_t j = 0; j < toks->len; j++) {
                if (i == j) continue;
                int32_t a = toks->data[i], b = toks->data[j];
                int64_t key = pack2(a, b);
                int32_t idx;
                if (strmap_get(&seen, (const char *)&key, (int32_t)sizeof(key), &idx)) {
                    pc.data[idx]++;
                } else {
                    idx = (int32_t)pa.len;
                    i32vec_push(&pa, a); i32vec_push(&pb, b); i32vec_push(&pc, 1);
                    char *owned = arena_strndup(&arena, (const char *)&key, (int32_t)sizeof(key));
                    strmap_put(&seen, owned, (int32_t)sizeof(key), idx);
                }
            }
        }
        i32vec_free(toks);
    }
    free(rel_buckets);
    strmap_free(&seen); arena_free(&arena);

    int32_t n = (int32_t)pa.len;
    size_t cnt = (n > 0) ? (size_t)n : 1u;
    /* sort by packed (a<<32|b) -- groups by a, sorted by b within each group */
    KI *ki = (KI *)malloc(sizeof(KI) * cnt);
    for (int32_t i = 0; i < n; i++) { ki[i].k = pack2(pa.data[i], pb.data[i]); ki[i].idx = i; }
    if (n > 0) qsort(ki, (size_t)n, sizeof(KI), ki_cmp);

    co->other_token = (int32_t *)malloc(sizeof(int32_t) * cnt);
    co->count = (int32_t *)malloc(sizeof(int32_t) * cnt);
    for (int32_t i = 0; i < n; i++) {
        int32_t p = ki[i].idx;
        co->other_token[i] = pb.data[p];
        co->count[i] = pc.data[p];
    }
    free(ki);

    co->offsets = (int32_t *)calloc((size_t)vocab_size + 1, sizeof(int32_t));
    for (int32_t i = 0; i < n; i++) co->offsets[pa.data[i] + 1]++;
    for (int32_t i = 1; i <= vocab_size; i++) co->offsets[i] += co->offsets[i - 1];

    i32vec_free(&pa); i32vec_free(&pb); i32vec_free(&pc);
}

static void co_occur_free(CoOccurIndex *co) {
    free(co->offsets); free(co->other_token); free(co->count);
    memset(co, 0, sizeof(*co));
}

static int co_occur_has(const CoOccurIndex *co, int32_t t, int32_t c) {
    if (t < 0 || t >= co->vocab_size) return 0;
    int32_t lo = co->offsets[t], hi = co->offsets[t + 1] - 1;
    while (lo <= hi) {
        int32_t mid = lo + (hi - lo) / 2;
        int32_t v = co->other_token[mid];
        if (v == c) return 1;
        if (v < c) lo = mid + 1; else hi = mid - 1;
    }
    return 0;
}

/* ============================================================== API */
void ivm_build(ImportanceVoteMatrix *ivm, const EdgeMatrix *edges, BridgeMatrix *bridges,
                RelationshipMatrix *rels, int32_t vocab_size) {
    memset(ivm, 0, sizeof(*ivm));
    ivm->edges = edges; ivm->bridges = bridges; ivm->vocab_size = vocab_size;
    ivm->important_weight = IVM_IMPORTANT_WEIGHT;
    ivm->influence_weight = IVM_INFLUENCE_WEIGHT;
    ivm->context_weight = IVM_CONTEXT_WEIGHT;
    ivm->context_influence_weight = IVM_CONTEXT_INFLUENCE_WEIGHT;
    ivm->bigram_witness_weight = IVM_BIGRAM_WITNESS_WEIGHT;
    ivm->adjacency_weight = IVM_ADJACENCY_WEIGHT;
    ivm->prev_current_weight = IVM_PREV_CURRENT_WEIGHT;
    ivm->triple_weight = IVM_TRIPLE_WEIGHT;
    ivm->whole_context_weight = IVM_WHOLE_CONTEXT_WEIGHT;

    token_rels_build(&ivm->token_rels, rels, bridges, vocab_size);
    ivm->important = (uint8_t *)malloc((size_t)vocab_size);
    build_important(ivm->important, bridges, vocab_size);
    bigram_rels_build(&ivm->bigram_rels, bridges, rels);
    co_occur_build(&ivm->co_occurring, &ivm->token_rels, rels->n_rels, vocab_size);
}

void ivm_free(ImportanceVoteMatrix *ivm) {
    token_rels_free(&ivm->token_rels);
    free(ivm->important);
    bigram_rels_free(&ivm->bigram_rels);
    co_occur_free(&ivm->co_occurring);
    memset(ivm, 0, sizeof(*ivm));
}

int32_t ivm_resolve_tie(const ImportanceVoteMatrix *ivm,
                         const int32_t *candidates, int32_t n_candidates,
                         const int32_t *context_tokens, int32_t n_context) {
    if (n_context == 0) return MSE_NONE;

    double *totals = (double *)calloc((size_t)(n_candidates ? n_candidates : 1), sizeof(double));
    int any_total = 0;
    /* dedup context tokens: Python's `knows` dict (keyed by token) means
     * a repeated context token is only ever processed once, no matter
     * how many times it appears in the caller's context_tokens. Real
     * callers always pass a Python `set` so this never bites there, but
     * this C API accepts a plain array and must not silently double-
     * count a token that happens to appear twice in it. */
    uint8_t *seen = (uint8_t *)calloc((size_t)ivm->vocab_size, 1);

    for (int32_t ci = 0; ci < n_context; ci++) {
        int32_t t = context_tokens[ci];
        if (t < 0 || t >= ivm->vocab_size) continue;
        if (seen[t]) continue;
        seen[t] = 1;
        if (!ivm->important[t] || mse_is_reserved(t)) continue;

        int32_t influence_t = 0;
        for (int32_t p = 0; p < n_candidates; p++) {
            int32_t c = candidates[p];
            if (c == t || mse_is_reserved(c)) continue;
            if (token_rels_intersect_any(&ivm->token_rels, t, c)) influence_t++;
        }
        if (influence_t == 0) continue;
        for (int32_t p = 0; p < n_candidates; p++) {
            int32_t c = candidates[p];
            if (c == t || mse_is_reserved(c)) continue;
            if (token_rels_intersect_any(&ivm->token_rels, t, c)) { totals[p] += influence_t; any_total = 1; }
        }
    }

    int32_t result = MSE_NONE;
    if (any_total) {
        double max_v = -1;
        for (int32_t p = 0; p < n_candidates; p++) if (totals[p] > 0 && totals[p] > max_v) max_v = totals[p];
        int32_t winner = MSE_NONE, count = 0;
        for (int32_t p = 0; p < n_candidates; p++) {
            if (totals[p] == max_v && max_v > 0) {
                count++;
                if (winner == MSE_NONE || candidates[p] < winner) winner = candidates[p];
            }
        }
        if (count == 1) result = winner;
    }
    free(totals);
    free(seen);
    return result;
}

void ivm_score_candidates(const ImportanceVoteMatrix *ivm,
                           const int32_t *candidates, int32_t n_candidates,
                           const int32_t *context_tokens, int32_t n_context,
                           int32_t current, int32_t previous,
                           double *out_scores) {
    for (int32_t i = 0; i < n_candidates; i++) out_scores[i] = 0.0;
    if (n_candidates == 0) return;

    int32_t vs = ivm->vocab_size;

    /* Every real caller (inference.py, and this port's own inference.c)
     * passes context_tokens as a deduplicated set -- score_candidates()
     * relies on that: V1/V2 are accumulated through a dict keyed by
     * token (idempotent to a repeat), while V3/V4/V6 are accumulated
     * through a plain running sum (NOT idempotent to a repeat). Rather
     * than replicate that inconsistency for a case that never occurs in
     * practice, dedupe once here so every layer sees each context token
     * exactly once, matching real-world behavior uniformly. */
    uint8_t *ctx_seen = (uint8_t *)calloc((size_t)vs, 1);
    i32vec ctx; i32vec_init(&ctx);
    for (int32_t i = 0; i < n_context; i++) {
        int32_t t = context_tokens[i];
        if (t < 0 || t >= vs || ctx_seen[t]) continue;
        ctx_seen[t] = 1;
        i32vec_push(&ctx, t);
    }
    free(ctx_seen);
    context_tokens = ctx.data;
    n_context = (int32_t)ctx.len;

    int32_t *cand_pos = (int32_t *)malloc(sizeof(int32_t) * (size_t)vs);
    for (int32_t i = 0; i < vs; i++) cand_pos[i] = -1;
    for (int32_t i = 0; i < n_candidates; i++)
        if (candidates[i] >= 0 && candidates[i] < vs) cand_pos[candidates[i]] = i;

    double *important_vote = (double *)calloc((size_t)n_candidates, sizeof(double));
    double *influence_vote = (double *)calloc((size_t)n_candidates, sizeof(double));
    double *raw_context = (double *)calloc((size_t)n_candidates, sizeof(double));
    double *context_influence_vote = (double *)calloc((size_t)n_candidates, sizeof(double));
    double *raw_adjacency = (double *)calloc((size_t)n_candidates, sizeof(double));
    double *bigram_witness_vote = (double *)calloc((size_t)n_candidates, sizeof(double));
    double *prev_current_vote = (double *)calloc((size_t)n_candidates, sizeof(double));
    double *triple_vote = (double *)calloc((size_t)n_candidates, sizeof(double));
    double *whole_context_vote = (double *)calloc((size_t)n_candidates, sizeof(double));

    /* V1/V2/V3/V4: single merged pass over context tokens, gated by
     * co_occurring's row (only candidates a token has ever actually
     * shared a relationship with are ever visited). */
    i32vec knows_pos; i32vec_init(&knows_pos);
    for (int32_t ci = 0; ci < n_context; ci++) {
        int32_t t = context_tokens[ci];
        if (t < 0 || t >= vs || mse_is_reserved(t)) continue;
        int32_t start = ivm->co_occurring.offsets[t], end = ivm->co_occurring.offsets[t + 1];
        if (start == end) continue;
        int is_important = ivm->important[t];
        knows_pos.len = 0;
        for (int32_t k = start; k < end; k++) {
            int32_t c = ivm->co_occurring.other_token[k];
            int32_t pos = cand_pos[c];
            if (pos < 0) continue;
            int32_t shared = ivm->co_occurring.count[k];
            raw_context[pos] += 1.0;
            context_influence_vote[pos] += ivm->context_influence_weight * (double)shared;
            if (is_important) i32vec_push(&knows_pos, pos);
        }
        if (is_important && knows_pos.len) {
            double inf = (double)knows_pos.len;
            for (size_t k = 0; k < knows_pos.len; k++) {
                int32_t pos = knows_pos.data[k];
                important_vote[pos] += ivm->important_weight * 1.0;
                influence_vote[pos] += ivm->influence_weight * inf;
            }
        }
    }
    i32vec_free(&knows_pos);

    /* V6: adjacency -- independent EdgeMatrix walk (see header comment) */
    for (int32_t ci = 0; ci < n_context; ci++) {
        int32_t t = context_tokens[ci];
        if (t < 0 || t >= vs || mse_is_reserved(t)) continue;
        if (t + 1 >= ivm->edges->vocab_size + 1 || !ivm->edges->index) continue;
        int32_t start = ivm->edges->index[t], end = ivm->edges->index[t + 1];
        for (int32_t k = start; k < end; k++) {
            int32_t c = ivm->edges->dst[k];
            if (c == t) continue;
            int32_t pos = cand_pos[c];
            if (pos < 0) continue;
            raw_adjacency[pos] += 1.0;
        }
    }

    /* V5: bigram witness -- gated to candidates V3/V6 already touched */
    if (current != MSE_NONE) {
        for (int32_t pos = 0; pos < n_candidates; pos++) {
            if (!(raw_context[pos] > 0 || raw_adjacency[pos] > 0)) continue;
            int32_t c = candidates[pos];
            if (mse_is_reserved(c)) continue;
            int32_t wn; const int32_t *witness = bigram_rels_lookup(&ivm->bigram_rels, current, c, &wn);
            if (!witness) continue;
            for (int32_t ci = 0; ci < n_context; ci++) {
                int32_t t = context_tokens[ci];
                if (t < 0 || t >= vs || mse_is_reserved(t) || t == c) continue;
                if (token_rels_row_intersects(&ivm->token_rels, t, witness, wn)) bigram_witness_vote[pos] += 1.0;
            }
        }
    }

    /* V7/V8: fixed-pair/triple checks on (previous, current) */
    int32_t cur_in_ctx = (current != MSE_NONE && current >= 0 && current < vs) ? 0 : 1; /* default true if current is None */
    int32_t prev_in_ctx = (previous != MSE_NONE && previous >= 0 && previous < vs) ? 0 : 1;
    for (int32_t ci = 0; ci < n_context; ci++) {
        if (current != MSE_NONE && context_tokens[ci] == current) cur_in_ctx = 1;
        if (previous != MSE_NONE && context_tokens[ci] == previous) prev_in_ctx = 1;
    }
    int v78_gate_safe = cur_in_ctx && prev_in_ctx;

    if (previous != MSE_NONE && current != MSE_NONE && !mse_is_reserved(previous) && !mse_is_reserved(current)) {
        i32vec shared_pc; i32vec_init(&shared_pc);
        token_rels_intersect(&ivm->token_rels, previous, current, &shared_pc);
        if (shared_pc.len) {
            for (int32_t pos = 0; pos < n_candidates; pos++) {
                int32_t c = candidates[pos];
                if (mse_is_reserved(c) || c == previous || c == current) continue;
                if (v78_gate_safe && !(raw_context[pos] > 0 || raw_adjacency[pos] > 0)) continue;
                if (token_rels_row_intersects(&ivm->token_rels, c, shared_pc.data, (int32_t)shared_pc.len))
                    prev_current_vote[pos] = ivm->prev_current_weight * 1.0;
            }
        }
        i32vec_free(&shared_pc);

        for (int32_t pos = 0; pos < n_candidates; pos++) {
            int32_t c = candidates[pos];
            if (mse_is_reserved(c) || c == previous || c == current) continue;
            if (v78_gate_safe && !(raw_context[pos] > 0 || raw_adjacency[pos] > 0)) continue;
            if (bm_has_triple(ivm->bridges, previous, current, c))
                triple_vote[pos] = ivm->triple_weight * 1.0;
        }
    }

    /* V9: unanimous -- requires every non-reserved context token (other
     * than C itself) to co-occur with C */
    {
        i32vec required_all; i32vec_init(&required_all);
        for (int32_t ci = 0; ci < n_context; ci++) {
            int32_t t = context_tokens[ci];
            if (t >= 0 && t < vs && !mse_is_reserved(t)) i32vec_push(&required_all, t);
        }
        if (required_all.len) {
            for (int32_t pos = 0; pos < n_candidates; pos++) {
                if (!(raw_context[pos] > 0 || raw_adjacency[pos] > 0)) continue; /* gate: V9 implies V3 for all, hence for >=1 */
                int32_t c = candidates[pos];
                if (mse_is_reserved(c)) continue;
                int has_required = 0, all_know = 1;
                for (size_t k = 0; k < required_all.len && all_know; k++) {
                    int32_t t = required_all.data[k];
                    if (t == c) continue;
                    has_required = 1;
                    if (!co_occur_has(&ivm->co_occurring, t, c)) all_know = 0;
                }
                if (has_required && all_know) whole_context_vote[pos] = ivm->whole_context_weight * 1.0;
            }
        }
        i32vec_free(&required_all);
    }

    for (int32_t pos = 0; pos < n_candidates; pos++) {
        double v3 = ivm->context_weight * raw_context[pos];
        double v6 = ivm->adjacency_weight * raw_adjacency[pos];
        double v5 = ivm->bigram_witness_weight * bigram_witness_vote[pos];
        out_scores[pos] = important_vote[pos] + influence_vote[pos] + v3 + context_influence_vote[pos]
                         + v5 + v6 + prev_current_vote[pos] + triple_vote[pos] + whole_context_vote[pos];
    }

    free(cand_pos);
    free(important_vote); free(influence_vote); free(raw_context); free(context_influence_vote); free(raw_adjacency);
    free(bigram_witness_vote); free(prev_current_vote); free(triple_vote); free(whole_context_vote);
    i32vec_free(&ctx);
}

int32_t ivm_select(const ImportanceVoteMatrix *ivm,
                    const int32_t *candidates, int32_t n_candidates,
                    const int32_t *context_tokens, int32_t n_context,
                    int32_t current, int32_t previous) {
    if (n_candidates == 0) return MSE_NONE;

    double *scores = (double *)malloc(sizeof(double) * (size_t)n_candidates);
    ivm_score_candidates(ivm, candidates, n_candidates, context_tokens, n_context, current, previous, scores);

    double max_score = scores[0];
    for (int32_t i = 1; i < n_candidates; i++) if (scores[i] > max_score) max_score = scores[i];

    i32vec top; i32vec_init(&top);
    for (int32_t i = 0; i < n_candidates; i++) if (scores[i] == max_score) i32vec_push(&top, candidates[i]);
    free(scores);
    /* sort ascending */
    { int32_t *d = top.data; size_t ln = top.len;
      for (size_t a = 1; a < ln; a++) { int32_t v = d[a]; size_t b = a; while (b > 0 && d[b-1] > v) { d[b] = d[b-1]; b--; } d[b] = v; } }

    if (top.len > 1 && current != MSE_NONE) {
        int32_t max_bf = -1;
        for (size_t i = 0; i < top.len; i++) { int32_t f = em_frequency(ivm->edges, current, top.data[i]); if (f > max_bf) max_bf = f; }
        i32vec next; i32vec_init(&next);
        for (size_t i = 0; i < top.len; i++) if (em_frequency(ivm->edges, current, top.data[i]) == max_bf) i32vec_push(&next, top.data[i]);
        i32vec_free(&top); top = next;
    }
    if (top.len > 1) {
        int32_t max_gf = -1;
        for (size_t i = 0; i < top.len; i++) {
            int32_t n; token_rels_row(&ivm->token_rels, top.data[i], &n);
            if (n > max_gf) max_gf = n;
        }
        i32vec next; i32vec_init(&next);
        for (size_t i = 0; i < top.len; i++) {
            int32_t n; token_rels_row(&ivm->token_rels, top.data[i], &n);
            if (n == max_gf) i32vec_push(&next, top.data[i]);
        }
        i32vec_free(&top); top = next;
    }

    int32_t winner = top.len ? top.data[0] : MSE_NONE;
    i32vec_free(&top);
    return winner;
}
