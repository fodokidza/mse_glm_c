/* noise.c — C port of noise.py's fast, anchor-independent
 * three-stage noise-cancellation scoring. See mse_noise.h for the
 * formula and the key simplification this implements.
 */
#include "mse_noise.h"
#include "mse_importance.h"
#include "mse_config.h"
#include <stdlib.h>
#include <string.h>

/* Builds ni->rt_offsets/rt_tokens: for each relationship, its literal
 * reconstructed sequence (mse_sequence_for_relationship), deduplicated
 * and with every STRUCTURAL token (PAD/UNK/BOS/EOS) stripped out —
 * exactly noise.py's tokens_for(rid) == set(sequence) - _STRUCTURAL.
 * Also derives vocab_all_size, the size of the union of every row —
 * stage 3's fixed pool size. */
static void build_rel_tokens(NoiseIndex *ni, RelationshipMatrix *rels, const BridgeMatrix *bridges) {
    int32_t n_rels = ni->n_rels;
    ni->rt_offsets = (int32_t *)calloc((size_t)n_rels + 1, sizeof(int32_t));

    i32vec *rows = (i32vec *)malloc(sizeof(i32vec) * (size_t)(n_rels > 0 ? n_rels : 1));
    uint8_t *seen_vocab = (uint8_t *)calloc((size_t)ni->vocab_size, 1);
    int32_t total = 0;

    for (int32_t r = 0; r < n_rels; r++) {
        i32vec seq; i32vec_init(&seq);
        mse_sequence_for_relationship(rels, bridges, r, &seq);

        uint8_t *seen_row = (uint8_t *)calloc((size_t)ni->vocab_size, 1);
        i32vec_init(&rows[r]);
        for (size_t k = 0; k < seq.len; k++) {
            int32_t tok = seq.data[k];
            if (tok < 0 || tok >= ni->vocab_size) continue;
            if (mse_is_structural(tok)) continue;
            if (seen_row[tok]) continue;
            seen_row[tok] = 1;
            i32vec_push(&rows[r], tok);
            if (!seen_vocab[tok]) { seen_vocab[tok] = 1; ni->vocab_all_size++; }
        }
        /* sort each row ascending so later intersections/unions can walk
         * it directly (not strictly required for correctness -- the
         * caller only ever unions/counts -- but keeps rows canonical) */
        { int32_t *d = rows[r].data; size_t ln = rows[r].len;
          for (size_t a = 1; a < ln; a++) { int32_t v = d[a]; size_t b = a; while (b > 0 && d[b-1] > v) { d[b] = d[b-1]; b--; } d[b] = v; } }
        free(seen_row);
        total += (int32_t)rows[r].len;
        i32vec_free(&seq);
    }
    free(seen_vocab);

    ni->rt_tokens = (int32_t *)malloc(sizeof(int32_t) * (size_t)(total > 0 ? total : 1));
    int32_t pos = 0;
    for (int32_t r = 0; r < n_rels; r++) {
        ni->rt_offsets[r] = pos;
        for (size_t k = 0; k < rows[r].len; k++) ni->rt_tokens[pos++] = rows[r].data[k];
        i32vec_free(&rows[r]);
    }
    ni->rt_offsets[n_rels] = pos;
    free(rows);
}

void noise_index_build(NoiseIndex *ni, RelationshipMatrix *rels, const BridgeMatrix *bridges,
                        const TokenRels *token_rels, int32_t vocab_size, double vote_weight) {
    memset(ni, 0, sizeof(*ni));
    ni->rels = rels;
    ni->token_rels = token_rels;
    ni->vocab_size = vocab_size;
    ni->n_rels = rels->n_rels;
    ni->vote_weight = vote_weight;

    build_rel_tokens(ni, rels, bridges);

    ni->avg_cache = (double *)calloc((size_t)vocab_size, sizeof(double));
    ni->avg_ok = (uint8_t *)calloc((size_t)vocab_size, 1);
}

void noise_index_free(NoiseIndex *ni) {
    free(ni->rt_offsets); free(ni->rt_tokens);
    free(ni->avg_cache); free(ni->avg_ok);
    memset(ni, 0, sizeof(*ni));
}

static inline void rt_row(const NoiseIndex *ni, int32_t rel, const int32_t **out, int32_t *out_n) {
    *out = ni->rt_tokens + ni->rt_offsets[rel];
    *out_n = ni->rt_offsets[rel + 1] - ni->rt_offsets[rel];
}

int noise_candidate_average(NoiseIndex *ni, int32_t cand, double *out) {
    if (cand < 0 || cand >= ni->vocab_size) return 0;
    if (ni->avg_ok[cand]) { *out = ni->avg_cache[cand]; return 1; }

    if (mse_is_structural(cand)) return 0;

    int32_t n_c;
    const int32_t *rels_c = token_rels_row(ni->token_rels, cand, &n_c);
    if (!rels_c || n_c == 0) return 0; /* unscorable -- no relationships at all */

    /* reach(cand) = union_{r in rels_c} tokens_for(r) -- dedup via a
     * vocab-sized scratch bitset (reused across calls but allocated
     * fresh each call; a candidate's average is cached forever after,
     * so this O(vocab) scratch cost is paid at most once per DISTINCT
     * candidate ever queried, matching noise.py's own per-instance
     * cache), plus a compact list so the stage-2 walk below only
     * visits reach's actual members instead of rescanning the vocab. */
    uint8_t *reach_seen = (uint8_t *)calloc((size_t)ni->vocab_size, 1);
    i32vec reach_list; i32vec_init(&reach_list);
    for (int32_t i = 0; i < n_c; i++) {
        int32_t r = rels_c[i];
        const int32_t *row; int32_t rn;
        rt_row(ni, r, &row, &rn);
        for (int32_t k = 0; k < rn; k++) {
            int32_t tok = row[k];
            if (!reach_seen[tok]) { reach_seen[tok] = 1; i32vec_push(&reach_list, tok); }
        }
    }

    /* stage2 = n_rels - |knowing_rids(cand)|, knowing_rids = union of
     * token_rels[t] for every t in reach(cand) -- via a bitset over
     * relationship ids (same role as the Python fast path's bitmask
     * OR, just expressed as a byte array instead of a big int). */
    uint8_t *rel_hit = (uint8_t *)calloc((size_t)ni->n_rels, 1);
    int32_t knowing_n = 0;
    for (size_t li = 0; li < reach_list.len; li++) {
        int32_t tok = reach_list.data[li];
        int32_t tn;
        const int32_t *trow = token_rels_row(ni->token_rels, tok, &tn);
        if (!trow) continue;
        for (int32_t k = 0; k < tn; k++) {
            int32_t r = trow[k];
            if (!rel_hit[r]) { rel_hit[r] = 1; knowing_n++; }
        }
    }
    int32_t reach_n = (int32_t)reach_list.len;
    free(rel_hit);
    free(reach_seen);
    i32vec_free(&reach_list);

    double stage1 = (double)(ni->n_rels - n_c);
    double stage2 = (double)(ni->n_rels - knowing_n);
    double stage3 = (double)(ni->vocab_all_size - reach_n);
    double avg = ni->vote_weight * (stage1 + stage2 + stage3) / 3.0;

    ni->avg_cache[cand] = avg;
    ni->avg_ok[cand] = 1;
    *out = avg;
    return 1;
}
