#include "mse_graph.h"
#include "mse_ctm.h"
#include "mse_ivm.h"
#include "mse_config.h"
#include "mse_importance.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <prefix> <vocab_size>\n", argv[0]); return 1; }
    const char *prefix = argv[1];
    int32_t vocab_size = atoi(argv[2]);
    char path[4096];

    EdgeMatrix edges; snprintf(path, sizeof(path), "%s.edges", prefix); em_load(&edges, path);
    BridgeMatrix bridges; snprintf(path, sizeof(path), "%s.bridges", prefix); bm_load(&bridges, path);
    RelationshipMatrix rels; snprintf(path, sizeof(path), "%s.rels", prefix); rm_load(&rels, path);

    ImportanceVoteMatrix ivm;
    ivm_build(&ivm, &edges, &bridges, &rels, vocab_size);

    printf("--- IMPORTANT ---\n");
    for (int32_t t = 0; t < vocab_size; t++) if (ivm.important[t]) printf("%d\n", t);

    /* Build a handful of representative (context, current, previous)
     * scenarios directly from real training sequences reconstructed
     * from the Relationship Matrix, so this exercises realistic
     * generation-shaped inputs, not synthetic ones. */
    printf("--- SCORE_CANDIDATES ---\n");
    i32vec all_candidates; i32vec_init(&all_candidates);
    for (int32_t t = 0; t < vocab_size; t++) i32vec_push(&all_candidates, t);

    int32_t n_scenarios = 0;
    for (int32_t rel_id = 0; rel_id < rels.n_rels && n_scenarios < 15; rel_id++) {
        i32vec seq; i32vec_init(&seq);
        mse_sequence_for_relationship(&rels, &bridges, rel_id, &seq);
        if (seq.len < 3) { i32vec_free(&seq); continue; }
        /* score at the point right after the first two tokens */
        int32_t previous = seq.data[0], current = seq.data[1];
        i32vec ctx; i32vec_init(&ctx);
        for (size_t k = 0; k <= 1 && k < seq.len; k++) i32vec_push(&ctx, seq.data[k]);

        double *scores = (double *)malloc(sizeof(double) * all_candidates.len);
        ivm_score_candidates(&ivm, all_candidates.data, (int32_t)all_candidates.len,
                              ctx.data, (int32_t)ctx.len, current, previous, scores);
        printf("scenario rel=%d prev=%d cur=%d ctx_n=%d\n", rel_id, previous, current, (int)ctx.len);
        for (size_t c = 0; c < all_candidates.len; c++) {
            if (scores[c] != 0.0) printf("  c=%d score=%.6f\n", all_candidates.data[c], scores[c]);
        }
        int32_t winner = ivm_select(&ivm, all_candidates.data, (int32_t)all_candidates.len,
                                     ctx.data, (int32_t)ctx.len, current, previous);
        printf("  winner=%d\n", winner);
        free(scores);
        i32vec_free(&ctx);
        i32vec_free(&seq);
        n_scenarios++;
    }

    printf("--- RESOLVE_TIE ---\n");
    for (int32_t a = 0; a < vocab_size && a < 100; a++) {
        for (int32_t b = a + 1; b < vocab_size && b < 100; b++) {
            int32_t cand[2] = { a, b };
            int32_t ctx[3] = { a, b, TOK_EOS };
            int32_t r = ivm_resolve_tie(&ivm, cand, 2, ctx, 3);
            if (r != MSE_NONE) printf("(%d,%d) -> %d\n", a, b, r);
        }
    }

    i32vec_free(&all_candidates);
    ivm_free(&ivm);
    em_free(&edges); bm_free(&bridges); rm_free(&rels);
    return 0;
}
