#include "mse_graph.h"
#include "mse_ctm.h"
#include "mse_config.h"
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

    TokenRels tr;
    token_rels_build(&tr, &rels, &bridges, vocab_size);

    printf("--- TOKEN_RELS ---\n");
    for (int32_t t = 0; t < vocab_size; t++) {
        int32_t n; const int32_t *row = token_rels_row(&tr, t, &n);
        if (n == 0) continue;
        printf("%d:", t);
        for (int32_t i = 0; i < n; i++) printf(" %d", row[i]);
        printf("\n");
    }

    ContextTriggerMatrix ctm;
    ctm_build(&ctm, &bridges, &rels, &tr, CTM_MIN_SUPPORT);

    printf("--- CTM_SIGS ---\n");
    for (int32_t i = 0; i < ctm.n_sigs; i++) {
        CtmClusterSig *sig = &ctm.sigs[i];
        printf("cluster=%d axis=%s members=", sig->cluster_id, sig->axis == BM_AXIS_BRIDGE ? "bridge" : "target");
        for (int32_t m = 0; m < sig->n_members; m++) printf("%d,", sig->members[m]);
        printf("\n");
        for (int32_t m = 0; m < sig->n_members; m++) {
            int32_t start = sig->sig_offsets[m], end = sig->sig_offsets[m + 1];
            /* sort by trigger id for a stable diffable dump */
            int32_t cnt = end - start;
            int32_t *idx = (int32_t *)malloc(sizeof(int32_t) * (cnt ? cnt : 1));
            for (int32_t k = 0; k < cnt; k++) idx[k] = start + k;
            for (int32_t a = 1; a < cnt; a++) {
                int32_t v = idx[a]; int32_t b = a;
                while (b > 0 && sig->sig_trigger[idx[b-1]] > sig->sig_trigger[v]) { idx[b] = idx[b-1]; b--; }
                idx[b] = v;
            }
            for (int32_t k = 0; k < cnt; k++)
                printf("  member=%d trigger=%d support=%d\n", sig->members[m], sig->sig_trigger[idx[k]], sig->sig_support[idx[k]]);
            free(idx);
        }
    }

    /* resolve_tie spot checks: for every pair of tokens sharing 2+ clusters via t_index,
       try ctm_resolve_tie with the full training vocabulary context (a stand-in for a
       real generation context) and print the result -- exercised more precisely in the
       companion Python script using the same candidate/context inputs. */
    printf("--- RESOLVE_TIE_SAMPLE ---\n");
    for (int32_t a = 0; a < vocab_size && a < 200; a++) {
        for (int32_t b = a + 1; b < vocab_size && b < 200; b++) {
            int32_t cand[2] = { a, b };
            int32_t ctx[4] = { a, b, TOK_BOS, TOK_EOS };
            int32_t r = ctm_resolve_tie(&ctm, &bridges, cand, 2, ctx, 4);
            if (r != -1) printf("(%d,%d) ctx=(%d,%d) -> %d\n", a, b, a, b, r);
        }
    }

    token_rels_free(&tr);
    ctm_free(&ctm);
    em_free(&edges); bm_free(&bridges); rm_free(&rels);
    return 0;
}
