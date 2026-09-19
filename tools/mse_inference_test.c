#include "mse_graph.h"
#include "mse_ctm.h"
#include "mse_ivm.h"
#include "mse_inference.h"
#include "mse_importance.h"
#include "mse_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_ids(const int32_t *d, int32_t n) {
    for (int32_t i = 0; i < n; i++) printf("%s%d", i ? " " : "", d[i]);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <prefix> <vocab_size>\n", argv[0]); return 1; }
    const char *prefix = argv[1];
    int32_t vocab_size = atoi(argv[2]);
    char path[4096];

    EdgeMatrix edges; snprintf(path, sizeof(path), "%s.edges", prefix); em_load(&edges, path);
    BridgeMatrix bridges; snprintf(path, sizeof(path), "%s.bridges", prefix); bm_load(&bridges, path);
    RelationshipMatrix rels; snprintf(path, sizeof(path), "%s.rels", prefix); rm_load(&rels, path);

    TokenRels tr; token_rels_build(&tr, &rels, &bridges, vocab_size);
    ContextTriggerMatrix ctm; ctm_build(&ctm, &bridges, &rels, &tr, CTM_MIN_SUPPORT);
    ImportanceVoteMatrix ivm; ivm_build(&ivm, &edges, &bridges, &rels, vocab_size);

    i32vec vocab; i32vec_init(&vocab);
    for (int32_t t = 0; t < vocab_size; t++) i32vec_push(&vocab, t);

    /* Build prompts from real training prefixes (first 1, 2, 3 tokens of
     * several relationships) -- realistic generation-shaped input. */
    i32vec prompt_starts; i32vec_init(&prompt_starts); /* rel_ids to use as prompt sources */
    for (int32_t r = 0; r < rels.n_rels && prompt_starts.len < 20; r++) i32vec_push(&prompt_starts, r);

    const char *mode_names[2] = { "strict", "open" };
    for (int mode_i = 0; mode_i < 2; mode_i++) {
        printf("=== MODE %s ===\n", mode_names[mode_i]);
        InferenceEngine ie;
        ie_init(&ie, &edges, &bridges, &rels, mode_i == 0 ? IE_MODE_STRICT : IE_MODE_OPEN,
                vocab.data, (int32_t)vocab.len);

        for (size_t pi = 0; pi < prompt_starts.len; pi++) {
            i32vec seq; i32vec_init(&seq);
            mse_sequence_for_relationship(&rels, &bridges, prompt_starts.data[pi], &seq);
            if (seq.len < 2) { i32vec_free(&seq); continue; }
            int32_t plen = (seq.len >= 3) ? 3 : (int32_t)seq.len; /* first up-to-3 tokens as prompt */

            for (int use_ctm = 0; use_ctm <= 1; use_ctm++) {
                i32vec ids; i32vec_init(&ids);
                i32vec_push(&ids, TOK_BOS);
                for (int32_t k = 0; k < plen; k++) i32vec_push(&ids, seq.data[k]);

                ie_generate(&ie, &ids, 12, use_ctm ? &ctm : NULL, mode_i == 1 ? &ivm : NULL);
                printf("prompt_rel=%zu use_ctm=%d ids=", pi, use_ctm);
                print_ids(ids.data, (int32_t)ids.len);
                printf("\n");
                i32vec_free(&ids);
            }
            i32vec_free(&seq);
        }
    }

    i32vec_free(&vocab);
    i32vec_free(&prompt_starts);
    token_rels_free(&tr);
    ctm_free(&ctm);
    ivm_free(&ivm);
    em_free(&edges); bm_free(&bridges); rm_free(&rels);
    return 0;
}
