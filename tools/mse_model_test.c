#include "mse_model.h"
#include "mse_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_stats(MseModelStats s) {
    printf("vocab_size=%d\n", s.vocab_size);
    printf("edges=%d\n", s.edges);
    printf("bridges=%d\n", s.bridges);
    printf("clustered_bridges=%d\n", s.clustered_bridges);
    printf("clusters=%d\n", s.clusters);
    printf("relationships=%d\n", s.relationships);
    printf("relationship_rows=%d\n", s.relationship_rows);
    printf("relationship_occurrences=%lld\n", (long long)s.relationship_occurrences);
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <corpus.txt> <vocab_size> <save_folder>\n", argv[0]); return 1; }
    const char *corpus_path = argv[1];
    int32_t vocab_size = atoi(argv[2]);
    const char *save_folder = argv[3];

    MseModel m;
    model_init(&m, vocab_size);
    if (model_train_from_file(&m, corpus_path) != 0) { fprintf(stderr, "train failed\n"); return 1; }

    printf("--- STATS (after train) ---\n");
    print_stats(model_stats(&m));

    model_build_context_triggers(&m, CTM_MIN_SUPPORT);
    model_build_importance_votes(&m);

    printf("--- SAVE/LOAD ROUND TRIP ---\n");
    if (model_save(&m, save_folder) != 0) { fprintf(stderr, "save failed\n"); return 1; }
    MseModel m2;
    model_init(&m2, vocab_size);
    if (model_load(&m2, save_folder) != 0) { fprintf(stderr, "load failed\n"); return 1; }
    MseModelStats s1 = model_stats(&m), s2 = model_stats(&m2);
    int stats_match = s1.vocab_size == s2.vocab_size && s1.edges == s2.edges && s1.bridges == s2.bridges &&
                       s1.clustered_bridges == s2.clustered_bridges && s1.clusters == s2.clusters &&
                       s1.relationships == s2.relationships && s1.relationship_rows == s2.relationship_rows &&
                       s1.relationship_occurrences == s2.relationship_occurrences;
    printf("stats_match=%d\n", stats_match);

    printf("--- GENERATE (various prompts/modes, using the RELOADED model) ---\n");
    const char *prompts[] = { "the cat", "the dog sat", "a boy and", "is this", "numbers like" };
    for (int i = 0; i < 5; i++) {
        char *strict = model_generate(&m2, prompts[i], GEN_MAX_TOKENS, IE_MODE_STRICT, 0, 0, NULL);
        printf("[strict]      '%s' -> '%s'\n", prompts[i], strict);
        free(strict);
        char *open = model_generate(&m2, prompts[i], GEN_MAX_TOKENS, IE_MODE_OPEN, 0, 0, NULL);
        printf("[open]        '%s' -> '%s'\n", prompts[i], open);
        free(open);
    }

    /* opt-ins rebuilt on the reloaded model (CTM/IVM are not persisted) */
    model_build_context_triggers(&m2, CTM_MIN_SUPPORT);
    model_build_importance_votes(&m2);
    for (int i = 0; i < 5; i++) {
        char *strict_ctm = model_generate(&m2, prompts[i], GEN_MAX_TOKENS, IE_MODE_STRICT, 1, 0, NULL);
        printf("[strict+ctm]  '%s' -> '%s'\n", prompts[i], strict_ctm);
        free(strict_ctm);
        char *strict_ivm = model_generate(&m2, prompts[i], GEN_MAX_TOKENS, IE_MODE_STRICT, 0, 1, NULL);
        printf("[strict+ivm]  '%s' -> '%s'\n", prompts[i], strict_ivm);
        free(strict_ivm);
    }

    model_free(&m2);
    model_free(&m);
    return 0;
}
