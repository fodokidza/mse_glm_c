#include "mse_model.h"
#include "mse_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_stats(const char *label, MseModelStats s) {
    printf("%s vocab=%d edges=%d bridges=%d clustered=%d clusters=%d rels=%d rows=%d occ=%lld\n",
           label, s.vocab_size, s.edges, s.bridges, s.clustered_bridges, s.clusters,
           s.relationships, s.relationship_rows, (long long)s.relationship_occurrences);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <corpus1.txt> <corpus2.txt> [corpus3.txt ...]\n", argv[0]); return 1; }

    MseModel m;
    model_init(&m, 400);
    if (model_train_from_file(&m, argv[1]) != 0) { fprintf(stderr, "train failed\n"); return 1; }
    print_stats("after_initial_train:", model_stats(&m));

    for (int i = 2; i < argc; i++) {
        FILE *f = fopen(argv[i], "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[i]); return 1; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        char *buf = (char *)malloc((size_t)sz + 1);
        size_t rd = fread(buf, 1, (size_t)sz, f);
        fclose(f);
        buf[rd] = '\0';

        MseTrainIncrementalResult r = model_train_incremental(&m, buf, (int32_t)rd, 0, 0);
        printf("--- merge %d (sentences_added=%d vocab_added=%d ctm_invalidated=%d) ---\n",
               i - 1, r.sentences_added, r.vocab_added, r.ctm_invalidated);
        print_stats("before:", r.before);
        print_stats("after: ", r.after);
        free(buf);
    }

    printf("--- FINAL GENERATE (post-merge) ---\n");
    const char *prompts[] = { "the cat", "the dog sat", "a boy and", "is this", "numbers like" };
    for (int i = 0; i < 5; i++) {
        char *strict = model_generate(&m, prompts[i], GEN_MAX_TOKENS, IE_MODE_STRICT, 0, 0, NULL);
        printf("[strict] '%s' -> '%s'\n", prompts[i], strict);
        free(strict);
        char *open = model_generate(&m, prompts[i], GEN_MAX_TOKENS, IE_MODE_OPEN, 0, 0, NULL);
        printf("[open]   '%s' -> '%s'\n", prompts[i], open);
        free(open);
    }

    model_free(&m);
    return 0;
}
