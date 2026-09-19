/* mse_chat.c — interactive REPL, C port of chat.py's core command set.
 *
 * Ports: plain-text generation, /mode strict|open, /stats, /quit.
 * Deferred (lower priority than a working end-to-end generation loop,
 * and each already has a dedicated, more detailed test tool covering
 * its logic): /scores, /bigram, /cache, /shared, /similarity,
 * /explain, /clusters. Those introspection commands' underlying logic
 * is already ported and validated (mse_ivm_test, mse_ctm_test,
 * mse_interpret_test) -- what's missing here is only the REPL-command
 * plumbing around them, not new engine logic.
 */
#include "mse_model.h"
#include "mse_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *rstrip(char *s) {
    size_t n = strlen(s);
    while (n > 0 && (s[n-1] == '\n' || s[n-1] == '\r' || s[n-1] == ' ' || s[n-1] == '\t')) s[--n] = '\0';
    return s;
}
static char *lstrip(char *s) {
    while (*s == ' ' || *s == '\t') s++;
    return s;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model_folder> [max_tokens]\n", argv[0]);
        fprintf(stderr, "  (train one first with mse_train or mse_model_test)\n");
        return 1;
    }
    const char *folder = argv[1];
    int32_t max_tokens = (argc > 2) ? atoi(argv[2]) : GEN_CHAT_CLI_DEFAULT_MAX_TOKENS;

    MseModel m;
    model_init(&m, TOKENIZER_DEFAULT_VOCAB_SIZE);
    if (model_load(&m, folder) != 0) {
        fprintf(stderr, "failed to load model from '%s'\n", folder);
        return 1;
    }
    MseModelStats s = model_stats(&m);
    printf("Loaded model: vocab=%d edges=%d bridges=%d (%d clustered, %d clusters) relationships=%d\n",
           s.vocab_size, s.edges, s.bridges, s.clustered_bridges, s.clusters, s.relationships);
    printf("Commands: /mode strict|open, /stats, /quit. Anything else generates a continuation.\n");

    IeMode mode = IE_MODE_STRICT;
    char line[4096];
    for (;;) {
        printf("%s> ", mode == IE_MODE_STRICT ? "strict" : "open");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        char *l = lstrip(rstrip(line));
        if (!*l) continue;

        if (strcmp(l, "/quit") == 0 || strcmp(l, "/exit") == 0) break;

        if (strncmp(l, "/mode", 5) == 0 && (l[5] == '\0' || l[5] == ' ')) {
            char *arg = lstrip(l + 5);
            if (strcmp(arg, "strict") == 0) { mode = IE_MODE_STRICT; printf("  mode -> strict\n"); }
            else if (strcmp(arg, "open") == 0) { mode = IE_MODE_OPEN; printf("  mode -> open\n"); }
            else printf("  Usage: /mode strict|open\n");
            continue;
        }

        if (strcmp(l, "/stats") == 0) {
            MseModelStats st = model_stats(&m);
            printf("  vocab_size=%d edges=%d bridges=%d clustered_bridges=%d clusters=%d "
                   "relationships=%d relationship_rows=%d relationship_occurrences=%lld\n",
                   st.vocab_size, st.edges, st.bridges, st.clustered_bridges, st.clusters,
                   st.relationships, st.relationship_rows, (long long)st.relationship_occurrences);
            continue;
        }

        char *text = model_generate(&m, l, max_tokens, mode, 0, 0, NULL);
        printf("  model> %s\n", text);
        free(text);
    }

    model_free(&m);
    return 0;
}
