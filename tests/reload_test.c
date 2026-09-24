#include "mse_tokenizer.h"
#include "mse_graph.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    MseTokenizer tok;
    printf("loading tok...\n");
    if (mse_tok_load(&tok, "/tmp/mse_model2.tok") != 0) { printf("load tok FAIL\n"); return 1; }
    printf("loaded tok\n");
    EdgeMatrix e; if (em_load(&e, "/tmp/mse_model2.edges") != 0) { printf("load edges FAIL\n"); return 1; }
    printf("loaded edges\n");
    BridgeMatrix b; if (bm_load(&b, "/tmp/mse_model2.bridges") != 0) { printf("load bridges FAIL\n"); return 1; }
    printf("loaded bridges\n");
    RelationshipMatrix r; if (rm_load(&r, "/tmp/mse_model2.rels") != 0) { printf("load rels FAIL\n"); return 1; }
    printf("loaded rels\n");

    printf("vocab_size=%d\n", mse_tok_vocab_size_actual(&tok));
    printf("chars=%d word_id_span=%d\n", tok.chars.max_id - TOK_WORD_BOUND, tok.words.count);
    printf("edges=%d bridges=%d rels=%d rows=%d\n", e.n, b.n, r.n_rels, r.n_rows);

    /* re-encode/decode a sentence through the reloaded tokenizer */
    const char *s = "The cat sat on the mat.";
    i32vec ids; i32vec_init(&ids);
    mse_tok_encode_for_training(&tok, s, (int32_t)strlen(s), &ids);
    printf("encoded, n=%zu\n", ids.len);
    char *dec = mse_tok_decode(&tok, ids.data, (int32_t)ids.len);
    printf("decoded: %s\n", dec);
    free(dec);
    i32vec_free(&ids);

    /* spot-check a successor query and a triple lookup still work post-load */
    i32vec succ; i32vec_init(&succ);
    em_successors(&e, 44 /* 'the' or similar, whatever id */, &succ);
    printf("successors(44) count=%zu\n", succ.len);
    i32vec_free(&succ);

    fflush(stdout);
    printf("before tok_free\n"); fflush(stdout);
    mse_tok_free(&tok);
    printf("before em_free\n"); fflush(stdout);
    em_free(&e);
    printf("before bm_free\n"); fflush(stdout);
    bm_free(&b);
    printf("before rm_free\n"); fflush(stdout);
    rm_free(&r);
    printf("OK\n");
    return 0;
}
