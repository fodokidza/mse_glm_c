/* mse_train.c — trains a tokenizer + Edge/Bridge/Relationship graphs
 * from a corpus file, prints stats, and dumps a debug report (vocab,
 * merges, edges, bridges, rels) in a simple line-oriented text format
 * so it can be diffed directly against a Python reference dump.
 */
#include "mse_tokenizer.h"
#include "mse_graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_file(const char *path, long *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    buf[rd] = '\0';
    fclose(f);
    if (out_len) *out_len = (long)rd;
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <corpus.txt> <out_prefix> [vocab_size] [--dump]\n", argv[0]);
        return 1;
    }
    const char *corpus_path = argv[1];
    const char *out_prefix = argv[2];
    int32_t vocab_size = (argc > 3) ? atoi(argv[3]) : TOKENIZER_DEFAULT_VOCAB_SIZE;
    int dump = (argc > 4 && strcmp(argv[4], "--dump") == 0);

    long clen;
    char *corpus = read_file(corpus_path, &clen);

    MseTokenizer tok;
    mse_tok_init(&tok, vocab_size);
    mse_tok_train(&tok, corpus, (int32_t)clen);

    int32_t n_sent; int32_t *lens;
    char **sents = mse_split_sentences(corpus, (int32_t)clen, &lens, &n_sent);
    i32vec *seqs = (i32vec *)malloc(sizeof(i32vec) * (size_t)(n_sent ? n_sent : 1));
    for (int32_t i = 0; i < n_sent; i++) {
        i32vec_init(&seqs[i]);
        mse_tok_encode_for_training(&tok, sents[i], lens[i], &seqs[i]);
    }

    EdgeMatrix edges; em_init(&edges);
    em_build(&edges, seqs, n_sent, mse_tok_vocab_size_actual(&tok));

    BridgeMatrix bridges; bm_init(&bridges);
    bm_build(&bridges, seqs, n_sent, mse_tok_vocab_size_actual(&tok));

    RelationshipMatrix rels; rm_init(&rels);
    rm_build(&rels, seqs, n_sent, &bridges);

    int32_t clustered = 0, maxc = 0;
    for (int32_t i = 0; i < bridges.n; i++) { if (bridges.cluster_id[i]) clustered++; if (bridges.cluster_id[i] > maxc) maxc = bridges.cluster_id[i]; }
    int64_t total_rel_occ = 0;
    for (int32_t i = 0; i < rels.n_rels; i++) total_rel_occ += rels.rel_count[i];

    printf("vocab_size=%d\n", mse_tok_vocab_size_actual(&tok));
    printf("sentences=%d\n", n_sent);
    printf("edges=%d\n", edges.n);
    printf("bridges=%d\n", bridges.n);
    printf("clustered_bridges=%d\n", clustered);
    printf("clusters=%d\n", maxc);
    printf("relationships=%d\n", rels.n_rels);
    printf("relationship_rows=%d\n", rels.n_rows);
    printf("relationship_occurrences=%lld\n", (long long)total_rel_occ);

    char path[4096];
    snprintf(path, sizeof(path), "%s.tok", out_prefix); mse_tok_save(&tok, path);
    snprintf(path, sizeof(path), "%s.edges", out_prefix); em_save(&edges, path);
    snprintf(path, sizeof(path), "%s.bridges", out_prefix); bm_save(&bridges, path);
    snprintf(path, sizeof(path), "%s.rels", out_prefix); rm_save(&rels, path);

    if (dump) {
        printf("--- CHARS ---\n");
        for (int32_t id = TOK_FIRST_FREE; id < mse_tok_vocab_size_actual(&tok); id++) {
            if (!cv_is_char_id(&tok.chars, id)) continue;
            char b = (char)cv_decode_id(&tok.chars, id);
            printf("%d\t%.*s\n", id, 1, &b);
        }
        printf("--- WORDS ---\n");
        for (int32_t id = TOK_FIRST_FREE; id < mse_tok_vocab_size_actual(&tok); id++) {
            if (cv_is_char_id(&tok.chars, id)) continue;
            int32_t l; const char *s = wv_decode_id(&tok.words, id, &l);
            if (l == 0) continue;
            printf("%d\t%.*s\n", id, l, s);
        }
        printf("--- EDGES ---\n");
        for (int32_t i = 0; i < edges.n; i++)
            printf("%d\t%d\t%d\n", edges.src[i], edges.dst[i], edges.count[i]);
        printf("--- BRIDGES ---\n");
        for (int32_t i = 0; i < bridges.n; i++)
            printf("%d\t%d\t%d\t%d\n", bridges.source[i], bridges.target[i], bridges.bridge[i], bridges.cluster_id[i]);
        printf("--- RELS ---\n");
        for (int32_t i = 0; i < rels.n_rows; i++)
            printf("%d\t%d\n", rels.r_triple[i], rels.r_rel[i]);
        printf("--- REL_COUNT ---\n");
        for (int32_t i = 0; i < rels.n_rels; i++)
            printf("%d\t%d\n", i, rels.rel_count[i]);
    }

    /* round-trip encode/decode sanity check on the first few sentences */
    printf("--- ROUNDTRIP (first 5 sentences) ---\n");
    for (int32_t i = 0; i < n_sent && i < 5; i++) {
        char *dec = mse_tok_decode(&tok, seqs[i].data, (int32_t)seqs[i].len);
        printf("[%d] %s\n", i, dec);
        free(dec);
    }

    for (int32_t i = 0; i < n_sent; i++) i32vec_free(&seqs[i]);
    free(seqs);
    mse_free_sentences(sents, n_sent);
    free(lens);
    rm_free(&rels); bm_free(&bridges); em_free(&edges);
    mse_tok_free(&tok);
    free(corpus);
    return 0;
}
