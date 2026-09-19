/* mse_train_corpus.c — C port of train_corpus.py: trains from a folder
 * of many .txt files instead of one string/file.
 *   Pass 1 (vocabulary): accumulate word frequencies file by file via
 *     the public MseWordFreq API (mse_tokenizer.h) -- only the current
 *     file's raw text and the cumulative (much smaller) word-count
 *     table are ever in memory together, never the whole corpus's raw
 *     text at once. One tokenizer is trained from the combined counts.
 *
 *   Pass 2 (graph): batches of --batch-size files at a time. The first
 *     batch calls model_build_graphs() (from scratch); every batch
 *     after that merges in via model_merge_graphs() (see model.c's
 *     comment -- full recompute over the union, same cost profile
 *     train_corpus.py documents: closer to O(batches * corpus_size)
 *     than O(corpus_size), so bigger --batch-size trades memory for
 *     speed on genuinely large corpora).
 *
 * Known limitation (same one train_corpus.py documents): each file's
 * full text is read into memory at once during Pass 2 (not chunked
 * within one file) -- this assumes a large CORPUS split across many
 * reasonably-sized FILES. If one file is itself enormous, pre-split it.
 */
#define _POSIX_C_SOURCE 200809L
#include "mse_model.h"
#include "mse_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <dirent.h>
#include <sys/stat.h>
#include <time.h>

typedef struct { char **paths; int32_t n, cap; } PathList;

static void pathlist_push(PathList *l, const char *p) {
    if (l->n == l->cap) { l->cap = l->cap ? l->cap * 2 : 64; l->paths = (char **)realloc(l->paths, sizeof(char *) * (size_t)l->cap); }
    l->paths[l->n] = strdup(p);
    l->n++;
}
static void pathlist_free(PathList *l) {
    for (int32_t i = 0; i < l->n; i++) free(l->paths[i]);
    free(l->paths);
}
static int has_txt_ext(const char *name) {
    size_t n = strlen(name);
    return n >= 4 && strcasecmp(name + n - 4, ".txt") == 0;
}
static void discover_txt_files(const char *folder, int recursive, PathList *out) {
    DIR *d = opendir(folder);
    if (!d) return;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", folder, ent->d_name);
        struct stat st;
        if (stat(path, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            if (recursive) discover_txt_files(path, recursive, out);
        } else if (S_ISREG(st.st_mode) && has_txt_ext(ent->d_name)) {
            pathlist_push(out, path);
        }
    }
    closedir(d);
}
static int cmp_str(const void *a, const void *b) { return strcmp(*(const char *const *)a, *(const char *const *)b); }

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static char *read_whole_file(const char *path, long *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = '\0';
    if (out_len) *out_len = (long)rd;
    return buf;
}

int main(int argc, char **argv) {
    const char *corpus_dir = NULL, *out_dir = NULL;
    int32_t vocab_size = TOKENIZER_DEFAULT_VOCAB_SIZE;
    int32_t batch_size = TRAIN_CORPUS_DEFAULT_BATCH_SIZE;
    int recursive = 1, quiet = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--corpus-dir") == 0 && i + 1 < argc) corpus_dir = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_dir = argv[++i];
        else if (strcmp(argv[i], "--vocab-size") == 0 && i + 1 < argc) vocab_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--no-recursive") == 0) recursive = 0;
        else if (strcmp(argv[i], "--quiet") == 0) quiet = 1;
        else { fprintf(stderr, "unknown or incomplete argument: %s\n", argv[i]); return 1; }
    }
    if (!corpus_dir || !out_dir) {
        fprintf(stderr,
            "usage: %s --corpus-dir <dir> --out <dir> [--vocab-size N] [--batch-size N] "
            "[--no-recursive] [--quiet]\n", argv[0]);
        return 1;
    }
    if (batch_size < 1) { fprintf(stderr, "--batch-size must be >= 1\n"); return 1; }

    PathList files; memset(&files, 0, sizeof(files));
    discover_txt_files(corpus_dir, recursive, &files);
    if (files.n == 0) { fprintf(stderr, "No .txt files found under '%s'\n", corpus_dir); return 1; }
    qsort(files.paths, (size_t)files.n, sizeof(char *), cmp_str);

    if (!quiet) {
        printf("  MSE Graph Language Model  -  Large-corpus training\n");
        printf("  corpus_dir   %s\n", corpus_dir);
        printf("  files found  %d  (%s)\n", files.n, recursive ? "recursive" : "top-level only");
        printf("  vocab_size   %d\n", vocab_size);
        printf("  batch_size   %d  files per merge step\n\n", batch_size);
    }

    double t0 = now_sec();

    /* ---- Pass 1: vocabulary from streamed word frequencies ---- */
    if (!quiet) printf("  Pass 1/2  building shared vocabulary...\n");
    MseWordFreq *wf = mse_wordfreq_create();
    for (int32_t i = 0; i < files.n; i++) {
        if (mse_wordfreq_add_file(wf, files.paths[i]) != 0) {
            fprintf(stderr, "warning: could not read %s, skipping\n", files.paths[i]);
            continue;
        }
        if (!quiet) printf("\r    [%d/%d] %s  (%d distinct words so far)          ",
                            i + 1, files.n, files.paths[i], mse_wordfreq_distinct_words(wf));
        fflush(stdout);
    }
    if (!quiet) printf("\n");

    BPETokenizer tok;
    mse_tok_init(&tok, vocab_size);
    mse_tok_train_from_wordfreq(&tok, wf);
    mse_wordfreq_free(wf);
    if (!quiet) printf("  vocabulary: %d tokens (%d merges)  %.2fs\n\n",
                        mse_tok_vocab_size_actual(&tok), tok.n_merges, now_sec() - t0);

    /* ---- Pass 2: graph, in batches, reusing incremental-training merge ---- */
    if (!quiet) printf("  Pass 2/2  building graph in batches of %d file(s)...\n", batch_size);
    MseModel m;
    model_init(&m, vocab_size);
    mse_tok_free(&m.tokenizer);
    m.tokenizer = tok; /* transfer ownership -- matches model.tokenizer = tok */

    int32_t n_batches = (files.n + batch_size - 1) / batch_size;
    double t1 = now_sec();
    for (int32_t bi = 0; bi < n_batches; bi++) {
        int32_t start = bi * batch_size;
        int32_t end = start + batch_size; if (end > files.n) end = files.n;

        i32vec *seqs = NULL; int32_t n_seq = 0, cap_seq = 0;
        for (int32_t fi = start; fi < end; fi++) {
            long len;
            char *text = read_whole_file(files.paths[fi], &len);
            if (!text) { fprintf(stderr, "warning: could not read %s, skipping\n", files.paths[fi]); continue; }
            int32_t n_sent; int32_t *lens;
            char **sents = mse_split_sentences(text, (int32_t)len, &lens, &n_sent);
            for (int32_t si = 0; si < n_sent; si++) {
                if (n_seq == cap_seq) { cap_seq = cap_seq ? cap_seq * 2 : 64; seqs = (i32vec *)realloc(seqs, sizeof(i32vec) * (size_t)cap_seq); }
                i32vec_init(&seqs[n_seq]);
                mse_tok_encode_for_training(&m.tokenizer, sents[si], lens[si], &seqs[n_seq]);
                n_seq++;
            }
            mse_free_sentences(sents, n_sent);
            free(lens);
            free(text);
        }

        if (bi == 0) model_build_graphs(&m, seqs, n_seq);
        else model_merge_graphs(&m, seqs, n_seq);

        for (int32_t si = 0; si < n_seq; si++) i32vec_free(&seqs[si]);
        free(seqs);

        if (!quiet) {
            MseModelStats s = model_stats(&m);
            printf("\r    [batch %d/%d]  edges %d  bridges %d  clusters %d  rels %d  %.1fs      ",
                   bi + 1, n_batches, s.edges, s.bridges, s.clusters, s.relationships, now_sec() - t1);
            fflush(stdout);
        }
    }
    if (!quiet) printf("\n  graph built  %.2fs\n\n", now_sec() - t1);

    if (model_save(&m, out_dir) != 0) { fprintf(stderr, "save failed\n"); return 1; }

    MseModelStats stats = model_stats(&m);
    if (!quiet) {
        printf("  ------------------------------------------------------------\n");
        printf("  Training complete\n\n");
        printf("  Output folder         %s\n", out_dir);
        printf("  Files processed        %d\n", files.n);
        printf("  Vocabulary             %d tokens\n", stats.vocab_size);
        printf("  Edge Matrix             %d unique bigrams\n", stats.edges);
        printf("  Bridge Matrix           %d unique triples\n", stats.bridges);
        printf("  Clustered triples       %d  (%d clusters)\n", stats.clustered_bridges, stats.clusters);
        printf("  Relationship rows       %d  (%d unique sentences)\n", stats.relationship_rows, stats.relationships);
        printf("  Total time              %.2fs\n\n", now_sec() - t0);
    }

    model_free(&m);
    pathlist_free(&files);
    return 0;
}
