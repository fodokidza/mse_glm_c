#include "mse_model.h"
#include "mse_config.h"
#include "mse_importance.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>

void model_init(MseModel *m, int32_t vocab_size) {
    memset(m, 0, sizeof(*m));
    mse_tok_init(&m->tokenizer, vocab_size);
    em_init(&m->edges);
    bm_init(&m->bridges);
    rm_init(&m->rels);
    i32vec_init(&m->open_vocab);
}

void model_free(MseModel *m) {
    mse_tok_free(&m->tokenizer);
    em_free(&m->edges);
    bm_free(&m->bridges);
    rm_free(&m->rels);
    if (m->trained) { /* engines/open_ctm only exist once a graph build has happened */
        ivm_free(&m->open_ctm);
    }
    if (m->has_noise) { noise_index_free(&m->noise); m->has_noise = 0; }
    i32vec_free(&m->open_vocab);
    if (m->has_ctm) ctm_free(&m->ctm);
    if (m->has_ivm) ivm_free(&m->ivm);
    memset(m, 0, sizeof(*m));
}

void model_all_candidate_tokens(const MseModel *m, i32vec *out) {
    int32_t vs = mse_tok_vocab_size_actual(&m->tokenizer);
    for (int32_t t = 0; t < vs; t++) if (!mse_is_reserved(t)) i32vec_push(out, t);
}

static void rebuild_open_engine(MseModel *m) {
    /* V10's data source is derived from the graphs about to be rebuilt
     * below -- drop it FIRST (mirrors model.py's own ordering fix: an
     * earlier revision invalidated noise state AFTER rebuilding the
     * open engine, which left a freshly-rebuilt open_ctm pointing at a
     * NoiseIndex built from the pre-merge structure for one call cycle).
     * Lazily rebuilt again on the next model_ensure_noise_layer() call
     * (i.e. the next Open Mode generate()), not eagerly here — nothing
     * pays for V10 until Open Mode actually runs. */
    if (m->has_noise) { noise_index_free(&m->noise); m->has_noise = 0; }

    if (m->trained) ivm_free(&m->open_ctm);
    i32vec_free(&m->open_vocab); i32vec_init(&m->open_vocab);
    model_all_candidate_tokens(m, &m->open_vocab);

    ie_init(&m->open_engine, &m->edges, &m->bridges, &m->rels, IE_MODE_OPEN,
            m->open_vocab.data, (int32_t)m->open_vocab.len);
    ivm_build(&m->open_ctm, &m->edges, &m->bridges, &m->rels, mse_tok_vocab_size_actual(&m->tokenizer));
}

static void build_graphs_from_seqs(MseModel *m, i32vec *seqs, int32_t n_seq) {
    int32_t vs = mse_tok_vocab_size_actual(&m->tokenizer);
    em_build(&m->edges, seqs, n_seq, vs);
    bm_build(&m->bridges, seqs, n_seq, vs);
    rm_build(&m->rels, seqs, n_seq, &m->bridges);
    ie_init(&m->strict_engine, &m->edges, &m->bridges, &m->rels, IE_MODE_STRICT, NULL, 0);
    m->trained = 1; /* must be set before rebuild_open_engine's ivm_free(&m->open_ctm) guard */
    rebuild_open_engine(m);
}

void model_build_graphs(MseModel *m, i32vec *seqs, int32_t n_seq) { build_graphs_from_seqs(m, seqs, n_seq); }

void model_merge_graphs(MseModel *m, i32vec *new_seqs, int32_t n_new) {
    int32_t vsz = mse_tok_vocab_size_actual(&m->tokenizer);

    /* Reconstruct every existing relationship's literal sentence content
     * FIRST, from the still-intact PRE-merge Bridge/Relationship
     * structure -- this MUST happen before the Bridge Matrix below is
     * overwritten, since sequence_for_relationship() reads it. */
    int32_t n_old_rels = m->rels.n_rels;
    i32vec *old_seqs = (i32vec *)malloc(sizeof(i32vec) * (size_t)(n_old_rels > 0 ? n_old_rels : 1));
    for (int32_t r = 0; r < n_old_rels; r++) {
        i32vec_init(&old_seqs[r]);
        mse_sequence_for_relationship(&m->rels, &m->bridges, r, &old_seqs[r]);
    }

    /* ---- Edge Matrix: union of (src,dst), counts summed ---- */
    {
        StrMap seen; strmap_init(&seen);
        Arena arena; arena_init(&arena);
        i32vec a, b, c; i32vec_init(&a); i32vec_init(&b); i32vec_init(&c);

        for (int32_t i = 0; i < m->edges.n; i++) {
            int64_t key = ((int64_t)(uint32_t)m->edges.src[i] << 32) | (uint32_t)m->edges.dst[i];
            int32_t idx = (int32_t)a.len;
            i32vec_push(&a, m->edges.src[i]); i32vec_push(&b, m->edges.dst[i]); i32vec_push(&c, m->edges.count[i]);
            char *owned = arena_strndup(&arena, (const char *)&key, (int32_t)sizeof(key));
            strmap_put(&seen, owned, (int32_t)sizeof(key), idx);
        }
        for (int32_t s = 0; s < n_new; s++) {
            i32vec *seq = &new_seqs[s];
            for (size_t i = 0; i + 1 < seq->len; i++) {
                int32_t sa = seq->data[i], sb = seq->data[i + 1];
                int64_t key = ((int64_t)(uint32_t)sa << 32) | (uint32_t)sb;
                int32_t idx;
                if (strmap_get(&seen, (const char *)&key, (int32_t)sizeof(key), &idx)) c.data[idx]++;
                else {
                    idx = (int32_t)a.len;
                    i32vec_push(&a, sa); i32vec_push(&b, sb); i32vec_push(&c, 1);
                    char *owned = arena_strndup(&arena, (const char *)&key, (int32_t)sizeof(key));
                    strmap_put(&seen, owned, (int32_t)sizeof(key), idx);
                }
            }
        }
        strmap_free(&seen); arena_free(&arena);
        em_build_from_pairs(&m->edges, &a, &b, &c, vsz);
        i32vec_free(&a); i32vec_free(&b); i32vec_free(&c);
    }

    /* ---- Bridge Matrix: union of (source,target,bridge) triples,
     * cluster_id recomputed from scratch over the merged set ---- */
    {
        StrMap seen; strmap_init(&seen);
        Arena arena; arena_init(&arena);
        i32vec ts, tt, tb; i32vec_init(&ts); i32vec_init(&tt); i32vec_init(&tb);

        for (int32_t i = 0; i < m->bridges.n; i++) {
            struct { int32_t s, t, b; } k = { m->bridges.source[i], m->bridges.target[i], m->bridges.bridge[i] };
            int32_t idx = (int32_t)ts.len;
            i32vec_push(&ts, k.s); i32vec_push(&tt, k.t); i32vec_push(&tb, k.b);
            char *owned = arena_strndup(&arena, (const char *)&k, (int32_t)sizeof(k));
            strmap_put(&seen, owned, (int32_t)sizeof(k), idx);
        }
        for (int32_t s = 0; s < n_new; s++) {
            i32vec *seq = &new_seqs[s];
            for (size_t i = 0; i + 2 < seq->len; i++) {
                struct { int32_t s, t, b; } k = { seq->data[i], seq->data[i + 2], seq->data[i + 1] };
                int32_t idx;
                if (!strmap_get(&seen, (const char *)&k, (int32_t)sizeof(k), &idx)) {
                    idx = (int32_t)ts.len;
                    i32vec_push(&ts, k.s); i32vec_push(&tt, k.t); i32vec_push(&tb, k.b);
                    char *owned = arena_strndup(&arena, (const char *)&k, (int32_t)sizeof(k));
                    strmap_put(&seen, owned, (int32_t)sizeof(k), idx);
                }
            }
        }
        strmap_free(&seen); arena_free(&arena);
        bm_build_from_triples(&m->bridges, &ts, &tt, &tb, vsz); /* replaces m->bridges */
        i32vec_free(&ts); i32vec_free(&tt); i32vec_free(&tb);
    }

    /* ---- Relationship Matrix: rebuild from scratch over
     * old_seqs + new_seqs against the NEW (post-merge) Bridge Matrix. */
    {
        int32_t n_total = n_old_rels + n_new;
        i32vec *all_seqs = (i32vec *)malloc(sizeof(i32vec) * (size_t)(n_total > 0 ? n_total : 1));
        for (int32_t i = 0; i < n_old_rels; i++) all_seqs[i] = old_seqs[i]; /* transfer ownership */
        for (int32_t i = 0; i < n_new; i++) {
            i32vec_init(&all_seqs[n_old_rels + i]);
            for (size_t k = 0; k < new_seqs[i].len; k++) i32vec_push(&all_seqs[n_old_rels + i], new_seqs[i].data[k]);
        }
        rm_build(&m->rels, all_seqs, n_total, &m->bridges);
        for (int32_t i = 0; i < n_total; i++) i32vec_free(&all_seqs[i]);
        free(all_seqs);
    }
    free(old_seqs);

    ie_init(&m->strict_engine, &m->edges, &m->bridges, &m->rels, IE_MODE_STRICT, NULL, 0);
    rebuild_open_engine(m);

    /* CTM/legacy-IVM are opt-in add-ons derived from the pre-merge
     * structure -- invalidated, not auto-rebuilt (matches model.py). */
    if (m->has_ctm) { ctm_free(&m->ctm); m->has_ctm = 0; }
    if (m->has_ivm) { ivm_free(&m->ivm); m->has_ivm = 0; }
}

static void encode_all_sentences(MseTokenizer *tok, const char *text, int32_t len, i32vec **out_seqs, int32_t *out_n);

MseTrainIncrementalResult model_train_incremental(MseModel *m, const char *corpus, int32_t len,
                                                    int extend_vocab, int32_t target_vocab_size) {
    MseTrainIncrementalResult r;
    r.before = model_stats(m);
    r.vocab_added = 0;
    if (extend_vocab) r.vocab_added = mse_tok_extend_vocab(&m->tokenizer, corpus, len, target_vocab_size);

    i32vec *seqs; int32_t n_seq;
    encode_all_sentences(&m->tokenizer, corpus, len, &seqs, &n_seq);
    int had_ctm = m->has_ctm;
    model_merge_graphs(m, seqs, n_seq);
    for (int32_t i = 0; i < n_seq; i++) i32vec_free(&seqs[i]);
    free(seqs);

    r.sentences_added = n_seq;
    r.after = model_stats(m);
    r.ctm_invalidated = had_ctm;
    return r;
}

static void encode_all_sentences(MseTokenizer *tok, const char *text, int32_t len, i32vec **out_seqs, int32_t *out_n) {
    int32_t n_sent; int32_t *lens;
    char **sents = mse_split_sentences(text, len, &lens, &n_sent);
    i32vec *seqs = (i32vec *)malloc(sizeof(i32vec) * (size_t)(n_sent ? n_sent : 1));
    for (int32_t i = 0; i < n_sent; i++) {
        i32vec_init(&seqs[i]);
        mse_tok_encode_for_training(tok, sents[i], lens[i], &seqs[i]);
    }
    mse_free_sentences(sents, n_sent);
    free(lens);
    *out_seqs = seqs; *out_n = n_sent;
}

void model_train(MseModel *m, const char *corpus, int32_t len) {
    mse_tok_train(&m->tokenizer, corpus, len);
    i32vec *seqs; int32_t n_seq;
    encode_all_sentences(&m->tokenizer, corpus, len, &seqs, &n_seq);
    build_graphs_from_seqs(m, seqs, n_seq);
    for (int32_t i = 0; i < n_seq; i++) i32vec_free(&seqs[i]);
    free(seqs);
}

int model_train_from_file(MseModel *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return -1; }
    char *buf = (char *)malloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = '\0';
    model_train(m, buf, (int32_t)rd);
    free(buf);
    return 0;
}

MseModelStats model_stats(const MseModel *m) {
    MseModelStats s;
    s.vocab_size = mse_tok_vocab_size_actual(&m->tokenizer);
    s.edges = m->edges.n;
    s.bridges = m->bridges.n;
    s.clustered_bridges = 0;
    for (int32_t i = 0; i < m->bridges.n; i++) if (m->bridges.cluster_id[i] != 0) s.clustered_bridges++;
    s.clusters = m->bridges.max_cluster_id > 0 ? m->bridges.max_cluster_id : 0;
    s.relationships = m->rels.n_rels;
    s.relationship_rows = m->rels.n_rows;
    s.relationship_occurrences = 0;
    for (int32_t i = 0; i < m->rels.n_rels; i++) s.relationship_occurrences += m->rels.rel_count[i];
    return s;
}

void model_build_context_triggers(MseModel *m, int32_t min_support) {
    if (m->has_ctm) ctm_free(&m->ctm);
    ctm_build(&m->ctm, &m->bridges, &m->rels, &m->open_ctm.token_rels, min_support);
    m->has_ctm = 1;
}

void model_build_importance_votes(MseModel *m) {
    if (m->has_ivm) ivm_free(&m->ivm);
    ivm_build(&m->ivm, &m->edges, &m->bridges, &m->rels, mse_tok_vocab_size_actual(&m->tokenizer));
    m->has_ivm = 1;
}

void model_ensure_noise_layer(MseModel *m) {
    if (m->has_noise && m->open_ctm.noise == &m->noise) return; /* already attached */
    if (m->has_noise) { noise_index_free(&m->noise); m->has_noise = 0; }
    noise_index_build(&m->noise, &m->rels, &m->bridges, &m->open_ctm.token_rels,
                       mse_tok_vocab_size_actual(&m->tokenizer), NOISE_VOTE_WEIGHT);
    m->has_noise = 1;
    m->open_ctm.noise = &m->noise;
    m->open_ctm.noise_weight = IVM_NOISE_WEIGHT;
}

char *model_generate(MseModel *m, const char *prompt, int32_t max_tokens,
                      IeMode mode, int use_ctm, int use_ivm, i32vec *out_ids) {
    i32vec ids; i32vec_init(&ids);
    mse_tok_encode(&m->tokenizer, prompt, (int32_t)strlen(prompt), &ids);

    if (mode == IE_MODE_OPEN) {
        model_ensure_noise_layer(m); /* V10 -- see mse_model.h */
        ie_generate(&m->open_engine, &ids, max_tokens, NULL, &m->open_ctm);
    } else {
        ContextTriggerMatrix *ctm = (use_ctm && m->has_ctm) ? &m->ctm : NULL;
        ImportanceVoteMatrix *ivm = (use_ivm && m->has_ivm) ? &m->ivm : NULL;
        ie_generate(&m->strict_engine, &ids, max_tokens, ctm, ivm);
    }

    char *decoded = mse_tok_decode(&m->tokenizer, ids.data, (int32_t)ids.len);
    if (out_ids) { for (size_t i = 0; i < ids.len; i++) i32vec_push(out_ids, ids.data[i]); }
    i32vec_free(&ids);
    return decoded;
}

int model_save(const MseModel *m, const char *folder) {
    char path[4096];
    mkdir(folder, 0755); /* ignore EEXIST -- best effort, single-level like os.makedirs for our use */
    snprintf(path, sizeof(path), "%s/tokenizer.tok", folder);
    if (mse_tok_save(&m->tokenizer, path) != 0) return -1;
    snprintf(path, sizeof(path), "%s/edges.bin", folder);
    if (em_save(&m->edges, path) != 0) return -1;
    snprintf(path, sizeof(path), "%s/bridges.bin", folder);
    if (bm_save(&m->bridges, path) != 0) return -1;
    snprintf(path, sizeof(path), "%s/relationships.bin", folder);
    if (rm_save(&m->rels, path) != 0) return -1;
    return 0;
}

int model_load(MseModel *m, const char *folder) {
    char path[4096];
    model_free(m);
    model_init(m, TOKENIZER_DEFAULT_VOCAB_SIZE);

    snprintf(path, sizeof(path), "%s/tokenizer.tok", folder);
    mse_tok_free(&m->tokenizer);
    if (mse_tok_load(&m->tokenizer, path) != 0) return -1;

    snprintf(path, sizeof(path), "%s/edges.bin", folder);
    if (em_load(&m->edges, path) != 0) return -1;
    snprintf(path, sizeof(path), "%s/bridges.bin", folder);
    if (bm_load(&m->bridges, path) != 0) return -1;
    snprintf(path, sizeof(path), "%s/relationships.bin", folder);
    if (rm_load(&m->rels, path) != 0) return -1;

    ie_init(&m->strict_engine, &m->edges, &m->bridges, &m->rels, IE_MODE_STRICT, NULL, 0);
    m->trained = 1;
    rebuild_open_engine(m);
    return 0;
}
