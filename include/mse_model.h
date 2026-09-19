/* mse_model.h — C port of model.py's MSEGraphLanguageModel.
 *
 * This layer is almost entirely assembly: it owns one of each engine
 * component already ported (tokenizer, three graph matrices, Strict
 * and Open Mode InferenceEngines, Open Mode's always-on IVM, and the
 * two opt-in add-ons CTM/legacy-IVM) and wires them together exactly
 * as model.py does. The only genuinely new logic here is
 * train_incremental()'s graph-merge, which is deferred for now (see
 * README) — everything else is direct reuse of Phase 1-3.
 */
#ifndef MSE_MODEL_H
#define MSE_MODEL_H

#include "mse_tokenizer.h"
#include "mse_graph.h"
#include "mse_ctm.h"
#include "mse_ivm.h"
#include "mse_inference.h"

typedef struct {
    int32_t vocab_size, edges, bridges, clustered_bridges, clusters;
    int32_t relationships, relationship_rows;
    int64_t relationship_occurrences;
} MseModelStats;

typedef struct {
    BPETokenizer        tokenizer;
    EdgeMatrix          edges;
    BridgeMatrix        bridges;
    RelationshipMatrix  rels;

    InferenceEngine strict_engine;
    InferenceEngine open_engine;
    i32vec          open_vocab;   /* all_candidate_tokens(): everything except PAD/UNK/BOS */
    ImportanceVoteMatrix open_ctm; /* Open Mode's PRIMARY mechanism, always built alongside training */

    ContextTriggerMatrix ctm;  int has_ctm; /* Strict Mode opt-in add-on */
    ImportanceVoteMatrix ivm;  int has_ivm; /* Strict Mode legacy opt-in tie-break */

    int trained; /* 0 until train()/train_from_file()/load() succeeds */
} MseModel;

void model_init(MseModel *m, int32_t vocab_size);
void model_free(MseModel *m);

void model_train(MseModel *m, const char *corpus, int32_t len);
int  model_train_from_file(MseModel *m, const char *path); /* 0 ok, -1 io error */

/* Every vocab id except PAD/UNK/BOS (EOS is deliberately included —
 * see model.py's all_candidate_tokens docstring). Appended to `out`. */
void model_all_candidate_tokens(const MseModel *m, i32vec *out);

MseModelStats model_stats(const MseModel *m);

/* Opt-in add-ons -- neither is built automatically. */
void model_build_context_triggers(MseModel *m, int32_t min_support);
void model_build_importance_votes(MseModel *m);

/* Rebuilds Edge/Bridge/Relationship as the union of the model's current
 * graphs and the structural facts in `new_seqs` (already-encoded token
 * sequences), with Bridge Matrix clusters recomputed from scratch over
 * the merged set. Open Mode (open_engine/open_ctm) is rebuilt
 * automatically; CTM/legacy-IVM opt-ins are invalidated (freed), not
 * rebuilt -- call model_build_context_triggers()/model_build_importance_votes()
 * again if you need them. See model.c's comment for the one documented,
 * provably-harmless ordering divergence from Python's set-based merge. */
/* Builds Edge/Bridge/Relationship from scratch over `seqs` (already-
 * encoded token sequences) -- the from-scratch counterpart to
 * model_merge_graphs(). Exposed publicly for tools/mse_train_corpus.c's
 * first batch (every batch after that uses model_merge_graphs()). */
void model_build_graphs(MseModel *m, i32vec *seqs, int32_t n_seq);

void model_merge_graphs(MseModel *m, i32vec *new_seqs, int32_t n_new);

typedef struct {
    int32_t sentences_added, vocab_added;
    MseModelStats before, after;
    int ctm_invalidated;
} MseTrainIncrementalResult;

/* Encodes `corpus`'s sentences with the model's EXISTING tokenizer
 * (optionally extending its vocabulary first, matching model.py's
 * train_incremental(extend_vocab=..., target_vocab_size=...)) and
 * merges them in via model_merge_graphs(). Requires an already-trained
 * or loaded model. */
MseTrainIncrementalResult model_train_incremental(MseModel *m, const char *corpus, int32_t len,
                                                    int extend_vocab, int32_t target_vocab_size);

/* generate(): encodes `prompt`, runs InferenceEngine.generate() in the
 * requested mode, and decodes the result. `use_ctm`/`use_ivm` are
 * Strict-Mode-only opt-ins (matching model.py; ignored in Open Mode,
 * which always uses open_ctm as ivm.h/inference.h's IVM). Returns a
 * newly malloc'd decoded string (caller frees); *out_ids, if non-NULL,
 * receives the raw generated token ids (caller-inited i32vec). */
char *model_generate(MseModel *m, const char *prompt, int32_t max_tokens,
                      IeMode mode, int use_ctm, int use_ivm, i32vec *out_ids);

/* Binary persistence: `folder` gets tokenizer.tok/edges.bin/bridges.bin/
 * relationships.bin (not JSON — see mse_format.h). Rebuilds both
 * InferenceEngines and open_ctm after a load, matching model.py's
 * load() classmethod. */
int model_save(const MseModel *m, const char *folder); /* 0 ok */
int model_load(MseModel *m, const char *folder);        /* 0 ok */

#endif /* MSE_MODEL_H */
