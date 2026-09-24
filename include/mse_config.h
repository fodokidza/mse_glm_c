/* mse_config.h — single source of truth for tunable constants, mirroring
 * the Python config.py. Every other header/source file includes this
 * instead of hardcoding a literal, same rule the Python codebase enforces.
 */
#ifndef MSE_CONFIG_H
#define MSE_CONFIG_H

/* ---- reserved / special token ids ---- */
enum {
    TOK_PAD = 0,
    TOK_UNK = 1,
    TOK_BOS = 2,
    TOK_EOS = 3,
    TOK_WORD_BOUND = 4,   /* tokenizer.py's CharWordTokenizer only -- marks
                            * "a new stage-1-spelled word starts here" (see
                            * mse_tokenizer.h). Deliberately NOT in RESERVED,
                            * same treatment as EOS: the model needs to be
                            * able to select it during generation too. */
    TOK_FIRST_FREE = 5    /* first id available for real vocabulary */
};

/* RESERVED = {PAD, UNK, BOS} (EOS and WORD_BOUND are intentionally
 * excluded -- see config.py: neither is excluded from vote layers). */
static inline int mse_is_reserved(int32_t t) {
    return t == TOK_PAD || t == TOK_UNK || t == TOK_BOS;
}

/* STRUCTURAL = RESERVED | {EOS} -- noise.py's own, slightly wider
 * exclusion set (see mse_noise.h): used only for deciding whether a
 * candidate can be scored at all, and for stripping a relationship's
 * reconstructed sequence down to the tokens that count as evidence.
 * WORD_BOUND is deliberately NOT in this set either, same reasoning
 * as above. */
static inline int mse_is_structural(int32_t t) {
    return mse_is_reserved(t) || t == TOK_EOS;
}

/* ---- tokenizer ---- */
#define TOKENIZER_DEFAULT_VOCAB_SIZE        2000
#define TOKENIZER_TRAIN_CLI_DEFAULT_VOCAB   1000
#define TOKENIZER_STREAM_CHUNK_SIZE         (1 << 20)

/* PUNCTUATION = ".,!?;:()\"'-" preserved as its own token(s).
 * NO_SPACE_BEFORE = PUNCTUATION - '(' */
#define MSE_IS_PUNCT(c) \
    ((c) == '.' || (c) == ',' || (c) == '!' || (c) == '?' || (c) == ';' || \
     (c) == ':' || (c) == '(' || (c) == ')' || (c) == '"'  || (c) == '\'' || \
     (c) == '-')
#define MSE_NO_SPACE_BEFORE(c) (MSE_IS_PUNCT(c) && (c) != '(')

/* ---- IVM (V1-V10) default weights — see ivm.py's module docstring.
 * IMPORTANT_WEIGHT/INFLUENCE_WEIGHT/CONTEXT_INFLUENCE_WEIGHT/
 * NOISE_WEIGHT are deliberately smaller than CONTEXT_WEIGHT BY DESIGN;
 * if you retune these, that ratio is a property you need to preserve
 * yourself, not something enforced automatically. */
#define IVM_IMPORTANT_WEIGHT          0.3
#define IVM_INFLUENCE_WEIGHT          0.0
#define IVM_CONTEXT_WEIGHT            1.1
#define IVM_CONTEXT_INFLUENCE_WEIGHT  0.0
#define IVM_BIGRAM_WITNESS_WEIGHT     0.03
#define IVM_ADJACENCY_WEIGHT          0.06
#define IVM_PREV_CURRENT_WEIGHT       1.0
#define IVM_TRIPLE_WEIGHT             1.0
#define IVM_WHOLE_CONTEXT_WEIGHT      1.0
#define IVM_NOISE_WEIGHT              0.00003   /* V10 -- see mse_noise.h */

/* ---- Noise (V10's data source, mse_noise.h/noise.c) ----
 * VOTE_WEIGHT is the multiplier baked into every candidate's
 * three-stage average (see noise_candidate_average()); IVM_NOISE_WEIGHT
 * above is applied ON TOP of that, once, when V10 is summed into the
 * final score -- same two-multiplier chain as ivm.py/noise.py. Unlike
 * the Python original's NoiseConfig.EAGER_BUILD switch, this port has
 * no eager-build mode: the noise index is always attached lazily, once,
 * on the first Open Mode generate() call (see model.c's
 * ensure_noise_layer()) -- there is no "force it into every build"
 * escape hatch here, since nothing here pays for it until Open Mode
 * actually runs. */
#define NOISE_VOTE_WEIGHT 1.0

/* ---- CTM ---- */
#define CTM_MIN_SUPPORT 1

/* ---- Interpret ---- */
#define INTERPRET_MIN_COVERAGE      0.5
#define INTERPRET_MIN_SIGNALS       2
#define INTERPRET_TOP_N             5
#define INTERPRET_MAX_PER_CLUSTER   3
#define INTERPRET_MIN_GROUP_SIZE    2
#define INTERPRET_BUILD_MATRIX_TOPN 50

/* ---- generation ---- */
#define GEN_MAX_TOKENS                 40
#define GEN_CHAT_CLI_DEFAULT_MAX_TOKENS 30
#define GEN_SERVER_DEFAULT_MAX_TOKENS   80

/* ---- train_corpus ---- */
#define TRAIN_CORPUS_DEFAULT_BATCH_SIZE 10

#include <stdint.h>

#endif /* MSE_CONFIG_H */
