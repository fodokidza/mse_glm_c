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
    TOK_FIRST_FREE = 4   /* first id available for real vocabulary */
};

/* RESERVED = {PAD, UNK, BOS} (EOS is intentionally excluded — see
 * config.py: it's the training-time terminator, not excluded from
 * vote layers). */
static inline int mse_is_reserved(int32_t t) {
    return t == TOK_PAD || t == TOK_UNK || t == TOK_BOS;
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

/* ---- IVM (V1-V9) default weights — see ivm.py's module docstring */
#define IVM_IMPORTANT_WEIGHT          0.4
#define IVM_INFLUENCE_WEIGHT          0.0003
#define IVM_CONTEXT_WEIGHT            1.0
#define IVM_CONTEXT_INFLUENCE_WEIGHT  0.0003
#define IVM_BIGRAM_WITNESS_WEIGHT     0.7
#define IVM_ADJACENCY_WEIGHT          0.6
#define IVM_PREV_CURRENT_WEIGHT       1.7
#define IVM_TRIPLE_WEIGHT             2.0
#define IVM_WHOLE_CONTEXT_WEIGHT      2.5

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
