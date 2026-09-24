/* mse_tokenizer.h — C port of tokenizer.py's CharWordTokenizer: a
 * from-scratch TWO-STAGE character/word tokenizer. NOT BPE — an
 * earlier BPE port existed here and has been deleted outright, not
 * merely replaced, matching the Python original (see its module
 * docstring: BPE's merge-pair scan was, on the corpus it was measured
 * against, the single slowest step in training).
 *
 *   STAGE 1 (CharVocab)  — every unique ASCII character seen across
 *     the corpus gets its own id, assigned in ascending byte order.
 *     Case-sensitive. A complete tokenizer by itself, if verbose.
 *   STAGE 2 (WordVocab)  — every multi-character word worth having an
 *     id, capped by vocab_size, assigned most-frequent-first (ties
 *     broken alphabetically). Built strictly on top of stage 1: a
 *     word's stored "recipe" is the sequence of stage-1 character ids
 *     that spells it. CASE-INSENSITIVE — "The"/"the"/"THE" share one
 *     entry, decoding it always gives back the stored lowercase form.
 *
 * ONE SHARED ID SPACE for both stages, right after the reserved
 * specials (PAD/UNK/BOS/EOS/WORD_BOUND) — every id anywhere in an
 * encoded stream means one thing to decode(), whichever stage minted
 * it. TRUE ZERO VOCABULARY LOSS: a word stage 2 has no entry for is
 * spelled out live through stage 1, WORD_BOUND-prefixed so a flat id
 * stream can tell where a fallback-spelled word starts — <UNK> is
 * only ever reached for a genuinely unseen *character*.
 *
 * Design departures from a literal translation (same reasoning as
 * before — see git history / VALIDATION.md for the BPE-era notes on
 * why normalize() never decodes UTF-8; that byte-oriented approach
 * carries over unchanged):
 *   - Stage 1's alphabet is provably ASCII-only (normalize()/segment()
 *     both funnel everything else to a space before either stage ever
 *     sees it — see mse_normalize()/mse_segment() below), so CharVocab
 *     is a flat 256-entry array indexed by raw byte value, not a
 *     StrMap — O(1) lookup either direction, no hashing, no arena.
 *   - A word's stage-1 "recipe" (Python's id_to_chars) is used only
 *     transiently, while building a new WordVocab entry's canonical
 *     decoded string — nothing outside tokenizer.c ever reads it in
 *     the Python original either (confirmed: only chars.vocab_size /
 *     chars.is_char_id / decode_id / words.vocab_size / decode_id are
 *     touched elsewhere), so it is never stored, never persisted.
 *   - WordVocab's string->id lookup is case-insensitive by construction:
 *     every key inserted is already lowercased, and every lookup
 *     lowercases its query first — no separate fold step + StrMap
 *     wrapper pair.
 */
#ifndef MSE_TOKENIZER_H
#define MSE_TOKENIZER_H

#include "mse_util.h"
#include "mse_config.h"

/* ------------------------------------------------------------ CharVocab
 * Stage 1. char_to_id[256]: -1 (MSE_TOK_NO_ID) until that raw byte has
 * been learned; id_to_char[id] is only ever populated for real
 * (non-special) ids returned via that array. Both arrays are fixed
 * size — no growth, no allocation beyond the struct itself. */
#define MSE_TOK_NO_ID (-1)
#define MSE_TOK_CHAR_ID_CAP 512  /* generous fixed cap: at most 256
                                    distinct bytes can ever be learned,
                                    plus the 5 reserved specials */

typedef struct {
    int32_t char_to_id[256];
    int32_t id_to_char[MSE_TOK_CHAR_ID_CAP];  /* indexed by id; -1 (MSE_TOK_NO_ID)
                                                  unless that id is a real char */
    int32_t max_id;           /* highest id ever handed out (>= TOK_WORD_BOUND) */
} CharVocab;

void    cv_init(CharVocab *cv);
/* Learns every byte across `bytes[0..n)` not already known, assigning
 * ids in ascending byte-value order starting at
 * max(cv->max_id, min_next_id - 1) + 1 — mirrors CharacterVocabulary
 * .build()'s min_next_id contract (see mse_tokenizer.h's module
 * docstring / tokenizer.py): pass the tokenizer's current shared
 * next-free-id on every call after the very first. `seen256` is a
 * caller-provided 256-byte array, seen256[b] != 0 iff byte b appears
 * in this call's input. */
void    cv_build(CharVocab *cv, const uint8_t *seen256, int32_t min_next_id);
int32_t cv_encode_char(const CharVocab *cv, uint8_t byte); /* -> id, or TOK_UNK */
/* One id -> its byte, or 0 if `id` is special/unknown (caller checks
 * cv_is_char_id first when that distinction matters). */
uint8_t cv_decode_id(const CharVocab *cv, int32_t id);
int     cv_is_char_id(const CharVocab *cv, int32_t id);

/* --------------------------------------------------------- WordVocab
 * Stage 2. token_to_id: lowercase word bytes -> id (StrMap). id info
 * is stored in parallel arrays indexed by (id - TOK_FIRST_FREE) —
 * words.count entries, growable. Only ever holds words of 2+ chars. */
typedef struct {
    StrMap    token_to_id;    /* lowercase word -> id */
    Arena     arena;          /* owns every id_to_token string */
    char    **id_to_token;    /* [i] = word for id (TOK_FIRST_FREE + i); growable */
    int32_t  *id_to_token_len;
    int32_t   count;          /* number of stage-2 entries */
    int32_t   cap;
} WordVocab;

void    wv_init(WordVocab *wv);
void    wv_free(WordVocab *wv);
/* Learns every word of 2+ characters in the (already lowercase-folded
 * and frequency-merged) `words`/`lens`/`freqs`/n arrays not already
 * known, assigning ids most-frequent-first (ties broken by byte-order
 * on the word itself) starting at max(cv->max_id, current max word
 * id) + 1. Mirrors TokenVocabulary.build(); see mse_tokenizer.h's
 * module docstring for why case-folding happens before this call, not
 * inside it. `cv` must already know every character these words are
 * spelled from. */
void    wv_build(WordVocab *wv, CharVocab *cv, char **words, int32_t *lens,
                  int32_t *freqs, int32_t n);
int     wv_lookup_lower(const WordVocab *wv, const char *word, int32_t len, int32_t *out_id);
/* One id -> its stored (lowercase) word string, or "" for a
 * special/unknown id. */
const char *wv_decode_id(const WordVocab *wv, int32_t id, int32_t *len_out);
int32_t wv_max_id(const WordVocab *wv); /* max id ever handed out, or 4 (TOK_WORD_BOUND) if none yet */

/* ------------------------------------------------------------ normalize
 * Case-FOLDING word segmentation into one space-joined string —
 * lowercase, drop anything outside [a-z0-9<punct>], isolate
 * punctuation (apostrophe-in-contraction excepted), collapse
 * whitespace. Returns a newly malloc'd NUL-terminated string; caller
 * frees. Kept for parity with tokenizer.py's public normalize(); this
 * tokenizer's own word segmentation is mse_segment() below (stage 1 is
 * explicitly case-sensitive — see module docstring). */
char *mse_normalize(const char *text, int32_t len, int32_t *out_len);

/* Case-PRESERVING word segmentation for stage 2 — the exact same rule
 * as mse_normalize() (isolate punctuation, collapse whitespace) minus
 * the lowercasing step. Returns a malloc'd array of malloc'd,
 * NUL-terminated word strings for ONE sentence's worth of text
 * (callers segment a whole corpus sentence-by-sentence via
 * mse_split_sentences() first, matching segment()'s own contract);
 * *out_n receives the count. Caller frees each string then the array
 * (mse_free_sentences() works for this too — same ownership shape). */
char **mse_segment(const char *text, int32_t len, int32_t **out_lens, int32_t *out_n);

/* Splits `text` into sentences per _SENT_SPLIT_RE + _finalize_sentences
 * (unchanged from the BPE-era port — this rule never depended on which
 * vocabulary algorithm consumes its output). Returns a malloc'd array
 * of malloc'd, NUL-terminated sentence strings (caller frees each
 * string, then the array); *out_n receives the count. */
char **mse_split_sentences(const char *text, int32_t len, int32_t **out_lens, int32_t *out_n);
void   mse_free_sentences(char **sents, int32_t n);

/* -------------------------------------------------------- MseTokenizer
 * Top-level tokenizer combining both stages. vocab_size is stage 2's
 * target CAP (distinct from vocab_size_actual — see
 * mse_tok_vocab_size_actual()'s doc comment: stage 1's character ids
 * occupy real id-space too, so the actual upper bound on any id
 * encode() can produce is always >= stage 2's own entry count). */
typedef struct {
    int32_t    vocab_size;      /* target cap for stage 2 */
    CharVocab  chars;
    WordVocab  words;

    /* word (string, case-preserved) -> cached id sequence, memoizing
     * _encode_word() exactly like the BPE-era port's ids_for_word()
     * cache did — a repeat word in the same or a later call skips
     * re-deriving its id list. Invalidated (cleared) by
     * mse_tok_extend_vocab() since new stage-2 entries can change how
     * an already-cached word resolves (see that function's doc). */
    StrMap     cache_index;
    Arena      cache_arena;
    i32vec    *cache_vecs;
    int32_t    cache_count, cache_cap;
} MseTokenizer;

void mse_tok_init(MseTokenizer *t, int32_t vocab_size);
void mse_tok_free(MseTokenizer *t);

void mse_tok_train(MseTokenizer *t, const char *corpus, int32_t len);
int  mse_tok_train_from_file(MseTokenizer *t, const char *path); /* 0 ok, -1 io error */

/* Grows stage 2's pre-defined-word budget from a NEW corpus without
 * touching any existing character or word id — every id already
 * assigned keeps that exact id, so every Edge/Bridge/Relationship
 * triple built under the old vocabulary stays valid. Returns the
 * number of new stage-2 entries actually added (0 if
 * target_vocab_size doesn't exceed the budget already in use, or the
 * new corpus has no new multi-char words left to pre-define). */
int32_t mse_tok_extend_vocab(MseTokenizer *t, const char *corpus, int32_t len,
                              int32_t target_vocab_size);

/* ------------------------------------------------------- MseWordFreq
 * Opaque word-frequency accumulator, exposed for large-corpus training
 * (see tools/mse_train_corpus.c): lets a caller build up one shared,
 * CASE-PRESERVING word-frequency table across MANY files/chunks —
 * each added one at a time, so only the current chunk's raw text and
 * the (much smaller) cumulative word-count table are ever in memory
 * together, never the whole corpus's raw text at once — then train a
 * single tokenizer from the combined counts. Case-preserving because
 * this is what stage 2's build() itself lowercase-folds internally
 * (see wv_build()) — the same accumulator mse_tok_train() builds
 * internally for a single corpus string; exposing it lets a caller
 * feed it many corpus pieces instead.
 */
typedef struct MseWordFreq MseWordFreq;
MseWordFreq *mse_wordfreq_create(void);
void         mse_wordfreq_free(MseWordFreq *wf);
/* Splits `text` into sentences, segments (case-preserving), and
 * accumulates word counts — exactly the first half of what
 * mse_tok_train() does. */
void mse_wordfreq_add_text(MseWordFreq *wf, const char *text, int32_t len);
/* Reads `path` fully into memory, adds it, frees the buffer. 0 ok, -1 io error. */
int  mse_wordfreq_add_file(MseWordFreq *wf, const char *path);
int32_t mse_wordfreq_distinct_words(const MseWordFreq *wf);
/* Trains `t`'s vocabulary (both stages) from the accumulated counts —
 * the second half of mse_tok_train(), operating on a pre-built
 * MseWordFreq instead of a single corpus string. */
void mse_tok_train_from_wordfreq(MseTokenizer *t, MseWordFreq *wf);

/* encode(): appends BOS + token ids for `text` to `out` (caller-owned,
 * pre-inited i32vec). Falls back to WORD_BOUND + stage-1 spelling for
 * any word stage 2 has no entry for — see module docstring. */
void mse_tok_encode(MseTokenizer *t, const char *text, int32_t len, i32vec *out);
/* encode_for_training(): encode() + trailing EOS. */
void mse_tok_encode_for_training(MseTokenizer *t, const char *text, int32_t len, i32vec *out);

/* decode(): returns a newly malloc'd, NUL-terminated string. A
 * stage-2 word id always decodes to its ONE stored lowercase spelling
 * regardless of this occurrence's original casing; a fallback-spelled
 * (WORD_BOUND-prefixed) run of stage-1 ids round-trips its exact
 * original casing. */
char *mse_tok_decode(MseTokenizer *t, const int32_t *ids, int32_t n);

/* max(every id actually assigned across BOTH stages) + 1 — graph.c
 * sizes its CSR index arrays from this number and then indexes them
 * with real token ids straight out of encode()'s output, so this MUST
 * be an upper bound on every id that can appear in a sequence, not
 * just stage 2's entry count (stage 1's character ids occupy real
 * id-space too — see module docstring's "ONE SHARED ID SPACE"). */
int32_t mse_tok_vocab_size_actual(const MseTokenizer *t);

/* Binary persistence (see tools/mse_format.h for the on-disk layout). */
int mse_tok_save(const MseTokenizer *t, const char *path);   /* 0 ok */
int mse_tok_load(MseTokenizer *t, const char *path);         /* 0 ok */

#endif /* MSE_TOKENIZER_H */
