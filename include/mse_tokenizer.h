/* mse_tokenizer.h — from-scratch BPE tokenizer, C port of tokenizer.py.
 *
 * Design note (why this isn't a literal line-by-line translation):
 * normalize() in Python funnels every input character down to
 * [a-z0-9 <punct>] (anything else, including all non-ASCII, is stripped
 * by _KEEP_CHARS_RE before anything else happens). That means the
 * *output* of normalize() is always a pure-ASCII byte string, so this
 * port scans input as raw bytes and never needs a UTF-8 decoder: any
 * byte >= 0x80 (a UTF-8 continuation/lead byte) is unconditionally
 * "not a kept character" and collapses to a space, byte for byte, with
 * no multi-byte decode step — exactly the same net result as decoding
 * codepoints and rejecting non-ASCII ones, but branch-free and O(n)
 * over bytes instead of over decoded codepoints.
 *
 * The vocabulary (token_to_id / id_to_token) is an arena-backed
 * StrMap instead of two Python dicts, and BPE merges are stored as
 * pairs of *vocabulary ids* rather than pairs of strings: every symbol
 * that can ever appear in a merge rule was already assigned a vocab id
 * at the moment it was created (base character or earlier merge
 * result), so id-pair comparison during both training and encode()
 * replaces all string concatenation/comparison with integer ops.
 */
#ifndef MSE_TOKENIZER_H
#define MSE_TOKENIZER_H

#include "mse_util.h"
#include "mse_config.h"

/* ------------------------------------------------------------- Vocab */
typedef struct {
    StrMap   str_to_id;
    Arena    arena;
    char   **id_to_str;   /* id -> NUL-terminated string (arena-owned) */
    int32_t *id_to_len;
    int32_t  count;        /* number of ids assigned (== next free id) */
    int32_t  cap;
} Vocab;

void    vocab_init(Vocab *v);
void    vocab_free(Vocab *v);
/* Returns existing id if `s` is already present, else assigns the next
 * sequential id (Python: `next_id`) and returns it. */
int32_t vocab_get_or_create(Vocab *v, const char *s, int32_t len);
int     vocab_lookup(const Vocab *v, const char *s, int32_t len, int32_t *out);
const char *vocab_str(const Vocab *v, int32_t id, int32_t *len_out);

/* --------------------------------------------------------- normalize */
/* Returns a newly malloc'd, NUL-terminated normalized string; caller
 * frees it. *out_len receives the length (excl. NUL). */
char *mse_normalize(const char *text, int32_t len, int32_t *out_len);

/* Splits `text` into sentences per _SENT_SPLIT_RE + _finalize_sentences.
 * Returns a malloc'd array of malloc'd, NUL-terminated sentence strings
 * (caller frees each string, then the array); *out_n receives the count. */
char **mse_split_sentences(const char *text, int32_t len, int32_t **out_lens, int32_t *out_n);
void   mse_free_sentences(char **sents, int32_t n);

/* -------------------------------------------------------- BPETokenizer */
/* vocab ids, applied in order; `merged` is precomputed at training
 * time (the id that (a,b) resolves to) so encode-time merge
 * application is pure integer id matching — no string concatenation
 * or re-lookup on the hot path, unlike the Python original which
 * rebuilds the concatenated string on every application. */
typedef struct { int32_t a, b, merged; } MseMerge;

typedef struct {
    int32_t   vocab_size;      /* target */
    Vocab     vocab;
    MseMerge *merges;
    int32_t   n_merges, merges_cap;

    /* word (string) -> cached id sequence, memoizing _ids_for_word */
    StrMap    cache_index;     /* word bytes -> index into cache_vecs */
    Arena     cache_arena;
    i32vec   *cache_vecs;
    int32_t   cache_count, cache_cap;
} BPETokenizer;

void mse_tok_init(BPETokenizer *t, int32_t vocab_size);
void mse_tok_free(BPETokenizer *t);

void mse_tok_train(BPETokenizer *t, const char *corpus, int32_t len);
int  mse_tok_train_from_file(BPETokenizer *t, const char *path); /* 0 ok, -1 io error */

/* Grows the vocabulary from a new corpus without touching any existing
 * id. Returns number of new vocab entries added. */
int32_t mse_tok_extend_vocab(BPETokenizer *t, const char *corpus, int32_t len,
                              int32_t target_vocab_size);

/* ------------------------------------------------------- MseWordFreq
 * Opaque word-frequency accumulator, exposed for large-corpus training
 * (see tools/mse_train_corpus.c): lets a caller build up one shared
 * word-frequency table across MANY files/chunks -- each added one at a
 * time, so only the current chunk's raw text and the (much smaller)
 * cumulative word-count table are ever in memory together, never the
 * whole corpus's raw text at once -- then train a single tokenizer
 * from the combined counts. This is the same underlying accumulator
 * mse_tok_train() builds internally for a single corpus string;
 * exposing it lets a caller feed it many corpus pieces instead.
 */
typedef struct MseWordFreq MseWordFreq;
MseWordFreq *mse_wordfreq_create(void);
void         mse_wordfreq_free(MseWordFreq *wf);
/* Splits `text` into sentences, normalizes, and accumulates word
 * counts -- exactly the first half of what mse_tok_train() does. */
void mse_wordfreq_add_text(MseWordFreq *wf, const char *text, int32_t len);
/* Reads `path` fully into memory, adds it, frees the buffer. 0 ok, -1 io error. */
int  mse_wordfreq_add_file(MseWordFreq *wf, const char *path);
int32_t mse_wordfreq_distinct_words(const MseWordFreq *wf);
/* Trains `t`'s vocabulary from the accumulated counts -- the second
 * half of mse_tok_train(), operating on a pre-built MseWordFreq
 * instead of a single corpus string. */
void mse_tok_train_from_wordfreq(BPETokenizer *t, MseWordFreq *wf);

/* encode(): appends BOS + token ids for `text` to `out` (caller-owned,
 * pre-inited i32vec). */
void mse_tok_encode(BPETokenizer *t, const char *text, int32_t len, i32vec *out);
/* encode_for_training(): encode() + trailing EOS. */
void mse_tok_encode_for_training(BPETokenizer *t, const char *text, int32_t len, i32vec *out);

/* decode(): returns a newly malloc'd, NUL-terminated string. */
char *mse_tok_decode(BPETokenizer *t, const int32_t *ids, int32_t n);

int32_t mse_tok_vocab_size_actual(const BPETokenizer *t);

/* Binary persistence (see tools/mse_format.h for the on-disk layout). */
int mse_tok_save(const BPETokenizer *t, const char *path);   /* 0 ok */
int mse_tok_load(BPETokenizer *t, const char *path);         /* 0 ok */

#endif /* MSE_TOKENIZER_H */
