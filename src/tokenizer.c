/* tokenizer.c — C port of tokenizer.py's CharWordTokenizer (two-stage
 * character/word design). See mse_tokenizer.h for the design notes on
 * why this isn't a literal translation. BPE (and everything BPE-only
 * — merges, PairMap-driven training) has been deleted outright, not
 * kept alongside this.
 */
#include "mse_tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

/* ============================================================ CharVocab */
void cv_init(CharVocab *cv) {
    for (int i = 0; i < 256; i++) cv->char_to_id[i] = MSE_TOK_NO_ID;
    for (int i = 0; i < MSE_TOK_CHAR_ID_CAP; i++) cv->id_to_char[i] = MSE_TOK_NO_ID;
    cv->max_id = TOK_WORD_BOUND; /* == 4; mirrors Python's SPECIAL_TOKENS max */
}

void cv_build(CharVocab *cv, const uint8_t *seen256, int32_t min_next_id) {
    int32_t next_id = (cv->max_id > min_next_id - 1 ? cv->max_id : min_next_id - 1) + 1;
    for (int b = 0; b < 256; b++) {
        if (!seen256[b]) continue;
        if (cv->char_to_id[b] != MSE_TOK_NO_ID) continue;
        cv->char_to_id[b] = next_id;
        if (next_id < MSE_TOK_CHAR_ID_CAP) cv->id_to_char[next_id] = b;
        if (next_id > cv->max_id) cv->max_id = next_id;
        next_id++;
    }
}

int32_t cv_encode_char(const CharVocab *cv, uint8_t byte) {
    int32_t id = cv->char_to_id[byte];
    return id == MSE_TOK_NO_ID ? TOK_UNK : id;
}

uint8_t cv_decode_id(const CharVocab *cv, int32_t id) {
    if (id < 0 || id >= MSE_TOK_CHAR_ID_CAP || cv->id_to_char[id] == MSE_TOK_NO_ID) return 0;
    return (uint8_t)cv->id_to_char[id];
}

int cv_is_char_id(const CharVocab *cv, int32_t id) {
    return id >= 0 && id < MSE_TOK_CHAR_ID_CAP && cv->id_to_char[id] != MSE_TOK_NO_ID;
}

/* ============================================================ WordVocab */
void wv_init(WordVocab *wv) {
    strmap_init(&wv->token_to_id);
    arena_init(&wv->arena);
    wv->id_to_token = NULL; wv->id_to_token_len = NULL;
    wv->count = 0; wv->cap = 0;
}

void wv_free(WordVocab *wv) {
    strmap_free(&wv->token_to_id);
    arena_free(&wv->arena);
    free(wv->id_to_token); free(wv->id_to_token_len);
    wv->id_to_token = NULL; wv->id_to_token_len = NULL; wv->count = 0; wv->cap = 0;
}

static void wv_ensure_cap(WordVocab *wv, int32_t n) {
    if (n <= wv->cap) return;
    int32_t newcap = wv->cap ? wv->cap * 2 : 64;
    while (newcap < n) newcap *= 2;
    wv->id_to_token = (char **)realloc(wv->id_to_token, sizeof(char *) * (size_t)newcap);
    wv->id_to_token_len = (int32_t *)realloc(wv->id_to_token_len, sizeof(int32_t) * (size_t)newcap);
    for (int32_t i = wv->cap; i < newcap; i++) { wv->id_to_token[i] = NULL; wv->id_to_token_len[i] = 0; }
    wv->cap = newcap;
}

int wv_lookup_lower(const WordVocab *wv, const char *word, int32_t len, int32_t *out_id) {
    char stackbuf[128];
    char *buf = len <= (int32_t)sizeof(stackbuf) ? stackbuf : (char *)malloc((size_t)len);
    for (int32_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)word[i];
        buf[i] = (char)((c >= 'A' && c <= 'Z') ? c + 32 : c);
    }
    int found = strmap_get(&wv->token_to_id, buf, len, out_id);
    if (buf != stackbuf) free(buf);
    return found;
}

const char *wv_decode_id(const WordVocab *wv, int32_t id, int32_t *len_out) {
    int32_t idx = id - TOK_FIRST_FREE;
    if (idx < 0 || idx >= wv->count) { if (len_out) *len_out = 0; return ""; }
    if (len_out) *len_out = wv->id_to_token_len[idx];
    return wv->id_to_token[idx];
}

int32_t wv_max_id(const WordVocab *wv) {
    return wv->count > 0 ? TOK_FIRST_FREE + wv->count - 1 : TOK_WORD_BOUND;
}

/* Direct setters used only by mse_tok_load(): the on-disk file records
 * the exact (id, bytes) mapping the original build produced, so a
 * reload should reproduce it exactly via direct assignment rather
 * than re-deriving ids through cv_build()'s/wv_build()'s own ordering
 * logic (which sorts/batches by criteria -- byte value, frequency --
 * that the file no longer carries, and isn't guaranteed to reproduce
 * the original id for every entry once char ids and word ids have
 * become interleaved across more than one train()/extend_vocab() call). */
static void cv_set_entry(CharVocab *cv, int32_t id, uint8_t byte) {
    cv->char_to_id[byte] = id;
    if (id < MSE_TOK_CHAR_ID_CAP) cv->id_to_char[id] = byte;
    if (id > cv->max_id) cv->max_id = id;
}

static void wv_set_entry(WordVocab *wv, int32_t id, const char *bytes, int32_t len) {
    int32_t idx = id - TOK_FIRST_FREE;
    wv_ensure_cap(wv, idx + 1);
    char *owned = arena_strndup(&wv->arena, bytes, len);
    wv->id_to_token[idx] = owned;
    wv->id_to_token_len[idx] = len;
    strmap_put(&wv->token_to_id, owned, len, id);
    if (idx + 1 > wv->count) wv->count = idx + 1;
}

/* most-frequent-first, ties broken by byte-order on the word itself —
 * mirrors TokenVocabulary.build()'s `sorted(candidates, key=lambda w:
 * (-folded[w], w))`. */
struct WvSortItem { const char *w; int32_t len; int32_t freq; };
static int wv_sort_cmp(const void *pa, const void *pb) {
    const struct WvSortItem *a = (const struct WvSortItem *)pa, *b = (const struct WvSortItem *)pb;
    if (a->freq != b->freq) return b->freq - a->freq; /* descending frequency */
    int32_t minlen = a->len < b->len ? a->len : b->len;
    int c = memcmp(a->w, b->w, (size_t)minlen);
    if (c != 0) return c;
    return a->len - b->len;
}

void wv_build(WordVocab *wv, CharVocab *cv, char **words, int32_t *lens, int32_t *freqs, int32_t n) {
    if (n == 0) return;
    struct WvSortItem *items = (struct WvSortItem *)malloc(sizeof(struct WvSortItem) * (size_t)n);
    int32_t m = 0;
    for (int32_t i = 0; i < n; i++) {
        if (lens[i] <= 1) continue;                      /* stage 2 never stores 1-char words */
        int32_t existing;
        if (wv_lookup_lower(wv, words[i], lens[i], &existing)) continue; /* already known */
        items[m].w = words[i]; items[m].len = lens[i]; items[m].freq = freqs[i];
        m++;
    }
    if (m == 0) { free(items); return; }
    qsort(items, (size_t)m, sizeof(struct WvSortItem), wv_sort_cmp);

    int32_t next_id = (cv->max_id > wv_max_id(wv) ? cv->max_id : wv_max_id(wv)) + 1;
    /* Capacity must cover the actual max index this batch will write to
     * -- (next_id + m - 1) - TOK_FIRST_FREE -- NOT wv->count + m: if
     * stage 1 has claimed ids past wv->count's current span (the usual
     * case whenever characters and words share one growing id space —
     * see module docstring), next_id can already be well beyond
     * TOK_FIRST_FREE + wv->count, and basing the reserve on wv->count
     * alone under-allocates. */
    int32_t max_idx = (next_id + m - 1) - TOK_FIRST_FREE;
    wv_ensure_cap(wv, max_idx + 1);
    for (int32_t i = 0; i < m; i++) {
        int32_t id = next_id++;
        int32_t idx = id - TOK_FIRST_FREE;
        /* canonical stored form: lowercase, byte-for-byte (chars.build()
         * already learned every lowercase-folded byte this word needs —
         * see mse_tok_train()'s dual-case chars.build() call) */
        char *lower = (char *)malloc((size_t)items[i].len);
        for (int32_t k = 0; k < items[i].len; k++) {
            unsigned char c = (unsigned char)items[i].w[k];
            lower[k] = (char)((c >= 'A' && c <= 'Z') ? c + 32 : c);
        }
        char *owned = arena_strndup(&wv->arena, lower, items[i].len);
        free(lower);
        wv->id_to_token[idx] = owned;
        wv->id_to_token_len[idx] = items[i].len;
        strmap_put(&wv->token_to_id, owned, items[i].len, id);
        if (idx + 1 > wv->count) wv->count = idx + 1;
    }
    free(items);
}

/* ============================================================ normalize
 * Shared core: lower (if fold_case) + filter -> isolate punctuation
 * (apostrophe-in-contraction excepted) -> collapse whitespace. Returns
 * a newly malloc'd, NUL-terminated, space-joined string.
 */
static int is_ascii_lower_or_digit(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'); }
static int is_ascii_alnum_cs(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}
static int is_ascii_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
static int is_kept_char(unsigned char c, int fold_case) {
    if (fold_case ? is_ascii_lower_or_digit(c) : is_ascii_alnum_cs(c)) return 1;
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') return 1;
    if (MSE_IS_PUNCT((char)c)) return 1;
    return 0;
}

static char *normalize_core(const char *text, int32_t len, int fold_case, int32_t *out_len) {
    /* Pass A: (lower if folding) + filter -> buf1 (exactly `len` bytes, 1:1 with input). */
    char *buf1 = (char *)malloc((size_t)len + 1);
    for (int32_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        unsigned char lc = (fold_case && c >= 'A' && c <= 'Z') ? (unsigned char)(c + 32) : c;
        buf1[i] = is_kept_char(lc, fold_case) ? (char)lc : ' ';
    }
    buf1[len] = '\0';

    /* Pass B: isolate punctuation except apostrophe -> buf2 (<= 3*len+1). */
    char *buf2 = (char *)malloc((size_t)len * 3 + 1);
    int32_t n2 = 0;
    for (int32_t i = 0; i < len; i++) {
        char c = buf1[i];
        if (MSE_IS_PUNCT(c) && c != '\'') {
            buf2[n2++] = ' '; buf2[n2++] = c; buf2[n2++] = ' ';
        } else {
            buf2[n2++] = c;
        }
    }
    free(buf1);

    /* Pass C: lone-apostrophe isolation -> buf3 (<= 3*n2+1). */
    char *buf3 = (char *)malloc((size_t)n2 * 3 + 1);
    int32_t n3 = 0;
    for (int32_t i = 0; i < n2; i++) {
        char c = buf2[i];
        if (c == '\'') {
            unsigned char prev = (i > 0) ? (unsigned char)buf2[i - 1] : 0;
            unsigned char next = (i + 1 < n2) ? (unsigned char)buf2[i + 1] : 0;
            int flanked = fold_case
                ? (is_ascii_lower_or_digit(prev) && is_ascii_lower_or_digit(next))
                : (is_ascii_alnum_cs(prev) && is_ascii_alnum_cs(next));
            if (flanked) {
                buf3[n3++] = '\'';
            } else {
                buf3[n3++] = ' '; buf3[n3++] = '\''; buf3[n3++] = ' ';
            }
        } else {
            buf3[n3++] = c;
        }
    }
    free(buf2);

    /* Pass D: collapse ANY whitespace run to a single space, then strip. */
    char *buf4 = (char *)malloc((size_t)n3 + 1);
    int32_t n4 = 0;
    int32_t i = 0;
    while (i < n3 && is_ascii_space(buf3[i])) i++;
    while (i < n3) {
        if (is_ascii_space(buf3[i])) {
            int32_t j = i;
            while (j < n3 && is_ascii_space(buf3[j])) j++;
            if (j < n3) buf4[n4++] = ' ';
            i = j;
        } else {
            buf4[n4++] = buf3[i++];
        }
    }
    free(buf3);
    buf4[n4] = '\0';
    if (out_len) *out_len = n4;
    return buf4;
}

char *mse_normalize(const char *text, int32_t len, int32_t *out_len) {
    return normalize_core(text, len, 1, out_len);
}

static char **split_space_joined(const char *norm, int32_t nlen, int32_t **out_lens, int32_t *out_n) {
    int32_t cap = 8, n = 0;
    char **words = (char **)malloc(sizeof(char *) * (size_t)cap);
    int32_t *lens = (int32_t *)malloc(sizeof(int32_t) * (size_t)cap);
    int32_t i = 0;
    while (i < nlen) {
        while (i < nlen && norm[i] == ' ') i++;
        int32_t start = i;
        while (i < nlen && norm[i] != ' ') i++;
        if (i > start) {
            if (n == cap) { cap *= 2; words = (char **)realloc(words, sizeof(char *) * (size_t)cap);
                             lens = (int32_t *)realloc(lens, sizeof(int32_t) * (size_t)cap); }
            int32_t wlen = i - start;
            char *w = (char *)malloc((size_t)wlen + 1);
            memcpy(w, norm + start, (size_t)wlen);
            w[wlen] = '\0';
            words[n] = w; lens[n] = wlen; n++;
        }
    }
    *out_lens = lens; *out_n = n;
    return words;
}

char **mse_segment(const char *text, int32_t len, int32_t **out_lens, int32_t *out_n) {
    int32_t nlen;
    char *norm = normalize_core(text, len, 0, &nlen); /* case-preserving */
    char **words = split_space_joined(norm, nlen, out_lens, out_n);
    free(norm);
    return words;
}

/* ======================================================= split_sentences
 * Unchanged from the BPE-era port -- this rule never depended on which
 * vocabulary algorithm consumes its output. */
static int is_delim(char c) { return c == '.' || c == '!' || c == '?' || c == '\n'; }
static int is_sent_punct(char c) { return c == '.' || c == '!' || c == '?'; }

static void trim_slice(const char *s, int32_t len, int32_t *start, int32_t *end) {
    int32_t a = 0, b = len;
    while (a < b && isspace((unsigned char)s[a])) a++;
    while (b > a && isspace((unsigned char)s[b - 1])) b--;
    *start = a; *end = b;
}

typedef struct { char **items; int32_t *lens; int32_t n, cap; } StrList;
static void strlist_init(StrList *l) { l->items = NULL; l->lens = NULL; l->n = 0; l->cap = 0; }
static void strlist_push(StrList *l, const char *s, int32_t len) {
    if (l->n == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 8;
        l->items = (char **)realloc(l->items, sizeof(char *) * (size_t)l->cap);
        l->lens = (int32_t *)realloc(l->lens, sizeof(int32_t) * (size_t)l->cap);
    }
    char *copy = (char *)malloc((size_t)len + 1);
    memcpy(copy, s, (size_t)len);
    copy[len] = '\0';
    l->items[l->n] = copy;
    l->lens[l->n] = len;
    l->n++;
}

static void finalize_one(StrList *out, const char *body, int32_t blen, const char *punct, int32_t plen) {
    int32_t bs, be;
    trim_slice(body, blen, &bs, &be);
    int32_t trimmed_len = be - bs;
    if (trimmed_len == 0 && plen == 0) return;
    if (plen == 0) {
        strlist_push(out, body + bs, trimmed_len);
    } else {
        char *tmp = (char *)malloc((size_t)trimmed_len + 1 + (size_t)plen);
        memcpy(tmp, body + bs, (size_t)trimmed_len);
        tmp[trimmed_len] = ' ';
        memcpy(tmp + trimmed_len + 1, punct, (size_t)plen);
        strlist_push(out, tmp, trimmed_len + 1 + plen);
        free(tmp);
    }
}

char **mse_split_sentences(const char *text, int32_t len, int32_t **out_lens, int32_t *out_n) {
    StrList out; strlist_init(&out);
    int32_t body_start = 0;
    int32_t i = 0;
    char punctbuf[64];
    while (i < len) {
        if (is_delim(text[i])) {
            int32_t j = i;
            int32_t plen = 0;
            while (j < len && is_delim(text[j])) {
                if (is_sent_punct(text[j]) && plen < (int32_t)sizeof(punctbuf)) punctbuf[plen++] = text[j];
                j++;
            }
            finalize_one(&out, text + body_start, i - body_start, punctbuf, plen);
            body_start = j;
            i = j;
        } else {
            i++;
        }
    }
    finalize_one(&out, text + body_start, len - body_start, NULL, 0);

    *out_n = out.n;
    *out_lens = out.lens;
    return out.items;
}

void mse_free_sentences(char **sents, int32_t n) {
    for (int32_t i = 0; i < n; i++) free(sents[i]);
    free(sents);
}

/* ================================================================ WordFreq
 * Insertion-ordered, CASE-PRESERVING word -> frequency table (mirrors
 * Python's Counter() fed by segment(), not normalize()). */
typedef struct {
    StrMap   word_to_idx;
    Arena    arena;
    char   **words;
    int32_t *lens;
    int32_t *freq;
    int32_t  n, cap;
} WordFreq;

static void wf_init(WordFreq *wf) {
    strmap_init(&wf->word_to_idx);
    arena_init(&wf->arena);
    wf->words = NULL; wf->lens = NULL; wf->freq = NULL; wf->n = 0; wf->cap = 0;
}
static void wf_free(WordFreq *wf) {
    strmap_free(&wf->word_to_idx);
    arena_free(&wf->arena);
    free(wf->words); free(wf->lens); free(wf->freq);
}
static void wf_add(WordFreq *wf, const char *s, int32_t len) {
    int32_t idx;
    if (strmap_get(&wf->word_to_idx, s, len, &idx)) { wf->freq[idx]++; return; }
    if (wf->n == wf->cap) {
        wf->cap = wf->cap ? wf->cap * 2 : 64;
        wf->words = (char **)realloc(wf->words, sizeof(char *) * (size_t)wf->cap);
        wf->lens = (int32_t *)realloc(wf->lens, sizeof(int32_t) * (size_t)wf->cap);
        wf->freq = (int32_t *)realloc(wf->freq, sizeof(int32_t) * (size_t)wf->cap);
    }
    char *owned = arena_strndup(&wf->arena, s, len);
    wf->words[wf->n] = owned; wf->lens[wf->n] = len; wf->freq[wf->n] = 1;
    strmap_put(&wf->word_to_idx, owned, len, wf->n);
    wf->n++;
}
static void wf_add_segmented(WordFreq *wf, const char *text, int32_t len) {
    int32_t n_words; int32_t *lens;
    char **words = mse_segment(text, len, &lens, &n_words);
    for (int32_t i = 0; i < n_words; i++) wf_add(wf, words[i], lens[i]);
    mse_free_sentences(words, n_words);
    free(lens);
}
static void wf_add_from_sentences(WordFreq *wf, char **sents, int32_t *lens, int32_t n_sent) {
    for (int32_t i = 0; i < n_sent; i++) wf_add_segmented(wf, sents[i], lens[i]);
}

/* build the union-of-bytes "seen256" scratch (both exact-case AND
 * lowercase-folded forms — see mse_tok_train()'s doc comment: stage 1
 * must learn both, since a word capitalized only at a sentence start
 * still needs its lowercase-folded spelling available to stage 2). */
static void seen_bytes_dual_case(const WordFreq *wf, uint8_t seen256[256]) {
    memset(seen256, 0, 256);
    for (int32_t wi = 0; wi < wf->n; wi++) {
        for (int32_t i = 0; i < wf->lens[wi]; i++) {
            unsigned char c = (unsigned char)wf->words[wi][i];
            seen256[c] = 1;
            if (c >= 'A' && c <= 'Z') seen256[c + 32] = 1;
        }
    }
}

/* =================================================== public MseWordFreq */
struct MseWordFreq { WordFreq wf; };

MseWordFreq *mse_wordfreq_create(void) {
    MseWordFreq *w = (MseWordFreq *)malloc(sizeof(MseWordFreq));
    wf_init(&w->wf);
    return w;
}

void mse_wordfreq_free(MseWordFreq *wf) {
    if (!wf) return;
    wf_free(&wf->wf);
    free(wf);
}

void mse_wordfreq_add_text(MseWordFreq *wf, const char *text, int32_t len) {
    int32_t n_sent; int32_t *lens;
    char **sents = mse_split_sentences(text, len, &lens, &n_sent);
    wf_add_from_sentences(&wf->wf, sents, lens, n_sent);
    mse_free_sentences(sents, n_sent);
    free(lens);
}

int mse_wordfreq_add_file(MseWordFreq *wf, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return -1; }
    char *buf = (char *)malloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = '\0';
    mse_wordfreq_add_text(wf, buf, (int32_t)rd);
    free(buf);
    return 0;
}

int32_t mse_wordfreq_distinct_words(const MseWordFreq *wf) { return wf->wf.n; }

/* ==================================================================== API
 * train_from_word_freq(): stage 1 first (dual-case bytes), then stage
 * 2 (case-folded + frequency-merged, capped to vocab_size) — mirrors
 * CharWordTokenizer._train_from_word_freq(). */
static void train_from_word_freq(MseTokenizer *t, WordFreq *wf, int32_t min_next_id) {
    uint8_t seen256[256];
    seen_bytes_dual_case(wf, seen256);
    cv_build(&t->chars, seen256, min_next_id);

    /* fold to lowercase + merge frequencies (StrMap keyed by lowercase
     * bytes), matching _train_from_word_freq()'s `multi_char` dict */
    StrMap folded_idx; strmap_init(&folded_idx);
    Arena folded_arena; arena_init(&folded_arena);
    char **fwords = NULL; int32_t *flens = NULL, *ffreq = NULL;
    int32_t fn = 0, fcap = 0;
    for (int32_t wi = 0; wi < wf->n; wi++) {
        if (wf->lens[wi] <= 1) continue;
        char stackbuf[128];
        int32_t wl = wf->lens[wi];
        char *lower = wl <= (int32_t)sizeof(stackbuf) ? stackbuf : (char *)malloc((size_t)wl);
        for (int32_t k = 0; k < wl; k++) {
            unsigned char c = (unsigned char)wf->words[wi][k];
            lower[k] = (char)((c >= 'A' && c <= 'Z') ? c + 32 : c);
        }
        int32_t idx;
        if (strmap_get(&folded_idx, lower, wl, &idx)) {
            ffreq[idx] += wf->freq[wi];
        } else {
            if (fn == fcap) {
                fcap = fcap ? fcap * 2 : 64;
                fwords = (char **)realloc(fwords, sizeof(char *) * (size_t)fcap);
                flens = (int32_t *)realloc(flens, sizeof(int32_t) * (size_t)fcap);
                ffreq = (int32_t *)realloc(ffreq, sizeof(int32_t) * (size_t)fcap);
            }
            char *owned = arena_strndup(&folded_arena, lower, wl);
            fwords[fn] = owned; flens[fn] = wl; ffreq[fn] = wf->freq[wi];
            strmap_put(&folded_idx, owned, wl, fn);
            fn++;
        }
        if (lower != stackbuf) free(lower);
    }

    /* cap to t->vocab_size most-frequent (wv_build's own sort handles
     * the tie-break; capping here just trims the candidate pool first,
     * matching _train_from_word_freq()'s `sorted(...)[:vocab_size]`) */
    if (fn > t->vocab_size) {
        struct WvSortItem *tmp = (struct WvSortItem *)malloc(sizeof(struct WvSortItem) * (size_t)fn);
        for (int32_t i = 0; i < fn; i++) tmp[i] = (struct WvSortItem){ fwords[i], flens[i], ffreq[i] };
        qsort(tmp, (size_t)fn, sizeof(struct WvSortItem), wv_sort_cmp);
        fn = t->vocab_size;
        char **kwords = (char **)malloc(sizeof(char *) * (size_t)fn);
        int32_t *klens = (int32_t *)malloc(sizeof(int32_t) * (size_t)fn);
        int32_t *kfreq = (int32_t *)malloc(sizeof(int32_t) * (size_t)fn);
        for (int32_t i = 0; i < fn; i++) { kwords[i] = (char *)tmp[i].w; klens[i] = tmp[i].len; kfreq[i] = tmp[i].freq; }
        free(tmp);
        free(fwords); free(flens); free(ffreq);
        fwords = kwords; flens = klens; ffreq = kfreq;
    }

    wv_build(&t->words, &t->chars, fwords, flens, ffreq, fn);

    strmap_free(&folded_idx); arena_free(&folded_arena);
    free(fwords); free(flens); free(ffreq);
}

void mse_tok_train_from_wordfreq(MseTokenizer *t, MseWordFreq *wf) {
    train_from_word_freq(t, &wf->wf, /*min_next_id=*/0);
}

void mse_tok_init(MseTokenizer *t, int32_t vocab_size) {
    t->vocab_size = vocab_size;
    cv_init(&t->chars);
    wv_init(&t->words);
    strmap_init(&t->cache_index);
    arena_init(&t->cache_arena);
    t->cache_vecs = NULL; t->cache_count = 0; t->cache_cap = 0;
}

void mse_tok_free(MseTokenizer *t) {
    wv_free(&t->words);
    strmap_free(&t->cache_index);
    arena_free(&t->cache_arena);
    for (int32_t i = 0; i < t->cache_count; i++) i32vec_free(&t->cache_vecs[i]);
    free(t->cache_vecs);
}

static void cache_clear(MseTokenizer *t) {
    strmap_free(&t->cache_index);
    strmap_init(&t->cache_index);
    arena_free(&t->cache_arena);
    arena_init(&t->cache_arena);
    for (int32_t i = 0; i < t->cache_count; i++) i32vec_free(&t->cache_vecs[i]);
    t->cache_count = 0;
}

void mse_tok_train(MseTokenizer *t, const char *corpus, int32_t len) {
    int32_t n_sent; int32_t *lens;
    char **sents = mse_split_sentences(corpus, len, &lens, &n_sent);
    WordFreq wf; wf_init(&wf);
    wf_add_from_sentences(&wf, sents, lens, n_sent);
    mse_free_sentences(sents, n_sent);
    free(lens);
    train_from_word_freq(t, &wf, 0);
    wf_free(&wf);
}

int mse_tok_train_from_file(MseTokenizer *t, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return -1; }
    char *buf = (char *)malloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = '\0';
    mse_tok_train(t, buf, (int32_t)rd);
    free(buf);
    return 0;
}

static int32_t next_free_id(const MseTokenizer *t) {
    int32_t wmax = wv_max_id(&t->words);
    return (t->chars.max_id > wmax ? t->chars.max_id : wmax) + 1;
}

int32_t mse_tok_extend_vocab(MseTokenizer *t, const char *corpus, int32_t len, int32_t target_vocab_size) {
    /* words.vocab_size - len(SPECIAL_TOKENS) == t->words.count exactly
     * (token_to_id is seeded with the 5 specials in Python; here the
     * specials never occupy a WordVocab slot at all, so count already
     * IS that difference). */
    if (target_vocab_size <= t->words.count) return 0;

    int32_t n_sent; int32_t *lens;
    char **sents = mse_split_sentences(corpus, len, &lens, &n_sent);
    WordFreq wf; wf_init(&wf);
    wf_add_from_sentences(&wf, sents, lens, n_sent);
    mse_free_sentences(sents, n_sent);
    free(lens);
    if (wf.n == 0) { wf_free(&wf); return 0; }

    int32_t start_size = t->words.count;
    t->vocab_size = target_vocab_size;

    uint8_t seen256[256];
    seen_bytes_dual_case(&wf, seen256);
    cv_build(&t->chars, seen256, next_free_id(t));

    int32_t remaining_budget = target_vocab_size - start_size;
    if (remaining_budget < 0) remaining_budget = 0;

    StrMap folded_idx; strmap_init(&folded_idx);
    Arena folded_arena; arena_init(&folded_arena);
    char **fwords = NULL; int32_t *flens = NULL, *ffreq = NULL;
    int32_t fn = 0, fcap = 0;
    for (int32_t wi = 0; wi < wf.n; wi++) {
        if (wf.lens[wi] <= 1) continue;
        int32_t existing;
        if (wv_lookup_lower(&t->words, wf.words[wi], wf.lens[wi], &existing)) continue;
        char stackbuf[128]; int32_t wl = wf.lens[wi];
        char *lower = wl <= (int32_t)sizeof(stackbuf) ? stackbuf : (char *)malloc((size_t)wl);
        for (int32_t k = 0; k < wl; k++) {
            unsigned char c = (unsigned char)wf.words[wi][k];
            lower[k] = (char)((c >= 'A' && c <= 'Z') ? c + 32 : c);
        }
        int32_t idx;
        if (strmap_get(&folded_idx, lower, wl, &idx)) {
            ffreq[idx] += wf.freq[wi];
        } else {
            if (fn == fcap) {
                fcap = fcap ? fcap * 2 : 64;
                fwords = (char **)realloc(fwords, sizeof(char *) * (size_t)fcap);
                flens = (int32_t *)realloc(flens, sizeof(int32_t) * (size_t)fcap);
                ffreq = (int32_t *)realloc(ffreq, sizeof(int32_t) * (size_t)fcap);
            }
            char *owned = arena_strndup(&folded_arena, lower, wl);
            fwords[fn] = owned; flens[fn] = wl; ffreq[fn] = wf.freq[wi];
            strmap_put(&folded_idx, owned, wl, fn);
            fn++;
        }
        if (lower != stackbuf) free(lower);
    }

    if (fn > remaining_budget) {
        struct WvSortItem *tmp = (struct WvSortItem *)malloc(sizeof(struct WvSortItem) * (size_t)fn);
        for (int32_t i = 0; i < fn; i++) tmp[i] = (struct WvSortItem){ fwords[i], flens[i], ffreq[i] };
        qsort(tmp, (size_t)fn, sizeof(struct WvSortItem), wv_sort_cmp);
        fn = remaining_budget;
        char **kwords = (char **)malloc(sizeof(char *) * (size_t)(fn ? fn : 1));
        int32_t *klens = (int32_t *)malloc(sizeof(int32_t) * (size_t)(fn ? fn : 1));
        int32_t *kfreq = (int32_t *)malloc(sizeof(int32_t) * (size_t)(fn ? fn : 1));
        for (int32_t i = 0; i < fn; i++) { kwords[i] = (char *)tmp[i].w; klens[i] = tmp[i].len; kfreq[i] = tmp[i].freq; }
        free(tmp);
        free(fwords); free(flens); free(ffreq);
        fwords = kwords; flens = klens; ffreq = kfreq;
    }

    wv_build(&t->words, &t->chars, fwords, flens, ffreq, fn);

    strmap_free(&folded_idx); arena_free(&folded_arena);
    free(fwords); free(flens); free(ffreq);
    wf_free(&wf);

    cache_clear(t); /* new stage-2 entries can change how a cached word resolves */
    return t->words.count - start_size;
}

int32_t mse_tok_vocab_size_actual(const MseTokenizer *t) {
    int32_t wmax = wv_max_id(&t->words);
    int32_t m = t->chars.max_id > wmax ? t->chars.max_id : wmax;
    return m + 1;
}

/* _encode_word(): one word -> a small id list, cached by exact
 * (case-preserved) spelling. */
static void encode_word(MseTokenizer *t, const char *word, int32_t wlen, i32vec *out) {
    int32_t cache_idx;
    if (strmap_get(&t->cache_index, word, wlen, &cache_idx)) {
        i32vec *cached = &t->cache_vecs[cache_idx];
        for (size_t i = 0; i < cached->len; i++) i32vec_push(out, cached->data[i]);
        return;
    }

    i32vec resolved; i32vec_init(&resolved);
    if (wlen == 1) {
        i32vec_push(&resolved, TOK_WORD_BOUND);
        i32vec_push(&resolved, cv_encode_char(&t->chars, (uint8_t)word[0]));
    } else {
        int32_t wid;
        if (wv_lookup_lower(&t->words, word, wlen, &wid)) {
            i32vec_push(&resolved, wid);
        } else {
            i32vec_push(&resolved, TOK_WORD_BOUND);
            for (int32_t i = 0; i < wlen; i++)
                i32vec_push(&resolved, cv_encode_char(&t->chars, (uint8_t)word[i]));
        }
    }

    if (t->cache_count == t->cache_cap) {
        t->cache_cap = t->cache_cap ? t->cache_cap * 2 : 64;
        t->cache_vecs = (i32vec *)realloc(t->cache_vecs, sizeof(i32vec) * (size_t)t->cache_cap);
    }
    t->cache_vecs[t->cache_count] = resolved; /* transfer ownership */
    char *owned = arena_strndup(&t->cache_arena, word, wlen);
    strmap_put(&t->cache_index, owned, wlen, t->cache_count);
    for (size_t i = 0; i < resolved.len; i++) i32vec_push(out, resolved.data[i]);
    t->cache_count++;
}

void mse_tok_encode(MseTokenizer *t, const char *text, int32_t len, i32vec *out) {
    i32vec_push(out, TOK_BOS);
    int32_t n_words; int32_t *lens;
    char **words = mse_segment(text, len, &lens, &n_words);
    for (int32_t i = 0; i < n_words; i++) encode_word(t, words[i], lens[i], out);
    mse_free_sentences(words, n_words);
    free(lens);
}

void mse_tok_encode_for_training(MseTokenizer *t, const char *text, int32_t len, i32vec *out) {
    mse_tok_encode(t, text, len, out);
    i32vec_push(out, TOK_EOS);
}

char *mse_tok_decode(MseTokenizer *t, const int32_t *ids, int32_t n) {
    /* Build the "words" list exactly like decode(): WORD_BOUND starts a
     * new word and renders nothing; a run of stage-1 char ids between
     * boundaries (or a stage-2 word id, already atomic) is one word;
     * PAD/UNK/BOS/EOS are dropped without affecting word boundaries. */
    StrList words; strlist_init(&words);
    char *cur = NULL; int32_t cur_len = 0, cur_cap = 0;
    for (int32_t i = 0; i < n; i++) {
        int32_t id = ids[i];
        if (id == TOK_PAD || id == TOK_UNK || id == TOK_BOS || id == TOK_EOS) continue;
        if (id == TOK_WORD_BOUND) {
            if (cur_len) { strlist_push(&words, cur, cur_len); cur_len = 0; }
            continue;
        }
        if (cv_is_char_id(&t->chars, id)) {
            if (cur_len + 1 > cur_cap) { cur_cap = cur_cap ? cur_cap * 2 : 16; cur = (char *)realloc(cur, (size_t)cur_cap); }
            cur[cur_len++] = (char)cv_decode_id(&t->chars, id);
        } else {
            if (cur_len) { strlist_push(&words, cur, cur_len); cur_len = 0; }
            int32_t tlen; const char *tok = wv_decode_id(&t->words, id, &tlen);
            if (tlen > 0) strlist_push(&words, tok, tlen);
        }
    }
    if (cur_len) strlist_push(&words, cur, cur_len);
    free(cur);

    /* punctuation-aware join (unchanged rule) */
    size_t cap = 64, olen = 0;
    char *out = (char *)malloc(cap);
    out[0] = '\0';
    for (int32_t i = 0; i < words.n; i++) {
        const char *w = words.items[i]; int32_t wl = words.lens[i];
        if (wl == 0) continue;
        int all_no_space_before = 1;
        for (int32_t k = 0; k < wl; k++) if (!MSE_NO_SPACE_BEFORE(w[k])) { all_no_space_before = 0; break; }
        int need_space = (olen > 0) && !all_no_space_before;
        size_t add = (size_t)wl + (need_space ? 1 : 0);
        if (olen + add + 1 > cap) { while (olen + add + 1 > cap) cap *= 2; out = (char *)realloc(out, cap); }
        if (need_space) out[olen++] = ' ';
        memcpy(out + olen, w, (size_t)wl);
        olen += (size_t)wl;
    }
    out[olen] = '\0';
    mse_free_sentences(words.items, words.n);
    free(words.lens);
    return out;
}


/* ============================================================ persistence
 * Custom flat binary format, version 2 (the BPE-era format was
 * version 1 -- see tools/mse_format.h). Every id from TOK_FIRST_FREE
 * up is either a char-vocab entry (a single raw byte) or a word-vocab
 * entry (a lowercase multi-byte string); the tag is implicit in the
 * length (== 1 -> char, > 1 -> word), same convention encode()/
 * decode() use to tell them apart at runtime. Recipes (Python's
 * id_to_chars) are NOT persisted -- nothing outside tokenizer.c/
 * tokenizer.py ever reads them; see mse_tokenizer.h's module
 * docstring. Reload reproduces the exact (id, bytes) mapping via
 * direct assignment (cv_set_entry/wv_set_entry above), not by
 * re-running cv_build()/wv_build()'s own ordering logic -- see those
 * setters' comment for why that distinction matters once char ids and
 * word ids have interleaved across more than one train()/
 * extend_vocab() call.
 */
#include "mse_format.h"

int mse_tok_save(const MseTokenizer *t, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int32_t vocab_count = mse_tok_vocab_size_actual(t);
    MseTokHeader h = { .magic = MSE_TOK_MAGIC, .version = 2,
                        .vocab_size_target = t->vocab_size,
                        .vocab_count = vocab_count,
                        .n_merges = 0 };
    fwrite(&h, sizeof(h), 1, f);
    for (int32_t id = TOK_FIRST_FREE; id < vocab_count; id++) {
        if (cv_is_char_id(&t->chars, id)) {
            int32_t l = 1;
            char b = (char)cv_decode_id(&t->chars, id);
            fwrite(&l, sizeof(int32_t), 1, f);
            fwrite(&b, 1, 1, f);
        } else {
            int32_t l; const char *s = wv_decode_id(&t->words, id, &l);
            fwrite(&l, sizeof(int32_t), 1, f);
            fwrite(s, 1, (size_t)l, f);
        }
    }
    fclose(f);
    return 0;
}

int mse_tok_load(MseTokenizer *t, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    MseTokHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1 || h.magic != MSE_TOK_MAGIC || h.version != 2) { fclose(f); return -1; }
    mse_tok_init(t, h.vocab_size_target);

    for (int32_t id = TOK_FIRST_FREE; id < h.vocab_count; id++) {
        int32_t l;
        if (fread(&l, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
        char *buf = (char *)malloc((size_t)l + 1);
        if (l > 0 && fread(buf, 1, (size_t)l, f) != (size_t)l) { free(buf); fclose(f); return -1; }
        buf[l] = '\0';
        if (l == 1) {
            cv_set_entry(&t->chars, id, (uint8_t)buf[0]);
        } else if (l > 1) {
            wv_set_entry(&t->words, id, buf, l);
        }
        free(buf);
    }
    fclose(f);
    return 0;
}
