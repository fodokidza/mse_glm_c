/* tokenizer.c — C port of tokenizer.py. See mse_tokenizer.h for the
 * design notes on why this isn't a literal translation (byte-level
 * normalize with no UTF-8 decode step, id-based merges instead of
 * string concatenation, arena-backed vocab).
 */
#include "mse_tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

/* ================================================================ Vocab */
static int32_t vocab_ensure_slot(Vocab *v, int32_t id) {
    if (id < v->cap) return 0;
    int32_t newcap = v->cap ? v->cap * 2 : 16;
    while (newcap <= id) newcap *= 2;
    v->id_to_str = (char **)realloc(v->id_to_str, sizeof(char *) * newcap);
    v->id_to_len = (int32_t *)realloc(v->id_to_len, sizeof(int32_t) * newcap);
    for (int32_t i = v->cap; i < newcap; i++) { v->id_to_str[i] = NULL; v->id_to_len[i] = 0; }
    v->cap = newcap;
    return 0;
}

static int32_t vocab_set(Vocab *v, const char *s, int32_t len, int32_t id) {
    vocab_ensure_slot(v, id);
    char *owned = arena_strndup(&v->arena, s, len);
    v->id_to_str[id] = owned;
    v->id_to_len[id] = len;
    strmap_put(&v->str_to_id, owned, len, id);
    if (id + 1 > v->count) v->count = id + 1;
    return id;
}

void vocab_init(Vocab *v) {
    strmap_init(&v->str_to_id);
    arena_init(&v->arena);
    v->id_to_str = NULL; v->id_to_len = NULL;
    v->count = 0; v->cap = 0;
    vocab_set(v, "<PAD>", 5, TOK_PAD);
    vocab_set(v, "<UNK>", 5, TOK_UNK);
    vocab_set(v, "<BOS>", 5, TOK_BOS);
    vocab_set(v, "<EOS>", 5, TOK_EOS);
}

void vocab_free(Vocab *v) {
    strmap_free(&v->str_to_id);
    arena_free(&v->arena);
    free(v->id_to_str); free(v->id_to_len);
    v->id_to_str = NULL; v->id_to_len = NULL; v->count = 0; v->cap = 0;
}

int vocab_lookup(const Vocab *v, const char *s, int32_t len, int32_t *out) {
    return strmap_get(&v->str_to_id, s, len, out);
}

const char *vocab_str(const Vocab *v, int32_t id, int32_t *len_out) {
    if (id < 0 || id >= v->count || !v->id_to_str[id]) { if (len_out) *len_out = 0; return ""; }
    if (len_out) *len_out = v->id_to_len[id];
    return v->id_to_str[id];
}

int32_t vocab_get_or_create(Vocab *v, const char *s, int32_t len) {
    int32_t id;
    if (vocab_lookup(v, s, len, &id)) return id;
    id = v->count; /* next sequential id */
    return vocab_set(v, s, len, id);
}

/* ============================================================ normalize */
static int is_ascii_lower_or_digit(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
}
static int is_ascii_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}
static int is_kept_char(unsigned char lc) {
    if (is_ascii_lower_or_digit(lc)) return 1;
    if (lc == ' ' || lc == '\t' || lc == '\n' || lc == '\r' || lc == '\f' || lc == '\v') return 1;
    if (MSE_IS_PUNCT((char)lc)) return 1;
    return 0;
}

char *mse_normalize(const char *text, int32_t len, int32_t *out_len) {
    /* Pass A: lower + filter -> buf1 (exactly `len` bytes, 1:1 with input). */
    char *buf1 = (char *)malloc((size_t)len + 1);
    for (int32_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        unsigned char lc = (c >= 'A' && c <= 'Z') ? (unsigned char)(c + 32) : c;
        buf1[i] = is_kept_char(lc) ? (char)lc : ' ';
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
            int flanked = is_ascii_lower_or_digit(prev) && is_ascii_lower_or_digit(next);
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

    /* Pass D: collapse ANY whitespace run (space, tab, newline, CR, FF,
     * VT — matching Python's \s+) to a single space, then strip. */
    char *buf4 = (char *)malloc((size_t)n3 + 1);
    int32_t n4 = 0;
    int32_t i = 0;
    while (i < n3 && is_ascii_space(buf3[i])) i++; /* skip leading */
    while (i < n3) {
        if (is_ascii_space(buf3[i])) {
            int32_t j = i;
            while (j < n3 && is_ascii_space(buf3[j])) j++;
            if (j < n3) buf4[n4++] = ' '; /* internal run -> one space (drop if trailing) */
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

/* ======================================================= split_sentences */
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
        l->items = (char **)realloc(l->items, sizeof(char *) * l->cap);
        l->lens = (int32_t *)realloc(l->lens, sizeof(int32_t) * l->cap);
    }
    char *copy = (char *)malloc((size_t)len + 1);
    memcpy(copy, s, (size_t)len);
    copy[len] = '\0';
    l->items[l->n] = copy;
    l->lens[l->n] = len;
    l->n++;
}

/* Shared by split_sentences() and (a future) streaming reader: given a
 * body slice and the punctuation to reattach, append "body punct" (or
 * just "body") if non-empty. */
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
    /* trailing body, no delimiter after it */
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
 * Insertion-ordered word -> frequency table, mirroring Python's
 * Counter() (a dict subclass: first-seen order is iteration order). */
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
        wf->words = (char **)realloc(wf->words, sizeof(char *) * wf->cap);
        wf->lens = (int32_t *)realloc(wf->lens, sizeof(int32_t) * wf->cap);
        wf->freq = (int32_t *)realloc(wf->freq, sizeof(int32_t) * wf->cap);
    }
    char *owned = arena_strndup(&wf->arena, s, len);
    wf->words[wf->n] = owned; wf->lens[wf->n] = len; wf->freq[wf->n] = 1;
    strmap_put(&wf->word_to_idx, owned, len, wf->n);
    wf->n++;
}
static void wf_add_from_normalized(WordFreq *wf, const char *norm, int32_t len) {
    int32_t i = 0;
    while (i < len) {
        while (i < len && norm[i] == ' ') i++;
        int32_t start = i;
        while (i < len && norm[i] != ' ') i++;
        if (i > start) wf_add(wf, norm + start, i - start);
    }
}

/* ============================================================ PWMap
 * pair(a,b) -> set-of-word-indices (as a growable array; membership
 * within one word's contribution is deduped by the caller before
 * pwmap_add_word is called, matching Python's `pair_words[pair].add(w)`
 * set semantics). See mse_util.h's PairMap docstring for why this and
 * PairMap are companions sharing the same key packing/hash. */
typedef struct { int64_t key; i32vec words; uint8_t occupied; } PWSlot;
typedef struct { PWSlot *slots; size_t cap, count; } PWMap;

static int64_t pw_pack(int32_t a, int32_t b) { return ((int64_t)(uint32_t)a << 32) | (uint32_t)b; }
static uint64_t pw_hash(int64_t key) {
    uint64_t h = (uint64_t)key;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}
static void pwmap_init(PWMap *m) { m->cap = 16; m->count = 0; m->slots = (PWSlot *)calloc(m->cap, sizeof(PWSlot)); }
static void pwmap_free(PWMap *m) {
    for (size_t i = 0; i < m->cap; i++) if (m->slots[i].occupied) i32vec_free(&m->slots[i].words);
    free(m->slots);
}
static void pwmap_grow(PWMap *m) {
    PWSlot *old = m->slots; size_t oldcap = m->cap;
    m->cap *= 2; m->count = 0;
    m->slots = (PWSlot *)calloc(m->cap, sizeof(PWSlot));
    size_t mask = m->cap - 1;
    for (size_t i = 0; i < oldcap; i++) {
        if (!old[i].occupied) continue;
        size_t j = (size_t)(pw_hash(old[i].key) & mask);
        while (m->slots[j].occupied) j = (j + 1) & mask;
        m->slots[j] = old[i];
        m->count++;
    }
    free(old);
}
static PWSlot *pwmap_find_or_create(PWMap *m, int64_t key) {
    if (m->count * 4 >= m->cap * 3) pwmap_grow(m);
    size_t mask = m->cap - 1;
    size_t i = (size_t)(pw_hash(key) & mask);
    for (;;) {
        PWSlot *s = &m->slots[i];
        if (!s->occupied) { s->occupied = 1; s->key = key; i32vec_init(&s->words); m->count++; return s; }
        if (s->key == key) return s;
        i = (i + 1) & mask;
    }
}
static PWSlot *pwmap_find(PWMap *m, int64_t key) {
    if (m->cap == 0) return NULL;
    size_t mask = m->cap - 1;
    size_t i = (size_t)(pw_hash(key) & mask);
    for (size_t probe = 0; probe < m->cap; probe++) {
        PWSlot *s = &m->slots[i];
        if (!s->occupied) return NULL;
        if (s->key == key) return s;
        i = (i + 1) & mask;
    }
    return NULL;
}
static void pwmap_add_word(PWMap *m, int32_t a, int32_t b, int32_t w) {
    PWSlot *s = pwmap_find_or_create(m, pw_pack(a, b));
    i32vec_push(&s->words, w);
}
static void pwmap_discard_word(PWMap *m, int32_t a, int32_t b, int32_t w) {
    PWSlot *s = pwmap_find(m, pw_pack(a, b));
    if (!s) return;
    for (size_t i = 0; i < s->words.len; i++) {
        if (s->words.data[i] == w) {
            s->words.data[i] = s->words.data[s->words.len - 1];
            s->words.len--;
            return;
        }
    }
}

/* ==================================================== BPE merge training */
typedef void (*pair_cb)(void *ctx, int32_t a, int32_t b);

static void distinct_pairs_in_word_and(i32vec *syms, pair_cb cb, void *ctx) {
    for (size_t i = 0; i + 1 < syms->len; i++) {
        int32_t a = syms->data[i], b = syms->data[i + 1];
        int dup = 0;
        for (size_t k = 0; k < i; k++) if (syms->data[k] == a && syms->data[k + 1] == b) { dup = 1; break; }
        if (!dup) cb(ctx, a, b);
    }
}

struct PWAddCtx { PWMap *pw; int32_t w; };
static void pw_add_cb(void *vctx, int32_t a, int32_t b) {
    struct PWAddCtx *c = (struct PWAddCtx *)vctx;
    pwmap_add_word(c->pw, a, b, c->w);
}

static void run_bpe_merges(BPETokenizer *t, WordFreq *wf, i32vec *word_symbols, int32_t target_vocab_size) {
    PairMap pm; pairmap_init(&pm);
    PWMap pw; pwmap_init(&pw);

    /* initial population, iterating words in first-seen order */
    for (int32_t wi = 0; wi < wf->n; wi++) {
        int32_t freq = wf->freq[wi];
        i32vec *syms = &word_symbols[wi];
        for (size_t i = 0; i + 1 < syms->len; i++)
            pairmap_add(&pm, syms->data[i], syms->data[i + 1], freq);
        struct PWAddCtx ctx = { &pw, wi };
        distinct_pairs_in_word_and(syms, pw_add_cb, &ctx);
    }

    while (t->vocab.count < target_vocab_size) {
        int32_t a, b, count;
        if (!pairmap_max(&pm, &a, &b, &count)) break;

        int32_t alen, blen;
        const char *astr = vocab_str(&t->vocab, a, &alen);
        const char *bstr = vocab_str(&t->vocab, b, &blen);
        char *merged = (char *)malloc((size_t)alen + blen);
        memcpy(merged, astr, (size_t)alen);
        memcpy(merged + alen, bstr, (size_t)blen);
        int32_t merged_id = vocab_get_or_create(&t->vocab, merged, alen + blen);
        free(merged);

        if (t->n_merges == t->merges_cap) {
            t->merges_cap = t->merges_cap ? t->merges_cap * 2 : 64;
            t->merges = (MseMerge *)realloc(t->merges, sizeof(MseMerge) * t->merges_cap);
        }
        t->merges[t->n_merges++] = (MseMerge){ a, b, merged_id };

        PWSlot *s = pwmap_find(&pw, pw_pack(a, b));
        i32vec affected; i32vec_init(&affected);
        if (s) {
            for (size_t i = 0; i < s->words.len; i++) i32vec_push(&affected, s->words.data[i]);
            s->words.len = 0;
        }

        for (size_t ai = 0; ai < affected.len; ai++) {
            int32_t wi = affected.data[ai];
            int32_t freq = wf->freq[wi];
            i32vec *syms = &word_symbols[wi];

            for (size_t i = 0; i + 1 < syms->len; i++)
                pairmap_add(&pm, syms->data[i], syms->data[i + 1], -freq);
            for (size_t i = 0; i + 1 < syms->len; i++) {
                int32_t pa = syms->data[i], pb = syms->data[i + 1];
                int dup = 0;
                for (size_t k = 0; k < i; k++) if (syms->data[k] == pa && syms->data[k + 1] == pb) { dup = 1; break; }
                if (!dup) pwmap_discard_word(&pw, pa, pb, wi);
            }

            i32vec new_syms; i32vec_init(&new_syms);
            size_t i = 0;
            while (i < syms->len) {
                if (i + 1 < syms->len && syms->data[i] == a && syms->data[i + 1] == b) {
                    i32vec_push(&new_syms, merged_id);
                    i += 2;
                } else {
                    i32vec_push(&new_syms, syms->data[i]);
                    i += 1;
                }
            }
            i32vec_free(syms);
            *syms = new_syms;

            for (size_t i2 = 0; i2 + 1 < syms->len; i2++)
                pairmap_add(&pm, syms->data[i2], syms->data[i2 + 1], freq);
            struct PWAddCtx ctx2 = { &pw, wi };
            distinct_pairs_in_word_and(syms, pw_add_cb, &ctx2);
        }
        i32vec_free(&affected);
    }

    pairmap_free(&pm);
    pwmap_free(&pw);
}

static void train_from_word_freq(BPETokenizer *t, WordFreq *wf) {
    /* distinct chars across all words -> base vocab entries, sorted
     * (bytes here are always a-z/0-9/punct, so ascending-byte order
     * IS lexicographic order of the single-char strings). */
    unsigned char seen[256] = {0};
    for (int32_t wi = 0; wi < wf->n; wi++)
        for (int32_t i = 0; i < wf->lens[wi]; i++)
            seen[(unsigned char)wf->words[wi][i]] = 1;
    for (int c = 0; c < 256; c++) {
        if (!seen[c]) continue;
        char ch = (char)c;
        int32_t existing;
        if (!vocab_lookup(&t->vocab, &ch, 1, &existing))
            vocab_get_or_create(&t->vocab, &ch, 1);
    }

    i32vec *word_symbols = (i32vec *)malloc(sizeof(i32vec) * (size_t)(wf->n ? wf->n : 1));
    for (int32_t wi = 0; wi < wf->n; wi++) {
        i32vec_init(&word_symbols[wi]);
        for (int32_t i = 0; i < wf->lens[wi]; i++) {
            int32_t id;
            vocab_lookup(&t->vocab, wf->words[wi] + i, 1, &id); /* guaranteed present */
            i32vec_push(&word_symbols[wi], id);
        }
    }

    run_bpe_merges(t, wf, word_symbols, t->vocab_size);

    for (int32_t wi = 0; wi < wf->n; wi++) i32vec_free(&word_symbols[wi]);
    free(word_symbols);
}

/* =================================================== public MseWordFreq
 * Opaque wrapper around the internal WordFreq -- see mse_tokenizer.h's
 * docstring for why this is exposed (bounded-memory multi-file/multi-
 * chunk training, e.g. tools/mse_train_corpus.c's Pass 1).
 */
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
    for (int32_t i = 0; i < n_sent; i++) {
        int32_t nlen;
        char *norm = mse_normalize(sents[i], lens[i], &nlen);
        wf_add_from_normalized(&wf->wf, norm, nlen);
        free(norm);
    }
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

void mse_tok_train_from_wordfreq(BPETokenizer *t, MseWordFreq *wf) {
    train_from_word_freq(t, &wf->wf);
}

/* ==================================================================== API */
void mse_tok_init(BPETokenizer *t, int32_t vocab_size) {
    t->vocab_size = vocab_size;
    vocab_init(&t->vocab);
    t->merges = NULL; t->n_merges = 0; t->merges_cap = 0;
    strmap_init(&t->cache_index);
    arena_init(&t->cache_arena);
    t->cache_vecs = NULL; t->cache_count = 0; t->cache_cap = 0;
}

void mse_tok_free(BPETokenizer *t) {
    vocab_free(&t->vocab);
    free(t->merges);
    strmap_free(&t->cache_index);
    arena_free(&t->cache_arena);
    for (int32_t i = 0; i < t->cache_count; i++) i32vec_free(&t->cache_vecs[i]);
    free(t->cache_vecs);
}

static void cache_clear(BPETokenizer *t) {
    strmap_free(&t->cache_index);
    strmap_init(&t->cache_index);
    arena_free(&t->cache_arena);
    arena_init(&t->cache_arena);
    for (int32_t i = 0; i < t->cache_count; i++) i32vec_free(&t->cache_vecs[i]);
    t->cache_count = 0;
}

void mse_tok_train(BPETokenizer *t, const char *corpus, int32_t len) {
    int32_t n_sent; int32_t *lens;
    char **sents = mse_split_sentences(corpus, len, &lens, &n_sent);
    WordFreq wf; wf_init(&wf);
    for (int32_t i = 0; i < n_sent; i++) {
        int32_t nlen;
        char *norm = mse_normalize(sents[i], lens[i], &nlen);
        wf_add_from_normalized(&wf, norm, nlen);
        free(norm);
    }
    mse_free_sentences(sents, n_sent);
    free(lens);
    train_from_word_freq(t, &wf);
    wf_free(&wf);
}

int mse_tok_train_from_file(BPETokenizer *t, const char *path) {
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

int32_t mse_tok_extend_vocab(BPETokenizer *t, const char *corpus, int32_t len, int32_t target_vocab_size) {
    int32_t start_size = mse_tok_vocab_size_actual(t);
    if (target_vocab_size <= start_size) return 0;

    int32_t n_sent; int32_t *lens;
    char **sents = mse_split_sentences(corpus, len, &lens, &n_sent);
    WordFreq wf; wf_init(&wf);
    for (int32_t i = 0; i < n_sent; i++) {
        int32_t nlen;
        char *norm = mse_normalize(sents[i], lens[i], &nlen);
        wf_add_from_normalized(&wf, norm, nlen);
        free(norm);
    }
    mse_free_sentences(sents, n_sent);
    free(lens);
    if (wf.n == 0) { wf_free(&wf); return 0; }

    unsigned char seen[256] = {0};
    for (int32_t wi = 0; wi < wf.n; wi++)
        for (int32_t i = 0; i < wf.lens[wi]; i++)
            seen[(unsigned char)wf.words[wi][i]] = 1;
    for (int c = 0; c < 256 && mse_tok_vocab_size_actual(t) < target_vocab_size; c++) {
        if (!seen[c]) continue;
        char ch = (char)c;
        int32_t existing;
        if (!vocab_lookup(&t->vocab, &ch, 1, &existing))
            vocab_get_or_create(&t->vocab, &ch, 1);
    }

    /* re-apply merges already learned so this corpus's words start from
     * the same symbol state a from-scratch encode() would produce */
    i32vec *word_symbols = (i32vec *)malloc(sizeof(i32vec) * (size_t)wf.n);
    for (int32_t wi = 0; wi < wf.n; wi++) {
        i32vec_init(&word_symbols[wi]);
        for (int32_t i = 0; i < wf.lens[wi]; i++) {
            int32_t id;
            if (!vocab_lookup(&t->vocab, wf.words[wi] + i, 1, &id)) id = -1;
            i32vec_push(&word_symbols[wi], id);
        }
        for (int32_t m = 0; m < t->n_merges; m++) {
            int32_t a = t->merges[m].a, b = t->merges[m].b, mid = t->merges[m].merged;
            i32vec *syms = &word_symbols[wi];
            i32vec ns; i32vec_init(&ns);
            size_t k = 0;
            while (k < syms->len) {
                if (k + 1 < syms->len && syms->data[k] == a && syms->data[k + 1] == b) {
                    i32vec_push(&ns, mid); k += 2;
                } else { i32vec_push(&ns, syms->data[k]); k += 1; }
            }
            i32vec_free(syms); *syms = ns;
        }
    }

    run_bpe_merges(t, &wf, word_symbols, target_vocab_size);

    for (int32_t wi = 0; wi < wf.n; wi++) i32vec_free(&word_symbols[wi]);
    free(word_symbols);
    wf_free(&wf);

    cache_clear(t); /* new merges can change how known words split */
    return mse_tok_vocab_size_actual(t) - start_size;
}

int32_t mse_tok_vocab_size_actual(const BPETokenizer *t) { return t->vocab.count; }

static void apply_merges_ids(BPETokenizer *t, const char *word, int32_t wlen, i32vec *out) {
    i32vec_init(out);
    for (int32_t i = 0; i < wlen; i++) {
        int32_t id;
        if (!vocab_lookup(&t->vocab, word + i, 1, &id)) id = -1;
        i32vec_push(out, id);
    }
    for (int32_t m = 0; m < t->n_merges; m++) {
        int32_t a = t->merges[m].a, b = t->merges[m].b, mid = t->merges[m].merged;
        i32vec ns; i32vec_init(&ns);
        size_t k = 0;
        while (k < out->len) {
            if (k + 1 < out->len && out->data[k] == a && out->data[k + 1] == b) {
                i32vec_push(&ns, mid); k += 2;
            } else { i32vec_push(&ns, out->data[k]); k += 1; }
        }
        i32vec_free(out); *out = ns;
    }
}

static void ids_for_word(BPETokenizer *t, const char *word, int32_t wlen, i32vec *out /* appended to */) {
    int32_t cache_idx;
    if (strmap_get(&t->cache_index, word, wlen, &cache_idx)) {
        i32vec *cached = &t->cache_vecs[cache_idx];
        for (size_t i = 0; i < cached->len; i++) i32vec_push(out, cached->data[i]);
        return;
    }
    i32vec syms;
    apply_merges_ids(t, word, wlen, &syms);
    i32vec resolved; i32vec_init(&resolved);
    for (size_t i = 0; i < syms.len; i++) {
        int32_t sid = syms.data[i];
        i32vec_push(&resolved, sid < 0 ? TOK_UNK : sid);
    }
    i32vec_free(&syms);

    if (t->cache_count == t->cache_cap) {
        t->cache_cap = t->cache_cap ? t->cache_cap * 2 : 64;
        t->cache_vecs = (i32vec *)realloc(t->cache_vecs, sizeof(i32vec) * t->cache_cap);
    }
    t->cache_vecs[t->cache_count] = resolved; /* transfer ownership */
    char *owned = arena_strndup(&t->cache_arena, word, wlen);
    strmap_put(&t->cache_index, owned, wlen, t->cache_count);
    for (size_t i = 0; i < resolved.len; i++) i32vec_push(out, resolved.data[i]);
    t->cache_count++;
}

void mse_tok_encode(BPETokenizer *t, const char *text, int32_t len, i32vec *out) {
    i32vec_push(out, TOK_BOS);
    int32_t nlen;
    char *norm = mse_normalize(text, len, &nlen);
    int32_t i = 0;
    while (i < nlen) {
        while (i < nlen && norm[i] == ' ') i++;
        int32_t start = i;
        while (i < nlen && norm[i] != ' ') i++;
        if (i > start) ids_for_word(t, norm + start, i - start, out);
    }
    free(norm);
}

void mse_tok_encode_for_training(BPETokenizer *t, const char *text, int32_t len, i32vec *out) {
    mse_tok_encode(t, text, len, out);
    i32vec_push(out, TOK_EOS);
}

char *mse_tok_decode(BPETokenizer *t, const int32_t *ids, int32_t n) {
    /* Build the "words" list exactly like decode(): runs of single-char
     * symbols concatenate into one word; multi-char symbols are their
     * own word; PAD/UNK/BOS/EOS flush the current word and are dropped. */
    StrList words; strlist_init(&words);
    char *cur = NULL; int32_t cur_len = 0, cur_cap = 0;
    for (int32_t i = 0; i < n; i++) {
        int32_t id = ids[i];
        if (id == TOK_PAD || id == TOK_UNK || id == TOK_BOS || id == TOK_EOS) {
            if (cur_len) { strlist_push(&words, cur, cur_len); cur_len = 0; }
            continue;
        }
        int32_t tlen;
        const char *tok = vocab_str(&t->vocab, id, &tlen);
        if (tlen == 1) {
            if (cur_len + 1 > cur_cap) { cur_cap = cur_cap ? cur_cap * 2 : 16; cur = (char *)realloc(cur, (size_t)cur_cap); }
            cur[cur_len++] = tok[0];
        } else {
            if (cur_len) { strlist_push(&words, cur, cur_len); cur_len = 0; }
            strlist_push(&words, tok, tlen);
        }
    }
    if (cur_len) strlist_push(&words, cur, cur_len);
    free(cur);

    /* punctuation-aware join */
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
 * Custom flat binary format (magic "MSET", version, then fixed-width
 * records) instead of JSON: no parser needed on load, and every array
 * loads with a single fread() into pre-sized memory. See tools/
 * mse_format.h for the exact layout shared with graph.c's format.
 */
#include "mse_format.h"

int mse_tok_save(const BPETokenizer *t, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    MseTokHeader h = { .magic = MSE_TOK_MAGIC, .version = 1,
                        .vocab_size_target = t->vocab_size,
                        .vocab_count = t->vocab.count,
                        .n_merges = t->n_merges };
    fwrite(&h, sizeof(h), 1, f);
    for (int32_t i = 0; i < t->vocab.count; i++) {
        int32_t l; const char *s = vocab_str(&t->vocab, i, &l);
        fwrite(&l, sizeof(int32_t), 1, f);
        fwrite(s, 1, (size_t)l, f);
    }
    fwrite(t->merges, sizeof(MseMerge), (size_t)t->n_merges, f);
    fclose(f);
    return 0;
}

int mse_tok_load(BPETokenizer *t, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    MseTokHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1 || h.magic != MSE_TOK_MAGIC) { fclose(f); return -1; }
    mse_tok_init(t, h.vocab_size_target);
    for (int32_t i = 0; i < h.vocab_count; i++) {
        int32_t l;
        if (fread(&l, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }
        char *buf = (char *)malloc((size_t)l + 1);
        if (l > 0 && fread(buf, 1, (size_t)l, f) != (size_t)l) { free(buf); fclose(f); return -1; }
        buf[l] = '\0';
        if (i >= TOK_FIRST_FREE) vocab_get_or_create(&t->vocab, buf, l);
        free(buf);
    }
    t->n_merges = h.n_merges;
    t->merges_cap = h.n_merges;
    t->merges = (MseMerge *)malloc(sizeof(MseMerge) * (size_t)(h.n_merges ? h.n_merges : 1));
    if (h.n_merges > 0 && fread(t->merges, sizeof(MseMerge), (size_t)h.n_merges, f) != (size_t)h.n_merges) { fclose(f); return -1; }
    fclose(f);
    return 0;
}
