/* mse_format.h — on-disk binary layout for saved tokenizer/graph state.
 *
 * Deliberately NOT JSON: every array is a fixed-width record dumped
 * with a single fwrite()/fread(), so loading a model is a handful of
 * bulk reads into pre-sized buffers instead of parsing a text tree —
 * an explicit example of "using C's capabilities" rather than
 * replicating json.dump/json.load. Endianness: this format is written
 * and read on the same architecture family (no byte-swapping is
 * performed); that matches the Python original's own portability
 * scope (it never claimed cross-machine binary portability either —
 * its "binary" format was JSON text, which sidesteps endianness by
 * not being binary at all. If cross-endian portability is ever
 * needed, add explicit htole32-style conversions at the fwrite/fread
 * call sites — the struct layouts below don't need to change).
 */
#ifndef MSE_FORMAT_H
#define MSE_FORMAT_H

#include <stdint.h>

#define MSE_TOK_MAGIC   0x4D534554u /* "MSET" */
#define MSE_EDGE_MAGIC  0x4D534545u /* "MSEE" */
#define MSE_BRIDGE_MAGIC 0x4D534542u /* "MSEB" */
#define MSE_REL_MAGIC   0x4D534552u /* "MSER" */

typedef struct {
    uint32_t magic;
    uint32_t version;         /* 1 = BPE-era format (no longer written by
                                * this build); 2 = two-stage char/word
                                * format (current — see mse_tokenizer.h) */
    int32_t  vocab_size_target;
    int32_t  vocab_count;
    int32_t  n_merges;        /* vestigial: BPE-only field, always 0 in a
                                * version-2 file. Kept (not removed) so the
                                * header's on-disk byte layout doesn't shift
                                * — nothing about version 2 depends on it. */
} MseTokHeader;

typedef struct {
    uint32_t magic;
    uint32_t version;
    int32_t  vocab_size;
    int32_t  n_edges;      /* length of src/dst/count arrays */
    int32_t  index_len;    /* vocab_size + 1 */
} MseEdgeHeader;

typedef struct {
    uint32_t magic;
    uint32_t version;
    int32_t  vocab_size;
    int32_t  n_triples;
    int32_t  index_len;    /* vocab_size + 1 */
    int32_t  n_tindex_entries; /* number of (token -> cluster list) rows */
} MseBridgeHeader;

typedef struct {
    uint32_t magic;
    uint32_t version;
    int32_t  n_rels;
    int32_t  n_rows;       /* length of r_triple/r_rel arrays */
    int32_t  index_len;    /* n_rels + 1 */
} MseRelHeader;

#endif /* MSE_FORMAT_H */
