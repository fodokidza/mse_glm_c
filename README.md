# MSE-GLM in C

A from-scratch C port of MSE-GLM (a deterministic, zero-weight, explainable
graph-based language model). Every module below was ported, then validated
by running your actual, unmodified Python source against the C build's own
output — not by re-deriving the logic from a description of what it does.
See `VALIDATION.md` for the full trail of what was compared, what diverged
and why, and every bug that was found and fixed (in both directions — three
were in the C port, a couple were in the Python-side test scripts written
for validation, and one is a genuine finding in the original Python codebase
itself).

Roughly 5,400 lines of C across 9 modules. Most of that size is deliberate:
explicit CSR arrays, arena allocators, and hash maps where Python reached
for a dict or set, because that reach-for-a-dict-or-set is most of what
made the original correct *and* slow, and this port wanted to keep the
first property while fixing the second.

## Quick start

```sh
make                                    # builds every tool into bin/
./bin/mse_train_corpus \
    --corpus-dir /path/to/txt/files \
    --out ./my_model \
    --vocab-size 4000 --batch-size 20
./bin/mse_chat ./my_model
```

Or, for a single corpus file instead of a folder:

```sh
./bin/mse_model_test corpus.txt 4000 ./my_model   # trains, saves, sanity-checks
./bin/mse_chat ./my_model
```

Inside `mse_chat`: type text to generate a continuation, `/mode strict|open`
to switch inference modes, `/stats` for model stats, `/quit` to exit.

## What this is a port of

MSE-GLM has no learned weights. Everything it does is deterministic:
counting, graph traversal, and set intersection over structures built
directly from training text. That's what makes a faithful C port both
possible and worth doing carefully — there's a ground truth to check
against at every layer, not just at the final output.

| Python module    | C files                              | What it does |
|-------------------|---------------------------------------|---------------|
| `config.py`        | `mse_config.h`                        | Every tunable constant, one place |
| `tokenizer.py`      | `mse_tokenizer.h/.c`                   | BPE tokenizer: normalize, split into sentences, train, encode, decode |
| `graph.py`          | `mse_graph.h/.c`                       | Edge Matrix (bigrams), Bridge Matrix (triples + dual-axis clustering), Relationship Matrix (training sentences) |
| `importance.py`*    | `mse_importance.h/.c`                  | Sequence reconstruction from a relationship id; per-triple trigger lookup |
| `ctm.py`            | `mse_ctm.h/.c`                         | `token_to_relationships`; the Context Trigger Matrix |
| `ivm.py`            | `mse_ivm.h/.c`                         | The 9-layer (V1-V9) Importance Vote Matrix — Open Mode's scoring engine |
| `inference.py`      | `mse_inference.h/.c`                   | Strict Mode's two-stage lineage pipeline; Open Mode; `generate()` |
| `interpret.py`      | `mse_interpret.h/.c`                   | Cluster Interpreter — proposes human-readable labels for clusters |
| `model.py`          | `mse_model.h/.c`                       | Orchestrator: train/save/load/generate, incremental training |
| `train_corpus.py`   | `tools/mse_train_corpus.c`             | Two-pass large-corpus training from a folder of files |
| `chat.py`*          | `tools/mse_chat.c`                     | Interactive REPL |

\* partially ported — see "What's not done" below for the specific pieces
left out of each.

## Design departures from a literal translation

The brief for this project was explicitly "utilize C's capabilities, don't
just translate" — here's what that meant in practice, concretely, not just
as a slogan:

- **`normalize()` never decodes UTF-8.** Python's own regex funnels every
  character down to `[a-z0-9<punct>]` before anything else happens, so any
  byte >= 0x80 is unconditionally "not kept" — this port exploits that and
  scans raw bytes with no decode step at all, branch-free, O(n) over bytes
  instead of over decoded codepoints.
- **BPE merges are vocabulary ids, not strings.** Every symbol that can ever
  appear in a merge rule already has a vocab id by the time it's used, so
  both training and `encode()` compare integers on the hot path instead of
  concatenating and comparing strings.
- **One hash map, not a different dict/set per call site.** Every dedup or
  grouping operation across the whole codebase — pairs, triples, whole
  token sequences — goes through the same generic `StrMap` (bytes -> index),
  by treating any fixed- or variable-width tuple as a raw byte string. One
  implementation to get right instead of five.
- **CSR everywhere a Python list-of-lists would have been.** Edge/Bridge/
  Relationship matrices, `token_to_relationships`, the co-occurrence index,
  the bigram-relationship index — all flat arrays with offset tables, not
  arrays of dynamically-sized Python objects.
- **An arena allocator for anything that lives as long as the model.**
  Vocabulary strings, mostly — bump-allocated in 1MB blocks, freed once as
  a block list, never individually.
- **A flat binary format instead of JSON.** Saving/loading a model is a
  handful of `fwrite`/`fread` calls into pre-sized buffers, not building
  and walking a text tree.
- **V6 (adjacency) and V8 (exact triple) never materialize their own Python-
  side sets.** They query the Edge Matrix / Bridge Matrix CSR directly —
  that data already exists in exactly the needed shape; a second copy of it
  as a hash set would just be duplication with a staleness risk.
- **`bm_build_from_triples`/`em_build_from_pairs`** split the "given an
  already-deduplicated set of triples/pairs, sort+cluster+index it" logic
  out of the normal from-scratch build path specifically so
  `model_merge_graphs`'s incremental-training path could reuse the *exact*
  same code instead of a second, riskier copy of ~80 lines of clustering
  logic.

## What's validated, and how

Every module above (except the explicitly-partial ones' unported pieces)
was checked against your real Python source, not a re-implementation of it,
using one of two methods:

1. **Direct comparison on a shared corpus** (Phase 1: tokenizer + graphs).
2. **A Python loader for the C port's own binary output**
   (`validation/mse_binary_loader.py`), so `ctm.py`/`ivm.py`/`inference.py`/
   `interpret.py`/`model.py` — completely unmodified — run against
   *identical* underlying graph data as the C build. This was necessary
   starting in Phase 2 because Phase 1 uncovered that Python's own BPE
   training isn't reproducible run-to-run (see below), which would
   otherwise make "train the same corpus on both sides and diff" a
   comparison against a moving target.

Concretely, this means: full V1-V9 IVM score arrays compared to 6 decimal
places; every Strict Mode rule/stage label reproduced exactly; cluster
interpretations, evidence signals, and sort order identical; full
`train()` -> `build_context_triggers()` -> `build_importance_votes()` ->
`save()` -> `load()` -> `generate()` pipelines byte-identical end to end,
including two rounds of `train_incremental()`. All of this on two corpora
(a small hand-written one, and your actual README.md at ~2000 vocab), plus
a synthetic 18MB/200-file/159K-sentence corpus for `train_corpus`
specifically (see below).

Every test tool used for this is in `tools/*_test.c` and the matching
Python scripts are in `validation/*.py` — you can rerun any of these
comparisons yourself; see `validation/README.md`.

Every test tool is also clean under `-fsanitize=address,undefined` — no
leaks, no use-after-free, no undefined behavior, checked at every phase,
not just at the end.

### Two known, honest divergences (not bugs — see VALIDATION.md for the full writeup)

1. **BPE training tie-breaks.** Python's `tokenizer.py` uses a plain `set()`
   to track which words contain a given symbol pair; when two pairs tie for
   "most frequent" during training, the tie-break depends on Python's
   per-process string-hash randomization. Verified directly: the same
   corpus trains a *different* vocabulary under different `PYTHONHASHSEED`
   values. This isn't rare — it showed up on a 780-line real corpus, not
   just adversarial input. The C port makes a different, but *fully
   reproducible*, choice instead of trying to replicate something that
   doesn't have one canonical answer to replicate.
2. **Incremental-merge array ordering.** Similarly, Python's `_merge_graphs`
   builds merged pair/triple sets via `set(old) | set(new)`, whose
   iteration order (before its final sort) depends on CPython's int-tuple
   hashing. This one's provably harmless rather than merely hard to avoid:
   every real consumer of within-row order either explicitly re-sorts
   (Strict Mode's Stage 1) or scans a whole row for an exact match
   (`frequency()`, `bm_has_triple()`) rather than relying on position — so
   the divergence is invisible to every query this codebase makes, and was
   confirmed invisible in practice too (`model_train_incremental`'s output
   matched Python's byte-for-byte across two merge rounds in testing).

## Large-corpus training

`mse_train_corpus` is a two-pass pipeline, same shape as `train_corpus.py`:

```sh
./bin/mse_train_corpus \
    --corpus-dir data/ \
    --out runs/big_model \
    --vocab-size 8000 \
    --batch-size 20
```

**Pass 1** builds one shared vocabulary by streaming word frequencies across
every `.txt` file under `--corpus-dir` (recursive by default) — only the
current file's text and the cumulative word-count table are ever in memory
together, never the whole corpus's raw text at once.

**Pass 2** builds the graph in batches of `--batch-size` files: the first
batch is a from-scratch build, every batch after that merges in via the
same incremental-training machinery `model_train_incremental` uses.

**This was tested on a real large run**, not just the unit-scale corpora
used for correctness validation: an 18MB synthetic corpus across 200 files
(generated with a template+substitution script, not real prose), producing
a 226-token vocabulary, 159,525 unique training sentences, and 2.55M
relationship rows. It completed successfully and was memory-clean under
ASan on a reduced slice. Two things worth knowing before pointing this at
something huge, both inherited directly from `train_corpus.py`'s own
design, not artifacts of the C port:

- **The per-batch merge is a full recompute over everything seen so far**
  (same reasoning as `train_incremental`: a cluster can only be discovered
  once every triple in it is known, so it can't be computed incrementally/
  append-only). Total cost across N batches is closer to
  O(N x corpus_size_so_far) than O(corpus_size) — confirmed directly: the
  same 18MB corpus took 76s at `--batch-size 20` (10 merge passes) and 20s
  at `--batch-size 200` (a single from-scratch build, no merging at all).
  **Start with a bigger `--batch-size` than you think you need** for
  genuinely large corpora; `--batch-size 1` is the safest memory profile,
  not the fastest one.
- **Pass 2 reads each file's full text into memory at once** (not chunked
  within a file). This assumes a large corpus split across many
  reasonably-sized files, which is the scenario both this and the Python
  original were built for — if one file is itself enormous, pre-split it.

## Build

```sh
make            # builds every tool into bin/
make clean
```

Tools built:

| Binary | Purpose |
|---|---|
| `mse_train` | Trains just the tokenizer+graphs (Phase 1) from one corpus file; `--dump` prints everything as tab-separated lines |
| `mse_ctm_test` | Dumps `token_to_relationships` + Context Trigger Matrix signatures |
| `mse_ivm_test` | Dumps important tokens, full V1-V9 score breakdowns, `resolve_tie` samples |
| `mse_inference_test` | Runs `generate()` in both modes from real training-sentence prefixes |
| `mse_interpret_test` | Dumps cluster interpretations, the CI Matrix, zero-cluster groups |
| `mse_model_test` | Full pipeline: train -> save -> load -> generate, both modes, both opt-ins |
| `mse_incremental_test` | Runs `train_incremental`/`model_merge_graphs` across several corpora |
| `mse_train_corpus` | The two-pass large-corpus pipeline described above |
| `mse_chat` | Interactive REPL |

No external dependencies — standard C11 and POSIX (`dirent.h`, `sys/stat.h`
for directory traversal in `mse_train_corpus`). Tested with gcc on Linux.

## Layout

```
include/
  mse_config.h       constants (mirrors config.py)
  mse_util.h         i32vec (growable array), Arena, StrMap, PairMap
  mse_tokenizer.h     Vocab, BPETokenizer, normalize/split_sentences, MseWordFreq
  mse_graph.h         EdgeMatrix, BridgeMatrix, RelationshipMatrix
  mse_importance.h    sequence_for_relationship, _trigger_for_triple
  mse_ctm.h           TokenRels (token_to_relationships), ContextTriggerMatrix
  mse_ivm.h           ImportanceVoteMatrix (V1-V9 scoring)
  mse_inference.h     InferenceEngine (Strict/Open Mode)
  mse_interpret.h     Cluster Interpreter
  mse_model.h         MSEGraphLanguageModel orchestrator
  mse_format.h        on-disk binary layout
src/
  *.c                 implementations, one file per header above
tools/
  mse_train.c, mse_ctm_test.c, mse_ivm_test.c, mse_inference_test.c,
  mse_interpret_test.c, mse_model_test.c, mse_incremental_test.c,
  mse_train_corpus.c, mse_chat.c
tests/
  reload_test.c       binary save/load round-trip smoke test
validation/
  mse_binary_loader.py, *_dump.py   the Python-side half of every comparison
  README.md                         how to rerun any of it yourself
VALIDATION.md          the full trail: what was compared, what was found
```

## What's not done

**`train_incremental`/`train_corpus` are now done** (this was the explicit
ask that prompted finishing this layer) — see above for both.

Still not ported, roughly in order of how much it'd matter if you needed it:

- **`server.py`'s HTTP layer.** This is the one piece deliberately left
  alone rather than attempted under time pressure: Python's version is a
  thin JSON API wrapper around `model.py` using Flask, which handles all
  HTTP/1.1 parsing, routing, and streaming-response semantics for free. A
  C equivalent means hand-rolling that over raw POSIX sockets — a
  substantial, genuinely new piece of systems code (not a port of existing
  logic) with real security surface (HTTP request parsing) that deserves
  its own careful, from-scratch validation pass rather than being rushed
  in alongside everything else. Everything it would call into
  (`model.py`'s API) is already ported and validated — what's missing is
  purely the HTTP transport on top.
- **`chat.py`'s remaining introspection commands** (`/scores`, `/bigram`,
  `/cache`, `/shared`, `/similarity`, `/explain`, `/clusters`). The logic
  behind every one of these is already ported and validated by the
  dedicated test tools (`mse_ivm_test`, `mse_ctm_test`, `mse_interpret_test`)
  — what's missing is only the REPL argument-parsing glue around calls this
  codebase already makes correctly elsewhere.
- **`analyse.py`** (a CLI for the same interpret.py/importance.py analysis
  functions, formatted as reports). Same situation: the underlying logic
  exists and is validated; this would be presentation/CLI plumbing on top.
- **`importance.py`'s read-only analysis functions**
  (`important_tokens_in_sequence`, `trigger_matrix`, `expected_importance`)
  — reporting/introspection, not on any code path anything else depends on.
- **IVM's opt-in sparse score cache** (`build_cache`/`enable_cache`). A pure
  performance optimization; the Python docstring itself guarantees it
  produces numerically identical output to the live path, which is what
  this port implements directly. Worth adding if profiling on a real
  workload shows `score_candidates` as a bottleneck, not before.

## Performance (informal — see VALIDATION.md for exact numbers per phase)

At the corpus sizes used for validation (up to ~2000 vocab / ~1000 lines),
every ported module ran faster than the Python original except one:
tokenizer+graph training ran ~4-8x faster, CTM+IVM construction ~6x faster,
`generate()` ~8-11x faster, the full `model.py` pipeline ~8.8x faster. The
one exception is `interpret.c` (the Cluster Interpreter), which ran
slightly *slower* than Python on the larger corpus — it's a read-only
offline analysis pass, never on the generation hot path, and the current
implementation allocates a fresh hash map per cluster rather than reusing
one; a real target if that module's own performance ever matters, not
hidden in these notes. None of this is a rigorous benchmark — no warm-up,
single runs, and only the corpus sizes validation happened to use — treat
it as "this isn't accidentally quadratic somewhere Python wasn't," not as
a performance guarantee at any particular scale.
