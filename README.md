# MSE-GLM in C

A from-scratch C port of MSE-GLM — a deterministic, zero-weight, explainable
graph-based language model. There are no learned weights: everything is
counting, graph traversal, and set intersection over structures built
directly from training text. Every module was ported from the original
Python source and then validated against it; see [`VALIDATION.md`](VALIDATION.md)
for the full trail and the "Recent changes" section below for what's changed
since that trail was written (tokenizer rewrite, V10 noise layer).

This document is a complete reference for every command-line tool in the
project — what it does, every flag and subcommand, and what it prints.

---

## Contents

- [Build](#build)
- [Quick start](#quick-start)
- [Command reference](#command-reference)
  - [`mse_train_corpus`](#mse_train_corpus) — train from a folder of files
  - [`mse_model_test`](#mse_model_test) — train from one file, full pipeline
  - [`mse_train`](#mse_train) — tokenizer + graphs only, debug dump
  - [`mse_chat`](#mse_chat) — interactive REPL
  - [`mse_incremental_test`](#mse_incremental_test) — incremental training over several corpora
  - [`mse_ctm_test`](#mse_ctm_test) — Context Trigger Matrix dump
  - [`mse_ivm_test`](#mse_ivm_test) — Importance Vote Matrix dump
  - [`mse_inference_test`](#mse_inference_test) — `generate()` in both modes
  - [`mse_interpret_test`](#mse_interpret_test) — Cluster Interpreter dump
- [On-disk model formats](#on-disk-model-formats)
- [Module map](#module-map)
- [Project layout](#project-layout)
- [Recent changes (tokenizer rewrite + V10)](#recent-changes-tokenizer-rewrite--v10)
- [Design departures from a literal translation](#design-departures-from-a-literal-translation)
- [What's validated, and how](#whats-validated-and-how)
- [What's not done](#whats-not-done)
- [Performance](#performance-informal)

---

## Build

```sh
make            # builds every tool below into bin/
make clean      # removes build artifacts and bin/
make test       # builds, then runs mse_train on sample.txt as a smoke test
```

No external dependencies — standard C11 and POSIX (`dirent.h`, `sys/stat.h`
for directory traversal in `mse_train_corpus`). Tested with gcc on Linux.
Override the compiler/flags with `CC` and `CFLAGS`, e.g. `make CC=clang`.

## Quick start

Train on a folder of `.txt` files and chat with the result:

```sh
make
./bin/mse_train_corpus --corpus-dir /path/to/txt/files --out ./my_model \
    --vocab-size 4000 --batch-size 20
./bin/mse_chat ./my_model
```

Or, for a single corpus file instead of a folder:

```sh
./bin/mse_model_test corpus.txt 4000 ./my_model   # trains, saves, sanity-checks
./bin/mse_chat ./my_model
```

---

## Command reference

Nine binaries build into `bin/`. Five (`mse_ctm_test`, `mse_ivm_test`,
`mse_inference_test`, `mse_interpret_test`, and `mse_train`'s own output)
work off a **prefix** produced by `mse_train` — four files named
`<prefix>.tok`, `<prefix>.edges`, `<prefix>.bridges`, `<prefix>.rels`. The
other four (`mse_train_corpus`, `mse_model_test`, `mse_incremental_test`,
`mse_chat`) work off a **model folder** produced by `model_save()` — see
[On-disk model formats](#on-disk-model-formats) for the difference.

### `mse_train_corpus`

Two-pass pipeline for training on a **folder** of many `.txt` files, the
tool you want for anything larger than a single file. C port of
`train_corpus.py`.

```sh
mse_train_corpus --corpus-dir <dir> --out <dir> [options]
```

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--corpus-dir <dir>` | yes | — | Folder to read `.txt` files from |
| `--out <dir>` | yes | — | Folder to write the trained model into (`model_save()` layout) |
| `--vocab-size N` | no | `2000` | Tokenizer vocabulary cap |
| `--batch-size N` | no | `10` | Files merged per training batch (must be ≥ 1) |
| `--no-recursive` | no | recursive | Only look at top-level files in `--corpus-dir`, don't descend into subfolders |
| `--quiet` | no | off | Suppress progress output |

```sh
./bin/mse_train_corpus \
    --corpus-dir data/ \
    --out runs/big_model \
    --vocab-size 8000 \
    --batch-size 20
```

**How it works:**

1. **Pass 1 (vocabulary).** Streams word frequencies file by file across
   every `.txt` file found under `--corpus-dir` — only the current file's
   text and the cumulative word-count table are ever in memory together,
   never the whole corpus at once. One tokenizer is trained from the
   combined counts.
2. **Pass 2 (graph).** Processes files in batches of `--batch-size`. The
   first batch is a from-scratch build (`model_build_graphs`); every batch
   after that merges in via the same incremental machinery
   `model_train_incremental` uses (`model_merge_graphs`).

**Cost profile to know before pointing this at something huge:** the
per-batch merge is a full recompute over everything seen so far (clusters
can only be discovered once every triple in them is known — not
incrementally appendable), so total cost is closer to
`O(batches × corpus_size_so_far)` than `O(corpus_size)`. A bigger
`--batch-size` trades memory for speed; `--batch-size 1` is the safest
memory profile, not the fastest. Also, each file's full text is read into
memory at once during Pass 2 (not chunked within a file) — pre-split any
single file that's itself enormous.

Verified on an 18MB/200-file/159,525-sentence synthetic corpus: 76s at
`--batch-size 20` vs. 20s at `--batch-size 200` (a single from-scratch
build, no merging).

### `mse_model_test`

Full single-file pipeline: train → save → load → generate, in both modes,
plus a save/load stats-match check. Good for a quick correctness smoke test
on one corpus file, and the simplest way to produce a model folder for
`mse_chat` from a single file.

```sh
mse_model_test <corpus.txt> <vocab_size> <save_folder>
```

| Argument | Meaning |
|---|---|
| `<corpus.txt>` | Path to a single training-text file |
| `<vocab_size>` | Tokenizer vocabulary cap |
| `<save_folder>` | Folder to save the model into |

```sh
./bin/mse_model_test sample.txt 2000 /tmp/my_model
```

Prints stats after training, confirms the save/load round trip preserves
them (`stats_match=1`), then runs five fixed sample prompts through both
Strict and Open Mode, on the **reloaded** model — CTM/IVM are not
persisted, so they're rebuilt after load before generation.

### `mse_train`

Trains just the tokenizer and the three graphs (Edge, Bridge, Relationship)
from one corpus file — the "Phase 1" layer, without CTM/IVM/inference on
top. Writes a `<prefix>.tok/.edges/.bridges/.rels` set of files, the input
format the four `*_test` tools below expect. Optionally dumps everything as
tab-separated lines for diffing against a Python reference dump.

```sh
mse_train <corpus.txt> <out_prefix> [vocab_size] [--dump]
```

| Argument | Default | Meaning |
|---|---|---|
| `<corpus.txt>` | — | Path to training text |
| `<out_prefix>` | — | Output prefix; writes `<out_prefix>.tok/.edges/.bridges/.rels` |
| `[vocab_size]` | `2000` | Tokenizer vocabulary cap |
| `[--dump]` | off | Print vocab, edges, bridges, relationships as tab-separated lines |

```sh
./bin/mse_train sample.txt /tmp/mse_model 500
./bin/mse_train sample.txt /tmp/mse_model 500 --dump > /tmp/dump.tsv
```

Always prints stats (`vocab_size`, `sentences`, `edges`, `bridges`,
`clustered_bridges`, `clusters`, `relationships`, `relationship_rows`,
`relationship_occurrences`) and a round-trip decode of the first 5
sentences, regardless of `--dump`.

### `mse_chat`

Interactive REPL against a trained model folder (from `mse_train_corpus`,
`mse_model_test`, or `mse_incremental_test`).

```sh
mse_chat <model_folder> [max_tokens]
```

| Argument | Default | Meaning |
|---|---|---|
| `<model_folder>` | — | Folder written by `model_save()` |
| `[max_tokens]` | `30` | Tokens generated per turn |

```sh
./bin/mse_chat ./my_model
./bin/mse_chat ./my_model 50
```

On startup it prints the loaded model's stats, then drops into a prompt
(`strict>` or `open>`) that reads one line at a time.

**Subcommands** (anything not matching one of these is treated as a prompt
and generates a continuation):

| Subcommand | Effect |
|---|---|
| `/mode strict` | Switch to Strict Mode (deterministic two-stage lineage pipeline) |
| `/mode open` | Switch to Open Mode (V1–V10 Importance Vote Matrix scoring) |
| `/stats` | Print `vocab_size`, `edges`, `bridges`, `clustered_bridges`, `clusters`, `relationships`, `relationship_rows`, `relationship_occurrences` |
| `/quit` or `/exit` | Exit the REPL |

> The Python original (`chat.py`) also has `/scores`, `/bigram`, `/cache`,
> `/shared`, `/similarity`, `/explain`, and `/clusters`. Their underlying
> logic is fully ported and validated — it's exercised by `mse_ivm_test`,
> `mse_ctm_test`, and `mse_interpret_test` below — but the REPL
> argument-parsing glue for them hasn't been added yet. See
> [What's not done](#whats-not-done).

### `mse_incremental_test`

Trains on one corpus, then merges in one or more additional corpus files
one at a time via `model_train_incremental`, printing stats before/after
each merge. Useful for exercising and inspecting incremental training
directly (as opposed to `mse_train_corpus`'s batched version of the same
machinery).

```sh
mse_incremental_test <corpus1.txt> <corpus2.txt> [corpus3.txt ...]
```

```sh
./bin/mse_incremental_test sample.txt sample2.txt sample3.txt
```

For each additional corpus it prints `sentences_added`, `vocab_added`,
`ctm_invalidated`, and full stats before/after that merge, then finishes
with the same five fixed prompts as `mse_model_test`, run in both modes on
the final, fully-merged model.

### `mse_ctm_test`

Dumps `token_to_relationships` and the Context Trigger Matrix for a model
already saved by `mse_train` (prefix form, **not** a `model_save()` folder).

```sh
mse_ctm_test <prefix> <vocab_size>
```

```sh
./bin/mse_train sample.txt /tmp/mse_model 500
./bin/mse_ctm_test /tmp/mse_model 500
```

Prints three sections: `--- TOKEN_RELS ---` (token → relationship-id list),
`--- CTM_SIGS ---` (per-cluster trigger signatures, member/trigger/support
tuples), and `--- RESOLVE_TIE_SAMPLE ---` (spot checks of the tie-breaking
logic between tokens sharing 2+ clusters).

### `mse_ivm_test`

Dumps the full V1–V10 Importance Vote Matrix for a model saved by
`mse_train`.

```sh
mse_ivm_test <prefix> <vocab_size>
```

```sh
./bin/mse_ivm_test /tmp/mse_model 500
```

Prints `--- IMPORTANT ---` (tokens flagged important), `---
SCORE_CANDIDATES ---` (up to 15 realistic `(context, current, previous)`
scenarios reconstructed from real training sentences, with per-candidate
scores and the selected winner), and `--- RESOLVE_TIE ---` (tie-break spot
checks).

### `mse_inference_test`

Runs `generate()` in both Strict and Open Mode from real training-sentence
prefixes, with and without the Context Trigger Matrix, on a model saved by
`mse_train`.

```sh
mse_inference_test <prefix> <vocab_size>
```

```sh
./bin/mse_inference_test /tmp/mse_model 500
```

For each of `strict`/`open` × `use_ctm=0/1`, runs up to 20 prompts (the
first 1–3 tokens of real training relationships, prefixed with `TOK_BOS`)
through `ie_generate` for 12 tokens and prints the resulting id sequence.

### `mse_interpret_test`

Dumps the Cluster Interpreter's output for a model saved by `mse_train`:
per-cluster label candidates, coverage, and evidence.

```sh
mse_interpret_test <prefix> <vocab_size>
```

```sh
./bin/mse_interpret_test /tmp/mse_model 500
```

Prints `--- INTERPRET_CLUSTER ---` (top-5 candidate interpretations per
cluster, one call per cluster id), `--- INTERPRET_ALL_CLUSTERS ---` (the
batched equivalent, `interpret_all_clusters`), and `---
BUILD_INTERPRETER_MATRIX ---` (the full CI Matrix).

---

## On-disk model formats

Two different layouts exist, and the tools aren't interchangeable across
them:

- **Prefix form** — written by `mse_train`, read by `mse_ctm_test`,
  `mse_ivm_test`, `mse_inference_test`, `mse_interpret_test`:
  `<prefix>.tok`, `<prefix>.edges`, `<prefix>.bridges`, `<prefix>.rels`.
- **Folder form** — written by `model_save()` (used by `mse_train_corpus`,
  `mse_model_test`, `mse_incremental_test`), read by `mse_chat` and
  `model_load()`: `<folder>/tokenizer.tok`, `<folder>/edges.bin`,
  `<folder>/bridges.bin`, `<folder>/relationships.bin`.

Both are flat binary formats (`fwrite`/`fread` into pre-sized buffers, no
text tree to build or walk) — see `mse_format.h`. Neither stores the
Context Trigger Matrix or Importance Vote Matrix; both are rebuilt after
load (`model_build_context_triggers` / `model_build_importance_votes`), and
`mse_chat` rebuilds them for you at startup implicitly by calling
`model_generate`, which triggers the lazy CTM/IVM build paths as needed.

The **tokenizer format is version 2** (see `mse_format.h`) — models saved
by an older build of this port are not loadable here, and vice versa (see
[Recent changes](#recent-changes-tokenizer-rewrite--v10)).

---

## Module map

| Python module | C files | What it does |
|---|---|---|
| `config.py` | `mse_config.h` | Every tunable constant, one place |
| `tokenizer.py` | `mse_tokenizer.h/.c` | Two-stage char/word tokenizer: normalize, split into sentences, train, encode, decode |
| `graph.py` | `mse_graph.h/.c` | Edge Matrix (bigrams), Bridge Matrix (triples + dual-axis clustering), Relationship Matrix (training sentences) |
| `importance.py`* | `mse_importance.h/.c` | Sequence reconstruction from a relationship id; per-triple trigger lookup |
| `ctm.py` | `mse_ctm.h/.c` | `token_to_relationships`; the Context Trigger Matrix |
| `ivm.py` | `mse_ivm.h/.c` | The 10-layer (V1–V10) Importance Vote Matrix — Open Mode's scoring engine |
| `noise.py` | `mse_noise.h/.c` | V10's data source: three-stage noise-cancellation scoring |
| `inference.py` | `mse_inference.h/.c` | Strict Mode's two-stage lineage pipeline; Open Mode; `generate()` |
| `interpret.py` | `mse_interpret.h/.c` | Cluster Interpreter — proposes human-readable labels for clusters |
| `model.py` | `mse_model.h/.c` | Orchestrator: train/save/load/generate, incremental training |
| `train_corpus.py` | `tools/mse_train_corpus.c` | Two-pass large-corpus training from a folder of files |
| `chat.py`* | `tools/mse_chat.c` | Interactive REPL |

\* partially ported — see [What's not done](#whats-not-done).

---

## Project layout

```
include/
  mse_config.h       constants (mirrors config.py)
  mse_util.h          i32vec (growable array), Arena, StrMap, PairMap
  mse_tokenizer.h      CharVocab, WordVocab, MseTokenizer, normalize/segment/split_sentences
  mse_graph.h          EdgeMatrix, BridgeMatrix, RelationshipMatrix
  mse_importance.h     sequence_for_relationship, _trigger_for_triple
  mse_ctm.h            TokenRels (token_to_relationships), ContextTriggerMatrix
  mse_ivm.h            ImportanceVoteMatrix (V1-V10 scoring)
  mse_noise.h          NoiseIndex (V10's data source)
  mse_inference.h      InferenceEngine (Strict/Open Mode)
  mse_interpret.h      Cluster Interpreter
  mse_model.h          MSEGraphLanguageModel orchestrator
  mse_format.h         on-disk binary layout
src/
  *.c                  implementations, one file per header above
tools/
  mse_train.c, mse_ctm_test.c, mse_ivm_test.c, mse_inference_test.c,
  mse_interpret_test.c, mse_model_test.c, mse_incremental_test.c,
  mse_train_corpus.c, mse_chat.c
tests/
  reload_test.c        binary save/load round-trip smoke test (not in `make all`;
                        build/run it manually against /tmp/mse_model2.*)
validation/
  mse_binary_loader.py, *_dump.py    the Python-side half of every comparison
  README.md                          how to rerun any of it yourself
VALIDATION.md           the full trail: what was compared, what was found
```

---

## Recent changes (tokenizer rewrite + V10)

The Python codebase moved on in two ways this port had fallen behind on;
both are now current:

1. **Tokenizer: BPE → two-stage character/word.** Python's `tokenizer.py`
   deleted BPE outright and replaced it with `CharWordTokenizer` (stage 1:
   every character gets an id; stage 2: common words get their own id on
   top of that, case-insensitively, capped by `vocab_size`). This port's
   `mse_tokenizer.h`/`tokenizer.c` were rewritten to match — a from-scratch
   reimplementation against the current Python source, not a translation of
   the old BPE files. `WORD_BOUND` (id 4) is a new reserved token;
   `TOK_FIRST_FREE` moved from 4 to 5. The on-disk tokenizer format bumped
   to version 2 — **old saved models are not loadable by this build**, and
   vice versa. `PairMap` (`mse_util.h`), BPE's merge-selection structure, is
   now dead code (kept, not deleted) since nothing calls it anymore.
2. **IVM: V1–V9 → V1–V10.** `ivm.py` gained a tenth weighted-voting layer
   sourced from `noise.py`'s three-stage noise-cancellation scoring. This
   port adds it as `mse_noise.h`/`noise.c` plus the corresponding
   `ivm.c`/`model.c` wiring: V10 is attached lazily, once, on the first Open
   Mode `model_generate()` call (`model_ensure_noise_layer()`), and
   correctly invalidated and rebuilt after every graph rebuild (train /
   `train_incremental` / load) so it can never score from a stale
   pre-merge structure. Unlike an earlier Python revision's
   per-`(anchor, candidate)` formula, this only implements noise.py's
   current, anchor-independent closed form — a candidate's noise score is
   a property of the candidate alone, computed once and cached, not once
   per context token. `IVMConfig`'s weights moved with `config.py` in the
   same pass: `INFLUENCE_WEIGHT`/`CONTEXT_INFLUENCE_WEIGHT` (V2/V4) are now
   `0.0`, and `NOISE_WEIGHT` (V10) defaults to `0.0001`.

**How this was verified** (lighter than `VALIDATION.md`'s original
methodology — no line-by-line Python-vs-C diff harness was rebuilt this
pass): the full existing test suite (`mse_ctm_test`, `mse_ivm_test`,
`mse_inference_test`, `mse_interpret_test`, `mse_model_test`,
`mse_incremental_test`, `reload_test`, `mse_train_corpus`) runs clean under
`-fsanitize=address,undefined` against the real sample corpora
(`sample.txt`/`sample2.txt`/`sample3.txt`) — zero leaks, zero UB, zero
crashes. Round-trip `encode`/`decode` was checked by hand against the
Python tokenizer's own segmentation rules. V10 was checked to (a)
contribute exactly 0 until `model_ensure_noise_layer()` runs, (b) change a
real fraction of candidate scores once attached, (c) leave `generate()`
deterministic, and (d) get correctly dropped and rebuilt across
`model_train_incremental()`.

**What this does NOT include:** a bit-for-bit score comparison against a
live Python model, the kind `VALIDATION.md`'s Phase 1/2 did for every other
module. If exact numeric parity with Python matters for your use, budget
time to rebuild that comparison harness for these two modules.

---

## Design departures from a literal translation

The brief for this project was explicitly "utilize C's capabilities, don't
just translate":

- **`normalize()` never decodes UTF-8.** Python's own regex funnels every
  character down to `[a-z0-9<punct>]`, so any byte ≥ 0x80 is unconditionally
  "not kept" — this port scans raw bytes with no decode step, branch-free,
  O(n) over bytes instead of decoded codepoints.
- **BPE merges are vocabulary ids, not strings.** (Historical note — see
  "Recent changes": BPE itself has since been removed.) Both training and
  `encode()` compared integers on the hot path instead of concatenating and
  comparing strings.
- **One hash map, not a different dict/set per call site.** Every dedup or
  grouping operation across the codebase — pairs, triples, whole token
  sequences — goes through the same generic `StrMap` (bytes → index).
- **CSR everywhere a Python list-of-lists would have been.** Edge/Bridge/
  Relationship matrices, `token_to_relationships`, the co-occurrence index,
  the bigram-relationship index — all flat arrays with offset tables.
- **An arena allocator for anything that lives as long as the model.**
  Vocabulary strings — bump-allocated in 1MB blocks, freed once as a block
  list.
- **A flat binary format instead of JSON.** Saving/loading a model is a
  handful of `fwrite`/`fread` calls into pre-sized buffers.
- **V6 (adjacency) and V8 (exact triple) never materialize their own
  Python-side sets.** They query the Edge Matrix / Bridge Matrix CSR
  directly.
- **`bm_build_from_triples`/`em_build_from_pairs`** split the "given an
  already-deduplicated set, sort+cluster+index it" logic out of the normal
  from-scratch build path so `model_merge_graphs`'s incremental path can
  reuse the exact same code instead of a second, riskier copy.

---

## What's validated, and how

Every module (except the explicitly-partial ones' unported pieces) was
checked against the real Python source, not a re-implementation of it,
using one of two methods:

1. **Direct comparison on a shared corpus** (Phase 1: tokenizer + graphs).
2. **A Python loader for the C port's own binary output**
   (`validation/mse_binary_loader.py`), so `ctm.py`/`ivm.py`/`inference.py`/
   `interpret.py`/`model.py` — completely unmodified — run against
   identical underlying graph data as the C build. This was necessary from
   Phase 2 on because Phase 1 uncovered that Python's own BPE training
   isn't reproducible run-to-run (see below).

Concretely: full V1–V9 IVM score arrays compared to 6 decimal places; every
Strict Mode rule/stage label reproduced exactly; cluster interpretations,
evidence signals, and sort order identical; full `train()` →
`build_context_triggers()` → `build_importance_votes()` → `save()` →
`load()` → `generate()` pipelines byte-identical end to end, including two
rounds of `train_incremental()`. Tested on two corpora (a small hand-written
one, and the project's own README at ~2000 vocab), plus a synthetic
18MB/200-file/159K-sentence corpus for `train_corpus` specifically.

Every test tool used for this is in `tools/*_test.c`, with matching Python
scripts in `validation/*.py` — rerun any comparison yourself; see
`validation/README.md`. Every test tool is also clean under
`-fsanitize=address,undefined` — no leaks, no use-after-free, no undefined
behavior, checked at every phase.

**Two known, honest divergences** (not bugs — full writeup in
`VALIDATION.md`):

1. **BPE training tie-breaks** (historical — BPE has since been removed;
   see "Recent changes"). Python's old tie-break depended on
   `PYTHONHASHSEED`-driven string-hash randomization; the same corpus
   trained a different vocabulary under different seeds. The C port made a
   different, but fully reproducible, choice instead of replicating
   something with no canonical answer.
2. **Incremental-merge array ordering.** Python's `_merge_graphs` builds
   merged pair/triple sets via `set(old) | set(new)`, whose iteration order
   depends on CPython's int-tuple hashing. Provably harmless: every real
   consumer of within-row order either explicitly re-sorts (Strict Mode's
   Stage 1) or scans the whole row for an exact match (`frequency()`,
   `bm_has_triple()`) rather than relying on position — confirmed invisible
   in practice too (`model_train_incremental`'s output matched Python's
   byte-for-byte across two merge rounds).

---

## What's not done

**`train_incremental`/`train_corpus` are done** — see
[`mse_train_corpus`](#mse_train_corpus) and
[`mse_incremental_test`](#mse_incremental_test) above.

Still not ported, roughly in order of how much it'd matter if you needed
it:

- **`server.py`'s HTTP layer.** Python's version is a thin JSON API wrapper
  around `model.py` using Flask, which handles HTTP/1.1 parsing, routing,
  and streaming for free. A C equivalent means hand-rolling that over raw
  POSIX sockets — a substantial, genuinely new piece of systems code (not a
  port of existing logic) with real security surface (HTTP request
  parsing) that deserves its own careful, from-scratch validation pass.
  Everything it would call into (`model.py`'s API) is already ported and
  validated — what's missing is purely the HTTP transport on top.
- **`chat.py`'s remaining introspection commands** (`/scores`, `/bigram`,
  `/cache`, `/shared`, `/similarity`, `/explain`, `/clusters`). The logic
  behind every one is already ported and validated by the dedicated test
  tools (`mse_ivm_test`, `mse_ctm_test`, `mse_interpret_test`) — what's
  missing is only the REPL argument-parsing glue around calls this
  codebase already makes correctly elsewhere.
- **`analyse.py`** — a CLI for the same `interpret.py`/`importance.py`
  analysis functions, formatted as reports. Same situation: the underlying
  logic exists and is validated; this would be presentation/CLI plumbing.
- **`importance.py`'s read-only analysis functions**
  (`important_tokens_in_sequence`, `trigger_matrix`, `expected_importance`)
  — reporting/introspection, not on any code path anything else depends on.
- **IVM's opt-in sparse score cache** (`build_cache`/`enable_cache`). A pure
  performance optimization; the Python docstring guarantees numerically
  identical output to the live path, which is what this port implements
  directly. Worth adding if profiling shows `score_candidates` as a
  bottleneck, not before.

---

## Performance (informal)

At the corpus sizes used for validation (up to ~2000 vocab / ~1000 lines),
every ported module ran faster than the Python original except one:
tokenizer+graph training ran ~4–8x faster, CTM+IVM construction ~6x faster,
`generate()` ~8–11x faster, the full `model.py` pipeline ~8.8x faster. The
exception is `interpret.c` (the Cluster Interpreter), which ran slightly
*slower* than Python on the larger corpus — it's a read-only offline
analysis pass, never on the generation hot path, and the current
implementation allocates a fresh hash map per cluster rather than reusing
one; a real target if that module's own performance ever matters.

None of this is a rigorous benchmark — no warm-up, single runs, and only
the corpus sizes validation happened to use. Treat it as "this isn't
accidentally quadratic somewhere Python wasn't," not as a performance
guarantee at any particular scale. See `VALIDATION.md` for exact numbers
per phase.
