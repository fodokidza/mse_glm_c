/* mse_interpret.h — C port of interpret.py: a read-only analysis pass
 * that proposes human-readable "interpreter tokens" for a dual-axis
 * Bridge Matrix cluster (e.g. labelling {cat, dog, pig} as "animal")
 * by combining Edge/Bridge/Relationship evidence. Never touches
 * generation — this only reads structures Phase 1/2 already built.
 *
 * `_rows_with_triple_id` needed no C equivalent: the Bridge Matrix's
 * own CSR row index for a source IS the triple_id (that's what
 * RelationshipMatrix.relationships_for_triple expects), so this port
 * just walks bridges->index[source]..[source+1] directly wherever
 * Python looked it up separately.
 */
#ifndef MSE_INTERPRET_H
#define MSE_INTERPRET_H

#include "mse_graph.h"
#include "mse_util.h"

typedef struct {
    int32_t interpreter_token;
    int32_t via_bridge_token;
    double  coverage;
    i32vec  members_covered;      /* sorted ascending */
    int32_t members_total;
    int     edge_corroborated;
    i32vec  shared_role_overlap;  /* sorted ascending cluster ids */
    i32vec  relationship_ids;     /* sorted ascending */
    int     n_evidence;           /* 1 (bridge_source_axis, always) + edge_corroborated + (shared_role nonempty) + (relationship_ids.len > 1) */
} InterpretCandidate;

typedef struct {
    int32_t cluster_id;
    BmAxis  axis;
    i32vec  members;               /* sorted ascending */
    InterpretCandidate *candidates;
    int32_t n_candidates;
} InterpretResult;

void interpret_result_free(InterpretResult *r);

/* Returns 1 and fills *out if cluster_id has >= 2 members, else 0
 * (matches Python's `None` return). Candidates are already sorted
 * (coverage desc, evidence-signal-count desc, ties broken by first-
 * discovery order — matches Python's stable sort) and truncated to
 * top_n. */
int interpret_cluster(BridgeMatrix *bridges, RelationshipMatrix *rels, const EdgeMatrix *edges,
                       int32_t cluster_id, int32_t top_n, InterpretResult *out);

typedef struct {
    InterpretResult *results;
    int32_t n_results;
} InterpretAllResult;

void interpret_all_free(InterpretAllResult *r);

/* Every cluster_id with at least one candidate at or above min_coverage
 * (up to max_per_cluster candidates each, or all of them if
 * max_per_cluster <= 0), sorted by each cluster's best coverage desc. */
void interpret_all_clusters(BridgeMatrix *bridges, RelationshipMatrix *rels, const EdgeMatrix *edges,
                             double min_coverage, int32_t max_per_cluster, InterpretAllResult *out);

/* Flattened rows: one per (cluster_id, interpreter) pair clearing both
 * min_coverage and min_signals -- the "final" CI Matrix. */
typedef struct {
    int32_t cluster_id;
    BmAxis  axis;
    i32vec  members;
    InterpretCandidate candidate; /* owns its own i32vecs; freed with the row */
} InterpreterMatrixRow;

typedef struct {
    InterpreterMatrixRow *rows;
    int32_t n_rows;
} InterpreterMatrix;

void interpreter_matrix_free(InterpreterMatrix *m);

void build_interpreter_matrix(BridgeMatrix *bridges, RelationshipMatrix *rels, const EdgeMatrix *edges,
                               double min_coverage, int32_t min_signals, int32_t max_per_cluster,
                               InterpreterMatrix *out);

/* discover_zero_cluster_groups: mines cluster_id==0 rows for the
 * "fix (bridge,target), source varies" grouping the Bridge Matrix's
 * two clustering rules never assign a cluster_id to. Different output
 * schema from the above (no real cluster_id to report). */
typedef struct {
    int32_t interpreter_token;
    int32_t via_bridge_token;
    i32vec  members;   /* sorted ascending */
    int32_t member_count;
    int     edge_corroborated;
    i32vec  shared_role_overlap;
    i32vec  relationship_ids;
    int     n_evidence; /* 1 (zero_cluster_source_axis) + edge_corroborated + shared_role + robustness */
} ZeroClusterGroup;

typedef struct {
    ZeroClusterGroup *groups;
    int32_t n_groups;
} ZeroClusterGroups;

void zero_cluster_groups_free(ZeroClusterGroups *g);

void discover_zero_cluster_groups(BridgeMatrix *bridges, RelationshipMatrix *rels, const EdgeMatrix *edges,
                                   int32_t min_group_size, ZeroClusterGroups *out);

#endif /* MSE_INTERPRET_H */
