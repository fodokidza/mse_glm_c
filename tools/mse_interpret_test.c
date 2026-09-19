#include "mse_graph.h"
#include "mse_interpret.h"
#include "mse_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_evidence(int edge, int shared_role, int robustness) {
    printf("bridge_source_axis");
    if (edge) printf(",edge_adjacency");
    if (shared_role) printf(",shared_role");
    if (robustness) printf(",relationship_robustness");
}

static void print_candidate(const InterpretCandidate *c) {
    printf("  interp=%d via_bridge=%d coverage=%.3f covered=", c->interpreter_token, c->via_bridge_token, c->coverage);
    for (size_t i = 0; i < c->members_covered.len; i++) printf("%d,", c->members_covered.data[i]);
    printf(" total=%d edge=%d shared_role=", c->members_total, c->edge_corroborated);
    for (size_t i = 0; i < c->shared_role_overlap.len; i++) printf("%d,", c->shared_role_overlap.data[i]);
    printf(" rels=");
    for (size_t i = 0; i < c->relationship_ids.len; i++) printf("%d,", c->relationship_ids.data[i]);
    printf(" evidence=[");
    print_evidence(c->edge_corroborated, c->shared_role_overlap.len > 0, c->relationship_ids.len > 1);
    printf("]\n");
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <prefix> <vocab_size>\n", argv[0]); return 1; }
    const char *prefix = argv[1];
    char path[4096];

    EdgeMatrix edges; snprintf(path, sizeof(path), "%s.edges", prefix); em_load(&edges, path);
    BridgeMatrix bridges; snprintf(path, sizeof(path), "%s.bridges", prefix); bm_load(&bridges, path);
    RelationshipMatrix rels; snprintf(path, sizeof(path), "%s.rels", prefix); rm_load(&rels, path);

    printf("--- INTERPRET_CLUSTER (all clusters, top_n=5) ---\n");
    for (int32_t cid = 1; cid <= bridges.max_cluster_id; cid++) {
        InterpretResult r;
        if (!interpret_cluster(&bridges, &rels, &edges, cid, INTERPRET_TOP_N, &r)) continue;
        printf("cluster=%d axis=%s members=", r.cluster_id, r.axis == BM_AXIS_BRIDGE ? "bridge" : "target");
        for (size_t i = 0; i < r.members.len; i++) printf("%d,", r.members.data[i]);
        printf("\n");
        for (int32_t i = 0; i < r.n_candidates; i++) print_candidate(&r.candidates[i]);
        interpret_result_free(&r);
    }

    printf("--- INTERPRET_ALL_CLUSTERS ---\n");
    InterpretAllResult all;
    interpret_all_clusters(&bridges, &rels, &edges, INTERPRET_MIN_COVERAGE, INTERPRET_MAX_PER_CLUSTER, &all);
    for (int32_t i = 0; i < all.n_results; i++) {
        InterpretResult *r = &all.results[i];
        printf("cluster=%d axis=%s\n", r->cluster_id, r->axis == BM_AXIS_BRIDGE ? "bridge" : "target");
        for (int32_t c = 0; c < r->n_candidates; c++) print_candidate(&r->candidates[c]);
    }
    interpret_all_free(&all);

    printf("--- BUILD_INTERPRETER_MATRIX ---\n");
    InterpreterMatrix mat;
    build_interpreter_matrix(&bridges, &rels, &edges, INTERPRET_MIN_COVERAGE, INTERPRET_MIN_SIGNALS, 0, &mat);
    for (int32_t i = 0; i < mat.n_rows; i++) {
        InterpreterMatrixRow *row = &mat.rows[i];
        printf("cluster=%d axis=%s ", row->cluster_id, row->axis == BM_AXIS_BRIDGE ? "bridge" : "target");
        print_candidate(&row->candidate);
    }
    interpreter_matrix_free(&mat);

    printf("--- DISCOVER_ZERO_CLUSTER_GROUPS ---\n");
    ZeroClusterGroups zg;
    discover_zero_cluster_groups(&bridges, &rels, &edges, INTERPRET_MIN_GROUP_SIZE, &zg);
    for (int32_t i = 0; i < zg.n_groups; i++) {
        ZeroClusterGroup *z = &zg.groups[i];
        printf("interp=%d via_bridge=%d members=", z->interpreter_token, z->via_bridge_token);
        for (size_t k = 0; k < z->members.len; k++) printf("%d,", z->members.data[k]);
        printf(" count=%d edge=%d shared_role=", z->member_count, z->edge_corroborated);
        for (size_t k = 0; k < z->shared_role_overlap.len; k++) printf("%d,", z->shared_role_overlap.data[k]);
        printf(" rels=");
        for (size_t k = 0; k < z->relationship_ids.len; k++) printf("%d,", z->relationship_ids.data[k]);
        printf(" evidence=[zero_cluster_source_axis");
        if (z->edge_corroborated) printf(",edge_adjacency");
        if (z->shared_role_overlap.len) printf(",shared_role");
        if (z->relationship_ids.len > 1) printf(",relationship_robustness");
        printf("]\n");
    }
    zero_cluster_groups_free(&zg);

    em_free(&edges); bm_free(&bridges); rm_free(&rels);
    return 0;
}
