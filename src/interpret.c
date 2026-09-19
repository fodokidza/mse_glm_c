#include "mse_interpret.h"
#include "mse_config.h"
#include <stdlib.h>
#include <string.h>

/* ============================================================ helpers */
static void add_unique(i32vec *s, int32_t v) {
    for (size_t i = 0; i < s->len; i++) if (s->data[i] == v) return;
    i32vec_push(s, v);
}
static void remove_value(i32vec *s, int32_t v) {
    for (size_t i = 0; i < s->len; i++) if (s->data[i] == v) { s->data[i] = s->data[s->len - 1]; s->len--; return; }
}
static void sort_i32vec(i32vec *v) {
    int32_t *d = v->data; size_t ln = v->len;
    for (size_t a = 1; a < ln; a++) { int32_t x = d[a]; size_t b = a; while (b > 0 && d[b-1] > x) { d[b] = d[b-1]; b--; } d[b] = x; }
}
static void sort_dedup_i32vec(i32vec *v) {
    sort_i32vec(v);
    size_t n = 0;
    for (size_t i = 0; i < v->len; i++) if (i == 0 || v->data[i] != v->data[i - 1]) v->data[n++] = v->data[i];
    v->len = n;
}
static int64_t pack2(int32_t a, int32_t b) { return ((int64_t)(uint32_t)a << 32) | (uint32_t)b; }
static int has_direct_edge(const EdgeMatrix *edges, int32_t a, int32_t b) { return em_frequency(edges, a, b) > 0; }

static int bm_find_cluster_members(BridgeMatrix *bridges, int32_t cluster_id, BmAxis *axis_out, i32vec *members_out) {
    i32vec src, tgt, brg; i32vec_init(&src); i32vec_init(&tgt); i32vec_init(&brg);
    BmAxis axis = bm_cluster_axis(bridges, cluster_id, &src, &tgt, &brg);
    if (axis == BM_AXIS_NONE) { i32vec_free(&src); i32vec_free(&tgt); i32vec_free(&brg); return 0; }
    for (size_t i = 0; i < src.len; i++) i32vec_push(members_out, axis == BM_AXIS_BRIDGE ? brg.data[i] : tgt.data[i]);
    i32vec_free(&src); i32vec_free(&tgt); i32vec_free(&brg);
    sort_dedup_i32vec(members_out);
    *axis_out = axis;
    return 1;
}

static void shared_role_overlap(BridgeMatrix *bridges, int32_t this_cluster_id,
                                 const i32vec *members, int32_t candidate, i32vec *out) {
    i32vec member_clusters; i32vec_init(&member_clusters);
    for (size_t i = 0; i < members->len; i++) {
        i32vec tmp; i32vec_init(&tmp);
        bm_tindex_get(bridges, members->data[i], &tmp);
        for (size_t k = 0; k < tmp.len; k++) add_unique(&member_clusters, tmp.data[k]);
        i32vec_free(&tmp);
    }
    remove_value(&member_clusters, this_cluster_id);
    sort_i32vec(&member_clusters);

    i32vec cand_clusters; i32vec_init(&cand_clusters);
    bm_tindex_get(bridges, candidate, &cand_clusters);
    remove_value(&cand_clusters, this_cluster_id);
    sort_i32vec(&cand_clusters);

    for (size_t i = 0; i < member_clusters.len; i++)
        for (size_t j = 0; j < cand_clusters.len; j++)
            if (member_clusters.data[i] == cand_clusters.data[j]) { i32vec_push(out, member_clusters.data[i]); break; }
    i32vec_free(&member_clusters); i32vec_free(&cand_clusters);
    sort_i32vec(out);
}

/* rel ids reachable from a triple_id (dedup + sorted, matching a Python set->sorted) */
static void triple_rel_ids(RelationshipMatrix *rels, int32_t triple_id, i32vec *out) {
    i32vec tmp; i32vec_init(&tmp);
    rm_relationships_for_triple(rels, triple_id, &tmp);
    for (size_t i = 0; i < tmp.len; i++) add_unique(out, tmp.data[i]);
    i32vec_free(&tmp);
}

/* ======================================================= bt_hits groups */
typedef struct { int64_t key; int32_t bridge, target; i32vec members; i32vec triple_ids; } BtGroup;

typedef struct {
    double neg_coverage; int32_t neg_evidence; int32_t orig_idx;
} CandSortKey;
static int cand_sort_cmp(const void *pa, const void *pb) {
    const CandSortKey *a = (const CandSortKey *)pa, *b = (const CandSortKey *)pb;
    if (a->neg_coverage != b->neg_coverage) return (a->neg_coverage < b->neg_coverage) ? -1 : 1;
    if (a->neg_evidence != b->neg_evidence) return (a->neg_evidence < b->neg_evidence) ? -1 : 1;
    return (a->orig_idx < b->orig_idx) ? -1 : (a->orig_idx > b->orig_idx ? 1 : 0);
}

int interpret_cluster(BridgeMatrix *bridges, RelationshipMatrix *rels, const EdgeMatrix *edges,
                       int32_t cluster_id, int32_t top_n, InterpretResult *out) {
    memset(out, 0, sizeof(*out));
    BmAxis axis; i32vec members; i32vec_init(&members);
    if (!bm_find_cluster_members(bridges, cluster_id, &axis, &members) || members.len == 0) {
        i32vec_free(&members);
        return 0;
    }

    uint8_t *is_member = (uint8_t *)calloc((size_t)bridges->vocab_size, 1);
    for (size_t i = 0; i < members.len; i++) if (members.data[i] < bridges->vocab_size) is_member[members.data[i]] = 1;

    StrMap key_to_idx; strmap_init(&key_to_idx);
    Arena arena; arena_init(&arena);
    BtGroup *groups = NULL; int32_t n_groups = 0, cap = 0;

    for (size_t mi = 0; mi < members.len; mi++) {
        int32_t src = members.data[mi];
        if (src < 0 || src + 1 >= bridges->vocab_size + 1 || !bridges->index) continue;
        int32_t start = bridges->index[src], end = bridges->index[src + 1];
        for (int32_t tid = start; tid < end; tid++) {
            int32_t target = bridges->target[tid], bridge = bridges->bridge[tid];
            if ((target < bridges->vocab_size && is_member[target]) || (bridge < bridges->vocab_size && is_member[bridge])) continue;
            int64_t key = pack2(bridge, target);
            int32_t gi;
            if (!strmap_get(&key_to_idx, (const char *)&key, (int32_t)sizeof(key), &gi)) {
                if (n_groups == cap) { cap = cap ? cap * 2 : 16; groups = (BtGroup *)realloc(groups, sizeof(BtGroup) * (size_t)cap); }
                gi = n_groups++;
                groups[gi].key = key; groups[gi].bridge = bridge; groups[gi].target = target;
                i32vec_init(&groups[gi].members); i32vec_init(&groups[gi].triple_ids);
                char *owned = arena_strndup(&arena, (const char *)&key, (int32_t)sizeof(key));
                strmap_put(&key_to_idx, owned, (int32_t)sizeof(key), gi);
            }
            i32vec_push(&groups[gi].members, src);
            i32vec_push(&groups[gi].triple_ids, tid);
        }
    }
    strmap_free(&key_to_idx); arena_free(&arena);
    free(is_member);

    InterpretCandidate *cands = (InterpretCandidate *)malloc(sizeof(InterpretCandidate) * (size_t)(n_groups ? n_groups : 1));
    for (int32_t g = 0; g < n_groups; g++) {
        BtGroup *grp = &groups[g];
        InterpretCandidate *c = &cands[g];
        memset(c, 0, sizeof(*c));
        c->interpreter_token = grp->target;
        c->via_bridge_token = grp->bridge;
        i32vec_init(&c->members_covered);
        for (size_t i = 0; i < grp->members.len; i++) i32vec_push(&c->members_covered, grp->members.data[i]);
        c->members_total = (int32_t)members.len;
        c->coverage = (double)grp->members.len / (double)members.len;

        int edge_ok = 1;
        for (size_t i = 0; i < grp->members.len; i++) if (!has_direct_edge(edges, grp->members.data[i], grp->target)) { edge_ok = 0; break; }
        c->edge_corroborated = edge_ok;

        i32vec_init(&c->shared_role_overlap);
        shared_role_overlap(bridges, cluster_id, &members, grp->target, &c->shared_role_overlap);

        i32vec_init(&c->relationship_ids);
        for (size_t i = 0; i < grp->members.len; i++) triple_rel_ids(rels, grp->triple_ids.data[i], &c->relationship_ids);
        sort_i32vec(&c->relationship_ids);

        c->n_evidence = 1 + (c->edge_corroborated ? 1 : 0) + (c->shared_role_overlap.len ? 1 : 0) + (c->relationship_ids.len > 1 ? 1 : 0);

        i32vec_free(&grp->members); i32vec_free(&grp->triple_ids);
    }
    free(groups);

    size_t n_groups_sz = (n_groups > 0) ? (size_t)n_groups : 1u;
    CandSortKey *keys = (CandSortKey *)malloc(sizeof(CandSortKey) * n_groups_sz);
    for (int32_t i = 0; i < n_groups; i++) { keys[i].neg_coverage = -cands[i].coverage; keys[i].neg_evidence = -cands[i].n_evidence; keys[i].orig_idx = i; }
    if (n_groups > 0) qsort(keys, (size_t)n_groups, sizeof(CandSortKey), cand_sort_cmp);

    int32_t n_keep = (top_n > 0 && top_n < n_groups) ? top_n : n_groups;
    size_t n_keep_sz = (n_keep > 0) ? (size_t)n_keep : 1u;
    InterpretCandidate *sorted_cands = (InterpretCandidate *)malloc(sizeof(InterpretCandidate) * n_keep_sz);
    for (int32_t i = 0; i < n_keep; i++) sorted_cands[i] = cands[keys[i].orig_idx];
    for (int32_t i = n_keep; i < n_groups; i++) {
        i32vec_free(&cands[keys[i].orig_idx].members_covered);
        i32vec_free(&cands[keys[i].orig_idx].shared_role_overlap);
        i32vec_free(&cands[keys[i].orig_idx].relationship_ids);
    }
    free(cands); free(keys);

    out->cluster_id = cluster_id; out->axis = axis;
    out->members = members;
    out->candidates = sorted_cands; out->n_candidates = n_keep;
    return 1;
}

void interpret_result_free(InterpretResult *r) {
    i32vec_free(&r->members);
    for (int32_t i = 0; i < r->n_candidates; i++) {
        i32vec_free(&r->candidates[i].members_covered);
        i32vec_free(&r->candidates[i].shared_role_overlap);
        i32vec_free(&r->candidates[i].relationship_ids);
    }
    free(r->candidates);
    memset(r, 0, sizeof(*r));
}

/* ==================================================== interpret_all_clusters */
static int32_t *distinct_sorted_nonzero_cluster_ids(BridgeMatrix *bridges, int32_t *out_n) {
    i32vec s; i32vec_init(&s);
    for (int32_t i = 0; i < bridges->n; i++) if (bridges->cluster_id[i] != 0) add_unique(&s, bridges->cluster_id[i]);
    sort_i32vec(&s);
    *out_n = (int32_t)s.len;
    return s.data; /* caller frees with free() */
}

typedef struct { double coverage; int32_t idx; } AllSortKey;
static int all_sort_cmp(const void *pa, const void *pb) {
    const AllSortKey *a = (const AllSortKey *)pa, *b = (const AllSortKey *)pb;
    if (a->coverage != b->coverage) return (a->coverage > b->coverage) ? -1 : 1; /* desc */
    return (a->idx < b->idx) ? -1 : (a->idx > b->idx ? 1 : 0); /* stable */
}

void interpret_all_clusters(BridgeMatrix *bridges, RelationshipMatrix *rels, const EdgeMatrix *edges,
                             double min_coverage, int32_t max_per_cluster, InterpretAllResult *out) {
    memset(out, 0, sizeof(*out));
    int32_t n_ids; int32_t *ids = distinct_sorted_nonzero_cluster_ids(bridges, &n_ids);
    int32_t scan_n = max_per_cluster > 0 ? max_per_cluster * 4 : 50;

    InterpretResult *results = (InterpretResult *)malloc(sizeof(InterpretResult) * (size_t)(n_ids ? n_ids : 1));
    int32_t n_results = 0;

    for (int32_t i = 0; i < n_ids; i++) {
        InterpretResult r;
        if (!interpret_cluster(bridges, rels, edges, ids[i], scan_n, &r)) continue;
        int32_t kept = 0;
        for (int32_t c = 0; c < r.n_candidates; c++) if (r.candidates[c].coverage >= min_coverage) kept++;
        if (max_per_cluster > 0 && kept > max_per_cluster) kept = max_per_cluster;
        if (kept == 0) { interpret_result_free(&r); continue; }

        InterpretResult filtered; memset(&filtered, 0, sizeof(filtered));
        filtered.cluster_id = r.cluster_id; filtered.axis = r.axis;
        i32vec_init(&filtered.members); for (size_t k = 0; k < r.members.len; k++) i32vec_push(&filtered.members, r.members.data[k]);
        filtered.candidates = (InterpretCandidate *)malloc(sizeof(InterpretCandidate) * (size_t)kept);
        filtered.n_candidates = kept;
        int32_t w = 0;
        for (int32_t c = 0; c < r.n_candidates && w < kept; c++) {
            if (r.candidates[c].coverage < min_coverage) continue;
            filtered.candidates[w] = r.candidates[c];
            /* transfer ownership: null out source so interpret_result_free(&r) below doesn't double-free */
            memset(&r.candidates[c], 0, sizeof(InterpretCandidate));
            w++;
        }
        interpret_result_free(&r);
        results[n_results++] = filtered;
    }
    free(ids);

    AllSortKey *keys = (AllSortKey *)malloc(sizeof(AllSortKey) * (size_t)(n_results ? n_results : 1));
    for (int32_t i = 0; i < n_results; i++) { keys[i].coverage = results[i].candidates[0].coverage; keys[i].idx = i; }
    qsort(keys, (size_t)n_results, sizeof(AllSortKey), all_sort_cmp);

    size_t n_results_sz = (n_results > 0) ? (size_t)n_results : 1u;
    InterpretResult *final = (InterpretResult *)malloc(sizeof(InterpretResult) * n_results_sz);
    for (int32_t i = 0; i < n_results; i++) final[i] = results[keys[i].idx];
    free(results); free(keys);

    out->results = final; out->n_results = n_results;
}

void interpret_all_free(InterpretAllResult *r) {
    for (int32_t i = 0; i < r->n_results; i++) interpret_result_free(&r->results[i]);
    free(r->results);
    memset(r, 0, sizeof(*r));
}

/* ================================================== build_interpreter_matrix */
void build_interpreter_matrix(BridgeMatrix *bridges, RelationshipMatrix *rels, const EdgeMatrix *edges,
                               double min_coverage, int32_t min_signals, int32_t max_per_cluster,
                               InterpreterMatrix *out) {
    memset(out, 0, sizeof(*out));
    int32_t n_ids; int32_t *ids = distinct_sorted_nonzero_cluster_ids(bridges, &n_ids);

    InterpreterMatrixRow *rows = NULL; int32_t n_rows = 0, cap = 0;

    for (int32_t i = 0; i < n_ids; i++) {
        InterpretResult r;
        if (!interpret_cluster(bridges, rels, edges, ids[i], INTERPRET_BUILD_MATRIX_TOPN, &r)) continue;
        int32_t added = 0;
        for (int32_t c = 0; c < r.n_candidates; c++) {
            if (max_per_cluster > 0 && added >= max_per_cluster) break;
            InterpretCandidate *cd = &r.candidates[c];
            if (!(cd->coverage >= min_coverage && cd->n_evidence >= min_signals)) continue;
            if (n_rows == cap) { cap = cap ? cap * 2 : 16; rows = (InterpreterMatrixRow *)realloc(rows, sizeof(InterpreterMatrixRow) * (size_t)cap); }
            InterpreterMatrixRow *row = &rows[n_rows++];
            row->cluster_id = r.cluster_id; row->axis = r.axis;
            i32vec_init(&row->members); for (size_t k = 0; k < r.members.len; k++) i32vec_push(&row->members, r.members.data[k]);
            row->candidate = *cd;
            memset(cd, 0, sizeof(*cd)); /* transfer ownership */
            added++;
        }
        interpret_result_free(&r);
    }
    free(ids);

    CandSortKey *keys = (CandSortKey *)malloc(sizeof(CandSortKey) * (size_t)(n_rows ? n_rows : 1));
    for (int32_t i = 0; i < n_rows; i++) { keys[i].neg_coverage = -rows[i].candidate.coverage; keys[i].neg_evidence = -rows[i].candidate.n_evidence; keys[i].orig_idx = i; }
    qsort(keys, (size_t)n_rows, sizeof(CandSortKey), cand_sort_cmp);

    size_t n_rows_sz = (n_rows > 0) ? (size_t)n_rows : 1u;
    InterpreterMatrixRow *final = (InterpreterMatrixRow *)malloc(sizeof(InterpreterMatrixRow) * n_rows_sz);
    for (int32_t i = 0; i < n_rows; i++) final[i] = rows[keys[i].orig_idx];
    free(rows); free(keys);

    out->rows = final; out->n_rows = n_rows;
}

void interpreter_matrix_free(InterpreterMatrix *m) {
    for (int32_t i = 0; i < m->n_rows; i++) {
        i32vec_free(&m->rows[i].members);
        i32vec_free(&m->rows[i].candidate.members_covered);
        i32vec_free(&m->rows[i].candidate.shared_role_overlap);
        i32vec_free(&m->rows[i].candidate.relationship_ids);
    }
    free(m->rows);
    memset(m, 0, sizeof(*m));
}

/* ============================================== discover_zero_cluster_groups */
typedef struct { int64_t key; int32_t bridge, target; i32vec sources; i32vec triple_ids; } ZGroup;

typedef struct { int32_t neg_count; int32_t neg_evidence; int32_t orig_idx; } ZSortKey;
static int z_sort_cmp(const void *pa, const void *pb) {
    const ZSortKey *a = (const ZSortKey *)pa, *b = (const ZSortKey *)pb;
    if (a->neg_count != b->neg_count) return (a->neg_count < b->neg_count) ? -1 : 1;
    if (a->neg_evidence != b->neg_evidence) return (a->neg_evidence < b->neg_evidence) ? -1 : 1;
    return (a->orig_idx < b->orig_idx) ? -1 : (a->orig_idx > b->orig_idx ? 1 : 0);
}

void discover_zero_cluster_groups(BridgeMatrix *bridges, RelationshipMatrix *rels, const EdgeMatrix *edges,
                                   int32_t min_group_size, ZeroClusterGroups *out) {
    memset(out, 0, sizeof(*out));
    StrMap key_to_idx; strmap_init(&key_to_idx);
    Arena arena; arena_init(&arena);
    ZGroup *groups = NULL; int32_t n_groups = 0, cap = 0;

    for (int32_t i = 0; i < bridges->n; i++) {
        if (bridges->cluster_id[i] != 0) continue;
        int32_t s = bridges->source[i], t = bridges->target[i], br = bridges->bridge[i];
        int64_t key = pack2(br, t);
        int32_t gi;
        if (!strmap_get(&key_to_idx, (const char *)&key, (int32_t)sizeof(key), &gi)) {
            if (n_groups == cap) { cap = cap ? cap * 2 : 16; groups = (ZGroup *)realloc(groups, sizeof(ZGroup) * (size_t)cap); }
            gi = n_groups++;
            groups[gi].key = key; groups[gi].bridge = br; groups[gi].target = t;
            i32vec_init(&groups[gi].sources); i32vec_init(&groups[gi].triple_ids);
            char *owned = arena_strndup(&arena, (const char *)&key, (int32_t)sizeof(key));
            strmap_put(&key_to_idx, owned, (int32_t)sizeof(key), gi);
        }
        /* Python's per_source is a dict-of-lists keyed by source; a
         * given source could in principle contribute more than one row
         * to the SAME (bridge,target) group only if it had two distinct
         * triples with the same (bridge,target) -- impossible (triples
         * are unique by (s,b,t)), so `per_source.keys()` == one entry
         * per distinct source here too. */
        i32vec_push(&groups[gi].sources, s);
        i32vec_push(&groups[gi].triple_ids, i);
    }
    strmap_free(&key_to_idx); arena_free(&arena);

    ZeroClusterGroup *zcands = (ZeroClusterGroup *)malloc(sizeof(ZeroClusterGroup) * (size_t)(n_groups ? n_groups : 1));
    int32_t n_out = 0;

    for (int32_t g = 0; g < n_groups; g++) {
        ZGroup *grp = &groups[g];
        int has_bridge = 0, has_target = 0;
        for (size_t i = 0; i < grp->sources.len; i++) {
            if (grp->sources.data[i] == grp->bridge) has_bridge = 1;
            if (grp->sources.data[i] == grp->target) has_target = 1;
        }
        if (has_bridge || has_target) { i32vec_free(&grp->sources); i32vec_free(&grp->triple_ids); continue; }

        i32vec members; i32vec_init(&members);
        for (size_t i = 0; i < grp->sources.len; i++) i32vec_push(&members, grp->sources.data[i]);
        sort_dedup_i32vec(&members); /* sources already distinct per group by construction, but sort for output */
        if ((int32_t)members.len < min_group_size) { i32vec_free(&members); i32vec_free(&grp->sources); i32vec_free(&grp->triple_ids); continue; }

        ZeroClusterGroup *zc = &zcands[n_out++];
        memset(zc, 0, sizeof(*zc));
        zc->interpreter_token = grp->target;
        zc->via_bridge_token = grp->bridge;
        zc->members = members;
        zc->member_count = (int32_t)members.len;

        int edge_ok = 1;
        for (size_t i = 0; i < members.len; i++) if (!has_direct_edge(edges, members.data[i], grp->target)) { edge_ok = 0; break; }
        zc->edge_corroborated = edge_ok;

        /* no pre-existing cluster_id to discard here */
        i32vec member_clusters; i32vec_init(&member_clusters);
        for (size_t i = 0; i < members.len; i++) {
            i32vec tmp; i32vec_init(&tmp);
            bm_tindex_get(bridges, members.data[i], &tmp);
            for (size_t k = 0; k < tmp.len; k++) add_unique(&member_clusters, tmp.data[k]);
            i32vec_free(&tmp);
        }
        i32vec cand_clusters; i32vec_init(&cand_clusters);
        bm_tindex_get(bridges, grp->target, &cand_clusters);
        i32vec_init(&zc->shared_role_overlap);
        for (size_t i = 0; i < member_clusters.len; i++)
            for (size_t j = 0; j < cand_clusters.len; j++)
                if (member_clusters.data[i] == cand_clusters.data[j]) { i32vec_push(&zc->shared_role_overlap, member_clusters.data[i]); break; }
        sort_i32vec(&zc->shared_role_overlap);
        i32vec_free(&member_clusters); i32vec_free(&cand_clusters);

        i32vec_init(&zc->relationship_ids);
        for (size_t i = 0; i < grp->sources.len; i++) triple_rel_ids(rels, grp->triple_ids.data[i], &zc->relationship_ids);
        sort_i32vec(&zc->relationship_ids);

        zc->n_evidence = 1 + (zc->edge_corroborated ? 1 : 0) + (zc->shared_role_overlap.len ? 1 : 0) + (zc->relationship_ids.len > 1 ? 1 : 0);

        i32vec_free(&grp->sources); i32vec_free(&grp->triple_ids);
    }
    free(groups);

    ZSortKey *keys = (ZSortKey *)malloc(sizeof(ZSortKey) * (size_t)(n_out ? n_out : 1));
    for (int32_t i = 0; i < n_out; i++) { keys[i].neg_count = -zcands[i].member_count; keys[i].neg_evidence = -zcands[i].n_evidence; keys[i].orig_idx = i; }
    qsort(keys, (size_t)n_out, sizeof(ZSortKey), z_sort_cmp);

    size_t n_out_sz = (n_out > 0) ? (size_t)n_out : 1u;
    ZeroClusterGroup *final = (ZeroClusterGroup *)malloc(sizeof(ZeroClusterGroup) * n_out_sz);
    for (int32_t i = 0; i < n_out; i++) final[i] = zcands[keys[i].orig_idx];
    free(zcands); free(keys);

    out->groups = final; out->n_groups = n_out;
}

void zero_cluster_groups_free(ZeroClusterGroups *g) {
    for (int32_t i = 0; i < g->n_groups; i++) {
        i32vec_free(&g->groups[i].members);
        i32vec_free(&g->groups[i].shared_role_overlap);
        i32vec_free(&g->groups[i].relationship_ids);
    }
    free(g->groups);
    memset(g, 0, sizeof(*g));
}
