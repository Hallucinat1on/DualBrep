"""8D joint-field wireframe extraction: parallel E/V mid-field pull, mutual verification,
V-dominated analytical cut balls, 1D weighted PCA thinning, and tangent-aware tracing.
"""
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree
from scipy.spatial import cKDTree

from .deadline import checkpoint
from .schemas import VertexNode


@dataclass
class Path:
    nodes: list
    points: np.ndarray
    closed: bool
    ports: list = field(default_factory=list)
    refined_mask: np.ndarray | None = None
    collapsed: bool = False


def _mesh_distances(points, provider):
    if provider is None or not hasattr(provider, 'oracle') or not len(points):
        return np.zeros(len(points), dtype=float)
    return np.sqrt(np.maximum(provider.oracle.closest(points)[0], 0.0))


def _weighted_l1_median(points, weights, iterations=4):
    w = np.asarray(weights, dtype=float)
    w_sum = float(w.sum())
    if w_sum <= 1e-30:
        w = np.ones(len(points), dtype=float)
    center = np.average(points, axis=0, weights=w)
    for _ in range(iterations):
        dist = np.linalg.norm(points - center, axis=1)
        inv = w / np.maximum(dist, 1e-6)
        center = np.average(points, axis=0, weights=inv)
    return center


def _pull_vertex_candidates(grid, provider, cfg):
    """Step 1a: Pull V-channel mid-field points to 0D vertex clusters via x + uV."""
    if not len(grid.points):
        return [], dict(mid_points=0, clusters=0)

    dV = grid.values[:, 2]
    uV = grid.offsets[:, 1]
    uV_norm = np.linalg.norm(uV, axis=1)
    cons_err = np.abs(uV_norm - dV)

    mask = (
        grid.valid[:, 2]
        & (dV >= cfg.vertex_mid_min)
        & (dV <= cfg.vertex_mid_max)
        & (cons_err <= cfg.vertex_consistency_tol)
    )
    if mask.sum() < 4:
        mask = (
            grid.valid[:, 2]
            & (dV > 1e-8)
            & (dV <= cfg.vertex_mid_max)
            & (cons_err <= cfg.vertex_consistency_tol)
        )
    if not mask.any():
        return [], dict(mid_points=0, clusters=0)

    pulled = grid.points[mask] + uV[mask]
    err = cons_err[mask]
    mesh_d = _mesh_distances(pulled, provider)
    near_mesh = mesh_d <= cfg.feature_sdf_limit
    if not near_mesh.any():
        return [], dict(mid_points=int(mask.sum()), clusters=0)
    pulled, err, mesh_d = pulled[near_mesh], err[near_mesh], mesh_d[near_mesh]

    sigma_c = max(0.5 * cfg.vertex_consistency_tol, 1e-4)
    sigma_s = max(cfg.feature_sdf_tolerance, 1e-4)
    weights = np.exp(-(err / sigma_c) ** 2) * np.exp(-(np.maximum(0.0, mesh_d - cfg.feature_sdf_tolerance) / sigma_s) ** 2)

    tree = cKDTree(pulled)
    pairs = tree.query_pairs(cfg.vertex_cluster_radius, output_type='ndarray')
    n = len(pulled)
    if len(pairs):
        adj = csr_matrix(
            (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
            shape=(n, n),
        )
        n_comp, labels = connected_components(adj + adj.T, directed=False)
    else:
        n_comp, labels = n, np.arange(n)

    candidates = []
    min_support = 4
    for cid in range(n_comp):
        idx = np.flatnonzero(labels == cid)
        if len(idx) < min_support:
            continue
        pts = pulled[idx]
        w = weights[idx]
        center = _weighted_l1_median(pts, w)
        spread = float(np.sqrt(np.average(np.sum((pts - center) ** 2, axis=1), weights=w)))
        candidates.append(dict(position=center, residual=spread, weight=float(w.sum()), support=len(idx)))

    return candidates, dict(mid_points=len(pulled), clusters=len(candidates))


def _pull_edge_points(grid, provider, cfg):
    """Step 1b: Pull E-channel mid-field points to 1D edge manifold via x + uE."""
    empty = (
        np.empty((0, 3)), np.empty(0), np.empty(0, bool),
        np.empty((0, 3)), np.empty(0),
    )
    if provider.presence()[0] == 'absent' or not len(grid.points):
        return empty

    dE = grid.values[:, 1]
    uE = grid.offsets[:, 0]
    uE_norm = np.linalg.norm(uE, axis=1)
    cons_err = np.abs(uE_norm - dE)

    mask = (
        grid.valid[:, 1]
        & (dE >= cfg.edge_mid_min)
        & (dE <= cfg.edge_mid_max)
        & (cons_err <= cfg.edge_consistency_tol)
    )
    if mask.sum() < 8:
        mask = (
            grid.valid[:, 1]
            & (dE > 1e-8)
            & (dE <= cfg.edge_mid_max)
            & (cons_err <= cfg.edge_consistency_tol)
        )
    if not mask.any():
        return empty

    src_xyz = grid.points[mask]
    src_dV = grid.values[mask, 2]
    src_v_valid = grid.valid[mask, 2]
    pulled = src_xyz + uE[mask]
    err = cons_err[mask]

    mesh_d = _mesh_distances(pulled, provider)
    near_mesh = mesh_d <= cfg.feature_sdf_limit
    if not near_mesh.any():
        return empty
    src_xyz = src_xyz[near_mesh]
    src_dV = src_dV[near_mesh]
    src_v_valid = src_v_valid[near_mesh]
    pulled = pulled[near_mesh]
    err = err[near_mesh]
    mesh_d = mesh_d[near_mesh]

    sigma_c = max(0.5 * cfg.edge_consistency_tol, 1e-4)
    sigma_s = max(cfg.feature_sdf_tolerance, 1e-4)
    weights = np.exp(-(err / sigma_c) ** 2) * np.exp(-(np.maximum(0.0, mesh_d - cfg.feature_sdf_tolerance) / sigma_s) ** 2)
    return src_xyz, src_dV, src_v_valid, pulled, weights


def _verify_vertices_with_edges(candidates, pulled_e, provider, cfg):
    """Step 2: E/V mutual verification to filter false vertices and build analytical cut balls."""
    if not candidates:
        return [], []

    centers = np.array([c['position'] for c in candidates], dtype=float)
    eval_res = provider.evaluate(centers)
    e_tree = cKDTree(pulled_e) if len(pulled_e) else None
    e_support_radius = max(0.008, 2.5 * cfg.fine_spacing)
    e_val_tol = max(0.005, 1.5 * cfg.edge_consistency_tol)
    e_usable = eval_res.valid[:, 1] & (~eval_res.saturated[:, 1] | (eval_res.values[:, 1] > 0)) & np.isfinite(eval_res.values[:, 1])

    verified = []
    rejected = []
    for i, cand in enumerate(candidates):
        e_count = len(e_tree.query_ball_point(cand['position'], e_support_radius)) if e_tree is not None else 0
        if e_tree is not None and e_count < 3:
            rejected.append(dict(reason='false_vertex_no_edge_support', position=cand['position'].tolist(),
                                 edge_support=e_count))
            continue
        if e_usable[i] and eval_res.values[i, 1] > e_val_tol:
            rejected.append(dict(reason='false_vertex_high_dE', position=cand['position'].tolist(),
                                 dE=float(eval_res.values[i, 1])))
            continue
        verified.append(cand)

    if not verified:
        return [], rejected

    # Deduplicate nearly-coincident verified vertices using a tight merge radius
    # decoupled from collapse_tolerance so genuine close CAD vertices survive.
    if cfg.allow_feature_collapse and len(verified) > 1:
        merge_tol = min(0.002, 0.7 * cfg.vertex_cluster_radius, cfg.collapse_tolerance)
        pos = np.array([v['position'] for v in verified])
        pairs = cKDTree(pos).query_pairs(merge_tol, output_type='ndarray')
        if len(pairs):
            n = len(verified)
            adj = csr_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
            n_comp, labels = connected_components(adj + adj.T, directed=False)
            merged = []
            for cid in range(n_comp):
                idx = np.flatnonzero(labels == cid)
                if len(idx) == 1:
                    merged.append(verified[idx[0]])
                else:
                    w = np.array([verified[k]['weight'] for k in idx], dtype=float)
                    p = np.average([verified[k]['position'] for k in idx], axis=0, weights=w)
                    res = float(max(verified[k]['residual'] for k in idx))
                    merged.append(dict(position=p, residual=res, weight=float(w.sum()),
                                       support=int(sum(verified[k]['support'] for k in idx))))
            verified = merged

    # Build VertexNodes with adaptive analytical cut balls r_i = min(r_default, 0.35 * d_min).
    v_pos = np.array([v['position'] for v in verified], dtype=float)
    if len(v_pos) > 1:
        d_min = cKDTree(v_pos).query(v_pos, k=2)[0][:, 1]
    else:
        d_min = np.full(len(v_pos), np.inf)

    vertices = []
    for i, cand in enumerate(verified):
        r_i = float(min(cfg.vertex_cut_radius, 0.35 * d_min[i]))
        vertices.append(VertexNode(i, cand['position'], cand['residual'], r_i))
    return vertices, rejected


def _cut_edges_by_vertices(src_xyz, src_dV, src_v_valid, pulled_e, weights_e, vertices):
    """Step 3: V-dominated junction masking to excise cut-locus dirty zones around vertices."""
    if not len(pulled_e) or not vertices:
        return pulled_e, weights_e

    v_pos = np.array([v.position for v in vertices], dtype=float)
    v_rad = np.array([v.support_radius for v in vertices], dtype=float)
    v_tree = cKDTree(v_pos)

    d_src, k_src = v_tree.query(src_xyz)
    d_pulled, k_pulled = v_tree.query(pulled_e)

    keep = (d_src > v_rad[k_src]) & (d_pulled > v_rad[k_pulled])
    keep &= (~src_v_valid) | (src_dV > v_rad[k_src])
    return pulled_e[keep], weights_e[keep]


def _micro_voxel_consensus(points, weights, voxel_size):
    """Collapse dense pulled points into weighted micro-voxel centroids for consensus & speed."""
    if not len(points):
        return np.empty((0, 3)), np.empty(0), np.empty(0, int)
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    n_vox = len(counts)
    w_sum = np.bincount(inverse, weights=weights, minlength=n_vox)
    safe_w = np.maximum(w_sum, 1e-30)
    centroids = np.column_stack([
        np.bincount(inverse, weights=points[:, d] * weights, minlength=n_vox) / safe_w
        for d in range(3)
    ])
    return centroids, w_sum, counts


def _pca_thin_and_tangents(points, weights, cfg):
    """Step 4a: 1D weighted PCA centerline thinning (no shrinkage on already-thin curves)."""
    n = len(points)
    if n == 0:
        return points, np.empty((0, 3)), np.empty(0)

    pts = points.copy()
    tangents = np.zeros((n, 3), dtype=float)
    linearity = np.zeros(n, dtype=float)
    r_pca = cfg.pca_thin_radius
    sigma_r = max(0.5 * r_pca, 1e-4)

    for _ in range(cfg.pca_thin_iterations):
        checkpoint('pca_thin')
        tree = cKDTree(pts)
        neighbors_list = tree.query_ball_point(pts, r_pca)
        next_pts = pts.copy()
        for i, nbrs in enumerate(neighbors_list):
            if len(nbrs) < 2:
                continue
            nbr_pts = pts[nbrs]
            if len(nbrs) == 2:
                vec = nbr_pts[1] - nbr_pts[0]
                norm = float(np.linalg.norm(vec))
                if norm > 1e-12:
                    tangents[i] = vec / norm
                    linearity[i] = 1.0
                continue
            d2 = np.sum((nbr_pts - pts[i]) ** 2, axis=1)
            nbr_w = weights[nbrs] * np.exp(-d2 / (2.0 * sigma_r * sigma_r))
            w_tot = float(nbr_w.sum())
            if w_tot <= 1e-30:
                continue
            mu = (nbr_pts * nbr_w[:, None]).sum(axis=0) / w_tot
            diff = nbr_pts - mu
            cov = (diff * nbr_w[:, None]).T @ diff / w_tot
            evals, evecs = np.linalg.eigh(cov)
            l1 = float(evals[2])
            if l1 <= 1e-12:
                continue
            l2 = float(max(evals[1], 0.0))
            t = evecs[:, 2]
            tangents[i] = t
            linearity[i] = (l1 - l2) / l1
            spread = np.sqrt(l2)
            # Contract only genuine noisy 1D tubes; skip razor-thin curves (<1e-4) and bimodal parallel gaps (>0.0015).
            if 1e-4 <= spread <= 0.0015:
                next_pts[i] = mu + float(np.dot(pts[i] - mu, t)) * t
        pts = next_pts

    return pts, tangents, linearity


def _cluster_tangent_aware(points, tangents, linearity, cfg):
    """Step 4b: Connect points that are close, tangent-aligned, and laterally collinear."""
    n = len(points)
    if n == 0:
        return []

    # Drop blob/sheet noise points with very low linearity when enough linear points exist.
    good = (np.linalg.norm(tangents, axis=1) > 0.5) & (linearity >= 0.30)
    active = np.flatnonzero(good) if good.sum() >= 3 else np.arange(n)
    if not len(active):
        return []

    sub_pts = points[active]
    sub_tan = tangents[active]
    pairs = cKDTree(sub_pts).query_pairs(cfg.edge_link_radius, output_type='ndarray')
    if not len(pairs):
        return []

    i, j = pairs[:, 0], pairs[:, 1]
    dot = np.einsum('ij,ij->i', sub_tan[i], sub_tan[j])
    cos_max = np.cos(np.deg2rad(42.0))
    align = np.abs(dot) >= cos_max

    sign = np.where(dot >= 0.0, 1.0, -1.0)[:, None]
    t_avg = sub_tan[i] + sign * sub_tan[j]
    t_norm = np.linalg.norm(t_avg, axis=1, keepdims=True)
    t_avg = np.divide(t_avg, np.maximum(t_norm, 1e-12))

    delta = sub_pts[j] - sub_pts[i]
    d_along = np.einsum('ij,ij->i', delta, t_avg)
    d_perp = np.linalg.norm(delta - d_along[:, None] * t_avg, axis=1)

    keep = align & (d_perp <= cfg.edge_lateral_tol)
    pairs = pairs[keep]
    if not len(pairs):
        return []

    m = len(active)
    dist = np.linalg.norm(sub_pts[pairs[:, 1]] - sub_pts[pairs[:, 0]], axis=1)
    graph = csr_matrix((dist, (pairs[:, 0], pairs[:, 1])), shape=(m, m))
    graph = graph + graph.T
    n_comp, labels = connected_components(graph, directed=False)

    clusters = []
    for cid in range(n_comp):
        loc = np.flatnonzero(labels == cid)
        if len(loc) < 3:
            continue
        clusters.append((active[loc], graph[loc][:, loc]))
    return clusters


def _mst_diameter_order(sub_graph):
    """Find the longest path (tree diameter) in the Euclidean MST of a cluster."""
    mst = minimum_spanning_tree(sub_graph)
    mst_sym = mst + mst.T
    d0 = dijkstra(mst_sym, directed=False, indices=0)
    end_a = int(np.argmax(d0))
    d1, pred = dijkstra(mst_sym, directed=False, indices=end_a, return_predecessors=True)
    end_b = int(np.argmax(d1))

    order = []
    curr = end_b
    while curr >= 0 and curr != end_a:
        order.append(curr)
        curr = int(pred[curr])
    order.append(end_a)
    order.reverse()
    return np.asarray(order, dtype=int)


def _resample_polyline(pts, step, closed=False):
    if len(pts) < 2:
        return pts
    work = np.vstack((pts, pts[:1])) if closed else pts
    seg = np.linalg.norm(np.diff(work, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 1e-9:
        return work[:2]
    cum = np.r_[0.0, np.cumsum(seg)]
    n_samples = max(16 if closed else 5, int(np.ceil(total / step)) + 1)
    s = np.linspace(0.0, total, n_samples)
    out = np.column_stack([np.interp(s, cum, work[:, d]) for d in range(3)])
    if closed:
        out[-1] = out[0]
    return out


def _end_tangent(pts, at_start):
    """Unit vector pointing outward from the start or end of a polyline."""
    if len(pts) < 2:
        return np.zeros(3)
    if at_start:
        k = min(len(pts) - 1, 3)
        vec = pts[0] - pts[k]
    else:
        k = max(0, len(pts) - 4)
        vec = pts[-1] - pts[k]
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else np.zeros(3)


def _extract_ordered_polylines(points, tangents, clusters, vertices, cfg):
    """Step 4c: Order each cluster via EMST diameter and detect closed loops."""
    polylines = []
    v_pos = np.array([v.position for v in vertices], dtype=float) if vertices else np.empty((0, 3))
    v_rad = np.array([v.support_radius for v in vertices], dtype=float) if vertices else np.empty(0)
    v_tree = cKDTree(v_pos) if len(v_pos) else None
    step = min(0.0025, cfg.fine_spacing)

    for global_idx, sub_graph in clusters:
        order = _mst_diameter_order(sub_graph)
        if len(order) < 3:
            continue
        ordered_pts = points[global_idx[order]]
        ordered_tan = tangents[global_idx[order]]
        arc_len = float(np.linalg.norm(np.diff(ordered_pts, axis=0), axis=1).sum())
        if arc_len < 0.5 * cfg.fine_spacing:
            continue

        gap = float(np.linalg.norm(ordered_pts[0] - ordered_pts[-1]))
        near_vertex = False
        if v_tree is not None:
            d0, k0 = v_tree.query(ordered_pts[0])
            d1, k1 = v_tree.query(ordered_pts[-1])
            near_vertex = (d0 <= v_rad[k0] + 0.005) or (d1 <= v_rad[k1] + 0.005)

        is_closed = (
            not near_vertex
            and len(ordered_pts) >= 6
            and gap <= 2.5 * cfg.edge_link_radius
            and abs(float(np.dot(ordered_tan[0], ordered_tan[-1]))) >= np.cos(np.deg2rad(50.0))
        )
        resampled = _resample_polyline(ordered_pts, step, closed=is_closed)
        polylines.append(Path([], resampled, is_closed, refined_mask=np.ones(len(resampled), dtype=bool)))
    return polylines


def _verify_polylines(polylines, provider, cfg):
    """Step 4d: Segment-level scalar dE & mesh proximity verification."""
    if not polylines:
        return [], 0
    probe_sets = []
    counts = []
    for p in polylines:
        idx = np.unique(np.linspace(0, len(p.points) - 1, min(7, len(p.points))).astype(int))
        probe_sets.append(p.points[idx])
        counts.append(len(idx))
    all_probes = np.vstack(probe_sets)
    res = provider.evaluate(all_probes)
    mesh_d = _mesh_distances(all_probes, provider)
    e_tol = max(0.005, 1.5 * cfg.edge_consistency_tol)

    kept = []
    rejected_count = 0
    offset = 0
    for p, cnt in zip(polylines, counts):
        sl = slice(offset, offset + cnt)
        offset += cnt
        ok = res.valid[sl, 1] & (~res.saturated[sl, 1] | (res.values[sl, 1] > 0)) & np.isfinite(res.values[sl, 1])
        dE_ok = (not ok.any()) or (float(np.median(res.values[sl, 1][ok])) <= e_tol)
        mesh_ok = float(np.max(mesh_d[sl])) <= cfg.feature_sdf_limit
        if dE_ok and mesh_ok:
            kept.append(p)
        else:
            rejected_count += 1
    return kept, rejected_count


def _attach_and_bridge(polylines, vertices, provider, cfg):
    """Step 4e: Tangent-aligned vertex attachment, weak-edge gap bridging, short-edge recovery."""
    attach_margin = max(0.006, 2.0 * cfg.fine_spacing)
    v_pos = np.array([v.position for v in vertices], dtype=float) if vertices else np.empty((0, 3))
    v_rad = np.array([v.support_radius for v in vertices], dtype=float) if vertices else np.empty(0)
    v_tree = cKDTree(v_pos) if len(v_pos) else None

    def find_vertex(p_end, t_out):
        if v_tree is None:
            return None
        max_r = float(v_rad.max()) + attach_margin
        cands = v_tree.query_ball_point(p_end, max_r)
        best_vid = None
        best_score = np.inf
        for vid in cands:
            vec = v_pos[vid] - p_end
            dist = float(np.linalg.norm(vec))
            if dist > v_rad[vid] + attach_margin:
                continue
            proj = float(np.dot(vec, t_out))
            if proj < -0.003:
                continue
            lat = float(np.linalg.norm(vec - max(proj, 0.0) * t_out))
            score = lat + 0.25 * dist
            if score < best_score:
                best_score = score
                best_vid = vid
        return best_vid

    closed_paths = []
    open_items = []  # [pts, vid_start, vid_end]
    for p in polylines:
        if p.closed:
            closed_paths.append(p)
            continue
        v0 = find_vertex(p.points[0], _end_tangent(p.points, True))
        v1 = find_vertex(p.points[-1], _end_tangent(p.points, False))
        open_items.append([p.points, v0, v1])

    # Gap bridging for unattached open ends that face each other across a weak-edge break.
    bridge_max = max(0.015, 3.5 * cfg.edge_link_radius)
    e_tol = max(0.005, 1.5 * cfg.edge_consistency_tol)
    cos_bridge = np.cos(np.deg2rad(35.0))
    changed = True
    while changed:
        changed = False
        best_pair = None
        best_dist = bridge_max
        for i in range(len(open_items)):
            pts_i, vi0, vi1 = open_items[i]
            for end_i in (0, 1):
                if (end_i == 0 and vi0 is not None) or (end_i == 1 and vi1 is not None):
                    continue
                p_a = pts_i[0] if end_i == 0 else pts_i[-1]
                t_a = _end_tangent(pts_i, end_i == 0)
                for j in range(i + 1, len(open_items)):
                    pts_j, vj0, vj1 = open_items[j]
                    for end_j in (0, 1):
                        if (end_j == 0 and vj0 is not None) or (end_j == 1 and vj1 is not None):
                            continue
                        p_b = pts_j[0] if end_j == 0 else pts_j[-1]
                        d = float(np.linalg.norm(p_b - p_a))
                        if d < 1e-8 or d >= best_dist:
                            continue
                        chord = (p_b - p_a) / d
                        t_b = _end_tangent(pts_j, end_j == 0)
                        if float(np.dot(t_a, chord)) < cos_bridge or float(np.dot(t_b, -chord)) < cos_bridge:
                            continue
                        mid = np.array([0.5 * (p_a + p_b)])
                        m_res = provider.evaluate(mid)
                        m_usable = bool(
                            m_res.valid[0, 1]
                            and (not m_res.saturated[0, 1] or m_res.values[0, 1] > 0)
                            and np.isfinite(m_res.values[0, 1])
                        )
                        if m_usable and m_res.values[0, 1] > e_tol:
                            continue
                        if _mesh_distances(mid, provider)[0] > cfg.feature_sdf_limit:
                            continue
                        best_dist = d
                        best_pair = (i, end_i, j, end_j)
        if best_pair is not None:
            i, end_i, j, end_j = best_pair
            pts_i, vi0, vi1 = open_items[i]
            pts_j, vj0, vj1 = open_items[j]
            if end_i == 0:
                pts_i = pts_i[::-1]
                vi0, vi1 = vi1, vi0
            if end_j == 1:
                pts_j = pts_j[::-1]
                vj0, vj1 = vj1, vj0
            merged_pts = np.vstack((pts_i, pts_j))
            open_items[i] = [merged_pts, vi0, vj1]
            open_items.pop(j)
            changed = True

    paths = list(closed_paths)
    unresolved = []
    for pts, v0, v1 in open_items:
        if v0 is not None and v1 is not None:
            if v0 == v1:
                paths.append(Path([v0], pts, False, refined_mask=np.ones(len(pts), dtype=bool)))
            else:
                paths.append(Path([v0, v1], pts, False, refined_mask=np.ones(len(pts), dtype=bool)))
        else:
            arc_len = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            if arc_len >= 2.0 * cfg.fine_spacing:
                unresolved.append(dict(
                    reason='dangling_edge_segment',
                    attached=[v for v in (v0, v1) if v is not None],
                    length=arc_len,
                ))

    # Short-edge fallback for close vertex pairs separated by <= ~2.5 * vertex_cut_radius.
    if len(vertices) >= 2:
        connected = {tuple(sorted(p.nodes)) for p in paths if len(p.nodes) == 2}
        max_short = 2.5 * cfg.vertex_cut_radius + cfg.fine_spacing
        short_tol = max(0.003, cfg.edge_consistency_tol)
        for a, b in sorted(cKDTree(v_pos).query_pairs(max_short)):
            if (a, b) in connected:
                continue
            p0, p1 = v_pos[a], v_pos[b]
            length = float(np.linalg.norm(p1 - p0))
            if length <= 1e-8:
                continue
            probes = p0 + np.linspace(0.2, 0.8, 5)[:, None] * (p1 - p0)
            res = provider.evaluate(probes)
            ok = res.valid[:, 1] & (~res.saturated[:, 1] | (res.values[:, 1] > 0)) & np.isfinite(res.values[:, 1])
            if not ok.all() or float(res.values[:, 1].max()) > short_tol:
                continue
            if float(_mesh_distances(probes, provider).max()) > cfg.feature_sdf_limit:
                continue
            if any(cKDTree(p.points).query(probes)[0].max() < 0.3 * length for p in paths if len(p.points)):
                continue
            pts = np.vstack((p0, probes, p1))
            paths.append(Path([a, b], pts, False, refined_mask=np.ones(len(pts), dtype=bool)))
            connected.add((a, b))

    return paths, unresolved


def extract_vector_wireframe(grid, provider, cfg):
    """Run the complete 8D vector-field wireframe extraction (V/E pull -> mutual verify -> trace)."""
    t0 = perf_counter()
    v_cands, v_stats = _pull_vertex_candidates(grid, provider, cfg)
    src_xyz, src_dV, src_v_valid, pulled_e, weights_e = _pull_edge_points(grid, provider, cfg)
    t_pull = perf_counter()

    vertices, v_rejected = _verify_vertices_with_edges(v_cands, pulled_e, provider, cfg)
    t_verify = perf_counter()

    cut_e, cut_w = _cut_edges_by_vertices(src_xyz, src_dV, src_v_valid, pulled_e, weights_e, vertices)
    vox_size = min(0.001, cfg.fine_spacing / 3.0)
    vox_pts, vox_w, vox_counts = _micro_voxel_consensus(cut_e, cut_w, vox_size)

    # Filter isolated 1-hit outlier micro-voxels when the cloud is populated.
    if len(vox_pts) > 20:
        d_nn = cKDTree(vox_pts).query(vox_pts, k=2)[0][:, 1]
        keep_vox = (vox_counts >= 2) | (d_nn <= cfg.edge_link_radius)
        vox_pts, vox_w = vox_pts[keep_vox], vox_w[keep_vox]

    thinned_pts, tangents, linearity = _pca_thin_and_tangents(vox_pts, vox_w, cfg)
    clusters = _cluster_tangent_aware(thinned_pts, tangents, linearity, cfg)
    polylines = _extract_ordered_polylines(thinned_pts, tangents, clusters, vertices, cfg)
    verified_lines, rejected_segments = _verify_polylines(polylines, provider, cfg)
    paths, unresolved = _attach_and_bridge(verified_lines, vertices, provider, cfg)
    t_trace = perf_counter()

    stats = dict(
        method='vector_pull',
        vertex_pull=v_stats,
        edge_pulled_points=int(len(pulled_e)),
        edge_after_vertex_cut=int(len(cut_e)),
        micro_voxels=int(len(vox_pts)),
        clusters=int(len(clusters)),
        rejected_segments=int(rejected_segments),
        paths=[dict(vertices=p.nodes, closed=p.closed, samples=len(p.points)) for p in paths],
        timings=dict(
            pull=t_pull - t0,
            verify=t_verify - t_pull,
            thin_and_trace=t_trace - t_verify,
        ),
    )
    return vertices, paths, v_rejected + unresolved, thinned_pts, stats
