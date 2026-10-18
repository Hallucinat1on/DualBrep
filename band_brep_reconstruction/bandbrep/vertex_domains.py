"""Locally scaled vertex domains from v-UDF votes.

A single global threshold cannot both cut every edge band and stay below half
of the shortest edge.  Here each vertex candidate gets its own radius, and band
nodes are assigned to their *nearest* candidate, so two neighbouring vertex
domains never merge into one component even when their balls touch.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def vertex_candidates(points, distances, provider, cfg):
    """Cluster gradient votes x - dV * grad(dV); returns centers and vote counts."""
    if not len(points):
        return np.empty((0, 3)), np.empty(0, int), 0.
    g, valid = provider.gradient(points, 2)
    norm = np.linalg.norm(g, axis=1)
    valid &= norm > 1e-10
    votes = points[valid] - distances[valid, None] * g[valid] / norm[valid, None]
    if not len(votes):
        return np.empty((0, 3)), np.empty(0, int), 0.
    merge = max(2 * cfg.localization_tolerance, 4 * cfg.field_error)
    pairs = cKDTree(votes).query_pairs(merge, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(votes), len(votes)))
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=count)
    keep = np.flatnonzero(sizes >= cfg.vertex_min_votes)
    centers = np.array([np.median(votes[labels == c], axis=0) for c in keep]).reshape(-1, 3)
    return centers, sizes[keep], merge


def local_vertex_domains(points, values, edge_band, support, provider, cfg):
    """Return (vertex_labels, centers, report); labels are -1 outside vertex domains."""
    labels = np.full(len(points), -1, np.int64)
    seeds = np.flatnonzero(edge_band & support & (values[:, 2] < cfg.tau_vertex))
    centers, votes, merge = vertex_candidates(points[seeds], values[seeds, 2], provider, cfg)
    report = dict(method='nearest_candidate_local_radius', seed_nodes=len(seeds), candidates=len(centers),
                  merge_radius=merge, radius_fraction=cfg.vertex_radius_fraction, cut_factor=cfg.vertex_cut_factor)
    if not len(centers):
        return labels, centers, report
    tree = cKDTree(centers)
    gap = tree.query(centers, k=2)[0][:, 1] if len(centers) > 1 else np.full(1, np.inf)
    band_ids = np.flatnonzero(edge_band)
    band_tree = cKDTree(points[band_ids[support[band_ids]]]) if np.any(support[band_ids]) else None
    support_ids = band_ids[support[band_ids]]
    radius = np.empty(len(centers))
    half_width = np.empty(len(centers))
    for i, c in enumerate(centers):
        near = band_tree.query_ball_point(c, cfg.tau_vertex) if band_tree is not None else []
        # Band half-width near this vertex: how far band nodes reach from the edge.
        w = float(np.quantile(values[support_ids[near], 1], .95)) if len(near) else cfg.face_threshold2
        half_width[i] = w
        # Upper bound keeps short edges / neighbours apart; lower bound must cut the band.
        radius[i] = min(cfg.tau_vertex, max(cfg.vertex_radius_fraction * gap[i], cfg.vertex_cut_factor * w))
    distance, owner = tree.query(points[band_ids])
    inside = distance < radius[owner]
    labels[band_ids[inside]] = owner[inside]
    used = np.unique(labels[labels >= 0])
    remap = np.full(len(centers), -1, np.int64)
    remap[used] = np.arange(len(used))
    labels[labels >= 0] = remap[labels[labels >= 0]]
    centers = centers[used]
    report.update(domains=len(used), dropped_empty_candidates=int(len(radius) - len(used)),
                  radius_min=float(radius[used].min()) if len(used) else None,
                  radius_max=float(radius[used].max()) if len(used) else None,
                  cut_limited=int(np.sum((cfg.vertex_cut_factor * half_width > cfg.vertex_radius_fraction * gap)[used])),
                  angle_control=False)
    return labels, centers, report
