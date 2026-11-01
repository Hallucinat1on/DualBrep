"""Deterministic geometric near-duplicate detection with topology checks.

Thresholds are this project's policy, not unpublished DualBrep parameters.
Coordinates are compared in the saved normalization; no rotation/ICP alignment.
"""

from collections import Counter
import time

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from veudf.preprocess.sampling import sample_edges


def sample_mesh(vertices, triangles, count, seed=0):
    """Area samples independent of triangle/vertex order."""
    tri = np.asarray(vertices, np.float64)[triangles]
    order = np.lexsort((tri[:, :, 2], tri[:, :, 1], tri[:, :, 0]), axis=-1)
    tri = np.take_along_axis(tri, order[:, :, None], axis=1)
    flat = tri.reshape(len(tri), 9)
    tri = tri[np.lexsort(tuple(flat[:, i] for i in range(8, -1, -1)))]
    area = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(tri), count, p=area / area.sum())
    a, b = rng.random((2, count))
    a = np.sqrt(a)
    points = (
        (1 - a[:, None]) * tri[chosen, 0]
        + (a * (1 - b))[:, None] * tri[chosen, 1]
        + (a * b)[:, None] * tri[chosen, 2]
    )
    return points.astype(np.float32)


def features(mesh, targets, cfg):
    # Retain the original 65536-point pool by default, even when supervision
    # becomes denser. Both arc-length sampling and feature RNG are independent
    # of the query stream, so changing pool sizes cannot change dedup features.
    edge_points = sample_edges(targets, cfg)
    rng = np.random.default_rng(cfg.seed)
    edges = (
        edge_points[
            rng.choice(
                len(edge_points),
                min(len(edge_points), cfg.dedup_edge_points),
                replace=False,
            )
        ]
        if len(edge_points)
        else np.empty((0, 3))
    )
    return dict(
        dedup_surface=sample_mesh(mesh[0], mesh[1], cfg.dedup_surface_points),
        dedup_edge=np.asarray(edges, np.float32).reshape(-1, 3),
        dedup_vertex=np.asarray(targets.vertices, np.float32).reshape(-1, 3),
    )


def topology_graph(topology):
    graph = nx.Graph()
    for face in topology["source_faces"]:
        graph.add_node(("f", face["source_face_id"]), kind="face")
    for vertex in topology["vertices"]:
        graph.add_node(("v", vertex["vertex_id"]), kind="vertex")
    for edge in topology["edges"]:
        curve_type = topology["source_edges"][edge["source_edge_ids"][0]]["curve_type"]
        node = ("e", edge["edge_id"])
        graph.add_node(
            node, kind=f'edge:{curve_type}:{edge["closed"]}:{edge["self_loop"]}'
        )
        for fid in edge["face_ids"]:
            graph.add_edge(node, ("f", fid))
        for vid in edge["vertex_ids"]:
            graph.add_edge(node, ("v", vid))
    return graph


def bidirectional(a, b):
    if not len(a) or not len(b):
        return (0.0, 0.0, 0.0) if len(a) == len(b) else (float("inf"),) * 3
    d = np.r_[cKDTree(a).query(b)[0], cKDTree(b).query(a)[0]]
    return float(np.sqrt(np.mean(d**2))), float(np.quantile(d, 0.95)), float(d.max())


class GraphMatchTimeout(RuntimeError):
    pass


class DeadlineGraphMatcher(nx.algorithms.isomorphism.GraphMatcher):
    """Bound VF2's combinatorial search, including rejected candidate pairs."""

    def __init__(self, a, b, deadline):
        self.deadline = deadline
        super().__init__(a, b, node_match=lambda x, y: x["kind"] == y["kind"])

    def _check(self):
        if time.monotonic() >= self.deadline:
            raise GraphMatchTimeout()

    def candidate_pairs_iter(self):
        for pair in super().candidate_pairs_iter():
            self._check()
            yield pair

    def match(self):
        self._check()
        yield from super().match()


def bounded_isomorphic(a, b, timeout_seconds=2.0):
    """True/False, or None when the bounded search is inconclusive."""
    if len(a) != len(b) or a.number_of_edges() != b.number_of_edges():
        return False, "size_mismatch"
    typed = lambda g: Counter((d["kind"], g.degree[n]) for n, d in g.nodes(data=True))
    if typed(a) != typed(b):
        return False, "typed_degree_mismatch"
    if nx.utils.graphs_equal(a, b):
        return True, "identical_graph"
    # WL is only a rejection test; a matching hash never proves isomorphism.
    if nx.weisfeiler_lehman_graph_hash(
        a, node_attr="kind", iterations=4
    ) != nx.weisfeiler_lehman_graph_hash(b, node_attr="kind", iterations=4):
        return False, "color_refinement_mismatch"
    matcher = DeadlineGraphMatcher(a, b, time.monotonic() + timeout_seconds)
    try:
        return matcher.is_isomorphic(), "bounded_vf2"
    except GraphMatchTimeout:
        return None, "inconclusive_timeout"
    finally:
        matcher.reset_recursion_limit()


class Deduplicator:
    """Feed candidates in priority order (test, val, train); earlier ones are representatives.

    Same geometry + same topology -> ``near_duplicate``.  Near-identical geometry
    across splits (topology different or inconclusive) -> ``near_duplicate_split_conflict``.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.buckets = {}

    def consider(self, uid, split, topology, feats):
        cfg = self.cfg
        key = (len(topology["source_faces"]), len(topology["source_edges"]))
        surface = feats["dedup_surface"]
        descriptor = np.r_[
            surface.min(0), surface.max(0), surface.mean(0), surface.var(0)
        ]
        graph = None
        for other in self.buckets.get(key, []):
            if (
                np.max(np.abs(descriptor - other["descriptor"]))
                > 4 * cfg.dedup_surface_p95
            ):
                continue
            rms, p95, _ = bidirectional(surface, other["feats"]["dedup_surface"])
            if rms > cfg.dedup_surface_rms or p95 > cfg.dedup_surface_p95:
                continue
            graph = graph or topology_graph(topology)
            erms = bidirectional(feats["dedup_edge"], other["feats"]["dedup_edge"])[0]
            vmax = bidirectional(feats["dedup_vertex"], other["feats"]["dedup_vertex"])[
                2
            ]
            same, method = (
                (bounded_isomorphic(graph, other["graph"]))
                if erms <= cfg.dedup_edge_rms and vmax <= cfg.dedup_vertex_max
                else (False, "target_geometry_mismatch")
            )
            scores = dict(
                surface_rms=rms,
                surface_p95=p95,
                topology_isomorphic=same,
                method=method,
            )
            if same:
                return dict(
                    status="rejected",
                    code="near_duplicate",
                    representative=other["uid"],
                    scores=scores,
                )
            if split != other["split"]:
                return dict(
                    status="rejected",
                    code="near_duplicate_split_conflict",
                    representative=other["uid"],
                    scores=scores,
                )
        self.buckets.setdefault(key, []).append(
            dict(
                uid=uid,
                split=split,
                descriptor=descriptor,
                feats=feats,
                graph=graph or topology_graph(topology),
            )
        )
        return dict(status="passed")
