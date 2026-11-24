"""Unit tests for 8D vector-field wireframe extraction: mutual verification, noise, narrow edges."""
import numpy as np

from bandbrep.config import Config
from bandbrep.edge_first_grid import GridSamples
from bandbrep.field_provider import FieldEvaluation
from bandbrep.vector_wireframe import extract_vector_wireframe


class _SyntheticProvider:
    def __init__(self, fn, presence=('present', 'present')):
        self.fn = fn
        self._presence = presence

    def presence(self):
        return self._presence

    def evaluate(self, q):
        q = np.asarray(q, float).reshape(-1, 3)
        raw = self.fn(q)
        vals = np.column_stack((np.zeros(len(q)), raw[:, :2]))
        offs = raw[:, 2:].reshape(len(q), 2, 3)
        valid = np.ones((len(q), 3), dtype=bool)
        sat = np.zeros((len(q), 3), dtype=bool)
        return FieldEvaluation(vals, offs, valid, sat, np.zeros_like(vals))


def _make_grid(points, provider):
    res = provider.evaluate(points)
    keys = np.rint(points / 0.003).astype(np.int64)
    return GridSamples(np.empty((0, 3), int), keys, points, res.values, res.offsets, res.valid, {})


def test_false_vertex_filtered_by_edge_mutual_verification():
    """A floating fake V cluster far from any edge must be rejected by E/V mutual verification."""
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([0.1, 0.0, 0.0])
    v_fake = np.array([0.05, 0.06, 0.0])
    verts = np.vstack((v0, v1, v_fake))

    def field(p):
        t = np.clip(p[:, 0] / 0.1, 0.0, 1.0)
        closest_e = np.column_stack((t * 0.1, np.zeros(len(p)), np.zeros(len(p))))
        u_e = closest_e - p
        d_e = np.linalg.norm(u_e, axis=1)
        diff_v = verts[None, :, :] - p[:, None, :]
        dist_v = np.linalg.norm(diff_v, axis=2)
        idx = np.argmin(dist_v, axis=1)
        d_v = dist_v[np.arange(len(p)), idx]
        u_v = diff_v[np.arange(len(p)), idx]
        return np.column_stack((d_e, d_v, u_e, u_v))

    xs = np.arange(-0.015, 0.116, 0.003)
    ys = np.arange(-0.015, 0.076, 0.003)
    zs = np.arange(-0.015, 0.016, 0.003)
    pts = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1).reshape(-1, 3)

    provider = _SyntheticProvider(field)
    grid = _make_grid(pts, provider)
    vertices, paths, rejected, _, _ = extract_vector_wireframe(grid, provider, Config())

    assert len(vertices) == 2
    assert any(r.get('reason') in ('false_vertex_no_edge_support', 'false_vertex_high_dE') for r in rejected)
    assert len(paths) == 1
    assert set(paths[0].nodes) == {0, 1}


def test_narrow_parallel_edges_stay_separated_and_noise_resilient():
    """Two parallel edges separated by 0.004 (< 2*0.005 old MC shell) stay cleanly separated under noise."""
    rng = np.random.default_rng(42)
    segments = [
        (np.array([0.0, 0.0, 0.0]), np.array([0.12, 0.0, 0.0])),
        (np.array([0.0, 0.004, 0.0]), np.array([0.12, 0.004, 0.0])),
    ]
    verts = np.array([
        [0.0, 0.0, 0.0], [0.12, 0.0, 0.0],
        [0.0, 0.004, 0.0], [0.12, 0.004, 0.0],
    ])

    def field(p):
        d2 = np.full(len(p), np.inf)
        u_e = np.zeros((len(p), 3))
        for a, b in segments:
            ab = b - a
            t = np.clip((p - a) @ ab / float(ab @ ab), 0.0, 1.0)
            off = a + t[:, None] * ab - p
            c = np.einsum('ij,ij->i', off, off)
            better = c < d2
            d2[better] = c[better]
            u_e[better] = off[better]
        d_e = np.sqrt(d2)
        diff_v = verts[None, :, :] - p[:, None, :]
        dist_v = np.linalg.norm(diff_v, axis=2)
        idx = np.argmin(dist_v, axis=1)
        d_v = dist_v[np.arange(len(p)), idx]
        u_v = diff_v[np.arange(len(p)), idx]
        return np.column_stack((d_e, d_v, u_e, u_v))

    xs = np.arange(-0.012, 0.133, 0.003)
    ys = np.arange(-0.012, 0.017, 0.003)
    zs = np.arange(-0.012, 0.013, 0.003)
    pts = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1).reshape(-1, 3)

    provider = _SyntheticProvider(field)
    grid = _make_grid(pts, provider)
    # Inject small Gaussian noise on offsets to simulate neural prediction noise.
    grid.offsets = grid.offsets + rng.normal(0.0, 0.0003, size=grid.offsets.shape)

    cfg = Config(allow_feature_collapse=False, edge_lateral_tol=0.0015, vertex_cluster_radius=0.0015)
    vertices, paths, _, _, _ = extract_vector_wireframe(grid, provider, cfg)

    assert len(vertices) == 4
    # Two long parallel edges (+ two 0.004 short connecting edges if dE=0.002 at midpoint -> wait, midpoint dE is 0.002).
    # Verify both long parallel edges (length ~0.12) are recovered separately!
    long_paths = [p for p in paths if np.linalg.norm(vertices[p.nodes[0]].position - vertices[p.nodes[1]].position) > 0.10]
    assert len(long_paths) == 2
