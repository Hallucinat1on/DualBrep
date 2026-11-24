"""Analytic 8D unit fields [dE, dV, uE(3), uV(3)] and closed meshes."""
import numpy as np
from bandbrep.field_provider import CallableUDFProvider, FieldEvaluation
from bandbrep.schemas import SurfaceMesh


class AnalyticFields:
    """Internal fixture for geometry unit tests."""
    def __init__(self, fn, presence=('present', 'present')):
        self.fn = fn
        self._presence = presence
        self.fingerprint = 'analytic-test-field'

    def presence(self):
        return self._presence

    def evaluate(self, q):
        v = np.asarray(self.fn(np.asarray(q, float)), float)
        offsets = np.zeros((len(v), 2, 3), dtype=float)
        return FieldEvaluation(v, offsets, np.isfinite(v), np.zeros_like(v, bool), np.zeros_like(v))

    def sdf(self, q):
        return self.evaluate(q).values[:, 0]


def mesh_of(mesh):
    return SurfaceMesh(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64))


def box():
    import trimesh
    from itertools import product

    vertices = np.array(list(product((-0.55, 0.55), repeat=3)))
    segments = [(a, b) for i, a in enumerate(vertices) for b in vertices[i + 1:] if np.count_nonzero(a != b) == 1]

    def values(p):
        p = np.asarray(p, float).reshape(-1, 3)
        d2 = np.full(len(p), np.inf)
        u_e = np.zeros((len(p), 3), dtype=float)
        for a, b in segments:
            ab = b - a
            t = np.clip((p - a) @ ab / float(ab @ ab), 0.0, 1.0)
            off = a + t[:, None] * ab - p
            cand = np.einsum('ij,ij->i', off, off)
            better = cand < d2
            d2[better] = cand[better]
            u_e[better] = off[better]
        de = np.sqrt(d2)
        v_diff = vertices[None, :, :] - p[:, None, :]
        v_dist = np.linalg.norm(v_diff, axis=2)
        v_idx = np.argmin(v_dist, axis=1)
        dv = v_dist[np.arange(len(p)), v_idx]
        u_v = v_diff[np.arange(len(p)), v_idx]
        return np.column_stack((de, dv, u_e, u_v))

    return mesh_of(trimesh.creation.box(extents=[1.1] * 3)), CallableUDFProvider(
        values, fingerprint='box-8d', presence=('present', 'present')
    )


def cylinder():
    import trimesh

    def values(p):
        p = np.asarray(p, float).reshape(-1, 3)
        r = np.linalg.norm(p[:, :2], axis=1)
        safe_r = np.maximum(r, 1e-12)
        dir_xy = np.where(r[:, None] > 1e-12, p[:, :2] / safe_r[:, None], np.array([1.0, 0.0]))
        closest_xy = dir_xy * 0.65
        closest_z = np.where(p[:, 2] >= 0.0, 0.35, -0.35)
        closest = np.column_stack((closest_xy, closest_z))
        u_e = closest - p
        de = np.linalg.norm(u_e, axis=1)
        dv = np.full(len(p), 3.0)
        u_v = np.zeros((len(p), 3), dtype=float)
        return np.column_stack((de, dv, u_e, u_v))

    radial = np.linspace(0, 0.65, 21)
    profile = np.vstack((
        np.column_stack((radial, np.full(21, -0.35))),
        np.column_stack((np.full(24, 0.65), np.linspace(-0.35, 0.35, 25)[1:])),
        np.column_stack((radial[-2::-1], np.full(20, 0.35))),
    ))
    return mesh_of(trimesh.creation.revolve(profile, sections=128)), CallableUDFProvider(
        values, fingerprint='cylinder-8d', presence=('present', 'absent')
    )


def hollow_sphere():
    import trimesh

    a = trimesh.creation.icosphere(subdivisions=3, radius=0.75)
    b = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
    b.invert()

    def values(q):
        out = np.zeros((len(q), 8), dtype=float)
        out[:, :2] = 3.0
        return out

    return mesh_of(trimesh.util.concatenate([a, b])), CallableUDFProvider(
        values, fingerprint='hollow-8d', presence=('absent', 'absent')
    )


def sphere_with_bore():
    import trimesh

    radius, bore = 0.75, 0.3
    z = np.sqrt(radius**2 - bore**2)
    zz = np.linspace(-z, z, 129)
    profile = np.vstack((
        np.column_stack((np.sqrt(radius**2 - zz**2), zz)),
        np.column_stack((np.full(39, bore), np.linspace(z, -z, 41)[1:-1])),
    ))

    def values(p):
        p = np.asarray(p, float).reshape(-1, 3)
        r = np.linalg.norm(p[:, :2], axis=1)
        safe_r = np.maximum(r, 1e-12)
        dir_xy = np.where(r[:, None] > 1e-12, p[:, :2] / safe_r[:, None], np.array([1.0, 0.0]))
        closest = np.column_stack((dir_xy * bore, np.where(p[:, 2] >= 0.0, z, -z)))
        u_e = closest - p
        de = np.linalg.norm(u_e, axis=1)
        return np.column_stack((de, np.full(len(p), 3.0), u_e, np.zeros((len(p), 3))))

    return mesh_of(trimesh.creation.revolve(profile, sections=128)), CallableUDFProvider(
        values, fingerprint='bore-8d', presence=('present', 'absent')
    )
