"""NumPy-only continuous signed distance to a closed oriented triangle mesh.

One reusable libigl AABB and triangle winding BVH per object. Positive is outside;
no rasterization, finite grid or CAD dependency is involved.
"""

import numpy as np

# libigl parallel_for starts native std::threads above small query counts,
# independently of OMP_NUM_THREADS. Bound calls so asset-level workers remain
# single threaded. Query batching does not change either acceleration tree.
QUERY_BATCH_SIZE = 512


class MeshSDFOracle:
    def __init__(self, vertices, faces):
        import igl
        import trimesh
        from veudf.geometry.sampling import face_normals, mesh_hash

        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        v, f = self.vertices, self.faces
        if v.ndim != 2 or v.shape[1:] != (3,) or not np.isfinite(v).all():
            raise ValueError("Invalid condition vertices")
        if (
            f.ndim != 2
            or f.shape[1:] != (3,)
            or not len(f)
            or f.min() < 0
            or f.max() >= len(v)
        ):
            raise ValueError("Invalid condition triangles")
        tri = v[f]
        if np.any(
            np.linalg.norm(
                np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
            )
            <= 1e-15
        ):
            raise ValueError("Degenerate condition triangles")
        mesh = trimesh.Trimesh(v, f, process=False)
        if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
            raise ValueError(
                "Condition mesh must be closed and consistently outward oriented"
            )
        self.normals = face_normals(v, f)
        self.tree = igl.AABB()
        self.tree.init(v, f)
        self.winding = igl.FastWindingNumberBVH()
        self.winding.init(v, f)
        # Topological consistency fixes orientation within a connected shell.
        # For multiple shells, validate nesting at an interior triangle location;
        # probing every tiny CAD triangle near float-precision winding kernels
        # would spuriously reject valid curved meshes.
        components = trimesh.graph.connected_components(
            mesh.face_adjacency, nodes=np.arange(len(f))
        )
        if len(components) > 1:
            area = mesh.area_faces
            selected = np.array([ids[np.argmax(area[ids])] for ids in components])
            centers = tri[selected].mean(1)
            eps = np.minimum(np.sqrt(area[selected]) * 1e-3, 1e-5)
            inner = self.winding.winding_number(
                centers - eps[:, None] * self.normals[selected]
            )
            outer = self.winding.winding_number(
                centers + eps[:, None] * self.normals[selected]
            )
            if np.any(inner < 0.5) or np.any(outer > 0.5):
                raise ValueError("Condition mesh has invalid shell orientation")
        self.fingerprint = mesh_hash(v, f)

    @staticmethod
    def _points(xyz):
        x = np.ascontiguousarray(xyz, dtype=np.float64).reshape(-1, 3)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite mesh query")
        return x

    def closest(self, xyz):
        x = self._points(xyz)
        if not len(x):
            return np.empty(0), np.empty(0, np.int64), x.copy()
        parts = [
            self.tree.squared_distance(
                self.vertices, self.faces, x[start : start + QUERY_BATCH_SIZE]
            )
            for start in range(0, len(x), QUERY_BATCH_SIZE)
        ]
        return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))

    def query(self, xyz):
        x = self._points(xyz)
        squared, _, _ = self.closest(x)
        if not len(x):
            return np.empty(0, np.float64)
        inside = (
            np.concatenate(
                [
                    self.winding.winding_number(x[start : start + QUERY_BATCH_SIZE])
                    for start in range(0, len(x), QUERY_BATCH_SIZE)
                ]
            )
            > 0.5
        )
        return np.sqrt(np.maximum(squared, 0)) * np.where(inside, -1.0, 1.0)

    def gradient(self, xyz):
        """Analytic gradient plus validity; ties and nonsmooth zero points are masked.

        Six small distance probes only validate differentiability; they do not
        replace the analytic gradient. This cost occurs only for gradient queries.
        """
        import igl

        x = self._points(xyz)
        d2, ids, closest = self.closest(x)
        distance = np.sqrt(np.maximum(d2, 0))
        sdf = self.query(x)
        far = distance > 1e-9
        g = self.normals[ids].astype(np.float64) if len(x) else x.copy()
        g[far] = (
            (x[far] - closest[far])
            / distance[far, None]
            * np.where(sdf[far] < 0, -1, 1)[:, None]
        )
        valid = np.ones(len(x), bool)
        if (~far).any():
            tri = self.vertices[self.faces[ids[~far]]]
            bary = igl.barycentric_coordinates(
                closest[~far], tri[:, 0], tri[:, 1], tri[:, 2]
            )
            valid[~far] = bary.min(1) > 1e-5
        eps = 1e-6
        fd = np.empty_like(x)
        for axis in range(3):
            delta = np.eye(3)[axis] * eps
            fd[:, axis] = (self.query(x + delta) - self.query(x - delta)) / (2 * eps)
        valid &= np.linalg.norm(fd - g, axis=1) < 2e-3
        return g.astype(np.float32), valid
