"""Pure geometry helpers: normalization transforms, sampling, alignment, and UDF queries.

No file-schema knowledge and no subprocesses here; everything operates on
in-memory meshes.  numpy/scipy/igl/open3d/trimesh are imported lazily.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from .common import PipelineError


def parse_transform(path: Path) -> dict[str, Any]:
    import numpy as np
    try:
        values = [float(item) for item in path.read_text().split()]
    except Exception as exc:
        raise PipelineError("normalization_mismatch", f"cannot read {path}: {exc}") from exc
    if len(values) != 4:
        raise PipelineError("normalization_mismatch", f"expected 4 values, got {len(values)}")
    translation = np.asarray(values[:3], dtype=np.float64)
    scale = float(values[3])
    if not np.isfinite(translation).all() or not math.isfinite(scale) or scale <= 0:
        raise PipelineError("normalization_mismatch", f"invalid transform values: {values}")
    forward = np.eye(4, dtype=np.float64)
    forward[:3, :3] *= scale
    forward[:3, 3] = scale * translation
    inverse = np.linalg.inv(forward)
    return {
        "formula": "x_normalized = scale * (x_original + translation)",
        "translation": translation.tolist(),
        "scale": scale,
        "forward_matrix": forward.tolist(),
        "inverse_matrix": inverse.tolist(),
    }


def transform_mesh(mesh: Any, transform: dict[str, Any]) -> Any:
    import numpy as np
    result = mesh.copy()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    translation = np.asarray(transform["translation"], dtype=np.float64)
    result.vertices = (vertices + translation[None, :]) * float(transform["scale"])
    return result


def sample_surface(mesh: Any, count: int, seed: int) -> tuple[Any, Any, Any, Any]:
    import numpy as np
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    doubled_area = np.linalg.norm(cross, axis=1)
    total = float(doubled_area.sum())
    if not math.isfinite(total) or total <= 0:
        raise PipelineError("mesh_invalid", "mesh has zero total area")
    rng = np.random.default_rng(seed)
    face_id = rng.choice(len(faces), size=count, replace=True, p=doubled_area / total)
    r1, r2 = rng.random((2, count))
    root = np.sqrt(r1)
    bary = np.stack((1.0 - root, root * (1.0 - r2), root * r2), axis=1)
    points = np.einsum("ni,nij->nj", bary, triangles[face_id])
    normals = cross[face_id] / doubled_area[face_id, None]
    return (points.astype(np.float32), normals.astype(np.float32),
            face_id.astype(np.int32), bary.astype(np.float32))


def alignment_metrics(mesh: Any, reference: Any, seed: int,
                      sample_count: int = 2048) -> dict[str, Any]:
    import numpy as np
    from scipy.spatial import cKDTree
    bounds_a = np.asarray(mesh.bounds, dtype=np.float64)
    bounds_b = np.asarray(reference.bounds, dtype=np.float64)
    center_a, center_b = bounds_a.mean(0), bounds_b.mean(0)
    extent_a, extent_b = np.ptp(bounds_a, axis=0), np.ptp(bounds_b, axis=0)
    denominator = np.maximum(np.maximum(extent_a, extent_b), 1e-12)
    pa = sample_surface(mesh, sample_count, seed)[0]
    pb = sample_surface(reference, sample_count, seed + 1)[0]
    da = cKDTree(pb).query(pa, workers=1)[0]
    db = cKDTree(pa).query(pb, workers=1)[0]
    all_dist = np.concatenate((da, db))
    return {
        "bbox_center_max_abs": float(np.max(np.abs(center_a - center_b))),
        "bbox_extent_max_relative": float(np.max(np.abs(extent_a - extent_b) / denominator)),
        "sampled_symmetric_mean": float(np.mean(all_dist)),
        "sampled_symmetric_p95": float(np.quantile(all_dist, 0.95)),
    }


def unsigned_distance(query_xyz: Any, mesh: Any, backend: str,
                      batch_size: int) -> tuple[Any, str]:
    import numpy as np
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    query = np.asarray(query_xyz, dtype=np.float64)
    if backend in ("auto", "igl"):
        try:
            import igl
            chunks = []
            for start in range(0, len(query), batch_size):
                dist = igl.signed_distance(
                    query[start:start + batch_size], vertices, faces,
                    sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED,
                )[0]
                chunks.append(np.asarray(dist).reshape(-1))
            return np.concatenate(chunks).astype(np.float32), "igl"
        except ImportError:
            if backend == "igl":
                raise PipelineError("missing_distance_backend", "libigl is not installed")
    if backend in ("auto", "open3d"):
        try:
            import open3d as o3d
            tensor_mesh = o3d.t.geometry.TriangleMesh(
                o3d.core.Tensor(vertices.astype(np.float32), o3d.core.Dtype.Float32),
                o3d.core.Tensor(faces, o3d.core.Dtype.Int32),
            )
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(tensor_mesh)
            chunks = []
            for start in range(0, len(query), batch_size):
                tensor = o3d.core.Tensor(query[start:start + batch_size].astype(np.float32),
                                         o3d.core.Dtype.Float32)
                chunks.append(scene.compute_distance(tensor).numpy())
            return np.concatenate(chunks).astype(np.float32), "open3d"
        except ImportError:
            if backend == "open3d":
                raise PipelineError("missing_distance_backend", "Open3D is not installed")
    if backend not in ("auto", "naive"):
        raise PipelineError("missing_distance_backend", f"unknown backend {backend}")
    import trimesh
    chunks = []
    for start in range(0, len(query), batch_size):
        _, dist, _ = trimesh.proximity.closest_point_naive(mesh, query[start:start + batch_size])
        chunks.append(np.asarray(dist).reshape(-1))
    return np.concatenate(chunks).astype(np.float32), "trimesh_naive"


def mesh_hash(vertices: Any, faces: Any) -> str:
    import numpy as np
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(vertices).tobytes())
    digest.update(np.ascontiguousarray(faces).tobytes())
    return digest.hexdigest()
