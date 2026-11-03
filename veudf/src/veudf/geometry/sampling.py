"""Deterministic area sampling and face normals on a fixed mesh."""

import hashlib
import numpy as np
from veudf.schema import GEOMETRY_VERSION
from veudf.utils import array_hash, seed_for, draw


def triangle_samples(
    vertices,
    faces,
    count,
    rng,
    ids=None,
    *,
    return_ids=False,
    return_barycentric=False,
    dtype=np.float32
):
    """Area-uniform barycentric samples of the final mesh; no CAD-face quota."""
    ids = np.arange(len(faces)) if ids is None else np.asarray(ids)
    tri = vertices[faces[ids]]
    area = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    if count and (not len(tri) or area.sum() <= 0):
        raise ValueError("No nondegenerate triangles")
    chosen = (
        rng.choice(len(tri), count, p=area / area.sum()) if count else np.empty(0, int)
    )
    uv = rng.random((count, 2))
    uv[uv.sum(1) > 1] = 1 - uv[uv.sum(1) > 1]
    t = tri[chosen]
    points = (
        t[:, 0] + uv[:, :1] * (t[:, 1] - t[:, 0]) + uv[:, 1:] * (t[:, 2] - t[:, 0])
    ).astype(dtype)
    if return_barycentric:
        return points, ids[chosen], uv
    return (points, ids[chosen]) if return_ids else points


def face_normals(vertices, faces):
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return (n / np.linalg.norm(n, axis=1, keepdims=True)).astype(np.float32)


def base_surface(
    mesh, count, build_seed=0, *, return_ids=False, return_barycentric=False
):
    vertices, faces = mesh
    rng = np.random.default_rng(
        seed_for(GEOMETRY_VERSION, mesh_hash(vertices, faces), int(build_seed))
    )
    points, ids, uv = triangle_samples(
        vertices, faces, count, rng, return_barycentric=True
    )
    result = points, face_normals(vertices, faces)[ids]
    if return_barycentric:
        return (*result, ids, uv)
    return (*result, ids) if return_ids else result


SURFACE_TOLERANCE = 2e-7  # float32 storage inside the normalized [-.9,.9] box


def surface_barycentric(points, triangles):
    """Stable cross-product coordinates, evaluated before float32 point storage."""
    a, b, c = np.moveaxis(np.asarray(triangles, float), 1, 0)
    p = np.asarray(points, float)
    normal = np.cross(b - a, c - a)
    denominator = np.einsum("ij,ij->i", normal, normal)
    u = np.einsum("ij,ij->i", np.cross(p - a, c - a), normal) / denominator
    v = np.einsum("ij,ij->i", np.cross(b - a, p - a), normal) / denominator
    return np.column_stack((u, v))


def attach_surface_coordinates(pool, mesh):
    """Validate provenance without nearest-surface searches or SDF queries."""
    ids = pool["triangle_ids"]
    if len(ids) and ((ids < 0).any() or (ids >= len(mesh[1])).any()):
        raise ValueError("Invalid surface triangle IDs")
    tri = mesh[0][mesh[1][ids]]
    uv = np.asarray(
        (
            pool["barycentric"]
            if "barycentric" in pool
            else surface_barycentric(pool["points"], tri)
        ),
        float,
    )
    weights = np.column_stack((1 - uv.sum(1), uv))
    if not np.isfinite(weights).all() or (weights < -1e-8).any():
        raise ValueError("Surface sample lies outside its triangle")
    reconstructed = np.einsum("ni,nij->nj", weights, tri)
    error = np.linalg.norm(reconstructed - pool["points"], axis=1)
    if (error > SURFACE_TOLERANCE).any():
        raise ValueError("Surface coordinates do not reconstruct stored points")
    pool["barycentric"] = uv  # retain float64 coordinates even on very skinny faces
    return float(error.max()) if len(error) else 0.0


def mesh_hash(vertices, faces):
    return hashlib.sha256(
        (array_hash(vertices) + array_hash(faces)).encode()
    ).hexdigest()


def geometry_rows(points, normals, count, seed):
    """Never read target labels, instance IDs or feature-dependent surface extensions."""
    identity = array_hash(points) + array_hash(normals)
    rng = np.random.default_rng(seed_for(GEOMETRY_VERSION, identity, int(seed)))
    return draw(rng, np.arange(len(points)), count)
