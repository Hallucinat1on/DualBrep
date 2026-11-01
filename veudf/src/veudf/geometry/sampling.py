"""Deterministic area sampling and face normals on a fixed mesh."""

import hashlib
import numpy as np
from veudf.schema import GEOMETRY_VERSION
from veudf.utils import array_hash, seed_for, draw


def triangle_samples(vertices, faces, count, rng, ids=None, *, return_ids=False):
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
    ).astype(np.float32)
    return (points, ids[chosen]) if return_ids else points


def face_normals(vertices, faces):
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return (n / np.linalg.norm(n, axis=1, keepdims=True)).astype(np.float32)


def base_surface(mesh, count, build_seed=0, *, return_ids=False):
    vertices, faces = mesh
    rng = np.random.default_rng(
        seed_for(GEOMETRY_VERSION, mesh_hash(vertices, faces), int(build_seed))
    )
    points, ids = triangle_samples(vertices, faces, count, rng, return_ids=True)
    result = points, face_normals(vertices, faces)[ids]
    return (*result, ids) if return_ids else result


def mesh_hash(vertices, faces):
    return hashlib.sha256(
        (array_hash(vertices) + array_hash(faces)).encode()
    ).hexdigest()


def geometry_rows(points, normals, count, seed):
    """Never read target labels, instance IDs or feature-dependent surface extensions."""
    identity = array_hash(points) + array_hash(normals)
    rng = np.random.default_rng(seed_for(GEOMETRY_VERSION, identity, int(seed)))
    return draw(rng, np.arange(len(points)), count)
