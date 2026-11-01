"""Geometry-only condition packages; no target labels or model imports."""

import json
from pathlib import Path
import numpy as np
from veudf.schema import GEOMETRY_VERSION
from veudf.utils import array_hash
from veudf.geometry.sampling import mesh_hash, base_surface
from veudf.geometry.mesh import MeshSDFOracle


def read_geometry(path):
    """Read a mesh condition and regenerate S0, without opening any GT labels."""
    p = Path(path)
    if (p / "condition").is_dir():
        p = p / "condition"
    meta = json.loads((p / "meta.json").read_text())
    if meta.get("geometry_version") != GEOMETRY_VERSION:
        raise ValueError("Unsupported geometry sampling version")
    vertices = np.load(p / "vertices.npy", allow_pickle=False)
    faces = np.load(p / "triangles.npy", allow_pickle=False)
    if vertices.dtype != np.float64 or faces.dtype != np.int64:
        raise ValueError("Condition mesh requires float64 vertices and int64 triangles")
    if mesh_hash(vertices, faces) != meta["mesh_sha256"]:
        raise ValueError("Condition mesh changed")
    oracle = MeshSDFOracle(vertices, faces)
    points, normal = base_surface(
        (vertices, faces), int(meta["surface_count"]), meta["build_seed"]
    )
    if (
        array_hash(points) != meta["points_sha256"]
        or array_hash(normal) != meta["normals_sha256"]
    ):
        raise ValueError("Geometry surface changed")
    return oracle, points, normal, meta


def write_condition(path, mesh, count=393216, build_seed=0, normalization=None):
    """Publish geometry-only arrays and all information needed to reproduce S0."""
    vertices, faces = np.asarray(mesh[0], np.float64), np.asarray(mesh[1], np.int64)
    mesh = vertices, faces
    if (
        not np.isfinite(vertices).all()
        or np.abs(vertices).max() > 0.900001
        or count < 1
    ):
        raise ValueError(
            "Condition requires positive S0 count and geometry inside [-.9,.9]^3"
        )
    MeshSDFOracle(vertices, faces)  # Reject invalid input before writing.
    points, normals = base_surface(mesh, count, build_seed)
    p = Path(path)
    p.mkdir(parents=True, exist_ok=False)
    np.save(p / "vertices.npy", vertices, allow_pickle=False)
    np.save(p / "triangles.npy", faces, allow_pickle=False)
    meta = dict(
        geometry_version=GEOMETRY_VERSION,
        mesh_sha256=mesh_hash(vertices, faces),
        surface_count=int(count),
        build_seed=int(build_seed),
        points_sha256=array_hash(points),
        normals_sha256=array_hash(normals),
        normalization=normalization or {},
    )
    (p / "meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False))
    return meta
