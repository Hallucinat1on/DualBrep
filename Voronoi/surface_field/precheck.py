"""Input validation: STEP topology checks and OBJ mesh checks.

All heavy dependencies (pythonocc, numpy, trimesh) are imported lazily inside
the functions so importing this module stays cheap.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .common import PipelineError


def _count_shapes(shape: Any, shape_type: Any) -> tuple[int, list[Any]]:
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopTools import TopTools_MapOfShape
    explorer = TopExp_Explorer(shape, shape_type)
    seen = TopTools_MapOfShape()
    values: list[Any] = []
    while explorer.More():
        item = explorer.Current()
        if not seen.Contains(item):
            seen.Add(item)
            values.append(item)
        explorer.Next()
    return len(values), values


def precheck_step(step_path: Path) -> dict[str, Any]:
    try:
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
        from OCC.Core.TopoDS import topods
        from OCC.Extend.DataExchange import read_step_file
    except ImportError as exc:
        raise PipelineError("missing_pythonocc", str(exc)) from exc

    try:
        shape = read_step_file(str(step_path), verbosity=False)
    except Exception as exc:
        raise PipelineError("step_read_failed", str(exc)) from exc
    if shape.IsNull():
        raise PipelineError("step_read_failed", "STEP reader returned a null shape")

    solid_count, _ = _count_shapes(shape, TopAbs_SOLID)
    shell_count, _ = _count_shapes(shape, TopAbs_SHELL)
    face_count, _ = _count_shapes(shape, TopAbs_FACE)
    edge_count, edges = _count_shapes(shape, TopAbs_EDGE)
    degenerated = sum(bool(BRep_Tool.Degenerated(topods.Edge(edge))) for edge in edges)
    valid = bool(BRepCheck_Analyzer(shape).IsValid())
    properties = GProp_GProps()
    try:
        brepgprop.VolumeProperties(shape, properties)
        volume = float(properties.Mass())
    except Exception:
        volume = math.nan
    result = {
        "solid_count": solid_count,
        "shell_count": shell_count,
        "face_count": face_count,
        "edge_count": edge_count,
        "degenerated_edge_count": degenerated,
        "brep_valid": valid,
        "volume": volume,
    }
    if solid_count != 1:
        raise PipelineError("unsupported_topology", f"expected one solid, found {solid_count}")
    if face_count < 2:
        raise PipelineError("unsupported_topology", f"need at least two faces, found {face_count}")
    if not valid:
        raise PipelineError("invalid_solid", "OpenCASCADE BRepCheck reports invalid shape")
    if degenerated:
        raise PipelineError("unsupported_topology", f"found {degenerated} degenerated edges")
    if not math.isfinite(volume) or abs(volume) <= 1e-18:
        raise PipelineError("invalid_solid", f"invalid solid volume: {volume}")
    return result


def load_triangle_mesh(path: Path) -> Any:
    try:
        import numpy as np
        import trimesh
    except ImportError as exc:
        raise PipelineError("missing_mesh_dependency", str(exc)) from exc
    try:
        mesh = trimesh.load(str(path), force="mesh", process=False, maintain_order=True)
    except Exception as exc:
        raise PipelineError("mesh_read_failed", str(exc)) from exc
    if not isinstance(mesh, trimesh.Trimesh):
        raise PipelineError("mesh_read_failed", f"expected Trimesh, got {type(mesh).__name__}")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise PipelineError("mesh_invalid", f"invalid vertices shape {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise PipelineError("mesh_invalid", f"invalid triangles shape {faces.shape}")
    if not np.isfinite(vertices).all():
        raise PipelineError("mesh_invalid", "non-finite vertex coordinates")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise PipelineError("mesh_invalid", "triangle index outside vertex range")
    return mesh


def precheck_obj(mesh: Any, require_watertight: bool = True) -> dict[str, Any]:
    import numpy as np
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri = vertices[faces]
    doubled_area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    extent = float(np.max(np.ptp(vertices, axis=0)))
    area_eps = max(1e-30, extent * extent * 1e-14)
    degenerate_count = int(np.count_nonzero(doubled_area <= area_eps))

    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = int(np.count_nonzero(counts == 1))
    nonmanifold_edges = int(np.count_nonzero(counts > 2))
    result = {
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(faces)),
        "degenerate_triangle_count": degenerate_count,
        "boundary_edge_count": boundary_edges,
        "nonmanifold_edge_count": nonmanifold_edges,
        "watertight": bool(mesh.is_watertight),
        "bounds": np.asarray(mesh.bounds, dtype=float).tolist(),
    }
    if degenerate_count:
        raise PipelineError("mesh_invalid", f"found {degenerate_count} degenerate triangles")
    if nonmanifold_edges:
        raise PipelineError("mesh_invalid", f"found {nonmanifold_edges} non-manifold edges")
    if require_watertight and not mesh.is_watertight:
        raise PipelineError("mesh_invalid", f"mesh is open ({boundary_edges} boundary edges)")
    return result
