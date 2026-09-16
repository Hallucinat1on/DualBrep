"""Experimental SciPy Voronoi backend for smoke tests only.

It reproduces the C++ normalization, tessellates each CAD face separately
(keeping face IDs), samples labeled surface sites, and extracts finite Voronoi
ridges between sites of different faces.  Outputs are marked
``experimental=true`` and must be regenerated with the C++ backend before
training; caches are not shared across backends.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

from .common import PipelineError, atomic_write_text, sha256_file, write_json


def _normalized_step_triangles(step_path: Path, deflection: float) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Tessellate each STEP face after applying the C++ normalization formula."""
    import numpy as np
    try:
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.BRepBndLib import brepbndlib
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopLoc import TopLoc_Location
        from OCC.Core.TopoDS import topods
        from OCC.Core.gp import gp_Trsf, gp_Vec
        from OCC.Extend.DataExchange import read_step_file
    except ImportError as exc:
        raise PipelineError("missing_pythonocc", str(exc)) from exc

    shape = read_step_file(str(step_path), verbosity=False)
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    spans = np.asarray([xmax - xmin, ymax - ymin, zmax - zmin], dtype=np.float64)
    if not np.isfinite(spans).all() or np.any(spans <= 0):
        raise PipelineError("normalization_mismatch", f"invalid STEP bbox spans: {spans.tolist()}")
    translation = -0.5 * np.asarray([xmin + xmax, ymin + ymax, zmin + zmax])
    scale = float(1.8 / spans.max())
    move = gp_Trsf()
    move.SetTranslationPart(gp_Vec(*translation.tolist()))
    resize = gp_Trsf()
    resize.SetScaleFactor(scale)
    resize.Multiply(move)
    transformer = BRepBuilderAPI_Transform(resize)
    transformer.Perform(shape)
    normalized_shape = transformer.Shape()
    BRepMesh_IncrementalMesh(normalized_shape, deflection, True, 0.1)

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    face_ids: list[int] = []
    explorer = TopExp_Explorer(normalized_shape, TopAbs_FACE)
    cad_face_id = 0
    while explorer.More():
        face = topods.Face(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is None:
            raise PipelineError("step_triangulation_failed", f"face {cad_face_id} has no triangulation")
        local_to_global = location.Transformation()
        offset = len(vertices)
        for index in range(1, triangulation.NbNodes() + 1):
            point = triangulation.Node(index).Transformed(local_to_global)
            vertices.append([point.X(), point.Y(), point.Z()])
        for index in range(1, triangulation.NbTriangles() + 1):
            triangle = triangulation.Triangle(index)
            a, b, c = triangle.Value(1), triangle.Value(2), triangle.Value(3)
            if face.Orientation() == TopAbs_REVERSED:
                a, c = c, a
            faces.append([offset + a - 1, offset + b - 1, offset + c - 1])
            face_ids.append(cad_face_id)
        cad_face_id += 1
        explorer.Next()
    if not faces:
        raise PipelineError("step_triangulation_failed", "STEP tessellation is empty")
    transform = {
        "formula": "x_normalized = scale * (x_original + translation)",
        "translation": translation.tolist(),
        "scale": scale,
    }
    return (np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32),
            np.asarray(face_ids, dtype=np.int32), transform)


def _labeled_surface_samples(vertices: Any, faces: Any, face_ids: Any,
                             count: int, min_per_face: int, seed: int) -> tuple[Any, Any]:
    import numpy as np
    triangles = vertices[faces]
    doubled_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    valid = doubled_area > 1e-16
    triangles, doubled_area, face_ids = triangles[valid], doubled_area[valid], face_ids[valid]
    unique = np.unique(face_ids)
    minimum_total = len(unique) * min_per_face
    if count < minimum_total:
        raise PipelineError("scipy_voronoi_budget_too_small",
                            f"need at least {minimum_total} points for {len(unique)} faces")
    rng = np.random.default_rng(seed)
    sampled_triangles: list[Any] = []
    sampled_labels: list[Any] = []
    remaining = count - minimum_total
    face_areas = np.asarray([doubled_area[face_ids == item].sum() for item in unique])
    extra = rng.multinomial(remaining, face_areas / face_areas.sum())
    for cad_face, extra_count in zip(unique, extra):
        mask = face_ids == cad_face
        local_triangles = triangles[mask]
        local_area = doubled_area[mask]
        local_count = min_per_face + int(extra_count)
        choice = rng.choice(len(local_triangles), size=local_count, replace=True,
                            p=local_area / local_area.sum())
        sampled_triangles.append(local_triangles[choice])
        sampled_labels.append(np.full(local_count, cad_face, dtype=np.int32))
    selected = np.concatenate(sampled_triangles, axis=0)
    labels = np.concatenate(sampled_labels)
    r1, r2 = rng.random((2, len(selected)))
    root = np.sqrt(r1)
    bary = np.stack((1.0 - root, root * (1.0 - r2), root * r2), axis=1)
    points = np.einsum("ni,nij->nj", bary, selected)
    return points, labels


def run_scipy_voronoi(step_path: Path, work_dir: Path, point_count: int,
                      min_per_face: int, deflection: float, seed: int,
                      resume: bool, overwrite: bool) -> None:
    """Experimental smoke-test fallback; not equivalent to the production C++ sampler."""
    import numpy as np
    import trimesh
    try:
        from scipy.spatial import QhullError, Voronoi
    except ImportError as exc:
        raise PipelineError("missing_scipy", str(exc)) from exc
    required = (work_dir / "voronoi.ply", work_dir / "normalized_mesh.ply",
                work_dir / "normalized_params.txt")
    marker = work_dir / "voronoi_backend.json"
    if resume and all(path.is_file() and path.stat().st_size > 0 for path in required) and marker.is_file():
        provenance = json.loads(marker.read_text())
        if (provenance.get("backend") == "scipy"
                and provenance.get("step_sha256") == sha256_file(step_path)
                and provenance.get("point_count") == point_count
                and provenance.get("min_per_face") == min_per_face
                and math.isclose(float(provenance.get("deflection", -1.0)), deflection,
                                 rel_tol=0.0, abs_tol=1e-15)):
            return
    if work_dir.exists():
        if not overwrite:
            raise PipelineError("partial_work_exists", f"incomplete work directory: {work_dir}")
        archive = work_dir.with_name(f"{work_dir.name}.failed-{int(time.time())}")
        os.replace(work_dir, archive)
    work_dir.mkdir(parents=True, exist_ok=False)

    vertices, faces, face_ids, transform = _normalized_step_triangles(step_path, deflection)
    points, labels = _labeled_surface_samples(
        vertices, faces, face_ids, point_count, min_per_face, seed)
    # Bounding corners make all surface sites interior to the convex hull.  Ridges
    # touching these synthetic sites are intentionally excluded below.
    corners = np.asarray([[x, y, z] for x in (-2.0, 2.0)
                          for y in (-2.0, 2.0) for z in (-2.0, 2.0)], dtype=np.float64)
    sites = np.concatenate((points, corners), axis=0)
    site_labels = np.concatenate((labels, np.full(len(corners), -1, dtype=np.int32)))
    try:
        diagram = Voronoi(sites, qhull_options="Qbb Qc Qz QJ")
    except QhullError as exc:
        raise PipelineError("scipy_voronoi_failed", str(exc)) from exc

    out_vertices: list[Any] = []
    out_faces: list[list[int]] = []
    for pair, ridge_indices in zip(diagram.ridge_points, diagram.ridge_vertices):
        if np.any(pair >= len(site_labels)):
            continue
        label_a, label_b = site_labels[pair[0]], site_labels[pair[1]]
        if label_a < 0 or label_b < 0 or label_a == label_b:
            continue
        if len(ridge_indices) < 3 or any(index < 0 for index in ridge_indices):
            continue
        polygon = np.asarray(diagram.vertices[ridge_indices], dtype=np.float64)
        if not np.isfinite(polygon).all():
            continue
        center = polygon.mean(axis=0)
        normal = sites[pair[1]] - sites[pair[0]]
        normal_norm = np.linalg.norm(normal)
        first = polygon[0] - center
        first_norm = np.linalg.norm(first)
        if normal_norm <= 1e-14 or first_norm <= 1e-14:
            continue
        normal /= normal_norm
        axis_u = first / first_norm
        axis_v = np.cross(normal, axis_u)
        axis_v_norm = np.linalg.norm(axis_v)
        if axis_v_norm <= 1e-14:
            continue
        axis_v /= axis_v_norm
        offsets = polygon - center
        order = np.argsort(np.arctan2(offsets @ axis_v, offsets @ axis_u))
        polygon = polygon[order]
        base = len(out_vertices)
        out_vertices.append(center)
        out_vertices.extend(polygon)
        for index in range(len(polygon)):
            out_faces.append([base, base + 1 + index, base + 1 + (index + 1) % len(polygon)])
    if len(out_faces) < 10:
        raise PipelineError("voronoi_empty", f"SciPy fallback generated {len(out_faces)} triangles")

    normalized_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    voronoi_mesh = trimesh.Trimesh(vertices=np.asarray(out_vertices),
                                   faces=np.asarray(out_faces, dtype=np.int32), process=False)
    normalized_mesh.export(work_dir / "normalized_mesh.ply")
    voronoi_mesh.export(work_dir / "voronoi.ply")
    trimesh.points.PointCloud(points).export(work_dir / "sampled_points.ply")
    atomic_write_text(work_dir / "normalized_params.txt",
                      " ".join(f"{value:.17g}" for value in
                               (*transform["translation"], transform["scale"])) + "\n")
    write_json(work_dir / "scipy_voronoi.json", {
        "experimental": True,
        "point_count": int(point_count),
        "min_per_face": int(min_per_face),
        "cad_face_count": int(len(np.unique(face_ids))),
        "ridge_triangle_count": int(len(out_faces)),
        "warning": "Smoke-test fallback; regenerate with the C++ backend before training.",
    })
    write_json(marker, {
        "backend": "scipy",
        "experimental": True,
        "step_sha256": sha256_file(step_path),
        "point_count": int(point_count),
        "min_per_face": int(min_per_face),
        "deflection": float(deflection),
    })
