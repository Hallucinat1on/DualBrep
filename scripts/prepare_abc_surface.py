#!/usr/bin/env python3
"""Build mesh-conditioned Voronoi-UDF surface-field data from ABC STEP/OBJ pairs.

The C++ ``calculate_voronoi`` executable remains the source of the Voronoi
partition.  This script owns reproducible pairing, validation, coordinate
alignment, per-triangle queries, and the on-disk training schema.

Typical smoke run::

    python scripts/prepare_abc_surface.py run \
      --abc-root /opt/data/private/yihengxu/Datasets/abc \
      --output-root /opt/data/private/yihengxu/Datasets/surface \
      --voronoi-exe Voronoi/build/calculate_voronoi/calculate_voronoi \
      --limit 3 --condition-points 4096
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "abc-surface-v1"
DEFAULT_ABC_ROOT = Path("/opt/data/private/yihengxu/Datasets/abc")
DEFAULT_OUTPUT_ROOT = Path("/opt/data/private/yihengxu/Datasets/surface")
DEFAULT_VORONOI_EXE = Path("Voronoi/build/calculate_voronoi/calculate_voronoi")
ASSET_RE = re.compile(
    r"^(?P<shape_id>\d{8})_(?P<token>[^_]+)_"
    r"(?P<kind>step|trimesh)_(?P<part>\d+)\.(?P<ext>step|stp|obj)$",
    re.IGNORECASE,
)


class PipelineError(RuntimeError):
    """An expected, classifiable per-shape failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Asset:
    shape_id: str
    token: str
    part: str
    kind: str
    path: Path

    @property
    def pair_key(self) -> tuple[str, str, str]:
        return self.shape_id, self.token, self.part


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True,
                                       ensure_ascii=False, default=_json_default) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False,
                              default=_json_default) + "\n" for row in rows)
    atomic_write_text(path, text)


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_asset(path: Path, expected_kind: str) -> Asset | None:
    match = ASSET_RE.match(path.name)
    if match is None or match.group("kind").lower() != expected_kind:
        return None
    return Asset(
        shape_id=match.group("shape_id"),
        token=match.group("token"),
        part=match.group("part"),
        kind=expected_kind,
        path=path.resolve(),
    )


def scan_assets(root: Path, kind: str) -> list[Asset]:
    suffixes = {"step": {".step", ".stp"}, "trimesh": {".obj"}}[kind]
    assets: list[Asset] = []
    if not root.is_dir():
        return assets
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            asset = parse_asset(path, kind)
            if asset is not None:
                assets.append(asset)
    return sorted(assets, key=lambda item: (item.pair_key, str(item.path)))


def stable_split(group_id: str, seed: int, train_ratio: float,
                 val_ratio: float) -> str:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def build_inventory(args: argparse.Namespace) -> list[dict[str, Any]]:
    abc_root = Path(args.abc_root).resolve()
    output_root = Path(args.output_root).resolve()
    steps = scan_assets(abc_root / "step", "step")
    objs = scan_assets(abc_root / "obj", "trimesh")

    by_step: dict[tuple[str, str, str], list[Asset]] = {}
    by_obj: dict[tuple[str, str, str], list[Asset]] = {}
    for asset in steps:
        by_step.setdefault(asset.pair_key, []).append(asset)
    for asset in objs:
        by_obj.setdefault(asset.pair_key, []).append(asset)

    records: list[dict[str, Any]] = []
    for key in sorted(set(by_step) | set(by_obj)):
        step_items, obj_items = by_step.get(key, []), by_obj.get(key, [])
        status = "paired"
        if not step_items:
            status = "missing_step"
        elif not obj_items:
            status = "missing_obj"
        elif len(step_items) != 1 or len(obj_items) != 1:
            status = "pair_conflict"
        shape_id, token, part = key
        group_id = token
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "shape_id": shape_id,
            "chunk": f"{int(shape_id) // 10000:04d}",
            "token": token,
            "part": part,
            "group_id": group_id,
            "split": stable_split(group_id, args.seed, args.train_ratio, args.val_ratio),
            "pair_status": status,
            "mesh_source": "abc_obj" if obj_items else None,
            "step_path": str(step_items[0].path) if len(step_items) == 1 else None,
            "obj_path": str(obj_items[0].path) if len(obj_items) == 1 else None,
            "step_candidates": [str(item.path) for item in step_items],
            "obj_candidates": [str(item.path) for item in obj_items],
        }
        if len(step_items) == 1:
            row["step_bytes"] = step_items[0].path.stat().st_size
        if len(obj_items) == 1:
            row["obj_bytes"] = obj_items[0].path.stat().st_size
        if args.hash_inputs and status == "paired":
            row["step_sha256"] = sha256_file(step_items[0].path)
            row["obj_sha256"] = sha256_file(obj_items[0].path)
        records.append(row)

    write_jsonl(output_root / "manifest.jsonl", records)
    split_root = output_root / "splits"
    for split in ("train", "val", "test"):
        ids = [row["shape_id"] for row in records
               if row["pair_status"] == "paired" and row["split"] == split]
        atomic_write_text(split_root / f"{split}.txt", "".join(f"{item}\n" for item in ids))
    counts: dict[str, int] = {}
    for row in records:
        counts[row["pair_status"]] = counts.get(row["pair_status"], 0) + 1
    write_json(output_root / "reports" / "inventory.json", {
        "schema_version": SCHEMA_VERSION,
        "abc_root": str(abc_root),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "step_files": len(steps),
        "obj_files": len(objs),
        "pair_counts": counts,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "hash_inputs": bool(args.hash_inputs),
    })
    write_json(output_root / "config.json", {
        "schema_version": SCHEMA_VERSION,
        "abc_root": str(abc_root),
        "output_root": str(output_root),
        "tau": getattr(args, "tau", 0.015),
        "seed": args.seed,
    })
    print(f"inventory: {len(steps)} STEP, {len(objs)} OBJ, "
          f"{counts.get('paired', 0)} exact pairs")
    return records


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def _count_shapes(shape: Any, shape_type: Any) -> tuple[int, list[Any]]:
    from OCC.Core.TopExp import TopExp_Explorer
    explorer = TopExp_Explorer(shape, shape_type)
    seen: set[int] = set()
    values: list[Any] = []
    while explorer.More():
        item = explorer.Current()
        key = int(item.HashCode(2147483647))
        if key not in seen:
            seen.add(key)
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


def run_voronoi(executable: Path, step_path: Path, work_dir: Path,
                timeout_seconds: int, resume: bool, overwrite: bool) -> None:
    required = (work_dir / "voronoi.ply", work_dir / "normalized_mesh.ply",
                work_dir / "normalized_params.txt")
    marker = work_dir / "voronoi_backend.json"
    if resume and all(path.is_file() and path.stat().st_size > 0 for path in required) and marker.is_file():
        provenance = json.loads(marker.read_text())
        if provenance.get("backend") == "cpp" and provenance.get("step_sha256") == sha256_file(step_path):
            return
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise PipelineError("missing_voronoi_executable", f"not executable: {executable}")
    if work_dir.exists():
        if not overwrite:
            raise PipelineError("partial_work_exists", f"incomplete work directory: {work_dir}")
        archive = work_dir.with_name(f"{work_dir.name}.failed-{int(time.time())}")
        os.replace(work_dir, archive)
    work_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = subprocess.run(
            [str(executable), str(step_path), str(work_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        atomic_write_text(work_dir / "voronoi.log", exc.stdout or "")
        raise PipelineError("voronoi_timeout", f"exceeded {timeout_seconds}s") from exc
    atomic_write_text(work_dir / "voronoi.log", result.stdout)
    if result.returncode != 0:
        raise PipelineError("voronoi_crash", f"exit code {result.returncode}")
    missing = [str(path.name) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise PipelineError("voronoi_empty", f"missing outputs: {', '.join(missing)}")
    write_json(marker, {
        "backend": "cpp",
        "experimental": False,
        "step_sha256": sha256_file(step_path),
        "executable": str(executable),
        "executable_sha256": sha256_file(executable),
    })


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


def _mesh_hash(vertices: Any, faces: Any) -> str:
    import numpy as np
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(vertices).tobytes())
    digest.update(np.ascontiguousarray(faces).tobytes())
    return digest.hexdigest()


def requested_processing_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "tau": float(args.tau),
        "condition_points": int(args.condition_points),
        "seed": int(args.seed),
        "voronoi_backend": args.voronoi_backend,
        "scipy_voronoi_points": int(args.scipy_voronoi_points),
        "scipy_min_face_points": int(args.scipy_min_face_points),
        "step_deflection": float(args.step_deflection),
        "distance_backend_requested": args.distance_backend,
        "alignment_samples": int(args.alignment_samples),
        "max_bbox_center_error": float(args.max_bbox_center_error),
        "max_bbox_extent_relative": float(args.max_bbox_extent_relative),
        "lipschitz_tolerance": float(args.lipschitz_tolerance),
        "allow_open_mesh": bool(args.allow_open_mesh),
    }


def process_one(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    shape_id = row["shape_id"]
    output_root = Path(args.output_root).resolve()
    sample_dir = output_root / "samples" / shape_id
    quality_file = sample_dir / "quality.json"
    processing_config = requested_processing_config(args)
    if args.resume and quality_file.is_file():
        previous = json.loads(quality_file.read_text())
        expected = previous.get("processing_config") == processing_config
        required = (sample_dir / "geometry.npz", sample_dir / "surface_field.npz",
                    sample_dir / "recon_sdf.ply", sample_dir / "udf_g.npy")
        if previous.get("status") == "success" and expected and all(path.is_file() for path in required):
            return previous
        if previous.get("status") == "success" and not args.overwrite:
            return {
                "schema_version": SCHEMA_VERSION,
                "shape_id": row["shape_id"],
                "status": "failed",
                "stage": "resume",
                "failure_code": "sample_configuration_mismatch",
                "message": "existing sample uses different settings; pass --overwrite",
                "elapsed_seconds": 0.0,
            }

    step_path = Path(row["step_path"])
    obj_path = Path(row["obj_path"])
    started = time.time()
    stage = "precheck"
    try:
        step_metrics = {} if args.skip_step_precheck else precheck_step(step_path)
        mesh = load_triangle_mesh(obj_path)
        obj_metrics = precheck_obj(mesh, require_watertight=not args.allow_open_mesh)
        if args.max_faces and len(mesh.faces) > args.max_faces:
            raise PipelineError("mesh_too_large", f"{len(mesh.faces)} > {args.max_faces}")

        stage = "voronoi"
        work_dir = output_root / "work" / shape_id
        if args.voronoi_backend == "cpp":
            executable = Path(args.voronoi_exe).resolve()
            run_voronoi(executable, step_path, work_dir, args.timeout,
                        resume=args.resume, overwrite=args.overwrite)
        else:
            if not args.allow_experimental_voronoi:
                raise PipelineError(
                    "experimental_backend_not_allowed",
                    "pass --allow-experimental-voronoi for smoke tests; do not publish these labels",
                )
            run_scipy_voronoi(
                step_path, work_dir, args.scipy_voronoi_points,
                args.scipy_min_face_points, args.step_deflection,
                args.seed + int(shape_id), args.resume, args.overwrite,
            )

        stage = "alignment"
        transform = parse_transform(work_dir / "normalized_params.txt")
        normalized = transform_mesh(mesh, transform)
        reference = load_triangle_mesh(work_dir / "normalized_mesh.ply")
        align = alignment_metrics(normalized, reference, args.seed + int(shape_id),
                                  args.alignment_samples)
        if align["bbox_center_max_abs"] > args.max_bbox_center_error:
            raise PipelineError("normalization_mismatch",
                                f"bbox center error {align['bbox_center_max_abs']:.6g}")
        if align["bbox_extent_max_relative"] > args.max_bbox_extent_relative:
            raise PipelineError("normalization_mismatch",
                                f"bbox extent error {align['bbox_extent_max_relative']:.6g}")

        stage = "field"
        vertices = np.asarray(normalized.vertices, dtype=np.float32)
        faces = np.asarray(normalized.faces, dtype=np.int32)
        triangles = vertices[faces]
        centers = triangles.mean(axis=1).astype(np.float32)
        cross = np.cross(triangles[:, 1] - triangles[:, 0],
                         triangles[:, 2] - triangles[:, 0])
        doubled_area = np.linalg.norm(cross, axis=1)
        normals = (cross / doubled_area[:, None]).astype(np.float32)
        face_area = (0.5 * doubled_area).astype(np.float32)
        adjacency = np.asarray(normalized.face_adjacency, dtype=np.int32).reshape(-1, 2)
        condition = sample_surface(normalized, args.condition_points,
                                   args.seed + int(shape_id))

        voronoi = load_triangle_mesh(work_dir / "voronoi.ply")
        udf_raw, actual_backend = unsigned_distance(
            centers, voronoi, args.distance_backend, args.query_batch_size)
        if not np.isfinite(udf_raw).all() or np.any(udf_raw < 0):
            raise PipelineError("distance_nonfinite", "distance result is negative or non-finite")
        udf_metric = np.minimum(udf_raw, args.tau).astype(np.float32)
        udf_target = (udf_metric / args.tau).astype(np.float32)

        violation = np.empty(0, dtype=np.float32)
        if len(adjacency):
            qa, qb = centers[adjacency[:, 0]], centers[adjacency[:, 1]]
            lhs = np.abs(udf_raw[adjacency[:, 0]] - udf_raw[adjacency[:, 1]])
            violation = lhs - np.linalg.norm(qa - qb, axis=1)
        max_violation = float(max(0.0, float(violation.max()))) if len(violation) else 0.0
        if max_violation > args.lipschitz_tolerance:
            raise PipelineError("distance_validation_failed",
                                f"max Lipschitz violation {max_violation:.6g}")

        stage = "write"
        samples_root = output_root / "samples"
        samples_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = samples_root / f".{shape_id}.{uuid.uuid4().hex}.tmp"
        tmp_dir.mkdir(parents=False, exist_ok=False)
        try:
            normalized.export(tmp_dir / "recon_sdf.ply")
            roundtrip = load_triangle_mesh(tmp_dir / "recon_sdf.ply")
            if (not np.array_equal(np.asarray(roundtrip.faces, dtype=np.int32), faces)
                    or not np.allclose(np.asarray(roundtrip.vertices), vertices, atol=1e-7)):
                raise PipelineError("face_index_mismatch", "PLY round-trip changed mesh ordering")
            np.savez_compressed(
                tmp_dir / "geometry.npz",
                vertices=vertices,
                triangles=faces,
                face_adjacency=adjacency,
                face_area=face_area,
                condition_points=condition[0],
                condition_normals=condition[1],
                condition_triangle_id=condition[2],
                condition_barycentric=condition[3],
            )
            np.savez_compressed(
                tmp_dir / "surface_field.npz",
                query_xyz=centers,
                query_normal=normals,
                query_triangle_id=np.arange(len(faces), dtype=np.int32),
                udf_raw=udf_raw,
                udf_metric=udf_metric,
                udf_target=udf_target,
                tau=np.float32(args.tau),
            )
            np.save(tmp_dir / "udf_g.npy", udf_metric)
            transform.update({
                "schema_version": SCHEMA_VERSION,
                "source_step": str(step_path),
                "source_obj": str(obj_path),
                "step_sha256": sha256_file(step_path),
                "obj_sha256": sha256_file(obj_path),
            })
            write_json(tmp_dir / "transform.json", transform)
            result = {
                "schema_version": SCHEMA_VERSION,
                "shape_id": shape_id,
                "status": "success",
                "stage": "validated",
                "step_path": str(step_path),
                "obj_path": str(obj_path),
                "split": row["split"],
                "mesh_source": "abc_obj",
                "mesh_hash": _mesh_hash(vertices, faces),
                "processing_config": processing_config,
                "step": step_metrics,
                "obj": obj_metrics,
                "alignment": align,
                "field": {
                    "tau": args.tau,
                    "voronoi_backend": args.voronoi_backend,
                    "experimental_voronoi": args.voronoi_backend != "cpp",
                    "distance_backend": actual_backend,
                    "minimum": float(udf_raw.min()),
                    "maximum": float(udf_raw.max()),
                    "mean": float(udf_raw.mean()),
                    "saturated_fraction": float(np.mean(udf_raw >= args.tau)),
                    "max_lipschitz_violation": max_violation,
                },
                "elapsed_seconds": time.time() - started,
            }
            write_json(tmp_dir / "quality.json", result)
            if sample_dir.exists():
                if not args.overwrite:
                    raise PipelineError("sample_exists", f"sample already exists: {sample_dir}")
                archive = sample_dir.with_name(f"{sample_dir.name}.old-{int(time.time())}")
                os.replace(sample_dir, archive)
            os.replace(tmp_dir, sample_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        return result
    except PipelineError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "shape_id": shape_id,
            "status": "failed",
            "stage": stage,
            "failure_code": exc.code,
            "message": str(exc),
            "step_path": str(step_path),
            "obj_path": str(obj_path),
            "elapsed_seconds": time.time() - started,
        }
    except Exception as exc:  # preserve unexpected failures per shape
        return {
            "schema_version": SCHEMA_VERSION,
            "shape_id": shape_id,
            "status": "failed",
            "stage": stage,
            "failure_code": "unexpected_error",
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "step_path": str(step_path),
            "obj_path": str(obj_path),
            "elapsed_seconds": time.time() - started,
        }


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("pair_status") == "paired"]
    if args.split != "all":
        selected = [row for row in selected if row.get("split") == args.split]
    if args.ids:
        wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
        selected = [row for row in selected if row["shape_id"] in wanted]
    selected.sort(key=lambda row: row["shape_id"])
    if args.limit is not None:
        selected = selected[:args.limit]
    return selected


def process_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = Path(args.output_root).resolve()
    manifest = Path(args.manifest).resolve() if args.manifest else output_root / "manifest.jsonl"
    rows = select_rows(read_manifest(manifest), args)
    if not rows:
        raise SystemExit("no paired STEP/OBJ rows selected")
    item_root = output_root / "reports" / "items"
    item_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] {row['shape_id']}", flush=True)
        result = process_one(row, args)
        write_json(item_root / f"{row['shape_id']}.json", result)
        results.append(result)
        suffix = "ok" if result["status"] == "success" else result["failure_code"]
        print(f"  {suffix} ({result['elapsed_seconds']:.2f}s)", flush=True)
    all_items = []
    for path in sorted(item_root.glob("*.json")):
        all_items.append(json.loads(path.read_text()))
    write_jsonl(output_root / "reports" / "process_results.jsonl", all_items)
    summary: dict[str, int] = {}
    for result in results:
        key = "success" if result["status"] == "success" else result["failure_code"]
        summary[key] = summary.get(key, 0) + 1
    write_json(output_root / "reports" / "last_run.json", {
        "selected": len(rows),
        "counts": summary,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    print("summary:", json.dumps(summary, sort_keys=True))
    return results


def add_inventory_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--abc-root", type=Path, default=DEFAULT_ABC_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--hash-inputs", action="store_true",
                        help="hash every paired input during inventory (slow)")


def add_process_arguments(parser: argparse.ArgumentParser, *, include_output: bool = True,
                          include_seed: bool = True) -> None:
    if include_output:
        parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--voronoi-exe", type=Path, default=DEFAULT_VORONOI_EXE)
    parser.add_argument("--voronoi-backend", choices=("cpp", "scipy"), default="cpp")
    parser.add_argument("--allow-experimental-voronoi", action="store_true")
    parser.add_argument("--scipy-voronoi-points", type=int, default=4000)
    parser.add_argument("--scipy-min-face-points", type=int, default=12)
    parser.add_argument("--step-deflection", type=float, default=0.001)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", help="comma-separated eight-digit shape IDs")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--tau", type=float, default=0.015)
    parser.add_argument("--condition-points", type=int, default=32768)
    if include_seed:
        parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--distance-backend", choices=("auto", "igl", "open3d", "naive"),
                        default="auto")
    parser.add_argument("--query-batch-size", type=int, default=65536)
    parser.add_argument("--alignment-samples", type=int, default=2048)
    parser.add_argument("--max-bbox-center-error", type=float, default=0.02)
    parser.add_argument("--max-bbox-extent-relative", type=float, default=0.05)
    parser.add_argument("--lipschitz-tolerance", type=float, default=2e-5)
    parser.add_argument("--max-faces", type=int, default=0)
    parser.add_argument("--allow-open-mesh", action="store_true")
    parser.add_argument("--skip-step-precheck", action="store_true",
                        help="testing only; do not use to publish data")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="archive and replace an incomplete/old per-shape output")


def validate_args(args: argparse.Namespace) -> None:
    if hasattr(args, "train_ratio"):
        if not 0 < args.train_ratio < 1 or not 0 <= args.val_ratio < 1:
            raise SystemExit("invalid split ratios")
        if args.train_ratio + args.val_ratio >= 1:
            raise SystemExit("train_ratio + val_ratio must be < 1")
    if hasattr(args, "tau") and args.tau <= 0:
        raise SystemExit("--tau must be positive")
    for name in ("condition_points", "query_batch_size", "alignment_samples"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    for name in ("scipy_voronoi_points", "scipy_min_face_points"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory", help="scan and pair ABC STEP/OBJ files")
    add_inventory_arguments(inventory)
    process = commands.add_parser("process", help="process paired rows from a manifest")
    add_process_arguments(process)
    run = commands.add_parser("run", help="build inventory, then process selected pairs")
    add_inventory_arguments(run)
    add_process_arguments(run, include_output=False, include_seed=False)
    args = parser.parse_args()
    validate_args(args)
    if args.command == "inventory":
        build_inventory(args)
    elif args.command == "process":
        process_manifest(args)
    else:
        build_inventory(args)
        args.manifest = Path(args.output_root) / "manifest.jsonl"
        process_manifest(args)


if __name__ == "__main__":
    main()
