"""Per-shape processing orchestration and batch manifest execution.

``process_one`` chains precheck -> voronoi backend -> alignment -> field
queries -> atomic export; ``process_manifest`` iterates a manifest and writes
the ``reports/`` summaries.  All per-shape failures are captured as
``PipelineError`` records instead of aborting the batch.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from .common import (SCHEMA_VERSION, PipelineError, sha256_file, write_json,
                     write_jsonl)
from .geometry import (alignment_metrics, mesh_hash, parse_transform,
                       sample_surface, transform_mesh, unsigned_distance)
from .inventory import read_manifest
from .precheck import load_triangle_mesh, precheck_obj, precheck_step
from .voronoi_cpp import run_voronoi
from .voronoi_scipy import run_scipy_voronoi


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
            with np.load(sample_dir / "surface_field.npz") as archive:
                if "udf_vertex_raw" in archive.files:  # v2 schema; v1 samples are regenerated
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
        # Face centroids and mesh vertices are queried in a single batch so both
        # share the same distance backend; the result is split back apart below.
        combined, actual_backend = unsigned_distance(
            np.concatenate((centers, vertices), axis=0), voronoi,
            args.distance_backend, args.query_batch_size)
        udf_raw = combined[:len(centers)]
        udf_vertex_raw = combined[len(centers):].astype(np.float32)
        if not np.isfinite(combined).all() or np.any(combined < 0):
            raise PipelineError("distance_nonfinite", "distance result is negative or non-finite")
        udf_metric = np.minimum(udf_raw, args.tau).astype(np.float32)
        udf_target = (udf_metric / args.tau).astype(np.float32)
        udf_vertex_metric = np.minimum(udf_vertex_raw, args.tau).astype(np.float32)
        udf_vertex_target = (udf_vertex_metric / args.tau).astype(np.float32)

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
                udf_vertex_raw=udf_vertex_raw,
                udf_vertex_metric=udf_vertex_metric,
                udf_vertex_target=udf_vertex_target,
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
                "mesh_hash": mesh_hash(vertices, faces),
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
                    "vertex_minimum": float(udf_vertex_raw.min()),
                    "vertex_maximum": float(udf_vertex_raw.max()),
                    "vertex_mean": float(udf_vertex_raw.mean()),
                    "vertex_saturated_fraction": float(np.mean(udf_vertex_raw >= args.tau)),
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
