"""Paired surface-point distance benchmark; never writes a production directory.

--baseline is an optional frozen previous distance.py, loaded only for timing.
The production implementation does not contain a legacy backend.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import numpy as np
from veudf.preprocess.cad import read_normalized, collect_targets, triangulate
from veudf.preprocess.condition import weld
from veudf.preprocess.config import Config
from veudf.preprocess.distance import EdgeOracle
from veudf.preprocess.pipeline import code_fingerprint, runtime_identity
from veudf.geometry.sampling import base_surface
from veudf.utils import file_hash, array_hash


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = Config().validate()
    manifest = json.loads(args.manifest.read_text())
    rows = [r for r in manifest if r["id"] in args.ids.split(",")]
    if len(rows) != len(args.ids.split(",")):
        raise ValueError("Missing benchmark sample")
    previous = None
    if args.baseline:
        frozen = args.output / "baseline_distance.py"
        shutil.copy2(args.baseline, frozen)
        spec = importlib.util.spec_from_file_location(
            "surface_distance_baseline", frozen
        )
        previous = importlib.util.module_from_spec(spec)
        import sys

        sys.modules[spec.name] = previous
        spec.loader.exec_module(previous)
    snapshot = args.output / "source_snapshot"
    package = Path(__file__).resolve().parents[1] / "src/veudf"
    shutil.copytree(
        package, snapshot / "veudf", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    identity = dict(
        code=code_fingerprint(),
        runtime=runtime_identity(),
        config=cfg.to_dict(),
        manifest_sha256=file_hash(args.manifest),
        count=args.count,
        audit_count=args.audit_count,
        baseline_sha256=file_hash(args.baseline) if args.baseline else None,
        note="Same surface points; speed excludes sampling-count reductions. Cold query includes lazy candidate index.",
    )
    (args.output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    results = []
    for row in rows:
        step = Path(row["step"])
        if file_hash(step) != row["source_sha256"]:
            raise ValueError("Source changed")
        shape, _ = read_normalized(step, cfg)
        targets = collect_targets(shape, cfg)
        raw = triangulate(targets, cfg)
        mesh, _ = weld(raw, cfg)
        q, _, ids = base_surface(mesh, args.count, return_ids=True)
        oracle = EdgeOracle(
            targets.edges,
            cfg,
            mesh=mesh,
            edge_faces=[e["face_ids"] for e in targets.topology["edges"]],
            triangle_face_ids=raw[2],
        )
        start = time.perf_counter()
        new = oracle.query(q, triangle_ids=ids)
        cold = time.perf_counter() - start
        cold_engine = oracle.report()
        start = time.perf_counter()
        warm = oracle.query(q, triangle_ids=ids)
        warm_seconds = time.perf_counter() - start
        np.testing.assert_array_equal(new["distance"], warm["distance"])
        n = min(args.audit_count, len(q))
        ref = EdgeOracle(targets.edges, cfg, "cad_reference").query(q[:n])
        error = float(np.abs(new["distance"][:n] - ref["distance"]).max())
        assert error <= cfg.label_tolerance
        unique = ~new["tie"][:n] & ~ref["tie"]
        np.testing.assert_array_equal(
            new["nearest_id"][:n][unique], ref["nearest_id"][unique]
        )
        np.testing.assert_array_equal(new["tie"][:n], ref["tie"])
        result = dict(
            id=row["id"],
            source_sha256=row["source_sha256"],
            points_sha256=array_hash(q),
            points=len(q),
            edges=len(targets.edges),
            triangles=len(mesh[1]),
            cold_seconds=cold,
            cold_engine=cold_engine,
            warm_seconds=warm_seconds,
            reference_max_error=error,
            engine=oracle.report(),
        )
        if previous:
            old = previous.EdgeOracle(targets.edges, cfg)
            start = time.perf_counter()
            baseline = old.query(q)
            elapsed = time.perf_counter() - start
            delta = float(np.abs(new["distance"] - baseline["distance"]).max())
            assert delta <= cfg.label_tolerance
            valid = ~new["tie"] & ~baseline["tie"]
            np.testing.assert_array_equal(
                new["nearest_id"][valid], baseline["nearest_id"][valid]
            )
            result.update(
                baseline_seconds=elapsed,
                baseline_occ_calls=old.calls,
                baseline_max_error=delta,
                cold_speedup=elapsed / cold,
                warm_speedup=elapsed / warm_seconds,
            )
        results.append(result)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    lines = [
        "# Surface distance engine: paired benchmark",
        "",
        "Identical points and CAD targets; no reduction in point count. Cold timing includes the candidate cache.",
        "",
        "| ID | edges | points | previous BVH s | new cold s | new warm s | cold speedup | max distance delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} | {r['edges']} | {r['points']} | {r.get('baseline_seconds',0):.4f} | {r['cold_seconds']:.4f} | {r['warm_seconds']:.4f} | {r.get('cold_speedup',0):.2f} | {r.get('baseline_max_error',r['reference_max_error']):.3g} |"
        )
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--ids", default="00000168,00000602,00000688")
    p.add_argument("--count", type=int, default=8192)
    p.add_argument("--audit-count", type=int, default=256)
    a = p.parse_args()
    if min(a.count, a.audit_count) < 1:
        p.error("Counts must be positive")
    run(a)
