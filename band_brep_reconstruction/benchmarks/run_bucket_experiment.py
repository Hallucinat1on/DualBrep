"""Frozen face-count-stratified bucket experiment.

Selection: passed production assets from a veudf dataset, 50 per 10-face bucket.
Per sample: one common prefix through vertices, one direct-distance curve replay,
and five scalar-only region-repair arms.  This pilot driver runs one sample per
worker so CAD target parsing, condition loading and exact E/V oracles are shared.
"""
import argparse
import csv
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODULE = REPO / "band_brep_reconstruction"
DEFAULT_DATASET = Path("/opt/data/private/yihengxu/Datasets/abc_ve_udf_neighborhood_pools_first50000_20261106")
sys.path[:0] = [str(MODULE), str(REPO / "veudf/src"), str(Path(__file__).parent)]

from run_mesh_input_experiment import ExactUDF, offline_source, read, save, digest  # noqa: E402
from experiment_scoring import score, geometry_score, fidelity_score  # noqa: E402

BUCKETS = [(2, 10)] + [(10 * i + 1, 10 * i + 10) for i in range(1, 10)]
HISTORICAL_IDS = {
    "00000168", "00000602", "00000688", "00002331", "00002720",
    "00002748", "00002978", "00003114", "00003261", "00003291",
    "00004610", "00004627", "00005094", "00007970", "00008133",
    "00008222", "00008280", "00008804", "00009340", "00009843",
}
ARMS = {
    "baseline": dict(region_repair="off", face_mean_filter=True),
    "no_mean": dict(region_repair="off", face_mean_filter=False),
    "small": dict(region_repair="small", face_mean_filter=False),
    "single_face": dict(region_repair="single_face", face_mean_filter=False),
    "joint": dict(region_repair="joint", face_mean_filter=False),
}
BASE_OVERRIDES = dict(
    query_batch_size=65536,
    refinement_coarse_edge=.03,
    refinement_target_edge=.003,
    refinement_rho_edge=.02,
    refinement_rho_vertex=.03,
    face_threshold1=.003,
    face_second_pass=False,
    face_filter_size=10,
    face_avg_udf_threshold=.003,
    band_threshold_inclusive=False,
    feature_sdf_tolerance=.004,
    curve_candidates="arcs",
    curve_recovery="direct_distance",
    save_queries=False,
    stop_after="solid",
    sample_timeout_seconds=10800.,
)


def short_id(sample_id):
    parts = str(sample_id).split("_")
    for part in parts:
        if len(part) == 8 and part.isdigit():
            return part
    return str(sample_id).split(".")[0]


def bucket_of(faces):
    return (2, 10) if faces <= 10 else (((faces - 1) // 10) * 10 + 1, ((faces - 1) // 10) * 10 + 10)


def bucket_name(bucket):
    return f"{bucket[0]}-{bucket[1]}"


def parse_buckets(text):
    if not text:
        return BUCKETS
    result = []
    for part in text.split(","):
        lo, hi = part.strip().split("-")
        result.append((int(lo), int(hi)))
    return result


def scan_dataset(dataset):
    dataset = Path(dataset)
    outcomes = [json.loads(line) for line in (dataset / "data/outcomes.jsonl").read_text().splitlines() if line.strip()]
    manifest = {row["sample_id"]: row for row in
                (json.loads(line) for line in (dataset / "data/manifest.jsonl").read_text().splitlines() if line.strip())}
    assets = dataset / "data/assets"
    rows = []
    for row in outcomes:
        if row.get("status") != "passed" or not row.get("counts"):
            continue
        asset_id = str(row.get("asset_id") or "")
        if not asset_id:
            continue
        folder = assets / asset_id
        if not folder.is_dir():
            continue
        condition = folder / "condition"
        if not all((condition / name).is_file() for name in ("vertices.npy", "triangles.npy", "meta.json")):
            continue
        counts = row["counts"]
        faces = int(counts["faces"])
        if not 2 <= faces <= 100:
            continue
        sid = short_id(row.get("sample_id", asset_id))
        if sid in HISTORICAL_IDS:
            continue
        source = manifest.get(row.get("sample_id"), {})
        step_path = row.get("step_path") or source.get("step_path") or row.get("source_path")
        if not step_path:
            continue
        rows.append(dict(
            id=sid, asset_id=asset_id, sample_id=row.get("sample_id"),
            family_id=row.get("family_id"), split=row.get("split"),
            face_count=faces, edge_count=int(counts["edges"]), vertex_count=int(counts["vertices"]),
            bucket=bucket_name(bucket_of(faces)),
            step_path=step_path,
            source_sha256=row.get("source_sha256") or source.get("source_sha256"),
            condition_dir=str(condition),
        ))
    return rows


def select_rows(rows, per_bucket=50, buckets=None, seed=0, allow_short=False):
    buckets = buckets or BUCKETS
    grouped = {b: [] for b in buckets}
    for row in rows:
        for b in buckets:
            if b[0] <= row["face_count"] <= b[1]:
                grouped[b].append(row)
                break
    selected = []
    counts = {}
    rng = np.random.default_rng(seed)
    for index, b in enumerate(buckets):
        ids = sorted(grouped[b], key=lambda r: r["asset_id"])
        counts[bucket_name(b)] = len(ids)
        need = min(per_bucket, len(ids))
        if len(ids) < per_bucket and not allow_short:
            raise RuntimeError(
                f"bucket {bucket_name(b)} has {len(ids)} assets, need {per_bucket}; "
                "resume preprocessing first or pass --allow-short for a pilot"
            )
        if need:
            order = rng.permutation(len(ids))[:need]
            selected.extend(ids[int(i)] for i in order)
    selected.sort(key=lambda r: (r["face_count"], r["id"]))
    if len({r["id"] for r in selected}) != len(selected):
        raise RuntimeError("duplicate short IDs in selection")
    return selected, counts


def configs():
    from bandbrep.config import load_config
    prefix = load_config(overrides={**BASE_OVERRIDES, "stop_after": "vertices", "curve_recovery": "projected"})
    curve = load_config(overrides={**BASE_OVERRIDES, "stop_after": "solid", "curve_recovery": "direct_distance"})
    repair = {name: load_config(overrides={**BASE_OVERRIDES, **options, "stop_after": "solid"})
              for name, options in ARMS.items()}
    return dict(prefix=prefix.to_dict(), curve=curve.to_dict(), repair={k: v.to_dict() for k, v in repair.items()})


def initialize(dataset, out, per_bucket=50, buckets=None, seed=0, jobs=2, allow_short=False, timeout=21600.):
    out = Path(out)
    if out.exists():
        raise RuntimeError("Use a new output directory")
    if not 1 <= jobs <= os.cpu_count():
        raise ValueError("invalid jobs")
    rows, counts = select_rows(scan_dataset(dataset), per_bucket, buckets, seed, allow_short)
    out.mkdir(parents=True)
    save(out / "manifest.json", rows)
    cfg = configs()
    policy = dict(
        dataset=str(Path(dataset).resolve()), seed=seed, per_bucket=per_bucket,
        buckets=[bucket_name(b) for b in (buckets or BUCKETS)], selected=len(rows),
        bucket_counts={k: v for k, v in counts.items()},
        jobs=jobs, sample_timeout_seconds=timeout,
        configs=cfg, arms=list(ARMS),
        scope="common prefix through vertices; one direct-distance curve replay; five region-repair arms",
        historical_excluded=sorted(HISTORICAL_IDS),
    )
    save(out / "campaign.json", policy)
    snapshot = out / "source_snapshot"
    code = {}
    folders = (MODULE / "bandbrep", MODULE / "benchmarks", REPO / "veudf/src/veudf")
    for folder in folders:
        for source in sorted(folder.rglob("*.py")):
            relative = source.relative_to(REPO)
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            code[str(relative)] = digest(target)
    common = {str(out / "manifest.json"): digest(out / "manifest.json"),
              str(out / "campaign.json"): digest(out / "campaign.json"),
              str(Path(dataset) / "data/outcomes.jsonl"): digest(Path(dataset) / "data/outcomes.jsonl"),
              str(Path(dataset) / "data/identity.json"): digest(Path(dataset) / "data/identity.json")}
    cases = {}
    for row in rows:
        files = {}
        for name in ("vertices.npy", "triangles.npy", "meta.json"):
            p = Path(row["condition_dir"]) / name
            files[str(p)] = digest(p)
        cases[row["id"]] = files
    save(out / "identity.json", dict(code=code, common=common, cases=cases))
    print(json.dumps(dict(selected=len(rows), bucket_counts=counts, output=str(out)), ensure_ascii=False, indent=2))


def verify(out, sid=None):
    identity = read(Path(out) / "identity.json")
    for relative, expected in identity["code"].items():
        if digest(Path(out) / "source_snapshot" / relative) != expected:
            raise RuntimeError("Frozen code changed: " + relative)
    for name, expected in identity["common"].items():
        if digest(name) != expected:
            raise RuntimeError("Frozen input changed: " + name)
    if sid is not None:
        for name, expected in identity["cases"][sid].items():
            if digest(name) != expected:
                raise RuntimeError("Frozen case input changed: " + name)


def prepare_asset(row):
    from bandbrep.surface_mesh import load_mesh
    from bandbrep.field_provider import CallableUDFProvider
    from veudf.preprocess.config import Config as DistanceConfig
    from veudf.preprocess.cad import read_normalized, collect_targets, triangulate
    from veudf.preprocess.condition import weld

    dc = DistanceConfig().validate()
    condition = Path(row["condition_dir"])
    mesh = load_mesh(condition)
    meta = json.loads((condition / "meta.json").read_text())
    shape, norm = read_normalized(row["step_path"], dc)
    old = meta.get("normalization", {}).get("world_to_normalized")
    if old is not None:
        np.testing.assert_allclose(norm["world_to_normalized"], old, rtol=0, atol=1e-10)
    targets = collect_targets(shape, dc)
    counts = dict(faces=len(targets.faces), edges=len(targets.edges), vertices=len(targets.vertices))
    expected = dict(faces=row["face_count"], edges=row["edge_count"], vertices=row["vertex_count"])
    if counts != expected:
        raise RuntimeError(f"target counts differ: {counts} != {expected}")
    raw = triangulate(targets, dc)
    _, oracle = weld(raw, dc)
    source = offline_source(targets, raw, oracle, norm)
    udf = ExactUDF(targets, dc)
    presence = ("present" if targets.edges else "absent", "present" if len(targets.vertices) else "absent")
    provider = CallableUDFProvider(
        udf.values, fingerprint="production-cad:" + str(row["source_sha256"]),
        presence=presence, band_width=.02,
    )
    return dict(mesh=mesh, provider=provider, udf=udf, targets=targets, raw=raw,
                oracle=oracle, source=source, norm=norm, meta=meta)


def load_bands(folder):
    from bandbrep.schemas import BandState
    with np.load(Path(folder) / "bands.npz") as a:
        return BandState(
            *(a[k] for k in ("points", "values", "edge_band", "vertex_band", "edge_region", "core_mask")),
            face_labels=a["face_labels"], exclusion_reasons=a["exclusion_reasons"],
            vertex_labels=a["vertex_labels"] if "vertex_labels" in a else None,
            vertex_centers=a["vertex_centers"] if "vertex_centers" in a else None,
            raw_face_labels=a["raw_face_labels"] if "raw_face_labels" in a else None,
            raw_exclusion_reasons=a["raw_exclusion_reasons"] if "raw_exclusion_reasons" in a else None,
            trusted_core_mask=a["trusted_core_mask"] if "trusted_core_mask" in a else None,
            edge_support_mask=a["edge_support_mask"] if "edge_support_mask" in a else None,
            absorbed_mask=a["absorbed_mask"] if "absorbed_mask" in a else None,
        )


def score_run(folder, source):
    from bandbrep.cache import load_topology
    folder = Path(folder)
    if not (folder / "incidence.json").is_file() or not (folder / "bands.npz").is_file():
        return None
    bands = load_bands(folder)
    value = score(load_topology(folder), bands, source)
    save(folder / "graph_score.json", value)
    return value


def prefix_run(prepared, row, out, cfg):
    from bandbrep.config import Config
    from bandbrep.reconstruct import reconstruct
    folder = Path(out) / "runs" / row["id"] / "prefix"
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(id=row["id"], stage="prefix", status="failed")
    try:
        run = reconstruct(prepared["mesh"], prepared["provider"], Config(**cfg), folder / "reconstruction")
        result["status"] = run.get("status", "vertices_ready")
        save(folder / "source_audit.json", dict(
            target_counts=dict(faces=len(prepared["targets"].faces), edges=len(prepared["targets"].edges),
                               vertices=len(prepared["targets"].vertices)),
            normalization=prepared["norm"], mesh=prepared["meta"].get("mesh_sha256")))
    except Exception as exc:
        result["error"] = dict(code=getattr(exc, "code", type(exc).__name__), message=str(exc),
                               details=getattr(exc, "details", {}))
        result["traceback"] = traceback.format_exc()
    result["seconds"] = time.monotonic() - started
    save(folder / "result.json", result)
    return result


def curve_run(prepared, row, out, prefix, cfg):
    from bandbrep.config import Config
    from bandbrep.field_provider import MeshFieldProvider
    from bandbrep.schemas import SurfaceMesh, write_json, jsonable
    from bandbrep.cache import load_topology
    from bandbrep.curve_distance_fitting import recover_distance_curves
    from bandbrep.geometry_refinement import validate_structure_coverage
    from bandbrep.postprocess import orient_boundaries
    from bandbrep.surface_fitting import initialize_face_surfaces
    from bandbrep.occ_assembly import assemble_brep, verify_export_and_roundtrip

    folder = Path(out) / "runs" / row["id"] / "curve"
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(id=row["id"], stage="curve", status="failed", timings={})
    try:
        prefix_dir = Path(prefix) / "reconstruction"
        with np.load(prefix_dir / "working_mesh.npz") as a:
            working = SurfaceMesh(a["vertices"], a["triangles"])
        from bandbrep.schemas import VertexNode
        vertices = [VertexNode(**{**v, "position": np.asarray(v["position"])})
                    for v in json.loads((prefix_dir / "vertices.json").read_text())]
        bands = load_bands(prefix_dir)
        topology = load_topology(prefix_dir)
        cfg_obj = Config(**cfg)
        provider = MeshFieldProvider(prepared["mesh"], prepared["provider"], cfg_obj)
        tick = time.monotonic()
        graph, curves, report = recover_distance_curves(topology, bands, vertices, provider, cfg_obj,
            progress=lambda data: write_json(folder / "curve_recovery_progress.json", data))
        result["timings"]["curve_recovery"] = time.monotonic() - tick
        write_json(folder / "logical_graph.json", graph)
        write_json(folder / "curves.json", curves)
        np.savez_compressed(folder / "refined_geometry.npz",
                            **{f"edge_{e.id}": e.samples for e in graph.edges},
                            vertices=np.array([v.position for v in vertices]).reshape(-1, 3))
        tick = time.monotonic()
        result["coverage"] = validate_structure_coverage(working, bands, curves, provider, cfg_obj)
        result["timings"]["global_coverage"] = time.monotonic() - tick
        result["curve_kinds"] = {str(i): c.kind for i, c in curves.items()}
        orient_boundaries(working, topology, graph, curves)
        write_json(folder / "cad_graph.json", graph)
        patches = initialize_face_surfaces(working, topology.face_labels, graph, curves,
                                           provider, cfg_obj,
                                           trusted=getattr(bands, "trusted_core_mask", None))
        write_json(folder / "face_patches.json", patches)
        solid, mapping = assemble_brep(graph, curves, patches, cfg_obj, folder)
        cad = verify_export_and_roundtrip(solid, graph, curves, patches, mapping, cfg_obj, folder)
        result["cad"] = cad
        result["score"] = score_run(prefix_dir, prepared["source"])
        if result["score"] is not None:
            save(folder / "graph_score.json", result["score"])
        result["geometry"] = geometry_score(graph.vertices, curves, prepared["source"])
        result["fidelity"] = fidelity_score(solid, prepared["source"])
        result["status"] = "solid_validated"
        result["joint_success"] = bool(result["status"] == "solid_validated" and (result.get("score") or {}).get("graph_good"))
    except Exception as exc:
        result["error"] = dict(code=getattr(exc, "code", type(exc).__name__), message=str(exc),
                               details=getattr(exc, "details", {}))
        result["traceback"] = traceback.format_exc()
    result["seconds"] = time.monotonic() - started
    save(folder / "result.json", result)
    save(folder / "run.json", {**result, "config": cfg})
    return result


def repair_run(prepared, row, out, prefix, cfg, arm):
    from bandbrep.config import Config
    from bandbrep.reconstruct import reconstruct_from_prepared
    from bandbrep.cache import load_center_fields
    from bandbrep.schemas import SurfaceMesh, write_json

    folder = Path(out) / "runs" / row["id"] / "repair" / arm
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(id=row["id"], stage="repair", arm=arm, status="failed", timings={})
    prefix_dir = Path(prefix) / "reconstruction"
    provenance = dict(prefix=str(prefix_dir), artifacts={
        name: digest(prefix_dir / name) for name in ("input_mesh.npz", "working_mesh.npz", "center_fields.npz")
    })
    try:
        with np.load(prefix_dir / "input_mesh.npz") as a:
            input_mesh = SurfaceMesh(a["vertices"], a["triangles"])
        input_mesh.report["normalization"] = prepared["meta"].get("normalization", {})
        with np.load(prefix_dir / "working_mesh.npz") as a:
            working = SurfaceMesh(a["vertices"], a["triangles"])
        centers = load_center_fields(prefix_dir / "center_fields.npz")
        tick = time.monotonic()
        reconstruction = reconstruct_from_prepared(
            input_mesh, working, centers, prepared["provider"], Config(**cfg),
            folder / "reconstruction", provenance=provenance)
        result["timings"]["reconstruction"] = time.monotonic() - tick
        run = read(folder / "reconstruction/run.json")
        result["status"] = run.get("status", "solid_validated")
        result["repair_attempts"] = run.get("repair_attempts", [])
        result["effective_mode"] = run.get("effective_config", {}).get("region_repair")
        result["score"] = score_run(folder / "reconstruction", prepared["source"])
        if reconstruction is not None and "graph" in reconstruction:
            result["geometry"] = geometry_score(reconstruction["graph"].vertices, reconstruction["curves"], prepared["source"])
            result["fidelity"] = fidelity_score(reconstruction["solid"], prepared["source"])
            result["joint_success"] = bool(result["status"] == "solid_validated" and (result.get("score") or {}).get("graph_good"))
    except Exception as exc:
        result["error"] = dict(code=getattr(exc, "code", type(exc).__name__), message=str(exc),
                               details=getattr(exc, "details", {}))
        result["traceback"] = traceback.format_exc()
        runpath = folder / "reconstruction/run.json"
        if runpath.exists():
            run = read(runpath)
            result["status"] = run.get("status", "failed")
            result["repair_attempts"] = run.get("repair_attempts", [])
            result["effective_mode"] = run.get("effective_config", {}).get("region_repair")
            result["failed_stage"] = run.get("failed_stage")
    result["seconds"] = time.monotonic() - started
    save(folder / "result.json", result)
    return result


def sample_worker(out, sid):
    from bandbrep.deadline import sample_budget
    out = Path(out)
    policy = read(out / "campaign.json")
    verify(out, sid)
    row = next(r for r in read(out / "manifest.json") if r["id"] == sid)
    record_dir = out / "records" / sid
    record_dir.mkdir(parents=True, exist_ok=True)
    result = dict(id=sid, bucket=row["bucket"], face_count=row["face_count"],
                  edge_count=row["edge_count"], vertex_count=row["vertex_count"], status="failed")
    started = time.monotonic()
    with sample_budget(policy["sample_timeout_seconds"]):
        try:
            prepared = prepare_asset(row)
            prefix_base = out / "runs" / sid / "prefix"
            if (prefix_base / "reconstruction/run.json").exists():
                prefix_run_data = read(prefix_base / "result.json") if (prefix_base / "result.json").exists() else {}
                prefix_status = read(prefix_base / "reconstruction/run.json").get("status")
                result["prefix"] = dict(prefix_run_data, status=prefix_status, reused=True)
            else:
                result["prefix"] = prefix_run(prepared, row, out, policy["configs"]["prefix"])
            if result["prefix"].get("status") not in ("vertices_ready", "solid_validated"):
                result["status"] = "prefix_failed"
            else:
                result["curve"] = curve_run(prepared, row, out, out / "runs" / sid / "prefix", policy["configs"]["curve"])
                result["repair"] = {}
                for arm in policy["arms"]:
                    result["repair"][arm] = repair_run(
                        prepared, row, out, out / "runs" / sid / "prefix", policy["configs"]["repair"][arm], arm)
                result["status"] = "completed"
        except Exception as exc:
            result["error"] = dict(code=getattr(exc, "code", type(exc).__name__), message=str(exc),
                                   details=getattr(exc, "details", {}))
            result["traceback"] = traceback.format_exc()
    result["seconds"] = time.monotonic() - started
    save(record_dir / "result.json", result)
    print(sid, result["status"], flush=True)
    return result


def launch(out, row):
    out = Path(out)
    sid = row["id"]
    record = out / "records" / sid / "result.json"
    if record.exists():
        return read(record)
    folder = out / "runs" / sid
    if folder.exists() and not (folder / "prefix").exists():
        raise RuntimeError("unrecorded partial sample: " + str(folder))
    snapshot = out / "source_snapshot"
    script = snapshot / "band_brep_reconstruction/benchmarks/run_bucket_experiment.py"
    env = {**os.environ, "PYTHONPATH": str(snapshot / "veudf/src") + ":" + str(snapshot / "band_brep_reconstruction"),
           **{k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
    logdir = out / "logs"; logdir.mkdir(exist_ok=True)
    timeout = read(out / "campaign.json")["sample_timeout_seconds"]
    start = time.monotonic(); reason = None
    with (logdir / (sid + ".log")).open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, str(script), "--worker", "--id", sid, "--output", str(out)],
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while proc.poll() is None:
            if time.monotonic() - start > timeout:
                reason = "sample_timeout"
                try: os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                break
            save(out / "active" / (sid + ".json"), dict(id=sid, pid=proc.pid,
                 elapsed_seconds=time.monotonic() - start, timeout_seconds=timeout))
            time.sleep(1)
    if record.exists():
        result = read(record)
    else:
        result = dict(id=sid, bucket=row["bucket"], status="failed",
                      error=dict(code=reason or "worker_exit", message=str(proc.returncode)))
    result["process"] = dict(returncode=proc.returncode, seconds=time.monotonic() - start, timeout=reason)
    save(record, result)
    (out / "active" / (sid + ".json")).unlink(missing_ok=True)
    return result


def run_batch(out, jobs=None):
    out = Path(out)
    verify(out)
    policy = read(out / "campaign.json")
    rows = read(out / "manifest.json")
    jobs = int(jobs or policy["jobs"])
    tasks = [r for r in rows if not (out / "records" / r["id"] / "result.json").exists()]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(launch, out, row) for row in tasks]
        for future in as_completed(futures):
            future.result()


def report(out):
    out = Path(out)
    rows = read(out / "manifest.json")
    records = {r["id"]: read(out / "records" / r["id"] / "result.json")
               for r in rows if (out / "records" / r["id"] / "result.json").exists()}
    summary = dict(total=len(rows), completed=len(records), buckets={}, arms={}, curve={})
    for row in rows:
        rec = records.get(row["id"])
        b = row["bucket"]
        summary["buckets"].setdefault(b, dict(total=0, completed=0))
        summary["buckets"][b]["total"] += 1
        if rec:
            summary["buckets"][b]["completed"] += 1
    for arm in ARMS:
        good = 0; total = 0; times = []
        for rec in records.values():
            item = (rec.get("repair") or {}).get(arm)
            if not item: continue
            total += 1; times.append(item.get("seconds", 0))
            good += bool(item.get("status") == "solid_validated" and (item.get("score") or {}).get("graph_good"))
        summary["arms"][arm] = dict(completed=total, joint_success=good,
                                    median_seconds=statistics.median(times) if times else None)
    good = 0; total = 0; times = []
    for rec in records.values():
        item = rec.get("curve")
        if not item: continue
        total += 1; times.append(item.get("seconds", 0))
        good += bool(item.get("status") == "solid_validated" and (item.get("score") or {}).get("graph_good"))
    summary["curve"] = dict(completed=total, joint_success=good,
                            median_seconds=statistics.median(times) if times else None)
    save(out / "summary.json", summary)
    lines = ["# 分桶实验", "", f"selected {len(rows)}; completed {len(records)}", "",
             "## 分桶", "", "| bucket | total | completed |", "|---|---:|---:|"]
    for b, item in sorted(summary["buckets"].items()):
        lines.append(f"| {b} | {item['total']} | {item['completed']} |")
    lines += ["", "## 曲线回放", "", f"{summary['curve']}", "", "## 五组修复", "",
              "| arm | completed | joint? | median seconds |", "|---|---:|---:|---:|"]
    for arm, item in summary["arms"].items():
        lines.append(f"| {arm} | {item['completed']} | {item['joint_success']} | {item['median_seconds']} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--initialize", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--report", action="store_true")
    modes.add_argument("--worker", action="store_true")
    p.add_argument("--id")
    p.add_argument("--per-bucket", type=int, default=50)
    p.add_argument("--buckets")
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--allow-short", action="store_true")
    p.add_argument("--timeout", type=float, default=21600.)
    a = p.parse_args()
    if a.initialize:
        initialize(a.dataset, a.output, a.per_bucket, parse_buckets(a.buckets), a.seed, a.jobs,
                   a.allow_short, a.timeout)
    elif a.run:
        run_batch(a.output, a.jobs)
    elif a.report:
        print(json.dumps(report(a.output), ensure_ascii=False, indent=2))
    else:
        if not a.id: p.error("--worker needs --id")
        sample_worker(a.output, a.id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
