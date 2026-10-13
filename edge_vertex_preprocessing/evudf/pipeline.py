import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import platform

import numpy as np
import OCC
import scipy
from scipy.spatial import cKDTree

from . import CHANNEL_NAMES, SCHEMA_VERSION, __version__
from .cad import PreprocessError, collect_targets, read_normalized, triangulate
from .distance import EdgeOracle, SDFOracle, VertexOracle
from .sampling import QUERY_SOURCES, sample_all


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_versions(cfg):
    result = dict(occ=OCC.VERSION, numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version())
    if cfg.sdf_backend.startswith("mesh"):
        try:
            result["libigl"] = version("libigl")
        except PackageNotFoundError:
            result["libigl"] = "not-installed"
    return result


def cache_key(source_hash, cfg, source_record=None):
    record = dict(source_sha256=source_hash, config=cfg.to_dict(),
                  code_sha256=code_fingerprint(), version=__version__,
                  runtime=runtime_versions(cfg), schema=SCHEMA_VERSION,
                  manifest_identity={k: source_record[k] for k in ("sample_id", "group_id", "split")
                                     if source_record and k in source_record})
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def verify_saved(path, expected_key=None):
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    quality = json.loads((path / "quality.json").read_text())
    if meta.get("schema_version") != SCHEMA_VERSION or meta.get("channel_names") != list(CHANNEL_NAMES):
        raise PreprocessError("quality_failed", "Saved channel/schema contract does not match")
    if expected_key is not None and meta.get("cache_key") != expected_key:
        raise PreprocessError("cache_mismatch", "Saved output belongs to different inputs/config/code")
    if quality.get("status") != "passed":
        raise PreprocessError("quality_failed", "Saved sample has not passed quality checks")
    for name, checksum in meta["artifact_sha256"].items():
        if sha256_file(path / name) != checksum:
            raise PreprocessError("quality_failed", f"Artifact hash mismatch: {name}")
    with np.load(path / "fields.npz", allow_pickle=False) as data:
        q, values = data["query_points"], data["query_fields"]
        if q.ndim != 2 or q.shape[1] != 3 or values.shape != q.shape:
            raise PreprocessError("quality_failed", "Saved field shape mismatch")
        if q.dtype != np.float32 or values.dtype != np.float32:
            raise PreprocessError("quality_failed", "Saved fields must be float32")
        if not np.isfinite(q).all() or not np.isfinite(values).all():
            raise PreprocessError("quality_failed", "Saved fields contain nonfinite values")
        if data["vertex_points"].shape != (meta["target_counts"]["vertices"], 3):
            raise PreprocessError("quality_failed", "Vertex count disagrees with metadata")
        if bool(data["has_vertices"]) != meta["has_vertices"] or bool(data["has_edges"]) != meta["has_edges"]:
            raise PreprocessError("quality_failed", "Empty-set flags disagree with metadata")
    return meta


def stratified_indices(sources, count, rng):
    count = min(count, len(sources))
    groups = [np.flatnonzero(sources == source) for source in np.unique(sources)]
    quota = max(1, count // len(groups))
    selected = np.concatenate([rng.choice(group, min(quota, len(group)), replace=False) for group in groups])
    if len(selected) > count:
        return rng.choice(selected, count, replace=False)
    remaining = np.setdiff1d(np.arange(len(sources)), selected)
    return np.r_[selected, rng.choice(remaining, count - len(selected), replace=False)]


def validate_geometry(targets, cfg):
    if len(targets.vertices) > 1:
        pairs = cKDTree(targets.vertices).query_pairs(cfg.geometry_tolerance)
        if pairs:
            raise PreprocessError("unsupported_overlap", f"Distinct target vertices within geometry tolerance: {sorted(pairs)[:5]}")


def validate_fields(solid, targets, mesh, arrays, cfg, rng):
    q = arrays["query_points"].astype(np.float64)
    values = arrays["query_fields"].astype(np.float64)
    if not np.isfinite(values).all() or np.min(values[:, 1:]) < 0:
        raise PreprocessError("quality_failed", "Nonfinite or negative UDF")
    geometry_budget = cfg.label_tolerance + cfg.geometry_tolerance
    sdf_budget = geometry_budget + (2 * cfg.mesh_deflection if cfg.sdf_backend.startswith("mesh") else 0)
    if targets.edges:
        violation = float(np.max(np.abs(values[:, 0]) - values[:, 1]))
        if violation > sdf_budget:
            raise PreprocessError("quality_failed", f"|SDF| > dE by {violation:g}")
    else:
        violation = 0.0
    if len(targets.vertices):
        if not targets.edges or np.max(values[:, 1] - values[:, 2]) > geometry_budget:
            raise PreprocessError("quality_failed", "dE <= dV / target containment check failed")
    src = arrays["query_source"]
    for source, channel in ((QUERY_SOURCES["edge_anchor"], 1), (QUERY_SOURCES["vertex_anchor"], 2)):
        mask = src == source
        if mask.any() and np.max(values[mask, channel]) > geometry_budget:
            raise PreprocessError("quality_failed", "Target zero-set check failed")
    if (src == QUERY_SOURCES["vertex_anchor"]).any():
        if np.max(values[src == QUERY_SOURCES["vertex_anchor"], 1]) > geometry_budget:
            raise PreprocessError("quality_failed", "Necessary vertex does not lie on target edges")
    selected = stratified_indices(src, cfg.reference_checks, rng)
    edge_reference = EdgeOracle(targets.edges, cfg, "cad_reference").query(q[selected])
    ref_edge = edge_reference["distance"]
    ref_vertex = VertexOracle(targets.vertices, cfg).query(q[selected])["distance"]
    ref_sdf = SDFOracle(solid, targets.faces, mesh, cfg, "cad").query(q[selected])
    errors = np.abs(values[selected] - np.column_stack((ref_sdf, ref_edge, ref_vertex)))
    limits = np.array([sdf_budget, geometry_budget, geometry_budget])
    if np.any(errors.max(axis=0) > limits):
        raise PreprocessError("quality_failed", f"Reference discrepancy: {errors.max(axis=0).tolist()}")
    unique = (~edge_reference["tie"] & ~arrays["nearest_tie_E"][selected]
              & (edge_reference["nearest_id"] >= 0))
    mismatches = int(np.sum(unique & (edge_reference["nearest_id"] != arrays["nearest_E_id"][selected])))
    if mismatches:
        raise PreprocessError("quality_failed", f"Unique nearest edge id disagrees with reference at {mismatches} queries")
    # Independent random-pair check, with empty-set channels excluded.
    pairs = rng.integers(len(q), size=(min(256, len(q)), 2))
    separation = np.linalg.norm(q[pairs[:, 0]] - q[pairs[:, 1]], axis=1)
    diff = np.abs(values[pairs[:, 0]] - values[pairs[:, 1]]) - separation[:, None]
    active = np.array([True, bool(targets.edges), bool(len(targets.vertices))])
    if np.any(np.max(diff[:, active], axis=0) > 2 * limits[active]):
        raise PreprocessError("quality_failed", "Distance Lipschitz check failed")
    return dict(status="passed", reference_query_count=len(selected),
                reference_max_error={name: float(errors[:, i].max()) if active[i] else None
                                     for i, name in enumerate(CHANNEL_NAMES)},
                reference_p99_error={name: float(np.percentile(errors[:, i], 99)) if active[i] else None
                                     for i, name in enumerate(CHANNEL_NAMES)},
                reference_unique_edge_id_mismatches=mismatches,
                surface_edge_containment_violation=max(0.0, violation),
                checks="topology/empty-set/zero-set/reference/containment/Lipschitz/finite/storage",
                sdf_mesh_error_certification="empirical reference subset; no global mesh error guarantee",
                intersection_scope="valid OCC solid; close distinct vertices rejected; global curve-overlap certification not implemented")


def build_sample(step_path, output, cfg, *, resume=False, overwrite=False, source_record=None):
    start = time.monotonic()
    cfg.validate()
    source = Path(step_path).resolve()
    output = Path(output).resolve()
    source_hash = sha256_file(source)
    if source_record and source_record.get("source_sha256") not in (None, source_hash):
        raise PreprocessError("source_hash_mismatch", "Manifest SHA256 disagrees with source file")
    key = cache_key(source_hash, cfg, source_record)
    if output.exists():
        if resume:
            verify_saved(output, key)
            return dict(status="resumed", output=str(output))
        if not overwrite:
            raise PreprocessError("output_exists", f"Output exists: {output}; use --resume or --overwrite")
        marker = output / "metadata.json"
        try:
            existing = json.loads(marker.read_text())
            if existing.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("unrecognized schema")
        except (OSError, ValueError) as exc:
            raise PreprocessError("output_exists", "--overwrite only replaces an identified evudf sample directory") from exc
    rng = np.random.default_rng(cfg.seed)
    timings = {}
    stage = time.monotonic()
    solid, transform = read_normalized(source, cfg)
    targets = collect_targets(solid, cfg)
    validate_geometry(targets, cfg)
    timings["cad_and_targets"] = time.monotonic() - stage
    stage = time.monotonic()
    mesh = triangulate(targets, cfg)
    arrays = sample_all(targets, mesh, cfg, rng)
    timings["mesh_and_sampling"] = time.monotonic() - stage
    stage = time.monotonic()
    edge = EdgeOracle(targets.edges, cfg)
    vertex = VertexOracle(targets.vertices, cfg)
    sdf = SDFOracle(solid, targets.faces, mesh, cfg)
    query = arrays["query_points"]
    values = np.empty(query.shape, dtype=np.float32)
    auxiliaries = {}
    for prefix in ("E", "V"):
        auxiliaries[f"nearest_{prefix}_id"] = np.empty(len(query), dtype=np.int32)
        auxiliaries[f"nearest_tie_{prefix}"] = np.empty(len(query), dtype=bool)
        auxiliaries[f"closest_{prefix}_points"] = np.empty(query.shape, dtype=np.float32)
        auxiliaries[f"gradient_{prefix}"] = np.empty(query.shape, dtype=np.float32)
        auxiliaries[f"gradient_valid_{prefix}"] = np.empty(len(query), dtype=bool)
    for head in range(0, len(query), cfg.batch_size):
        tail = min(head + cfg.batch_size, len(query))
        points = query[head:tail].astype(np.float64)
        er, vr = edge.query(points), vertex.query(points)
        values[head:tail] = np.column_stack((sdf.query(points), er["distance"], vr["distance"]))
        for prefix, result in (("E", er), ("V", vr)):
            for dest, src in ((f"nearest_{prefix}_id", "nearest_id"), (f"nearest_tie_{prefix}", "tie"),
                              (f"closest_{prefix}_points", "closest"), (f"gradient_{prefix}", "gradient"),
                              (f"gradient_valid_{prefix}", "gradient_valid")):
                auxiliaries[dest][head:tail] = result[src]
    arrays.update(auxiliaries)
    arrays["query_fields"] = values
    timings["field_queries"] = time.monotonic() - stage
    stage = time.monotonic()
    quality = validate_fields(solid, targets, mesh, arrays, cfg, rng)
    timings["validation"] = time.monotonic() - stage
    quality.update(timings_seconds=timings, edge_cad_query_calls=edge.calls,
                   total_seconds_before_write=time.monotonic() - start)
    metadata = dict(schema_version=SCHEMA_VERSION, channel_names=list(CHANNEL_NAMES),
                    cache_key=key, package_version=__version__, code_sha256=code_fingerprint(),
                    runtime_versions=runtime_versions(cfg), config=cfg.to_dict(),
                    source_path=str(source), source_sha256=source_hash, source_record=source_record,
                    normalization=transform, has_edges=bool(targets.edges),
                    has_vertices=bool(len(targets.vertices)),
                    clip_values=[cfg.clip_sdf, cfg.clip_edge, cfg.clip_vertex],
                    field_units="normalized Euclidean length; nonempty distances are not clipped",
                    empty_set_contract="empty channel filled with its clip value; nearest id=-1; geometry/gradient invalid",
                    query_sources=QUERY_SOURCES,
                    actual_query_counts={name: int(np.sum(arrays["query_source"] == idx)) for name, idx in QUERY_SOURCES.items()},
                    target_counts=dict(faces=len(targets.faces), edges=len(targets.edges), vertices=len(targets.vertices)))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    backup = None
    try:
        np.savez_compressed(temporary / "fields.npz", **arrays)
        write_json(temporary / "topology.json", targets.topology)
        write_json(temporary / "quality.json", quality)
        if cfg.save_mesh:
            np.savez_compressed(temporary / "surface_mesh.npz", vertices=mesh[0], triangles=mesh[1], face_ids=mesh[2])
        metadata["artifact_sha256"] = {p.name: sha256_file(p) for p in temporary.iterdir()}
        write_json(temporary / "metadata.json", metadata)
        verify_saved(temporary, key)
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output.name}.old-", dir=output.parent))
            backup.rmdir()
            os.replace(output, backup)
        os.replace(temporary, output)
        if backup is not None:
            shutil.rmtree(backup)
    except BaseException:
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return dict(status="passed", output=str(output), counts=metadata["target_counts"],
                query_count=len(query), seconds=time.monotonic() - start)
