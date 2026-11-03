"""One STEP -> atomic memory-mapped surface pools and independent evaluation points."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
from scipy.spatial import cKDTree
from veudf.schema import SCHEMA, CHANNELS, EMPTY_DISTANCE, GEOMETRY_VERSION, VALID, TIE
from veudf.utils import array_hash, file_hash, seed_for
from veudf.geometry.sampling import mesh_hash, attach_surface_coordinates
from veudf import __version__
from veudf.preprocess.cad import (
    PreprocessError,
    collect_targets,
    read_normalized,
    triangulate,
)
from veudf.preprocess.condition import weld
from veudf.preprocess.dedup import features
from veudf.preprocess.distance import EdgeOracle, VertexOracle
from veudf.preprocess.filtering import audit_shape, check_solid_faces
from veudf.preprocess.sampling import sample_pools

sha256_file = file_hash


def runtime_identity():
    import OCC
    from importlib.metadata import version

    return dict(
        pythonocc=OCC.VERSION,
        numpy=np.__version__,
        libigl=version("libigl"),
        scipy=version("scipy"),
        trimesh=version("trimesh"),
    )


def code_fingerprint():
    digest = hashlib.sha256(json.dumps(runtime_identity(), sort_keys=True).encode())
    package = Path(__file__).resolve().parents[1]
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode() + path.read_bytes())
    return digest.hexdigest()


def validate_geometry(targets, cfg):
    if len(targets.vertices) > 1 and cKDTree(targets.vertices).query_pairs(
        cfg.geometry_tolerance
    ):
        raise PreprocessError(
            "unsupported_overlap", "Distinct target vertices within tolerance"
        )


def quantiles(values):
    v = np.asarray(values)
    if not len(v):
        return None
    return dict(
        zip(
            ["p50", "p95", "p99", "max"],
            map(float, np.quantile(v, [0.5, 0.95, 0.99, 1])),
        )
    )


def label_pool(directory, pool, mesh, targets, cfg, edge, vertex):
    directory.mkdir(parents=True)
    surface_error = attach_surface_coordinates(pool, mesh)
    x = pool["points"]
    for key, value in pool.items():
        np.save(directory / f"{key}.npy", value, allow_pickle=False)
    n = len(x)
    values = np.lib.format.open_memmap(
        directory / "fields.npy", mode="w+", dtype=np.float32, shape=(n, 2)
    )
    nearest = np.lib.format.open_memmap(
        directory / "nearest_ids.npy", mode="w+", dtype=np.int32, shape=(n, 2)
    )
    flags = np.lib.format.open_memmap(
        directory / "flags.npy", mode="w+", dtype=np.uint8, shape=(n, 2)
    )
    present = np.array([bool(targets.edges), bool(len(targets.vertices))])
    for start in range(0, n, cfg.batch_size):
        stop = start + cfg.batch_size
        q = x[start:stop].astype(np.float64)
        hint = (
            pool["source_instance"][start:stop]
            if directory.name == "edge_surface"
            else None
        )
        e = edge.query(
            q, triangle_ids=pool["triangle_ids"][start:stop], source_ids=hint
        )
        v = vertex.query(q)
        values[start:stop] = np.column_stack((e["distance"], v["distance"]))
        nearest[start:stop] = np.column_stack((e["nearest_id"], v["nearest_id"]))
        flags[start:stop] = (
            present.astype(np.uint8) * VALID
            + np.column_stack((e["tie"], v["tie"])).astype(np.uint8) * TIE
        )
    if n and (not np.isfinite(values).all() or np.any(values < 0)):
        raise PreprocessError("quality_failed", "Invalid surface distance labels")
    for c, exists in enumerate(present):
        if not exists and (
            np.any(values[:, c] != np.float32(EMPTY_DISTANCE))
            or np.any(nearest[:, c] != -1)
        ):
            raise PreprocessError("quality_failed", "Invalid empty channel")
    if directory.name in ("edge_surface", "vertex_surface") and n:
        c = int(directory.name == "vertex_surface")
        if values[:, c].max() > cfg.feature_radii[-1] + cfg.label_tolerance:
            raise PreprocessError(
                "quality_failed", "Feature surface exceeds its distance bound"
            )
    if present[1] and np.any(values[:, 0] > values[:, 1] + cfg.label_tolerance):
        raise PreprocessError("quality_failed", "dE > dV")
    channel = int(directory.name == "vertex_surface")
    strata = np.searchsorted(cfg.feature_radii, values[:, channel], side="left").astype(
        np.uint8
    )
    np.save(directory / "strata.npy", strata[:, None], allow_pickle=False)
    values.flush()
    nearest.flush()
    flags.flush()
    rng = np.random.default_rng(seed_for(cfg.seed, "label-audit", directory.name))
    rows = rng.choice(n, min(n, cfg.reference_checks), replace=False)
    pool["_audit_xyz"], pool["_audit_values"] = x[rows], np.asarray(values[rows])
    pool["_audit_ids"], pool["_audit_flags"] = np.asarray(nearest[rows]), np.asarray(
        flags[rows]
    )
    coverage = {}
    for c, name, total in (
        (0, "edge", len(targets.edges)),
        (1, "vertex", len(targets.vertices)),
    ):
        ids = nearest[:, c]
        counts = np.bincount(ids[ids >= 0], minlength=total)
        coverage[name] = dict(
            counts=counts.tolist(), missing=np.flatnonzero(counts == 0).tolist()
        )
    return dict(
        count=n,
        surface_reconstruction_max_error=surface_error,
        coverage=coverage,
        distance_quantiles=[
            quantiles(values[:, c]) if present[c] else None for c in range(2)
        ],
        tie_counts=[int(((flags[:, c] & TIE) != 0).sum()) for c in range(2)],
    )


def audit(pools, targets, cfg):
    q = np.concatenate([p["_audit_xyz"] for p in pools.values()])
    values = np.concatenate([p["_audit_values"] for p in pools.values()])
    ids = np.concatenate([p["_audit_ids"] for p in pools.values()])
    flags = np.concatenate([p["_audit_flags"] for p in pools.values()])
    reference = [
        EdgeOracle(targets.edges, cfg, "cad_reference").query(q),
        VertexOracle(targets.vertices, cfg).query(q),
    ]
    errors = np.abs(values - np.column_stack([r["distance"] for r in reference]))
    err = errors.max(0)
    if err.max() > cfg.label_tolerance:
        raise PreprocessError("quality_failed", "CAD reference mismatch")
    for c, r in enumerate(reference):
        stored_tie = (flags[:, c] & TIE) != 0
        if np.any(stored_tie != r["tie"]):
            raise PreprocessError("quality_failed", "Nearest target ambiguity mismatch")
        unique = ~r["tie"] & (r["nearest_id"] >= 0)
        if np.any(ids[unique, c] != r["nearest_id"][unique]):
            raise PreprocessError(
                "quality_failed", "Unique nearest target identity mismatch"
            )
    # Full global E/V distances remain 1-Lipschitz even between different faces.
    if len(q) > 1:
        displacement = np.linalg.norm(q[1:].astype(float) - q[:-1], axis=1)
        for c, exists in enumerate((bool(targets.edges), bool(len(targets.vertices)))):
            if (
                exists
                and np.max(np.abs(np.diff(values[:, c])) - displacement)
                > 2 * cfg.label_tolerance
            ):
                raise PreprocessError(
                    "quality_failed", "Surface UDF Lipschitz mismatch"
                )
    return dict(reference_max_error=err.tolist(), reference_count=len(q))


def build_asset(record, output, cfg):
    start = time.monotonic()
    timings = {}
    output = Path(output)
    if output.exists():
        raise PreprocessError("output_exists", str(output))
    step = Path(record["step_path"])
    source_hash = file_hash(step)
    if record.get("source_sha256") not in (None, source_hash):
        raise PreprocessError("source_hash_mismatch", "Manifest does not match STEP")
    solid, normalization = read_normalized(step, cfg)
    check_solid_faces(solid, cfg)
    targets = collect_targets(solid, cfg)
    validate_geometry(targets, cfg)
    filters = audit_shape(solid, targets, cfg)
    cad_mesh = triangulate(targets, cfg)
    timings["cad"] = time.monotonic() - start
    t = time.monotonic()
    mesh, oracle = weld(cad_mesh, cfg)
    timings["condition"] = time.monotonic() - t
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        condition = temp / "condition"
        condition.mkdir()
        for name, x in [
            ("vertices", mesh[0]),
            ("triangles", mesh[1]),
        ]:
            np.save(condition / f"{name}.npy", x, allow_pickle=False)
        t = time.monotonic()
        seed = seed_for(record["sample_id"], cfg.seed, "training-pools")
        pools, reachability = sample_pools(targets, mesh, cfg, seed)
        timings["sampling"] = time.monotonic() - t
        t = time.monotonic()
        edge = EdgeOracle(
            targets.edges,
            cfg,
            mesh=mesh,
            edge_faces=[e["face_ids"] for e in targets.topology["edges"]],
            triangle_face_ids=cad_mesh[2],
        )
        vertex = VertexOracle(targets.vertices, cfg)
        statistics = {
            name: label_pool(
                temp / "pools" / name, pool, mesh, targets, cfg, edge, vertex
            )
            for name, pool in pools.items()
        }
        timings["labels"] = time.monotonic() - t
        t = time.monotonic()
        quality = audit(pools, targets, cfg)
        timings["audit"] = time.monotonic() - t
        t = time.monotonic()
        # Independent positions, not held-out rows of the training pools.
        k = cfg.evaluation_points_per_pool
        eval_pools, eval_reach = sample_pools(
            targets,
            mesh,
            replace(cfg, seed=seed_for(cfg.seed, "evaluation"), feature_min_samples=1),
            seed_for(seed, "evaluation"),
            counts=[k, max(k, len(targets.edges)), max(k, len(targets.vertices))],
            geometry_count=k,
        )
        for name, pool in eval_pools.items():
            label_pool(
                temp / "evaluation" / name, pool, mesh, targets, cfg, edge, vertex
            )
        quality["evaluation"] = audit(eval_pools, targets, cfg)
        timings["evaluation"] = time.monotonic() - t
        n = cfg.geometry_surface_pool_count
        sp = pools["surface"]
        geometry_meta = dict(
            geometry_version=GEOMETRY_VERSION,
            build_seed=cfg.seed,
            surface_count=n,
            mesh_sha256=mesh_hash(*mesh),
            points_sha256=array_hash(sp["points"][:n]),
            normals_sha256=array_hash(sp["normals"][:n]),
            normalization=normalization,
        )
        (condition / "meta.json").write_text(json.dumps(geometry_meta, indent=2))
        d = temp / "audit"
        d.mkdir()
        np.save(d / "triangle_cad_face_ids.npy", cad_mesh[2], allow_pickle=False)
        for name, x in features(cad_mesh, targets, cfg).items():
            np.save(d / f"{name}.npy", x, allow_pickle=False)
        meta = dict(
            schema=SCHEMA,
            channel_names=list(CHANNELS),
            package_version=__version__,
            runtime=runtime_identity(),
            asset_id=record["sample_id"],
            family_id=record["group_id"],
            split=record["split"],
            source_sha256=source_hash,
            normalization=normalization,
            config=cfg.to_dict(),
            code_sha256=code_fingerprint(),
            has_edges=bool(targets.edges),
            has_vertices=bool(len(targets.vertices)),
            empty_distance=EMPTY_DISTANCE,
            distance_definition="global_retained_cad_euclidean_on_condition_mesh",
            distance_engine=edge.report(),
            target_counts=dict(
                faces=len(targets.faces),
                edges=len(targets.edges),
                vertices=len(targets.vertices),
            ),
            filter=filters,
            statistics=statistics,
            reachability=reachability,
            evaluation_reachability=eval_reach,
            quality=quality,
            timings_seconds=dict(timings, total=time.monotonic() - start),
        )
        (temp / "meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False))
        (temp / "topology.json").write_text(json.dumps(targets.topology, indent=2))
        hashes = {
            str(p.relative_to(temp)): file_hash(p)
            for p in sorted(temp.rglob("*"))
            if p.is_file()
        }
        os.replace(temp, output)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return dict(
        asset_id=record["sample_id"],
        family_id=record["group_id"],
        split=record["split"],
        source_sha256=source_hash,
        hashes=hashes,
        counts=meta["target_counts"],
        seconds=time.monotonic() - start,
    )


def verify_asset(directory, hashes):
    directory = Path(directory)
    actual = {
        str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
    }
    if actual != set(hashes):
        raise PreprocessError("quality_failed", "Asset file inventory mismatch")
    for name, digest in hashes.items():
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or file_hash(directory / p) != digest:
            raise PreprocessError("quality_failed", f"Asset artifact changed: {name}")
