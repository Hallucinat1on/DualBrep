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
from veudf.schema import SCHEMA, CHANNELS, EMPTY_DISTANCE, GEOMETRY_VERSION, SURFACE
from veudf.utils import array_hash, file_hash, seed_for
from veudf.geometry.sampling import mesh_hash
from veudf import __version__
from veudf.preprocess.cad import (
    PreprocessError,
    collect_targets,
    read_normalized,
    triangulate,
)
from veudf.preprocess.condition import weld
from veudf.preprocess.dedup import features
from veudf.preprocess.distance import FeatureUDFOracle
from veudf.preprocess.reference import (
    ReferenceEdgeOracle,
    CADSDFOracle,
    ReferenceVertexOracle,
)
from veudf.preprocess.filtering import audit_shape, check_solid_faces
from veudf.preprocess.sampling import sample_pools, zero_pools

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


def label_pool(directory, pool, oracle, targets, cfg, feature, reference=None):
    directory.mkdir(parents=True)
    cached_sdf = pool.pop("_sdf", None)
    for key, value in pool.items():
        np.save(directory / f"{key}.npy", value, allow_pickle=False)
    x = pool["points"]
    n = len(x)
    values = np.lib.format.open_memmap(
        directory / "fields.npy", mode="w+", dtype=np.float32, shape=(n, 3)
    )
    nearest = np.lib.format.open_memmap(
        directory / "nearest_ids.npy", mode="w+", dtype=np.int32, shape=(n, 2)
    )
    flags = np.lib.format.open_memmap(
        directory / "flags.npy", mode="w+", dtype=np.uint8, shape=(n,)
    )
    saturated = np.lib.format.open_memmap(
        directory / "saturated.npy", mode="w+", dtype=bool, shape=(n, 2)
    )
    elapsed = dict(udf=0.0, sdf=0.0)
    for start in range(0, n, cfg.label_batch_size):
        q = x[start : start + cfg.label_batch_size].astype(np.float64)
        t = time.monotonic()
        if reference is None:
            e, v = feature.query(q)
            saturated[start : start + len(q)] = np.column_stack(
                (e["saturated"], v["saturated"])
            )
        else:
            e, v = reference[0].query(q), reference[1].query(q)
            saturated[start : start + len(q)] = False
        elapsed["udf"] += time.monotonic() - t
        t = time.monotonic()
        if cached_sdf is None:
            sdf = oracle.query(q)
        else:
            sdf = cached_sdf[start : start + len(q)].copy()
            missing = ~np.isfinite(sdf)
            if missing.any():
                sdf[missing] = oracle.query(q[missing])
        values[start : start + len(q)] = np.column_stack(
            (sdf, e["distance"], v["distance"])
        )
        nearest[start : start + len(q)] = np.column_stack(
            (e["nearest_id"], v["nearest_id"])
        )
        flags[start : start + len(q)] = e["tie"].astype(np.uint8) | (
            v["tie"].astype(np.uint8) << 1
        )
        elapsed["sdf"] += time.monotonic() - t
    if n and (not np.isfinite(values).all() or np.any(values[:, 1:] < 0)):
        raise PreprocessError("quality_failed", "Invalid field labels")
    empty_distance = cfg.udf_truncation if reference is None else EMPTY_DISTANCE
    for c, present in enumerate([bool(targets.edges), bool(len(targets.vertices))]):
        if not present and np.any(values[:, c + 1] != np.float32(empty_distance)):
            raise PreprocessError("quality_failed", "Invalid empty channel")
    if directory.name in ("edge_neighborhood", "vertex_neighborhood") and n:
        c = 1 if directory.name == "edge_neighborhood" else 2
        surface_rows = pool["sample_type"] == SURFACE
        if (
            surface_rows.any()
            and np.max(values[surface_rows, c])
            > cfg.feature_radii[-1] + cfg.label_tolerance
        ):
            raise PreprocessError(
                "quality_failed", "Feature surface exceeds its distance bound"
            )
    if len(targets.vertices) and np.any(
        values[:, 1] > values[:, 2] + cfg.label_tolerance
    ):
        raise PreprocessError("quality_failed", "dE > dV")
    if (
        "zero_mask" in pool
        and n
        and np.any(values[:, 1:][pool["zero_mask"]] > cfg.label_tolerance)
    ):
        raise PreprocessError("quality_failed", "CAD zero-set residual")
    if "strata" not in pool:
        channel = 2 if directory.name == "vertex_neighborhood" else 1
        layer = np.searchsorted(
            cfg.feature_radii, values[:, channel], side="left"
        ).astype(np.uint8)
        np.save(directory / "strata.npy", layer[:, None], allow_pickle=False)
    values.flush()
    nearest.flush()
    flags.flush()
    saturated.flush()
    rng = np.random.default_rng(seed_for(cfg.seed, "label-audit", directory.name))
    rows = rng.choice(n, min(n, cfg.reference_checks), replace=False)
    pool["_audit_xyz"] = x[rows]
    pool["_audit_values"] = np.asarray(values[rows])
    pool["_audit_ids"] = np.asarray(nearest[rows])
    pool["_audit_flags"] = np.asarray(flags[rows])
    pool["_audit_saturated"] = np.asarray(saturated[rows])
    nearest_coverage = {}
    for c, name, total in [
        (0, "edge", len(targets.edges)),
        (1, "vertex", len(targets.vertices)),
    ]:
        ids = nearest[:, c]
        counts = np.bincount(ids[ids >= 0], minlength=total)
        nearest_coverage[name] = dict(
            counts=counts.tolist(), missing=np.flatnonzero(counts == 0).tolist()
        )
    return dict(
        count=n,
        sample_type_counts=np.bincount(pool["sample_type"], minlength=3).tolist(),
        timings_seconds=elapsed,
        saturated_counts=np.count_nonzero(saturated, axis=0).tolist(),
        unique_coordinates=len(np.unique(x, axis=0)),
        mesh_abs=quantiles(np.abs(values[:, 0])),
        positive=int((values[:, 0] > 0).sum()),
        negative=int((values[:, 0] < 0).sum()),
        coverage=nearest_coverage,
        perturbation_counts=(
            {
                "/".join(map(str, key)): int(count)
                for key, count in zip(
                    *np.unique(
                        np.column_stack(
                            (
                                pool["sample_type"],
                                pool["perturbation_mode"],
                                pool["perturbation_layer"],
                            )
                        ),
                        axis=0,
                        return_counts=True,
                    )
                )
            }
            if "perturbation_mode" in pool
            else None
        ),
        band_counts={
            str(h): int((np.abs(values[:, 0]) <= h).sum()) for h in [0.01, 0.02, 0.03]
        },
    )


def audit(pools, solid, targets, cfg, truncated=True):
    q = np.concatenate([p["_audit_xyz"] for p in pools.values()])
    values = np.concatenate([p["_audit_values"] for p in pools.values()])
    er = ReferenceEdgeOracle(targets.edges, cfg, exhaustive=True).query(q)
    e = er["distance"]
    stored_ids = np.concatenate([p["_audit_ids"] for p in pools.values()])
    stored_flags = np.concatenate([p["_audit_flags"] for p in pools.values()])
    stored_saturated = np.concatenate([p["_audit_saturated"] for p in pools.values()])
    vr = ReferenceVertexOracle(targets.vertices, cfg).query(q)
    v = vr["distance"]
    for c, ref in enumerate((er, vr)):
        unique = (
            ~ref["tie"] & ((stored_flags & (1 << c)) == 0) & (ref["nearest_id"] >= 0)
        )
        if truncated:
            unique &= ref["distance"] < cfg.udf_truncation - cfg.label_tolerance
        if np.any(stored_ids[unique, c] != ref["nearest_id"][unique]):
            raise PreprocessError(
                "quality_failed", "Unique nearest feature identity mismatch"
            )
        if np.any(stored_ids[stored_saturated[:, c], c] != -1):
            raise PreprocessError(
                "quality_failed", "Saturated points claim a nearest feature"
            )
    reference = np.column_stack((e, v))
    expected = reference.copy()
    if truncated:
        expected = np.minimum(expected, cfg.udf_truncation)
    err = np.max(np.abs(expected - values[:, 1:]), axis=0)
    if err.max() > cfg.label_tolerance + cfg.geometry_tolerance:
        raise PreprocessError("quality_failed", "CAD reference mismatch")
    cad_sdf = CADSDFOracle(solid, targets.faces, cfg).query(q)
    if (
        targets.edges
        and np.max(np.abs(cad_sdf) - reference[:, 0])
        > cfg.label_tolerance + 2 * cfg.mesh_deflection
    ):
        raise PreprocessError("quality_failed", "CAD containment mismatch")
    # Both E/V distance functions must remain 1-Lipschitz on audited pairs.
    if len(q) > 1:
        d = np.linalg.norm(q[1:].astype(np.float64) - q[:-1], axis=1)
        for c, present in enumerate([bool(targets.edges), bool(len(targets.vertices))]):
            if (
                present
                and np.max(np.abs(np.diff(values[:, c + 1])) - d)
                > 2 * cfg.label_tolerance
            ):
                raise PreprocessError("quality_failed", "UDF Lipschitz mismatch")
    return dict(
        reference_max_error=err.tolist(),
        mesh_vs_cad=quantiles(np.abs(values[:, 0] - cad_sdf)),
    )


def build_asset(record, output, cfg):
    start = time.monotonic()
    cpu_start = time.process_time()
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
        pools, reachability = sample_pools(targets, mesh, oracle, cfg, seed)
        pools.update(zero_pools(targets, cfg))
        timings["sampling"] = time.monotonic() - t
        t = time.monotonic()
        feature = FeatureUDFOracle(targets, cfg)
        timings["feature_preparation"] = time.monotonic() - t
        t = time.monotonic()
        statistics = {
            name: label_pool(temp / "pools" / name, pool, oracle, targets, cfg, feature)
            for name, pool in pools.items()
        }
        timings["training_labels"] = time.monotonic() - t
        t = time.monotonic()
        quality = audit(pools, solid, targets, cfg)
        timings["training_audit"] = time.monotonic() - t
        t = time.monotonic()
        # Independent positions, not held-out rows of the training pools.
        k = cfg.evaluation_points_per_pool
        es, vs = max(k, len(targets.edges)), max(k, len(targets.vertices))
        en, vn = round(k * 0.3), round(k * 0.2)
        eval_pools, eval_reach = sample_pools(
            targets,
            mesh,
            oracle,
            replace(
                cfg,
                seed=seed_for(cfg.seed, "evaluation"),
                feature_min_samples=1,
                feature_surface_ratios=(es / (es + en), vs / (vs + vn)),
            ),
            seed_for(seed, "evaluation"),
            counts=[k, k // 2, es + en, vs + vn],
            geometry_count=k,
        )
        eval_pools.update(zero_pools(targets, cfg, evaluation=True))
        timings["evaluation_sampling"] = time.monotonic() - t
        t = time.monotonic()
        reference = ReferenceEdgeOracle(targets.edges, cfg), ReferenceVertexOracle(
            targets.vertices, cfg
        )
        for name, pool in eval_pools.items():
            label_pool(
                temp / "evaluation" / name,
                pool,
                oracle,
                targets,
                cfg,
                feature,
                reference,
            )
        timings["evaluation_labels"] = time.monotonic() - t
        t = time.monotonic()
        quality["evaluation"] = audit(eval_pools, solid, targets, cfg, truncated=False)
        timings["evaluation_audit"] = time.monotonic() - t
        t = time.monotonic()
        if cfg.volume_pool_count:
            for parent, count, stream in [
                ("pools", cfg.volume_pool_count, "volume-training"),
                ("evaluation", cfg.evaluation_volume_count, "volume-evaluation"),
            ]:
                rng = np.random.default_rng(seed_for(seed, stream))
                xyz = rng.uniform(-1, 1, (count, 3)).astype(np.float32)
                dest = temp / parent / "volume"
                dest.mkdir()
                np.save(dest / "points.npy", xyz, allow_pickle=False)
                sdf = np.concatenate(
                    [
                        oracle.query(xyz[i : i + cfg.batch_size])
                        for i in range(0, count, cfg.batch_size)
                    ]
                ).astype(np.float32)
                np.save(dest / "sdf.npy", sdf[:, None], allow_pickle=False)
        timings["volume"] = time.monotonic() - t
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
            empty_distance=cfg.udf_truncation,
            evaluation_empty_distance=EMPTY_DISTANCE,
            band_max=cfg.band_max,
            udf_truncation=cfg.udf_truncation,
            evaluation_udf="exact_cad",
            labeling=dict(
                method="feature_neighborhood_min",
                occ_calls=feature.calls,
                batches=feature.batches,
            ),
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
        cpu_seconds=time.process_time() - cpu_start,
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
