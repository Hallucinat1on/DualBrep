"""Continuous CAD mesh surface and 3D near-surface/feature pools with independent random streams."""

import numpy as np
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_AbscissaPoint, GCPnts_UniformAbscissa
from veudf.schema import POOLS, SURFACE, SPATIAL, ZERO
from veudf.utils import allocate, seed_for
from veudf.geometry.sampling import base_surface, triangle_samples, face_normals
from veudf.preprocess.cad import PreprocessError, xyz
from veudf.geometry.mesh import QUERY_BATCH_SIZE

FEATURE_3D_PERTURBATION_MODE = 2


def curve_points(edge, count, tolerance, closed=False):
    curve = BRepAdaptor_Curve(edge)
    if count == 1:
        return np.asarray([xyz(curve.Value(curve.FirstParameter()))], np.float64)
    requested = max(2, int(count)) + int(closed)
    sampler = GCPnts_UniformAbscissa(curve, requested, tolerance)
    if not sampler.IsDone() or sampler.NbPoints() != requested:
        raise PreprocessError("sampling_failed", "Arc-length sampling failed")
    return np.array(
        [
            xyz(curve.Value(sampler.Parameter(i + 1)))
            for i in range(requested - int(closed))
        ],
        np.float64,
    )


def sample_edges(targets, cfg):
    """Independent fixed arc-length pool used only for deduplication."""
    if not targets.edges:
        return np.empty((0, 3))
    lengths = [
        GCPnts_AbscissaPoint.Length(BRepAdaptor_Curve(e), cfg.solver_tolerance)
        for e in targets.edges
    ]
    counts = allocate(cfg.dedup_edge_pool_count, lengths)
    return np.concatenate(
        [
            curve_points(
                e,
                max(2, n),
                cfg.solver_tolerance,
                targets.topology["edges"][i]["closed"],
            )
            for i, (e, n) in enumerate(zip(targets.edges, counts))
        ]
    )


def zero_pools(targets, cfg, evaluation=False):
    budget = (
        max(cfg.evaluation_zero_edge_count, len(targets.edges))
        if evaluation
        else cfg.zero_edge_count
    )
    if len(targets.edges) > budget:
        raise PreprocessError(
            "sampling_failed", "Edge zero budget cannot cover every edge"
        )
    points, ids = [], []
    if targets.edges:
        lengths = [
            GCPnts_AbscissaPoint.Length(BRepAdaptor_Curve(e), cfg.solver_tolerance)
            for e in targets.edges
        ]
        minimum = min(cfg.zero_edge_min_samples, budget // len(targets.edges))
        counts = minimum + allocate(budget - minimum * len(targets.edges), lengths)
        for i, (edge, n) in enumerate(zip(targets.edges, counts)):
            # Midpoints of equal arc-length cells exclude endpoints and closed seams.
            samples = curve_points(edge, 2 * int(n) + 1, cfg.solver_tolerance)[1::2]
            points.append(samples)
            ids.extend([i] * len(samples))

    def bank(x, instances, mask):
        x = np.asarray(x, np.float32).reshape(-1, 3)
        return dict(
            points=x,
            source_instance=np.asarray(instances, np.int32),
            sample_type=np.full(len(x), ZERO, np.uint8),
            zero_mask=np.tile(mask, (len(x), 1)).astype(bool),
        )

    return dict(
        edge_zero=bank(np.concatenate(points) if points else [], ids, [True, False]),
        vertex_zero=bank(
            targets.vertices, np.arange(len(targets.vertices)), [True, True]
        ),
    )


def _query_sdf_batched(oracle, points):
    if not len(points):
        return np.empty(0, np.float64)
    if len(points) <= QUERY_BATCH_SIZE:
        return oracle.query(points)
    return np.concatenate(
        [
            oracle.query(points[start : start + QUERY_BATCH_SIZE])
            for start in range(0, len(points), QUERY_BATCH_SIZE)
        ]
    )


def _empty_neighborhood():
    return dict(
        points=np.empty((0, 3), np.float32),
        normals=np.empty((0, 3), np.float32),
        triangle_ids=np.empty(0, np.int32),
        source_instance=np.empty(0, np.int32),
        sample_type=np.empty(0, np.uint8),
        perturbation_mode=np.empty(0, np.uint8),
        perturbation_layer=np.empty(0, np.uint8),
        _sdf=np.empty(0, np.float64),
    )


def _orthonormal_frame(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-12
    tangents = np.zeros_like(vectors)
    tangents[valid] = vectors[valid] / norms[valid]
    tangents[~valid] = np.array([0.0, 0.0, 1.0])
    ref = np.zeros_like(tangents)
    use_z = np.abs(tangents[:, 2]) < 0.9
    ref[use_z, 2] = 1.0
    ref[~use_z, 0] = 1.0
    e1 = np.cross(tangents, ref)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    e2 = np.cross(tangents, e1)
    return e1, e2, valid


def _unit_directions(rng, count):
    d = rng.normal(size=(count, 3))
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    zero = norms[:, 0] < 1e-12
    if zero.any():
        d[zero] = np.array([1.0, 0.0, 0.0])
        norms[zero] = 1.0
    return d / norms


def _near_surface_normal_strata(band_max):
    if band_max > 0.025:
        return [0.0, 0.002, 0.01, 0.025, float(band_max)], [0.25, 0.35, 0.25, 0.15]
    return [0.0, 0.002, 0.01, float(band_max)], [0.3, 0.4, 0.3]


def spatial_pool(base, count, oracle, cfg, seed):
    """Perturb surface points with 4-layer normal offsets (80%) and 4-scale isotropic noise (20%)."""
    if not count:
        return dict(
            points=np.empty((0, 3), np.float32),
            _sdf=np.empty(0, np.float64),
            source_instance=np.empty(0, np.int32),
            sample_type=np.empty(0, np.uint8),
            perturbation_mode=np.empty(0, np.uint8),
            perturbation_layer=np.empty(0, np.uint8),
        )
    rng = np.random.default_rng(seed)
    chunks, values, instances, modes, layers = [], [], [], [], []
    bounds, normal_weights = _near_surface_normal_strata(cfg.band_max)
    normal_count, isotropic_count = allocate(count, [0.8, 0.2])
    for mode, quotas in enumerate(
        (
            allocate(int(normal_count), normal_weights),
            allocate(int(isotropic_count), [1, 1, 1, 1]),
        )
    ):
        for layer, quota in enumerate(quotas):
            remaining = int(quota)
            for _ in range(cfg.sampling_rounds):
                if not remaining:
                    break
                idx = rng.integers(
                    len(base["points"]), size=min(max(remaining * 2, 128), 131072)
                )
                p = base["points"][idx].astype(np.float64)
                if mode == 0:
                    delta = rng.uniform(
                        bounds[layer], bounds[layer + 1], len(p)
                    ) * rng.choice([-1, 1], len(p))
                    q = p + base["normals"][idx] * delta[:, None]
                else:
                    q = p + rng.normal(size=p.shape) * [0.001, 0.005, 0.01, 0.02][layer]
                q = q.astype(np.float32)
                inside = (np.abs(q) <= 1).all(1)
                q, idx = q[inside], idx[inside]
                sdf = _query_sdf_batched(oracle, q)
                good = np.abs(sdf) <= cfg.band_max
                accepted = q[good][:remaining]
                chunks.append(accepted)
                values.append(sdf[good][:remaining])
                instances.extend(base["source_instance"][idx[good]][:remaining])
                modes.extend([mode] * len(accepted))
                layers.extend([layer] * len(accepted))
                remaining -= len(accepted)
            if remaining:
                raise PreprocessError(
                    "sampling_failed",
                    f"Spatial quota unfilled: mode={mode}, layer={layer}, missing={remaining}",
                )
    return dict(
        points=np.concatenate(chunks) if chunks else np.empty((0, 3), np.float32),
        _sdf=np.concatenate(values) if values else np.empty(0, np.float64),
        source_instance=np.asarray(instances, np.int32),
        sample_type=np.full(count, SPATIAL, np.uint8),
        perturbation_mode=np.asarray(modes, np.uint8),
        perturbation_layer=np.asarray(layers, np.uint8),
    )


def sample_feature_3d_pool(targets, count, kind, oracle, cfg, seed):
    """Pure 3D radial tube/ball sampling around CAD edges/vertices within |s_mesh| <= band_max."""
    entities = targets.edges if kind == "edge" else targets.vertices
    if not len(entities) or not count:
        return _empty_neighborhood(), dict(
            mode="feature_3d_radial", unresolved=[], transfers=[], coverage=[]
        )
    if count < len(entities) * cfg.feature_min_samples:
        raise PreprocessError(
            "sampling_budget", f"{kind}: budget below per-instance minimum"
        )
    rng = np.random.default_rng(seed)
    if kind == "edge":
        weights = []
        edge_segments = []
        spacing = cfg.feature_curve_spacing
        for i, edge in enumerate(entities):
            length = GCPnts_AbscissaPoint.Length(
                BRepAdaptor_Curve(edge), cfg.solver_tolerance
            )
            closed = bool(targets.topology["edges"][i]["closed"])
            n_seeds = max(4, int(np.ceil(length / spacing)) + 1)
            seeds = curve_points(edge, n_seeds, cfg.solver_tolerance, closed=closed)
            if closed:
                starts = seeds
                ends = np.roll(seeds, -1, axis=0)
            else:
                starts = seeds[:-1]
                ends = seeds[1:]
            deltas = ends - starts
            e1, e2, valid = _orthonormal_frame(deltas)
            edge_segments.append((starts, deltas, e1, e2, valid))
            weights.append(np.sqrt(length))
    else:
        weights = np.ones(len(entities), np.float64)
        edge_segments = None

    quotas = (
        allocate(count - len(entities) * cfg.feature_min_samples, weights)
        + cfg.feature_min_samples
    )
    slots = []
    for i, quota in enumerate(quotas):
        layer_quotas = allocate(int(quota), cfg.feature_weights)
        for layer, need in enumerate(layer_quotas):
            if int(need) > 0:
                slots.append([i, layer, int(need)])

    collected_pts = []
    collected_sdf = []
    collected_inst = []
    collected_layer = []
    coverage_counts = np.zeros(len(entities), np.int64)
    radii = cfg.feature_radii

    for round_idx in range(cfg.sampling_rounds):
        active = [slot for slot in slots if slot[2] > 0]
        if not active:
            break
        oversample = 1.15 if round_idx == 0 else 2.0
        candidate_chunks = []
        spans = []
        offset = 0
        for slot in active:
            inst, layer, need = slot
            draw_n = max(int(np.ceil(need * oversample)), need + 8)
            low = float(radii[layer - 1]) if layer > 0 else 0.0
            high = float(radii[layer])
            r = rng.uniform(low, high, draw_n)
            if kind == "edge":
                starts, deltas, e1, e2, valid = edge_segments[inst]
                seg_idx = rng.integers(len(starts), size=draw_n)
                u = rng.random(draw_n)
                origins = starts[seg_idx] + u[:, None] * deltas[seg_idx]
                theta = rng.uniform(0.0, 2.0 * np.pi, draw_n)
                dirs = (
                    np.cos(theta)[:, None] * e1[seg_idx]
                    + np.sin(theta)[:, None] * e2[seg_idx]
                )
                bad_dir = ~valid[seg_idx]
                if bad_dir.any():
                    dirs[bad_dir] = _unit_directions(rng, int(bad_dir.sum()))
                q = (origins + r[:, None] * dirs).astype(np.float32)
            else:
                origin = np.asarray(entities[inst], np.float64)
                dirs = _unit_directions(rng, draw_n)
                q = (origin[None, :] + r[:, None] * dirs).astype(np.float32)
            candidate_chunks.append(q)
            spans.append((slot, offset, offset + draw_n))
            offset += draw_n

        all_q = np.concatenate(candidate_chunks, axis=0)
        inside = (np.abs(all_q) <= 1).all(1)
        good = np.zeros(len(all_q), dtype=bool)
        all_sdf = np.full(len(all_q), np.nan, dtype=np.float64)
        if inside.any():
            sdf_in = _query_sdf_batched(oracle, all_q[inside])
            all_sdf[inside] = sdf_in
            good[inside] = np.abs(sdf_in) <= cfg.band_max

        for slot, start, end in spans:
            inst, layer, need = slot
            local_good = np.flatnonzero(good[start:end])
            if not len(local_good):
                continue
            pick = start + local_good[:need]
            taken = len(pick)
            collected_pts.append(all_q[pick])
            collected_sdf.append(all_sdf[pick])
            collected_inst.append(np.full(taken, inst, np.int32))
            collected_layer.append(np.full(taken, layer, np.uint8))
            coverage_counts[inst] += taken
            slot[2] -= taken

    unfilled = [slot for slot in slots if slot[2] > 0]
    if unfilled:
        inst, layer, missing = unfilled[0]
        raise PreprocessError(
            "sampling_failed",
            f"{kind} {inst}: 3D radial quota unfilled (layer={layer}, missing={missing})",
        )

    points = np.concatenate(collected_pts, axis=0)
    total = len(points)
    return dict(
        points=points,
        normals=np.zeros((total, 3), np.float32),
        triangle_ids=np.full(total, -1, np.int32),
        source_instance=np.concatenate(collected_inst, axis=0),
        sample_type=np.full(total, SPATIAL, np.uint8),
        perturbation_mode=np.full(total, FEATURE_3D_PERTURBATION_MODE, np.uint8),
        perturbation_layer=np.concatenate(collected_layer, axis=0),
        _sdf=np.concatenate(collected_sdf, axis=0),
    ), dict(
        mode="feature_3d_radial",
        unresolved=[],
        transfers=[],
        coverage=[
            dict(instance=i, count=int(coverage_counts[i]))
            for i in range(len(entities))
        ],
    )


def sample_pools(targets, mesh, oracle, cfg, seed, counts=None, geometry_count=None):
    counts = (
        allocate(cfg.pool_count, cfg.pool_weights)
        if counts is None
        else np.asarray(counts)
    )
    geometry_count = (
        cfg.geometry_surface_pool_count if geometry_count is None else geometry_count
    )
    s, n, e, v = map(int, counts)
    feature_counts = []
    for count, present in (
        (e, bool(targets.edges)),
        (v, bool(len(targets.vertices))),
    ):
        if present:
            feature_counts.append(count)
        else:
            n += count
            feature_counts.append(0)
    if s < geometry_count:
        raise ValueError("Surface pool smaller than geometry prefix")
    p, normal, base_ids = base_surface(mesh, geometry_count, cfg.seed, return_ids=True)
    extra, extra_ids = triangle_samples(
        *mesh,
        s - geometry_count,
        np.random.default_rng(seed_for(seed, "extra")),
        return_ids=True,
    )
    sp = dict(
        points=np.concatenate((p, extra)),
        normals=np.concatenate((normal, face_normals(*mesh)[extra_ids])),
        triangle_ids=np.concatenate((base_ids, extra_ids)).astype(np.int32),
        source_instance=np.full(s, -1, np.int32),
        sample_type=np.full(s, SURFACE, np.uint8),
    )
    npool = spatial_pool(sp, n, oracle, cfg, seed_for(seed, "near"))
    features = []
    reachability = {}
    for count, kind in zip(feature_counts, ("edge", "vertex")):
        pool, reach = sample_feature_3d_pool(
            targets, count, kind, oracle, cfg, seed_for(seed, kind, "spatial3d")
        )
        features.append(pool)
        reachability[kind] = reach
    return dict(zip(POOLS, [sp, npool, *features])), reachability
