"""Continuous CAD mesh surface/feature pools with independent random streams."""

import numpy as np
from scipy.spatial import cKDTree
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_AbscissaPoint, GCPnts_UniformAbscissa
from veudf.schema import POOLS, SURFACE, SPATIAL, ZERO
from veudf.utils import allocate, seed_for
from veudf.geometry.sampling import base_surface, triangle_samples, face_normals
from veudf.preprocess.cad import PreprocessError, xyz
from veudf.geometry.mesh import QUERY_BATCH_SIZE


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


def proposal_fragments(mesh, max_edge, distance, radius):
    """Longest-edge bisection preserves each plane without overrefining thin CAD triangles."""
    pending = mesh[0][mesh[1]]
    parents = np.arange(len(pending))
    triangles, indices = [], []
    for _ in range(32):
        edges = pending[:, [1, 2, 0]] - pending
        # CAD may retain subnormal roundoff components at exact axes. Ignore
        # these only in the refinement metric, never in stored mesh geometry.
        edges[np.abs(edges) < 1e-15] = 0
        lengths = np.sqrt(np.einsum("ijk,ijk->ij", edges, edges))
        # A 1-Lipschitz distance and the centroid radius bound safely discard
        # whole fragments outside the target neighborhood before subdividing.
        keep = distance(pending.mean(1)) <= radius + lengths.max(1) * (2 / 3) + 1e-12
        pending, parents, lengths = pending[keep], parents[keep], lengths[keep]
        longest = lengths.argmax(1)
        split = lengths.max(1) > max_edge
        triangles.append(pending[~split])
        indices.append(parents[~split])
        if not split.any():
            t = np.concatenate(triangles)
            return (
                t.reshape(-1, 3),
                np.arange(len(t) * 3).reshape(-1, 3),
                np.concatenate(indices),
            )
        tri = pending[split]
        order = (longest[split, None] + np.arange(3)) % 3
        tri = np.take_along_axis(tri, order[..., None], axis=1)
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        mid = (a + b) / 2
        pending = np.concatenate((np.stack((a, mid, c), 1), np.stack((mid, b, c), 1)))
        parents = np.tile(parents[split], 2)
    raise PreprocessError("sampling_failed", "Proposal refinement exceeded its budget")


class FeatureSurfaceSampler:
    def __init__(self, mesh, cfg):
        import igl

        self.cfg = cfg
        original_v, original_f = mesh
        self.original_mesh = mesh
        self.normals = face_normals(*mesh)
        self.v, self.f = mesh
        tri = self.v[self.f]
        self.centers = tri.mean(1)
        self.radii = np.linalg.norm(tri - self.centers[:, None], axis=-1).max(1)
        self.bvh = igl.AABB()
        self.bvh.init(original_v, original_f)

    def pool(self, targets, count, kind, seed):
        cfg = self.cfg
        entities = targets.edges if kind == "edge" else targets.vertices
        if not len(entities) or not count:
            return self.empty(), dict(unresolved=[], transfers=[], coverage=[])
        rng = np.random.default_rng(seed)
        proposals, weights, unresolved = [], [], []
        spacing = cfg.feature_curve_spacing
        for i, entity in enumerate(entities):
            if kind == "edge":
                length = GCPnts_AbscissaPoint.Length(
                    BRepAdaptor_Curve(entity), cfg.solver_tolerance
                )
                seeds = curve_points(
                    entity,
                    max(3, int(np.ceil(length / spacing)) + 1),
                    cfg.solver_tolerance,
                    targets.topology["edges"][i]["closed"],
                )
                seed_tree = cKDTree(seeds)
                distance = lambda x, t=seed_tree: t.query(x)[0]
                error_bound = (
                    length
                    / (
                        len(seeds)
                        if targets.topology["edges"][i]["closed"]
                        else len(seeds) - 1
                    )
                    / 2
                )
                error_bound += cfg.solver_tolerance
                weight = np.sqrt(length)
            else:
                seeds = np.asarray(entity)[None]
                distance = lambda x, p=seeds: np.linalg.norm(x - p, axis=1)
                error_bound = 0.0
                weight = 1.0
            minimum = float(
                np.sqrt(
                    min(
                        self.bvh.squared_distance(
                            *self.original_mesh, seeds[start : start + QUERY_BATCH_SIZE]
                        )[0].min()
                        for start in range(0, len(seeds), QUERY_BATCH_SIZE)
                    )
                )
            )
            # Query each triangle once against the target seed tree. Expanding
            # a triangle-center radius for every edge seed materializes a huge
            # Cartesian neighborhood on CAD meshes with a few large planes.
            ids = np.flatnonzero(
                distance(self.centers)
                <= cfg.feature_radii[-1] + self.radii + cfg.solver_tolerance
            )
            available = np.flatnonzero(np.asarray(cfg.feature_radii) > minimum + 1e-8)
            if not len(ids) or not len(available):
                unresolved.append(i)
                continue
            proposals.append(
                (i, ids, distance, int(available[0]), minimum, error_bound)
            )
            weights.append(weight)
        if not proposals:
            return self.empty(), dict(unresolved=unresolved, transfers=[], coverage=[])
        if count < len(proposals) * cfg.feature_min_samples:
            raise PreprocessError(
                "sampling_budget", f"{kind}: budget below per-instance minimum"
            )
        quotas = (
            allocate(count - len(proposals) * cfg.feature_min_samples, weights)
            + cfg.feature_min_samples
        )
        points, normals, triangle_ids, instances, transfers, coverage = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for quota, (i, ids, distance, first, minimum, error_bound) in zip(
            quotas, proposals
        ):
            # Refine only this target neighborhood, keeping the original face
            # IDs. The final condition mesh and triangle normals never change.
            pv, pf, parents = proposal_fragments(
                (self.v, self.f[ids]),
                cfg.feature_radii[-1],
                distance,
                cfg.feature_radii[-1],
            )
            parents = ids[parents]
            centers = pv.reshape(-1, 3, 3).mean(1)
            fragment_radius = cfg.feature_radii[-1] * (2 / 3)
            layer = allocate(int(quota), cfg.feature_weights)
            if first:
                layer[first] += layer[:first].sum()
                layer[:first] = 0
                transfers.append(
                    dict(instance=i, minimum_estimate=minimum, first_layer=first)
                )
            collected, collected_ids = [], []
            for b, need in enumerate(layer):
                need = int(need)
                low = cfg.feature_radii[b - 1] if b else 0.0
                layer_ids = np.flatnonzero(
                    distance(centers)
                    <= cfg.feature_radii[b] + fragment_radius + error_bound
                )
                acceptance = 0.125
                for _ in range(cfg.sampling_rounds):
                    if not need:
                        break
                    candidate, candidate_ids = triangle_samples(
                        pv,
                        pf,
                        min(max(int(np.ceil(need / acceptance * 1.2)), 256), 65536),
                        rng,
                        layer_ids,
                        return_ids=True,
                    )
                    d = distance(candidate.astype(np.float64))
                    good = (d <= cfg.feature_radii[b]) & (
                        (d - error_bound > low) if b else (d >= 0)
                    )
                    acceptance = max(float(good.mean()), 1 / len(candidate))
                    take = candidate[good][:need]
                    collected.append(take)
                    collected_ids.append(parents[candidate_ids[good][:need]])
                    need -= len(take)
                if need and b + 1 < len(layer):
                    layer[b + 1] += need
                    transfers.append(
                        dict(instance=i, from_layer=b, to_layer=b + 1, count=need)
                    )
                elif need:
                    raise PreprocessError(
                        "sampling_failed",
                        f"{kind} {i}: unable to fill reachable surface pool",
                    )
            p = np.concatenate(collected).astype(np.float32)
            points.append(p)
            ti = np.concatenate(collected_ids)
            triangle_ids.append(ti)
            normals.append(self.normals[ti])
            instances.extend([i] * len(p))
            coverage.append(dict(instance=i, count=len(p)))
        p = np.concatenate(points)
        return dict(
            points=p,
            normals=np.concatenate(normals),
            triangle_ids=np.concatenate(triangle_ids).astype(np.int32),
            source_instance=np.asarray(instances, np.int32),
        ), dict(unresolved=unresolved, transfers=transfers, coverage=coverage)

    @staticmethod
    def empty():
        return dict(
            points=np.empty((0, 3), np.float32),
            normals=np.empty((0, 3), np.float32),
            source_instance=np.empty(0, np.int32),
            triangle_ids=np.empty(0, np.int32),
        )


def spatial_pool(base, count, oracle, cfg, seed):
    """Perturb one surface source, preserving accepted mode/layer quotas and owner."""
    rng = np.random.default_rng(seed)
    chunks, values, instances, modes, layers = [], [], [], [], []
    bounds = [0, 0.002, 0.01, cfg.band_max]
    normal_count, isotropic_count = allocate(count, [0.8, 0.2])
    for mode, quotas in enumerate(
        (
            allocate(int(normal_count), [0.3, 0.4, 0.3]),
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
                sdf = oracle.query(q)
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
        _sdf=np.concatenate(values) if values else np.empty(0),
        source_instance=np.asarray(instances, np.int32),
        sample_type=np.full(count, SPATIAL, np.uint8),
        perturbation_mode=np.asarray(modes, np.uint8),
        perturbation_layer=np.asarray(layers, np.uint8),
    )


def merge_neighborhood(surface, spatial):
    """Surface normals/triangle IDs are meaningful only on sample_type=SURFACE."""
    n, m = len(surface["points"]), len(spatial["points"])
    return dict(
        points=np.concatenate((surface["points"], spatial["points"])),
        normals=np.concatenate((surface["normals"], np.zeros((m, 3), np.float32))),
        triangle_ids=np.concatenate(
            (surface["triangle_ids"], np.full(m, -1, np.int32))
        ),
        source_instance=np.concatenate(
            (surface["source_instance"], spatial["source_instance"])
        ),
        sample_type=np.concatenate(
            (np.full(n, SURFACE, np.uint8), spatial["sample_type"])
        ),
        perturbation_mode=np.concatenate(
            (np.full(n, 255, np.uint8), spatial["perturbation_mode"])
        ),
        perturbation_layer=np.concatenate(
            (np.full(n, 255, np.uint8), spatial["perturbation_layer"])
        ),
        _sdf=np.concatenate((np.full(n, np.nan), spatial["_sdf"])),
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
    fs = FeatureSurfaceSampler(mesh, cfg)
    feature_bases, feature_spatial_counts, reachability = [], [], {}
    for count, ratio, kind in zip(
        (e, v), cfg.feature_surface_ratios, ("edge", "vertex")
    ):
        surface_count, spatial_count = map(int, allocate(count, [ratio, 1 - ratio]))
        # A feature spatial sample needs a surface seed, even for a space-only quota.
        present = bool(targets.edges) if kind == "edge" else bool(len(targets.vertices))
        base, reach = fs.pool(
            targets,
            max(surface_count, int(spatial_count > 0)) if present else 0,
            kind,
            seed_for(seed, kind),
        )
        reachability[kind] = reach
        if not len(base["points"]):
            s += surface_count
            n += spatial_count
            spatial_count = 0
        feature_bases.append((base, surface_count))
        feature_spatial_counts.append(spatial_count)
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
    for (base, surface_count), count, kind in zip(
        feature_bases, feature_spatial_counts, ("edge", "vertex")
    ):
        spatial = spatial_pool(
            base, count, oracle, cfg, seed_for(seed, kind, "spatial")
        )
        surface = {key: value[:surface_count] for key, value in base.items()}
        features.append(merge_neighborhood(surface, spatial))
    return dict(zip(POOLS, [sp, npool, *features])), reachability
