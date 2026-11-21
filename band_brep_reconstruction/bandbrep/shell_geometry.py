"""Sphere fitting and multi-scale V-shell vertex recovery."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .deadline import checkpoint
from .edge_first_grid import sample_local_box
from .edge_first_mc import extract_shell
from .schemas import VertexNode
from .surface_mesh import components


def fit_sphere(points, weights):
    mean = np.average(points, axis=0, weights=weights)
    q = points - mean
    design = np.column_stack((2 * q, np.ones(len(q))))
    fit, _, rank, _ = np.linalg.lstsq(weights[:, None] * design, weights * np.sum(q * q, axis=1), rcond=None)
    if rank < 4:
        return None
    initial = np.r_[fit[:3], np.sqrt(max(fit[3] + fit[:3] @ fit[:3], 1e-16))]
    opt = least_squares(lambda p: weights * (np.linalg.norm(q - p[:3], axis=1) - p[3]), initial,
                        loss='soft_l1', f_scale=.15 * initial[3], max_nfev=20)
    center, radius = mean + opt.x[:3], opt.x[3]
    radial = points - center
    lengths = np.linalg.norm(radial, axis=1)
    direction = radial / np.maximum(lengths[:, None], 1e-15)
    coverage = np.linalg.norm(np.average(direction, axis=0, weights=weights))
    return center, float(radius), float(np.quantile(abs(lengths - radius), .9)), float(coverage)


def _refine_center_with_v(center, samples, distance, h):
    if len(samples) < 8:
        return center, None
    def residuals(p):
        return np.linalg.norm(samples - center - p[:3], axis=1) + p[3] - distance
    opt = least_squares(residuals, np.zeros(4), loss='soft_l1', f_scale=.25 * h, max_nfev=25)
    conditioned = np.linalg.svd(opt.jac, compute_uv=False)
    res90 = float(np.quantile(abs(opt.fun), .9))
    if conditioned[-1] > .05 * conditioned[0] and np.linalg.norm(opt.x[:3]) < 1.2 * h and res90 < .35 * h:
        return center + opt.x[:3], res90
    return center, None


def _split_cached_v_minima(pts_box, q, d, h, level):
    """Split a merged V-shell component using local minima of cached V observations."""
    lo, hi = pts_box.min(axis=0) - h, pts_box.max(axis=0) + h
    in_box = np.all((q >= lo) & (q <= hi), axis=1) & (d <= level * 1.4)
    if in_box.sum() < 8:
        return []
    sub_q, sub_d = q[in_box], d[in_box]
    tree = cKDTree(sub_q)
    # A voxel is a local minimum if no neighbor within sqrt(3)*h*1.05 has a strictly smaller V.
    nbrs = tree.query_ball_point(sub_q, np.sqrt(3) * h * 1.05)
    minima = []
    for i, ids in enumerate(nbrs):
        if sub_d[i] <= level * 0.95 and sub_d[i] <= sub_d[ids].min() + 1e-12:
            minima.append(i)
    if not minima:
        return []
    # Merge plateau minima closer than 1.5*h by greedy lowest-V order.
    minima.sort(key=lambda i: sub_d[i])
    seeds = []
    for i in minima:
        if all(np.linalg.norm(sub_q[i] - sub_q[s]) >= 1.5 * h for s in seeds):
            seeds.append(i)
    if len(seeds) < 2:
        return []
    seed_pos = sub_q[seeds]
    # Voronoi-partition cached V points around seeds and refine each center independently.
    owner = cKDTree(seed_pos).query(sub_q)[1]
    recovered = []
    for k, pos in enumerate(seed_pos):
        mask = owner == k
        if mask.sum() < 8:
            continue
        refined, res90 = _refine_center_with_v(pos, sub_q[mask], sub_d[mask], h)
        if res90 is not None:
            recovered.append((refined, level * 0.5, res90))
    return recovered if len(recovered) >= 2 else []


def _extract_local_v_spheres(pts_box, provider, cfg, h_cur, r_cur, remaining_levels):
    """Locally refine a merged V-shell bbox at finer spacing and smaller isovalue."""
    if provider is None or remaining_levels <= 0:
        return []
    h_sub = h_cur * 0.5
    r_sub = max(1.35 * h_sub, r_cur * 0.5)
    pad = max(2.5 * h_cur, r_cur + 2 * h_sub)
    lo = pts_box.min(axis=0) - pad
    hi = pts_box.max(axis=0) + pad
    local_grid = sample_local_box(provider, lo, hi, h_sub, cfg)
    if not len(local_grid.points):
        return []
    v_shell, v_faces, _ = extract_shell(local_grid, cfg, channel=2, spacing=h_sub, level=r_sub)
    if not len(v_faces):
        return []
    pairs = v_faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    _, labels = components(pairs, len(v_shell))
    areas = np.linalg.norm(np.cross(v_shell[v_faces[:, 1]] - v_shell[v_faces[:, 0]],
                                    v_shell[v_faces[:, 2]] - v_shell[v_faces[:, 0]]), axis=1)
    raw_weights = np.bincount(v_faces.ravel(), weights=np.repeat(areas, 3), minlength=len(v_shell))
    valid = local_grid.valid[:, 2] & (local_grid.values[:, 2] <= 2 * r_sub)
    q, d = local_grid.points[valid], local_grid.values[valid, 2]
    tree = cKDTree(q) if len(q) else None
    found = []
    for label in np.unique(labels):
        ids = (labels == label) & (raw_weights > 0)
        if ids.sum() < 6:
            continue
        comp_pts = v_shell[ids]
        fit = fit_sphere(comp_pts, np.sqrt(raw_weights[ids] / raw_weights[ids].mean()))
        if fit is not None:
            center, radius, residual, coverage = fit
            if .45 * h_sub <= radius <= 2 * r_sub and residual <= .6 * h_sub and coverage <= .65:
                if tree is not None:
                    qids = tree.query_ball_point(center, 2.2 * radius)
                    if len(qids) >= 8:
                        refined, res90 = _refine_center_with_v(center, q[qids], d[qids], h_sub)
                        if res90 is not None:
                            center, residual = refined, min(residual, res90)
                found.append((center, radius, residual))
                continue
        # Still merged at this level: try cached minima then recurse finer if budget allows.
        sub = _split_cached_v_minima(comp_pts, q, d, h_sub, r_sub) if len(q) else []
        if not sub and remaining_levels > 1:
            sub = _extract_local_v_spheres(comp_pts, provider, cfg, h_sub, r_sub, remaining_levels - 1)
        found.extend(sub)
    return found


def shell_vertices(shell, faces, grid, cfg, provider=None):
    """Fit V spheres with free radius; split merged shells via cached minima or local refinement."""
    if not len(faces):
        return [], []
    pairs = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    _, labels = components(pairs, len(shell))
    areas = np.linalg.norm(np.cross(shell[faces[:, 1]] - shell[faces[:, 0]],
                                    shell[faces[:, 2]] - shell[faces[:, 0]]), axis=1)
    raw_weights = np.bincount(faces.ravel(), weights=np.repeat(areas, 3), minlength=len(shell))
    valid = grid.valid[:, 2] & (grid.values[:, 2] <= 2 * cfg.vertex_level)
    q, d = grid.points[valid], grid.values[valid, 2]
    tree = cKDTree(q) if len(q) else None
    fitted, rejected = [], []
    h = cfg.fine_spacing
    for label in np.unique(labels):
        checkpoint('shell_vertices')
        ids = (labels == label) & (raw_weights > 0)
        comp_pts = shell[ids]
        fit = fit_sphere(comp_pts, np.sqrt(raw_weights[ids] / raw_weights[ids].mean())) if ids.sum() >= 6 else None
        if fit is None:
            rejected.append(dict(reason='vertex_shell_insufficient', component=int(label)))
            continue
        center, radius, residual, coverage = fit
        if .45 * h <= radius <= 2 * cfg.vertex_level and residual <= .6 * h and coverage <= .63:
            fitted.append((center, radius, residual, True, 0.0))
            continue
        # Merged or distorted V-shell: try cached V local minima first, then local finer box.
        split = _split_cached_v_minima(comp_pts, q, d, h, cfg.vertex_level) if len(q) else []
        if not split and cfg.local_refine_levels > 0 and provider is not None:
            levels = 1 if cfg.allow_feature_collapse else cfg.local_refine_levels
            split = _extract_local_v_spheres(comp_pts, provider, cfg, h, cfg.vertex_level, levels)
        if split:
            fitted.extend((c, r, e, False, 0.0) for c, r, e in split)
        elif cfg.allow_feature_collapse:
            # Collapse an inseparable V-shell cluster into a single super-vertex whose cut ball
            # covers the entire cluster so outgoing long edges attach cleanly.
            w_comp = raw_weights[ids]
            c_super = np.average(comp_pts, axis=0, weights=w_comp)
            if len(q):
                lo_c, hi_c = comp_pts.min(axis=0) - h, comp_pts.max(axis=0) + h
                in_c = np.all((q >= lo_c) & (q <= hi_c), axis=1)
                if in_c.any():
                    best_i = int(np.argmin(d[in_c]))
                    c_cand = q[in_c][best_i]
                    refined_c, res90 = _refine_center_with_v(c_cand, q[in_c], d[in_c], h)
                    c_super = refined_c if res90 is not None else c_cand
            cluster_rad = float(np.linalg.norm(comp_pts - c_super, axis=1).max()) + 0.5 * h
            fitted.append((c_super, radius, residual, False, cluster_rad))
        else:
            rejected.append(dict(reason='vertex_shell_ambiguous', component=int(label),
                                 radius=radius, residual=residual, coverage=coverage))
    if not fitted:
        return [], rejected
    # Deduplicate any near-identical vertex candidates from overlapping splits.
    fitted.sort(key=lambda item: item[2])
    deduped = []
    min_sep = h * 0.25
    for item in fitted:
        if all(np.linalg.norm(item[0] - kept[0]) >= min_sep for kept in deduped):
            deduped.append(item)
    pos = np.asarray([f[0] for f in deduped])
    gaps = cKDTree(pos).query(pos, k=2)[0][:, 1] if len(pos) > 1 else np.full(len(pos), np.inf)
    refined = []
    for i, (center, radius, residual, needs_refine, min_cut_rad) in enumerate(deduped):
        if needs_refine and tree is not None:
            ids = tree.query_ball_point(center, min(2.5 * radius, .42 * gaps[i]))
            if len(ids) >= 8:
                samples, distance = q[ids], d[ids]
                def residuals(p):
                    return np.linalg.norm(samples - center - p[:3], axis=1) + p[3] - distance
                opt = least_squares(residuals, np.zeros(4), loss='soft_l1', f_scale=.25 * h, max_nfev=20)
                conditioned = np.linalg.svd(opt.jac, compute_uv=False)
                if (conditioned[-1] > .05 * conditioned[0] and np.linalg.norm(opt.x[:3]) < .6 * h
                        and np.quantile(abs(opt.fun), .9) < .3 * h):
                    center = center + opt.x[:3]
        refined.append((center, residual, min_cut_rad))
    pos = np.asarray([x[0] for x in refined])
    gaps = cKDTree(pos).query(pos, k=2)[0][:, 1] if len(pos) > 1 else np.full(len(pos), np.inf)
    return [VertexNode(i, c, e, min(max(2 * h, mrad), .45 * gaps[i])) for i, (c, e, mrad) in enumerate(refined)], rejected
