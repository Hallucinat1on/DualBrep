"""Trace the thinned graph outside known vertex balls before fitting curves."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from .deadline import checkpoint
from .surface_fitting import frame
from .surface_mesh import components


@dataclass
class Path:
    nodes: list
    points: np.ndarray
    closed: bool
    ports: list | None = None


def cut_at_vertices(points, links, vertices):
    """Split voxel links at exact sphere intersections and keep exterior segments."""
    if not vertices or not len(links):
        return points, links, {}
    centers = np.asarray([v.position for v in vertices])
    radii = np.asarray([v.support_radius for v in vertices])
    tree = cKDTree(centers)
    start, stop = points[links].transpose(1, 0, 2)
    directions = stop - start
    nearby = tree.query_ball_point((start + stop) / 2,
                                   radii.max() + np.linalg.norm(directions, axis=1) / 2)
    nodes, exterior, ports = list(points), [], {}
    for row, candidates in enumerate(nearby):
        if row % 4096 == 0:
            checkpoint('skeleton_vertex_ports')
        a, b = (int(i) for i in links[row])
        p, d = start[row], directions[row]
        aa = d @ d
        intervals = []
        for vid in candidates:
            offset = p - centers[vid]
            bb = offset @ d
            discriminant = bb * bb - aa * (offset @ offset - radii[vid] ** 2)
            if discriminant <= 1e-24:
                continue
            root = np.sqrt(discriminant)
            lo, hi = (-bb - root) / aa, (-bb + root) / aa
            if max(0., lo) < min(1., hi):
                if abs(lo) <= 1e-10:
                    ports[a] = int(vid)
                if abs(hi - 1) <= 1e-10:
                    ports[b] = int(vid)
                intervals.append((max(0., lo), min(1., hi), vid))

        def port(t, vid):
            if t <= 1e-10:
                node = a
            elif t >= 1 - 1e-10:
                node = b
            else:
                node = len(nodes)
                nodes.append(p + t * d)
            ports[node] = int(vid)
            return node

        # Fitted cut balls use at most 0.4 of the nearest-vertex distance,
        # so their intervals along a link cannot overlap.
        cursor, last = 0., a
        for lo, hi, vid in sorted(intervals):
            if lo > cursor + 1e-10:
                exterior.append((last, port(lo, vid)))
            cursor = hi
            if hi < 1 - 1e-10:
                last = port(hi, vid)
        if cursor < 1 - 1e-10:
            exterior.append((last, b))
    return np.asarray(nodes), np.asarray(exterior, dtype=int).reshape(-1, 2), ports


def trace_paths(points, links, vertices):
    cut_points, exterior, ports = cut_at_vertices(points, links, vertices)
    if not len(exterior):
        return [], []
    count, labels = components(exterior, len(cut_points))
    adjacency = [[] for _ in cut_points]
    for u, v in exterior:
        adjacency[u].append(v)
        adjacency[v].append(u)
    groups = [[] for _ in range(count)]
    for i, neighbors in enumerate(adjacency):
        if neighbors:
            groups[labels[i]].append(i)
    paths, unresolved = [], []
    for group in groups:
        if not group:
            continue
        ends = [i for i in group if len(adjacency[i]) == 1]
        contacts = [i for i in group if i in ports]
        if not ends and not contacts and all(len(adjacency[i]) == 2 for i in group):
            closed = True
            start = group[0]
        elif len(ends) == 2 and set(ends) == set(contacts) and all(len(adjacency[i]) <= 2 for i in group):
            closed = False
            start = ends[0]
        else:
            reason = ('skeleton_junction_unresolved' if any(len(adjacency[i]) > 2 for i in group)
                      else 'skeleton_end_unattached')
            unresolved.append(dict(reason=reason, vertices=sorted({ports[i] for i in contacts}),
                                   points=len(group), position=cut_points[group].mean(axis=0)))
            continue
        ordered = [start]
        prev, node = -1, start
        for _ in range(len(group)):
            choices = [j for j in adjacency[node] if j != prev]
            if not choices:
                break
            prev, node = node, min(choices)
            ordered.append(node)
            if node == start:
                break
        ids = [] if closed else [ports[ordered[0]], ports[ordered[-1]]]
        positions = cut_points[ordered]
        paths.append(Path(ids, positions, closed, None if closed else positions[[0, -1]].tolist()))
    return paths, unresolved


def refine_voxel_centers(paths, grid, cfg):
    """Fit a local transverse center to cached E squared, keeping graph incidence fixed."""
    h = cfg.fine_spacing
    valid = grid.valid[:, 1] & (grid.values[:, 1] <= cfg.shell_radius + 2 * h)
    observations = grid.points[valid]
    if not len(observations):
        return
    distances = grid.values[valid, 1] / h
    tree = cKDTree(observations)
    for path in paths:
        checkpoint('skeleton_center_refinement')
        points = path.points[:-1] if path.closed else path.points
        if len(points) < 3:
            continue
        smooth = gaussian_filter1d(points, 2., axis=0, mode='wrap' if path.closed else 'nearest')
        if path.closed:
            tangent = np.roll(smooth, -1, axis=0) - np.roll(smooth, 1, axis=0)
        else:
            tangent = np.gradient(smooth, axis=0)
        neighborhoods = tree.query_ball_point(points, 3 * h)
        refined = points.copy()
        for i, ids in enumerate(neighborhoods):
            if len(ids) < 8 or np.linalg.norm(tangent[i]) < 1e-12:
                continue
            axis = tangent[i] / np.linalg.norm(tangent[i])
            sample = (observations[ids] - points[i]) / h
            use = np.abs(sample @ axis) <= 1.25
            if use.sum() < 8:
                continue
            x, y = frame(axis)
            uv = sample[use] @ np.stack((x, y), axis=1)
            value = distances[np.asarray(ids)[use]]
            design = np.column_stack((2 * uv, np.ones(len(uv))))
            target = np.einsum('ij,ij->i', uv, uv) - value ** 2
            weight = 1 / (1 + value ** 2)
            for _ in range(2):
                fit, _, rank, _ = np.linalg.lstsq(design * weight[:, None], target * weight, rcond=None)
                weight = weight / np.sqrt(1 + (design @ fit - target) ** 2)
            if rank < 3 or np.linalg.norm(fit[:2]) > 1.0:
                continue
            radial = uv - fit[:2]
            covariance = radial.T @ radial
            eigen = np.linalg.eigvalsh(covariance)
            if eigen[0] < .15 * eigen[1] or np.quantile(abs(design @ fit - target), .75) > 1.:
                continue
            refined[i] += h * (fit[0] * x + fit[1] * y)
        if path.closed:
            path.points = np.vstack((refined, refined[0]))
        else:
            # The ports establish incidence. The first voxels beyond a cut
            # ball can still be influenced by the other arms of the junction.
            length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            margin = min(2 * h, .25 * length[-1])
            core = np.minimum(length, length[-1] - length) >= margin
            path.points = refined[core] if core.sum() >= 4 else refined
