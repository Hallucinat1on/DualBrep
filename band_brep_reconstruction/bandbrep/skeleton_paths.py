"""Trace the thinned graph outside known vertex balls before fitting curves."""
from collections import deque
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
    refined_mask: np.ndarray | None = None
    collapsed: bool = False


def cut_at_vertices(points, links, vertices, radii_override=None):
    """Split voxel links at exact sphere intersections and keep exterior segments."""
    if not vertices or not len(links):
        return points, links, {}
    centers = np.asarray([v.position for v in vertices])
    radii = (np.asarray([v.support_radius for v in vertices])
             if radii_override is None else np.asarray(radii_override, float))
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
        if aa <= 1e-24:
            continue
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

        cursor, last = 0., a
        for lo, hi, vid in sorted(intervals):
            if lo > cursor + 1e-10:
                exterior.append((last, port(lo, vid)))
            if hi > cursor:
                cursor = hi
                if hi < 1 - 1e-10:
                    last = port(hi, vid)
        if cursor < 1 - 1e-10:
            exterior.append((last, b))
    return np.asarray(nodes), np.asarray(exterior, dtype=int).reshape(-1, 2), ports


def _prune_spurs(adjacency, ports, max_steps=3):
    """Remove short dead-end thinning hairs that do not touch any vertex port."""
    for _ in range(max_steps):
        removed = False
        for i, nbrs in enumerate(adjacency):
            if len(nbrs) == 1 and i not in ports:
                j = nbrs[0]
                if len(adjacency[j]) >= 2:
                    adjacency[i].clear()
                    adjacency[j].remove(i)
                    removed = True
        if not removed:
            break


def _expand_cut_radii_for_acute_junctions(points, links, vertices, shell_radius, spacing):
    """Locally expand cut balls when acute/tangent tubes merge just outside 2*h."""
    if not vertices or not len(links):
        return None
    centers = np.asarray([v.position for v in vertices])
    base_radii = np.asarray([v.support_radius for v in vertices], float)
    gaps = cKDTree(centers).query(centers, k=2)[0][:, 1] if len(centers) > 1 else np.full(len(centers), np.inf)
    max_radii = np.minimum(0.45 * gaps, max(3.2 * shell_radius, 0.016))
    radii = base_radii.copy()
    tree = cKDTree(centers)
    for _ in range(2):
        cut_pts, ext, ports = cut_at_vertices(points, links, vertices, radii)
        if not len(ext):
            break
        adj = [[] for _ in cut_pts]
        for u, v in ext:
            adj[int(u)].append(int(v))
            adj[int(v)].append(int(u))
        _prune_spurs(adj, ports, max_steps=2)
        hi_nodes = [i for i, nbrs in enumerate(adj) if len(nbrs) > 2]
        if not hi_nodes:
            break
        changed = False
        d_near, v_near = tree.query(cut_pts[hi_nodes])
        for dist, vid in zip(d_near, v_near):
            vid = int(vid)
            needed = float(dist) + 0.6 * spacing
            if radii[vid] < needed <= max_radii[vid]:
                radii[vid] = needed
                changed = True
        if not changed:
            break
    return radii


def _extract_chain(start, first_next, adjacency):
    ordered = [start, first_next]
    prev, node = start, first_next
    while len(adjacency[node]) == 2:
        nxt = adjacency[node][0] if adjacency[node][1] == prev else adjacency[node][1]
        if nxt == start:
            ordered.append(nxt)
            break
        ordered.append(nxt)
        prev, node = node, nxt
    return ordered


def _flatten_narrow_bubbles(cut_points, adjacency, ports, max_width):
    """Collapse narrow parallel 2-chain bubbles between deg>2 junctions into a single chain."""
    changed = True
    while changed:
        changed = False
        key_nodes = {i for i, nbrs in enumerate(adjacency) if len(nbrs) != 2 or i in ports}
        chains = []
        visited = set()
        for u in key_nodes:
            for nxt in list(adjacency[u]):
                edge_key = (u, nxt)
                if edge_key in visited:
                    continue
                chain = _extract_chain(u, nxt, adjacency)
                visited.add((chain[0], chain[1]))
                visited.add((chain[-1], chain[-2]))
                chains.append(chain)
        by_ends = {}
        for ch in chains:
            key = (min(ch[0], ch[-1]), max(ch[0], ch[-1]))
            if key[0] != key[1]:
                by_ends.setdefault(key, []).append(ch)
        for key, group in by_ends.items():
            if len(group) < 2:
                continue
            # Keep the shortest chain and remove any parallel duplicate within max_width.
            group.sort(key=len)
            base = group[0]
            base_tree = cKDTree(cut_points[base])
            for dup in group[1:]:
                if base_tree.query(cut_points[dup])[0].max() <= max_width:
                    for a, b in zip(dup, dup[1:]):
                        if b in adjacency[a]:
                            adjacency[a].remove(b)
                        if a in adjacency[b]:
                            adjacency[b].remove(a)
                    changed = True
                    break
            if changed:
                break


def trace_paths(points, links, vertices, *, shell_radius=0.005, spacing=0.003,
                allow_collapse=False, collapse_tolerance=0.005):
    radii = _expand_cut_radii_for_acute_junctions(points, links, vertices, shell_radius, spacing)
    cut_points, exterior, ports = cut_at_vertices(points, links, vertices, radii)
    if not len(exterior):
        return [], []
    adjacency = [[] for _ in cut_points]
    for u, v in exterior:
        adjacency[int(u)].append(int(v))
        adjacency[int(v)].append(int(u))
    _prune_spurs(adjacency, ports, max_steps=3)
    if allow_collapse:
        _flatten_narrow_bubbles(cut_points, adjacency, ports, max(collapse_tolerance * 1.5, 2 * shell_radius))
        _prune_spurs(adjacency, ports, max_steps=3)
    active_edges = [(u, v) for u, nbrs in enumerate(adjacency) for v in nbrs if u < v]
    if not active_edges:
        return [], []
    exterior_clean = np.asarray(active_edges, dtype=int).reshape(-1, 2)
    count, labels = components(exterior_clean, len(cut_points))
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
        junctions = [i for i in group if len(adjacency[i]) > 2]
        if not ends and not contacts and not junctions:
            start = group[0]
            ordered = _extract_chain(start, min(adjacency[start]), adjacency)
            positions = cut_points[ordered]
            paths.append(Path([], positions, True, None))
            continue
        if len(ends) == 2 and set(ends) == set(contacts) and not junctions:
            start = ends[0]
            ordered = _extract_chain(start, adjacency[start][0], adjacency)
            ids = [ports[ordered[0]], ports[ordered[-1]]]
            positions = cut_points[ordered]
            paths.append(Path(ids, positions, False, positions[[0, -1]].tolist()))
            continue
        if junctions:
            gpts = cut_points[group]
            unresolved.append(dict(
                reason='skeleton_junction_unresolved',
                vertices=sorted({ports[i] for i in contacts}),
                points=len(group),
                position=gpts.mean(axis=0),
                bbox=[gpts.min(axis=0).tolist(), gpts.max(axis=0).tolist()],
                _support_points=gpts,
                _junction_points=cut_points[junctions],
            ))
        else:
            gpts = cut_points[group]
            unresolved.append(dict(
                reason='skeleton_end_unattached',
                vertices=sorted({ports[i] for i in contacts}),
                points=len(group),
                position=gpts.mean(axis=0),
                bbox=[gpts.min(axis=0).tolist(), gpts.max(axis=0).tolist()],
            ))
    return paths, unresolved


def recover_cut_ball_short_paths(points, links, vertices, existing_paths, spacing):
    """Recover short edges whose entire skeleton segment lay inside two vertex cut balls."""
    if len(vertices) < 2 or not len(links):
        return []
    connected = {tuple(sorted(p.nodes)) for p in existing_paths if len(p.nodes) == 2}
    centers = np.asarray([v.position for v in vertices])
    radii = np.asarray([v.support_radius for v in vertices])
    adj = [[] for _ in range(len(points))]
    for u, v in links:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    skel_tree = cKDTree(points)
    v_tree = cKDTree(centers)
    max_search = float(2 * radii.max() + 3 * spacing)
    added = []
    for a, b in sorted(v_tree.query_pairs(max_search)):
        if (a, b) in connected:
            continue
        p0, p1 = centers[a], centers[b]
        dist = float(np.linalg.norm(p1 - p0))
        if dist > radii[a] + radii[b] + 2.5 * spacing or dist <= 1e-10:
            continue
        # Find skeleton nodes inside ball a and ball b.
        in_a = set(skel_tree.query_ball_point(p0, max(radii[a], 1.5 * spacing)))
        in_b = set(skel_tree.query_ball_point(p1, max(radii[b], 1.5 * spacing)))
        if not in_a or not in_b:
            continue
        # Restrict BFS to corridor around segment a-b and avoid any third vertex ball.
        corridor = set(skel_tree.query_ball_point(0.5 * (p0 + p1), 0.5 * dist + 2 * spacing))
        blocked = set()
        for c_id in v_tree.query_ball_point(0.5 * (p0 + p1), 0.5 * dist + radii.max() + spacing):
            if c_id not in (a, b):
                blocked.update(skel_tree.query_ball_point(centers[c_id], radii[c_id]))
        allowed = (corridor - blocked) | in_a | in_b
        # Shortest path in skeleton links from in_a to in_b.
        q = deque(in_a)
        parent = {u: -1 for u in in_a}
        hit = None
        while q:
            u = q.popleft()
            if u in in_b:
                hit = u
                break
            for w in adj[u]:
                if w in allowed and w not in parent:
                    parent[w] = u
                    q.append(w)
        if hit is None:
            continue
        chain = []
        cur = hit
        while cur != -1:
            chain.append(cur)
            cur = parent[cur]
        chain = chain[::-1]
        pts = points[chain]
        if len(pts) < 2:
            pts = np.vstack((p0, 0.5 * (p0 + p1), p1))
        else:
            pts = np.vstack((p0, pts, p1))
        added.append(Path([a, b], pts, False))
        connected.add((a, b))
    return added


def refine_voxel_centers(paths, grid, cfg, *, spacing=None, level=None):
    """Fit a local transverse center to cached E squared, tracking refined confidence."""
    h = cfg.fine_spacing if spacing is None else float(spacing)
    r = cfg.shell_radius if level is None else float(level)
    valid = grid.valid[:, 1] & (grid.values[:, 1] <= r + h)
    observations = grid.points[valid]
    if not len(observations):
        for path in paths:
            path.refined_mask = np.zeros(len(path.points), dtype=bool)
        return
    distances = grid.values[valid, 1] / h
    tree = cKDTree(observations)
    query_radius = min(2.2 * h, r + h)
    for path in paths:
        checkpoint('skeleton_center_refinement')
        points = path.points[:-1] if path.closed else path.points
        if len(points) < 3:
            path.refined_mask = np.zeros(len(path.points), dtype=bool)
            continue
        smooth = gaussian_filter1d(points, 2., axis=0, mode='wrap' if path.closed else 'nearest')
        if path.closed:
            tangent = np.roll(smooth, -1, axis=0) - np.roll(smooth, 1, axis=0)
        else:
            tangent = np.gradient(smooth, axis=0)
        neighborhoods = tree.query_ball_point(points, query_radius)
        refined = points.copy()
        moved = np.zeros(len(points), dtype=bool)
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
            for _ in range(3):
                fit, _, rank, _ = np.linalg.lstsq(design * weight[:, None], target * weight, rcond=None)
                res = design @ fit - target
                weight = weight / np.sqrt(1 + (res * 2.0) ** 2)
            if rank < 3 or np.linalg.norm(fit[:2]) > 1.0:
                continue
            radial = uv - fit[:2]
            covariance = radial.T @ radial
            eigen = np.linalg.eigvalsh(covariance)
            if eigen[0] < .12 * eigen[1] or np.quantile(abs(design @ fit - target), .75) > 1.2:
                continue
            refined[i] += h * (fit[0] * x + fit[1] * y)
            moved[i] = True
        if path.closed:
            path.points = np.vstack((refined, refined[0]))
            path.refined_mask = np.r_[moved, moved[0]]
        else:
            length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            margin = min(2 * h, .25 * length[-1])
            core = np.minimum(length, length[-1] - length) >= margin
            if core.sum() >= 4:
                path.points = refined[core]
                path.refined_mask = moved[core]
            else:
                path.points = refined
                path.refined_mask = moved
