"""Embed curve barriers in a surface copy, then recover oriented face uses."""
from collections import defaultdict
import heapq

import numpy as np
from shapely import LineString, constrained_delaunay_triangles
from shapely.ops import polygonize, unary_union

from .boundary_cycles import complete_face_regions_and_cycles
from .deadline import checkpoint
from .schemas import SurfaceMesh, require
from .surface_mesh import components, mesh_edges, normals as mesh_normals


EPS = 2e-9
SNAP = 1e-5


def _edge_point(a, b, c, d):
    """Find a point on mesh edge a-b near path segment c-d."""
    edge, path = b - a, d - c
    u, t = np.linalg.lstsq(np.column_stack((edge, -path)), c - a, rcond=None)[0]
    if t < 0 or t > 1:
        u = (c + np.clip(t, 0, 1) * path - a) @ edge / max(edge @ edge, 1e-30)
    p = a + np.clip(u, 0, 1) * edge
    t = np.clip((p - c) @ path / max(path @ path, 1e-30), 0, 1)
    return p, np.linalg.norm(p - (c + t * path))


def _simplify_segments(segs):
    """Remove immediate backtracking spikes (u -> v -> u) from a routed curve."""
    stack = []
    for tri, a, b in segs:
        if a == b:
            continue
        if stack and stack[-1][1] == b and stack[-1][2] == a:
            stack.pop()
            continue
        stack.append((tri, a, b))
    # Also collapse short 2-step micro-backtracks (u -> v -> w -> v -> u) if any remain.
    changed = True
    while changed and len(stack) >= 2:
        changed = False
        cleaned = []
        for item in stack:
            if cleaned and cleaned[-1][1] == item[2] and cleaned[-1][2] == item[1]:
                cleaned.pop()
                changed = True
            else:
                cleaned.append(item)
        stack = cleaned
    return stack


class SurfaceCut:
    def __init__(self, mesh, oracle, tolerance):
        self.mesh, self.oracle, self.tolerance = mesh, oracle, tolerance
        self.points = list(mesh.vertices)
        self.point_ids = {self.key(p): i for i, p in enumerate(mesh.vertices)}
        self.tri_normals = mesh_normals(mesh)
        edges, inverse, counts = mesh_edges(mesh.triangles)
        self.face_edges = edges[inverse]
        self.edge_faces = defaultdict(list)
        self.vertex_faces = defaultdict(set)
        for fid, triangle in enumerate(mesh.triangles):
            for node in triangle:
                self.vertex_faces[int(node)].add(fid)
            for edge in self.face_edges[fid]:
                self.edge_faces[tuple(edge)].append(fid)
        require(np.all(counts == 2), 'cut_mesh_nonmanifold', 'The surface cut requires a closed input')
        self.neighbors = [[] for _ in mesh.triangles]
        for edge, (a, b) in self.edge_faces.items():
            self.neighbors[a].append((b, edge))
            self.neighbors[b].append((a, edge))
        self.edge_nodes = defaultdict(set)
        self.cuts = defaultdict(list)
        self.barriers = {}
        self.coarse_barriers = {}
        self.claimed_interior = set()

    @staticmethod
    def key(point):
        return tuple(np.round(point / EPS).astype(np.int64))

    def point(self, point):
        key = self.key(point)
        if key not in self.point_ids:
            self.point_ids[key] = len(self.points)
            self.points.append(np.asarray(point))
        return self.point_ids[key]

    def on_edge(self, edge, p):
        a, b = self.mesh.vertices[list(edge)]
        d = b - a
        t = np.clip((p - a) @ d / max(d @ d, 1e-30), 0, 1)
        return a + t * d

    def edge_node(self, edge, point, *, avoid_claimed=False):
        p = self.on_edge(edge, point)
        a, b = self.mesh.vertices[list(edge)]
        edge_vec = b - a
        edge_len = float(np.linalg.norm(edge_vec))
        snap = max(EPS * 2, min(SNAP, 0.05 * edge_len))
        if avoid_claimed and edge_len > EPS * 10:
            # Physically nudge away from claimed interior endpoints AND claimed Steiner nodes
            # on this edge so self.point(p) cannot hash onto another edge's interior node.
            nudge_dist = min(0.05 * edge_len, max(SNAP * 3, EPS * 10))
            if int(edge[0]) in self.claimed_interior and np.linalg.norm(p - a) <= snap * 2:
                p = a + (nudge_dist / edge_len) * edge_vec
            elif int(edge[1]) in self.claimed_interior and np.linalg.norm(p - b) <= snap * 2:
                p = b - (nudge_dist / edge_len) * edge_vec
            for node in list(self.edge_nodes[edge]):
                if node in self.claimed_interior and np.linalg.norm(p - self.points[node]) <= snap * 2:
                    # Nudge toward the farther endpoint of this mesh edge.
                    target = b if np.linalg.norm(b - p) >= np.linalg.norm(a - p) else a
                    step_dir = target - p
                    step_len = float(np.linalg.norm(step_dir))
                    if step_len > EPS * 10:
                        p = p + min(0.25, nudge_dist / step_len) * step_dir
        for node in edge:
            if avoid_claimed and int(node) in self.claimed_interior:
                continue
            if np.linalg.norm(p - self.points[node]) <= snap:
                return int(node)
        for node in self.edge_nodes[edge]:
            if avoid_claimed and node in self.claimed_interior:
                continue
            if np.linalg.norm(p - self.points[node]) <= snap:
                return node
        node = self.point(p)
        self.edge_nodes[edge].add(node)
        return node

    def locate(self, triangle, point, *, is_endpoint=False):
        tri = self.mesh.triangles[triangle]
        verts = self.mesh.vertices[tri]
        min_edge = float(min(np.linalg.norm(verts[1] - verts[0]),
                             np.linalg.norm(verts[2] - verts[1]),
                             np.linalg.norm(verts[0] - verts[2])))
        snap = max(EPS * 2, min(SNAP, 0.05 * min_edge))
        for node in tri:
            if not is_endpoint and int(node) in self.claimed_interior:
                continue
            if np.linalg.norm(self.points[node] - point) <= snap:
                return int(node), self.vertex_faces[int(node)]
        best_edge = None
        best_dist = snap
        for edge in map(tuple, self.face_edges[triangle]):
            dist = float(np.linalg.norm(self.on_edge(edge, point) - point))
            if dist <= best_dist:
                best_dist = dist
                best_edge = edge
        if best_edge is not None:
            return self.edge_node(best_edge, point, avoid_claimed=not is_endpoint), set(self.edge_faces[best_edge])
        p = np.asarray(point, float)
        if not is_endpoint:
            existing = self.point_ids.get(self.key(p))
            if existing is not None and existing in self.claimed_interior:
                centroid = verts.mean(axis=0)
                delta = centroid - p
                dlen = float(np.linalg.norm(delta))
                if dlen > EPS * 10:
                    p = p + min(0.1, max(SNAP * 3, EPS * 10) / dlen) * delta
        return self.point(p), {int(triangle)}

    def route(self, src, dst, starts, ends):
        """Route through triangles in the local continuous segment corridor."""
        same = starts & ends
        if same:
            return [(min(same), src, dst)]
        p, q = self.points[src], self.points[dst]
        corridor = min(self.tolerance, 1.5 * np.linalg.norm(p - q) + EPS)
        costs = dict.fromkeys(starts, 0.)
        previous = {}
        queue = [(0., t) for t in sorted(starts)]
        heapq.heapify(queue)
        last = None
        while queue:
            value, triangle = heapq.heappop(queue)
            if value != costs[triangle]:
                continue
            if triangle in ends:
                last = triangle
                break
            n_cur = self.tri_normals[triangle]
            for neighbor, edge in self.neighbors[triangle]:
                a, b = self.mesh.vertices[list(edge)]
                mid, error = _edge_point(a, b, p, q)
                if error > corridor:
                    continue
                # Mild penalty against flipping onto opposite-facing thin-wall sheets,
                # and strong penalty against pinching onto another edge's interior mesh vertex.
                n_dot = float(n_cur @ self.tri_normals[neighbor])
                flip_penalty = corridor * 0.15 * max(0.0, -n_dot)
                pinch_penalty = 0.0
                if self.claimed_interior:
                    for v_end in edge:
                        if int(v_end) in self.claimed_interior and int(v_end) not in (src, dst):
                            if np.linalg.norm(mid - self.points[int(v_end)]) <= max(SNAP * 2, EPS * 4):
                                pinch_penalty = corridor * 0.8
                                break
                cost = value + error + corridor * .05 + flip_penalty + pinch_penalty
                if cost < costs.get(neighbor, np.inf):
                    costs[neighbor] = cost
                    previous[neighbor] = (triangle, edge, mid)
                    heapq.heappush(queue, (cost, neighbor))
        require(last is not None, 'curve_embedding_disconnected', 'Projected curve jumps across surface sheets')
        parts = []
        while last in previous:
            before, edge, mid = previous[last]
            node = self.edge_node(edge, mid, avoid_claimed=True)
            parts.append((last, node, dst))
            last, dst = before, node
        return [(last, src, dst)] + parts[::-1]

    def embed(self, graph, curves, spacing, coarse_paths):
        mesh_anchors = {}
        for edge in graph.edges:
            if edge.id in coarse_paths:
                path = coarse_paths[edge.id]
                mesh_anchors.update(zip(edge.vertices, (path[0], path[-1])))
                interior = path[1:-1] if edge.vertices else path[:-1]
                self.claimed_interior.update(int(n) for n in interior)
                for u, v in zip(path, path[1:]):
                    self.coarse_barriers[(min(u, v), max(u, v))] = (edge.id, u, v)
        shared = {vertex: (node, self.vertex_faces[node]) for vertex, node in mesh_anchors.items()}
        for edge in graph.edges:
            if edge.id in coarse_paths:
                continue
            checkpoint('edge_first_embedding')
            curve = curves[edge.id]
            coarse = curve.evaluate(np.linspace(0, 1, 129))
            lengths = np.linalg.norm(np.diff(coarse, axis=0), axis=1)
            total_len = float(lengths.sum())
            count = max(17, int(np.ceil(total_len / spacing)) + 1)
            q = curve.evaluate(np.linspace(0, 1, count))
            if edge.vertices and total_len > 4.0 * spacing:
                # Skip interior samples cramped right next to shared vertices so acute-angle
                # incident edges enter distinct mesh sectors cleanly.
                keep_q = np.ones(len(q), dtype=bool)
                keep_q[1:-1] = ((np.linalg.norm(q[1:-1] - q[0], axis=1) >= 1.5 * spacing)
                                & (np.linalg.norm(q[1:-1] - q[-1], axis=1) >= 1.5 * spacing))
                if keep_q.sum() >= 3:
                    q = q[keep_q]
            sq, tids, projected = self.oracle.closest(q)
            require(sq.max() <= self.tolerance ** 2, 'curve_embedding_offset',
                    'Curve is too far from the input surface', edge=edge.id, offset=float(np.sqrt(sq.max())))
            endpoints = dict(zip((0, len(q) - 1), edge.vertices))
            anchors = []
            for i, (f, p) in enumerate(zip(tids, projected)):
                vertex = endpoints.get(i)
                if vertex in mesh_anchors:
                    anchor = shared[vertex]
                    require(np.linalg.norm(self.points[anchor[0]] - p) <= self.tolerance,
                            'curve_embedding_vertex_conflict', 'Coarse endpoint and exact projection differ',
                            vertex=vertex, edge=edge.id)
                    anchors.append(anchor)
                else:
                    anchors.append(self.locate(f, p, is_endpoint=(vertex is not None)))
            if edge.vertices:
                for index, vertex in zip((0, -1), edge.vertices):
                    anchor = shared.setdefault(vertex, anchors[index])
                    require(np.linalg.norm(self.points[anchor[0]] - self.points[anchors[index][0]]) <= max(SNAP * 4, EPS * 4),
                            'curve_embedding_vertex_conflict', 'Uses of a vertex project to different sheets', vertex=vertex)
                    anchors[index] = anchor
            elif edge.closed:
                anchors[-1] = anchors[0]
            edge_segs = []
            for (src, starts), (dst, ends) in zip(anchors, anchors[1:]):
                if src != dst:
                    edge_segs.extend(self.route(src, dst, starts, ends))
            simplified = _simplify_segments(edge_segs)
            end_ids = {anchors[0][0], anchors[-1][0]} if edge.vertices else set()
            for triangle, a, b in simplified:
                if a != b:
                    self.cuts[triangle].append((a, b, edge.id))
                    if a not in end_ids:
                        self.claimed_interior.add(a)
                    if b not in end_ids:
                        self.claimed_interior.add(b)

    def _expanded(self, fid, a, b):
        p, q = self.points[a], self.points[b]
        for edge in map(tuple, self.face_edges[fid]):
            if max(np.linalg.norm(self.on_edge(edge, x) - x) for x in (p, q)) <= EPS * 4:
                delta = q - p
                if delta @ delta <= 1e-24:
                    return [a, b]
                nodes = [(0., a), (1., b)]
                for node in self.edge_nodes[edge] - {a, b}:
                    t = (self.points[node] - p) @ delta / (delta @ delta)
                    if 0 < t < 1:
                        nodes.append((t, node))
                return [node for _, node in sorted(nodes)]
        return [a, b]

    def add_barrier(self, u, v, eid):
        key = (min(u, v), max(u, v))
        old = self.barriers.setdefault(key, (eid, u, v))
        require(old == (eid, u, v), 'curve_embedding_overlap',
                'Barriers overlap or backtrack on the surface', edge=eid, other_edge=old[0])

    def triangulate(self):
        for key, (eid, u, v) in self.coarse_barriers.items():
            face = self.edge_faces[key][0]
            nodes = self._expanded(face, u, v)
            for a, b in zip(nodes, nodes[1:]):
                self.add_barrier(a, b, eid)
        triangles = []
        for fid, face in enumerate(self.mesh.triangles):
            if fid % 512 == 0:
                checkpoint('edge_first_cut_triangles')
            if not self.cuts[fid] and not any(tuple(e) in self.edge_nodes for e in self.face_edges[fid]):
                triangles.append(face)
                continue
            a, b, c = self.mesh.vertices[face]
            axis = (b - a) / np.linalg.norm(b - a)
            normal = np.cross(axis, c - a)
            basis = np.vstack((axis, np.cross(normal / np.linalg.norm(normal), axis)))
            uv_to_id, uv_list, segments = {}, [], set()
            tri_barriers = []

            def uv(node):
                result = tuple((self.points[node] - a) @ basis.T)
                if result not in uv_to_id:
                    uv_to_id[result] = node
                    uv_list.append((result, node))
                return result

            def resolve_uv(pt):
                key = (float(pt[0]), float(pt[1]))
                nid = uv_to_id.get(key)
                if nid is not None:
                    return nid
                best_nid, best_d2 = None, 1e-22
                for coord, cid in uv_list:
                    d2 = (coord[0] - key[0]) ** 2 + (coord[1] - key[1]) ** 2
                    if d2 < best_d2:
                        best_d2, best_nid = d2, cid
                if best_nid is not None:
                    uv_to_id[key] = best_nid
                    return best_nid
                # Lift interior intersection point back to 3D on the triangle plane.
                p3 = a + key[0] * basis[0] + key[1] * basis[1]
                nid = self.point(p3)
                uv_to_id[key] = nid
                uv_list.append((key, nid))
                return nid

            def add_segments(nodes, eid=None):
                for u, v in zip(nodes, nodes[1:]):
                    key = tuple(sorted((u, v)))
                    segments.add(key)
                    if eid is not None:
                        tri_barriers.append((u, v, eid))

            for u, v in zip(face, np.roll(face, -1)):
                add_segments(self._expanded(fid, u, v))
            for u, v, eid in self.cuts[fid]:
                add_segments(self._expanded(fid, u, v), eid)

            lines = [LineString((uv(u), uv(v))) for u, v in segments]
            # Check if any cut segments cross inside this triangle; if so, node them with unary_union.
            has_crossing = False
            if len(tri_barriers) >= 2:
                cut_lines = [LineString((uv(u), uv(v))) for u, v, _ in tri_barriers]
                for i_c in range(len(cut_lines)):
                    for j_c in range(i_c + 1, len(cut_lines)):
                        if cut_lines[i_c].crosses(cut_lines[j_c]):
                            has_crossing = True
                            break
                    if has_crossing:
                        break

            if has_crossing:
                noded = unary_union(lines)
                noded_lines = list(noded.geoms) if hasattr(noded, 'geoms') else [noded]
                noded_pts = []
                for ln in noded_lines:
                    if ln.geom_type != 'LineString':
                        continue
                    for coord in ln.coords:
                        nid = resolve_uv(coord)
                        noded_pts.append((uv(nid), nid))
                # Split each barrier segment along any interior noded vertices lying on it.
                for u, v, eid in tri_barriers:
                    pu, pv = np.asarray(uv(u)), np.asarray(uv(v))
                    seg_vec = pv - pu
                    seg_len2 = float(seg_vec @ seg_vec)
                    if seg_len2 <= 1e-24:
                        continue
                    on_seg = [(0.0, u), (1.0, v)]
                    seen_ids = {u, v}
                    for coord, nid in noded_pts:
                        if nid in seen_ids:
                            continue
                        dvec = np.asarray(coord) - pu
                        t = float(dvec @ seg_vec) / seg_len2
                        if 1e-9 < t < 1.0 - 1e-9:
                            perp = np.linalg.norm(dvec - t * seg_vec)
                            if perp <= max(1e-10, np.sqrt(seg_len2) * 1e-6):
                                seen_ids.add(nid)
                                on_seg.append((t, nid))
                    on_seg.sort(key=lambda item: item[0])
                    for (_, w0), (_, w1) in zip(on_seg, on_seg[1:]):
                        self.add_barrier(w0, w1, eid)
                polygons = list(polygonize(noded))
            else:
                for u, v, eid in tri_barriers:
                    self.add_barrier(u, v, eid)
                polygons = list(polygonize(lines))

            area = np.linalg.norm(normal) * np.linalg.norm(b - a) / 2
            require(abs(sum(p.area for p in polygons) - area) <= max(1e-14, area * 1e-6),
                    'curve_embedding_crossing', 'Local cuts do not partition the original triangle', triangle=fid)
            for polygon in polygons:
                for part in constrained_delaunay_triangles(polygon).geoms:
                    coordinates = np.asarray(part.exterior.coords)[:3]
                    u, v = coordinates[1:] - coordinates[0]
                    if u[0] * v[1] - u[1] * v[0] < 0:
                        coordinates = coordinates[[0, 2, 1]]
                    triangles.append([resolve_uv(p) for p in coordinates])
        return SurfaceMesh(np.asarray(self.points), np.array(triangles), dict(self.mesh.report))


def label_barriers(triangles, graph, barriers, edge_index=None):
    """Group original or cut triangles and assign the oriented sides of each barrier."""
    edges, inverse, counts = mesh_edges(triangles) if edge_index is None else edge_index
    require(np.all(counts == 2), 'cut_mesh_nonmanifold',
            'Each mesh edge must have two incident triangles')
    halfedges = np.argsort(inverse.ravel(), kind='stable').reshape(-1, 2)
    starts = triangles.ravel()[halfedges]
    require(np.all(starts[:, 0] != starts[:, 1]), 'cut_mesh_nonmanifold',
            'Shared edges must have opposite triangle directions')
    adjacent = halfedges // 3

    keys = np.asarray(list(barriers), dtype=np.int64).reshape(-1, 2)
    base = int(triangles.max()) + 1
    edge_keys = edges[:, 0] * base + edges[:, 1]
    barrier_keys = keys[:, 0] * base + keys[:, 1]
    selected = np.searchsorted(edge_keys, barrier_keys)
    require(np.all(selected < len(edges)) and np.array_equal(edges[selected], keys),
            'curve_embedding_missing_cut', 'Constraint was not embedded into the mesh')
    allowed = np.ones(len(edges), dtype=bool)
    allowed[selected] = False
    graph.face_count, labels = components(adjacent[allowed], len(triangles))

    sides = defaultdict(set)
    for index, (eid, start, _) in zip(selected, barriers.values()):
        left, right = labels[adjacent[index]]
        if starts[index, 0] != start:
            left, right = right, left
        sides[eid].add((int(left), int(right)))
    for edge in graph.edges:
        pair = sides[edge.id]
        require(len(pair) == 1 and all(a != b for a, b in pair), 'edge_neighbors_unresolved',
                'Projected boundary must separate two consistently oriented faces', edge=edge.id, sides=sorted(pair))
        edge.faces = next(iter(pair))
        edge.left_face = edge.faces[0]
    complete_face_regions_and_cycles(graph)
    return labels


def cut_surface(mesh, oracle, graph, curves, cfg, *, save_partial=None, coarse_paths=None):
    cutter = SurfaceCut(mesh, oracle, cfg.feature_sdf_limit)
    cutter.embed(graph, curves, cfg.fine_spacing, coarse_paths or {})
    cut = cutter.triangulate()
    constraints = np.asarray(list(cutter.barriers.values()), dtype=int).reshape(-1, 3)
    if save_partial is not None:
        save_partial(cut, constraints)
    return cut, label_barriers(cut.triangles, graph, cutter.barriers), constraints
