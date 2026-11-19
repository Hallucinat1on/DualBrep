"""Embed curve barriers in a surface copy, then recover oriented face uses."""
from collections import defaultdict
import heapq

import numpy as np
from shapely import LineString, constrained_delaunay_triangles
from shapely.ops import polygonize

from .boundary_cycles import complete_face_regions_and_cycles
from .deadline import checkpoint
from .schemas import SurfaceMesh, require
from .surface_mesh import components, mesh_edges


EPS = 2e-9


def _edge_point(a, b, c, d):
    """Find a point on mesh edge a-b near path segment c-d."""
    edge, path = b - a, d - c
    u, t = np.linalg.lstsq(np.column_stack((edge, -path)), c - a, rcond=None)[0]
    if t < 0 or t > 1:
        u = (c + np.clip(t, 0, 1) * path - a) @ edge / (edge @ edge)
    p = a + np.clip(u, 0, 1) * edge
    t = np.clip((p - c) @ path / max(path @ path, 1e-30), 0, 1)
    return p, np.linalg.norm(p - (c + t * path))


class SurfaceCut:
    def __init__(self, mesh, oracle, tolerance):
        self.mesh, self.oracle, self.tolerance = mesh, oracle, tolerance
        self.points = list(mesh.vertices)
        self.point_ids = {self.key(p): i for i, p in enumerate(mesh.vertices)}
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
        t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
        return a + t * (b - a)

    def edge_node(self, edge, point):
        p = self.on_edge(edge, point)
        for node in edge:
            if np.linalg.norm(p - self.points[node]) <= EPS * 2:
                return int(node)
        for node in self.edge_nodes[edge]:
            if np.linalg.norm(p - self.points[node]) <= EPS * 2:
                return node
        node = self.point(p)
        self.edge_nodes[edge].add(node)
        return node

    def locate(self, triangle, point):
        for node in self.mesh.triangles[triangle]:
            if np.linalg.norm(self.points[node] - point) <= EPS * 2:
                return int(node), self.vertex_faces[node]
        for edge in map(tuple, self.face_edges[triangle]):
            if np.linalg.norm(self.on_edge(edge, point) - point) <= EPS * 2:
                return self.edge_node(edge, point), set(self.edge_faces[edge])
        return self.point(point), {int(triangle)}

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
            for neighbor, edge in self.neighbors[triangle]:
                a, b = self.mesh.vertices[list(edge)]
                mid, error = _edge_point(a, b, p, q)
                if error > corridor:
                    continue
                cost = value + error + corridor * .05
                if cost < costs.get(neighbor, np.inf):
                    costs[neighbor] = cost
                    previous[neighbor] = (triangle, edge, mid)
                    heapq.heappush(queue, (cost, neighbor))
        require(last is not None, 'curve_embedding_disconnected', 'Projected curve jumps across surface sheets')
        parts = []
        while last in previous:
            before, edge, mid = previous[last]
            node = self.edge_node(edge, mid)
            parts.append((last, node, dst))
            last, dst = before, node
        return [(last, src, dst)] + parts[::-1]

    def embed(self, graph, curves, spacing, coarse_paths):
        mesh_anchors = {}
        for edge in graph.edges:
            if edge.id in coarse_paths:
                path = coarse_paths[edge.id]
                mesh_anchors.update(zip(edge.vertices, (path[0], path[-1])))
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
            count = max(17, int(np.ceil(lengths.sum() / spacing)) + 1)
            q = curve.evaluate(np.linspace(0, 1, count))
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
                    anchors.append(self.locate(f, p))
            if edge.vertices:
                for index, vertex in zip((0, -1), edge.vertices):
                    anchor = shared.setdefault(vertex, anchors[index])
                    require(np.linalg.norm(self.points[anchor[0]] - self.points[anchors[index][0]]) <= EPS * 4,
                            'curve_embedding_vertex_conflict', 'Uses of a vertex project to different sheets', vertex=vertex)
                    anchors[index] = anchor
            elif edge.closed:
                anchors[-1] = anchors[0]
            for (src, starts), (dst, ends) in zip(anchors, anchors[1:]):
                if src != dst:
                    for triangle, a, b in self.route(src, dst, starts, ends):
                        if a != b:
                            self.cuts[triangle].append((a, b, edge.id))

    def _expanded(self, fid, a, b):
        # A path lying on an original mesh edge must also honor split points
        # inserted later by another path or by the opposite triangle.
        p, q = self.points[a], self.points[b]
        for edge in map(tuple, self.face_edges[fid]):
            if max(np.linalg.norm(self.on_edge(edge, x) - x) for x in (p, q)) <= EPS * 4:
                delta = q - p
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
            uv_to_id, segments = {}, set()

            def uv(node):
                result = tuple((self.points[node] - a) @ basis.T)
                uv_to_id[result] = node
                return result

            def add_segments(nodes, eid=None):
                for u, v in zip(nodes, nodes[1:]):
                    key = tuple(sorted((u, v)))
                    segments.add(key)
                    if eid is not None:
                        self.add_barrier(u, v, eid)

            for u, v in zip(face, np.roll(face, -1)):
                add_segments(self._expanded(fid, u, v))
            for u, v, eid in self.cuts[fid]:
                add_segments(self._expanded(fid, u, v), eid)
            polygons = list(polygonize([LineString((uv(u), uv(v))) for u, v in segments]))
            area = np.linalg.norm(normal) * np.linalg.norm(b - a) / 2
            require(abs(sum(p.area for p in polygons) - area) <= max(1e-14, area * 1e-6),
                    'curve_embedding_crossing', 'Local cuts do not partition the original triangle', triangle=fid)
            for polygon in polygons:
                for part in constrained_delaunay_triangles(polygon).geoms:
                    coordinates = np.asarray(part.exterior.coords)[:3]
                    u, v = coordinates[1:] - coordinates[0]
                    if u[0] * v[1] - u[1] * v[0] < 0:
                        coordinates = coordinates[[0, 2, 1]]
                    triangles.append([uv_to_id[tuple(p)] for p in coordinates])
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
    # Preserve the constraint-to-curve map, including on label failure.
    constraints = np.asarray(list(cutter.barriers.values()), dtype=int).reshape(-1, 3)
    if save_partial is not None:
        save_partial(cut, constraints)
    return cut, label_barriers(cut.triangles, graph, cutter.barriers), constraints
