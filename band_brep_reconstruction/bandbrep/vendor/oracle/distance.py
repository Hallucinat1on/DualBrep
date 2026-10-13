"""Finite CAD-edge distance queries and exact finite-vertex nearest neighbours."""
from dataclasses import dataclass
import heapq
import itertools

import numpy as np
from scipy.spatial import cKDTree
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Circle
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_OUT, TopAbs_ON
from OCC.Core.gp import gp_Pnt

from .cad import PreprocessError, bounds, compound, xyz


def make_tool(shape, tolerance):
    tool = BRepExtrema_DistShapeShape()
    tool.LoadS2(shape)
    tool.SetDeflection(tolerance)
    return tool


def solve(tool, point_vertex):
    tool.LoadS1(point_vertex)
    tool.Perform()
    if not tool.IsDone() or tool.NbSolution() < 1:
        raise PreprocessError("distance_solver_failed", "OCC distance query did not converge")
    distance = float(tool.Value())
    if not np.isfinite(distance) or distance < 0:
        raise PreprocessError("distance_solver_failed", f"Invalid CAD distance: {distance}")
    closest = xyz(tool.PointOnShape2(1))
    if not np.isfinite(closest).all():
        raise PreprocessError("distance_solver_failed", "Nonfinite closest point")
    return distance, closest, tool.NbSolution() > 1


@dataclass
class Node:
    lo: np.ndarray
    hi: np.ndarray
    indices: tuple = ()
    children: tuple = ()


def build_bvh(boxes, indices):
    lo = np.min([boxes[i][0] for i in indices], axis=0)
    hi = np.max([boxes[i][1] for i in indices], axis=0)
    if len(indices) <= 2:
        return Node(lo, hi, tuple(indices))
    centers = np.array([(boxes[i][0] + boxes[i][1]) / 2 for i in indices])
    axis = int(np.argmax(np.ptp(centers, axis=0)))
    order = np.argsort(centers[:, axis], kind="stable")
    ordered = [indices[i] for i in order]
    mid = len(ordered) // 2
    return Node(lo, hi, children=(build_bvh(boxes, ordered[:mid]), build_bvh(boxes, ordered[mid:])))


def box_distance(point, node):
    return float(np.linalg.norm(np.maximum(np.maximum(node.lo - point, point - node.hi), 0)))


def empty_result(count, clip):
    return dict(distance=np.full(count, clip, dtype=np.float64),
                closest=np.zeros((count, 3), dtype=np.float64),
                nearest_id=np.full(count, -1, dtype=np.int32),
                tie=np.zeros(count, dtype=bool), gradient=np.zeros((count, 3)),
                gradient_valid=np.zeros(count, dtype=bool))


def add_gradient(result, points, tolerance):
    valid = (result["distance"] > tolerance) & ~result["tie"] & (result["nearest_id"] >= 0)
    result["gradient_valid"] = valid
    result["gradient"] = np.zeros((len(points), 3), dtype=np.float64)
    result["gradient"][valid] = ((points[valid] - result["closest"][valid])
                                 / result["distance"][valid, None])
    return result


class EdgeOracle:
    def __init__(self, edges, cfg, backend=None):
        self.cfg = cfg
        self.backend = backend or cfg.edge_backend
        if self.backend not in ("cad_reference", "cad_bvh"):
            raise ValueError("Unsupported edge distance backend")
        self.edges = edges
        self.tools = [make_tool(edge, cfg.solver_tolerance) for edge in edges]
        self.circles = {}
        for index, edge in enumerate(edges):
            curve = BRepAdaptor_Curve(edge)
            if curve.GetType() == GeomAbs_Circle:
                circle = curve.Circle()
                self.circles[index] = (xyz(circle.Location()), xyz(circle.Axis().Direction()))
        self.boxes = [bounds(edge, cfg.geometry_tolerance) for edge in edges]
        self.root = build_bvh(self.boxes, list(range(len(edges)))) if edges else None
        self.calls = 0

    def query(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        result = empty_result(len(points), self.cfg.clip_edge)
        if not self.edges:
            return result
        slack = self.cfg.tie_tolerance + 2 * self.cfg.solver_tolerance
        for row, point in enumerate(points):
            query_vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(*point)).Vertex()
            best, closest, nearest, ambiguous = float("inf"), None, -1, False
            candidates = []
            if self.backend == "cad_reference":
                candidates = list(range(len(self.edges)))
            else:
                counter = itertools.count()
                heap = [(box_distance(point, self.root), next(counter), self.root)]
            while candidates or (self.backend != "cad_reference" and heap):
                if candidates:
                    index = candidates.pop()
                else:
                    lower, _, node = heapq.heappop(heap)
                    if lower > best + slack:
                        continue
                    if node.children:
                        for child in node.children:
                            lower = box_distance(point, child)
                            if lower <= best + slack:
                                heapq.heappush(heap, (lower, next(counter), child))
                        continue
                    candidates.extend(node.indices)
                    continue
                distance, projected, local_tie = solve(self.tools[index], query_vertex)
                self.calls += 1
                if index in self.circles:
                    center, normal = self.circles[index]
                    displacement = point - center
                    radial = displacement - np.dot(displacement, normal) * normal
                    # The whole finite arc is equidistant from an axis point;
                    # the kernel need not report infinitely many solutions.
                    local_tie |= np.linalg.norm(radial) <= self.cfg.solver_tolerance
                if abs(np.linalg.norm(point - projected) - distance) > self.cfg.label_tolerance:
                    raise PreprocessError("distance_solver_failed", "Closest-point distance disagrees with CAD value")
                if distance < best:
                    is_tie = best - distance <= self.cfg.tie_tolerance
                    ambiguous = (ambiguous if is_tie else False) or is_tie or local_tie
                    best, closest, nearest = distance, projected, index
                elif distance - best <= self.cfg.tie_tolerance:
                    ambiguous = True
            result["distance"][row] = best
            result["closest"][row] = closest
            result["nearest_id"][row] = nearest
            result["tie"][row] = ambiguous
        return add_gradient(result, points, self.cfg.solver_tolerance)


class VertexOracle:
    def __init__(self, vertices, cfg):
        self.vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        self.cfg = cfg
        self.tree = cKDTree(self.vertices) if len(self.vertices) else None

    def query(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        result = empty_result(len(points), self.cfg.clip_vertex)
        if self.tree is None or not len(points):
            return result
        k = min(2, len(self.vertices))
        distance, index = self.tree.query(points, k=k, eps=0)
        if k == 2:
            result["tie"] = distance[:, 1] - distance[:, 0] <= self.cfg.tie_tolerance
            distance, index = distance[:, 0], index[:, 0]
        result["distance"] = distance
        result["nearest_id"] = index.astype(np.int32)
        result["closest"] = self.vertices[index]
        return add_gradient(result, points, self.cfg.solver_tolerance)


class SDFOracle:
    def __init__(self, solid, faces, mesh, cfg, backend=None):
        self.cfg = cfg
        self.backend = backend or cfg.sdf_backend
        if self.backend not in ("cad", "mesh", "mesh_fast"):
            raise ValueError("Unsupported SDF backend")
        self.classifier = BRepClass3d_SolidClassifier(solid)
        # Distance to boundary faces, NOT distance to solid (zero for interior).
        self.tool = make_tool(compound(faces), cfg.solver_tolerance)
        self.mesh = mesh
        self.igl = None
        if self.backend.startswith("mesh"):
            try:
                import igl
            except ImportError as exc:
                raise PreprocessError("missing_dependency", "mesh SDF needs libigl; install it or use --sdf-backend cad") from exc
            self.igl = igl
        if self.backend == "mesh_fast":
            v, f = self.mesh[:2]
            triangles = v[f]
            volume = np.einsum("ij,ij->i", triangles[:, 0],
                               np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
            if not np.isfinite(volume) or volume <= 0:
                raise PreprocessError("sdf_sign_failed", "Fast mesh SDF requires positive oriented mesh volume")

    def query(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        values = np.empty(len(points), dtype=np.float64)
        if self.backend == "mesh_fast":
            if len(points):
                v, f = self.mesh[:2]
                signed, _, closest, _ = self.igl.signed_distance(points, v, f,
                    sign_type=self.igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)
                # libigl scales magnitude by the approximate winding value.
                # Use the actual closest-point norm, and winding only for sign.
                unsigned = np.linalg.norm(points - closest, axis=1)
                values[:] = np.where(signed < 0, -unsigned, unsigned)
                values[unsigned <= self.cfg.solver_tolerance] = 0
                ambiguous = ((unsigned > self.cfg.solver_tolerance)
                             & (np.abs(signed) < 0.1 * unsigned))
                for i in np.flatnonzero(ambiguous):
                    self.classifier.Perform(gp_Pnt(*points[i]), self.cfg.solver_tolerance)
                    state = self.classifier.State()
                    if state == TopAbs_IN:
                        values[i] = -unsigned[i]
                    elif state == TopAbs_OUT:
                        values[i] = unsigned[i]
                    elif state == TopAbs_ON:
                        values[i] = 0
                    else:
                        raise PreprocessError("sdf_sign_failed", "Ambiguous winding and unknown CAD sign")
            if not np.isfinite(values).all():
                raise PreprocessError("sdf_sign_failed", "Nonfinite fast mesh SDF")
            return values
        if self.backend == "mesh" and len(points):
            v, f = self.mesh[:2]
            values[:] = self.igl.signed_distance(points, v, f,
                sign_type=self.igl.SIGNED_DISTANCE_TYPE_UNSIGNED)[0]
        for i, point in enumerate(points):
            pnt = gp_Pnt(*point)
            if self.backend == "cad":
                values[i] = solve(self.tool, BRepBuilderAPI_MakeVertex(pnt).Vertex())[0]
            self.classifier.Perform(pnt, self.cfg.solver_tolerance)
            state = self.classifier.State()
            if state == TopAbs_IN:
                values[i] *= -1
            elif state == TopAbs_ON:
                values[i] = 0
            elif state != TopAbs_OUT:
                raise PreprocessError("sdf_sign_failed", "Solid classifier returned UNKNOWN")
        if not np.isfinite(values).all():
            raise PreprocessError("distance_solver_failed", "Nonfinite SDF values")
        return values
