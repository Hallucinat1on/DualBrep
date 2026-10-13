"""OCC topology / triangulation, adapted from DualBrep occ_utils conventions.

No imports from DualBrep: copying the enclosing directory is sufficient.
Normalization uses geometric bounding boxes once, not separate mesh bounds.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import gzip
from pathlib import Path
import tempfile

import numpy as np
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepTools import BRepTools_WireExplorer
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GeomAbs import GeomAbs_Circle, GeomAbs_Ellipse
from OCC.Core.TopAbs import (TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX, TopAbs_WIRE,
                             TopAbs_SOLID, TopAbs_FORWARD, TopAbs_REVERSED)
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.TopoDS import TopoDS_Compound, topods
from OCC.Core.gp import gp_Trsf, gp_Vec
from OCC.Extend.DataExchange import read_step_file


class PreprocessError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def xyz(point):
    return np.array([point.X(), point.Y(), point.Z()], dtype=np.float64)


def unique_shapes(shape, kind):
    index = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, kind, index)
    return index, [index.FindKey(i) for i in range(1, index.Size() + 1)]


def compound(shapes):
    result = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(result)
    for shape in shapes:
        builder.Add(result, shape)
    return result


def bounds(shape, padding=0.0):
    box = Bnd_Box()
    # No sampled-point or triangulation boxes for distance pruning.
    brepbndlib.AddOptimal(shape, box, False, True)
    if box.IsVoid() or box.IsOpen():
        raise PreprocessError("invalid_bbox", "No finite geometry bounding box")
    values = np.array(box.Get(), dtype=np.float64)
    if not np.isfinite(values).all():
        raise PreprocessError("invalid_bbox", "Nonfinite bounding box")
    return values[:3] - padding, values[3:] + padding


@contextmanager
def step_input(path):
    path = Path(path)
    if path.suffix.lower() == ".gz":
        with tempfile.TemporaryDirectory(prefix="evudf-step-") as tmp:
            dest = Path(tmp) / "input.step"
            with gzip.open(path, "rb") as src, dest.open("wb") as out:
                import shutil
                shutil.copyfileobj(src, out)
            yield dest
    else:
        yield path


def read_normalized(path, cfg):
    with step_input(path) as input_path:
        try:
            imported = read_step_file(str(input_path), verbosity=False)
        except Exception as exc:
            raise PreprocessError("read_failed", str(exc)) from exc
    if imported is None or imported.IsNull():
        raise PreprocessError("read_failed", "STEP contains no shape")
    _, solids = unique_shapes(imported, TopAbs_SOLID)
    if len(solids) != 1:
        code = "open_boundary" if not solids else "unsupported_multi_solid"
        raise PreprocessError(code, f"Expected one closed solid, got {len(solids)}")
    solid = topods.Solid(solids[0])
    # Refuse loose shell / wire content outside the chosen solid.
    for kind in (TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX):
        solid_index, _ = unique_shapes(solid, kind)
        _, all_items = unique_shapes(imported, kind)
        if any(solid_index.FindIndex(item) == 0 for item in all_items):
            raise PreprocessError("invalid_solid", "Loose topology outside the sole solid")
    if not BRepCheck_Analyzer(solid).IsValid():
        raise PreprocessError("invalid_solid", "BRepCheck_Analyzer rejected the source solid")
    properties = GProp_GProps()
    brepgprop.VolumeProperties(solid, properties)
    volume = float(properties.Mass())
    if not np.isfinite(volume) or volume <= 0:
        raise PreprocessError("invalid_solid", "Source has nonpositive or nonfinite oriented volume")
    lower, upper = bounds(solid)
    extent = float(np.max(upper - lower))
    if extent <= 0:
        raise PreprocessError("invalid_bbox", "Solid has zero extent")
    center = (lower + upper) / 2
    scale = 2 * cfg.bounding / extent
    translation = gp_Trsf()
    translation.SetTranslationPart(gp_Vec(*(-center)))
    transform = gp_Trsf()
    transform.SetScaleFactor(scale)
    transform.Multiply(translation)
    normalized = topods.Solid(BRepBuilderAPI_Transform(solid, transform, True).Shape())
    if not BRepCheck_Analyzer(normalized).IsValid():
        raise PreprocessError("invalid_solid", "Normalization produced an invalid solid")
    lo, hi = bounds(normalized)
    if np.max(np.abs(np.r_[lo, hi])) >= 1:
        raise PreprocessError("invalid_bbox", "Normalized geometry leaves the query domain")
    matrix = np.eye(4)
    matrix[:3, :3] *= scale
    matrix[:3, 3] = -center * scale
    metadata = dict(center=center.tolist(), scale=scale, source_volume=volume, normalized_volume=volume * scale**3,
                    world_to_normalized=matrix.tolist(),
                    normalized_to_world=np.linalg.inv(matrix).tolist(),
                    source_bounds=[lower.tolist(), upper.tolist()],
                    normalized_bounds=[lo.tolist(), hi.tolist()],
                    source_units="as converted by STEP reader; distances follow saved transform")
    return normalized, metadata


@dataclass
class Targets:
    edges: list
    vertices: np.ndarray
    faces: list
    topology: dict


def collect_targets(solid, cfg):
    _, faces = unique_shapes(solid, TopAbs_FACE)
    ei, raw_edges = unique_shapes(solid, TopAbs_EDGE)
    vi, raw_vertices = unique_shapes(solid, TopAbs_VERTEX)
    faces = [topods.Face(f) for f in faces]
    edges = [topods.Edge(e.Oriented(TopAbs_FORWARD)) for e in raw_edges]
    vertices = [topods.Vertex(v) for v in raw_vertices]
    uses = [[] for _ in edges]
    wires = []
    max_tolerance = 0.0
    for item in faces + edges + vertices:
        max_tolerance = max(max_tolerance, BRep_Tool.Tolerance(item))
    if max_tolerance > cfg.geometry_tolerance:
        raise PreprocessError("tolerance_exceeded", f"CAD tolerance {max_tolerance:g} exceeds budget")
    for fid, face in enumerate(faces):
        wire_exp = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_exp.More():
            wire = topods.Wire(wire_exp.Current())
            raw_count = 0
            raw_exp = TopExp_Explorer(wire, TopAbs_EDGE)
            while raw_exp.More():
                raw_count += 1
                raw_exp.Next()
            wid = len(wires)
            local = []
            ordered = BRepTools_WireExplorer(wire, face)
            while ordered.More():
                edge = topods.Edge(ordered.Current())
                orientation = edge.Orientation()
                if orientation not in (TopAbs_FORWARD, TopAbs_REVERSED):
                    raise PreprocessError("unsupported_same_face_edge", "INTERNAL/EXTERNAL edge use")
                sid = ei.FindIndex(edge) - 1
                use = dict(source_edge_id=sid, source_face_id=fid, source_wire_id=wid,
                           position=len(local), orientation=int(orientation))
                local.append(use)
                uses[sid].append(use)
                ordered.Next()
            if len(local) != raw_count:
                raise PreprocessError("open_boundary", "Wire explorer did not traverse every edge use")
            wires.append(dict(source_wire_id=wid, source_face_id=fid, uses=local))
            wire_exp.Next()

    source_edges, kept, dropped, endpoints = [], [], [], []
    for sid, edge in enumerate(edges):
        if BRep_Tool.Degenerated(edge):
            raise PreprocessError("degenerate_edge", f"Source edge {sid} is degenerate")
        try:
            curve = BRepAdaptor_Curve(edge)
            first, last = curve.FirstParameter(), curve.LastParameter()
            if not np.isfinite([first, last]).all() or last <= first:
                raise ValueError("Invalid finite curve interval")
            if not np.isfinite(np.r_[xyz(curve.Value(first)), xyz(curve.Value(last))]).all():
                raise ValueError("Nonfinite curve")
        except Exception as exc:
            raise PreprocessError("missing_curve", f"Edge {sid}: {exc}") from exc
        use_list = uses[sid]
        fids = sorted({u["source_face_id"] for u in use_list})
        seams = [fid for fid in fids if BRep_Tool.IsClosed(edge, faces[fid])]
        a, b = topexp.FirstVertex(edge), topexp.LastVertex(edge)
        if a.IsNull() or b.IsNull():
            raise PreprocessError("missing_curve", f"Edge {sid} has no topological endpoints")
        vids = [vi.FindIndex(a) - 1, vi.FindIndex(b) - 1]
        if min(vids) < 0:
            raise PreprocessError("invalid_solid", "Endpoint missing from vertex map")
        for vertex, parameter in zip((a, b), (first, last)):
            if np.linalg.norm(xyz(BRep_Tool.Pnt(vertex)) - xyz(curve.Value(parameter))) > cfg.geometry_tolerance:
                raise PreprocessError("tolerance_exceeded", f"Edge {sid} endpoint gap exceeds budget")
        source_edges.append(dict(source_edge_id=sid, uses=use_list, face_ids=fids,
                                 source_vertex_ids=vids, parameter_range=[first, last],
                                 curve_type=int(curve.GetType()), seam_faces=seams))
        if seams:
            if len(use_list) != 2 or len(fids) != 1 or use_list[0]["orientation"] == use_list[1]["orientation"]:
                raise PreprocessError("nonmanifold_use", f"Seam {sid} has unexpected uses")
            dropped.append(dict(source_edge_id=sid, reason="seam_excluded"))
            continue
        if len(use_list) != 2:
            code = "open_boundary" if len(use_list) < 2 else "nonmanifold_use"
            raise PreprocessError(code, f"Edge {sid} has {len(use_list)} uses")
        if len(fids) != 2:
            raise PreprocessError("unsupported_same_face_edge", f"Non-seam edge {sid} lies within one face")
        kept.append(sid)
        endpoints.append(vids)

    incident = {i: set() for i in range(len(vertices))}
    for target_id, pair in enumerate(endpoints):
        for v in pair:
            incident[v].add(target_id)
    removed, keep_vertices = [], []
    for vid, vertex in enumerate(vertices):
        connected = incident[vid]
        if not connected:
            removed.append(dict(source_vertex_id=vid, reason="no_retained_edge"))
            continue
        removable = False
        if len(connected) == 1:
            tid = next(iter(connected))
            curve = BRepAdaptor_Curve(edges[kept[tid]])
            # Deliberately conservative: only full analytic periodic loops.
            removable = (endpoints[tid] == [vid, vid]
                         and curve.GetType() in (GeomAbs_Circle, GeomAbs_Ellipse)
                         and curve.IsPeriodic()
                         and abs(curve.LastParameter() - curve.FirstParameter() - curve.Period()) <= 1e-8)
        if removable:
            removed.append(dict(source_vertex_id=vid, reason="loop_anchor_removed"))
        else:
            keep_vertices.append(vid)
    vertex_map = {sid: i for i, sid in enumerate(keep_vertices)}
    vertex_xyz = np.array([xyz(BRep_Tool.Pnt(vertices[i])) for i in keep_vertices], dtype=np.float64).reshape(-1, 3)
    target_edges = []
    for tid, sid in enumerate(kept):
        pair = endpoints[tid]
        target_pair = [vertex_map[v] for v in pair if v in vertex_map]
        if len(target_pair) not in (0, 2):
            raise PreprocessError("ambiguous_canonicalization", "Only one endpoint was removed")
        target_edges.append(dict(edge_id=tid, source_edge_ids=[sid],
                                 source_curve_pieces=[dict(source_edge_id=sid,
                                     parameter_range=source_edges[sid]["parameter_range"], orientation=0)],
                                 face_ids=source_edges[sid]["face_ids"], vertex_ids=target_pair,
                                 closed=pair[0] == pair[1],
                                 self_loop=len(target_pair) == 2 and target_pair[0] == target_pair[1]))
    topology = dict(filter_version="nonseam_interface_v1", canonicalization_version="analytic_loop_anchor_v1",
                    canonicalization_status="conservative: other splits preserved",
                    source_faces=[dict(source_face_id=i, orientation=int(f.Orientation())) for i, f in enumerate(faces)],
                    source_edges=source_edges, source_wires=wires,
                    source_vertices=[dict(source_vertex_id=i, xyz=xyz(BRep_Tool.Pnt(v)).tolist()) for i, v in enumerate(vertices)],
                    edges=target_edges,
                    vertices=[dict(vertex_id=vertex_map[sid], source_vertex_id=sid,
                                   xyz=vertex_xyz[vertex_map[sid]].tolist(), edge_ids=sorted(incident[sid]))
                              for sid in keep_vertices],
                    excluded_edges=dropped, excluded_vertices=removed,
                    max_cad_tolerance=max_tolerance)
    return Targets([edges[i] for i in kept], vertex_xyz, faces, topology)


def triangulate(targets, cfg):
    result_v, result_f, result_ids = [], [], []
    # Absolute deflection, unlike DualBrep's original relative setting.
    mesher = BRepMesh_IncrementalMesh(compound(targets.faces), cfg.mesh_deflection, False, cfg.mesh_angle, False)
    if not mesher.IsDone():
        raise PreprocessError("meshing_failed", "OCC mesher did not finish")
    offset = 0
    for fid, face in enumerate(targets.faces):
        loc = TopLoc_Location()
        mesh = BRep_Tool.Triangulation(face, loc)
        if mesh is None or mesh.NbTriangles() == 0:
            raise PreprocessError("meshing_failed", f"Face {fid} has no triangles")
        trsf = loc.Transformation()
        points = np.array([xyz(mesh.Node(i).Transformed(trsf)) for i in range(1, mesh.NbNodes() + 1)])
        triangles = np.array([[mesh.Triangle(i).Value(k) - 1 for k in (1, 2, 3)]
                              for i in range(1, mesh.NbTriangles() + 1)], dtype=np.int64)
        if face.Orientation() == TopAbs_REVERSED:
            triangles = triangles[:, ::-1]
        result_v.append(points)
        result_f.append(triangles + offset)
        result_ids.append(np.full(len(triangles), fid, dtype=np.int32))
        offset += len(points)
    return np.concatenate(result_v), np.concatenate(result_f), np.concatenate(result_ids)
