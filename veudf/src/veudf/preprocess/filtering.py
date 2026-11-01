"""Dataset eligibility; it never removes or modifies CAD entities."""

import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopoDS import topods

from veudf.preprocess.cad import PreprocessError, unique_shapes


def check_face_count(count, cfg):
    if count < cfg.min_faces:
        raise PreprocessError(
            "face_count_below_min", f"{count} faces < {cfg.min_faces}"
        )
    if count > cfg.max_faces:
        raise PreprocessError(
            "face_count_above_max", f"{count} faces > {cfg.max_faces}"
        )


def check_solid_faces(solid, cfg):
    """Cheap rejection before target extraction."""
    _, faces = unique_shapes(solid, TopAbs_FACE)
    check_face_count(len(faces), cfg)


def face_edge_counts(topology):
    """Unique source edges per face over all wires, including seams."""
    counts = [set() for _ in topology["source_faces"]]
    for wire in topology["source_wires"]:
        for use in wire["uses"]:
            counts[wire["source_face_id"]].add(use["source_edge_id"])
    return [len(c) for c in counts]


def check_metrics(face_edges, lengths, cfg):
    """Pure numeric predicate (exact boundary regression checks)."""
    check_face_count(len(face_edges), cfg)
    excessive = [i for i, n in enumerate(face_edges) if n > cfg.max_face_edges]
    if excessive:
        raise PreprocessError(
            "face_edge_count_exceeded",
            f"Faces {excessive[:5]} exceed {cfg.max_face_edges} edges",
        )
    lengths = np.asarray(lengths, np.float64)
    if not len(lengths) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise PreprocessError(
            "invalid_edge_length", "Nonfinite/nonpositive source arc length"
        )
    short = np.flatnonzero(lengths < cfg.min_edge_length)
    if len(short):
        raise PreprocessError(
            "edge_too_short",
            f"Source edges {short[:5].tolist()} shorter than {cfg.min_edge_length}",
        )
    return dict(
        face_count=len(face_edges),
        max_face_edges=max(face_edges),
        min_source_edge_length=float(lengths.min()),
    )


def audit_shape(solid, targets, cfg):
    face_edges = face_edge_counts(targets.topology)
    check_face_count(len(face_edges), cfg)
    if max(face_edges) > cfg.max_face_edges:  # before arc-length integration
        check_metrics(face_edges, [1.0], cfg)
    _, edges = unique_shapes(solid, TopAbs_EDGE)
    lengths = []
    for sid, edge in enumerate(edges):
        if BRep_Tool.Degenerated(topods.Edge(edge)):
            raise PreprocessError("degenerate_edge", f"Source edge {sid} is degenerate")
        curve = BRepAdaptor_Curve(topods.Edge(edge))
        value = GCPnts_AbscissaPoint.Length(curve, cfg.solver_tolerance)
        if abs(value - cfg.min_edge_length) <= 10 * cfg.solver_tolerance:
            value = GCPnts_AbscissaPoint.Length(curve, min(cfg.solver_tolerance, 1e-10))
        lengths.append(value)
    return check_metrics(face_edges, lengths, cfg)
