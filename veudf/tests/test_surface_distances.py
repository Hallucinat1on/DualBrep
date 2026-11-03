"""Numerical and semantic regressions for the new global surface distance engine."""

from dataclasses import replace
import numpy as np
import pytest
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.Geom import Geom_Circle
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2
from veudf.preprocess.config import Config, load_config
from veudf.preprocess.distance import EdgeOracle, VertexOracle
from veudf.geometry.sampling import attach_surface_coordinates
from veudf.schema import EMPTY_DISTANCE

CFG = Config()


def segment(a, b):
    return BRepBuilderAPI_MakeEdge(gp_Pnt(*a), gp_Pnt(*b)).Edge()


def compare(edges, q):
    fast = EdgeOracle(edges, CFG)
    reference = EdgeOracle(edges, CFG, "cad_reference")
    a, b = fast.query(q), reference.query(q)
    np.testing.assert_allclose(a["distance"], b["distance"], atol=2e-7, rtol=0)
    unique = ~a["tie"] & ~b["tie"]
    np.testing.assert_array_equal(a["nearest_id"][unique], b["nearest_id"][unique])
    np.testing.assert_array_equal(a["tie"], b["tie"])
    return fast, a


@pytest.mark.parametrize(
    "first,last", [(0, np.pi / 2), (5.5, 7.5), (0, 2 * np.pi), (-2.0, 0.8)]
)
def test_finite_rotated_arcs_match_occ(first, last):
    circle = Geom_Circle(gp_Ax2(gp_Pnt(0.2, -0.1, 0.3), gp_Dir(0.2, 0.3, 1)), 0.7)
    edge = BRepBuilderAPI_MakeEdge(circle, first, last).Edge()
    q = np.random.default_rng(92).uniform(-1, 1, (200, 3))
    fast, _ = compare([edge], q)
    assert fast.calls == 0


def test_arc_distinct_endpoints_and_axis_ties():
    circle = Geom_Circle(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 1)
    edge = BRepBuilderAPI_MakeEdge(circle, 0, np.pi / 2).Edge()
    _, result = compare(
        [edge], np.array([[0, 0, 1], [-1, -1, 0], [2, 0, 0], [1, 0, 0]])
    )
    assert result["tie"].tolist() == [True, True, False, False]


def test_complex_curve_uses_exact_occ_fallback():
    p = TColgp_Array1OfPnt(1, 5)
    for i, q in enumerate(
        [(-0.8, 0, 0), (-0.4, 0.4, 0.2), (0, -0.2, 0), (0.4, 0.3, -0.2), (0.8, 0, 0)], 1
    ):
        p.SetValue(i, gp_Pnt(*q))
    curve = GeomAPI_PointsToBSpline(p).Curve()
    edge = BRepBuilderAPI_MakeEdge(curve).Edge()
    fast, _ = compare(
        [edge, segment((-0.8, -0.8, 0), (0.8, -0.8, 0))],
        np.random.default_rng(5).uniform(-1, 1, (100, 3)),
    )
    assert fast.calls > 0


def test_face_hint_does_not_hide_thin_wall_nearest_edge():
    mesh = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0, 0.5, 0]]), np.array(
        [[0, 1, 2]]
    )
    edges = [
        segment((-1, -1, 0), (1, -1, 0)),
        segment((1, -1, 0), (1, 1, 0)),
        segment((1, 1, 0), (-1, 1, 0)),
        segment((-1, 1, 0), (-1, -1, 0)),
        segment((0, -0.3, -0.001), (0, 0.3, -0.001)),
    ]
    oracle = EdgeOracle(
        edges,
        CFG,
        mesh=mesh,
        edge_faces=[[0]] * 4 + [[1]],
        triangle_face_ids=np.array([0]),
    )
    q = np.array([[0, 0, 0], [0.05, 0, 0]])
    result = oracle.query(q, triangle_ids=np.zeros(2, int), source_ids=np.zeros(2, int))
    assert result["nearest_id"].tolist() == [4, 4]
    np.testing.assert_allclose(
        result["distance"], [0.001, np.hypot(0.05, 0.001)], atol=1e-12
    )
    oracle.query(q, triangle_ids=np.zeros(2, int))
    assert oracle.report()["cache_hits"] > 0
    with pytest.raises(ValueError, match="candidate bound"):
        oracle.query([[9, 9, 9]], triangle_ids=np.array([0]))


def test_global_vertex_distance_and_absence():
    oracle = VertexOracle([[1, 0, 0], [-1, 0, 0], [0, 0, 0.001]], CFG)
    r = oracle.query([[0, 0, 0]])
    assert r["nearest_id"][0] == 2
    assert VertexOracle([[1, 0, 0], [-1, 0, 0]], CFG).query([[0, 0, 0]])["tie"][0]
    empty = VertexOracle([], CFG).query([[0, 0, 0]])
    assert empty["distance"][0] == EMPTY_DISTANCE and empty["nearest_id"][0] == -1


def test_surface_provenance_rejects_off_mesh_samples():
    mesh = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]])
    p = dict(
        points=np.array([[0.2, 0.3, 0.01]]),
        triangle_ids=np.array([0]),
        barycentric=np.array([[0.2, 0.3]]),
    )
    with pytest.raises(ValueError, match="reconstruct"):
        attach_surface_coordinates(p, mesh)
    p["points"][0, 2] = 0
    attach_surface_coordinates(p, mesh)


@pytest.mark.parametrize(
    "obsolete",
    [
        "band_max",
        "volume_pool_count",
        "evaluation_volume_count",
        "zero_checks_per_edge",
    ],
)
def test_old_config_keys_are_rejected(obsolete):
    with pytest.raises(ValueError, match="Unknown"):
        load_config(overrides={obsolete: 1})
