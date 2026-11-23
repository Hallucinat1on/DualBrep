"""End-to-end synthetic STEP tests for degenerate-edge cones and tilted spherical caps."""
import math
import numpy as np
import pytest
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeSphere,
)
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCC.Extend.DataExchange import read_step_file, write_step_file

from bandbrep.cad_source import load_cad_source
from bandbrep.config import Config
from bandbrep.edge_first import reconstruct


def _make_full_cone():
    return BRepPrimAPI_MakeCone(1.0, 0.0, 2.0).Shape()


def _make_oblique_cut_full_cone():
    cone = BRepPrimAPI_MakeCone(1.0, 0.0, 2.5).Shape()
    # Cut the lower portion with a plane tilted by ~20 degrees so the base loop is an ellipse.
    tilted_ax = gp_Ax2(gp_Pnt(0.0, 0.0, 0.45), gp_Dir(math.sin(math.radians(20)), 0.0, math.cos(math.radians(20))))
    cutter = BRepPrimAPI_MakeBox(tilted_ax, 6.0, 6.0, 3.0).Shape()
    # Shift cutter downward along its local -Z so it removes the bottom below z ~ 0.45.
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.gp import gp_Trsf, gp_Vec
    ax_dir = tilted_ax.Direction()
    x_dir = tilted_ax.XDirection()
    y_dir = tilted_ax.YDirection()
    trsf = gp_Trsf()
    trsf.SetTranslation(
        gp_Vec(
            -3.0 * x_dir.X() - 3.0 * y_dir.X() - 3.0 * ax_dir.X(),
            -3.0 * x_dir.Y() - 3.0 * y_dir.Y() - 3.0 * ax_dir.Y(),
            -3.0 * x_dir.Z() - 3.0 * y_dir.Z() - 3.0 * ax_dir.Z(),
        )
    )
    cutter = BRepBuilderAPI_Transform(cutter, trsf, True).Shape()
    return BRepAlgoAPI_Cut(cone, cutter).Shape()


def _make_tilted_half_cone():
    axis = gp_Ax2(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 1.0, 2.0), gp_Dir(1.0, -1.0, 0.0))
    return BRepPrimAPI_MakeCone(axis, 1.0, 0.0, 2.0, math.pi).Shape()


def _make_tilted_spherical_cap():
    axis = gp_Ax2(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 1.0, 2.0), gp_Dir(1.0, -1.0, 0.0))
    return BRepPrimAPI_MakeSphere(axis, 1.0, math.radians(15.0), math.pi / 2.0).Shape()


def _make_cone_sphere_side_patch():
    """Cone intersected with an off-axis sphere: local conical patch with 1 non-wrapping closed loop."""
    cone = BRepPrimAPI_MakeCone(1.0, 0.0, 2.0).Shape()
    sphere = BRepPrimAPI_MakeSphere(gp_Pnt(0.85, 0.0, 0.7), 0.45).Shape()
    return BRepAlgoAPI_Common(cone, sphere).Shape()


@pytest.mark.parametrize(
    "name,factory,expected_counts,expected_kind",
    [
        ("full_cone", _make_full_cone, (2, 1, 0), "cone"),
        ("oblique_cut_cone", _make_oblique_cut_full_cone, (2, 1, 0), "cone"),
        ("tilted_half_cone", _make_tilted_half_cone, (3, 4, 3), "cone"),
        ("tilted_spherical_cap", _make_tilted_spherical_cap, (2, 1, 0), "sphere"),
        ("cone_sphere_side_patch", _make_cone_sphere_side_patch, (2, 3, 3), "cone"),
    ],
)
def test_degenerate_edge_step_reconstruction(
    tmp_path, name, factory, expected_counts, expected_kind
):
    step_in = tmp_path / f"{name}_in.step"
    write_step_file(factory(), str(step_in))
    mesh, udf = load_cad_source(step_in, fingerprint=name)
    out = tmp_path / name
    result = reconstruct(mesh, udf, Config(face_samples=250), out)
    graph = result["graph"]
    assert (graph.face_count, len(graph.edges), len(graph.vertices)) == expected_counts
    assert any(p.kind == expected_kind for p in result["patches"].values())
    assert result["report"]["stages"]["cad"]["roundtrip_checked"]
    step_out = out / "reconstruction.step"
    assert step_out.is_file()
    reloaded = read_step_file(str(step_out), verbosity=False)
    assert reloaded is not None and not reloaded.IsNull()
    assert BRepCheck_Analyzer(reloaded).IsValid()


def test_non_wrapping_single_closed_loop_on_cone_skips_conical_cap(tmp_path):
    """A single closed loop on a cone side with Δu = 0 must not trigger make_conical_cap."""
    from bandbrep.edge_first_graph import Path, fit_geometry
    from bandbrep.occ_assembly import assemble_brep, _wraps_u_once
    from bandbrep.schemas import CoedgeUse, EdgeInstance, LogicalGraph
    from bandbrep.surface_fitting import FacePatch, frame

    axis = np.array([0.0, 0.0, 1.0])
    x, y = frame(axis)
    cfg = Config()
    cone_o = np.zeros(3)
    angle = -math.atan(0.5)  # R(z) = 1.0 - 0.5*z
    cone_patch = FacePatch(
        0,
        "cone",
        dict(origin=cone_o, axis=axis, x=x, y=y, radius=1.0, angle=angle),
        np.array([1.0, 0.0, 0.0]),
        {},
    )
    sc = np.array([0.85, 0.0, 0.7])
    sr = 0.45
    # Exact z-bounds where the intersection loop crosses u=0: (1 - 0.5*z - sc_x)^2 + (z - sc_z)^2 = sr^2
    a_q = 0.25 + 1.0
    b_q = -(1.0 - sc[0]) - 2.0 * sc[2]
    c_q = (1.0 - sc[0]) ** 2 + sc[2] ** 2 - sr**2
    disc = math.sqrt(b_q * b_q - 4.0 * a_q * c_q)
    z_lo = (-b_q - disc) / (2.0 * a_q)
    z_hi = (-b_q + disc) / (2.0 * a_q)
    # Cosine-clustered z sampling gives dense points near both turnarounds z_lo and z_hi.
    phi = np.linspace(0.0, np.pi, 200)
    zs = 0.5 * (z_lo + z_hi) - 0.5 * (z_hi - z_lo) * np.cos(phi)
    rz = 1.0 - 0.5 * zs
    cos_u = np.clip((rz**2 + sc[0] ** 2 + (zs - sc[2]) ** 2 - sr**2) / (2.0 * rz * sc[0]), -1.0, 1.0)
    us = np.arccos(cos_u)
    upper = np.column_stack((rz * np.cos(us), rz * np.sin(us), zs))
    lower = np.column_stack((rz[-2:0:-1] * np.cos(-us[-2:0:-1]), rz[-2:0:-1] * np.sin(-us[-2:0:-1]), zs[-2:0:-1]))
    loop = np.vstack((upper, lower, upper[:1]))
    curve = fit_geometry(0, Path([], loop, True), [], cfg)
    assert not _wraps_u_once(cone_patch, curve)

    # Sphere patch aligned to equator (axis = +y, x = +z, y = +x -> center of patch at -x has v = 0).
    sax = np.array([0.0, 1.0, 0.0])
    sx, sy = frame(sax)
    cone_patch.data["orientation_point"] = cone_patch.evaluate(np.array([[0.0, 0.7 / np.cos(angle)]]))[0]
    cone_patch.normal = np.array([1.0, 0.0, 0.5]) / np.hypot(1.0, 0.5)
    sphere_patch = FacePatch(
        1,
        "sphere",
        dict(origin=sc, axis=sax, x=sx, y=sy, radius=sr, orientation_point=sc + np.array([-sr, 0.0, 0.0])),
        np.array([-1.0, 0.0, 0.0]),
        {},
    )
    edge = EdgeInstance(0, (0, 1), loop, True, [], -1, cfg.support_radius)
    graph = LogicalGraph(
        vertices=[],
        edges=[edge],
        face_count=2,
        cycles={0: [[CoedgeUse(0, 0, True)]], 1: [[CoedgeUse(1, 0, False)]]},
    )
    solid, mapping = assemble_brep(graph, {0: curve}, {0: cone_patch, 1: sphere_patch}, cfg, tmp_path)
    assert BRepCheck_Analyzer(solid).IsValid()
    assert not any(
        item["kind"] == "explicit_periodic_seam"
        for item in mapping["postprocess_edges"]
    )

