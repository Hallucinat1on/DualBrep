import numpy as np
import pytest

from bandbrep.config import Config
from bandbrep.curve_fitting import CurveRecord
from bandbrep.edge_first_embedding import _simplify_segments, cut_surface
from bandbrep.edge_first_graph import Path, fit_geometry
from bandbrep.edge_first_grid import GridSamples
from bandbrep.edge_first_mc import extract_shell
from bandbrep.fast_faces import project_wireframe
from bandbrep.schemas import EdgeInstance, LogicalGraph, ReconstructionError, VertexNode
from bandbrep.shell_geometry import fit_sphere
from helpers import box


def test_missing_mc_corners_never_create_a_cap():
    keys = np.indices((3, 3, 3)).reshape(3, -1).T
    points = keys * .003
    values = np.zeros((27, 3))
    values[:, 1] = points[:, 1]
    valid = np.ones((27, 3), bool)
    valid[keys[:, 0] == 2, 1] = False
    samples = GridSamples(np.zeros((1, 3), int), keys, points, values, valid,
                          np.arange(27).reshape(1, 3, 3, 3), {})
    vertices, faces, report = extract_shell(samples, Config(coarse_spacing=.006))
    assert len(faces) == 4 and report['clipped_cells'] > 0
    np.testing.assert_allclose(vertices[:, 1], .005, atol=1e-8)
    assert vertices[:, 0].max() <= .003 + 1e-8


def test_fit_sub_grid_edge_and_simplify_backtrack():
    ends = np.array([[0., 0., 0.], [.0025, 0., 0.]])
    support = np.linspace(ends[0], ends[1], 7)
    curve = fit_geometry(0, Path([0, 1], support, False), ends, Config())
    assert curve.kind == 'line'
    np.testing.assert_array_equal(curve.evaluate(np.array([0., 1.])), ends)
    segs = [(0, 1, 2), (1, 2, 3), (1, 3, 2), (0, 2, 4)]
    assert _simplify_segments(segs) == [(0, 1, 2), (0, 2, 4)]


def test_v_shell_sphere_radius_is_not_fixed_to_level():
    import trimesh
    xyz=trimesh.creation.icosphere(subdivisions=2,radius=.003).vertices
    center=np.array([.22,-.15,.13])
    found,radius,error,coverage=fit_sphere(xyz+center,np.ones(len(xyz)))
    np.testing.assert_allclose(found,center,atol=1e-10)
    assert abs(radius-.003)<1e-10 and error<1e-10 and coverage<.1


def test_explicit_cut_preserves_strip_narrower_than_fine_grid():
    from veudf.geometry.mesh import MeshSDFOracle

    mesh, _ = box()
    oracle = MeshSDFOracle(mesh.vertices, mesh.triangles)
    corners = [[-.55, -.55], [.55, -.55], [.55, .55], [-.55, .55]]
    positions = np.array([xy + [z] for z in (-.00075, .00075) for xy in corners])
    graph = LogicalGraph(0, [VertexNode(i, p, 0., .001) for i, p in enumerate(positions)], [])
    curves = {}
    for eid, p in enumerate(positions):
        following = 4 * (eid // 4) + (eid + 1) % 4
        q = positions[following]
        graph.edges.append(EdgeInstance(eid, (), np.stack((p, q)), False, [eid, following], -1, .001))
        curves[eid] = CurveRecord(eid, 'line', dict(p0=p, p1=q), 0.)
    with pytest.raises(ReconstructionError, match='no distinct nearby mesh anchors'):
        project_wireframe(mesh, graph, curves, Config())
    cut, labels, barriers = cut_surface(mesh, oracle, graph, curves, Config())
    assert graph.face_count == 3
    assert set(barriers[:, 0]) == set(range(8))
    centers = cut.vertices[cut.triangles].mean(1)
    middle = abs(centers[:, 2]) < .00075
    assert middle.any() and len(np.unique(labels[middle])) == 1
    assert (labels == labels[middle][0]).sum() == middle.sum()


def test_patch_fallback_adds_internal_seam_and_preserves_logical_faces(tmp_path, monkeypatch):
    from itertools import combinations, product

    from bandbrep import face_patches
    from bandbrep.edge_first import export
    from bandbrep.field_provider import MeshFieldProvider
    from bandbrep.schemas import ReconstructionError

    mesh, udf = box()
    cfg = Config(face_samples=80)
    provider = MeshFieldProvider(mesh,udf,cfg)
    coords = np.asarray(list(product((-.55,.55),repeat=3)))
    graph = LogicalGraph(0,[VertexNode(i,p,0.,.006) for i,p in enumerate(coords)],[])
    curves = {}
    for a,b in combinations(range(len(coords)),2):
        if np.count_nonzero(coords[a]!=coords[b]) == 1:
            eid = len(graph.edges)
            graph.edges.append(EdgeInstance(eid,(),coords[[a,b]],False,[a,b],-1,.006))
            curves[eid]=CurveRecord(eid,'line',dict(p0=coords[a],p1=coords[b]),0.)
    cut,labels,constraints=cut_surface(mesh,provider.oracle,graph,curves,cfg)
    real_fit=face_patches.initialize_face_surfaces

    def fail_unsplit(*args):
        if args[2].face_count==6:
            raise ReconstructionError('surface_boundary_incompatible','force patch retry',{'face':0})
        return real_fit(*args)

    monkeypatch.setattr(face_patches,'initialize_face_surfaces',fail_unsplit)
    patches,revised,records,mapping,seams=face_patches.fit_with_patch_fallback(
        cut,labels,constraints,graph,curves,provider,cfg)
    assert graph.face_count==6 and revised.face_count==7
    assert len(revised.edges)==len(graph.edges)+1 and len(seams)==1
    assert mapping[-1]==0
    assert export(revised,records,patches,cfg,tmp_path)[2]['roundtrip_checked']


def test_dyadic_scout_spacings_and_feature_collapse():
    from bandbrep.edge_first_graph import _collapse_close_vertices
    cfg = Config()
    assert cfg.scout_spacings() == (0.024, 0.012, 0.006, 0.003)
    verts = [VertexNode(0, np.array([0., 0., 0.]), 0., .002),
             VertexNode(1, np.array([.001, 0., 0.]), 0., .002),
             VertexNode(2, np.array([.1, 0., 0.]), 0., .002)]
    paths = [Path([0, 1], np.array([[0., 0., 0.], [.001, 0., 0.]]), False),
             Path([1, 2], np.array([[.001, 0., 0.], [.1, 0., 0.]]), False)]
    new_v, new_p, collapsed = _collapse_close_vertices(verts, paths, Config(allow_feature_collapse=True))
    assert len(new_v) == 2 and len(new_p) == 1 and len(collapsed) == 1
    assert new_p[0].nodes == [0, 1]


def test_split_cached_v_minima_separates_close_vertices():
    from bandbrep.shell_geometry import _split_cached_v_minima
    h = 0.003
    coords = np.mgrid[-0.012:0.024:h, -0.012:0.012:h, -0.012:0.012:h].reshape(3, -1).T
    c0, c1 = np.array([0., 0., 0.]), np.array([0.0119, 0., 0.])
    d = np.minimum(np.linalg.norm(coords - c0, axis=1), np.linalg.norm(coords - c1, axis=1))
    pts_box = np.vstack((c0, c1))
    recovered = _split_cached_v_minima(pts_box, coords, d, h, 0.005)
    assert len(recovered) == 2
    found = np.sort([r[0][0] for r in recovered])
    np.testing.assert_allclose(found, [0.0, 0.0119], atol=5e-4)

