import numpy as np

from bandbrep.config import Config
from bandbrep.curve_fitting import CurveRecord
from bandbrep.edge_first_embedding import cut_surface
from bandbrep.edge_first_graph import add_short_paths, fit_paths
from bandbrep.edge_first_grid import GridSamples
from bandbrep.edge_first_mc import extract_shell
from bandbrep.schemas import EdgeInstance, LogicalGraph, VertexNode
from helpers import AnalyticFields, box


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


def test_short_edge_keeps_side_observations_even_without_fine_grid_support():
    length = .0025
    vertices = [VertexNode(i, np.array([x, 0., 0.]), 0., .001)
                for i, x in enumerate((0., length))]

    def fields(points):
        delta = points - np.column_stack((np.clip(points[:, 0], 0, length), np.zeros((len(points), 2))))
        return np.column_stack((points[:, 2], np.linalg.norm(delta, axis=1),
                                np.minimum(np.linalg.norm(points, axis=1),
                                           np.linalg.norm(points - vertices[1].position, axis=1))))

    provider = AnalyticFields(fields)
    cfg = Config()
    paths = []
    add_short_paths(paths, vertices, provider, cfg)
    assert len(paths) == 1 and paths[0].observations[0].shape == (35, 3)
    grid = GridSamples(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)),
                       np.empty((0, 3)), np.empty((0, 3), bool), np.empty((0,)), {})
    unresolved = []
    graph, curves = fit_paths(paths, vertices, grid, provider, cfg, unresolved)
    assert not unresolved and graph.edges[0].vertices == [0, 1]
    np.testing.assert_allclose(curves[0].evaluate(np.array([0., 1.])),
                               [v.position for v in vertices], atol=1e-10)


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
    cut, labels, barriers = cut_surface(mesh, oracle, graph, curves, Config())
    assert graph.face_count == 3
    assert set(barriers[:, 0]) == set(range(8))
    centers = cut.vertices[cut.triangles].mean(1)
    middle = abs(centers[:, 2]) < .00075
    assert middle.any() and len(np.unique(labels[middle])) == 1
    assert (labels == labels[middle][0]).sum() == middle.sum()
