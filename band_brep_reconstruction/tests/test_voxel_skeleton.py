"""Thinning keeps anchored branches and cut-ball ports determine incidence."""
from itertools import product

import numpy as np

from bandbrep.config import Config
from bandbrep.edge_first_grid import GridSamples
from bandbrep.schemas import VertexNode
from bandbrep.skeleton_paths import trace_paths
from bandbrep.voxel_skeleton import thin_volume


def test_thinning_keeps_branches_of_y_junction():
    cfg = Config()
    h = cfg.fine_spacing
    keys = np.array(list(product(range(-4, 15), repeat=3)))
    points = keys * h
    ends = [np.zeros(3), *(np.eye(3) * 10 * h)]
    distance = []
    for end in ends[1:]:
        t = np.clip(points @ end / (end @ end), 0., 1.)
        distance.append(np.linalg.norm(points - t[:, None] * end, axis=1))
    values = np.column_stack((np.zeros(len(keys)), np.min(distance, axis=0), np.zeros(len(keys))))
    grid = GridSamples(np.empty((0, 3)), keys, points, values, np.ones_like(values, bool), None, {})
    vertices = [VertexNode(i, p, 0., 2*h) for i, p in enumerate(ends)]
    positions, links, report = thin_volume(grid, vertices, cfg)
    paths, unresolved = trace_paths(positions, links, vertices)
    assert not unresolved
    assert {tuple(sorted(path.nodes)) for path in paths} == {(0, 1), (0, 2), (0, 3)}
    assert report['protected_vertices'] == 4
    assert report['skeleton_nodes'] < report['foreground'] / 4
    for path in paths:
        for vertex_id, port in zip(path.nodes, path.ports):
            v = vertices[vertex_id]
            np.testing.assert_allclose(np.linalg.norm(port - v.position), v.support_radius, atol=1e-10)


def test_short_link_between_two_cut_balls_has_two_ports():
    points = np.array([[0., 0., 0.], [.003, 0., 0.]])
    vertices = [VertexNode(i, p, 0., .001) for i, p in enumerate(points)]
    paths, unresolved = trace_paths(points, np.array([[0, 1]]), vertices)
    assert not unresolved
    assert len(paths) == 1
    assert paths[0].nodes == [0, 1]
    np.testing.assert_allclose(paths[0].points, [[.001, 0., 0.], [.002, 0., 0.]])


def test_thinning_does_not_treat_unknown_neighbors_as_exterior():
    cfg = Config()
    keys = np.array(list(product(range(-2, 3), repeat=3)))
    values = np.ones((len(keys), 3))
    values[np.all(keys == 0, axis=1), 1] = 0.
    valid = np.ones_like(values, bool)
    valid[np.all(keys == [1, 0, 0], axis=1), 1] = False
    grid = GridSamples(np.empty((0, 3)), keys, keys * cfg.fine_spacing, values, valid, None, {})
    points, links, report = thin_volume(grid, [], cfg)
    assert report['unknown_interface'] == 1
    assert not len(links)
    np.testing.assert_array_equal(points, [[0., 0., 0.]])
