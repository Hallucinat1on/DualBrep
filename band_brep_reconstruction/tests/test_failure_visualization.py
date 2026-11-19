"""Scientific display must retain rare labels and source IDs when reducing data."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'benchmarks/visualize_reconstruction_failures.py'
spec = importlib.util.spec_from_file_location('failure_visualization', SCRIPT)
viz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viz)


def test_sampling_preserves_rare_contamination_and_is_reproducible():
    labels = np.r_[np.zeros(5000, int), np.ones(3, int), np.full(1, 2)]
    ids = viz.stratified_indices(labels, 120)
    assert len(ids) == 120
    assert len(np.unique(ids)) == 120
    assert set(labels[ids]) == {0, 1, 2}
    np.testing.assert_array_equal(ids, viz.stratified_indices(labels, 120))
    with pytest.raises(ValueError):
        viz.stratified_indices(labels, 2)


def test_roi_selection_retains_source_triangle_indices_and_rare_face():
    vertices = np.vstack([[[i, 0, 0], [i, .1, 0], [i, 0, .1]] for i in range(10)])
    triangles = np.arange(30).reshape(-1, 3)
    labels = np.array([0] * 8 + [7, 9])
    ids = viz.triangle_subset(vertices, triangles, labels, np.array([[-1,-1,-1],[8.5,1,1]]), 3)
    assert np.all(ids < 9)
    assert 8 in ids
    assert set(labels[ids]) == {0, 7}


def test_displacement_vectors_use_null_breaks_and_keep_original_positions():
    a = np.array([[0., 0., 0.], [1., 2., 3.]])
    b = a + .01
    trace = viz.connectors(a, b, 'error')
    assert trace['x'][2] is None and trace['x'][5] is None
    np.testing.assert_allclose(trace['meta']['starts'], a)
    np.testing.assert_allclose(trace['meta']['ends'], b)
    import json
    json.dumps(trace, allow_nan=False)


def test_mesh_compaction_keeps_original_geometry():
    v = np.array([[0,0,0], [1,0,0], [0,1,0], [7,7,7]])
    t = viz.mesh_trace(v, np.array([[2,0,1]]), 'F9', '#ffffff')
    pts = np.c_[t['x'], t['y'], t['z']]
    np.testing.assert_array_equal(pts[np.c_[t['i'], t['j'], t['k']]], v[[[2,0,1]]])
    assert viz.mesh_trace(v, np.empty((0,3), int), 'empty', '#ffffff') is None
