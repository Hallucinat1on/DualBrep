from types import SimpleNamespace

import numpy as np

from bandbrep.vendor.dualbrep_segmentation import segment_faces


def test_connected_first_pass_component_stays_whole():
    mesh = SimpleNamespace(face_adjacency=np.array([[0, 1], [1, 2], [2, 3], [3, 4]]))
    count, labels, reasons, report = segment_faces(
        mesh, np.array([.02, .02, .003, .02, .02]), .003, 2, .003
    )
    assert count == 1
    np.testing.assert_array_equal(labels, [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(reasons, [0, 0, 0, 0, 0])
    assert report["second_pass_enabled"] is False


def test_threshold_and_component_filters_keep_distinct_rejection_reasons():
    mesh = SimpleNamespace(face_adjacency=np.array([[0, 1], [1, 2], [2, 3], [3, 4]]))
    count, labels, reasons, _ = segment_faces(
        mesh, np.array([.003, .003, .001, .008, .001]), .003, 2, .003
    )
    assert count == 0
    np.testing.assert_array_equal(labels, [-1, -1, -1, -1, -1])
    np.testing.assert_array_equal(reasons, [4, 4, 1, 3, 1])
