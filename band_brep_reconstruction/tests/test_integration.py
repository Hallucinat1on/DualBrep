"""End-to-end geometry-only wireframe and both face reconstruction paths."""
import numpy as np
import pytest

from bandbrep.config import Config
from bandbrep.edge_first import reconstruct
from bandbrep.field_provider import MeshFieldProvider
from helpers import box, cylinder


@pytest.mark.parametrize('factory,expected,method', [
    (box, (6, 12, 8), 'coarse_projection'),
    (cylinder, (3, 2, 0), 'cut_mesh'),
])
def test_edge_first_step(tmp_path, monkeypatch, factory, expected, method):
    def forbidden(*args, **kwargs):
        pytest.fail('Default shell geometry must not evaluate UDF gradients')

    monkeypatch.setattr(MeshFieldProvider, 'gradient', forbidden)
    mesh, udf = factory()
    result = reconstruct(mesh, udf, Config(face_samples=200), tmp_path)
    graph = result['graph']
    assert (graph.face_count, len(graph.edges), len(graph.vertices)) == expected
    assert result['report']['stages']['cad']['roundtrip_checked']
    assert result['report']['face_method'] == method
    assert (tmp_path / 'reconstruction.step').is_file()
    assert (tmp_path / 'split_mesh.npz').exists() == (method == 'cut_mesh')
    assert result['report']['extraction'] == 'thinning'
    assert (tmp_path / 'edge_skeleton.npz').is_file()
    assert len(result['report']['stages']['wireframe']['skeleton']['paths']) == expected[1]
    if method == 'coarse_projection':
        with np.load(tmp_path / 'face_regions.npz') as regions:
            np.testing.assert_array_equal(regions['vertices'], mesh.vertices)
            np.testing.assert_array_equal(regions['triangles'], mesh.triangles)
            labels = regions['face_labels']
        assert len(np.unique(labels)) == 6
        for label in np.unique(labels):
            assert (labels == label).sum() == 2
    else:
        assert not result['report']['stages']['fast_faces']['accepted']
    for edge in graph.edges:
        assert len(set(edge.faces)) == 2
        if edge.vertices:
            np.testing.assert_allclose(
                result['curves'][edge.id].evaluate(np.array([0., 1.])),
                [graph.vertices[i].position for i in edge.vertices], atol=1e-8,
            )
