"""Both extraction routes must produce a strictly shared edge graph and STEP."""
import numpy as np
import pytest

from bandbrep.config import Config
from bandbrep.edge_first import reconstruct
from helpers import box, cylinder


@pytest.mark.parametrize('extraction', ['trace', 'mc'])
@pytest.mark.parametrize('factory,expected', [(box, (6, 12, 8)), (cylinder, (3, 2, 0))])
def test_edge_first_step(tmp_path, extraction, factory, expected):
    mesh, udf = factory()
    result = reconstruct(mesh, udf, Config(extraction=extraction, face_samples=200), tmp_path)
    graph = result['graph']
    assert (graph.face_count, len(graph.edges), len(graph.vertices)) == expected
    assert result['report']['stages']['cad']['roundtrip_checked']
    assert (tmp_path / 'reconstruction.step').is_file()
    assert (tmp_path / 'edge_shell.npz').exists() == (extraction == 'mc')
    for edge in graph.edges:
        assert len(set(edge.faces)) == 2
        if edge.vertices:
            np.testing.assert_allclose(
                result['curves'][edge.id].evaluate(np.array([0., 1.])),
                [graph.vertices[i].position for i in edge.vertices], atol=1e-8,
            )
    if extraction == 'mc':
        assert result['report']['stages']['wireframe']['shell']['triangles'] > 0
