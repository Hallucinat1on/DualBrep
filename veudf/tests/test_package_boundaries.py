"""The data package must work without loading a model or the CAD kernel."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from veudf.io.predictions import read_prediction, write_prediction


def test_data_interfaces_do_not_import_training_or_cad(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {str(Path.cwd() / 'veudf/src')!r}); import veudf.schema, veudf.utils; "
            "import veudf.geometry.mesh, veudf.io.condition, veudf.io.dataset; "
            "import veudf.io.predictions, veudf.postprocess.mesh_cleanup; "
            "assert not any(n.split('.')[0] in {'torch', 'hy3dshape', 'OCC'} "
            "for n in sys.modules)",
        ],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_prediction_roundtrip_preserves_masks_and_coordinates(tmp_path):
    arrays = dict(
        query_points=np.array([[0, 0, 0], [0.5, 0.4, 0.3]], np.float32),
        query_fields=np.array([[0.01, 0.02], [np.nan, 0.3]], np.float32),
        valid=np.array([[True, True], [False, False]]),
        saturated=np.zeros((2, 2), bool),
        presence_support=np.array([20, 10]),
        has_edges=np.array(True),
        has_vertices=np.array(True),
    )
    metadata = dict(
        schema_version="edge_vertex_surface_prediction_v1",
        channel_names=["dE", "dV"],
        normalization=dict(center=[1, 2, 3], scale=0.5),
    )
    output = tmp_path / "prediction"
    write_prediction(output, arrays, metadata)
    actual, actual_meta = read_prediction(output)
    for name, value in arrays.items():
        np.testing.assert_array_equal(actual[name], value)
    assert actual_meta == metadata
    with pytest.raises(FileExistsError):
        write_prediction(output, arrays, metadata)


def test_prediction_rejects_misaligned_masks_before_writing(tmp_path):
    with pytest.raises(ValueError, match="misaligned"):
        write_prediction(
            tmp_path / "invalid",
            dict(
                query_points=np.zeros((2, 3)),
                query_fields=np.zeros((2, 2)),
                valid=np.ones((1, 2), bool),
                saturated=np.zeros((2, 2), bool),
            ),
            dict(
                schema_version="edge_vertex_surface_prediction_v1",
                channel_names=["dE", "dV"],
            ),
        )
    assert not (tmp_path / "invalid").exists()
