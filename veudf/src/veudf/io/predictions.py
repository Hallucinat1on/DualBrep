"""Model-independent prediction interchange for field postprocessing."""

import json
from pathlib import Path

import numpy as np

PREDICTION_SCHEMA = "edge_vertex_surface_prediction_v1"
PREDICTION_CHANNELS = ["dE", "dV"]


def write_prediction(directory, arrays, metadata):
    """Write evaluated fields, masks and their coordinate-system metadata."""
    destination = Path(directory)
    validate_prediction(arrays, metadata)
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination / "fields.npz", **arrays)
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return destination / "fields.npz"


def read_prediction(directory):
    """Load model outputs without importing PyTorch or loading model weights."""
    source = Path(directory)
    metadata = json.loads((source / "metadata.json").read_text())
    with np.load(source / "fields.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    validate_prediction(arrays, metadata)
    return arrays, metadata


def validate_prediction(arrays, metadata):
    if (
        metadata.get("schema_version") != PREDICTION_SCHEMA
        or metadata.get("channel_names") != PREDICTION_CHANNELS
    ):
        raise ValueError("Unsupported prediction schema/channels")
    points = np.asarray(arrays["query_points"])
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Prediction coordinates must be finite [Q,3]")
    shape = (len(points), 2)
    fields = np.asarray(arrays["query_fields"])
    valid = np.asarray(arrays["valid"])
    saturated = np.asarray(arrays["saturated"])
    if fields.shape != shape or valid.shape != shape or saturated.shape != shape:
        raise ValueError("Prediction coordinates, fields and masks are misaligned")
    if valid.dtype != np.bool_ or saturated.dtype != np.bool_:
        raise ValueError("Prediction masks must be boolean")
    if not np.isfinite(fields[valid]).all():
        raise ValueError("Valid prediction values must be finite")
