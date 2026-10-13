"""Small migration-friendly reader; no OCC dependency required for training."""
import json
from pathlib import Path
import numpy as np

from . import CHANNEL_NAMES, SCHEMA_VERSION


def load_fields(directory, normalize=False):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["schema_version"] != SCHEMA_VERSION or metadata["channel_names"] != list(CHANNEL_NAMES):
        raise ValueError("Unsupported field schema / channel order")
    with np.load(directory / "fields.npz", allow_pickle=False) as file:
        arrays = {key: file[key] for key in file.files}
    if arrays["query_fields"].shape != arrays["query_points"].shape:
        raise ValueError("Queries and three-channel labels do not align")
    if normalize:
        values = arrays["query_fields"].copy()
        clips = np.asarray(metadata["clip_values"], dtype=np.float32)
        values[:, 0] = np.clip(values[:, 0], -clips[0], clips[0]) / clips[0]
        for channel, present in ((1, bool(arrays["has_edges"])), (2, bool(arrays["has_vertices"]))):
            values[:, channel] = np.minimum(values[:, channel], clips[channel]) / clips[channel] if present else 1
        arrays["query_fields"] = values
        for channel, prefix in ((1, "E"), (2, "V")):
            arrays[f"gradient_valid_{prefix}"] &= np.abs(values[:, channel]) < 1
            arrays[f"gradient_{prefix}"] /= clips[channel]
    return arrays, metadata
