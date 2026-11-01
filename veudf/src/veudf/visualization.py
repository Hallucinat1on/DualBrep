"""PLY export helpers for distance fields (no OCC dependency)."""

from pathlib import Path

import numpy as np

# Low distance -> navy, cyan, green, yellow, red -> high distance.
COLOR_STOPS = np.array(
    [[30, 40, 170], [0, 180, 240], [50, 200, 110], [255, 220, 40], [220, 35, 35]],
    dtype=np.float64,
)


def distance_colors(distance, maximum, scale="linear", log_offset=1e-5):
    value = np.clip(np.asarray(distance, dtype=np.float64), 0, maximum)
    value = (
        np.log1p(value / log_offset) / np.log1p(maximum / log_offset)
        if scale == "log"
        else value / maximum
    )
    positions = np.linspace(0, 1, len(COLOR_STOPS))
    return np.rint(
        np.column_stack(
            [np.interp(value, positions, COLOR_STOPS[:, c]) for c in range(3)]
        )
    ).astype(np.uint8)


def write_ply(path, xyz, colors, distance=None):
    """Little-endian binary PLY with RGB and optional unmodified scalar distance."""
    properties = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if distance is not None:
        properties.append(("distance", "<f4"))
    properties += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    data = np.empty(len(xyz), dtype=properties)
    for axis, name in enumerate("xyz"):
        data[name] = xyz[:, axis]
    if distance is not None:
        data["distance"] = distance
    for axis, name in enumerate(("red", "green", "blue")):
        data[name] = colors[:, axis]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(data)}"]
    header += [
        f"property {'float' if t == '<f4' else 'uchar'} {n}" for n, t in properties
    ] + ["end_header"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(("\n".join(header) + "\n").encode())
        stream.write(data.tobytes())
