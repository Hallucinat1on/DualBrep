#!/usr/bin/env python3
"""Export processed `edge_vertex_neighborhood_pools_v3` assets as PLY.

The generated files can be opened in MeshLab, CloudCompare, Blender, etc.

Layout for one asset::

    <output>/<asset_id>/mesh.ply
    <output>/<asset_id>/combined_<channel>.ply
    <output>/<asset_id>/combined_pools.ply
    <output>/<asset_id>/pools/<pool>_<channel>.ply

``combined_<channel>.ply`` colors distance over all pools, so different
channels produce different files.  ``combined_pools.ply`` uses fixed
colors for pool membership instead of distance.

Channels are ``abs_s_mesh``, ``dE`` and ``dV``.  For ``dE``/``dV`` the
displayed range is the stored truncated UDF range (default 0.05).  For
``abs_s_mesh`` the default display range is 0.03.

Examples::

    PYTHONPATH=veudf/src python veudf/scripts/visualize_assets.py \
        --root /path/to/data --limit 3 --output /path/to/viz

    PYTHONPATH=veudf/src python veudf/scripts/visualize_assets.py \
        --root /path/to/data \
        --asset 0000_...-41577b4d4473 \
        --asset 0000_...-22a372b4fa4e \
        --output /path/to/viz --max-points 50000
"""

import argparse
import json
from pathlib import Path

import numpy as np

from veudf.visualization import distance_colors, write_ply

POOLS = (
    "surface",
    "near_surface",
    "edge_neighborhood",
    "vertex_neighborhood",
    "edge_zero",
    "vertex_zero",
)
POOL_COLORS = {
    "surface": (55, 110, 235),
    "near_surface": (0, 190, 235),
    "edge_neighborhood": (40, 190, 90),
    "vertex_neighborhood": (255, 155, 35),
    "edge_zero": (225, 55, 55),
    "vertex_zero": (190, 70, 220),
}
CHANNELS = ("abs_s_mesh", "dE", "dV")
CHANNEL_INDEX = {"abs_s_mesh": 0, "dE": 1, "dV": 2}


def write_mesh_ply(path, vertices, faces, color=(170, 170, 170)):
    """Write a binary little-endian triangle mesh PLY."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    vertex_data = np.empty(
        len(vertices),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    for axis, name in enumerate("xyz"):
        vertex_data[name] = vertices[:, axis]
    vertex_data["red"] = color[0]
    vertex_data["green"] = color[1]
    vertex_data["blue"] = color[2]
    face_data = np.empty(
        len(faces), dtype=[("n", "u1"), ("vertex_indices", "<i4", (3,))]
    )
    face_data["n"] = 3
    face_data["vertex_indices"] = faces
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(("\n".join(header) + "\n").encode())
        stream.write(vertex_data.tobytes())
        stream.write(face_data.tobytes())


def sample_indices(n, maximum, rng):
    if maximum <= 0 or n <= maximum:
        return np.arange(n)
    return np.sort(rng.choice(n, size=maximum, replace=False))


def discover_assets(root, asset_ids, limit, split):
    assets_dir = root / "assets"
    if not assets_dir.is_dir():
        raise SystemExit(f"missing assets directory: {assets_dir}")
    by_id = {p.name: p for p in assets_dir.iterdir() if p.is_dir() and not p.name.startswith(".")}
    if asset_ids:
        selected = []
        for value in asset_ids:
            candidate = Path(value)
            if candidate.is_dir():
                selected.append(candidate)
            elif value in by_id:
                selected.append(by_id[value])
            else:
                raise SystemExit(f"asset not found: {value}")
        return selected
    selected = [by_id[name] for name in sorted(by_id)]
    if split:
        index_path = root / "index.json"
        if not index_path.is_file():
            raise SystemExit("--split requires index.json")
        index = json.loads(index_path.read_text())
        allowed = {a["asset_id"] for a in index["assets"] if a["split"] == split}
        selected = [p for p in selected if p.name in allowed]
    if limit:
        selected = selected[:limit]
    return selected


def export_asset(asset_dir, output, max_points, sdf_max, udf_max, scale, rng):
    asset_dir = Path(asset_dir)
    name = asset_dir.name
    out = Path(output) / name
    out.mkdir(parents=True, exist_ok=True)
    condition = asset_dir / "condition"
    vertices = np.load(condition / "vertices.npy", mmap_mode="r", allow_pickle=False)
    triangles = np.load(condition / "triangles.npy", mmap_mode="r", allow_pickle=False)
    write_mesh_ply(out / "mesh.ply", vertices, triangles)

    summary = {"asset_id": name, "source": str(asset_dir), "pools": {}}
    combined = {channel: [] for channel in CHANNELS}
    pool_colored = []

    for pool in POOLS:
        pool_dir = asset_dir / "pools" / pool
        points_path = pool_dir / "points.npy"
        fields_path = pool_dir / "fields.npy"
        if not points_path.is_file() or not fields_path.is_file():
            continue
        points = np.load(points_path, mmap_mode="r", allow_pickle=False)
        fields = np.load(fields_path, mmap_mode="r", allow_pickle=False)
        n = len(points)
        if fields.shape != (n, 3):
            raise SystemExit(f"bad fields shape for {pool}: {fields.shape} vs {n}")
        take = sample_indices(n, max_points, rng)
        xyz = np.asarray(points[take], dtype=np.float32)
        values = np.asarray(fields[take], dtype=np.float32)
        summary["pools"][pool] = {"points": int(n), "exported": int(len(take))}

        for channel in CHANNELS:
            raw = values[:, CHANNEL_INDEX[channel]]
            if channel == "abs_s_mesh":
                shown = np.abs(raw)
                maximum = sdf_max
            else:
                shown = np.clip(raw, 0, udf_max)
                maximum = udf_max
            colors = distance_colors(shown, maximum, scale=scale)
            write_ply(out / "pools" / f"{pool}_{channel}.ply", xyz, colors, distance=raw)
            combined[channel].append((xyz, shown, raw))
        pool_colored.append(
            (xyz, np.tile(np.array([POOL_COLORS[pool]], dtype=np.uint8), (len(xyz), 1)))
        )

    for channel, parts in combined.items():
        if not parts:
            continue
        xyz = np.concatenate([p[0] for p in parts], axis=0)
        shown = np.concatenate([p[1] for p in parts], axis=0)
        raw = np.concatenate([p[2] for p in parts], axis=0)
        maximum = sdf_max if channel == "abs_s_mesh" else udf_max
        colors = distance_colors(shown, maximum, scale=scale)
        write_ply(out / f"combined_{channel}.ply", xyz, colors, distance=raw)

    if pool_colored:
        xyz = np.concatenate([p[0] for p in pool_colored], axis=0)
        colors = np.concatenate([p[1] for p in pool_colored], axis=0)
        write_ply(out / "combined_pools.ply", xyz, colors)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export processed v3 asset point pools and condition mesh as PLY",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root", required=True, help="dataset root containing assets/")
    parser.add_argument("--asset", action="append", default=[], help="asset id or directory; repeatable")
    parser.add_argument("--limit", type=int, default=1, help="number of assets when --asset is absent")
    parser.add_argument("--split", choices=("train", "val", "test"), help="filter by index.json split")
    parser.add_argument("--output", required=True, help="visualization output directory")
    parser.add_argument("--max-points", type=int, default=30000, help="sample limit per pool per channel")
    parser.add_argument("--sdf-max", type=float, default=0.03, help="display range for abs(s_mesh)")
    parser.add_argument("--udf-max", type=float, default=0.05, help="display range for dE/dV")
    parser.add_argument("--scale", choices=("linear", "log"), default="linear")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.limit < 1 or args.max_points < 1 or args.udf_max <= 0 or args.sdf_max <= 0:
        parser.error("limit/max-points/sdf-max/udf-max must be positive")

    root = Path(args.root).resolve()
    assets = discover_assets(root, args.asset, args.limit, args.split)
    if not assets:
        raise SystemExit("no assets selected")
    rng = np.random.default_rng(args.seed)
    for asset in assets:
        summary = export_asset(
            asset,
            args.output,
            args.max_points,
            args.sdf_max,
            args.udf_max,
            args.scale,
            rng,
        )
        print(json.dumps({"asset_id": summary["asset_id"], "output": str(Path(args.output) / summary["asset_id"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
