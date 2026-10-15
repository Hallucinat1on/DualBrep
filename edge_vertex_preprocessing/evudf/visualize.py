"""Export aligned UDF observations as colored, binary PLY point clouds."""
import argparse
import json
from pathlib import Path

import numpy as np

from .io import load_fields


# Low distance -> navy, cyan, green, yellow, red -> high distance.
COLOR_STOPS = np.array([
    [30, 40, 170], [0, 180, 240], [50, 200, 110],
    [255, 220, 40], [220, 35, 35],
], dtype=np.float64)


def distance_colors(distance, maximum, scale="linear", log_offset=1e-5):
    value = np.clip(np.asarray(distance, dtype=np.float64), 0, maximum)
    if scale == "log":
        value = np.log1p(value / log_offset) / np.log1p(maximum / log_offset)
    else:
        value = value / maximum
    positions = np.linspace(0, 1, len(COLOR_STOPS))
    return np.rint(np.column_stack([
        np.interp(value, positions, COLOR_STOPS[:, channel]) for channel in range(3)
    ])).astype(np.uint8)


def write_ply(path, xyz, colors, distance=None, source=None):
    """Write little-endian PLY with RGB and optional unmodified scalar distance."""
    properties = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if distance is not None:
        properties.append(("distance", "<f4"))
    properties += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if source is not None:
        properties.append(("query_source", "u1"))
    data = np.empty(len(xyz), dtype=properties)
    for axis, name in enumerate(("x", "y", "z")):
        data[name] = xyz[:, axis]
    for channel, name in enumerate(("red", "green", "blue")):
        data[name] = colors[:, channel]
    if distance is not None:
        data["distance"] = distance
    if source is not None:
        data["query_source"] = source
    header = ["ply", "format binary_little_endian 1.0",
              "comment coordinates and distances use normalized object units",
              f"element vertex {len(data)}"]
    for name, dtype in properties:
        header.append(f"property {'float' if dtype == '<f4' else 'uchar'} {name}")
    header.append("end_header")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(("\n".join(header) + "\n").encode("ascii"))
        data.tofile(file)


def export_sample(sample_dir, output_dir, args):
    arrays, meta = load_fields(sample_dir, normalize=False)
    xyz, fields = arrays["query_points"], arrays["query_fields"]
    if not np.isfinite(xyz).all() or not np.isfinite(fields).all():
        raise ValueError("Query coordinates / field values contain nonfinite values")
    if np.any(fields[:, 1:] < 0):
        raise ValueError("UDF values must be nonnegative")
    selected = np.ones(len(xyz), dtype=bool)
    sources = arrays.get("query_source")
    if args.selection == "surface":
        if sources is None:
            raise ValueError("--selection surface requires query_source")
        selected &= sources == meta["query_sources"]["on_surface"]
    if args.surface_band is not None:
        selected &= np.abs(fields[:, 0]) <= args.surface_band
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(sample=str(Path(sample_dir).resolve()),
                  channel_order=meta["channel_names"],
                  units="normalized object coordinates / Euclidean length",
                  normalization=meta.get("normalization"),
                  selection=args.selection, surface_band=args.surface_band,
                  max_distance=args.max_distance, color_scale=args.scale,
                  log_offset=args.log_offset if args.scale == "log" else None,
                  max_points=args.max_points, seed=args.seed,
                  channels={}, references=[])
    for name, channel, presence_key in (
        ("e_udf", 1, "has_edges"), ("v_udf", 2, "has_vertices")
    ):
        # Empty-set values are placeholders, not measured distances.
        if not bool(arrays[presence_key]):
            report["channels"][name] = dict(status="absent", exported_points=0,
                                           reason="Target set is empty; placeholder channel not exported")
            continue
        mask = selected.copy()
        if args.max_distance is not None:
            mask &= fields[:, channel] <= args.max_distance
        ids = np.flatnonzero(mask)
        candidate_count = len(ids)
        if args.max_points and len(ids) > args.max_points:
            rng = np.random.default_rng(args.seed)
            ids = np.sort(rng.choice(ids, args.max_points, replace=False))
        if not len(ids):
            report["channels"][name] = dict(status="no_selected_points", exported_points=0)
            continue
        maximum = args.color_max if args.color_max is not None else float(meta["clip_values"][channel])
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError("Color range upper bound must be finite and positive")
        distance = fields[ids, channel]
        colors = distance_colors(distance, maximum, args.scale, args.log_offset)
        filename = name + ".ply"
        write_ply(output_dir / filename, xyz[ids], colors, distance,
                  sources[ids] if sources is not None else None)
        # Legend positions invert the color normalization, including log scale.
        fractions = np.linspace(0, 1, len(COLOR_STOPS))
        legend_distances = (args.log_offset * np.expm1(fractions * np.log1p(maximum / args.log_offset))
                            if args.scale == "log" else fractions * maximum)
        report["channels"][name] = dict(
            status="exported", file=filename, channel=meta["channel_names"][channel],
            selected_points=candidate_count, exported_points=len(ids),
            distance_min=float(distance.min()), distance_max=float(distance.max()),
            color_min=0.0, color_max=maximum,
            color_saturated_points=int(np.sum(distance >= maximum)),
            legend=[dict(distance=float(d), rgb=c.astype(int).tolist())
                    for d, c in zip(legend_distances, COLOR_STOPS)],
        )
    if args.references:
        for key, filename in (("edge_points", "reference_edges.ply"),
                              ("vertex_points", "reference_vertices.ply")):
            points = arrays[key][:, :3]
            if len(points):
                write_ply(output_dir / filename, points,
                          np.full((len(points), 3), 255, dtype=np.uint8))
                report["references"].append(dict(file=filename, points=len(points),
                                                  meaning="CAD target samples, not reconstructed geometry"))
    (output_dir / "visualization.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def positive_number(text):
    value = float(text)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Sample directory, fields.npz, or batch directory containing samples")
    parser.add_argument("--output", type=Path,
                        help="Output root; default: <input directory>/visualizations")
    parser.add_argument("--selection", choices=("all", "surface"), default="all",
                        help="all queries, or only queries tagged on_surface")
    parser.add_argument("--surface-band", type=positive_number,
                        help="Keep queries with abs(SDF) <= this distance")
    parser.add_argument("--max-distance", type=positive_number,
                        help="Filter each UDF separately to distances <= this value")
    parser.add_argument("--color-max", type=positive_number,
                        help="Upper color bound; default: each channel's clip value")
    parser.add_argument("--scale", choices=("linear", "log"), default="linear")
    parser.add_argument("--log-offset", type=positive_number, default=1e-5,
                        help="Distance offset for log1p color scaling")
    parser.add_argument("--max-points", type=int, default=200000,
                        help="Maximum query points per PLY; 0 exports all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--references", action="store_true",
                        help="Also export white CAD edge/vertex target samples for overlay")
    args = parser.parse_args(argv)
    if args.max_points < 0 or args.seed < 0:
        parser.error("--max-points and --seed must be nonnegative")
    path = args.input.resolve()
    if path.is_file():
        if path.name != "fields.npz":
            parser.error("input file must be fields.npz with metadata.json beside it")
        path = path.parent
    if not path.is_dir():
        parser.error("input directory does not exist")
    single = (path / "fields.npz").is_file()
    samples = [path] if single else sorted(p.parent for p in path.rglob("fields.npz"))
    if not samples:
        parser.error("no fields.npz found")
    output = args.output.resolve() if args.output else path / "visualizations"
    # Refuse overwriting a channel export from an earlier, different selection.
    destinations = [output if single else output / s.relative_to(path) for s in samples]
    occupied = [p for p in destinations if any((p / name).exists() for name in
                ("e_udf.ply", "v_udf.ply", "reference_edges.ply", "reference_vertices.ply", "visualization.json"))]
    if occupied:
        parser.error(f"visualizations already exist in {occupied[0]}; choose a new --output")
    failures = 0
    for sample, dest in zip(samples, destinations):
        try:
            report = export_sample(sample, dest, args)
            print(json.dumps(dict(sample=sample.name, output=str(dest),
                                  channels={k: dict(status=v["status"], points=v["exported_points"])
                                            for k, v in report["channels"].items()})))
        except (ValueError, KeyError, OSError) as exc:
            failures += 1
            print(json.dumps(dict(sample=sample.name, status="failed", error=str(exc))))
    return 1 if failures else 0
