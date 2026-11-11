#!/usr/bin/env python3
"""Sample points on a reconstructed SDF mesh and color them by UDF.

The UDF scalar is taken from ``udf_g.npy`` produced by ``ae_reconstruct.py``:
one value per ``recon_sdf.ply`` triangle.  ``recon_udf.ply`` is *not* a scalar
field; it is the UDF iso-surface at ``runtime.udf_threshold`` and therefore
cannot be used directly as a per-point UDF value.

Example
-------
python visualize_sdf_udf.py \
    --recon-dir output_udf_demo/recon/00003291_3 \
    --num-points 200000 \
    --out output_udf_demo/udf_points/00003291_3.ply
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample recon_sdf.ply and write an RGB+UDF point cloud PLY."
    )
    p.add_argument(
        "--recon-dir",
        type=Path,
        help="Directory containing recon_sdf.ply and udf_g.npy.",
    )
    p.add_argument("--sdf", type=Path, help="Path to recon_sdf.ply.")
    p.add_argument(
        "--udf-npy",
        type=Path,
        help="Path to udf_g.npy (one value per SDF triangle).",
    )
    p.add_argument("--out", type=Path, required=True, help="Output .ply path.")
    p.add_argument(
        "--num-points",
        type=int,
        default=200000,
        help="Number of surface samples (default: 200000).",
    )
    p.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    p.add_argument(
        "--cmap",
        default="turbo",
        help="Matplotlib colormap name (default: turbo).",
    )
    p.add_argument(
        "--clip",
        type=float,
        default=0.015,
        help="Saturation value for UDF color mapping (default: 0.015).",
    )
    p.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Color-map minimum. Default: 0.",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Color-map maximum. Default: --clip.",
    )
    p.add_argument(
        "--scale",
        choices=("linear", "log"),
        default="linear",
        help="Color scaling (default: linear).",
    )
    p.add_argument(
        "--log-offset",
        type=float,
        default=1e-5,
        help="Offset for --scale log (default: 1e-5).",
    )
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.recon_dir is not None:
        sdf = args.recon_dir / "recon_sdf.ply"
        udf = args.recon_dir / "udf_g.npy"
    else:
        if args.sdf is None or args.udf_npy is None:
            raise SystemExit(
                "Provide either --recon-dir, or both --sdf and --udf-npy."
            )
        sdf, udf = args.sdf, args.udf_npy

    if not sdf.exists():
        raise SystemExit(f"recon_sdf.ply not found: {sdf}")
    if not udf.exists():
        raise SystemExit(
            f"udf_g.npy not found: {udf}\n"
            "recon_udf.ply is an iso-surface, not a scalar field. "
            "Use the udf_g.npy written by ae_reconstruct.py."
        )
    return sdf, udf


def sample_surface(mesh, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted surface sampling with deterministic face indices."""
    import trimesh

    try:
        points, face_ids = trimesh.sample.sample_surface(
            mesh, count, seed=seed
        )
    except TypeError:
        # Older trimesh versions do not accept seed=.
        np.random.seed(seed)
        points, face_ids = mesh.sample(count, return_index=True)
    return np.asarray(points, np.float64), np.asarray(face_ids, np.int64)


def colorize(
    values: np.ndarray,
    vmin: float,
    vmax: float,
    cmap_name: str,
    scale: str,
    log_offset: float,
) -> np.ndarray:
    """Return uint8 RGB colors for scalar values."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    if vmax <= vmin:
        raise SystemExit(f"vmax ({vmax}) must be greater than vmin ({vmin})")

    if scale == "log":
        if vmin < 0 or log_offset <= 0:
            raise SystemExit("log scale requires vmin >= 0 and log_offset > 0")
        transform = mcolors.LogNorm(vmin=max(vmin, log_offset), vmax=vmax)
    else:
        transform = mcolors.Normalize(vmin=vmin, vmax=vmax)

    mapper = cm.ScalarMappable(norm=transform, cmap=cmap_name)
    rgb = mapper.to_rgba(np.clip(values, vmin, vmax))[:, :3]
    # ScalarMappable returns float RGBA in [0, 1].
    if rgb.max() <= 1.0 + 1e-8:
        rgb = rgb * 255.0
    return np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)


def write_ply(
    path: Path,
    xyz: np.ndarray,
    udf: np.ndarray,
    rgb: np.ndarray,
) -> None:
    """Write a binary little-endian PLY with xyz, udf, and RGB."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("udf", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    data = np.empty(n, dtype=dtype)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["udf"] = udf
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"comment generated by visualize_sdf_udf.py",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property float udf",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
            "",
        ]
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def main() -> None:
    args = parse_args()
    sdf_path, udf_path = resolve_paths(args)

    import trimesh

    mesh = trimesh.load(sdf_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise SystemExit(f"Not a triangle mesh: {sdf_path}")

    udf_per_face = np.load(udf_path)
    if udf_per_face.ndim != 1:
        udf_per_face = udf_per_face.reshape(-1)
    if len(udf_per_face) != len(mesh.faces):
        raise SystemExit(
            f"udf_g length {len(udf_per_face)} != mesh faces {len(mesh.faces)}"
        )

    points, face_ids = sample_surface(mesh, args.num_points, args.seed)
    udf_values = udf_per_face[face_ids].astype(np.float64)

    vmin = 0.0 if args.vmin is None else args.vmin
    vmax = args.clip if args.vmax is None else args.vmax
    colors = colorize(
        udf_values,
        vmin=vmin,
        vmax=vmax,
        cmap_name=args.cmap,
        scale=args.scale,
        log_offset=args.log_offset,
    )

    # Keep xyz float32 in the PLY for smaller files.
    write_ply(args.out, points.astype(np.float32), udf_values.astype(np.float32), colors)

    print(f"sdf       : {sdf_path}")
    print(f"udf       : {udf_path}")
    print(f"output    : {args.out}")
    print(f"points    : {len(points)}")
    print(
        "udf stats : "
        f"min={udf_values.min():.6f} "
        f"p50={np.quantile(udf_values, 0.5):.6f} "
        f"p95={np.quantile(udf_values, 0.95):.6f} "
        f"max={udf_values.max():.6f}"
    )
    print(f"color     : {args.cmap}, vmin={vmin:g}, vmax={vmax:g}, {args.scale}")


if __name__ == "__main__":
    main()
