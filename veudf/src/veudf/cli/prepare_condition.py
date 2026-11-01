#!/usr/bin/env python3
"""Prepare an independent geometry-only condition from a closed triangle mesh."""

import argparse

import numpy as np
import trimesh
from veudf.io.condition import write_condition


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--surface-count", type=int, default=393216)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--already-normalized", action="store_true")
    args = p.parse_args(argv)
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    # Match the fixed CAD weld precision; never fill holes or change winding.
    mesh.merge_vertices(digits_vertex=7)
    mesh.remove_unreferenced_vertices()
    v, f = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)
    normalization = {}
    if not args.already_normalized:
        center = (v.max(0) + v.min(0)) / 2
        scale = 1.8 / np.ptp(v, axis=0).max()
        v = (v - center) * scale
        normalization = dict(center=center.tolist(), scale=float(scale))
    if not np.isfinite(v).all() or np.abs(v).max() > 0.900001 or args.surface_count < 1:
        raise ValueError(
            "Expected valid mesh inside [-0.9,0.9]^3 and positive surface count"
        )
    write_condition(args.output, (v, f), args.surface_count, args.seed, normalization)
    print(args.output)


if __name__ == "__main__":
    main()
