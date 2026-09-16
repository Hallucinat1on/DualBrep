"""Command-line interface for the surface-field data pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import DEFAULT_ABC_ROOT, DEFAULT_OUTPUT_ROOT, DEFAULT_VORONOI_EXE
from .inventory import build_inventory
from .pipeline import process_manifest


def add_inventory_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--abc-root", type=Path, default=DEFAULT_ABC_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--hash-inputs", action="store_true",
                        help="hash every paired input during inventory (slow)")


def add_process_arguments(parser: argparse.ArgumentParser, *, include_output: bool = True,
                          include_seed: bool = True) -> None:
    if include_output:
        parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--voronoi-exe", type=Path, default=DEFAULT_VORONOI_EXE)
    parser.add_argument("--voronoi-backend", choices=("cpp", "scipy"), default="cpp")
    parser.add_argument("--allow-experimental-voronoi", action="store_true")
    parser.add_argument("--scipy-voronoi-points", type=int, default=4000)
    parser.add_argument("--scipy-min-face-points", type=int, default=12)
    parser.add_argument("--step-deflection", type=float, default=0.001)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", help="comma-separated eight-digit shape IDs")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--tau", type=float, default=0.015)
    parser.add_argument("--condition-points", type=int, default=32768)
    if include_seed:
        parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--distance-backend", choices=("auto", "igl", "open3d", "naive"),
                        default="auto")
    parser.add_argument("--query-batch-size", type=int, default=65536)
    parser.add_argument("--alignment-samples", type=int, default=2048)
    parser.add_argument("--max-bbox-center-error", type=float, default=0.02)
    parser.add_argument("--max-bbox-extent-relative", type=float, default=0.05)
    parser.add_argument("--lipschitz-tolerance", type=float, default=2e-5)
    parser.add_argument("--max-faces", type=int, default=0)
    parser.add_argument("--allow-open-mesh", action="store_true")
    parser.add_argument("--skip-step-precheck", action="store_true",
                        help="testing only; do not use to publish data")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="archive and replace an incomplete/old per-shape output")


def validate_args(args: argparse.Namespace) -> None:
    if hasattr(args, "train_ratio"):
        if not 0 < args.train_ratio < 1 or not 0 <= args.val_ratio < 1:
            raise SystemExit("invalid split ratios")
        if args.train_ratio + args.val_ratio >= 1:
            raise SystemExit("train_ratio + val_ratio must be < 1")
    if hasattr(args, "tau") and args.tau <= 0:
        raise SystemExit("--tau must be positive")
    for name in ("condition_points", "query_batch_size", "alignment_samples"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    for name in ("scipy_voronoi_points", "scipy_min_face_points"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="surface_field",
        description=("Build mesh-conditioned Voronoi-UDF surface-field data from ABC "
                     "STEP/OBJ pairs.  Run from the repository root as "
                     "'python -m Voronoi.surface_field ...'."),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory", help="scan and pair ABC STEP/OBJ files")
    add_inventory_arguments(inventory)
    process = commands.add_parser("process", help="process paired rows from a manifest")
    add_process_arguments(process)
    run = commands.add_parser("run", help="build inventory, then process selected pairs")
    add_inventory_arguments(run)
    add_process_arguments(run, include_output=False, include_seed=False)
    args = parser.parse_args()
    validate_args(args)
    if args.command == "inventory":
        build_inventory(args)
    elif args.command == "process":
        process_manifest(args)
    else:
        build_inventory(args)
        args.manifest = Path(args.output_root) / "manifest.jsonl"
        process_manifest(args)
