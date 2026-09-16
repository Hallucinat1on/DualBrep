"""Asset scanning, STEP/OBJ pairing, stable splits, and manifest IO."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .common import (ASSET_RE, SCHEMA_VERSION, Asset, atomic_write_text,
                     sha256_file, write_json, write_jsonl)


def parse_asset(path: Path, expected_kind: str) -> Asset | None:
    match = ASSET_RE.match(path.name)
    if match is None or match.group("kind").lower() != expected_kind:
        return None
    return Asset(
        shape_id=match.group("shape_id"),
        token=match.group("token"),
        part=match.group("part"),
        kind=expected_kind,
        path=path.resolve(),
    )


def scan_assets(root: Path, kind: str) -> list[Asset]:
    suffixes = {"step": {".step", ".stp"}, "trimesh": {".obj"}}[kind]
    assets: list[Asset] = []
    if not root.is_dir():
        return assets
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            asset = parse_asset(path, kind)
            if asset is not None:
                assets.append(asset)
    return sorted(assets, key=lambda item: (item.pair_key, str(item.path)))


def stable_split(group_id: str, seed: int, train_ratio: float,
                 val_ratio: float) -> str:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def build_inventory(args: argparse.Namespace) -> list[dict[str, Any]]:
    abc_root = Path(args.abc_root).resolve()
    output_root = Path(args.output_root).resolve()
    steps = scan_assets(abc_root / "step", "step")
    objs = scan_assets(abc_root / "obj", "trimesh")

    by_step: dict[tuple[str, str, str], list[Asset]] = {}
    by_obj: dict[tuple[str, str, str], list[Asset]] = {}
    for asset in steps:
        by_step.setdefault(asset.pair_key, []).append(asset)
    for asset in objs:
        by_obj.setdefault(asset.pair_key, []).append(asset)

    records: list[dict[str, Any]] = []
    for key in sorted(set(by_step) | set(by_obj)):
        step_items, obj_items = by_step.get(key, []), by_obj.get(key, [])
        status = "paired"
        if not step_items:
            status = "missing_step"
        elif not obj_items:
            status = "missing_obj"
        elif len(step_items) != 1 or len(obj_items) != 1:
            status = "pair_conflict"
        shape_id, token, part = key
        group_id = token
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "shape_id": shape_id,
            "chunk": f"{int(shape_id) // 10000:04d}",
            "token": token,
            "part": part,
            "group_id": group_id,
            "split": stable_split(group_id, args.seed, args.train_ratio, args.val_ratio),
            "pair_status": status,
            "mesh_source": "abc_obj" if obj_items else None,
            "step_path": str(step_items[0].path) if len(step_items) == 1 else None,
            "obj_path": str(obj_items[0].path) if len(obj_items) == 1 else None,
            "step_candidates": [str(item.path) for item in step_items],
            "obj_candidates": [str(item.path) for item in obj_items],
        }
        if len(step_items) == 1:
            row["step_bytes"] = step_items[0].path.stat().st_size
        if len(obj_items) == 1:
            row["obj_bytes"] = obj_items[0].path.stat().st_size
        if args.hash_inputs and status == "paired":
            row["step_sha256"] = sha256_file(step_items[0].path)
            row["obj_sha256"] = sha256_file(obj_items[0].path)
        records.append(row)

    write_jsonl(output_root / "manifest.jsonl", records)
    split_root = output_root / "splits"
    for split in ("train", "val", "test"):
        ids = [row["shape_id"] for row in records
               if row["pair_status"] == "paired" and row["split"] == split]
        atomic_write_text(split_root / f"{split}.txt", "".join(f"{item}\n" for item in ids))
    counts: dict[str, int] = {}
    for row in records:
        counts[row["pair_status"]] = counts.get(row["pair_status"], 0) + 1
    write_json(output_root / "reports" / "inventory.json", {
        "schema_version": SCHEMA_VERSION,
        "abc_root": str(abc_root),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "step_files": len(steps),
        "obj_files": len(objs),
        "pair_counts": counts,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "hash_inputs": bool(args.hash_inputs),
    })
    write_json(output_root / "config.json", {
        "schema_version": SCHEMA_VERSION,
        "abc_root": str(abc_root),
        "output_root": str(output_root),
        "tau": getattr(args, "tau", 0.015),
        "seed": args.seed,
    })
    print(f"inventory: {len(steps)} STEP, {len(objs)} OBJ, "
          f"{counts.get('paired', 0)} exact pairs")
    return records


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows
