"""Read and validate dataset indexes and optional volume supervision."""

import json
from pathlib import Path
import numpy as np
from veudf.schema import SCHEMA


def discover_assets(root):
    root = Path(root)
    index = json.loads((root / "index.json").read_text())
    if index.get("schema") != SCHEMA:
        raise ValueError("Unsupported surface-pools dataset schema")
    records = index["assets"]
    seen = set()
    families = {}
    sources = {}
    for r in records:
        uid = r["asset_id"]
        if uid in seen or Path(uid).name != uid or uid in (".", ".."):
            raise ValueError("Unsafe/duplicate asset id")
        if r["split"] not in ("train", "val", "test"):
            raise ValueError("Invalid split")
        seen.add(uid)
        for key, store in [("family_id", families), ("source_sha256", sources)]:
            if store.setdefault(r[key], r["split"]) != r["split"]:
                raise ValueError("Family/source split leakage")
        r["directory"] = str(root / "assets" / uid)
    return records


def load_volume(path, required):
    """Read precomputed SDF-only volume labels. No geometry oracle in workers."""
    if not required:
        return {}
    if not (path / "points.npy").is_file() or not (path / "sdf.npy").is_file():
        raise ValueError(
            "Volume supervision requested but volume pool is missing; rebuild data with volume_pool_count > 0"
        )
    bank = {
        name: np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("points", "sdf")
    }
    n = len(bank["points"])
    if n < required or bank["points"].shape != (n, 3) or bank["sdf"].shape != (n, 1):
        raise ValueError("Invalid/insufficient volume pool")
    if any(a.dtype != np.float32 for a in bank.values()):
        raise ValueError("Volume points/labels must be float32")
    return bank
