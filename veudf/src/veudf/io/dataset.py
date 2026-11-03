"""Read and validate surface-only dataset indexes."""

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


def load_surface_pool(asset, name):
    """Memory-map a new-schema supervision pool; CAD provenance is GT, not condition input."""
    from veudf.schema import POOLS, CHANNELS

    asset = Path(asset)
    meta = json.loads((asset / "meta.json").read_text())
    if meta.get("schema") != SCHEMA or meta.get("channel_names") != list(CHANNELS):
        raise ValueError("Unsupported surface UDF schema/channels")
    if name not in POOLS:
        raise ValueError("Unknown surface pool")
    bank = {
        key: np.load(
            asset / "pools" / name / f"{key}.npy", mmap_mode="r", allow_pickle=False
        )
        for key in (
            "points",
            "normals",
            "triangle_ids",
            "barycentric",
            "fields",
            "nearest_ids",
            "flags",
            "source_instance",
            "strata",
        )
    }
    n = len(bank["points"])
    layout = dict(
        points=((n, 3), np.float32),
        normals=((n, 3), np.float32),
        triangle_ids=((n,), np.int32),
        barycentric=((n, 2), np.float64),
        fields=((n, 2), np.float32),
        nearest_ids=((n, 2), np.int32),
        flags=((n, 2), np.uint8),
        source_instance=((n,), np.int32),
        strata=((n, 1), np.uint8),
    )
    for key, (shape, dtype) in layout.items():
        if bank[key].shape != shape or bank[key].dtype != dtype:
            raise ValueError("Invalid surface pool array: " + key)
    return bank
