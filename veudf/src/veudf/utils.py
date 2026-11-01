"""Deterministic hashes, seeds and sampling budgets; no CAD or model imports."""

import hashlib
import json
from pathlib import Path
import numpy as np


def seed_for(*parts):
    return int(
        hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:15], 16
    )


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(
        str((a.shape, str(a.dtype))).encode() + a.tobytes()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def allocate(total, weights):
    w = np.asarray(weights, np.float64)
    if total < 0 or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Invalid sampling budget/weights")
    q = total * w / w.sum()
    n = np.floor(q).astype(np.int64)
    n[np.argsort(-(q - n), kind="stable")[: total - int(n.sum())]] += 1
    return n


def draw(rng, rows, count):
    rows = np.asarray(rows, np.int64)
    if not count:
        return rows[:0]
    if not len(rows):
        raise ValueError("Cannot sample an empty pool")
    return rng.choice(rows, count, replace=len(rows) < count)
