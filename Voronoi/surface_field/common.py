"""Shared constants, error type, and atomic IO helpers for the surface-field pipeline.

This module must only depend on the standard library so that ``--help`` works
without numpy/pythonocc/trimesh installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "abc-surface-v2"  # v2: surface_field.npz adds vertex-level udf_* arrays
DEFAULT_ABC_ROOT = Path("/opt/data/private/yihengxu/Datasets/abc")
DEFAULT_OUTPUT_ROOT = Path("/opt/data/private/yihengxu/Datasets/surface")
DEFAULT_VORONOI_EXE = Path("Voronoi/build/calculate_voronoi/calculate_voronoi")
ASSET_RE = re.compile(
    r"^(?P<shape_id>\d{8})_(?P<token>[^_]+)_"
    r"(?P<kind>step|trimesh)_(?P<part>\d+)\.(?P<ext>step|stp|obj)$",
    re.IGNORECASE,
)


class PipelineError(RuntimeError):
    """An expected, classifiable per-shape failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Asset:
    shape_id: str
    token: str
    part: str
    kind: str
    path: Path

    @property
    def pair_key(self) -> tuple[str, str, str]:
        return self.shape_id, self.token, self.part


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True,
                                       ensure_ascii=False, default=_json_default) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False,
                              default=_json_default) + "\n" for row in rows)
    atomic_write_text(path, text)


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()
