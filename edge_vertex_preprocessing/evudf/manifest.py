"""ABC/raw STEP indexing. No OCC or project dependency."""
import hashlib
import json
from pathlib import Path
import re


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_step(path):
    return path.name.lower().endswith((".step", ".stp", ".step.gz", ".stp.gz"))


def safe_id(relative):
    label = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(relative))[:120].strip(".") or "shape"
    return label + "-" + hashlib.sha256(str(relative).encode()).hexdigest()[:12]


def scan_directory(root, *, seed=0, train_fraction=0.8, val_fraction=0.1):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    if not (0 <= train_fraction <= 1 and 0 <= val_fraction <= 1 and train_fraction + val_fraction <= 1):
        raise ValueError("Split fractions must be nonnegative and sum to <= 1")
    records, seen_hashes, parents = [], {}, {}

    def find(group):
        parents.setdefault(group, group)
        if parents[group] != group:
            parents[group] = find(parents[group])
        return parents[group]

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)
    for path in sorted(p for p in root.rglob("*") if p.is_file() and is_step(p)):
        relative = path.relative_to(root)
        checksum = file_hash(path)
        # ABC filename: numeric model prefix + original Onshape document id.
        match = re.match(r"^\d{8}_([0-9a-fA-F]{24})_", path.name)
        group = match.group(1).lower() if match else "source:" + checksum
        find(group)
        duplicate = seen_hashes.get(checksum)
        if duplicate is not None:
            # Merge WHOLE source groups, including their other parts.
            union(group, duplicate["source_group_id"])
        record = dict(sample_id=safe_id(relative), step_path=str(path),
                      archive_member=str(relative), source_sha256=checksum,
                      source_group_id=group,
                      group_source="ABC filename document id" if match else "source content hash")
        if duplicate is not None:
            record["duplicate_of"] = duplicate["sample_id"]
        seen_hashes.setdefault(checksum, record)
        records.append(record)
    for record in records:
        group = find(record["source_group_id"])
        score = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16) / 2**64
        record["group_id"] = group
        record["split"] = "train" if score < train_fraction else "val" if score < train_fraction + val_fraction else "test"
    return records


def load_manifest(path):
    path = Path(path).resolve()
    records, seen_ids = [], set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row.get("sample_id")
        if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", sid) or sid in (".", ".."):
            raise ValueError(f"Unsafe/missing sample_id at line {line_number}")
        if sid in seen_ids:
            raise ValueError(f"Duplicate sample_id: {sid}")
        seen_ids.add(sid)
        source = Path(row["step_path"])
        if not source.is_absolute():
            source = path.parent / source
        row["step_path"] = str(source.resolve())
        records.append(row)
    return records
