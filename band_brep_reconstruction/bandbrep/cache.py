"""Versioned, hashed, JSON/NPZ-only stage caches (no executable pickle)."""
import hashlib
import json
from pathlib import Path
from .veudf_contract import require_veudf_v3

def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def code_hash():
    root=Path(__file__).parent;h=hashlib.sha256()
    for path in sorted(root.rglob('*.py')):h.update(str(path.relative_to(root)).encode());h.update(path.read_bytes())
    return h.hexdigest()

def cache_key(mesh,udf,cfg):
    dataset_schema = require_veudf_v3()
    from veudf.geometry.sampling import mesh_hash
    from veudf.schema import GEOMETRY_VERSION
    import veudf.geometry.mesh as mesh_module
    import veudf.geometry.sampling as sampling_module
    config=cfg.to_dict();config.pop('stop_after',None)
    d={'mesh':mesh_hash(mesh.vertices,mesh.triangles),'normalization':mesh.report.get('normalization',{}),
       'udf':udf.fingerprint,'band_width':udf.band_width,'presence':udf.presence(),
       'config':config,'code_hash':code_hash(),'geometry_version':GEOMETRY_VERSION,'veudf_schema':dataset_schema,
       'mesh_oracle_code':file_hash(mesh_module.__file__),'surface_sampling_code':file_hash(sampling_module.__file__)}
    return hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest(),d
