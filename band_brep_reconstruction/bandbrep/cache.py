"""Versioned, hashed, JSON/NPZ-only stage caches (no executable pickle)."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .schemas import VertexNode,EdgeInstance,LogicalGraph,CoedgeUse,BandTopology
from .curve_fitting import CurveRecord
from .surface_fitting import FacePatch

def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def code_hash():
    root=Path(__file__).parent;h=hashlib.sha256()
    for path in sorted(root.rglob('*.py')):h.update(str(path.relative_to(root)).encode());h.update(path.read_bytes())
    return h.hexdigest()

def cache_key(mesh,udf,cfg):
    from veudf.geometry.sampling import mesh_hash
    from veudf.schema import GEOMETRY_VERSION
    import veudf.geometry.mesh as mesh_module
    import veudf.geometry.sampling as sampling_module
    config=cfg.to_dict();config.pop('stop_after',None)
    d={'mesh':mesh_hash(mesh.vertices,mesh.triangles),'normalization':mesh.report.get('normalization',{}),
       'udf':udf.fingerprint,'band_width':udf.band_width,'presence':udf.presence(),
       'config':config,'code_hash':code_hash(),'geometry_version':GEOMETRY_VERSION,
       'mesh_oracle_code':file_hash(mesh_module.__file__),'surface_sampling_code':file_hash(sampling_module.__file__)}
    return hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest(),d

def load_graph(path):
    d=json.loads(Path(path).read_text());vertices=[VertexNode(**{**v,'position':np.array(v['position'])}) for v in d['vertices']]
    edges=[EdgeInstance(**{**e,'samples':np.array(e['samples'])}) for e in d['edges']]
    cycles={int(k):[[CoedgeUse(**u) for u in c] for c in v] for k,v in d['cycles'].items()}
    return LogicalGraph(d['face_count'],vertices,edges,cycles,d.get('report',{}))

def load_records(path,kind):
    d=json.loads(Path(path).read_text());out={}
    for i,v in d.items():
        data={key:np.array(value) if isinstance(value,list) else value for key,value in v['data'].items()}
        if kind=='curve':out[int(i)]=CurveRecord(**{**v,'data':data})
        else:out[int(i)]=FacePatch(**{**v,'data':data,'normal':np.array(v['normal'])})
    return out


def load_topology(directory):
    directory=Path(directory)
    d=json.loads((directory/'incidence.json').read_text())
    a=np.load(directory/'topology_labels.npz')
    return BandTopology(d['face_count'],a['face_labels'],a['edge_labels'],a['vertex_labels'],
                        d['edges'],d['vertices'],a['contacts'],d['report'])


def load_center_fields(path):
    from .field_provider import CenterFields
    with np.load(path, allow_pickle=False) as a:
        return CenterFields(a['points'], a['values'], str(a['provider_fingerprint'].item()))
