"""Versioned, hashed, JSON/NPZ-only stage caches (no executable pickle)."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .schemas import VertexNode,EdgeInstance,LogicalGraph,CoedgeUse
from .curve_fitting import CurveRecord
from .surface_fitting import FacePatch

def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def code_hash():
    root=Path(__file__).parent;h=hashlib.sha256()
    for path in sorted(root.rglob('*.py')):h.update(str(path.relative_to(root)).encode());h.update(path.read_bytes())
    return h.hexdigest()

def cache_key(provider,cfg,face_model=None):
    config=cfg.to_dict();config.pop('stop_after',None)
    d={'provider':provider.fingerprint,'config':config,'code_hash':code_hash(),
       'face_model':getattr(face_model,'fingerprint',None)}
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
