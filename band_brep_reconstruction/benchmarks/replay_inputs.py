"""Shared inputs and artifact helpers for frozen Post1109 replays."""
import hashlib
import json
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
MODULE=REPO/'band_brep_reconstruction'
sys.path[:0]=[str(MODULE),str(REPO/'veudf/src'),str(Path(__file__).parent)]


def read(path):return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    from bandbrep.schemas import jsonable
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(jsonable(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)




def offline_source(targets,raw,oracle,normalization):
    from types import SimpleNamespace
    from scipy.spatial import cKDTree
    from offline_support import PolylineDistance
    edge=PolylineDistance(targets.edges) if targets.edges else None
    return SimpleNamespace(targets=targets,mesh=raw,normalization=normalization,edge=edge,
        vtree=cKDTree(targets.vertices) if len(targets.vertices) else None,
        truth_face=lambda q:raw[2][oracle.closest(q)[1]],sdf=oracle.query)
