from dataclasses import dataclass, field, asdict
import json
import os
from pathlib import Path
import numpy as np

class ReconstructionError(RuntimeError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.details = code, details or {}

def require(condition, code, message, **details):
    if not condition: raise ReconstructionError(code, message, details)

def jsonable(obj):
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,np.generic): return obj.item()
    if hasattr(obj,'__dataclass_fields__'): return jsonable(asdict(obj))
    if isinstance(obj,dict): return {str(k):jsonable(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)): return [jsonable(v) for v in obj]
    return obj

def write_json(path, value):
    path=Path(path)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

@dataclass
class SurfaceMesh:
    vertices: np.ndarray
    triangles: np.ndarray
    report: dict = field(default_factory=dict)

@dataclass
class VertexNode:
    id: int
    position: np.ndarray
    residual: float
    support_radius: float

@dataclass
class EdgeInstance:
    id: int
    faces: tuple
    samples: np.ndarray
    closed: bool
    vertices: list
    left_face: int
    support_radius: float

@dataclass
class CoedgeUse:
    face: int
    edge: int
    forward: bool

@dataclass
class LogicalGraph:
    face_count: int
    vertices: list
    edges: list
    cycles: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)
