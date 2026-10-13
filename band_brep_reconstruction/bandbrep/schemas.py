from dataclasses import dataclass, field, asdict
import json
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
    Path(path).write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

@dataclass
class SurfaceMesh:
    vertices: np.ndarray
    triangles: np.ndarray
    report: dict = field(default_factory=dict)

@dataclass
class BandState:
    values: np.ndarray
    edge_band: np.ndarray
    vertex_band: np.ndarray
    edge_region: np.ndarray
    core_mask: np.ndarray
    triangle_core: np.ndarray
    report: dict = field(default_factory=dict)
    parent_triangles: np.ndarray = None

@dataclass
class ArrivalState:
    costs: np.ndarray
    vertex_labels: np.ndarray
    triangle_labels: np.ndarray
    source_nodes: dict

@dataclass
class InterfaceGraph:
    points: np.ndarray
    segments: list

@dataclass
class VertexNode:
    id: int
    position: np.ndarray
    residual: float
    support_radius: float
    ports: list = field(default_factory=list)

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
