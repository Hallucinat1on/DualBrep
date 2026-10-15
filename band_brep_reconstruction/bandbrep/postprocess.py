"""Geometric boundary orientation and wire construction, AFTER curve fitting."""
import numpy as np
from scipy.spatial import cKDTree
from .surface_mesh import normals
from .schemas import require
from .boundary_cycles import complete_face_regions_and_cycles


def orient_boundaries(mesh, topology, graph, curves):
    centers=mesh.vertices[mesh.triangles].mean(1);nrm=normals(mesh)
    regions={}
    for face in range(topology.face_count):
        ids=np.flatnonzero(topology.face_labels==face)
        regions[face]=(ids,cKDTree(centers[ids]))
    for edge in graph.edges:
        q=curves[edge.id].evaluate(np.linspace(.05,.95,64))
        tangent=np.gradient(q,axis=0)
        signs=[]
        for face in edge.faces:
            ids,tree=regions[face];closest=ids[tree.query(q)[1]]
            inward=centers[closest]-q
            score=np.einsum('ij,ij->i',np.cross(nrm[closest],tangent),inward)
            signs.append(float(np.median(score)))
        require(signs[0]*signs[1]<0,'boundary_orientation_ambiguous',
                'Face interior supports do not determine opposite edge uses',edge=edge.id,scores=signs)
        edge.left_face=edge.faces[0 if signs[0]>0 else 1]
    return complete_face_regions_and_cycles(graph)
