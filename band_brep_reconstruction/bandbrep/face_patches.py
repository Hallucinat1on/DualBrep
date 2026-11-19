"""Bounded patch fallback using interior mesh paths between logical vertices."""
from collections import defaultdict
from copy import deepcopy
from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .boundary_cycles import complete_face_regions_and_cycles
from .deadline import checkpoint
from .edge_first_graph import Path, fit_geometry
from .schemas import EdgeInstance, ReconstructionError, require
from .surface_fitting import initialize_face_surfaces
from .surface_mesh import components


def incidence(mesh):
    result = defaultdict(list)
    for fid, tri in enumerate(mesh.triangles):
        for u, v in zip(tri, np.roll(tri, -1)):
            result[tuple(sorted((u, v)))].append((fid, u, v))
    return result


def face_sides(uses, labels, a, b):
    left, right = uses[tuple(sorted((a, b)))]
    if left[1:] != (a, b):
        left, right = right, left
    return int(labels[left[0]]), int(labels[right[0]])


def interior_paths(mesh, labels, graph, face, uses, cfg):
    nodes = {i for e in graph.edges if face in e.faces for i in e.vertices}
    if len(nodes) < 2:
        return
    nodes = sorted(nodes)
    boundary, interior, adjacency = set(), [], []
    for (u, v), (a, b) in uses.items():
        membership = (labels[a[0]] == face, labels[b[0]] == face)
        if all(membership):
            interior.append((u, v))
            adjacency.append((a[0], b[0]))
        elif any(membership):
            boundary.update((u, v))
    if not boundary or not interior:
        return
    boundary_nodes = sorted(boundary)
    gap, near = cKDTree(mesh.vertices[boundary_nodes]).query([graph.vertices[i].position for i in nodes])
    anchors = {i: boundary_nodes[j] for i,j,d in zip(nodes,near,gap) if d <= cfg.feature_sdf_limit+1e-8}
    triangles = np.flatnonzero(labels==face)
    adjacency = np.asarray(adjacency)
    edges = np.asarray(interior)
    xyz = mesh.vertices
    plane = np.linalg.svd(xyz[mesh.triangles[triangles]].reshape(-1,3)-
                          xyz[mesh.triangles[triangles]].mean(axis=(0,1)), full_matrices=False)[2][-1]
    # Try separated anchor pairs first, avoiding many expensive projections.
    pairs = sorted(combinations(anchors,2),key=lambda e: -np.linalg.norm(
        xyz[anchors[e[0]]]-xyz[anchors[e[1]]]))[:24]
    for a,b in pairs:
        checkpoint('patch_split')
        u,v = anchors[a], anchors[b]
        if u==v:
            continue
        allowed = np.array([int(p) not in boundary or p in (u,v) for p in edges.ravel()]).reshape(-1,2).all(1)
        choices = edges[allowed]
        if not len(choices):
            continue
        diff=xyz[choices[:,0]]-xyz[choices[:,1]]
        lengths=np.linalg.norm(diff,axis=1)
        # Prefer shorter internal paths, with a mild cost for steep chart change.
        weight=lengths + .3*abs(diff@plane)
        local=coo_matrix((weight,(choices[:,0],choices[:,1])),shape=(len(xyz),len(xyz))).tocsr()
        d,prev=dijkstra(local,directed=False,indices=u,return_predecessors=True)
        if not np.isfinite(d[v]):
            continue
        path=[v]
        while path[-1]!=u:
            path.append(int(prev[path[-1]]))
        path=path[::-1]
        seam={tuple(sorted(edge)) for edge in zip(path,path[1:])}
        keep=np.array([edge not in seam for edge in interior])
        count, sub=components(np.searchsorted(triangles,adjacency[keep]),len(triangles))
        if count==2 and np.bincount(sub).min()>=2:
            yield a,b,path,triangles,sub


def patch_candidates(mesh, labels, constraints, graph, curves, face, cfg, uses):
    for a,b,path,triangles,split in interior_paths(mesh,labels,graph,face,uses,cfg):
        new_graph=deepcopy(graph)
        new_curves=curves.copy()
        new_labels=labels.copy()
        new_id=new_graph.face_count
        new_labels[triangles[split==1]]=new_id
        new_graph.face_count+=1
        eid=len(new_graph.edges)
        try:
            points=mesh.vertices[path]
            ends=np.array([graph.vertices[v].position for v in (a,b)])
            seam=fit_geometry(eid,Path([a,b],points,False),ends,cfg)
            sides={face_sides(uses,new_labels,u,v) for u,v in zip(path,path[1:])}
            require(len(sides)==1,'patch_seam_sides','Patch seam has inconsistent adjacent labels')
            pair=next(iter(sides))
            require(pair[0]!=pair[1],'patch_seam_open','Internal seam does not separate the face')
            mapped=defaultdict(set)
            for edge,u,v in constraints:
                mapped[edge].add(face_sides(uses,new_labels,u,v))
            for edge in new_graph.edges:
                pairs=mapped[edge.id]
                require(len(pairs)==1,'patch_crosses_external_edge','Patch splits an original shared edge')
                edge.faces=next(iter(pairs))
                edge.left_face=edge.faces[0]
            new_graph.edges.append(EdgeInstance(eid,pair,points,False,[a,b],pair[0],cfg.support_radius))
            new_curves[eid]=seam
            new_constraints=np.vstack((constraints,[(eid,u,v) for u,v in zip(path,path[1:])]))
            complete_face_regions_and_cycles(new_graph)
        except (ReconstructionError,np.linalg.LinAlgError):
            continue
        yield new_graph,new_curves,new_labels,new_constraints,dict(face=new_id,logical_face=face,edge=eid)


def fit_with_patch_fallback(mesh, labels, constraints, graph, curves, provider, cfg):
    """Allow a small number of internal seams; keep the original graph for scoring."""
    attempts=[]
    uses=None
    mapping=list(range(graph.face_count))
    for _ in range(cfg.patch_budget):
        try:
            return initialize_face_surfaces(mesh,labels,graph,curves,provider,cfg),graph,curves,mapping,attempts
        except ReconstructionError as exc:
            face=exc.details.get('face')
            if (exc.code not in ('surface_boundary_incompatible','surface_fit_failed','surface_fold')
                    or face is None or len(attempts)+1>=cfg.patch_budget):
                raise
            if uses is None:
                uses=incidence(mesh)
            best=None
            last=exc
            for candidate in patch_candidates(mesh,labels,constraints,graph,curves,face,cfg,uses):
                proposed,pc,pl,constraint,info=candidate
                try:
                    patches=initialize_face_surfaces(mesh,pl,proposed,pc,provider,cfg)
                except ReconstructionError as inner:
                    last=inner
                    # Advance only if both pieces of the old failed face now fit.
                    if inner.details.get('face') not in (face, info['face']) and best is None:
                        best=candidate
                    continue
                info['logical_face']=mapping[face]
                return patches,proposed,pc,mapping+[mapping[face]],attempts+[info]
            if best is None:
                raise exc from (None if exc is last else last)
            graph,curves,labels,constraints,info=best
            info['logical_face']=mapping[face]
            mapping.append(mapping[face])
            attempts.append(info)
    raise AssertionError('Patch iteration budget exhausted')
