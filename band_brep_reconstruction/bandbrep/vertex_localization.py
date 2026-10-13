import numpy as np
from scipy.optimize import least_squares
from scipy.cluster.hierarchy import linkage,fcluster
from .schemas import VertexNode,require
from .surface_mesh import mesh_edges,components,evaluate_batched

def distance_center(points,distances,initial=None,weights=None):
    points=np.asarray(points,float);d=np.asarray(distances,float)
    A=2*(points[1:]-points[0]);rhs=np.sum(points[1:]**2,axis=1)-np.sum(points[0]**2)-(d[1:]**2-d[0]**2)
    center,_,rank,_=np.linalg.lstsq(A,rhs,rcond=None)
    if rank<3:
        require(initial is not None,'vertex_rank_deficient','Distance-center observations do not span 3D')
        center=initial
    w=np.ones(len(d)) if weights is None else np.sqrt(weights)
    result=least_squares(lambda v:w*(np.linalg.norm(points-v,axis=1)-d),center,loss='soft_l1',f_scale=max(1e-6,np.median(d)*.02))
    return result.x,float(np.max(np.abs(np.linalg.norm(points-result.x,axis=1)-d))),rank

def localize_vertices(provider,mesh,bands,cfg):
    if provider.presence()[1]=='absent':return []
    ids=np.flatnonzero(bands.vertex_band)
    require(len(ids)>0,'vertex_unobserved','Vertex channel present but no vertex region found; refine sampling')
    # Gradient votes can split a single connected band containing several vertices.
    pts=mesh.vertices[ids];g,valid=provider.gradient(pts,2)
    d=bands.values[ids,2];norm=np.linalg.norm(g,axis=1)
    votes=pts[valid]-d[valid,None]*g[valid]/np.maximum(norm[valid,None],1e-12)
    require(len(votes)>0,'vertex_gradient_invalid','No usable vertex votes')
    # Small vote-scale clustering, never a tau_V-sized spatial merge.
    if len(votes)>5000:
        selected=np.linspace(0,len(votes)-1,5000).astype(int);votes=votes[selected]
    groups=fcluster(linkage(votes,method='complete'),t=max(cfg.localization_tolerance*3,cfg.field_error*3),criterion='distance') if len(votes)>1 else np.ones(1,int)
    candidates=[];rng=np.random.default_rng(cfg.seed)
    for group in np.unique(groups):
        v=np.median(votes[groups==group],axis=0)
        # Verify on independent 3D shell queries, then shrink to isolate nearest V.
        radius=max(cfg.localization_tolerance*8,min(cfg.tau_vertex*.2,.01))
        for _ in range(4):
            direction=rng.normal(size=(120,3));direction/=np.linalg.norm(direction,axis=1)[:,None]
            shell=v+direction*radius*rng.uniform(.5,1,(len(direction),1))
            ds=provider.evaluate(shell).values[:,2]
            center,residual,rank=distance_center(shell,ds,v)
            if residual<=cfg.localization_tolerance and np.linalg.norm(center-v)<radius:break
            radius*=.5;v=center
        require(residual<=cfg.localization_tolerance,'vertex_mixed_support','Cannot fit one vertex to local v-UDF observations',residual=residual)
        if any(np.linalg.norm(center-other.position)<cfg.localization_tolerance*2 for other in candidates):continue
        values=provider.evaluate(center[None]).values[0]
        require(values[2]<=cfg.localization_tolerance and values[1]<=cfg.localization_tolerance and abs(values[0])<=cfg.surface_tolerance,
                'vertex_geometry_invalid','Localized vertex fails field checks',values=values.tolist())
        candidates.append(VertexNode(len(candidates),center,residual,radius))
    return candidates
