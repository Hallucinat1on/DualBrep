import numpy as np
from scipy.optimize import least_squares
from .schemas import VertexNode,require

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

def localize_vertices(provider, bands, topology, cfg):
    """One distance-center fit per instantiated absorbing vertex domain."""
    candidates=[];rng=np.random.default_rng(cfg.seed)
    for domain in topology.vertices:
        ids=np.asarray(domain['nodes'],int);pts=bands.points[ids];d=bands.values[ids,2]
        g,valid=provider.gradient(pts,2);norm=np.linalg.norm(g,axis=1)
        valid &= norm>1e-10
        votes=pts[valid]-d[valid,None]*g[valid]/norm[valid,None]
        require(len(votes)>0,'vertex_gradient_invalid','No usable vertex votes',vertex=domain['id'])
        v=np.median(votes,axis=0)
        radius=max(cfg.localization_tolerance*8,min(cfg.tau_vertex*.2,.01))
        for _ in range(4):
            direction=rng.normal(size=(120,3));direction/=np.linalg.norm(direction,axis=1)[:,None]
            shell=v+direction*radius*rng.uniform(.5,1,(len(direction),1))
            ds=provider.evaluate(shell).values[:,2]
            center,residual,rank=distance_center(shell,ds,v)
            if residual<=cfg.localization_tolerance and np.linalg.norm(center-v)<radius:break
            radius*=.5;v=center
        require(residual<=cfg.localization_tolerance,'vertex_mixed_support',
                'Vertex domain does not support a single distance center',vertex=domain['id'],residual=residual)
        values=provider.evaluate(center[None]).values[0]
        require(values[2]<=cfg.localization_tolerance and values[1]<=cfg.localization_tolerance and abs(values[0])<=cfg.surface_tolerance,
                'vertex_geometry_invalid','Localized vertex fails field checks',vertex=domain['id'],values=values.tolist())
        candidates.append(VertexNode(domain['id'],center,residual,radius))
    return candidates
