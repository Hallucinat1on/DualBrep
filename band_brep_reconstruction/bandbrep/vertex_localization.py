import numpy as np
from scipy.optimize import least_squares
from .schemas import require

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
