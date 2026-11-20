"""Order spatial samples of geometric shell sections."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra, connected_components
from .schemas import require


def circle_order(points, data, ends, cfg):
    angle=np.mod(np.arctan2((points-data['center'])@data['y'],(points-data['center'])@data['x']),2*np.pi)
    if not len(ends):
        order=np.argsort(angle);q=points[order]
        # Do not silently close a sparse open arc whose endpoints were missed.
        gaps=np.diff(np.r_[np.sort(angle),np.sort(angle)[0]+2*np.pi])
        require(gaps.max()<max(.35,8*np.median(gaps[gaps>1e-7])),'curve_closure_unobserved','Circular support has a large uncovered arc')
        return np.vstack((q,q[0]))
    a=np.mod(np.arctan2((ends[0]-data['center'])@data['y'],(ends[0]-data['center'])@data['x']),2*np.pi)
    if len(ends)==1:
        order=np.argsort(np.mod(angle-a,2*np.pi));return np.vstack((ends[0],points[order],ends[0]))
    b=np.mod(np.arctan2((ends[1]-data['center'])@data['y'],(ends[1]-data['center'])@data['x']),2*np.pi)
    sweep=(b-a)%(2*np.pi);rel=(angle-a)%(2*np.pi)
    if np.mean(rel<=sweep+.01)>=.95:order=np.argsort(rel)
    elif np.mean((a-angle)%(2*np.pi)<=2*np.pi-sweep+.01)>=.95:order=np.argsort((a-angle)%(2*np.pi))
    else:return None
    return np.vstack((ends[0],points[order],ends[1]))


def order_general(points, ends, cfg, *, warnings=None):
    """Local projected-support graph, not global face competition.

    Open paths use an MST backbone. A closed path is cut locally then reclosed;
    coverage checks report branch loss on the resulting path.
    """
    cell=max(cfg.curve_tolerance*.15,1e-7)
    _,ids=np.unique(np.round(points/cell).astype(np.int64),axis=0,return_index=True)
    p=points[np.sort(ids)]
    require(len(p)>=4,'insufficient_edge_support','Too few distinct curve samples')
    k=min(8,len(p));distance,near=cKDTree(p).query(p,k=k)
    rows=np.repeat(np.arange(len(p)),k-1);cols=near[:,1:].ravel();weights=distance[:,1:].ravel()
    # Collapse reciprocal edges with a min rather than sparse duplicate summation.
    pairs=np.sort(np.column_stack((rows,cols)),axis=1)
    pairs,first=np.unique(pairs,axis=0,return_index=True);weights=weights[first]
    graph=coo_matrix((np.r_[weights,weights],(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(len(p),len(p))).tocsr()
    count,_=connected_components(graph,directed=False)
    require(count==1,'curve_support_disconnected','Projected support is not one connected curve')
    tree=minimum_spanning_tree(graph);tree=tree+tree.T
    if len(ends)==2:
        start,end=cKDTree(p).query(ends)[1]
    else:
        start=int(np.argmax(dijkstra(tree,directed=False,indices=0)))
        end=int(np.argmax(dijkstra(tree,directed=False,indices=start)))
    _,previous=dijkstra(tree,directed=False,indices=start,return_predecessors=True)
    path=[int(end)]
    while path[-1]!=start:
        parent=int(previous[path[-1]])
        require(parent>=0,'curve_path_missing','Curve backbone is disconnected')
        path.append(parent)
    q=p[path[::-1]]
    if len(ends)==2:q=np.vstack((ends[0],q,ends[1]))
    elif len(ends)==1:
        # The terminal vertex joins the two ends; curve validation checks the gap.
        q=np.vstack((ends[0],q,ends[0]))
    else:
        step=np.median(np.linalg.norm(np.diff(q,axis=0),axis=1))
        require(np.linalg.norm(q[-1]-q[0])<=max(8*step,cfg.curve_tolerance*4),
                'curve_closure_unobserved','No vertex association and support does not close')
        q=np.vstack((q,q[0]))
    # Point-to-segment distance checks coverage without a sparse sampling penalty.
    delta=q[1:]-q[:-1];worst=0.
    tree=cKDTree((q[1:]+q[:-1])*.5)
    for chunk in np.array_split(p,max(1,int(np.ceil(len(p)/256)))):
        near=tree.query(chunk,k=min(8,len(delta)))[1]
        if near.ndim==1:near=near[:,None]
        rel=chunk[:,None]-q[near];directions=delta[near]
        t=np.clip(np.einsum('nsi,nsi->ns',rel,directions)/np.maximum((directions*directions).sum(2),1e-20),0,1)
        dist=np.linalg.norm(rel-t[:,:,None]*directions,axis=2).min(1)
        # Candidate-segment distance is an upper bound: it may reject, never
        # certify missing support through an optimistic nearest-point estimate.
        worst=max(worst,float(dist.max()))
    limit = cfg.curve_tolerance * 3
    if worst > limit:
        require(warnings is not None, 'curve_support_branching', 'Ordered curve misses its support', error=worst)
        warnings.append(dict(reason='curve_support_branching', details=dict(error=worst, limit=limit)))
    return q
