"""UDF localization inside instantiated supports, followed by geometric ordering.

No arrival-cost interface or vertex port is used to create edge identities.
Closed/self-loop handling here belongs to geometric postprocessing, not incidence.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra, connected_components
from .schemas import EdgeInstance, LogicalGraph, require
from .curve_fitting import fit_circle


def project_support(points, provider, radius, cfg):
    original=np.asarray(points,float);q=original.copy()
    for _ in range(32):
        d=provider.evaluate(q).values[:,1];g,valid=provider.gradient(q,1)
        norm=np.linalg.norm(g,axis=1);active=valid&(norm>1e-10)&(d>cfg.curve_tolerance*.02)
        if not active.any():break
        proposed=q-.8*np.minimum(d,radius*.3)[:,None]*g/np.maximum(norm[:,None],1e-12)
        safe=np.linalg.norm(proposed-original,axis=1)<=radius
        q[active&safe]=proposed[active&safe]
    v=provider.evaluate(q).values
    valid=(v[:,1]<=cfg.curve_tolerance)&(np.abs(v[:,0])<=cfg.surface_tolerance)
    require(valid.sum()>=3 and valid.mean()>=.8,'edge_localization_failed',
            'Band samples do not consistently localize to an edge',localized=int(valid.sum()),total=len(q))
    return q[valid]


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


def order_general(points, ends, cfg):
    """Local projected-support graph, not global face competition.

    Open paths use an MST backbone. A closed path is cut locally then reclosed;
    coverage checks reject branch loss rather than inventing a connecting curve.
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
    require(worst<=cfg.curve_tolerance*3,'curve_support_branching','Ordered curve misses its support',error=worst)
    return q


def build_edge_geometry(topology, bands, vertices, provider, cfg):
    edges=[]
    for item in topology.edges:
        require(len(item['faces'])==2,'edge_neighbors_unresolved','Curve requires two confirmed neighboring faces',edge=item['id'],faces=item['faces'])
        vids=list(item['vertices']);require(len(vids)<=2,'edge_vertices_unresolved','Edge touches more than two vertex candidates',edge=item['id'],vertices=vids)
        radius=max(cfg.tau_edge*3,cfg.surface_tolerance*4)
        points=project_support(bands.points[item['nodes']],provider,radius,cfg)
        ends=np.array([vertices[v].position for v in vids]).reshape(-1,3)
        ordered=None
        if len(ends)==2:
            direction=ends[1]-ends[0];length=np.linalg.norm(direction)
            require(length>cfg.localization_tolerance,'coincident_vertex_candidates','Distinct vertex candidates have coincident geometry')
            t=(points-ends[0])@direction/(length*length)
            residual=np.linalg.norm(points-(ends[0]+t[:,None]*direction),axis=1)
            if residual.max()<=cfg.curve_tolerance:
                ordered=np.vstack((ends[0],points[np.argsort(t)],ends[1]))
        if ordered is None and len(points)>=6:
            data,error=fit_circle(points)
            if error<=cfg.curve_tolerance:ordered=circle_order(points,data,ends,cfg)
        if ordered is None:ordered=order_general(points,ends,cfg)
        keep=np.r_[True,np.linalg.norm(np.diff(ordered,axis=0),axis=1)>1e-9];ordered=ordered[keep]
        # A single distinct incident vertex is represented twice only now.
        endpoint_uses=vids if len(vids)!=1 else [vids[0],vids[0]]
        edges.append(EdgeInstance(item['id'],tuple(item['faces']),ordered,len(vids)==0,
                                  endpoint_uses,-1,radius))
    return LogicalGraph(topology.face_count,vertices,edges,report={'incidence_source':'band_components'})
