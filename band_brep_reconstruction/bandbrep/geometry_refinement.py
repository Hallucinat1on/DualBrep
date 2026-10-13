import numpy as np
from .schemas import require

def refine_edge_geometry(edge,provider,vertices,cfg):
    original=edge.samples.copy();points=original.copy()
    fixed=np.zeros(len(points),bool)
    if edge.vertices:fixed[[0,-1]]=True
    for _ in range(20):
        values=provider.evaluate(points).values
        d=values[:,1];g,valid=provider.gradient(points,1)
        norm=np.linalg.norm(g,axis=1)
        active=valid&~fixed&(d>cfg.curve_tolerance*.03)
        if not active.any():break
        step=.8*np.minimum(d,edge.support_radius*.3)
        delta=step[:,None]*g/np.maximum(norm[:,None],1e-12)
        proposed=points-delta
        # Trust region keeps projection attached to its original topological branch.
        safe=np.linalg.norm(proposed-original,axis=1)<=edge.support_radius
        points[active&safe]=proposed[active&safe]
    if edge.vertices:points[0]=vertices[edge.vertices[0]].position;points[-1]=vertices[edge.vertices[1]].position
    if edge.closed:points[-1]=points[0]
    v=provider.evaluate(points).values
    require(v[:,1].max()<=cfg.curve_tolerance,'edge_localization_failed','Interface does not project to its supported edge valley',edge=edge.id,max_udf=float(v[:,1].max()))
    require(np.abs(v[:,0]).max()<=cfg.surface_tolerance,'edge_surface_failed','Localized edge leaves SDF surface',edge=edge.id)
    keep=np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-9]
    edge.samples=points[keep]
    return edge.samples

def validate_structure_coverage(mesh,bands,curves,provider,cfg):
    """Detect unrepresented edge valleys, independently of recovered face labels."""
    from scipy.spatial import cKDTree
    if provider.presence()[0]=='absent':return {'tested':0,'edge_field_absent':True}
    ids=np.flatnonzero(bands.values[:,1]<cfg.tau_edge*.6)
    require(len(ids)>0,'edge_coverage_unobserved','No near-edge validation positions')
    if len(ids)>2048:ids=ids[np.linspace(0,len(ids)-1,2048).astype(int)]
    points=mesh.vertices[ids].copy()
    for _ in range(8):
        v=provider.evaluate(points).values;g,valid=provider.gradient(points,1)
        norm=np.linalg.norm(g,axis=1);active=valid&(v[:,1]>cfg.curve_tolerance*.05)
        points[active]-=.8*v[active,1,None]*g[active]/np.maximum(norm[active,None],1e-12)
    v=provider.evaluate(points).values;valid=(v[:,1]<=cfg.curve_tolerance)&(np.abs(v[:,0])<=cfg.surface_tolerance)
    require(valid.any(),'edge_coverage_unlocalized','Cannot localize independent edge-field probes')
    samples=np.vstack([c.evaluate(np.linspace(0,1,4097)) for c in curves.values()]) if curves else np.empty((0,3))
    require(len(samples)>0,'edge_coverage_missing','Edge field present but no curves recovered')
    distance=cKDTree(samples).query(points[valid])[0]
    require(distance.max()<=cfg.curve_tolerance*3,'edge_valley_unrepresented','Detected edge UDF zero branch not covered by any recovered curve; reduce bands / refine',error=float(distance.max()))
    return {'tested':len(points),'localized':int(valid.sum()),'max_curve_distance':float(distance.max()),'continuous_coverage_certified':False}
