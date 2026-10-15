import numpy as np
from .schemas import require

def validate_structure_coverage(mesh,bands,curves,provider,cfg):
    """Detect unrepresented edge valleys, independently of recovered face labels."""
    from scipy.spatial import cKDTree
    if provider.presence()[0]=='absent':return {'tested':0,'edge_field_absent':True}
    ids=np.flatnonzero(bands.values[:,1]<cfg.tau_edge*.6)
    require(len(ids)>0,'edge_coverage_unobserved','No near-edge validation positions')
    if len(ids)>2048:ids=ids[np.linspace(0,len(ids)-1,2048).astype(int)]
    points=bands.points[ids].copy()
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
