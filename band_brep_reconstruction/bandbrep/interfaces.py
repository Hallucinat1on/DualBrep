import itertools
import numpy as np
from .schemas import InterfaceGraph,require
from .surface_mesh import normals


def lower_envelope_segments(costs,tolerance=1e-10):
    """Equal-cost line clipped by ALL other sources in barycentric coordinates."""
    segments=[]
    for a,b in itertools.combinations(range(len(costs)),2):
        d=costs[a]-costs[b]
        if np.max(np.abs(d))<tolerance:continue
        roots=[]
        for u,v in ((0,1),(1,2),(2,0)):
            if abs(d[u])<=tolerance:roots.append(np.eye(3)[u])
            if d[u]*d[v]<0:
                t=d[u]/(d[u]-d[v]);roots.append((1-t)*np.eye(3)[u]+t*np.eye(3)[v])
        unique=[]
        for p in roots:
            if not any(np.linalg.norm(p-q)<1e-9 for q in unique):unique.append(p)
        if len(unique)<2:continue
        p,q=unique[0],unique[1];low,high=0.,1.
        for c in range(len(costs)):
            if c in (a,b):continue
            h=costs[a]-costs[c];h0,h1=h@p,h@q
            if h0>tolerance and h1>tolerance:low,high=1.,0.;break
            if h0>tolerance:low=max(low,h0/(h0-h1))
            if h1>tolerance:high=min(high,h0/(h0-h1))
        if high-low>1e-9:segments.append((a,b,p+low*(q-p),p+high*(q-p)))
    return segments

def extract_interfaces(mesh,bands,arrivals,provider,cfg):
    points=[];segments=[];cache={};nrm=normals(mesh)
    def node(bary,tri,pair,tid):
        zero=np.flatnonzero(np.abs(bary)<1e-8)
        if len(zero)>=2:key=('v',int(tri[np.argmax(bary)]),*pair)
        elif len(zero)==1:
            edge=tuple(sorted(tri[np.flatnonzero(bary>1e-8)].tolist()));key=('e',*edge,*pair)
        else:key=('interior',tid,tuple(np.round(bary,9)),*pair)
        if key not in cache:cache[key]=len(points);points.append(bary@mesh.vertices[tri])
        return cache[key]
    for tid,tri in enumerate(mesh.triangles):
        if bands.vertex_band[tri].any():continue
        c=arrivals.costs[:,tri].copy()
        finite=np.isfinite(c).all(axis=1)
        if finite.sum()<2:continue
        labels=np.flatnonzero(finite);c=c[finite]
        for a,b,p,q in lower_envelope_segments(c):
            pair=(int(labels[a]),int(labels[b]));u=node(p,tri,pair,tid);v=node(q,tri,pair,tid)
            if u==v:continue
            direction=np.asarray(points[v])-points[u]
            # Side of a's lower-cost region from affine cost gradient.
            d=c[a]-c[b];J=mesh.vertices[tri[1:]]-mesh.vertices[tri[0]]
            gradient=np.linalg.lstsq(J,d[1:]-d[0],rcond=None)[0]
            left=pair[0] if np.dot(np.cross(nrm[tid],direction),gradient)<0 else pair[1]
            segments.append({'nodes':(u,v),'faces':pair,'left':left,'triangle':tid})
    if provider.presence()[0]=='present':require(len(segments)>0,'no_interfaces','No two-source interface; bands may not isolate faces')
    return InterfaceGraph(np.asarray(points,float).reshape(-1,3),segments)
