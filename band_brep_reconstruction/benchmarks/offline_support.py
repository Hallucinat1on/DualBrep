"""Historical nearest-polyline scoring, isolated from reconstruction queries.

PolylineDistance copied unchanged from the frozen 2026-10-28 experiment_fields.py
class. Only offline instance matching/coverage uses it; input UDF is exact CAD.
"""
import numpy as np
from scipy.spatial import cKDTree
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve

class PolylineDistance:
    def __init__(self, edges, spacing=0.0002):
        points=[];owners=[];next_valid=[];self.pieces=[]
        for eid,edge in enumerate(edges):
            curve=BRepAdaptor_Curve(edge);n=256
            while True:
                u=np.linspace(curve.FirstParameter(),curve.LastParameter(),n)
                p=np.array([[q.X(),q.Y(),q.Z()] for q in (curve.Value(t) for t in u)])
                gap=float(np.linalg.norm(np.diff(p,axis=0),axis=1).max())
                if gap<=spacing:break
                if n>1000000:raise RuntimeError('Polyline sampling budget exceeded')
                n=int(n*min(16,max(2,gap/spacing*1.2)))
            self.pieces.append(p)
            valid=np.ones(n,bool);valid[-1]=False
            points.append(p);owners.append(np.full(n,eid));next_valid.append(valid)
        self.points=np.vstack(points);self.owners=np.concatenate(owners)
        self.next_valid=np.concatenate(next_valid);self.tree=cKDTree(self.points)

    def query(self, xyz):
        x=np.asarray(xyz,float).reshape(-1,3)
        distance,nearest=self.tree.query(x,workers=1)
        closest=self.points[nearest].copy();owner=self.owners[nearest].copy()
        for start in (nearest-1,nearest):
            valid=(start>=0)&(start<len(self.points)-1)
            valid[valid]&=self.next_valid[start[valid]]
            ids=np.flatnonzero(valid)
            a=self.points[start[ids]];delta=self.points[start[ids]+1]-a
            t=np.clip(np.einsum('ij,ij->i',x[ids]-a,delta)/np.maximum(np.einsum('ij,ij->i',delta,delta),1e-30),0,1)
            q=a+t[:,None]*delta;d=np.linalg.norm(x[ids]-q,axis=1)
            take=d<distance[ids];ids=ids[take]
            distance[ids]=d[take];closest[ids]=q[take];owner[ids]=self.owners[start[ids]]
        return distance,closest,owner
