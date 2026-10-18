"""Benchmark-only CAD fields, offline truth, and deterministic correlated perturbations."""
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from bandbrep.field_provider import FieldProvider
from bandbrep.vendor.oracle.config import Config as OracleConfig
from bandbrep.vendor.oracle.cad import read_normalized, collect_targets, triangulate
from bandbrep.vendor.oracle.distance import SDFOracle, EdgeOracle
import igl


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


class BenchmarkCAD(FieldProvider):
    def __init__(self, step):
        self.config=OracleConfig(sdf_backend='mesh_fast')
        solid,self.normalization=read_normalized(step,self.config)
        self.targets=collect_targets(solid,self.config)
        self.mesh=triangulate(self.targets,self.config)
        self.surface=SDFOracle(solid,self.targets.faces,self.mesh,self.config)
        self.edge=PolylineDistance(self.targets.edges) if self.targets.edges else None
        self.vtree=cKDTree(self.targets.vertices) if len(self.targets.vertices) else None
        self.fingerprint='benchmark-polyline-v2:'+hashlib.sha256(Path(step).read_bytes()).hexdigest()
        self.queries=0;self.sdf_queries=0

    def presence(self):return ('present' if self.edge else 'absent','present' if self.vtree else 'absent')

    def sdf(self, xyz):
        self.sdf_queries+=len(xyz)
        return self.surface.query(xyz)

    def evaluate(self, xyz):
        p=np.asarray(xyz,float).reshape(-1,3);self.queries+=len(p)
        de=self.edge.query(p)[0] if self.edge else np.full(len(p),.02)
        dv=self.vtree.query(p)[0] if self.vtree else np.full(len(p),.02)
        return self.result(np.column_stack((self.sdf(p),de,dv)))

    def gradient(self, xyz, channel, step=1e-5):
        p=np.asarray(xyz,float).reshape(-1,3)
        if channel==1 and self.edge:
            d,q,_=self.edge.query(p)
        elif channel==2 and self.vtree:
            d,i=self.vtree.query(p);q=self.targets.vertices[i]
        else:return super().gradient(p,channel,step)
        g=(p-q)/np.maximum(d[:,None],1e-12)
        return g,np.isfinite(g).all(axis=1)&(d>1e-8)

    def truth_face(self, xyz):
        v,f,ids=self.mesh
        _,i,_=igl.point_mesh_squared_distance(np.asarray(xyz,float),v,f)
        return ids[i]

    def validation(self, seed=20261016):
        rng=np.random.default_rng(seed)
        p=rng.uniform(-.95,.95,(128,3))
        if self.edge:
            chosen=self.edge.points[rng.integers(len(self.edge.points),size=128)]
            p=np.vstack((p,chosen+rng.normal(0,.005,chosen.shape)))
            ref=EdgeOracle(self.targets.edges,self.config).query(p)['distance']
            error=np.abs(ref-self.edge.query(p)[0])
            de={'queries':len(p),'max':float(error.max()),'mean':float(error.mean()),'passed':bool(error.max()<=1e-5)}
        else:de={'queries':0,'passed':True,'max':0.}
        q=p[:32]
        from bandbrep.vendor.oracle.distance import solve
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
        from OCC.Core.gp import gp_Pnt
        unsigned=np.array([solve(self.surface.tool,BRepBuilderAPI_MakeVertex(gp_Pnt(*x)).Vertex())[0] for x in q])
        se=np.abs(np.abs(self.sdf(q))-unsigned)
        return {'edge_reference':de,'sdf_unsigned_reference':{'queries':len(q),'max':float(se.max()),'mean':float(se.mean())},
            'scope':'Finite reference checks; no global field-error guarantee. SDF sign uses mesh_fast winding.'}


class PerturbedField(FieldProvider):
    def __init__(self, provider, amplitude, seed=20261016, channel=-1):
        self.provider=provider;self.amplitude=amplitude;self.channel=channel
        rng=np.random.default_rng(seed)
        direction=rng.normal(size=(12,3));direction/=np.linalg.norm(direction,axis=1)[:,None]
        self.wave=direction*rng.uniform(10,70,(12,1));self.phase=rng.uniform(0,2*np.pi,(12,3))
        self.error=provider.error+amplitude
        self.fingerprint=f'{provider.fingerprint}:bounded-spectral-v1:{amplitude}:{seed}:{channel}'

    def presence(self):return self.provider.presence()

    def noise(self, p):
        phase=np.asarray(p)@self.wave.T
        values=np.sin(phase[:,:,None]+self.phase[None]).mean(axis=1)*self.amplitude
        if self.channel>=0:values[:,np.arange(3)!=self.channel]=0
        return values

    def sdf(self, p):return self.provider.sdf(p)+self.noise(p)[:,0]

    def evaluate(self, p):
        values=self.provider.evaluate(p).values+self.noise(p)
        values[:,1:]=np.maximum(values[:,1:],0)
        for i,present in enumerate(self.presence(),1):
            if present=='absent':values[:,i]=.02
        return self.result(values)


class ScalarOnly(FieldProvider):
    def __init__(self, provider):
        self._source=provider;self.fingerprint=provider.fingerprint
        self.error=provider.error;self.strict_error_bound=provider.strict_error_bound
    def presence(self):return self._source.presence()
    def sdf(self,p):return self._source.sdf(p)
    def evaluate(self,p):return self._source.evaluate(p)
    def gradient(self,p,channel,step=1e-5):return self._source.gradient(p,channel,step)
