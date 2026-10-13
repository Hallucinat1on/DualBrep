"""Only scalar fields / gradients cross this boundary; topology is never exposed."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .schemas import require

@dataclass
class FieldEvaluation:
    values: np.ndarray
    valid: np.ndarray
    saturated: np.ndarray
    error: np.ndarray
    strict_error_bound: bool = False

class FieldProvider:
    channel_names = ('sdf','dE','dV')
    strict_error_bound = False
    error = 0.
    fingerprint = 'unidentified-provider'
    def presence(self): return ('unknown','unknown')
    def evaluate(self,xyz): raise NotImplementedError
    def sdf(self,xyz): return self.evaluate(xyz).values[:,0]
    def gradient(self,xyz,channel,step=1e-5):
        xyz=np.asarray(xyz,dtype=float).reshape(-1,3)
        g=np.empty_like(xyz)
        for a in range(3):
            offset=np.eye(3)[a]*step
            if channel==0: g[:,a]=(self.sdf(xyz+offset)-self.sdf(xyz-offset))/(2*step)
            else: g[:,a]=(self.evaluate(xyz+offset).values[:,channel]-self.evaluate(xyz-offset).values[:,channel])/(2*step)
        norm=np.linalg.norm(g,axis=1)
        return g,np.isfinite(g).all(axis=1)&(norm>1e-8)
    def result(self,values,saturated=None):
        values=np.asarray(values,float).reshape(-1,3)
        require(np.isfinite(values).all(),'invalid_field','Nonfinite field output')
        require(np.all(values[:,1:]>=-1e-8),'invalid_field','UDF must be nonnegative')
        values[:,1:]=np.maximum(values[:,1:],0)
        return FieldEvaluation(values,np.ones_like(values,bool),
            np.zeros_like(values,bool) if saturated is None else saturated,
            np.full_like(values,self.error),self.strict_error_bound)

class CallableFieldProvider(FieldProvider):
    def __init__(self,fn,presence=('unknown','unknown'),gradient=None,error=0.,fingerprint=None,
                 clip_values=None,normalized=False,strict_error_bound=False):
        self.fn,self._presence,self.gradfn=fn,tuple(presence),gradient
        require(all(v in ('present','absent','unknown') for v in presence),'invalid_presence','invalid presence')
        self.error,self.strict_error_bound=error,strict_error_bound
        self.fingerprint=fingerprint or 'callable-unspecified'
        self.clips=np.asarray(clip_values,float) if clip_values is not None else None
        self.normalized=normalized
        if normalized: require(self.clips is not None and np.all(self.clips>0),'invalid_scale','clip_values required')
    def presence(self): return self._presence
    def evaluate(self,xyz):
        v=np.asarray(self.fn(np.asarray(xyz,float)),float)
        saturated=np.abs(v)>=1-1e-7 if self.normalized else np.zeros_like(v,bool)
        if self.normalized: v=v*self.clips
        return self.result(v,saturated)
    def gradient(self,xyz,channel,step=1e-5):
        if self.gradfn is None: return super().gradient(xyz,channel,step)
        g=np.asarray(self.gradfn(np.asarray(xyz,float),channel),float)
        if self.normalized:g=g*self.clips[channel]
        return g,np.isfinite(g).all(axis=1)&(np.linalg.norm(g,axis=1)>1e-8)

class CadOracleProvider(FieldProvider):
    """Explicit oracle experiment; private CAD stays outside reconstruction stages."""
    def __init__(self,step_path,metadata=None,sdf_backend='cad',error=0.):
        from .vendor.oracle.config import Config
        from .vendor.oracle.cad import read_normalized,collect_targets,triangulate
        from .vendor.oracle.distance import EdgeOracle,VertexOracle,SDFOracle
        cfg=Config(sdf_backend=sdf_backend)
        solid,norm=read_normalized(step_path,cfg)
        if metadata:
            expected=metadata['normalization']['world_to_normalized']
            require(np.allclose(norm['world_to_normalized'],expected,rtol=1e-10,atol=1e-10),
                    'normalization_mismatch','Saved and oracle coordinate transforms disagree')
        targets=collect_targets(solid,cfg)
        mesh=triangulate(targets,cfg)
        self._sdf=SDFOracle(solid,targets.faces,mesh,cfg)
        self._edge=EdgeOracle(targets.edges,cfg)
        self._vertex=VertexOracle(targets.vertices,cfg)
        self._presence=('present' if targets.edges else 'absent','present' if len(targets.vertices) else 'absent')
        self.error=error
        self.normalization=norm
        digest=hashlib.sha256(Path(step_path).read_bytes()).hexdigest()
        self.fingerprint=f'cad:{digest}:{sdf_backend}:{error}'
        # Exact vectorized full-circle queries; finite arcs stay on the copied oracle.
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
        from OCC.Core.GeomAbs import GeomAbs_Circle
        self._circles=[]
        for edge in targets.edges:
            c=BRepAdaptor_Curve(edge)
            if c.GetType()!=GeomAbs_Circle or abs(c.LastParameter()-c.FirstParameter()-2*np.pi)>1e-7:
                self._circles=None;break
            circle=c.Circle()
            self._circles.append((np.array([circle.Location().X(),circle.Location().Y(),circle.Location().Z()]),
                np.array([circle.Axis().Direction().X(),circle.Axis().Direction().Y(),circle.Axis().Direction().Z()]),circle.Radius()))
    def presence(self): return self._presence
    def sdf(self,xyz): return self._sdf.query(xyz)
    def evaluate(self,xyz):
        xyz=np.asarray(xyz,float).reshape(-1,3)
        if self._circles:
            distances=[]
            for center,normal,radius in self._circles:
                delta=xyz-center;z=delta@normal
                radial=np.linalg.norm(delta-z[:,None]*normal,axis=1)
                distances.append(np.sqrt(z*z+(radial-radius)**2))
            de=np.min(distances,axis=0)
        else: de=self._edge.query(xyz)['distance']
        return self.result(np.column_stack((self.sdf(xyz),de,self._vertex.query(xyz)['distance'])))

class SampledFieldProvider(FieldProvider):
    """Moving local affine interpolation. Approximation, NOT a continuous-field oracle."""
    def __init__(self,directory,neighbors=24,validation_fraction=.05,seed=0):
        directory=Path(directory)
        metadata=json.loads((directory/'metadata.json').read_text())
        require(metadata['channel_names']==list(self.channel_names),'channel_order','Expected [sdf,dE,dV]')
        with np.load(directory/'fields.npz',allow_pickle=False) as f:
            # Deliberate whitelist: do not read closest points or instance IDs.
            xyz=np.asarray(f['query_points'],float);values=np.asarray(f['query_fields'],float)
            presence=tuple('present' if bool(f[k]) else 'absent' for k in ('has_edges','has_vertices'))
        require(xyz.shape==values.shape and xyz.shape[1]==3,'field_shape','Invalid sampled field shapes')
        unique,idx=np.unique(xyz,axis=0,return_index=True)
        self.xyz,self.values=unique,values[idx]
        self._presence=presence;self.neighbors=min(neighbors,len(unique))
        require(self.neighbors>=4,'insufficient_samples','Need at least four unique points')
        rng=np.random.default_rng(seed);order=rng.permutation(len(unique))
        n=max(1,int(len(unique)*validation_fraction));hold=order[:n];train=order[n:]
        require(len(train)>=self.neighbors,'insufficient_samples','Not enough training positions')
        self.tree=cKDTree(unique[train]);self._train_values=self.values[train]
        prediction,spacing=self._interpolate(unique[hold])
        residual=np.abs(prediction-self.values[hold])
        self.validation={'mode':'sampled_local_affine','unique_positions':len(unique),'held_out':n,
                         'mae':residual.mean(axis=0).tolist(),'max_error':residual.max(axis=0).tolist(),
                         'p95':np.quantile(residual,.95,axis=0).tolist(),'strict_error_bound':False,
                         'limitation':'Finite samples and local affine fits do not certify unsampled zero sets.'}
        self.error=float(np.max(self.validation['p95']))
        self.tree=cKDTree(unique);self._train_values=self.values
        self.fingerprint='sampled:'+hashlib.sha256((directory/'fields.npz').read_bytes()).hexdigest()
        self.normalization=metadata.get('normalization')
    def presence(self):return self._presence
    def _interpolate(self,xyz):
        xyz=np.asarray(xyz,float).reshape(-1,3)
        dist,ids=self.tree.query(xyz,k=self.neighbors)
        delta=self.tree.data[ids]-xyz[:,None,:]
        scale=np.maximum(dist[:,-1],1e-8)
        A=np.concatenate((np.ones((*dist.shape,1)),delta/scale[:,None,None]),axis=2)
        w=1/np.maximum(dist,scale[:,None]*.05)**2
        lhs=np.einsum('nki,nk,nkj->nij',A,w,A)
        lhs+=np.eye(4)[None]*1e-8*np.trace(lhs,axis1=1,axis2=2)[:,None,None]
        rhs=np.einsum('nki,nk,nkc->nic',A,w,self._train_values[ids])
        v=np.linalg.solve(lhs,rhs)[:,0,:]
        v[:,1:]=np.maximum(v[:,1:],0)
        return v,dist[:,0]
    def evaluate(self,xyz):return self.result(self._interpolate(xyz)[0])
