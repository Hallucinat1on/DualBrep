"""Analytic unit fields and closed meshes; no production compatibility layer."""
import numpy as np
from bandbrep.field_provider import CallableUDFProvider, FieldEvaluation
from bandbrep.schemas import SurfaceMesh

class AnalyticFields:
    """Internal three-channel fixture for geometry unit tests."""
    def __init__(self,fn,presence=('present','present')):self.fn=fn;self._presence=presence
    def presence(self):return self._presence
    def evaluate(self,q):
        v=np.asarray(self.fn(np.asarray(q,float)),float)
        return FieldEvaluation(v,np.isfinite(v),np.zeros_like(v,bool),np.zeros_like(v))
    def sdf(self,q):return self.evaluate(q).values[:,0]
    def gradient(self,q,channel,step=1e-5):
        q=np.asarray(q,float);g=np.empty_like(q)
        for a in range(3):
            d=np.eye(3)[a]*step
            g[:,a]=(self.evaluate(q+d).values[:,channel]-self.evaluate(q-d).values[:,channel])/(2*step)
        return g,np.isfinite(g).all(1)&(np.linalg.norm(g,axis=1)>1e-8)

def mesh_of(mesh):return SurfaceMesh(np.asarray(mesh.vertices,np.float64),np.asarray(mesh.faces,np.int64))

def box():
    import trimesh
    from itertools import product
    vertices=np.array(list(product((-.55,.55),repeat=3)))
    segments=[(a,b) for i,a in enumerate(vertices) for b in vertices[i+1:] if np.count_nonzero(a!=b)==1]
    def values(p):
        distances=[]
        for a,b in segments:
            t=np.clip((p-a)@(b-a)/np.sum((b-a)**2),0,1)
            distances.append(np.linalg.norm(p-a-t[:,None]*(b-a),axis=1))
        dv=np.linalg.norm(p[:,None]-vertices[None],axis=2).min(axis=1)
        return np.column_stack((np.min(distances,axis=0),dv))
    return mesh_of(trimesh.creation.box(extents=[1.1]*3)),CallableUDFProvider(values,fingerprint='box',presence=('present','present'))

def cylinder():
    import trimesh
    def values(p):
        de=np.hypot(np.linalg.norm(p[:,:2],axis=1)-.65,np.abs(p[:,2])-.35)
        return np.column_stack((de,np.full(len(p),3.)))
    # Segment both axial and radial directions before revolving; a single cap
    # fan has extremely skinny triangles and exhausts a red/green L5 budget.
    radial=np.linspace(0,.65,21)
    profile=np.vstack((np.column_stack((radial,np.full(21,-.35))),
        np.column_stack((np.full(24,.65),np.linspace(-.35,.35,25)[1:])),
        np.column_stack((radial[-2::-1],np.full(20,.35)))))
    return mesh_of(trimesh.creation.revolve(profile,sections=128)),CallableUDFProvider(values,fingerprint='cylinder',presence=('present','absent'))

def hollow_sphere():
    import trimesh
    a=trimesh.creation.icosphere(subdivisions=3,radius=.75)
    b=trimesh.creation.icosphere(subdivisions=3,radius=.4);b.invert()
    return mesh_of(trimesh.util.concatenate([a,b])),CallableUDFProvider(lambda q:np.full((len(q),2),3.),fingerprint='hollow',presence=('absent','absent'))

def sphere_with_bore():
    import trimesh
    radius,bore=.75,.3;z=np.sqrt(radius**2-bore**2)
    zz=np.linspace(-z,z,129)
    profile=np.vstack((np.column_stack((np.sqrt(radius**2-zz**2),zz)),
        np.column_stack((np.full(39,bore),np.linspace(z,-z,41)[1:-1]))))
    def values(p):
        de=np.hypot(np.linalg.norm(p[:,:2],axis=1)-bore,np.abs(p[:,2])-z)
        return np.column_stack((de,np.full(len(p),3.)))
    return mesh_of(trimesh.creation.revolve(profile,sections=128)),CallableUDFProvider(values,fingerprint='bore',presence=('present','absent'))
