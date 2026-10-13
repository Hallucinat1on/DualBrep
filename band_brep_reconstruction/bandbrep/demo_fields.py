"""Analytic regression fields; CSG demo has correct zero set, not exact global SDF."""
import numpy as np
from .field_provider import CallableFieldProvider

def cylinder(radius=.65,half_height=.35):
    def fn(p):
        radial=np.linalg.norm(p[:,:2],axis=1);q=np.column_stack((radial-radius,np.abs(p[:,2])-half_height))
        sdf=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(axis=1),0)
        de=np.sqrt((radial-radius)**2+(np.abs(p[:,2])-half_height)**2)
        return np.column_stack((sdf,de,np.full(len(p),.02)))
    return CallableFieldProvider(fn,('present','absent'),fingerprint=f'analytic-cylinder:{radius}:{half_height}')

def sphere_with_bore(radius=.75,bore_radius=.3):
    z=np.sqrt(radius**2-bore_radius**2)
    def fn(p):
        radial=np.linalg.norm(p[:,:2],axis=1)
        s=np.maximum(np.linalg.norm(p,axis=1)-radius,bore_radius-radial)
        de=np.sqrt((radial-bore_radius)**2+(np.abs(p[:,2])-z)**2)
        return np.column_stack((s,de,np.full(len(p),.02)))
    return CallableFieldProvider(fn,('present','absent'),fingerprint=f'analytic-sphere-bore:{radius}:{bore_radius}')

def box(half_extent=.55):
    from itertools import product
    vertices=np.array(list(product((-half_extent,half_extent),repeat=3)))
    segments=[(a,b) for i,a in enumerate(vertices) for b in vertices[i+1:] if np.count_nonzero(a!=b)==1]
    def fn(p):
        q=np.abs(p)-half_extent;s=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(axis=1),0)
        distances=[]
        for a,b in segments:
            t=np.clip((p-a)@(b-a)/np.sum((b-a)**2),0,1);distances.append(np.linalg.norm(p-a-t[:,None]*(b-a),axis=1))
        dv=np.linalg.norm(p[:,None]-vertices[None],axis=2).min(axis=1)
        return np.column_stack((s,np.min(distances,axis=0),dv))
    return CallableFieldProvider(fn,('present','present'),fingerprint=f'analytic-box:{half_extent}')

def hollow_sphere(outer=.75,inner=.4):
    def fn(p):
        r=np.linalg.norm(p,axis=1)
        return np.column_stack((np.maximum(r-outer,inner-r),np.full((len(p),2),.02)))
    return CallableFieldProvider(fn,('absent','absent'),fingerprint=f'analytic-hollow-sphere:{outer}:{inner}')

def split_sphere(radius=.75,z=.1):
    ring=np.sqrt(radius*radius-z*z)
    def fn(p):
        radial=np.linalg.norm(p[:,:2],axis=1)
        return np.column_stack((np.linalg.norm(p,axis=1)-radius,np.sqrt((radial-ring)**2+(p[:,2]-z)**2),np.full(len(p),.02)))
    return CallableFieldProvider(fn,('present','absent'),fingerprint=f'analytic-split-sphere:{radius}:{z}')
