from dataclasses import dataclass
import numpy as np
from .schemas import require

@dataclass
class CurveRecord:
    id: int
    kind: str
    data: dict
    residual: float
    def evaluate(self,t):
        t=np.asarray(t,float)
        if self.kind=='line':return (1-t[:,None])*self.data['p0']+t[:,None]*self.data['p1']
        if self.kind=='circle':
            d=self.data;a=d['start']+t*d['sweep']
            return d['center']+d['radius']*(np.cos(a)[:,None]*d['x']+np.sin(a)[:,None]*d['y'])
        knots=self.data['breaks'];controls=self.data['controls'];i=np.clip(np.searchsorted(knots,t,side='right')-1,0,len(controls)-1)
        u=(t-knots[i])/(knots[i+1]-knots[i]);b=np.column_stack(((1-u)**3,3*(1-u)**2*u,3*(1-u)*u*u,u**3))
        return np.einsum('ni,nij->nj',b,controls[i])

def arclength_parameters(points):
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    require(s[-1]>1e-12,'zero_length_edge','Curve support has zero length')
    return s/s[-1]

def fit_circle(points):
    center=points.mean(axis=0);_,_,vt=np.linalg.svd(points-center,full_matrices=False)
    normal=vt[-1];x=vt[0];y=np.cross(normal,x);uv=np.column_stack(((points-center)@x,(points-center)@y))
    A=np.column_stack((2*uv,np.ones(len(uv))));b=np.sum(uv*uv,axis=1)
    fit=np.linalg.lstsq(A,b,rcond=None)[0];radius=np.sqrt(max(0,fit[2]+fit[0]**2+fit[1]**2))
    c=center+fit[0]*x+fit[1]*y;delta=points-c
    angles=np.unwrap(np.arctan2(delta@y,delta@x));sweep=angles[-1]-angles[0]
    error=np.sqrt(((delta@normal)**2)+(np.linalg.norm(delta-(delta@normal)[:,None]*normal,axis=1)-radius)**2)
    return dict(center=c,radius=radius,x=x,y=y,normal=normal,start=angles[0],sweep=sweep),float(error.max())

def fit_fixed_arc(points,v0,v1):
    """Circular arc through both fixed endpoints; center on the chord bisector.

    Returns (data, max_error) or None.  Endpoints are reproduced to round-off,
    so shared vertices never move.
    """
    chord=v1-v0;length=np.linalg.norm(chord)
    if length<1e-9 or len(points)<3:return None
    c=chord/length;m=(v0+v1)/2
    normal=np.linalg.svd(np.vstack((points,v0,v1))-np.vstack((points,v0,v1)).mean(axis=0),full_matrices=False)[2][-1]
    normal=normal-(normal@c)*c
    if np.linalg.norm(normal)<1e-6:return None
    normal/=np.linalg.norm(normal);w=np.cross(normal,c)
    q=points-m;a=q@w;b=q@c;h=length/2
    # |q - s w| = sqrt(h^2 + s^2)  <=>  a^2 + b^2 - 2 a s = h^2  (in-plane, linear in s)
    keep=np.abs(a)>1e-9
    if keep.sum()<2:return None
    s=float(np.sum(a[keep]*(a[keep]**2+b[keep]**2-h*h))/np.sum(2*a[keep]**2))
    center=m+s*w;radius=float(np.hypot(h,s))
    x=(v0-center)/radius;y=np.cross(normal,x)
    end=v1-center;angle_end=float(np.arctan2(end@y,end@x))
    angles=np.arctan2((points-center)@y,(points-center)@x)
    sweeps=[np.mod(angle_end,2*np.pi),np.mod(angle_end,2*np.pi)-2*np.pi]
    best=None
    for sweep in sweeps:
        if abs(sweep)<1e-9:continue
        rel=np.mod(angles,2*np.pi) if sweep>0 else np.mod(angles,2*np.pi)-2*np.pi
        inside=np.mean((rel/sweep>=-1e-6)&(rel/sweep<=1+1e-6))
        if best is None or inside>best[0]:best=(inside,sweep)
    if best is None or best[0]<.95:return None
    delta=points-center;off=delta@normal
    error=np.sqrt(off**2+(np.linalg.norm(delta-off[:,None]*normal,axis=1)-radius)**2)
    return dict(center=center,radius=radius,x=x,y=y,normal=normal,start=0.,sweep=float(best[1])),float(error.max())
