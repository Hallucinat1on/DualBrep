from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from .schemas import require
from .surface_mesh import evaluate_batched

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

def fit_cubic_fixed_ends(points,t,v0,v1,weights=None):
    t=np.asarray(t);b0=(1-t)**3;b1=3*(1-t)**2*t;b2=3*(1-t)*t*t;b3=t**3
    A=np.column_stack((b1,b2));rhs=points-b0[:,None]*v0-b3[:,None]*v1
    w=np.sqrt(np.ones(len(t)) if weights is None else weights)[:,None]
    internal,_,rank,_=np.linalg.lstsq(w*A,w*rhs,rcond=None)
    require(rank==2,'curve_rank_deficient','Insufficient internal parameter coverage')
    return np.vstack((v0,internal,v1))

def fit_circle(points):
    center=points.mean(axis=0);_,_,vt=np.linalg.svd(points-center,full_matrices=False)
    normal=vt[-1];x=vt[0];y=np.cross(normal,x);uv=np.column_stack(((points-center)@x,(points-center)@y))
    A=np.column_stack((2*uv,np.ones(len(uv))));b=np.sum(uv*uv,axis=1)
    fit=np.linalg.lstsq(A,b,rcond=None)[0];radius=np.sqrt(max(0,fit[2]+fit[0]**2+fit[1]**2))
    c=center+fit[0]*x+fit[1]*y;delta=points-c
    angles=np.unwrap(np.arctan2(delta@y,delta@x));sweep=angles[-1]-angles[0]
    error=np.sqrt(((delta@normal)**2)+(np.linalg.norm(delta-(delta@normal)[:,None]*normal,axis=1)-radius)**2)
    return dict(center=c,radius=radius,x=x,y=y,normal=normal,start=angles[0],sweep=sweep),float(error.max())

def fit_piecewise(points,cfg):
    """Hard shared joins + parameter-scaled C1 constraints on a single logical edge."""
    t=arclength_parameters(points)
    for segments in (1,2,4,8,16,32,64):
        if segments>cfg.max_segments:break
        breaks=np.linspace(0,1,segments+1)
        joins=np.column_stack([np.interp(breaks,t,points[:,a]) for a in range(3)])
        A=np.zeros((len(t),2*segments));rhs=points.copy()
        indices=np.clip(np.searchsorted(breaks,t,side='right')-1,0,segments-1)
        u=(t-breaks[indices])/(breaks[indices+1]-breaks[indices])
        for j in range(segments):
            mask=indices==j;q=u[mask]
            A[mask,2*j]=3*(1-q)**2*q;A[mask,2*j+1]=3*(1-q)*q*q
            rhs[mask]-=(1-q[:,None])**3*joins[j]+q[:,None]**3*joins[j+1]
        B=np.zeros((max(0,segments-1),2*segments));C=[]
        for j in range(segments-1):
            h0=breaks[j+1]-breaks[j];h1=breaks[j+2]-breaks[j+1]
            B[j,2*j+1]=1/h0;B[j,2*j+2]=1/h1;C.append(joins[j+1]*(1/h0+1/h1))
        from .surface_fitting import constrained_lstsq
        controls=constrained_lstsq(A,rhs,B,np.asarray(C).reshape(-1,3),regularization=1e-10)
        nets=np.array([np.vstack((joins[j],controls[2*j:2*j+2],joins[j+1])) for j in range(segments)])
        curve=CurveRecord(-1,'bezier_spline',{'breaks':breaks,'controls':nets},0.)
        error=np.linalg.norm(curve.evaluate(t)-points,axis=1).max()
        if error<=cfg.curve_tolerance:curve.residual=float(error);return curve
    require(False,'curve_fit_failed','Piecewise fixed-end C1 fitting exceeds segment / error budget')

def fit_shared_curves(graph,provider,cfg):
    curves={}
    for edge in graph.edges:
        p=edge.samples;t=arclength_parameters(p);record=None
        if not edge.closed:
            q=(1-t[:,None])*p[0]+t[:,None]*p[-1];err=np.linalg.norm(q-p,axis=1).max()
            if err<=cfg.curve_tolerance:record=CurveRecord(edge.id,'line',{'p0':p[0],'p1':p[-1]},float(err))
        if record is None and len(p)>=6:
            data,err=fit_circle(p)
            if err<=cfg.curve_tolerance and (edge.closed or not edge.vertices):
                if edge.closed:data['sweep']=np.sign(data['sweep'])*2*np.pi
                record=CurveRecord(edge.id,'circle',data,err)
        if record is None:record=fit_piecewise(p,cfg);record.id=edge.id
        length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
        count=max(500,len(p)*2,int(np.ceil(length/(cfg.curve_tolerance*.5)))+1)
        test=record.evaluate(np.linspace(0,1,count))
        values=evaluate_batched(provider,test,cfg.query_batch_size)
        require(values[:,1].max()<=cfg.curve_tolerance*2,'curve_field_failed','Fitted curve violates edge UDF',edge=edge.id,error=float(values[:,1].max()))
        require(np.abs(values[:,0]).max()<=cfg.surface_tolerance,'curve_surface_failed','Fitted curve violates SDF',edge=edge.id)
        cover=cKDTree(test).query(p)[0].max()
        require(cover<=cfg.curve_tolerance*2,'curve_coverage_failed','Fitted curve loses branch coverage',edge=edge.id,error=float(cover))
        if edge.vertices:
            ends=record.evaluate(np.array([0.,1.]))
            target=np.array([graph.vertices[v].position for v in edge.vertices])
            require(np.linalg.norm(ends-target,axis=1).max()<1e-8,'curve_endpoint_moved','Curve moved fixed endpoint')
        curves[edge.id]=record
    return curves
