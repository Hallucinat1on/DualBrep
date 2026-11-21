"""Numerical curve utilities and constrained geometric spline fitting."""
import numpy as np
from scipy.spatial import cKDTree
from .curve_fitting import CurveRecord, arclength_parameters
from .deadline import checkpoint


def bezier_basis(t, breaks):
    t=np.asarray(t);k=len(breaks)-1
    span=np.clip(np.searchsorted(breaks,t,side='right')-1,0,k-1)
    u=(t-breaks[span])/(breaks[span+1]-breaks[span])
    values=np.column_stack(((1-u)**3,3*(1-u)**2*u,3*(1-u)*u*u,u**3))
    basis=np.zeros((len(t),3*k+1))
    basis[np.arange(len(t))[:,None],3*span[:,None]+np.arange(4)]=values
    return basis


def curve_derivatives(record,t):
    t=np.asarray(t);d=record.data
    if record.kind=='line':
        return np.broadcast_to(d['p1']-d['p0'],(len(t),3)),np.zeros((len(t),3))
    if record.kind=='circle':
        a=d['start']+t*d['sweep'];x,y=d['x'],d['y']
        first=d['radius']*d['sweep']*(-np.sin(a)[:,None]*x+np.cos(a)[:,None]*y)
        second=-d['radius']*d['sweep']**2*(np.cos(a)[:,None]*x+np.sin(a)[:,None]*y)
        return first,second
    breaks=d['breaks'];span=np.clip(np.searchsorted(breaks,t,side='right')-1,0,len(breaks)-2)
    h=breaks[span+1]-breaks[span];u=(t-breaks[span])/h;p=d['controls'][span]
    first=3*((1-u)[:,None]**2*(p[:,1]-p[:,0])+2*(u*(1-u))[:,None]*(p[:,2]-p[:,1])+u[:,None]**2*(p[:,3]-p[:,2]))/h[:,None]
    second=6*((1-u)[:,None]*(p[:,2]-2*p[:,1]+p[:,0])+u[:,None]*(p[:,3]-2*p[:,2]+p[:,1]))/h[:,None]**2
    return first,second


def closest_parameters(record,points,initial=None):
    """Global coarse seeds plus safeguarded local Newton, with both endpoints.

    This is a numerical closest-point search, not a certified global bound.
    Multiple coarse seeds prevent a single warm start from fixing a wrong span.
    """
    p=np.asarray(points,float);d=record.data
    if record.kind=='line':
        direction=d['p1']-d['p0']
        t=np.clip((p-d['p0'])@direction/max(direction@direction,1e-30),0,1)
        return t,np.linalg.norm(p-record.evaluate(t),axis=1)
    if record.kind=='circle':
        rel=p-d['center'];angle=np.arctan2(rel@d['y'],rel@d['x'])
        phase=np.mod(np.sign(d['sweep'])*(angle-d['start']),2*np.pi)
        t=phase/abs(d['sweep']);outside=t>1
        ends=record.evaluate(np.array([0.,1.]))
        t[outside]=(np.linalg.norm(p[outside]-ends[1],axis=1)<np.linalg.norm(p[outside]-ends[0],axis=1)).astype(float)
        return t,np.linalg.norm(p-record.evaluate(t),axis=1)
    breaks=d['breaks']
    grid=np.unique(np.concatenate([np.linspace(a,b,17) for a,b in zip(breaks[:-1],breaks[1:])]))
    ids=cKDTree(record.evaluate(grid)).query(p,k=3)[1]
    candidates=grid[ids]
    if initial is not None:candidates=np.column_stack((candidates,initial))
    q=np.repeat(p,candidates.shape[1],axis=0);t=candidates.ravel()
    for _ in range(8):
        checkpoint('curve_closest_points')
        c=record.evaluate(t);first,second=curve_derivatives(record,t);delta=c-q
        numerator=np.einsum('ij,ij->i',delta,first)
        denominator=np.einsum('ij,ij->i',first,first)+np.einsum('ij,ij->i',delta,second)
        step=np.zeros_like(t);good=denominator>1e-16
        step[good]=np.clip(numerator[good]/denominator[good],-.1,.1)
        proposed=np.clip(t-step,0,1)
        current_error=(delta*delta).sum(1)
        for _ in range(6):
            bad=((record.evaluate(proposed)-q)**2).sum(1)>current_error+1e-24
            if not bad.any():break
            proposed[bad]=(proposed[bad]+t[bad])*.5
        if np.max(np.abs(proposed-t),initial=0)<1e-11:
            t=proposed;break
        t=proposed
    candidates=np.column_stack((t.reshape(len(p),-1),np.zeros(len(p)),np.ones(len(p))))
    distance=np.linalg.norm(record.evaluate(candidates.ravel()).reshape(len(p),-1,3)-p[:,None],axis=2)
    best=np.argmin(distance,axis=1)
    return candidates[np.arange(len(p)),best],distance[np.arange(len(p)),best]


def _record(edge_id,breaks,poles):
    return CurveRecord(edge_id,'bezier_spline',dict(breaks=breaks,
        controls=np.array([poles[3*j:3*j+4] for j in range(len(breaks)-1)])),0.)


def _constraints(breaks,periodic,ends=None):
    k=len(breaks)-1;m=3*k+1;rows=[]
    for j in range(1,k):
        row=np.zeros(m);h0=breaks[j]-breaks[j-1];h1=breaks[j+1]-breaks[j]
        row[3*j-1]=-1/h0;row[3*j]=1/h0+1/h1;row[3*j+1]=-1/h1
        rows.append(row)
    if periodic:
        if ends is None:
            row=np.zeros(m);row[-1]=1;row[0]=-1;rows.append(row)
        row=np.zeros(m);h0=breaks[1]-breaks[0];h1=breaks[-1]-breaks[-2]
        row[-1]+=1/h1;row[-2]-=1/h1;row[1]-=1/h0;row[0]+=1/h0;rows.append(row)
    if not periodic or ends is not None:
        for index in (0,m-1):
            row=np.zeros(m);row[index]=1;rows.append(row)
    rhs=np.zeros((len(rows),3))
    if ends is not None:rhs[-2:]=ends
    return np.asarray(rows),rhs


def _constrained_lstsq(A, X, B, C, regularization=1e-10):
    from scipy import linalg
    from .schemas import require
    A = np.asarray(A, float); B = np.asarray(B, float); X = np.asarray(X, float); C = np.asarray(C, float)
    n = A.shape[1]
    if not len(B):
        return np.linalg.lstsq(np.vstack((A, np.sqrt(regularization) * np.eye(n))),
                               np.vstack((X, np.zeros((n, X.shape[1])))), rcond=None)[0]
    P = np.linalg.lstsq(B, C, rcond=1e-11)[0]
    require(np.max(np.abs(B @ P - C)) < 1e-7, 'boundary_constraints_conflict', 'Curve constraints are inconsistent')
    N = linalg.null_space(B, rcond=1e-11)
    if N.shape[1]:
        lhs = np.vstack((A @ N, np.sqrt(regularization) * N))
        rhs = np.vstack((X - A @ P, -np.sqrt(regularization) * P))
        P += N @ np.linalg.lstsq(lhs, rhs, rcond=1e-11)[0]
    return P


def projected_seed(ordered,ends,breaks,edge_id,periodic):
    """Robust point fit constrained to shared endpoints or a C1 periodic seam."""
    basis=bezier_basis(arclength_parameters(ordered),breaks)
    a,rhs=_constraints(breaks,periodic,ends if len(ends)==2 else None)
    weights=np.ones(len(ordered));poles=None
    for _ in range(4):
        w=np.sqrt(weights)[:,None]
        poles=_constrained_lstsq(w*basis,w*ordered,a,rhs,regularization=1e-10)
        residual=np.linalg.norm(basis@poles-ordered,axis=1)
        scale=max(1e-6,1.4826*np.median(abs(residual-np.median(residual))))
        weights=np.minimum(1.,1.5*scale/np.maximum(residual,1e-12))
    return _record(edge_id,breaks,poles)
