"""Mesh-derived geometry and continuous metric E/V queries.

Only the mesh supplies SDF. No sampled-field interpolation or CAD truth loader
is part of the reconstruction input API.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .schemas import require
from .deadline import checkpoint

EMPTY_DISTANCE = 2 * np.sqrt(3.)
UDF_GRADIENT_STEP = 1e-5


@dataclass
class FieldEvaluation:
    values: np.ndarray
    valid: np.ndarray
    saturated: np.ndarray
    error: np.ndarray
    strict_error_bound: bool = False


@dataclass
class CenterFields:
    """Validated final triangle-center queries, bound to one continuous provider."""
    points: np.ndarray
    values: np.ndarray
    provider_fingerprint: str

    def validate(self, points, provider):
        require(self.provider_fingerprint == provider.fingerprint,
                'center_fields_identity', 'Center queries belong to another provider')
        require(np.array_equal(self.points, points) and self.values.shape == points.shape,
                'center_fields_alignment', 'Center queries do not match the triangle ordering')
        required = np.array([True] + [p == 'present' for p in provider.presence()])
        require(np.isfinite(self.values[:, required]).all()
                and (self.values[:, 1:][:, required[1:]] >= 0).all(),
                'invalid_field_query', 'Cached center fields are invalid')
        return self.values


def sampled_udf_gradient(evaluate, xyz, channel, step=UDF_GRADIENT_STEP,
                         *, domain_mask=None, batch_size=8192):
    """Central differences of E/V values only; never call a gradient backend.

    The center and all six spatial probes must be usable. A zero center value
    or a vanishing difference is unsuitable for distance-based localization.
    """
    require(channel in (0,1), 'field_channel', 'UDF gradient channel must be E=0 or V=1')
    require(np.isfinite(step) and step > 0, 'gradient_step', 'Gradient step must be finite and positive')
    q = np.asarray(xyz, float).reshape(-1,3)
    g = np.zeros_like(q); valid = np.zeros(len(q), bool)
    deltas = np.eye(3)*step
    for start in range(0,len(q),batch_size):
        checkpoint('field_gradient')
        points = q[start:start+batch_size]
        mask = np.isfinite(points).all(1)
        if domain_mask is not None:
            mask &= domain_mask(points)
            for delta in deltas:
                mask &= domain_mask(points+delta) & domain_mask(points-delta)
        ids = np.flatnonzero(mask)
        if not len(ids):continue
        center = evaluate(points[ids])
        good = (center.valid[:,channel] & ~center.saturated[:,channel]
                & np.isfinite(center.values[:,channel]) & (center.values[:,channel]>1e-8))
        ids = ids[good]
        if not len(ids):continue
        probes = points[ids]
        slopes = np.empty_like(probes); good = np.ones(len(ids),bool)
        for axis,delta in enumerate(deltas):
            plus,minus = evaluate(probes+delta),evaluate(probes-delta)
            slopes[:,axis] = (plus.values[:,channel]-minus.values[:,channel])/(2*step)
            good &= (plus.valid[:,channel] & minus.valid[:,channel]
                     & ~plus.saturated[:,channel] & ~minus.saturated[:,channel]
                     & (plus.values[:,channel]>=0) & (minus.values[:,channel]>=0))
        good &= np.isfinite(slopes).all(1) & (np.linalg.norm(slopes,axis=1)>1e-8)
        g[start+ids[good]] = slopes[good]
        valid[start+ids[good]] = True
    return g,valid


class CallableUDFProvider:
    """Continuous metric [dE,dV]; gradients sample values at spatial offsets.

    Gradient channels are 0=E, 1=V. Presence may be omitted and inferred on
    deterministic mesh-surface samples. No derivative callback is accepted.
    """
    def __init__(self, fn, *, fingerprint, presence=None, band_width=.02):
        require(isinstance(fingerprint, str) and bool(fingerprint), 'field_identity', 'Supply a field fingerprint')
        require(np.isfinite(band_width) and band_width > 0, 'field_domain', 'Invalid field band width')
        require(presence is None or (len(presence) == 2 and all(v in ('present','absent','unknown') for v in presence)),
                'invalid_presence', 'Expected E/V presence')
        self.fn = fn
        self.fingerprint, self.band_width = fingerprint, float(band_width)
        self._presence = tuple(presence) if presence is not None else None

    def presence(self):
        return self._presence

    def evaluate(self, xyz):
        q = np.asarray(xyz, float).reshape(-1, 3)
        result = self.fn(q)
        if hasattr(result, 'values'):
            values = np.asarray(result.values, float)
            valid = np.asarray(result.valid, bool)
            saturated = np.asarray(result.saturated, bool)
        else:
            values = np.asarray(result, float)
            valid = np.isfinite(values)
            saturated = values < 0
        require(values.shape == (len(q), 2) and valid.shape == values.shape and saturated.shape == values.shape,
                'field_shape', 'Continuous UDF must return [N,2], with aligned masks')
        saturated = saturated | (values < 0)
        valid = valid & np.isfinite(values)
        return FieldEvaluation(values, valid, saturated, np.zeros_like(values))

    def gradient(self, xyz, channel, step=UDF_GRADIENT_STEP):
        return sampled_udf_gradient(self.evaluate, xyz, channel, step)


class ProductionUDFProvider(CallableUDFProvider):
    """Adapt the production FieldProviderAdapter without using its SDF output."""
    def __init__(self, adapter):
        self.adapter = adapter
        require(adapter.sdf_source=="condition","condition_source","Use the input mesh condition for the production adapter")
        self.mesh_fingerprint=adapter.condition.fingerprint
        def evaluate(q):
            r = adapter.evaluate(q)
            return FieldEvaluation(r.values[:,1:],r.valid[:,1:],r.saturated[:,1:],r.error[:,1:])
        policy = dict(band_width=adapter.band_width, presence=adapter.presence(),
                      presence_support=np.asarray(getattr(adapter,'presence_support',[])).tolist())
        digest = hashlib.sha256((adapter.fingerprint+json.dumps(policy,sort_keys=True)).encode()).hexdigest()
        super().__init__(evaluate, fingerprint='production:'+digest, presence=adapter.presence(),
                         band_width=adapter.band_width)
        self.normalization = getattr(adapter,'normalization',None)


class MeshFieldProvider:
    """Internal [mesh SDF,dE,dV], including query-domain and gradient validity."""
    channel_names = ('sdf','dE','dV')
    strict_error_bound = False
    error = 0.

    def __init__(self, mesh, udf, cfg):
        from veudf.geometry.mesh import MeshSDFOracle
        from veudf.geometry.sampling import base_surface, mesh_hash
        identity=mesh_hash(mesh.vertices,mesh.triangles)
        require(getattr(udf,"mesh_fingerprint",identity)==identity,"condition_mismatch","Decoder condition differs from the input mesh")
        self.oracle = MeshSDFOracle(mesh.vertices,mesh.triangles)
        self.udf, self.band_width = udf, float(udf.band_width)
        self.normalization = mesh.report.get('normalization',{})
        self.batch_size = cfg.query_batch_size
        self.query_report = dict(outside_band=0, backtracked_steps=0)
        self._presence = udf.presence()
        self.presence_support = None
        if self._presence is None:
            points,_ = base_surface((mesh.vertices,mesh.triangles),cfg.presence_samples,cfg.seed)
            support = np.zeros(2,int);usable=np.zeros(2,int)
            for start in range(0,len(points),cfg.query_batch_size):
                checkpoint("presence")
                r = udf.evaluate(points[start:start+cfg.query_batch_size])
                good=r.valid&~r.saturated&np.isfinite(r.values)
                usable+=good.sum(0)
                support += ((r.values < cfg.presence_tau)&good).sum(0)
            require((usable>=cfg.presence_min_points).all(),"presence_unknown","Insufficient valid surface queries to infer presence")
            self.presence_support = support.tolist()
            self._presence = tuple('present' if n >= cfg.presence_min_points else 'absent' for n in support)
        require(len(self._presence)==2 and all(p in ('present','absent') for p in self._presence),
                'presence_unknown','Resolve E/V presence before reconstruction')
        require(self._presence != ('absent','present'),'invalid_presence','Vertices require edges')
        policy = dict(mesh=mesh_hash(mesh.vertices,mesh.triangles), udf=udf.fingerprint,
                      band_width=self.band_width, presence=self._presence,
                      udf_gradient=dict(method='central_difference',step=UDF_GRADIENT_STEP),
                      normalization=self.normalization, presence_support=self.presence_support)
        self.fingerprint = 'mesh-udf:'+hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()

    def presence(self):
        return self._presence

    def sdf(self, xyz):
        checkpoint("mesh_distance")
        q = np.asarray(xyz,float).reshape(-1,3)
        require(np.isfinite(q).all(),'invalid_field_query','Nonfinite query coordinates')
        return self.oracle.query(q)

    def domain_mask(self, xyz):
        q = np.asarray(xyz,float).reshape(-1,3)
        finite = np.isfinite(q).all(1)
        mask = finite & (np.abs(q)<=1.000001).all(1)
        ids = np.flatnonzero(mask)
        mask[ids] &= np.abs(self.sdf(q[ids])) <= self.band_width
        return mask

    def evaluate(self, xyz):
        q = np.asarray(xyz,float).reshape(-1,3)
        sdf = self.sdf(q)
        values = np.full((len(q),3), EMPTY_DISTANCE)
        values[:,0] = sdf
        valid = np.zeros_like(values,bool);valid[:,0] = np.isfinite(sdf)
        saturated = np.zeros_like(valid)
        inside = (np.abs(q)<=1.000001).all(1) & (np.abs(sdf)<=self.band_width)
        self.query_report['outside_band'] += int((~inside).sum())
        ids = np.flatnonzero(inside)
        if len(ids) and 'present' in self._presence:
            r = self.udf.evaluate(q[ids])
            values[ids,1:] = r.values
            valid[ids,1:] = r.valid
            saturated[ids,1:] = r.saturated
        for c,p in enumerate(self._presence,1):
            if p=='absent':
                values[:,c] = EMPTY_DISTANCE;valid[:,c] = False
        return FieldEvaluation(values,valid,saturated,np.zeros_like(values))

    def gradient(self, xyz, channel, step=UDF_GRADIENT_STEP):
        checkpoint("field_gradient")
        q = np.asarray(xyz,float).reshape(-1,3)
        require(channel in (0,1,2),'field_channel','Unknown channel')
        if channel==0:return self.oracle.gradient(q)
        g = np.zeros_like(q);valid = np.zeros(len(q),bool)
        if self._presence[channel-1]=='absent':return g,valid
        return sampled_udf_gradient(self.udf.evaluate, q, channel-1, step,
                                    domain_mask=self.domain_mask, batch_size=self.batch_size)


def checked_values(provider, xyz, batch=8192):
    """Require only present channels; absent sentinels are not valid geometry."""
    q = np.asarray(xyz,float).reshape(-1,3)
    required = np.array([True]+[p=='present' for p in provider.presence()])
    parts=[]
    for start in range(0,len(q),batch):
        checkpoint("field_values")
        r=provider.evaluate(q[start:start+batch])
        checkpoint("field_values")
        require(np.all(r.valid[:,required]) and np.all(~r.saturated[:,required])
                and np.isfinite(r.values[:,required]).all(),'invalid_field_query',
                'Required field channel is invalid, outside its band, or clipped')
        parts.append(r.values)
    return np.concatenate(parts) if parts else np.empty((0,3))


def bounded_update(provider, current, proposed):
    """Backtrack off-band steps without projecting them onto the mesh."""
    if not hasattr(provider,'domain_mask'):return proposed
    result=np.asarray(proposed,float).copy()
    for _ in range(16):
        bad=~provider.domain_mask(result)
        if not bad.any():return result
        provider.query_report['backtracked_steps'] += int(bad.sum())
        result[bad]=(current[bad]+result[bad])*.5
    bad=~provider.domain_mask(result)
    result[bad]=current[bad]
    return result
