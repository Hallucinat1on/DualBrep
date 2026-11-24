"""Mesh-derived geometry and continuous 8D E/V joint field queries [dE, dV, uE(3), uV(3)].

Only the mesh supplies SDF. External field providers supply the 8D joint target:
scalar unsigned distances [dE, dV] and 3D closest-point displacement vectors [uE, uV].
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np

from .deadline import checkpoint
from .schemas import require

EMPTY_DISTANCE = 2 * np.sqrt(3.0)


@dataclass
class FieldEvaluation:
    values: np.ndarray      # [N, 2] ([dE, dV]) or [N, 3] ([sdf, dE, dV])
    offsets: np.ndarray     # [N, 2, 3] ([uE, uV])
    valid: np.ndarray       # aligned with values
    saturated: np.ndarray   # aligned with values
    error: np.ndarray       # aligned with values
    strict_error_bound: bool = False


class CallableUDFProvider:
    """Continuous 8D joint field [dE, dV, uE_x, uE_y, uE_z, uV_x, uV_y, uV_z]."""

    def __init__(self, fn, *, fingerprint, presence, band_width=0.02):
        require(isinstance(fingerprint, str) and bool(fingerprint), 'field_identity', 'Supply a field fingerprint')
        require(np.isfinite(band_width) and band_width > 0, 'field_domain', 'Invalid field band width')
        require(presence is not None and len(presence) == 2 and all(v in ('present', 'absent') for v in presence),
                'invalid_presence', 'Expected E/V presence')
        self.fn = fn
        self.fingerprint = fingerprint
        self.band_width = float(band_width)
        self._presence = tuple(presence)

    def presence(self):
        return self._presence

    def evaluate(self, xyz):
        q = np.asarray(xyz, float).reshape(-1, 3)
        result = self.fn(q)
        if isinstance(result, FieldEvaluation):
            values = np.asarray(result.values, float)
            offsets = np.asarray(result.offsets, float)
            valid = np.asarray(result.valid, bool)
            saturated = np.asarray(result.saturated, bool)
            require(values.shape == (len(q), 2) and offsets.shape == (len(q), 2, 3),
                    'field_shape', 'FieldEvaluation must have values [N,2] and offsets [N,2,3]')
            require(valid.shape == values.shape and saturated.shape == values.shape,
                    'field_shape', 'FieldEvaluation masks must align with values [N,2]')
        else:
            raw = np.asarray(result, float)
            require(raw.shape == (len(q), 8), 'field_shape',
                    'Continuous 8D field must return [N,8]: [dE, dV, uE(3), uV(3)]')
            values = raw[:, :2]
            offsets = raw[:, 2:].reshape(len(q), 2, 3)
            valid = np.isfinite(values) & np.isfinite(offsets).all(axis=2)
            saturated = values < 0
        saturated = saturated | (values < 0)
        valid = valid & np.isfinite(values) & np.isfinite(offsets).all(axis=2)
        return FieldEvaluation(values, offsets, valid, saturated, np.zeros_like(values))


class MeshFieldProvider:
    """Internal [mesh SDF, dE, dV] + 3D closest-point displacements [uE(3), uV(3)]."""
    channel_names = ('sdf', 'dE', 'dV')
    strict_error_bound = False
    error = 0.0

    def __init__(self, mesh, udf, cfg):
        from veudf.geometry.mesh import MeshSDFOracle
        from veudf.geometry.sampling import mesh_hash

        identity = mesh_hash(mesh.vertices, mesh.triangles)
        require(getattr(udf, 'mesh_fingerprint', identity) == identity,
                'condition_mismatch', 'Decoder condition differs from the input mesh')
        self.oracle = MeshSDFOracle(mesh.vertices, mesh.triangles)
        self.udf = udf
        self.band_width = float(udf.band_width)
        self.normalization = mesh.report.get('normalization', {})
        self.batch_size = cfg.query_batch_size
        self.query_report = dict(outside_band=0, backtracked_steps=0)
        self._presence = udf.presence()
        require(self._presence is not None and len(self._presence) == 2
                and all(p in ('present', 'absent') for p in self._presence),
                'presence_unknown', 'Resolve E/V presence before reconstruction')
        require(self._presence != ('absent', 'present'), 'invalid_presence', 'Vertices require edges')
        policy = dict(
            mesh=identity,
            udf=udf.fingerprint,
            band_width=self.band_width,
            presence=self._presence,
            field_target='8d_udf_gdf',
            normalization=self.normalization,
        )
        self.fingerprint = 'mesh-udf8d:' + hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()

    def presence(self):
        return self._presence

    def sdf(self, xyz):
        checkpoint('mesh_distance')
        q = np.asarray(xyz, float).reshape(-1, 3)
        require(np.isfinite(q).all(), 'invalid_field_query', 'Nonfinite query coordinates')
        return self.oracle.query(q)

    def domain_mask(self, xyz):
        q = np.asarray(xyz, float).reshape(-1, 3)
        finite = np.isfinite(q).all(1)
        mask = finite & (np.abs(q) <= 1.000001).all(1)
        ids = np.flatnonzero(mask)
        if len(ids):
            mask[ids] &= np.abs(self.sdf(q[ids])) <= self.band_width
        return mask

    def evaluate(self, xyz):
        q = np.asarray(xyz, float).reshape(-1, 3)
        sdf = self.sdf(q)
        values = np.full((len(q), 3), EMPTY_DISTANCE)
        values[:, 0] = sdf
        offsets = np.zeros((len(q), 2, 3), dtype=float)
        valid = np.zeros_like(values, bool)
        valid[:, 0] = np.isfinite(sdf)
        saturated = np.zeros_like(valid)
        inside = (np.abs(q) <= 1.000001).all(1) & (np.abs(sdf) <= self.band_width)
        self.query_report['outside_band'] += int((~inside).sum())
        ids = np.flatnonzero(inside)
        if len(ids) and 'present' in self._presence:
            r = self.udf.evaluate(q[ids])
            values[ids, 1:] = r.values
            offsets[ids] = r.offsets
            valid[ids, 1:] = r.valid
            saturated[ids, 1:] = r.saturated
        for c, p in enumerate(self._presence, 1):
            if p == 'absent':
                values[:, c] = EMPTY_DISTANCE
                offsets[:, c - 1] = 0.0
                valid[:, c] = False
                saturated[:, c] = False
        return FieldEvaluation(values, offsets, valid, saturated, np.zeros_like(values))


def checked_values(provider, xyz, batch=8192):
    """Require only present channels; absent sentinels are not valid geometry."""
    q = np.asarray(xyz, float).reshape(-1, 3)
    required = np.array([True] + [p == 'present' for p in provider.presence()])
    parts = []
    for start in range(0, len(q), batch):
        checkpoint('field_values')
        r = provider.evaluate(q[start:start + batch])
        checkpoint('field_values')
        require(np.all(r.valid[:, required]) and np.all(~r.saturated[:, required])
                and np.isfinite(r.values[:, required]).all(),
                'invalid_field_query',
                'Required field channel is invalid, outside its band, or clipped')
        parts.append(r.values)
    return np.concatenate(parts) if parts else np.empty((0, 3))


def bounded_update(provider, current, proposed):
    """Backtrack off-band steps without projecting them onto the mesh."""
    if not hasattr(provider, 'domain_mask'):
        return proposed
    result = np.asarray(proposed, float).copy()
    for _ in range(16):
        bad = ~provider.domain_mask(result)
        if not bad.any():
            return result
        provider.query_report['backtracked_steps'] += int(bad.sum())
        result[bad] = (current[bad] + result[bad]) * 0.5
    bad = ~provider.domain_mask(result)
    result[bad] = current[bad]
    return result
