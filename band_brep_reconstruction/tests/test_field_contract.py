import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from bandbrep.config import Config,load_config
from bandbrep.schemas import ReconstructionError,SurfaceMesh
from bandbrep.field_provider import (CallableUDFProvider,MeshFieldProvider,ProductionUDFProvider,
                                    FieldEvaluation,checked_values,bounded_update)
from bandbrep.surface_mesh import load_mesh,validate_mesh
from bandbrep.reconstruct import reconstruct
from helpers import box,hollow_sphere


class ContractTests(unittest.TestCase):
    def test_l5_defaults_and_removed_config(self):
        cfg=Config().validate()
        self.assertEqual((cfg.refinement_coarse_edge,cfg.refinement_target_edge,cfg.refinement_rho_edge,cfg.refinement_rho_vertex,cfg.sample_timeout_seconds),(.02,.005,.02,.03,10800.))
        self.assertEqual((cfg.vertex_domains,cfg.band_connectivity),('local','vertex'))
        for obsolete in ('resolution','refinement_max_rounds','max_triangles'):
            with self.assertRaises(ValueError):load_config(overrides={obsolete:257})
        for coarse in (True, 0., -.01, float('nan'), .001):
            with self.assertRaises(ValueError):Config(refinement_coarse_edge=coarse).validate()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.json';p.write_text('{"sample_timeout_seconds": 9}')
            self.assertEqual(load_config(p,{'sample_timeout_seconds':10}).sample_timeout_seconds,10)

    def test_mesh_sdf_and_metric_udf_gradient(self):
        mesh,udf=box();p=MeshFieldProvider(mesh,udf,Config())
        q=np.array([[.551,.53,.52]])
        self.assertAlmostEqual(p.sdf(q)[0],.001)
        g,valid=p.gradient(q,2);expected=q[0]-[.55,.55,.55];expected/=np.linalg.norm(expected)
        self.assertTrue(valid[0]);np.testing.assert_allclose(g[0],expected,atol=1e-7)
        self.assertTrue(checked_values(p,q).shape==(1,3))

    def test_absent_masks_do_not_block_surface(self):
        mesh,udf=hollow_sphere();p=MeshFieldProvider(mesh,udf,Config())
        r=p.evaluate(mesh.vertices[:3]);self.assertFalse(r.valid[:,1:].any())
        self.assertTrue(np.isfinite(checked_values(p,mesh.vertices[:3])).all())
        self.assertFalse(p.gradient(mesh.vertices[:3],1)[1].any())

    def test_off_band_never_queries_udf_and_update_backtracks(self):
        mesh,udf=box();p=MeshFieldProvider(mesh,udf,Config())
        def forbidden(q):raise AssertionError('off-band model query')
        udf.fn=forbidden
        with self.assertRaises(ReconstructionError):checked_values(p,[[0,0,0]])
        start=np.array([[.55,.1,.1]])
        q=bounded_update(p,start,start+[.2,0,0]);self.assertTrue(p.domain_mask(q).all())
        self.assertGreater(p.query_report['backtracked_steps'],0)

    def test_negative_nan_and_saturated_required_fields_rejected(self):
        mesh,_=box()
        for value in (-.1,float('nan')):
            udf=CallableUDFProvider(lambda q:np.full((len(q),2),value),fingerprint='bad',presence=('present','present'))
            with self.assertRaises(ReconstructionError):checked_values(MeshFieldProvider(mesh,udf,Config()),mesh.vertices[:1])
        udf=CallableUDFProvider(lambda q:FieldEvaluation(np.ones((len(q),2)),np.ones((len(q),2),bool),np.ones((len(q),2),bool),0),fingerprint='clipped',presence=('present','present'))
        with self.assertRaises(ReconstructionError):checked_values(MeshFieldProvider(mesh,udf,Config()),mesh.vertices[:1])

    def test_gradients_query_center_and_six_spatial_offsets(self):
        calls=[]
        def values(q):
            calls.append(q.copy())
            return np.column_stack((2+q@[1.,2.,3.],3+q@[4.,5.,6.]))
        udf=CallableUDFProvider(values,fingerprint='sampled')
        q=np.array([[.1,.2,.3]])
        g,valid=udf.gradient(q,1)
        np.testing.assert_allclose(g,[[4,5,6]],atol=1e-9)
        self.assertTrue(valid[0]);self.assertEqual(len(calls),7)
        np.testing.assert_array_equal(calls[0],q)
        for axis in range(3):
            delta=np.eye(3)[axis]*1e-5
            np.testing.assert_array_equal(calls[1+2*axis],q+delta)
            np.testing.assert_array_equal(calls[2+2*axis],q-delta)
        for step in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ReconstructionError):udf.gradient(q,0,step)

    def test_mesh_gradient_bypasses_custom_derivatives_and_batches_samples(self):
        mesh,source=box();calls=[]
        def values(q):
            calls.append(q.copy());return source.evaluate(q)
        def forbidden(*args):raise AssertionError('external gradient was called')
        udf=SimpleNamespace(evaluate=values,gradient=forbidden,presence=source.presence,
                            fingerprint='values-only',band_width=.02)
        p=MeshFieldProvider(mesh,udf,Config(query_batch_size=2))
        q=np.array([[.551,.53,.52],[.552,.53,.52],[.553,.53,.52]])
        g,valid=p.gradient(q,2)
        expected=q-[.55,.55,.55];expected/=np.linalg.norm(expected,axis=1)[:,None]
        np.testing.assert_allclose(g,expected,atol=1e-7)
        self.assertTrue(valid.all());self.assertTrue(all(len(v)<=2 for v in calls))
        self.assertEqual(sum(len(v) for v in calls),7*len(q))
        calls.clear()
        # The center is inside the band, but its positive x probe is outside.
        g,valid=p.gradient([[.57-5e-6,.1,.1]],2)
        self.assertFalse(valid[0]);self.assertEqual(calls,[])
        np.testing.assert_array_equal(g,[[0,0,0]])

    def test_gradient_rejects_bad_probe_values_and_zero_distance(self):
        q=np.array([[.1,.2,.3]])
        for failure in ('invalid','saturated','negative','nan','zero_center','flat'):
            with self.subTest(failure=failure):
                def values(points):
                    v=np.column_stack((1+points[:,0],2+points[:,1]))
                    valid=np.ones(v.shape,bool);saturated=np.zeros(v.shape,bool)
                    probe=points[:,0]>q[0,0]
                    if failure=='invalid':valid[probe,1]=False
                    if failure=='saturated':saturated[probe,1]=True
                    if failure=='negative':v[probe,1]=-.1
                    if failure=='nan':v[probe,1]=np.nan
                    if failure=='zero_center':v[:,1]=np.linalg.norm(points-q,axis=1)
                    if failure=='flat':v[:,1]=1.
                    return FieldEvaluation(v,valid,saturated,0)
                udf=CallableUDFProvider(values,fingerprint=failure)
                g,valid=udf.gradient(q,1)
                self.assertFalse(valid[0]);np.testing.assert_array_equal(g,[[0,0,0]])
                # A bad V value must not invalidate the independent E derivative.
                self.assertTrue(udf.gradient(q,0)[1][0])

    def test_production_values_only_adapter_and_condition_identity(self):
        mesh,_=box()
        from veudf.geometry.sampling import mesh_hash
        def gradient(q,channel):
            raise AssertionError('production gradient must not be used')
        adapter=SimpleNamespace(evaluate=lambda q:FieldEvaluation(np.column_stack((q[:,0],2+q[:,0],2+2*q[:,1])),np.ones((len(q),3),bool),np.zeros((len(q),3),bool),np.zeros((len(q),3))),
            gradient=gradient,band_width=.02,presence=lambda:('present','present'),fingerprint='fake',
            condition=SimpleNamespace(fingerprint=mesh_hash(mesh.vertices,mesh.triangles)),sdf_source='condition')
        udf=ProductionUDFProvider(adapter);p=MeshFieldProvider(mesh,udf,Config())
        g,valid=p.gradient(mesh.vertices[:1],2)
        self.assertTrue(valid[0]);np.testing.assert_allclose(g,[[0,2,0]],atol=1e-9)
        adapter.condition.fingerprint='wrong'
        with self.assertRaises(ReconstructionError):MeshFieldProvider(mesh,ProductionUDFProvider(adapter),Config())

    def test_presence_inference_uses_surface_without_labels(self):
        mesh,source=box()
        udf=CallableUDFProvider(source.fn,fingerprint='unknown',band_width=.02)
        p=MeshFieldProvider(mesh,udf,Config())
        self.assertEqual(p.presence(),('present','present'));self.assertGreater(p.presence_support[1],0)
        invalid=CallableUDFProvider(lambda q:np.full((len(q),2),np.nan),fingerprint='invalid')
        with self.assertRaises(ReconstructionError):MeshFieldProvider(mesh,invalid,Config(presence_samples=50))

    def test_condition_load_and_tamper_detection(self):
        from veudf.io.condition import write_condition
        mesh,_=box()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'condition';write_condition(path,(mesh.vertices,mesh.triangles),count=32)
            loaded=load_mesh(tmp);np.testing.assert_array_equal(mesh.triangles,loaded.triangles)
            v=np.load(path/'vertices.npy');v[0,0]+=.01;np.save(path/'vertices.npy',v)
            with self.assertRaises(ReconstructionError):load_mesh(tmp)

    def test_invalid_mesh_writes_failed_diagnostics(self):
        mesh,udf=box();mesh.triangles=mesh.triangles[:-1]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ReconstructionError):reconstruct(mesh,udf,Config(),tmp)
            self.assertEqual(json.loads((Path(tmp)/'quality.json').read_text())['failed_stage'],'mesh')

    def test_cache_identity_mesh_field_domain_and_corruption(self):
        from bandbrep.cache import cache_key
        mesh,udf=box();cfg=Config()
        key,_=cache_key(mesh,udf,cfg)
        self.assertEqual(key,cache_key(mesh,udf,replace(cfg,stop_after='seeds'))[0])
        self.assertNotEqual(key,cache_key(mesh,udf,replace(cfg,face_second_pass=False))[0])
        udf.band_width=.03;self.assertNotEqual(key,cache_key(mesh,udf,cfg)[0]);udf.band_width=.02
        mesh.vertices[0,0]+=.001;self.assertNotEqual(key,cache_key(mesh,udf,cfg)[0]);mesh.vertices[0,0]-=.001
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'x.json').write_text('{}')
            (out/'run.json').write_text(json.dumps({'cache_key':cache_key(mesh,udf,cfg)[0],'status':'failed','artifacts':{'x.json':'wrong'}}))
            with self.assertRaises(ReconstructionError) as error:reconstruct(mesh,udf,cfg,tmp,resume=True)
            self.assertEqual(error.exception.code,'cache_corrupt')

    def test_recording_preserves_metric_and_masks(self):
        from bandbrep.query_recording import RecordingProvider
        mesh,udf=box()
        with tempfile.TemporaryDirectory() as tmp:
            p=RecordingProvider(MeshFieldProvider(mesh,udf,Config()),tmp)
            p.evaluate(mesh.vertices[:1]);p.sdf([[0,0,0]]);p.flush()
            data=np.load(Path(tmp)/'000000.npz')
            self.assertEqual(data['query_fields'].shape,(2,3));self.assertTrue(np.isnan(data['query_fields'][1,1]))
            self.assertEqual(data['valid'][1].tolist(),[True,False,False])
