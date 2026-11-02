import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from bandbrep.config import Config
from bandbrep.reconstruct import reconstruct
from bandbrep.schemas import ReconstructionError
from helpers import box,cylinder,hollow_sphere,sphere_with_bore


class PipelineTests(unittest.TestCase):
    def test_box_resume_vertices_to_step(self):
        mesh,udf=box()
        from veudf.geometry.sampling import mesh_hash
        udf.mesh_fingerprint=mesh_hash(mesh.vertices,mesh.triangles)
        cfg=Config(face_samples=400,save_queries=False,stop_after='vertices')
        with tempfile.TemporaryDirectory() as tmp:
            result=reconstruct(mesh,udf,cfg,tmp)
            self.assertEqual(result['status'],'vertices_ready')
            result=reconstruct(mesh,udf,replace(cfg,stop_after='solid'),tmp,resume=True)
            g=result['graph']
            self.assertEqual((g.face_count,len(g.edges),len(g.vertices)),(6,12,8))
            self.assertEqual(result['report']['status'],'solid_validated')
            self.assertTrue((Path(tmp)/'reconstruction.step').exists())
            a=np.load(Path(tmp)/'input_mesh.npz');np.testing.assert_array_equal(a['vertices'],mesh.vertices)
            self.assertFalse((Path(tmp)/'sdf_surface.npz').exists())
            self.assertEqual(json.loads((Path(tmp)/'run.json').read_text())['stages'][0]['stage'],'mesh')
            run=json.loads((Path(tmp)/'run.json').read_text())
            self.assertEqual([s['stage'] for s in run['stages'][:3]],['mesh','coarse_refinement','fine_refinement'])
            band_report=next(s['report'] for s in run['stages'] if s['stage']=='bands')
            self.assertTrue(band_report['reused_center_queries'])
            self.assertEqual(band_report['field_query_points'],0)
            self.assertTrue((Path(tmp)/'coarse_mesh.npz').exists())
            self.assertTrue((Path(tmp)/'center_fields.npz').exists())
            self.assertFalse((Path(tmp)/'refinement_progress.json').exists())
            # Refinement preserves the original geometry, with each child tied to a parent.
            a=np.load(Path(tmp)/'working_mesh.npz');self.assertEqual(len(a['triangles']),len(a['parent_triangles']))

    def test_cylinder_full_step(self):
        mesh,udf=cylinder()
        with tempfile.TemporaryDirectory() as tmp:
            result=reconstruct(mesh,udf,Config(face_samples=400,save_queries=False),tmp)
            g=result['graph'];self.assertEqual((g.face_count,len(g.edges),len(g.vertices)),(3,2,0))
            self.assertEqual(result['report']['status'],'solid_validated')

    def test_cavity_absent_fields_and_resume(self):
        mesh,udf=hollow_sphere();cfg=Config(stop_after='seeds',save_queries=False)
        with tempfile.TemporaryDirectory() as tmp:
            reconstruct(mesh,udf,cfg,tmp)
            result=reconstruct(mesh,udf,replace(cfg,stop_after='solid'),tmp,resume=True)
            g=result['graph'];self.assertEqual((g.face_count,len(g.edges),len(g.vertices)),(2,0,0))
            self.assertEqual(result['report']['status'],'solid_validated')

    def test_sample_timeout_writes_failure(self):
        mesh,udf=box()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ReconstructionError) as ctx:reconstruct(mesh,udf,Config(save_queries=False,sample_timeout_seconds=.001),tmp)
            self.assertEqual(ctx.exception.code,'sample_timeout')
            report=json.loads((Path(tmp)/'quality.json').read_text())
            self.assertIn(report['failed_stage'],('mesh','coarse_refinement'));self.assertEqual(report['status'],'failed')
            self.assertFalse((Path(tmp)/'reconstruction.step').exists())

    def test_same_face_pair_two_loops_step(self):
        mesh,udf=sphere_with_bore()
        with tempfile.TemporaryDirectory() as tmp:
            result=reconstruct(mesh,udf,Config(face_samples=400,save_queries=False),tmp)
            g=result['graph'];self.assertEqual((g.face_count,len(g.edges),len(g.vertices)),(2,2,0))
            self.assertEqual(g.edges[0].faces,g.edges[1].faces)
            self.assertEqual(result['report']['status'],'solid_validated')

    def test_resume_after_fine_failure_reuses_completed_coarse_stage(self):
        from unittest.mock import patch
        mesh,udf=hollow_sphere()
        cfg=Config(stop_after='seeds',save_queries=False,refinement_coarse_edge=.1,refinement_target_edge=.05)
        with tempfile.TemporaryDirectory() as tmp:
            with patch('bandbrep.reconstruct.fine_refine_surface',side_effect=ReconstructionError('probe_failure','test failure')):
                with self.assertRaises(ReconstructionError):reconstruct(mesh,udf,cfg,tmp)
            run=json.loads((Path(tmp)/'run.json').read_text())
            self.assertEqual(run['failed_stage'],'fine_refinement')
            self.assertEqual([s['stage'] for s in run['stages']],['mesh','coarse_refinement'])
            self.assertNotIn('coarse_refinement_progress.json',run['artifacts'])
            # Ephemeral progress is excluded from immutable stage caches.
            (Path(tmp)/'coarse_refinement_progress.json').write_text('{}')
            with patch('bandbrep.reconstruct.coarse_refine_surface',side_effect=AssertionError('coarse rerun')):
                result=reconstruct(mesh,udf,cfg,tmp,resume=True)
            self.assertEqual(result['status'],'face_seeds_ready')
