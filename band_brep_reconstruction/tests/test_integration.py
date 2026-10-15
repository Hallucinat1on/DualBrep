import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from bandbrep.config import Config
from bandbrep.demo_fields import cylinder,box,sphere_with_bore
from bandbrep.reconstruct import reconstruct

@unittest.skipUnless(importlib.util.find_spec('OCC') and importlib.util.find_spec('mcubes'),'OCC/PyMCubes required')
class PipelineTests(unittest.TestCase):
    def check_shape(self,provider,expected,resolution=49):
        with tempfile.TemporaryDirectory() as tmp:
            # These fixtures isolate unchanged geometry/CAD stages with resolved bands.
            # Default narrow-threshold behavior is tested separately below.
            cfg=Config(resolution=resolution,face_samples=400,save_queries=False,
                       face_threshold1=.025,face_threshold2=.05,face_filter_size=1)
            result=reconstruct(provider,cfg,tmp);g=result['graph']
            self.assertEqual((g.face_count,len(g.edges),len(g.vertices)),expected)
            self.assertEqual(json.loads((Path(tmp)/'run.json').read_text())['status'],'solid_validated')
            self.assertTrue((Path(tmp)/'reconstruction.step').exists())
            return g
    def test_cylinder_full_step(self):self.check_shape(cylinder(),(3,2,0))
    def test_box_shared_vertices_full_step(self):self.check_shape(box(),(6,12,8),65)
    def test_sphere_bore_same_pair_two_loops_full_step(self):
        graph=self.check_shape(sphere_with_bore(),(2,2,0),65)
        self.assertEqual(graph.edges[0].faces,graph.edges[1].faces)
    def test_cavity_shells_without_edges(self):
        from bandbrep.demo_fields import hollow_sphere
        self.check_shape(hollow_sphere(),(2,0,0),65)
    def test_smooth_split_sphere_retains_boundary(self):
        from bandbrep.demo_fields import split_sphere
        self.check_shape(split_sphere(),(2,1,0),65)
    def test_resume_seeds_to_graph_without_hidden_topology(self):
        from bandbrep.demo_fields import hollow_sphere
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as tmp:
            provider=hollow_sphere();cfg=Config(resolution=33,stop_after='seeds',save_queries=False)
            reconstruct(provider,cfg,tmp)
            result=reconstruct(provider,replace(cfg,stop_after='graph'),tmp,resume=True)
            self.assertEqual(result['status'],'logical_graph_ready')
            graph=json.loads((Path(tmp)/'incidence.json').read_text());self.assertEqual(graph['face_count'],2);self.assertEqual(graph['edges'],[])

    def test_coarse_segmentation_errors_are_recorded_not_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=reconstruct(box(),Config(resolution=49,stop_after='graph',save_queries=False),tmp)
            data=json.loads((Path(tmp)/'incidence.json').read_text())
            self.assertEqual(result['status'],'logical_graph_ready')
            self.assertEqual(data['face_count'],6)
            self.assertEqual(result['stages'][1]['report']['segmentation']['method'],'dualbrep_two_pass_triangle_components')
            self.assertFalse((Path(tmp)/'arrivals.npz').exists())
            self.assertFalse((Path(tmp)/'interfaces.npz').exists())
            self.assertFalse((Path(tmp)/'reconstruction.step').exists())

    def test_resume_localized_vertices_to_full_step(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as tmp:
            cfg=Config(resolution=65,face_samples=400,stop_after='vertices',save_queries=False,
                       face_threshold1=.025,face_threshold2=.05,face_filter_size=1)
            provider=box();result=reconstruct(provider,cfg,tmp)
            self.assertEqual(result['status'],'vertices_ready')
            self.assertFalse((Path(tmp)/'curves.json').exists())
            self.assertFalse((Path(tmp)/'cad_graph.json').exists())
            result=reconstruct(provider,replace(cfg,stop_after='curves'),tmp,resume=True)
            self.assertEqual(result['status'],'curves_ready')
            result=reconstruct(provider,replace(cfg,stop_after='solid'),tmp,resume=True)
            self.assertEqual(result['report']['status'],'solid_validated')
