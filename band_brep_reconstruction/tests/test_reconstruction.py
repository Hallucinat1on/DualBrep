import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from bandbrep.config import Config
from bandbrep.schemas import SurfaceMesh,InterfaceGraph,ReconstructionError
from bandbrep.field_provider import CallableFieldProvider,SampledFieldProvider
from bandbrep.surface_mesh import mesh_edges,split_edges
from bandbrep.face_seeds import core_components
from bandbrep.interfaces import lower_envelope_segments
from bandbrep.vertex_localization import distance_center
from bandbrep.curve_fitting import fit_cubic_fixed_ends,fit_circle
from bandbrep.surface_fitting import constrained_lstsq,fit_four_sided_bezier
from bandbrep.edge_graph import instantiate_edges
from bandbrep.boundary_cycles import complete_face_regions_and_cycles
from bandbrep.reconstruct import reconstruct

class CoreTests(unittest.TestCase):
    def test_config_rejects_unresolved_band(self):
        with self.assertRaises(ValueError):Config(max_band_edge=.1).validate()
    def test_channel_scale_gradient(self):
        p=CallableFieldProvider(lambda x:np.column_stack((x[:,0],np.abs(x[:,1]),np.abs(x[:,2]))),normalized=True,clip_values=[2,3,4],presence=('present','present'))
        x=np.array([[.1,.2,.3]]);np.testing.assert_allclose(p.evaluate(x).values,[[.2,.6,1.2]])
        np.testing.assert_allclose(p.gradient(x,1)[0],[[0,3,0]],atol=1e-8)
    def test_negative_udf_rejected(self):
        with self.assertRaises(ReconstructionError):CallableFieldProvider(lambda x:np.full((len(x),3),-1.)).evaluate([[0,0,0]])
    def test_isolated_core_preserved(self):
        r=core_components(np.array([[0,1]]),np.array([True,True,False,True]));self.assertEqual(r[0],r[1]);self.assertNotEqual(r[0],r[3]);self.assertEqual(r[2],-1)
    def test_conforming_split(self):
        m=SurfaceMesh(np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]]),np.array([[0,1,2],[1,3,2]]))
        e,_,_=mesh_edges(m.triangles);r,_=split_edges(m,np.flatnonzero((e==[1,2]).all(axis=1)))
        self.assertEqual(len(r.vertices),5);self.assertEqual(len(r.triangles),4);_,_,c=mesh_edges(r.triangles);self.assertEqual(np.sum(c==1),4)
    def test_third_source_blocks_pair(self):self.assertEqual(lower_envelope_segments(np.array([[0.,1,1],[1,0,1],[-1,-1,-1]])),[])
    def test_three_source_envelope(self):
        c=np.array([[0.,2,2],[2,0,2],[2,2,0]]);s=lower_envelope_segments(c);self.assertEqual(len(s),3)
        for a,b,p,q in s:self.assertAlmostEqual(c[a]@p,c[b]@p);self.assertLessEqual(c[a]@q,np.min(c@q)+1e-8)
    def test_distance_center_not_centroid(self):
        v=np.array([.2,-.1,.3]);p=np.random.default_rng(1).uniform(0,.2,(100,3))+v;r,e,rank=distance_center(p,np.linalg.norm(p-v,axis=1))
        np.testing.assert_allclose(r,v,atol=1e-8);self.assertEqual(rank,3);self.assertGreater(np.linalg.norm(p.mean(axis=0)-v),.1)
    def test_rank_deficient_vertex(self):
        with self.assertRaises(ReconstructionError):distance_center(np.array([[0.,0,0],[1,0,0],[0,1,0]]),[1,1,1])
    def test_fixed_curve_endpoints(self):
        t=np.linspace(0,1,100);p=np.column_stack((t,t*(1-t),t*0));net=fit_cubic_fixed_ends(p,t,p[0],p[-1]);np.testing.assert_array_equal(net[0],p[0]);np.testing.assert_array_equal(net[-1],p[-1])
    def test_circle_not_polynomial_bezier(self):
        t=np.linspace(0,2*np.pi,120);p=np.column_stack((np.cos(t),np.sin(t),t*0));d,e=fit_circle(p);self.assertLess(e,1e-12);self.assertAlmostEqual(d['radius'],1.)
    def test_hard_boundary_constraints(self):
        B=np.array([[1.,1,0]]);C=np.array([[1.,2,3]]);P=constrained_lstsq(np.eye(3),np.zeros((3,3)),B,C);np.testing.assert_allclose(B@P,C,atol=1e-10)
    def test_conflicting_boundary_constraints(self):
        with self.assertRaises(ReconstructionError):constrained_lstsq(np.eye(2),np.zeros((2,3)),np.array([[1,0],[1,0]]),np.array([[0.,0,0],[1,0,0]]))
    def test_four_sides_whole_boundary(self):
        u=np.linspace(0,1,4);bottom=np.column_stack((u,0*u,.2*np.sin(np.pi*u)));top=bottom+[0,1,0];left=np.column_stack((0*u,u,0*u));right=left+[1,0,0]
        uv=np.random.default_rng(1).random((40,2));net=fit_four_sided_bezier(uv,np.column_stack((uv,np.zeros(40))),bottom,right,top,left)
        np.testing.assert_allclose(net[:,0],bottom,atol=1e-15);np.testing.assert_allclose(net[:,-1],top,atol=1e-15)
    def test_same_face_pair_two_closed_edges(self):
        points=[];segments=[]
        for z in (-.4,.4):
            offset=len(points);t=np.linspace(0,2*np.pi,20,endpoint=False);points.extend(np.column_stack((np.cos(t)*.3,np.sin(t)*.3,np.full(20,z))))
            segments.extend({'nodes':(offset+i,offset+(i+1)%20),'faces':(0,1),'left':0,'triangle':i} for i in range(20))
        g=instantiate_edges(InterfaceGraph(np.asarray(points),segments),[],None,Config(),2);self.assertEqual(len(g.edges),2);self.assertTrue(all(e.closed for e in g.edges));complete_face_regions_and_cycles(g);self.assertEqual(len(g.cycles[0]),2)
    def test_unknown_presence_is_not_absence(self):
        from bandbrep.bands import build_bands
        with self.assertRaises(ReconstructionError):build_bands(SurfaceMesh(np.zeros((3,3)),np.array([[0,1,2]])),CallableFieldProvider(lambda x:np.zeros_like(x)),Config())
    def test_sampled_reader_ignores_hidden_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);xyz=np.random.default_rng(1).random((300,3));v=np.column_stack((xyz[:,0]-.5,xyz[:,1],xyz[:,2]));(p/'metadata.json').write_text(json.dumps({'channel_names':['sdf','dE','dV']}))
            np.savez(p/'fields.npz',query_points=xyz,query_fields=v,has_edges=True,has_vertices=True,vertex_points=[[999,999,999]])
            provider=SampledFieldProvider(p);np.testing.assert_allclose(provider.evaluate([[.3,.4,.5]]).values,[[-.2,.4,.5]],atol=1e-5)
    def test_failure_report_no_false_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider=CallableFieldProvider(lambda x:np.column_stack((np.ones(len(x)),np.zeros((len(x),2)))),('absent','absent'))
            with self.assertRaises(ReconstructionError):reconstruct(provider,Config(resolution=9),tmp)
            report=json.loads((Path(tmp)/'quality.json').read_text());self.assertEqual(report['status'],'failed');self.assertFalse((Path(tmp)/'reconstruction.step').exists())

if __name__=='__main__':unittest.main()

class ExtendedTests(unittest.TestCase):
    def test_self_loop_has_two_ports_one_vertex(self):
        from bandbrep.schemas import VertexNode
        t=np.linspace(0,2*np.pi,20,endpoint=False);pts=np.column_stack((np.cos(t)*.3,np.sin(t)*.3,t*0))
        segments=[{'nodes':(i,(i+1)%20),'faces':(0,1),'left':0,'triangle':i} for i in range(20)]
        v=VertexNode(0,pts[0],0.,.01);g=instantiate_edges(InterfaceGraph(pts,segments),[v],None,Config(),2)
        self.assertEqual(len(g.edges),1);self.assertEqual(g.edges[0].vertices,[0,0]);self.assertEqual(len(v.ports),2)
    def test_face_scoring_handles_label_permutation(self):
        from bandbrep.evaluation import match_face_regions
        result=match_face_regions([1,1,0,0],[0,0,1,1]);self.assertEqual(result['mapping'],{0:1,1:0});self.assertEqual(result['merged_predictions'],[])
    def test_query_recording_keeps_metric_values(self):
        from bandbrep.query_recording import RecordingProvider
        with tempfile.TemporaryDirectory() as tmp:
            p=RecordingProvider(CallableFieldProvider(lambda x:np.column_stack((x[:,0],abs(x[:,1]),abs(x[:,2])))),tmp)
            p.evaluate(np.array([[.1,.2,.3]]));p.sdf(np.array([[.4,0,0]]));p.flush()
            data=np.load(Path(tmp)/'000000.npz');self.assertEqual(data['query_fields'][0,1],.2);self.assertTrue(np.isnan(data['query_fields'][1,1]))
    def test_cache_key_changes_for_field_and_config(self):
        from bandbrep.cache import cache_key
        p=CallableFieldProvider(lambda x:np.zeros_like(x),fingerprint='field-one')
        a,_=cache_key(p,Config(stop_after='seeds'));b,_=cache_key(p,Config(stop_after='solid'));self.assertEqual(a,b)
        c,_=cache_key(p,Config(tau_edge=.03));self.assertNotEqual(a,c)
    def test_corrupted_cache_rejected(self):
        from bandbrep.demo_fields import hollow_sphere
        # A tiny manually authored manifest exercises hash checking before any geometry call.
        from bandbrep.cache import cache_key
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);provider=hollow_sphere();cfg=Config(resolution=9);key,_=cache_key(provider,cfg)
            (p/'run.json').write_text(json.dumps({'cache_key':key,'status':'failed','artifacts':{'x.json':'incorrect'}}));(p/'x.json').write_text('{}')
            with self.assertRaises(ReconstructionError) as ctx:reconstruct(provider,cfg,p,resume=True)
            self.assertEqual(ctx.exception.code,'cache_corrupt')
