import unittest
from types import SimpleNamespace
from unittest.mock import patch
import json
from pathlib import Path
import tempfile
import numpy as np
from scipy.spatial import cKDTree
from bandbrep.config import Config
from bandbrep.schemas import VertexNode, ReconstructionError
from bandbrep.curve_fitting import CurveRecord
from bandbrep.curve_distance_fitting import (recover_distance_curves,closest_parameters,
    split_bezier,refine_bezier_distance,curve_derivatives,continuity_nullspace)
from helpers import AnalyticFields
from bandbrep.edge_geometry import project_support


class DistanceCurveTests(unittest.TestCase):
    def test_projection_skips_already_converged_points(self):
        points=np.array([[0.,0,0],[.1,.00001,0],[.2,.002,0],[.3,.003,0]])
        provider=AnalyticFields(lambda q:np.column_stack((q[:,2],np.abs(q[:,1]),np.ones(len(q)))))
        calls=[];source=provider.gradient
        def gradient(q,channel,step=1e-5):
            calls.append(q.copy());return source(q,channel,step)
        provider.gradient=gradient
        result=project_support(points,provider,.075,Config())
        self.assertTrue(calls)
        self.assertTrue(all(np.all(q[:,1]>.00002) for q in calls))
        self.assertLess(np.abs(result[:,1]).max(),.00002)
    def test_fixed_line_uses_cached_nonzero_distances_without_gradients(self):
        x=np.linspace(-.4,.4,300)
        points=np.column_stack((x,.002*np.sin(np.arange(len(x))),np.zeros(len(x))))
        provider=AnalyticFields(lambda q:np.column_stack((q[:,2],np.hypot(q[:,1],q[:,2]),np.ones(len(q)))))
        provider.gradient=lambda *args: self.fail('A valid fixed line must not query a UDF gradient')
        bands=SimpleNamespace(points=points,values=provider.evaluate(points).values)
        vertices=[VertexNode(i,p,0.,.01) for i,p in enumerate([[-.4,0,0],[.4,0,0]])]
        topo=SimpleNamespace(face_count=2,edges=[dict(id=0,faces=[0,1],vertices=[0,1],nodes=list(range(len(x))))])
        graph,curves,report=recover_distance_curves(topo,bands,vertices,provider,Config(curve_recovery='direct_distance'))
        self.assertEqual(curves[0].kind,'line')
        self.assertLess(curves[0].residual,1e-12)
        self.assertEqual(report['edges'][0]['initialization_points'],0)
        self.assertEqual(graph.edges[0].vertices,[0,1])

    def test_direct_circle_and_self_loop_keep_one_logical_curve(self):
        angles=np.linspace(0,2*np.pi,300,endpoint=False)
        radius=.4+.002*np.cos(angles*5)
        points=np.column_stack((radius*np.cos(angles),radius*np.sin(angles),np.zeros(len(angles))))
        provider=AnalyticFields(lambda q:np.column_stack((q[:,2],np.hypot(np.linalg.norm(q[:,:2],axis=1)-.4,q[:,2]),np.ones(len(q)))))
        bands=SimpleNamespace(points=points,values=provider.evaluate(points).values)
        for anchored in (False,True):
            topo=SimpleNamespace(face_count=2,edges=[dict(id=0,faces=[0,1],vertices=[0] if anchored else [],nodes=list(range(len(points))))])
            vertices=[VertexNode(0,np.array([.4,0,0]),0.,.01)] if anchored else []
            graph,curves,report=recover_distance_curves(topo,bands,vertices,provider,
                Config(curve_recovery='direct_distance',curve_init_samples=64,curve_fit_samples=128))
            self.assertEqual(curves[0].kind,'circle')
            self.assertLess(curves[0].residual,1e-5)
            self.assertEqual(len(graph.edges),1)
            if anchored:np.testing.assert_allclose(curves[0].evaluate(np.array([0.,1.])),[[.4,0,0]]*2,atol=1e-10)

    def test_single_cubic_distance_optimization_needs_no_udf(self):
        controls=np.array([[[0.,0,0],[.3,.4,0],[.7,-.4,0],[1.,0,0]]])
        truth=CurveRecord(0,'bezier_spline',dict(breaks=np.array([0.,1.]),controls=controls),0.)
        t=np.linspace(.02,.98,300);first,_=curve_derivatives(truth,t)
        normal=np.column_stack((-first[:,1],first[:,0],np.zeros(len(t))))
        normal/=np.linalg.norm(normal,axis=1)[:,None]
        angle=np.arange(len(t))*2.399963
        direction=np.cos(angle)[:,None]*normal+np.sin(angle)[:,None]*[0,0,1]
        distances=np.full(len(t),.003);points=truth.evaluate(t)+distances[:,None]*direction
        perturbed=controls.copy();perturbed[0,1]+=[.005,.02,.005];perturbed[0,2]+=[-.005,-.02,-.003]
        initial=CurveRecord(0,'bezier_spline',dict(breaks=np.array([0.,1.]),controls=perturbed),0.)
        fitted,_=refine_bezier_distance(initial,points,distances,Config(curve_distance_iterations=40))
        self.assertEqual(len(fitted.data['controls']),1)
        np.testing.assert_allclose(fitted.evaluate(np.array([0.,1.])),controls[0,[0,-1]],atol=1e-12)
        self.assertLess(np.max(abs(closest_parameters(fitted,points)[1]-distances)),2e-5)
        self.assertLess(np.max(np.linalg.norm(fitted.evaluate(t)-truth.evaluate(t),axis=1)),1e-4)

    def test_general_open_curve_end_to_end_uses_one_cubic_when_sufficient(self):
        true=CurveRecord(0,'bezier_spline',dict(breaks=np.array([0.,1.]),
            controls=np.array([[[-.4,0,0],[-.2,.35,0],[.2,-.35,0],[.4,0,0]]])),0.)
        dense=true.evaluate(np.linspace(0,1,50001));tree=cKDTree(dense)
        t=np.linspace(.001,.999,600);tangent,_=curve_derivatives(true,t)
        normal=np.column_stack((-tangent[:,1],tangent[:,0],np.zeros(len(t))))
        normal/=np.linalg.norm(normal,axis=1)[:,None]
        points=true.evaluate(t)+(.002*np.cos(np.arange(len(t))*2.4))[:,None]*normal
        provider=AnalyticFields(lambda q:np.column_stack((q[:,2],tree.query(q)[0],np.ones(len(q)))))
        bands=SimpleNamespace(points=points,values=provider.evaluate(points).values)
        ends=true.evaluate(np.array([0.,1.]));vertices=[VertexNode(i,p,0.,.01) for i,p in enumerate(ends)]
        topo=SimpleNamespace(face_count=2,edges=[dict(id=0,faces=[0,1],vertices=[0,1],nodes=list(range(len(points))))])
        graph,curves,report=recover_distance_curves(topo,bands,vertices,provider,
            Config(curve_recovery='direct_distance',curve_init_samples=128,curve_distance_iterations=30))
        self.assertEqual(curves[0].kind,'bezier_spline')
        self.assertEqual(len(curves[0].data['controls']),1)
        self.assertEqual(len(graph.edges),1)
        np.testing.assert_allclose(curves[0].evaluate(np.array([0.,1.])),ends,atol=1e-10)
        self.assertLess(curves[0].residual,.001)

    def test_open_arc_keeps_fixed_endpoints_and_correct_branch(self):
        t=np.linspace(0,np.pi/2,200);radius=.4+.002*np.cos(np.arange(len(t))*2.4)
        points=np.column_stack((radius*np.cos(t),radius*np.sin(t),np.zeros(len(t))))
        provider=AnalyticFields(lambda q:np.column_stack((q[:,2],np.hypot(np.linalg.norm(q[:,:2],axis=1)-.4,q[:,2]),np.ones(len(q)))))
        bands=SimpleNamespace(points=points,values=provider.evaluate(points).values)
        vertices=[VertexNode(i,np.array(p),0.,.01) for i,p in enumerate([[.4,0,0],[0,.4,0]])]
        topo=SimpleNamespace(face_count=2,edges=[dict(id=0,faces=[0,1],vertices=[0,1],nodes=list(range(len(t))))])
        _,curves,_=recover_distance_curves(topo,bands,vertices,provider,Config(curve_recovery='direct_distance'))
        self.assertEqual(curves[0].kind,'circle')
        self.assertAlmostEqual(abs(curves[0].data['sweep']),np.pi/2,places=3)
        np.testing.assert_allclose(curves[0].evaluate(np.array([0.,1.])),[v.position for v in vertices],atol=1e-10)

    def test_complex_open_curve_adds_local_spans_within_one_edge(self):
        x=np.linspace(-.4,.4,50001)
        dense=np.column_stack((x,.08*np.sin(4*np.pi*(x+.4)/.8),x*0));tree=cKDTree(dense)
        t=np.linspace(-.399,.399,1000)
        y=.08*np.sin(4*np.pi*(t+.4)/.8)
        slope=.08*4*np.pi/.8*np.cos(4*np.pi*(t+.4)/.8)
        normal=np.column_stack((-slope,np.ones(len(t)),t*0));normal/=np.linalg.norm(normal,axis=1)[:,None]
        points=np.column_stack((t,y,t*0))+(.0015*np.cos(np.arange(len(t))*2.4))[:,None]*normal
        provider=AnalyticFields(lambda q:np.column_stack((q[:,2],tree.query(q)[0],np.ones(len(q)))))
        bands=SimpleNamespace(points=points,values=provider.evaluate(points).values)
        vertices=[VertexNode(i,p,0.,.01) for i,p in enumerate(dense[[0,-1]])]
        topo=SimpleNamespace(face_count=2,edges=[dict(id=0,faces=[0,1],vertices=[0,1],nodes=list(range(len(t))))])
        graph,curves,report=recover_distance_curves(topo,bands,vertices,provider,
            Config(curve_recovery='direct_distance',max_segments=12,curve_distance_iterations=30))
        self.assertEqual(len(graph.edges),1)
        self.assertGreater(len(curves[0].data['controls']),1)
        self.assertLessEqual(len(curves[0].data['controls']),12)
        self.assertLessEqual(curves[0].residual,.001)
        np.testing.assert_allclose(curves[0].evaluate(np.array([0.,1.])),dense[[0,-1]],atol=1e-10)

    def test_local_split_preserves_geometry_and_c1_steps_allow_join_motion(self):
        curve=CurveRecord(0,'bezier_spline',dict(breaks=np.array([0.,1.]),
            controls=np.array([[[0.,0,0],[.2,.5,0],[.8,-.5,0],[1.,0,0]]])),0.)
        split=split_bezier(curve,0);split=split_bezier(split,1)
        t=np.linspace(0,1,301)
        np.testing.assert_allclose(split.evaluate(t),curve.evaluate(t),atol=1e-12)
        z=continuity_nullspace(split.data['breaks'],False)
        self.assertGreater(np.linalg.norm(z[3]),.1)
        self.assertLess(np.linalg.norm(z[[0,-1]]),1e-12)
        poles=np.vstack((split.data['controls'][0],*[v[1:] for v in split.data['controls'][1:]]))
        poles+=z@np.random.default_rng(1).normal(size=(z.shape[1],3))*.01
        moved=CurveRecord(0,'bezier_spline',dict(breaks=split.data['breaks'],controls=np.array([poles[3*j:3*j+4] for j in range(3)])),0.)
        for join in moved.data['breaks'][1:-1]:
            left=curve_derivatives(moved,np.array([join-1e-10]))[0]
            right=curve_derivatives(moved,np.array([join+1e-10]))[0]
            np.testing.assert_allclose(left,right,atol=1e-7)

    def test_periodic_free_curve_preserves_closed_position_and_tangent(self):
        angles=np.linspace(0,2*np.pi,300,endpoint=False)
        radius=.4+.001*np.cos(np.arange(len(angles))*2.4)
        points=np.column_stack((radius*np.cos(angles),radius*np.sin(angles),angles*0))
        provider=AnalyticFields(lambda q:np.column_stack((q[:,2],np.hypot(np.linalg.norm(q[:,:2],axis=1)-.4,q[:,2]),np.ones(len(q)))))
        bands=SimpleNamespace(points=points,values=provider.evaluate(points).values)
        topology=SimpleNamespace(face_count=2,edges=[dict(id=0,faces=[0,1],vertices=[],nodes=list(range(len(points))))])
        graph,curves,_=recover_distance_curves(topology,bands,[],provider,
            Config(curve_recovery='direct_distance',curve_candidates='basic',max_segments=8,
                   curve_init_samples=128,curve_fit_samples=256,curve_refine_samples=512,curve_distance_iterations=30))
        curve=curves[0];self.assertTrue(graph.edges[0].closed)
        self.assertEqual(curve.kind,'bezier_spline')
        np.testing.assert_allclose(curve.evaluate(np.array([0.])),curve.evaluate(np.array([1.])),atol=1e-10)
        first=curve_derivatives(curve,np.array([0.,1.]))[0]
        np.testing.assert_allclose(first[0],first[1],atol=1e-9)

    def test_config_rejects_invalid_modes_and_sample_budgets(self):
        for kwargs in [dict(curve_recovery='other'),dict(curve_init_samples=0),
                       dict(curve_fit_samples=128),dict(curve_distance_iterations=True)]:
            with self.assertRaises(ValueError):Config(**kwargs).validate()

    def test_cli_curve_mode_overrides_json(self):
        from bandbrep.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.json'
            for stored,flags,expected in [('direct_distance',[],'direct_distance'),
                    ('projected',['--curve-recovery','direct_distance'],'direct_distance')]:
                path.write_text(json.dumps(dict(curve_recovery=stored)))
                args=['reconstruct','--mesh','mesh','--udf-model','model.pt','--output',tmp,'--config',str(path),*flags]
                with patch('sys.argv',args),patch('bandbrep.cli.load_mesh'), \
                        patch('bandbrep.udf_model.TorchScriptUDFProvider'),patch('bandbrep.cli.reconstruct') as run:
                    main()
                self.assertEqual(run.call_args.args[2].curve_recovery,expected)


if __name__=='__main__':unittest.main()
