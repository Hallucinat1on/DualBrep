import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial import cKDTree
from bandbrep.config import Config
from bandbrep.schemas import VertexNode, ReconstructionError
from bandbrep.curve_fitting import CurveRecord
from bandbrep.curve_distance_fitting import (fit_edge_distances,closest_parameters,
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
        vertices=[VertexNode(i,p,0.,.01) for i,p in enumerate([[-.4,0,0],[.4,0,0]])]
        curve, report = fit_edge_distances(0, points, provider.evaluate(points).values[:, 1],
                                           np.asarray([v.position for v in vertices]), False, provider, Config())
        self.assertEqual(curve.kind,'line')
        self.assertLess(curve.residual,1e-12)
        self.assertEqual(report['initialization_points'],0)

    def test_direct_circle_and_self_loop_keep_one_logical_curve(self):
        angles=np.linspace(0,2*np.pi,300,endpoint=False)
        radius=.4+.002*np.cos(angles*5)
        points=np.column_stack((radius*np.cos(angles),radius*np.sin(angles),np.zeros(len(angles))))
        provider=AnalyticFields(lambda q:np.column_stack((q[:,2],np.hypot(np.linalg.norm(q[:,:2],axis=1)-.4,q[:,2]),np.ones(len(q)))))
        for anchored in (False,True):
            vertices=[VertexNode(0,np.array([.4,0,0]),0.,.01)] if anchored else []
            curve, _ = fit_edge_distances(0, points, provider.evaluate(points).values[:, 1],
                                          np.asarray([v.position for v in vertices]).reshape(-1, 3),
                                          not anchored, provider, Config(curve_init_samples=64, curve_fit_samples=128))
            self.assertEqual(curve.kind,'circle')
            self.assertLess(curve.residual,1e-5)
            if anchored:np.testing.assert_allclose(curve.evaluate(np.array([0.,1.])),[[.4,0,0]]*2,atol=1e-10)

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
        ends=true.evaluate(np.array([0.,1.]));vertices=[VertexNode(i,p,0.,.01) for i,p in enumerate(ends)]
        curve, _ = fit_edge_distances(0, points, provider.evaluate(points).values[:, 1], ends, False,
                                      provider, Config(curve_init_samples=128, curve_distance_iterations=30))
        self.assertEqual(curve.kind,'bezier_spline')
        self.assertEqual(len(curve.data['controls']),1)
        np.testing.assert_allclose(curve.evaluate(np.array([0.,1.])),ends,atol=1e-10)
        self.assertLess(curve.residual,.001)

    def test_open_arc_keeps_fixed_endpoints_and_correct_branch(self):
        t=np.linspace(0,np.pi/2,200);radius=.4+.002*np.cos(np.arange(len(t))*2.4)
        points=np.column_stack((radius*np.cos(t),radius*np.sin(t),np.zeros(len(t))))
        provider=AnalyticFields(lambda q:np.column_stack((q[:,2],np.hypot(np.linalg.norm(q[:,:2],axis=1)-.4,q[:,2]),np.ones(len(q)))))
        vertices=[VertexNode(i,np.array(p),0.,.01) for i,p in enumerate([[.4,0,0],[0,.4,0]])]
        curve, _ = fit_edge_distances(0, points, provider.evaluate(points).values[:, 1],
                                      np.asarray([v.position for v in vertices]), False, provider, Config())
        self.assertEqual(curve.kind,'circle')
        self.assertAlmostEqual(abs(curve.data['sweep']),np.pi/2,places=3)
        np.testing.assert_allclose(curve.evaluate(np.array([0.,1.])),[v.position for v in vertices],atol=1e-10)

    def test_complex_open_curve_adds_local_spans_within_one_edge(self):
        x=np.linspace(-.4,.4,50001)
        dense=np.column_stack((x,.08*np.sin(4*np.pi*(x+.4)/.8),x*0));tree=cKDTree(dense)
        t=np.linspace(-.399,.399,1000)
        y=.08*np.sin(4*np.pi*(t+.4)/.8)
        slope=.08*4*np.pi/.8*np.cos(4*np.pi*(t+.4)/.8)
        normal=np.column_stack((-slope,np.ones(len(t)),t*0));normal/=np.linalg.norm(normal,axis=1)[:,None]
        points=np.column_stack((t,y,t*0))+(.0015*np.cos(np.arange(len(t))*2.4))[:,None]*normal
        provider=AnalyticFields(lambda q:np.column_stack((q[:,2],tree.query(q)[0],np.ones(len(q)))))
        vertices=[VertexNode(i,p,0.,.01) for i,p in enumerate(dense[[0,-1]])]
        curve, _ = fit_edge_distances(0, points, provider.evaluate(points).values[:, 1], dense[[0,-1]], False,
                                      provider, Config(max_segments=12,curve_distance_iterations=30))
        self.assertGreater(len(curve.data['controls']),1)
        self.assertLessEqual(len(curve.data['controls']),12)
        self.assertLessEqual(curve.residual,.001)
        np.testing.assert_allclose(curve.evaluate(np.array([0.,1.])),dense[[0,-1]],atol=1e-10)

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
        with patch('bandbrep.curve_distance_fitting._circle_candidate', return_value=None):
            curve, _ = fit_edge_distances(0, points, provider.evaluate(points).values[:, 1],
                                          np.empty((0, 3)), True, provider,
                Config(max_segments=8, curve_init_samples=128, curve_fit_samples=256,
                       curve_refine_samples=512, curve_distance_iterations=30))
        self.assertEqual(curve.kind,'bezier_spline')
        np.testing.assert_allclose(curve.evaluate(np.array([0.])),curve.evaluate(np.array([1.])),atol=1e-10)
        first=curve_derivatives(curve,np.array([0.,1.]))[0]
        np.testing.assert_allclose(first[0],first[1],atol=1e-9)

    def test_config_rejects_invalid_sample_budgets(self):
        for kwargs in [dict(curve_init_samples=0),
                       dict(curve_fit_samples=128),dict(curve_distance_iterations=True)]:
            with self.assertRaises(ValueError):Config(**kwargs).validate()

if __name__=='__main__':unittest.main()
