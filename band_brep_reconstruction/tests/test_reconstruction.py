import unittest
import numpy as np
from bandbrep.face_seeds import core_components
from bandbrep.vertex_localization import distance_center
from bandbrep.curve_fitting import fit_cubic_fixed_ends,fit_circle
from bandbrep.surface_fitting import constrained_lstsq,fit_four_sided_bezier
from bandbrep.schemas import ReconstructionError

class CoreTests(unittest.TestCase):
    def test_isolated_core_preserved(self):
        r=core_components(np.array([[0,1]]),np.array([True,True,False,True]));self.assertEqual(r[0],r[1]);self.assertNotEqual(r[0],r[3]);self.assertEqual(r[2],-1)
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
    def test_face_scoring_handles_label_permutation(self):
        from bandbrep.evaluation import match_face_regions
        result=match_face_regions([1,1,0,0],[0,0,1,1]);self.assertEqual(result['mapping'],{0:1,1:0});self.assertEqual(result['merged_predictions'],[])
