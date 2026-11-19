import unittest
import numpy as np
from bandbrep.vertex_localization import distance_center
from bandbrep.curve_fitting import fit_circle
from bandbrep.surface_fitting import constrained_lstsq
from bandbrep.schemas import ReconstructionError

class CoreTests(unittest.TestCase):
    def test_distance_center_not_centroid(self):
        v=np.array([.2,-.1,.3]);p=np.random.default_rng(1).uniform(0,.2,(100,3))+v;r,e,rank=distance_center(p,np.linalg.norm(p-v,axis=1))
        np.testing.assert_allclose(r,v,atol=1e-8);self.assertEqual(rank,3);self.assertGreater(np.linalg.norm(p.mean(axis=0)-v),.1)
    def test_rank_deficient_vertex(self):
        with self.assertRaises(ReconstructionError):distance_center(np.array([[0.,0,0],[1,0,0],[0,1,0]]),[1,1,1])
    def test_circle_not_polynomial_bezier(self):
        t=np.linspace(0,2*np.pi,120);p=np.column_stack((np.cos(t),np.sin(t),t*0));d,e=fit_circle(p);self.assertLess(e,1e-12);self.assertAlmostEqual(d['radius'],1.)
    def test_hard_boundary_constraints(self):
        B=np.array([[1.,1,0]]);C=np.array([[1.,2,3]]);P=constrained_lstsq(np.eye(3),np.zeros((3,3)),B,C);np.testing.assert_allclose(B@P,C,atol=1e-10)
    def test_conflicting_boundary_constraints(self):
        with self.assertRaises(ReconstructionError):constrained_lstsq(np.eye(2),np.zeros((2,3)),np.array([[1,0],[1,0]]),np.array([[0.,0,0],[1,0,0]]))
