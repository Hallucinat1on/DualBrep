import unittest
import numpy as np


class SurfaceParameterizationTests(unittest.TestCase):
    def test_spline_uv_batched_rhs(self):
        from bandbrep.surface_fitting import FacePatch
        knots=np.r_[np.zeros(4),np.ones(4)]
        u,v=np.meshgrid(np.linspace(0,1,4),np.linspace(0,1,4),indexing='ij')
        controls=np.stack((u,v,np.zeros_like(u)),axis=-1)
        patch=FacePatch(0,'bspline',{'origin':np.zeros(3),'x':np.array([1.,0.,0.]),
            'y':np.array([0.,1.,0.]),'uv_lo':np.zeros(2),'uv_extent':np.ones(2),
            'knots':knots,'degree':3,'controls':controls},np.array([0.,0.,1.]),{})
        points=np.array([[.1,.2,0.],[.3,.4,0.],[.7,.8,0.]])
        np.testing.assert_allclose(patch.uv(points),points[:,:2],atol=1e-8)
