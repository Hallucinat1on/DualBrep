import unittest
import numpy as np
from bandbrep.config import Config
from bandbrep.curve_fitting import fit_fixed_arc, CurveRecord
from bandbrep.field_provider import CallableFieldProvider
from bandbrep.vertex_domains import local_vertex_domains
from bandbrep.surface_fitting import revolution_candidates, frame
from bandbrep.occ_assembly import shift_periods


class ImprovementTests(unittest.TestCase):
    def test_fixed_arc_reproduces_shared_endpoints(self):
        c=np.array([.1,-.2,.3]);r=.4;a=np.linspace(.2,1.4,40)
        x,y=frame(np.array([0.,1.,1.]))
        pts=c+r*(np.cos(a)[:,None]*x+np.sin(a)[:,None]*y)
        data,err=fit_fixed_arc(pts[1:-1],pts[0],pts[-1])
        self.assertLess(err,1e-9)
        ends=CurveRecord(0,'circle',data,err).evaluate(np.array([0.,1.]))
        np.testing.assert_allclose(ends,pts[[0,-1]],atol=1e-12)

    def test_close_vertices_keep_separate_domains(self):
        v=np.array([[0.,0.,0.],[.02,0.,0.]])
        fn=lambda p:np.column_stack((np.zeros(len(p)),np.abs(p[:,1])+np.abs(p[:,2]),
                                     np.linalg.norm(p[:,None]-v[None],axis=2).min(axis=1)))
        provider=CallableFieldProvider(fn,('present','present'))
        x=np.linspace(-.03,.05,161);points=np.column_stack((x,np.zeros_like(x),np.zeros_like(x)))
        values=fn(points);band=np.ones(len(points),bool)
        labels,centers,report=local_vertex_domains(points,values,band,band,provider,Config(tau_vertex=.02))
        self.assertEqual(report['domains'],2)
        self.assertEqual(len(centers),2)
        # Nodes midway between the two vertices belong to neither domain: an edge remains.
        mid=np.abs(x-.01)<.0005
        self.assertTrue(np.all(labels[mid]==-1))
        self.assertEqual(len(set(labels[labels>=0].tolist())),2)

    def test_cone_is_recovered_from_points_and_normals(self):
        rng=np.random.default_rng(0);axis=np.array([.2,.1,1.]);axis/=np.linalg.norm(axis);x,y=frame(axis)
        o=np.array([.1,.0,-.1]);r0=.3;k=.4
        u=rng.uniform(0,np.pi,400);z=rng.uniform(-.1,.2,400);rho=r0+k*z
        radial=np.cos(u)[:,None]*x+np.sin(u)[:,None]*y
        pts=o+rho[:,None]*radial+z[:,None]*axis
        nrm=radial-k*axis[None];nrm/=np.linalg.norm(nrm,axis=1)[:,None]
        out={c.kind:c for c in revolution_candidates(pts,nrm,[],np.empty((0,3)))}
        self.assertIn('cone',out)
        cone=out['cone'];err=np.linalg.norm(cone.evaluate(cone.uv(pts))-pts,axis=1).max()
        self.assertLess(err,1e-6)

    def test_period_shift_keeps_curve_in_one_interval(self):
        # Planning puts the cut inside a boundary gap, so pcurves never cross it.
        uv=np.column_stack((np.linspace(3.5,5.0,10),np.zeros(10)))
        shifted=shift_periods(uv.copy(),np.array([-np.pi]),'cylinder')
        self.assertTrue(np.all((shifted[:,0]>=-np.pi)&(shifted[:,0]<np.pi)))
        np.testing.assert_allclose(shifted[:,0],uv[:,0]-2*np.pi)

    def test_new_options_validate(self):
        for values in ({'vertex_domains':'ball'},{'surface_candidates':'all'},{'curve_candidates':'nurbs'},
                       {'infer_short_edges':1},{'periodic_frame_planning':'yes'},{'vertex_radius_fraction':.6}):
            with self.assertRaises(ValueError):Config(**values).validate()
        Config(vertex_domains='local',infer_short_edges=True,surface_candidates='revolution',
               curve_candidates='arcs',periodic_frame_planning=True).validate()


if __name__=='__main__':unittest.main()
