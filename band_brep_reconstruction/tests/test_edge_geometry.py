import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree
from bandbrep.config import Config
from bandbrep.schemas import VertexNode
from bandbrep.field_provider import CallableFieldProvider
from bandbrep.edge_geometry import build_edge_geometry
from bandbrep.curve_fitting import fit_shared_curves


class EdgeGeometryTests(unittest.TestCase):
    def test_general_open_support_fixed_endpoints(self):
        t=np.linspace(-.4,.4,8001)
        dense=np.column_stack((t,.8*t*t,t*0));tree=cKDTree(dense)
        provider=CallableFieldProvider(lambda x:np.column_stack((x[:,2],tree.query(x)[0],np.minimum(np.linalg.norm(x-dense[0],axis=1),np.linalg.norm(x-dense[-1],axis=1)))),presence=('present','present'))
        points=dense[::40]
        vertices=[VertexNode(i,p,0.,.01) for i,p in enumerate([dense[0],dense[-1]])]
        topology=SimpleNamespace(face_count=2,edges=[dict(id=0,faces=[0,1],vertices=[0,1],nodes=list(range(len(points))))])
        graph=build_edge_geometry(topology,SimpleNamespace(points=points),vertices,provider,Config())
        curves=fit_shared_curves(graph,provider,Config());ends=curves[0].evaluate(np.array([0.,1.]))
        np.testing.assert_allclose(ends,[dense[0],dense[-1]],atol=1e-10)
        self.assertEqual(graph.edges[0].vertices,[0,1])
    def test_self_loop_endpoint_uses_are_deferred_until_geometry(self):
        t=np.linspace(0,2*np.pi,240,endpoint=False);points=np.column_stack((.4*np.cos(t),.4*np.sin(t),t*0));anchor=points[0]
        provider=CallableFieldProvider(lambda x:np.column_stack((x[:,2],np.hypot(np.linalg.norm(x[:,:2],axis=1)-.4,x[:,2]),np.linalg.norm(x-anchor,axis=1))),presence=('present','present'))
        topology=SimpleNamespace(face_count=2,edges=[dict(id=0,faces=[0,1],vertices=[0],nodes=list(range(len(points))))])
        graph=build_edge_geometry(topology,SimpleNamespace(points=points),[VertexNode(0,anchor,0.,.01)],provider,Config())
        self.assertEqual(topology.edges[0]['vertices'],[0])
        self.assertEqual(graph.edges[0].vertices,[0,0])
        curves=fit_shared_curves(graph,provider,Config())
        np.testing.assert_allclose(curves[0].evaluate(np.array([0.,1.])),[anchor,anchor],atol=1e-10)

if __name__=='__main__':unittest.main()
