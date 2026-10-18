import unittest
import numpy as np
from bandbrep.config import Config
from bandbrep.schemas import BandState, SurfaceMesh
from bandbrep.surface_mesh import face_adjacency, vertex_adjacency
from bandbrep.topology import collect_labels, recover_topology


class ExperimentOptionsTests(unittest.TestCase):
    def test_corner_connectivity_is_opt_in(self):
        tri=np.array([[0,1,2],[2,3,4]])
        self.assertEqual(len(face_adjacency(tri)),0)
        self.assertEqual(vertex_adjacency(tri).tolist(),[[0,1]])
        b=BandState(np.zeros((2,3)),np.zeros((2,3)),np.ones(2,bool),np.zeros(2,bool),np.ones(2,bool),np.zeros(2,bool))
        m=SurfaceMesh(np.zeros((5,3)),tri)
        self.assertEqual(len(recover_topology(m,b,np.full(2,-1)).edges),2)
        self.assertEqual(len(recover_topology(m,b,np.full(2,-1),Config(band_connectivity='vertex')).edges),1)

    def test_exclusion_transfers_labels_but_not_geometry_support(self):
        labels=np.array([0,-1,-1,-1,1])
        band=labels<0;vertex=np.zeros(5,bool)
        b=BandState(np.zeros((5,3)),np.zeros((5,3)),band,vertex,band,~band,
            face_labels=labels,exclusion_reasons=np.array([0,1,3,1,0]))
        t=collect_labels(np.array([[0,1],[1,2],[2,3],[3,4]]),b,labels)
        self.assertEqual(t.edges[0]['faces'],[0,1])
        self.assertEqual(t.edges[0]['nodes'],[1,2,3])
        self.assertEqual(t.edges[0]['support_nodes'],[1,3])
        self.assertEqual(t.node_labels(2),({0,1},{0}))

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

    def test_empty_gt_label_bin_is_not_a_recovered_face(self):
        import sys
        from pathlib import Path
        from types import SimpleNamespace
        from bandbrep.schemas import BandTopology
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
        from experiment_scoring import score
        labels=np.array([1,1]);inactive=np.full(2,-1)
        topo=BandTopology(2,labels,inactive,inactive,[],[],np.empty((0,2),int))
        bands=BandState(np.zeros((2,3)),np.zeros((2,3)),np.zeros(2,bool),np.zeros(2,bool),np.zeros(2,bool),np.ones(2,bool))
        source=SimpleNamespace(targets=SimpleNamespace(faces=[0,1],edges=[],vertices=np.empty((0,3)),topology={'edges':[],'vertices':[]}),
            edge=None,vtree=None,truth_face=lambda points:np.ones(len(points),int))
        result=score(topo,bands,source)
        self.assertFalse(result['face_good'])
        self.assertEqual(result['missing_faces'],1)

    def test_invalid_options(self):
        for values in ({'band_connectivity':'knn'},{'band_propagation':'winner'}):
            with self.assertRaises(ValueError):Config(**values).validate()


if __name__=='__main__':unittest.main()
