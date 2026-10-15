import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from bandbrep.vendor.dualbrep_segmentation import hierarchical_segmentation
from bandbrep.bands import build_bands
from bandbrep.config import Config
from bandbrep.schemas import SurfaceMesh
from bandbrep.field_provider import CallableFieldProvider
from bandbrep.topology import recover_topology


def chain(n):
    return SimpleNamespace(face_adjacency=np.column_stack((np.arange(n-1),np.arange(1,n))))


class TwoPassTests(unittest.TestCase):
    def test_split_and_keep_parent_semantics(self):
        mesh=chain(5)
        _,labels,reasons,report=hierarchical_segmentation(mesh,np.array([.02,.02,.007,.02,.02]),.005,.01,2)
        np.testing.assert_array_equal(labels,[0,0,-1,1,1])
        np.testing.assert_array_equal(reasons,[0,0,2,0,0])
        self.assertEqual(report['split_parents'],[0])
        # A single valid high-distance child does not replace the parent.
        _,labels,reasons,_=hierarchical_segmentation(mesh,np.array([.02,.02,.007,.006,.006]),.005,.01,2)
        np.testing.assert_array_equal(labels,np.zeros(5))
        np.testing.assert_array_equal(reasons,np.zeros(5))

    def test_rejected_subclusters_are_not_edges(self):
        d=np.array([.02,.02,.007,.02,.007,.02,.02])
        _,labels,reasons,_=hierarchical_segmentation(chain(7),d,.005,.01,2)
        np.testing.assert_array_equal(labels,[0,0,-1,-1,-1,1,1])
        np.testing.assert_array_equal(reasons,[0,0,2,5,2,0,0])

    def test_empty_and_mean_filtered(self):
        n,l,r,_=hierarchical_segmentation(chain(2),np.array([.001,.001]),.005,.01,1)
        self.assertEqual(n,0);np.testing.assert_array_equal(r,[1,1])
        n,l,r,_=hierarchical_segmentation(chain(2),np.array([.006,.006]),.005,.01,1,.007)
        self.assertEqual(n,0);np.testing.assert_array_equal(r,[4,4])

    def test_band_partition_and_rejected_nodes_do_not_propagate(self):
        # Disconnected single triangles are size-rejected, not invented edges.
        mesh=SurfaceMesh(np.array([[0.,0,0],[1,0,0],[0,1,0]]),np.array([[0,1,2]]))
        p=CallableFieldProvider(lambda x:np.tile([0.,.008,0.],(len(x),1)),presence=('present','present'))
        _,b=build_bands(mesh,p,Config())
        self.assertEqual(b.exclusion_reasons.tolist(),[3])
        self.assertFalse((b.core_mask|b.edge_band|b.vertex_band).any())
        t=recover_topology(mesh,b,b.face_labels)
        self.assertEqual(t.edges,[]);self.assertEqual(t.vertices,[])
        # When accepted, this same point is core, never in the old .025 edge band.
        _,b=build_bands(mesh,p,Config(face_filter_size=1))
        self.assertTrue(b.core_mask.all());self.assertFalse(b.edge_band.any())

    def test_config_validation(self):
        for kw in ({'face_filter_size':0},{'face_threshold1':.02,'face_threshold2':.01},
                   {'face_threshold1':float('nan')}):
            with self.assertRaises(ValueError):Config(**kw).validate()

    def test_labels_match_original_dualbrep_function(self):
        # Optional repository-only reference; standalone installed module has no dependency.
        source=Path(__file__).resolve().parents[2]/'clustering.py'
        if not source.exists():self.skipTest('Original DualBrep source not present in portable copy')
        tree=ast.parse(source.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='hierarchical_segmentation')
        env=dict(np=np,csr_matrix=csr_matrix,connected_components=connected_components,defaultdict=defaultdict)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
        rng=np.random.default_rng(20261015)
        for _ in range(100):
            n=50;pairs=np.column_stack((rng.integers(n,size=80),rng.integers(n,size=80)))
            mesh=SimpleNamespace(face_adjacency=pairs);d=rng.choice([.001,.004,.006,.009,.02],n)
            kwargs=dict(threshold1=.005,threshold2=.01,filter_size=int(rng.integers(1,5)),avg_udf_threshold=.003)
            expected=env['hierarchical_segmentation'](mesh,d,**kwargs)
            actual=hierarchical_segmentation(mesh,d,**kwargs)
            self.assertEqual(actual[0],expected[0]);np.testing.assert_array_equal(actual[1],expected[1])
            np.testing.assert_array_equal(actual[1]>=0,actual[2]==0)
