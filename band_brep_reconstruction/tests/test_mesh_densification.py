import itertools
import unittest
from unittest.mock import Mock
from bandbrep.deadline import sample_budget
import time
import numpy as np
from bandbrep.mesh_densification import split_edges, mesh_metrics, coarse_refine_surface, marked_long_edges
from bandbrep.schemas import SurfaceMesh, ReconstructionError
from bandbrep.surface_mesh import mesh_edges


def tetrahedron():
    return SurfaceMesh(np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]]),
                       np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]]))


class DensificationTests(unittest.TestCase):
    def test_all_marked_edge_patterns_are_closed_and_preserve_geometry(self):
        mesh=tetrahedron();before=mesh_metrics(mesh)
        for marks in itertools.product([False, True], repeat=6):
            refined,parent,_=split_edges(mesh,np.array(marks),mesh_edges(mesh.triangles))
            after=mesh_metrics(refined)
            self.assertEqual(after['components'],before['components'])
            self.assertEqual(after['euler'],before['euler'])
            self.assertAlmostEqual(after['area'],before['area'])
            self.assertAlmostEqual(after['volume'],before['volume'])
            self.assertEqual(len(parent),len(refined.triangles))
            for i in range(4):
                selected=refined.vertices[refined.triangles[parent==i]]
                base=mesh.vertices[mesh.triangles[i]]
                normal=np.cross(base[1]-base[0],base[2]-base[0])
                np.testing.assert_allclose((selected-base[0])@normal,0,atol=1e-14)

    def test_uniform_shared_midpoints_and_parent_multiplicity(self):
        mesh=tetrahedron();provider=Mock()
        refined,parent,report=split_edges(mesh,np.ones(6,bool),mesh_edges(mesh.triangles))
        self.assertEqual(len(refined.vertices),10)
        self.assertEqual(len(refined.triangles),16)
        np.testing.assert_array_equal(np.bincount(parent),[4,4,4,4])
        provider.evaluate.assert_not_called()

    def test_expired_deadline_stops_before_split(self):
        with self.assertRaises(ReconstructionError) as error:
            with sample_budget(.001):
                time.sleep(.002)
                split_edges(tetrahedron(),np.ones(6,bool),mesh_edges(tetrahedron().triangles))
        self.assertEqual(error.exception.code,'sample_timeout')

    def test_coarse_geometry_only_target_and_parent_mapping(self):
        source=tetrahedron()
        mesh,parent,report=coarse_refine_surface(source,.2)
        self.assertTrue(report['target_reached'])
        self.assertLessEqual(report['refined']['edge_quantiles'][-1],.2*(1+1e-10))
        self.assertEqual(report['refined']['euler'],report['original']['euler'])
        for i in range(4):
            base=source.vertices[source.triangles[i]]
            normal=np.cross(base[1]-base[0],base[2]-base[0])
            np.testing.assert_allclose((mesh.vertices[mesh.triangles[parent==i]]-base[0])@normal,0,atol=1e-14)

    def test_short_edges_are_never_marked(self):
        mesh=tetrahedron();mesh.vertices*=np.array([1.,.01,.01])
        index=mesh_edges(mesh.triangles)
        selected=np.ones(len(mesh.triangles),bool)
        marked=marked_long_edges(mesh,index,selected,.1)
        lengths=np.linalg.norm(mesh.vertices[index[0][:,0]]-mesh.vertices[index[0][:,1]],axis=1)
        np.testing.assert_array_equal(marked,lengths>.1*(1+1e-10))
        self.assertTrue((lengths<=.1).any());self.assertTrue(marked.any())
        final,_,_=coarse_refine_surface(mesh,.1)
        self.assertLessEqual(mesh_metrics(final)['edge_quantiles'][-1],.1*(1+1e-10))

    def test_packed_edge_index_matches_canonical_pairs(self):
        rng=np.random.default_rng(19)
        for triangles in (tetrahedron().triangles,rng.integers(0,100,size=(1000,3)),np.empty((0,3),np.int64)):
            directed=triangles[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
            expected=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
            got=mesh_edges(triangles)
            np.testing.assert_array_equal(got[0],expected[0])
            np.testing.assert_array_equal(got[1].ravel(),expected[1])
            np.testing.assert_array_equal(got[2],expected[2])
