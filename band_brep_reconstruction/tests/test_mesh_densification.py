import itertools
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch
from bandbrep.deadline import sample_budget
import time
import numpy as np
from bandbrep.mesh_densification import (split_edges, mesh_metrics, coarse_refine_surface,
                                         fine_refine_surface, local_selection, marked_long_edges)
from bandbrep.config import Config
from bandbrep.field_provider import CenterFields
from bandbrep.bands import build_bands
from bandbrep.schemas import SurfaceMesh, ReconstructionError
from bandbrep.surface_mesh import mesh_edges
from helpers import AnalyticFields


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

    def test_local_queries_new_positions_and_reaches_target(self):
        positions=[]
        def values(x):
            positions.append(x.copy())
            return np.column_stack((np.zeros(len(x)),np.zeros(len(x)),np.ones(len(x))))
        provider=AnalyticFields(values,presence=('present','absent'))
        refined,parent,report,centers=fine_refine_surface(tetrahedron(),provider,
            Config(refinement_target_edge=.04,refinement_coarse_edge=.1))
        self.assertTrue(any(len(p)>4 for p in positions))
        edges=mesh_edges(refined.triangles)[0]
        self.assertLessEqual(np.linalg.norm(refined.vertices[edges[:,0]]-refined.vertices[edges[:,1]],axis=1).max(),.04)
        self.assertEqual(len(parent),len(refined.triangles))
        self.assertGreater(len(report["history"]),4)

    def test_timeout_does_not_claim_success(self):
        provider=AnalyticFields(lambda x:np.column_stack((np.zeros(len(x)),np.zeros(len(x)),np.ones(len(x)))),
                                       presence=('present','absent'))
        with self.assertRaises(ReconstructionError) as error:
            with sample_budget(.001):
                fine_refine_surface(tetrahedron(),provider,Config(refinement_target_edge=.000001))
        self.assertEqual(error.exception.code,'sample_timeout')

    def test_lipschitz_selector_catches_center_miss_and_absence(self):
        mesh=SurfaceMesh(np.array([[0.,0,0],[1,0,0],[0,1,0]]),np.array([[0,1,2]]))
        vertices=np.tile([0.,.1,.5],(3,1));center=np.array([[0.,.4,.5]])
        selected,_,_=local_selection(mesh,vertices,center,('present','absent'),Config())
        self.assertTrue(selected[0])
        selected,_,_=local_selection(mesh,vertices,center,('absent','absent'),Config())
        self.assertFalse(selected[0])

    def test_coarse_geometry_only_target_and_parent_mapping(self):
        source=tetrahedron()
        with patch('bandbrep.mesh_densification.evaluate_batched', side_effect=AssertionError('field used')):
            mesh,parent,report=coarse_refine_surface(source,.2)
        self.assertTrue(report['target_reached'])
        self.assertLessEqual(report['refined']['edge_quantiles'][-1],.2*(1+1e-10))
        self.assertEqual(report['refined']['euler'],report['original']['euler'])
        provider=AnalyticFields(lambda x:np.column_stack((np.zeros(len(x)),np.zeros(len(x)),np.ones(len(x)))),
                                presence=('present','absent'))
        final,fine_parent,report,fields=fine_refine_surface(mesh,provider,
            Config(refinement_coarse_edge=.2,refinement_target_edge=.13))
        original_parent=parent[fine_parent]
        for i in range(4):
            base=source.vertices[source.triangles[i]]
            normal=np.cross(base[1]-base[0],base[2]-base[0])
            np.testing.assert_allclose((final.vertices[final.triangles[original_parent==i]]-base[0])@normal,0,atol=1e-14)
        np.testing.assert_array_equal(fields.points,final.vertices[final.triangles].mean(1))

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

    def test_center_fields_reuse_rejects_misalignment_and_other_provider(self):
        mesh=tetrahedron();points=mesh.vertices[mesh.triangles].mean(1)
        p=AnalyticFields(lambda x:np.column_stack((np.zeros(len(x)),np.ones(len(x)),np.ones(len(x)))),
                         presence=('absent','absent'))
        fields=CenterFields(points,p.evaluate(points).values,p.fingerprint)
        cfg=Config(vertex_domains='threshold',face_filter_size=1)
        _,direct=build_bands(mesh,p,cfg)
        p.evaluate=Mock(side_effect=AssertionError('requeried centers'))
        _,reused=build_bands(mesh,p,cfg,fields)
        np.testing.assert_array_equal(direct.values,reused.values)
        np.testing.assert_array_equal(direct.face_labels,reused.face_labels)
        self.assertTrue(reused.report['reused_center_queries'])
        self.assertEqual(reused.report['field_query_points'],0)
        with self.assertRaises(ReconstructionError):
            build_bands(mesh,p,cfg,replace(fields,points=points[::-1]))
        with self.assertRaises(ReconstructionError):
            build_bands(mesh,p,cfg,replace(fields,provider_fingerprint='other'))

    def test_packed_edge_index_matches_canonical_pairs(self):
        rng=np.random.default_rng(19)
        for triangles in (tetrahedron().triangles,rng.integers(0,100,size=(1000,3)),np.empty((0,3),np.int64)):
            directed=triangles[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
            expected=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
            got=mesh_edges(triangles)
            np.testing.assert_array_equal(got[0],expected[0])
            np.testing.assert_array_equal(got[1].ravel(),expected[1])
            np.testing.assert_array_equal(got[2],expected[2])
