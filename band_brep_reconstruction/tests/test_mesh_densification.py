import itertools
import unittest
from dataclasses import replace
from unittest.mock import Mock
import numpy as np
from bandbrep.mesh_densification import (RefinementConfig, split_edges, mesh_metrics,
                                         densify_surface_mesh, local_selection)
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
            refined,parent,_=split_edges(mesh,np.array(marks),100)
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
        refined,parent,report=split_edges(mesh,np.ones(6,bool),1500000)
        self.assertEqual(len(refined.vertices),10)
        self.assertEqual(len(refined.triangles),16)
        np.testing.assert_array_equal(np.bincount(parent),[4,4,4,4])
        provider.evaluate.assert_not_called()

    def test_budget_rejects_before_allocating_children(self):
        with self.assertRaises(ReconstructionError) as error:
            split_edges(tetrahedron(),np.ones(6,bool),15)
        self.assertEqual(error.exception.code,'mesh_resource_limit')

    def test_local_queries_new_positions_and_reaches_target(self):
        positions=[]
        def values(x):
            positions.append(x.copy())
            return np.column_stack((np.zeros(len(x)),np.zeros(len(x)),np.ones(len(x))))
        provider=AnalyticFields(values,presence=('present','absent'))
        refined,parent,report=densify_surface_mesh(tetrahedron(),provider,
            RefinementConfig(target_edge=.4,max_rounds=3))
        self.assertTrue(any(len(p)>4 for p in positions))
        edges=mesh_edges(refined.triangles)[0]
        self.assertLessEqual(np.linalg.norm(refined.vertices[edges[:,0]]-refined.vertices[edges[:,1]],axis=1).max(),.4)
        self.assertEqual(len(parent),len(refined.triangles))

    def test_target_unmet_does_not_claim_success(self):
        provider=AnalyticFields(lambda x:np.column_stack((np.zeros(len(x)),np.zeros(len(x)),np.ones(len(x)))),
                                       presence=('present','absent'))
        with self.assertRaises(ReconstructionError) as error:
            densify_surface_mesh(tetrahedron(),provider,RefinementConfig(target_edge=.001,max_rounds=1))
        self.assertEqual(error.exception.code,'refinement_target_unmet')

    def test_lipschitz_selector_catches_center_miss_and_absence(self):
        mesh=SurfaceMesh(np.array([[0.,0,0],[1,0,0],[0,1,0]]),np.array([[0,1,2]]))
        vertices=np.tile([0.,.1,.5],(3,1));center=np.array([[0.,.4,.5]])
        selected,_,_=local_selection(mesh,vertices,center,('present','absent'),RefinementConfig())
        self.assertTrue(selected[0])
        selected,_,_=local_selection(mesh,vertices,center,('absent','absent'),RefinementConfig())
        self.assertFalse(selected[0])
