import ast
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from bandbrep.vendor.dualbrep_segmentation import hierarchical_segmentation
from bandbrep.bands import build_bands
from bandbrep.config import Config, load_config
from bandbrep.schemas import SurfaceMesh
from helpers import AnalyticFields
from bandbrep.topology import recover_topology


def chain(n):
    return SimpleNamespace(face_adjacency=np.column_stack((np.arange(n-1),np.arange(1,n))))


class TwoPassTests(unittest.TestCase):
    def test_first_pass_only_keeps_parent_without_building_split_graph(self):
        d=np.array([.02,.02,.007,.02,.02])
        with patch('bandbrep.vendor.dualbrep_segmentation.connected_components',
                   wraps=connected_components) as components:
            n,labels,reasons,report=hierarchical_segmentation(chain(5),d,.005,.01,2,second_pass=False)
        self.assertEqual(components.call_count,1)
        self.assertEqual(n,1)
        np.testing.assert_array_equal(labels,np.zeros(5))
        np.testing.assert_array_equal(reasons,np.zeros(5))
        self.assertFalse(report['second_pass_enabled'])
        self.assertEqual(report['split_parents'],[])

    def test_first_pass_only_filters_and_relabels_sparse_components(self):
        d=np.array([.02,.001,.006,.006,.001,.02,.02,.001,.02,.02])
        n,labels,reasons,report=hierarchical_segmentation(chain(10),d,.005,.01,2,.01,second_pass=False)
        self.assertEqual(n,2)
        np.testing.assert_array_equal(labels,[-1,-1,-1,-1,-1,0,0,-1,1,1])
        np.testing.assert_array_equal(reasons,[3,1,4,4,1,0,0,1,0,0])
        self.assertEqual((report['pass1_components'],report['pass1_valid']),(4,2))
        for d in (np.array([.001,.001]),np.array([.02,.001])):
            n,labels,_,report=hierarchical_segmentation(chain(2),d,.005,.01,2,second_pass=False)
            self.assertEqual(n,0);np.testing.assert_array_equal(labels,[-1,-1])
            self.assertFalse(report['second_pass_enabled'])

    def test_band_masks_follow_selected_segmentation_passes(self):
        x=np.arange(7,dtype=float)
        mesh=SurfaceMesh(np.column_stack((x,x*x,np.zeros(7))),
                         np.array([[0,i+1,i+2] for i in range(5)]))
        d=np.array([.02,.02,.007,.02,.02])
        provider=AnalyticFields(lambda q:np.column_stack((np.zeros(5),d,np.ones(5))),
                                presence=('present','absent'))
        for enabled,labels,edges,method in (
                (True,[0,0,-1,1,1],[False,False,True,False,False],'dualbrep_two_pass_triangle_components'),
                (False,[0,0,0,0,0],[False]*5,'dualbrep_first_pass_triangle_components')):
            _,bands=build_bands(mesh,provider,Config(face_filter_size=2,face_second_pass=enabled))
            np.testing.assert_array_equal(bands.face_labels,labels)
            np.testing.assert_array_equal(bands.edge_band,edges)
            np.testing.assert_array_equal(bands.core_mask,np.array(labels)>=0)
            self.assertEqual(bands.report['segmentation']['method'],method)
            self.assertEqual(bands.report['segmentation']['second_pass_enabled'],enabled)

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
        p=AnalyticFields(lambda x:np.tile([0.,.008,0.],(len(x),1)),presence=('present','present'))
        _,b=build_bands(mesh,p,Config(band_propagation="distance_only"))
        self.assertEqual(b.exclusion_reasons.tolist(),[3])
        self.assertFalse((b.core_mask|b.edge_band|b.vertex_band).any())
        t=recover_topology(mesh,b,b.face_labels)
        self.assertEqual(t.edges,[]);self.assertEqual(t.vertices,[])
        # When accepted, this same point is core, never in the old .025 edge band.
        _,b=build_bands(mesh,p,Config(face_filter_size=1))
        self.assertTrue(b.core_mask.all());self.assertFalse(b.edge_band.any())

    def test_config_validation(self):
        self.assertTrue(Config().face_second_pass)
        for kw in ({'face_filter_size':0},{'face_threshold1':.02,'face_threshold2':.01},
                   {'face_threshold1':float('nan')}):
            with self.assertRaises(ValueError):Config(**kw).validate()
        for value in (0,1,'false',None):
            with self.assertRaises(ValueError):Config(face_second_pass=value).validate()

    def test_cli_second_pass_overrides_config(self):
        from bandbrep.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'config.json'
            for stored,flag,expected in ((True,[],True),(False,[],False),
                    (True,['--no-face-second-pass'],False),(False,['--face-second-pass'],True)):
                config.write_text(json.dumps({'face_second_pass':stored}))
                self.assertEqual(load_config(config).face_second_pass,stored)
                argv=['reconstruct','--mesh','mesh','--udf-model','model.pt',
                      '--output',tmp,'--config',str(config),*flag]
                with patch('sys.argv',argv), patch('bandbrep.cli.load_mesh'), \
                        patch('bandbrep.udf_model.TorchScriptUDFProvider'), \
                        patch('bandbrep.cli.reconstruct') as run:
                    main()
                self.assertEqual(run.call_args.args[2].face_second_pass,expected)

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
