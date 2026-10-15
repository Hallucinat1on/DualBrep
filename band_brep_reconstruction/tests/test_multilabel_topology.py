import unittest
from collections import deque
import numpy as np
from bandbrep.schemas import BandState, SurfaceMesh
from bandbrep.bands import build_bands
from bandbrep.config import Config,load_config
from bandbrep.face_seeds import extract_face_seeds
from bandbrep.field_provider import CallableFieldProvider
from bandbrep.topology import collect_labels


def masks(n, regular, vertex):
    r=np.zeros(n,bool);v=r.copy();r[regular]=True;v[vertex]=True
    return BandState(np.zeros((n,3)),np.zeros((n,3)),r|v,v,r,~(r|v))


def absorbing_flood(pairs,bands,faces,source_order):
    """Independent per-message reference; late labels must traverse V again."""
    adjacency=[[] for _ in faces]
    for a,b in pairs:adjacency[a].append(b);adjacency[b].append(a)
    result=[set() for _ in faces];q=deque()
    for n in source_order:
        if faces[n]>=0:q.append((n,int(faces[n])))
    while q:
        node,label=q.popleft()
        if label in result[node]:continue
        result[node].add(label)
        for target in adjacency[node]:
            if bands.core_mask[target]:continue
            if bands.vertex_band[node] and not bands.vertex_band[target]:continue
            q.append((target,label))
    return result


class MultilabelTests(unittest.TestCase):
    def test_vertex_fills_without_relay_and_late_sources(self):
        # AB and BC strips join only through a three-node vertex sink; D touches V directly.
        pairs=np.array([[0,3],[1,4],[3,4],[4,7],[7,8],[8,9],[9,5],[5,6],[6,2],[1,5],[10,8]])
        b=masks(11,[3,4,5,6],[7,8,9]);f=np.array([0,1,2,-1,-1,-1,-1,-1,-1,-1,3])
        t=collect_labels(pairs,b,f)
        self.assertEqual(t.edges[0]['faces'],[0,1]);self.assertEqual(t.edges[1]['faces'],[1,2])
        self.assertEqual(t.vertices[0]['faces'],[0,1,2,3]);self.assertEqual(t.vertices[0]['edges'],[0,1])
        for n in [7,8,9]:self.assertEqual(t.node_labels(n),(set(range(4)),{0,1}))
        for order in [range(11),reversed(range(11))]:
            expected=absorbing_flood(pairs,b,f,order)
            self.assertEqual([t.node_labels(n)[0] for n in range(11)],expected)
    def test_same_neighbor_faces_keep_disconnected_edges(self):
        pairs=np.array([[0,2],[2,3],[3,1],[0,4],[4,5],[5,1]])
        t=collect_labels(pairs,masks(6,[2,3,4,5],[]),np.array([0,1,-1,-1,-1,-1]))
        self.assertEqual(len(t.edges),2);self.assertEqual(t.edges[0]['faces'],t.edges[1]['faces'])
        self.assertEqual(t.vertices,[]);self.assertEqual(t.edges[0]['vertices'],[])
    def test_vertex_edges_use_actual_ids_not_face_subset(self):
        pairs=np.array([[0,2],[2,3],[3,1],[0,4],[4,5],[5,1],[3,6]])
        t=collect_labels(pairs,masks(7,[2,3,4,5],[6]),np.array([0,1,-1,-1,-1,-1,-1]))
        self.assertEqual(t.vertices[0]['faces'],[0,1]);self.assertEqual(t.vertices[0]['edges'],[0])
        self.assertEqual(t.edges[1]['vertices'],[])
    def test_arbitrary_label_count_is_preserved(self):
        pairs=np.array([[0,3],[1,3],[2,3],[3,4]])
        t=collect_labels(pairs,masks(5,[3],[4]),np.array([0,1,2,-1,-1]))
        self.assertEqual(t.edges[0]['faces'],[0,1,2]);self.assertEqual(t.edges[0]['status'],'unresolved')
        self.assertEqual(t.node_labels(4),({0,1,2},{0}))
    def test_nested_vertex_mask_and_triangle_center_core(self):
        mesh=SurfaceMesh(np.array([[0.,0,0],[1,0,0],[0,1,0]]),np.array([[0,1,2]]))
        p=CallableFieldProvider(lambda x:np.column_stack((x[:,2],np.full(len(x),.1),np.zeros(len(x)))),presence=('present','present'))
        _,b=build_bands(mesh,p,Config(face_filter_size=1));self.assertFalse(b.vertex_band.any())
        labels,_=extract_face_seeds(mesh,b,p,Config());np.testing.assert_array_equal(labels,[0])
        self.assertEqual(b.points.shape,(1,3));np.testing.assert_allclose(b.points,[[1/3,1/3,0]])
    def test_removed_competition_config_rejected(self):
        with self.assertRaises(ValueError):load_config(overrides={'band_refinement':'legacy'})

if __name__=='__main__':unittest.main()
