import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

@unittest.skipUnless(importlib.util.find_spec('torch'),'optional torch missing')
class NeuralTests(unittest.TestCase):
    def test_training_checkpoint_and_metric_autograd(self):
        import torch
        from bandbrep.neural_fields import fit_fields,NeuralFieldProvider
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);xyz=np.random.default_rng(2).uniform(-.5,.5,(160,3)).astype(np.float32)
            v=np.column_stack((xyz[:,0],np.abs(xyz[:,1]),np.abs(xyz[:,2])))
            (p/'metadata.json').write_text(json.dumps({'channel_names':['sdf','dE','dV']}))
            np.savez(p/'fields.npz',query_points=np.vstack((xyz,xyz)),query_fields=np.vstack((v,v)),has_edges=True,has_vertices=True)
            report=fit_fields(p,p/'model.pt',steps=2,batch_size=32,width=16,layers=2,frequencies=1)
            self.assertEqual(report['unique_positions'],160)
            provider=NeuralFieldProvider(p/'model.pt');out=provider.evaluate(xyz[:3]);self.assertEqual(out.values.shape,(3,3));self.assertTrue((out.values[:,1:]>=0).all())
            gradient,valid=provider.gradient(xyz[:3],1);self.assertEqual(gradient.shape,(3,3));self.assertTrue(valid.all())
    def test_face_adapter_calls_only_face_decoder(self):
        import torch
        from bandbrep.model_adapter import DualBrepFaceInitializer
        class Model:
            def encode(self,data,test):return data
            def decode_face(self,data,uv):
                return {'face_norm':torch.zeros((1,4,4,3)),'face_bbox':data['face_input_bbox']}
            def inference(self,*args):raise AssertionError('edge inference must not be used')
        adapter=object.__new__(DualBrepFaceInitializer);adapter.device='cpu';adapter.model=Model()
        p=np.random.default_rng(1).random((20,3));n=np.tile([0.,0.,1.],(20,1))
        grids=adapter.predict({0:(p,n)});self.assertEqual(grids[0].shape,(4,4,3))
