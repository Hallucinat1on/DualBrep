import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np

@unittest.skipUnless(importlib.util.find_spec('torch'),'optional torch missing')
class NeuralTests(unittest.TestCase):
    def test_two_channel_metric_script_and_autograd(self):
        import torch
        from bandbrep.udf_model import TorchScriptUDFProvider
        class Decoder(torch.nn.Module):
            def forward(self,x):return torch.stack((2*x[:,0].abs(),3*x[:,1].abs()),dim=1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'udf.pt';torch.jit.script(Decoder()).save(str(path))
            p=TorchScriptUDFProvider(path,presence=('present','present'))
            q=np.array([[.1,.2,.3]])
            np.testing.assert_allclose(p.evaluate(q).values,[[.2,.6]],atol=1e-7)
            g,valid=p.gradient(q,1);self.assertTrue(valid[0]);np.testing.assert_allclose(g,[[0,3,0]])

    def test_mesh_cli_without_sdf_model(self):
        import torch
        import subprocess,sys,json
        from helpers import hollow_sphere
        from veudf.io.condition import write_condition
        class Empty(torch.nn.Module):
            def forward(self,x):return x[:,:2]*0+3.
        mesh,_=hollow_sphere()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);write_condition(path/'condition',(mesh.vertices,mesh.triangles),count=64)
            torch.jit.script(Empty()).save(str(path/'ev.pt'))
            cmd=[sys.executable,str(Path(__file__).resolve().parents[1]/'reconstruct.py'),
                 '--mesh',str(path/'condition'),'--udf-model',str(path/'ev.pt'),
                 '--output',str(path/'out'),'--stop-after','graph']
            subprocess.run(cmd,check=True,capture_output=True,text=True)
            run=json.loads((path/'out/run.json').read_text())
            self.assertEqual(run['status'],'logical_graph_ready');self.assertEqual(run['presence'],['absent','absent'])
            self.assertTrue((path/'out/working_mesh.npz').exists())
            self.assertFalse((path/'out/sdf_surface.npz').exists())
