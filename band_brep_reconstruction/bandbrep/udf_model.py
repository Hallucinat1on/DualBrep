"""A frozen TorchScript decoder taking [N,3] and returning metric [N,2]."""
import hashlib
from pathlib import Path
import numpy as np
from .field_provider import CallableUDFProvider
from .schemas import require


class TorchScriptUDFProvider(CallableUDFProvider):
    def __init__(self,path,device='cpu',band_width=.02,presence=None):
        import torch
        self.model=torch.jit.load(str(path),map_location=device).eval()
        self.device=device
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        def values(q):
            with torch.no_grad():
                result=self.model(torch.as_tensor(q,dtype=torch.float32,device=device))
            require(result.shape==(len(q),2),'field_shape','Decoder must return metric [N,2] E/V')
            return result.cpu().numpy()
        def gradient(q,channel):
            with torch.enable_grad():
                x=torch.tensor(q,dtype=torch.float32,device=device,requires_grad=True)
                y=self.model(x)
                if not y.requires_grad:return np.zeros_like(q),np.zeros(len(q),bool)
                g=torch.autograd.grad(y[:,channel].sum(),x,allow_unused=True)[0]
            if g is None:return np.zeros_like(q),np.zeros(len(q),bool)
            g=g.detach().cpu().numpy()
            return g,np.isfinite(g).all(1)
        digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        super().__init__(values,fingerprint='torchscript:'+digest+':'+device,
                         presence=presence,gradient=gradient,band_width=band_width)
