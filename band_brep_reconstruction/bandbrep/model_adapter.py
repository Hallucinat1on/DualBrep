"""Optional face-only adapter using the COPIED DualBrep network."""
import json
import numpy as np
from .schemas import require

class DualBrepFaceInitializer:
    def __init__(self,checkpoint,model_config,device='cpu'):
        import torch
        import hashlib
        self.fingerprint=hashlib.sha256(open(checkpoint,"rb").read()).hexdigest()
        from .vendor.dualbrep_model import Parametrizer
        config=json.loads(open(model_config).read()) if isinstance(model_config,(str,bytes)) else model_config
        self.model=Parametrizer(config).to(device).eval();self.device=device
        state=torch.load(checkpoint,map_location=device,weights_only=False)
        state=state.get('state_dict',state)
        state={key.removeprefix('model.'):value for key,value in state.items()}
        self.model.load_state_dict(state,strict=True)
    def predict(self,regions):
        import torch
        from .vendor.dualbrep_model import normalize_coord0516,denormalize_coord0516
        keys=sorted(regions);points=np.stack([np.column_stack(regions[key]) for key in keys]).astype(np.float32)
        points=torch.as_tensor(points,device=self.device)
        _,bbox=normalize_coord0516(points)
        data={'face_sample_points':points,'face_input_bbox':bbox,
              'face_attn_mask':torch.zeros((len(keys),len(keys)),dtype=torch.bool,device=self.device)}
        with torch.no_grad():
            encoded=self.model.encode(data,True)
            decoded=self.model.decode_face(encoded,None)
            grids=denormalize_coord0516(decoded['face_norm'],decoded['face_bbox'])[...,:3].cpu().numpy()
        require(len(grids)==len(keys),'face_model_shape','Unexpected decoded face shape')
        return dict(zip(keys,grids))
