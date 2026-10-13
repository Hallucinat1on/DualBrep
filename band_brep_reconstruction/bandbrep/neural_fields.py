"""Supervised three-channel field fitting, unique-coordinate train/validation split."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .field_provider import FieldProvider
from .schemas import require,write_json

class FieldMLP(nn.Module):
    def __init__(self,width=128,layers=5,frequencies=5):
        super().__init__();self.frequencies=frequencies
        features=3*(1+2*frequencies)
        seq=[]
        for i in range(layers):seq.extend((nn.Linear(features if i==0 else width,width),nn.Softplus(beta=10)))
        seq.append(nn.Linear(width,3));self.net=nn.Sequential(*seq)
    def forward(self,x):
        encoded=[x]
        for i in range(self.frequencies):encoded.extend((torch.sin(x*(2**i)*torch.pi),torch.cos(x*(2**i)*torch.pi)))
        values=self.net(torch.cat(encoded,dim=-1))
        return torch.cat((values[:,:1],torch.nn.functional.softplus(values[:,1:],beta=30)),dim=1)

class NeuralFieldProvider(FieldProvider):
    def __init__(self,path,device='cpu'):
        checkpoint=torch.load(path,map_location=device,weights_only=False)
        self.model=FieldMLP(**checkpoint['architecture']).to(device).eval();self.model.load_state_dict(checkpoint['state_dict'])
        self.device=device;self._presence=tuple(checkpoint['presence']);self.validation=checkpoint['validation']
        self.normalization=checkpoint.get('normalization');self.error=float(max(self.validation['p95']))
        self.fingerprint='mlp:'+hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def presence(self):return self._presence
    def evaluate(self,xyz):
        with torch.no_grad():values=self.model(torch.as_tensor(np.asarray(xyz),dtype=torch.float32,device=self.device)).cpu().numpy()
        return self.result(values)
    def gradient(self,xyz,channel,step=1e-5):
        x=torch.as_tensor(np.asarray(xyz),dtype=torch.float32,device=self.device).requires_grad_(True)
        g=torch.autograd.grad(self.model(x)[:,channel].sum(),x)[0].detach().cpu().numpy()
        return g,np.isfinite(g).all(axis=1)&(np.linalg.norm(g,axis=1)>1e-8)

def fit_fields(sample_dir,output,steps=5000,batch_size=4096,device='cpu',seed=0,width=128,layers=5,frequencies=5):
    p=Path(sample_dir);metadata=json.loads((p/'metadata.json').read_text())
    require(metadata['channel_names']==['sdf','dE','dV'],'channel_order','Expected metric [sdf,dE,dV]')
    with np.load(p/'fields.npz',allow_pickle=False) as f:
        xyz=f['query_points'];values=f['query_fields'];presence=tuple('present' if bool(f[k]) else 'absent' for k in ('has_edges','has_vertices'))
    xyz,index=np.unique(xyz,axis=0,return_index=True);values=values[index]
    rng=np.random.default_rng(seed);order=rng.permutation(len(xyz));n=max(1,int(.05*len(xyz)));hold=order[:n];train=order[n:]
    torch.manual_seed(seed);architecture=dict(width=width,layers=layers,frequencies=frequencies);model=FieldMLP(**architecture).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    x=torch.tensor(xyz,dtype=torch.float32,device=device);y=torch.tensor(values,dtype=torch.float32,device=device)
    # Balance far/near strata using only scalar distances, never query source IDs.
    near=train[np.min(np.abs(values[train]),axis=1)<.03]
    require(len(near)>0,'no_near_samples','No near-zero samples available')
    for step in range(steps):
        ids=np.r_[rng.choice(train,batch_size//2),rng.choice(near,batch_size-batch_size//2)]
        pred=model(x[ids]);res=pred-y[ids]
        weights=1+4*torch.exp(-torch.abs(y[ids])/.02)
        loss=(weights*torch.abs(res)).mean()
        opt.zero_grad();loss.backward();opt.step()
        if step%max(1,steps//10)==0:print(f'field step={step} loss={loss.item():.6g}',flush=True)
    errors=[]
    with torch.no_grad():
        for ids in np.array_split(hold,max(1,len(hold)//batch_size+1)):
            errors.append(torch.abs(model(x[ids])-y[ids]).cpu().numpy())
    e=np.concatenate(errors)
    validation={'mode':'supervised_mlp','unique_positions':len(xyz),'held_out':len(hold),'mae':e.mean(axis=0).tolist(),
                'p95':np.quantile(e,.95,axis=0).tolist(),'max_error':e.max(axis=0).tolist(),'strict_error_bound':False,
                'steps':steps,'seed':seed,'architecture':architecture}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'architecture':architecture,'state_dict':model.cpu().state_dict(),'presence':presence,'validation':validation,
                'normalization':metadata.get('normalization')},output)
    write_json(output.with_suffix('.validation.json'),validation)
    return validation
