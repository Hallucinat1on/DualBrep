"""Append-only query chunks; includes finite differences and SDF grid queries."""
from pathlib import Path
import numpy as np
from .field_provider import FieldProvider

class RecordingProvider(FieldProvider):
    def __init__(self,provider,path,batch_size=8192):
        self.provider=provider;self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.counter=max([int(p.stem) for p in self.path.glob("*.npz")],default=-1)+1;self.buffer=[];self.rows=0;self.batch_size=batch_size
        self.fingerprint=provider.fingerprint;self.error=provider.error;self.strict_error_bound=provider.strict_error_bound
        if hasattr(provider,'validation'):self.validation=provider.validation
        if hasattr(provider,'normalization'):self.normalization=provider.normalization
    def presence(self):return self.provider.presence()
    def _save(self,xyz,values):
        if len(xyz):self.buffer.append((np.asarray(xyz,float).copy(),np.asarray(values,float).copy()));self.rows+=len(xyz)
        if self.rows>=self.batch_size:self.flush()
    def flush(self):
        if not self.buffer:return
        points=np.concatenate([v[0] for v in self.buffer]);values=np.concatenate([v[1] for v in self.buffer])
        np.savez_compressed(self.path/f'{self.counter:06d}.npz',query_points=points,query_fields=values)
        self.counter+=1;self.buffer=[];self.rows=0
    def evaluate(self,xyz):
        result=self.provider.evaluate(xyz);self._save(xyz,result.values);return result
    def sdf(self,xyz):
        result=self.provider.sdf(xyz);values=np.full((len(result),3),np.nan);values[:,0]=result;self._save(xyz,values);return result
    def gradient(self,xyz,channel,step=1e-5):
        # Keep analytic / autograd gradients without substituting nearest points.
        if type(self.provider).gradient is not FieldProvider.gradient:return self.provider.gradient(xyz,channel,step)
        return super().gradient(xyz,channel,step)
