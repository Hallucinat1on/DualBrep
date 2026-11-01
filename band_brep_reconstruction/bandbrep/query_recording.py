"""Append-only metric query chunks with validity masks."""
from pathlib import Path
import numpy as np

class RecordingProvider:
    def __init__(self,provider,path,batch_size=8192):
        self.provider=provider;self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.counter=max([int(p.stem) for p in self.path.glob('*.npz')],default=-1)+1
        self.buffer=[];self.rows=0;self.batch_size=batch_size
    def __getattr__(self,name):return getattr(self.provider,name)
    def _save(self,xyz,values,valid,saturated):
        if len(xyz):
            self.buffer.append(tuple(np.asarray(v).copy() for v in (xyz,values,valid,saturated)))
            self.rows+=len(xyz)
        if self.rows>=self.batch_size:self.flush()
    def flush(self):
        if not self.buffer:return
        arrays=[np.concatenate([v[i] for v in self.buffer]) for i in range(4)]
        np.savez_compressed(self.path/f'{self.counter:06d}.npz',
                            **dict(zip(('query_points','query_fields','valid','saturated'),arrays)))
        self.counter+=1;self.buffer=[];self.rows=0
    def evaluate(self,xyz):
        r=self.provider.evaluate(xyz);self._save(xyz,r.values,r.valid,r.saturated);return r
    def sdf(self,xyz):
        result=self.provider.sdf(xyz);values=np.full((len(result),3),np.nan);values[:,0]=result
        valid=np.zeros_like(values,bool);valid[:,0]=np.isfinite(result)
        self._save(xyz,values,valid,np.zeros_like(valid));return result
    def gradient(self,xyz,channel,step=1e-5):
        return self.provider.gradient(xyz,channel,step)
