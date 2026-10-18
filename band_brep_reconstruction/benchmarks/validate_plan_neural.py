"""Supervised single-shape MLP field test; not generative or cross-shape generalization."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from bandbrep.neural_fields import fit_fields,NeuralFieldProvider
from bandbrep.schemas import write_json
from experiment_fields import BenchmarkCAD
from run_reconstruction_experiments import BASE,prepare,graph_case,full,error,source_hash


def make_samples(source,root,seed):
    rng=np.random.default_rng(seed);parts=[rng.uniform(-1,1,(30000,3))]
    vertices,triangles,_=source.mesh
    tri=vertices[triangles[rng.integers(len(triangles),size=24000)]]
    weights=rng.exponential(size=(len(tri),3));weights/=weights.sum(axis=1)[:,None]
    points=np.einsum('ni,nij->nj',weights,tri)
    parts.extend((points,points+rng.normal(0,.003,points.shape)))
    if source.edge:
        points=source.edge.points[rng.integers(len(source.edge.points),size=30000)]
        sigma=rng.choice([.0002,.001,.005,.02],len(points))
        parts.append(points+rng.normal(size=points.shape)*sigma[:,None])
    if len(source.targets.vertices):
        points=source.targets.vertices[rng.integers(len(source.targets.vertices),size=12000)]
        parts.append(points+rng.normal(0,.015,points.shape))
    points=np.vstack(parts);values=source.evaluate(points).values
    root.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(root/'fields.npz',query_points=points,query_fields=values,
        has_edges=source.presence()[0]=='present',has_vertices=source.presence()[1]=='present')
    write_json(root/'metadata.json',{'channel_names':['sdf','dE','dV'],'normalization':source.normalization,'seed':seed,'source':source.fingerprint})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=5000);args=parser.parse_args()
    torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu';rows=[]
    for mid in ('00000007','00000022'):
        root=args.output/mid;root.mkdir(parents=True,exist_ok=True)
        row={'id':mid,'device':device,'steps':args.steps,'scope':'single-shape supervised field fit; independent query holdout only'}
        try:
            source=BenchmarkCAD(next((args.data/mid).glob('*.step')));make_samples(source,root/'samples',20261016)
            checkpoint=root/'field.pt'
            row['validation']=fit_fields(root/'samples',checkpoint,steps=args.steps,batch_size=2048,device=device,seed=20261016)
            model=NeuralFieldProvider(checkpoint,device)
            cfg=replace(BASE,query_batch_size=8192)
            mesh,points,values,provider=prepare(source,cfg,root,model)
            bands,topo,scored=graph_case(source,mesh,points,values,provider,cfg)
            row['score']=scored;row['full']=full(source,mesh,bands,topo,cfg,root/'full',field=model)
        except Exception as exc:row['error']=error(exc)
        write_json(root/'result.json',row);rows.append(row)
    write_json(args.output/'results.json',{'rows':rows,'source_hash':source_hash()})


if __name__=='__main__':main()
