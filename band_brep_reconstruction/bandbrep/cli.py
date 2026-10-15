import argparse
import json
from pathlib import Path
from .config import load_config
from .field_provider import CadOracleProvider,SampledFieldProvider
from .reconstruct import reconstruct

def main():
    p=argparse.ArgumentParser(description='Portable three-field band-label B-rep reconstruction')
    p.add_argument('--provider',choices=('cad','sampled','neural'),required=True)
    p.add_argument('--sample-dir',type=Path)
    p.add_argument('--field-model',type=Path)
    p.add_argument('--face-checkpoint',type=Path)
    p.add_argument('--face-config',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--step',type=Path)
    p.add_argument('--sdf-backend',choices=('cad','mesh_fast','mesh'),default='cad')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--config',type=Path)
    p.add_argument('--resolution',type=int)
    p.add_argument('--stop-after',choices=('seeds','graph','vertices','curves','surfaces','solid'))
    args=p.parse_args();cfg=load_config(args.config,{'resolution':args.resolution,'stop_after':args.stop_after})
    if args.provider=='cad':
        metadata=json.loads((args.sample_dir/'metadata.json').read_text()) if args.sample_dir else None
        step=args.step or (Path(metadata['source_path']) if metadata else None)
        if step is None:p.error('cad needs --step or --sample-dir')
        provider=CadOracleProvider(step,metadata,args.sdf_backend,cfg.field_error)
    elif args.provider=='neural':
        if args.field_model is None:p.error('neural needs --field-model')
        from .neural_fields import NeuralFieldProvider
        provider=NeuralFieldProvider(args.field_model,args.device)
    else:
        if args.sample_dir is None:p.error('sampled needs --sample-dir')
        provider=SampledFieldProvider(args.sample_dir)
    model=None
    if args.face_checkpoint:
        if not args.face_config:p.error('--face-checkpoint requires --face-config')
        from .model_adapter import DualBrepFaceInitializer
        model=DualBrepFaceInitializer(str(args.face_checkpoint),str(args.face_config),args.device)
    try:reconstruct(provider,cfg,args.output,model,args.resume)
    except Exception as exc:p.exit(1,f'Reconstruction failed: {exc}; diagnostics: {args.output}/quality.json\n')
