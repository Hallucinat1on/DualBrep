import argparse
from pathlib import Path
from .config import load_config
from .surface_mesh import load_mesh
from .reconstruct import reconstruct


def main():
    p=argparse.ArgumentParser(description='Mesh + continuous metric E/V to B-rep (geometry-only coarse + near-field fine refinement)')
    p.add_argument('--mesh',type=Path,required=True,help='production condition directory or normalized mesh file')
    models=p.add_mutually_exclusive_group(required=True)
    models.add_argument('--udf-model',type=Path,help='TorchScript decoder: [N,3] -> metric [N,2] E/V')
    models.add_argument('--vae-bundle',type=Path,help='production hy3dshape VAE bundle; requires --flow-bundle or --asset')
    generation=p.add_mutually_exclusive_group()
    generation.add_argument('--flow-bundle',type=Path,help='condition-only production Flow generation')
    generation.add_argument('--asset',type=Path,help='explicit VAE reconstruction evaluation; reads asset E/V targets')
    p.add_argument('--flow-steps',type=int,default=5)
    p.add_argument('--band-width',type=float,default=.02,help='TorchScript training band; production bundle uses its own config')
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--config',type=Path)
    p.add_argument('--stop-after',choices=('seeds','graph','vertices','curves','surfaces','solid'))
    p.add_argument('--face-second-pass',action=argparse.BooleanOptionalAction,default=None,
                   help='enable second-pass face splitting; --no-face-second-pass keeps only pass 1 (default: enabled)')
    args=p.parse_args()
    if args.vae_bundle and not (args.flow_bundle or args.asset):p.error('--vae-bundle requires --flow-bundle or --asset')
    if args.udf_model and (args.flow_bundle or args.asset):p.error('--flow-bundle/--asset require --vae-bundle')
    cfg=load_config(args.config,{'stop_after':args.stop_after,'face_second_pass':args.face_second_pass})
    try:
        mesh=load_mesh(args.mesh)
        if args.udf_model:
            from .udf_model import TorchScriptUDFProvider
            udf=TorchScriptUDFProvider(args.udf_model,args.device,args.band_width)
        else:
            from .field_provider import ProductionUDFProvider
            from hy3dshape.inference.pipeline import FieldVAEPipeline,FieldFlowPipeline
            from veudf.geometry.sampling import mesh_hash
            from .schemas import require
            if args.flow_bundle:
                if not args.mesh.is_dir():p.error('production Flow requires a condition directory')
                pipeline=FieldFlowPipeline(args.vae_bundle,args.flow_bundle,args.device,
                                          cfg.query_batch_size,sdf_source='condition')
                adapter=pipeline.generate(args.mesh,seed=cfg.seed,steps=args.flow_steps)
            else:
                pipeline=FieldVAEPipeline(args.vae_bundle,args.device,cfg.query_batch_size,sdf_source='condition')
                adapter=pipeline.reconstruct(args.asset,seed=cfg.seed)
            require(adapter.condition.fingerprint==mesh_hash(mesh.vertices,mesh.triangles),
                    'condition_mismatch','Decoder condition must be exactly the input mesh')
            udf=ProductionUDFProvider(adapter)
        reconstruct(mesh,udf,cfg,args.output,resume=args.resume)
    except Exception as exc:
        p.exit(1,f'Reconstruction failed: {exc}; stage diagnostics, when initialized: {args.output}/quality.json\n')
