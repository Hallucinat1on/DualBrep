#!/usr/bin/env python3
import argparse
from bandbrep.neural_fields import fit_fields
p=argparse.ArgumentParser();p.add_argument('--sample-dir',required=True);p.add_argument('--output',required=True)
p.add_argument('--steps',type=int,default=5000);p.add_argument('--batch-size',type=int,default=4096)
p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=0)
a=p.parse_args();fit_fields(a.sample_dir,a.output,a.steps,a.batch_size,a.device,a.seed)
