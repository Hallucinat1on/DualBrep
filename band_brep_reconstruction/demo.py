#!/usr/bin/env python3
import argparse
from bandbrep.config import load_config
from bandbrep.demo_fields import cylinder,sphere_with_bore,box
from bandbrep.reconstruct import reconstruct
p=argparse.ArgumentParser();p.add_argument('--shape',choices=('cylinder','sphere_bore','box'),default='cylinder')
p.add_argument('--output',required=True);p.add_argument('--config');p.add_argument('--stop-after',choices=('seeds','graph','vertices','curves','surfaces','solid'))
a=p.parse_args();provider={'cylinder':cylinder,'sphere_bore':sphere_with_bore,'box':box}[a.shape]()
reconstruct(provider,load_config(a.config,{'stop_after':a.stop_after}),a.output)
