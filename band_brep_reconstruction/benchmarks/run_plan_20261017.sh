#!/bin/bash
# 2026-10-17 batch: dev set (S30+H100) x 5 variants, fresh holdout H100b x 2 variants.
set -u
cd /opt/yihengxu/DualBrep/band_brep_reconstruction
PY=/root/miniconda3/envs/Dualbrep/bin/python
OUT=/opt/yihengxu/DualBrep/band_brep_reconstruction/outputs/plan_20261017
DATA=/opt/yihengxu/Datasets/abc/step
H100=/opt/yihengxu/DualBrep/band_brep_reconstruction/outputs/plan_20261016/main/holdout_manifest.json
H100B=$OUT/holdout_b_manifest.json
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for v in all base no_vertex no_surface no_cad; do
  echo "$(date +%T) dev $v"
  $PY benchmarks/run_reconstruction_experiments.py --data $DATA --output $OUT/dev_$v --phase pipeline --variant $v \
      --holdout-manifest $H100 --jobs 8 --timeout 600 > $OUT/dev_$v.log 2>&1
done
for v in all base; do
  echo "$(date +%T) holdout_b $v"
  $PY benchmarks/run_reconstruction_experiments.py --data $DATA --output $OUT/h100b_$v --phase pipeline --variant $v \
      --holdout-manifest $H100B --sets holdout --jobs 8 --timeout 600 > $OUT/h100b_$v.log 2>&1
done
echo "$(date +%T) DONE"
