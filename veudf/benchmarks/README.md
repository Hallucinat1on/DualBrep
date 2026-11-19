# Full preprocessing benchmark

Run the production feature-first builder on a reproducible random selection of
real STEP assets, with the full configured sampling budget:

```bash
PYTHONPATH=veudf/src /miniconda/envs/hunyuan/bin/python -u \
  veudf/benchmarks/benchmark_preprocessing.py \
  --source-root /path/to/source_dataset \
  --output /path/to/new_benchmark_directory \
  --config veudf/configs/surface_pools.json \
  --count 10 --seed 20261105 --workers 10
```

The source dataset supplies only the population and original STEP manifest.
Its stored coordinates and UDF labels are never used to build new data.
The runner freezes the current preprocessing package and configuration under
`run/`, runs complete filtering, sampling, labeling, audits and publication, and
writes `summary.json` and `summary.md`. It requires all selected samples to pass
the new labeling, geometry and publication audits before reporting success.

Mean asset wall time and batch throughput are reported separately: with parallel
workers, batch wall time divided by sample count is not single-asset latency.
CPU seconds and stage timings are also recorded.

The earlier distance-only two-strategy experiment and its original source were
archived under `hy3dshape/output/udf_truncation_benchmark_20261105/run/`. The old
experimental labeling implementations are not part of the production package.

2026-11-06：benchmark按新邻域schema重建STEP，不再断言新旧池坐标逐字节一致。源root仅用于选择STEP清单。
