# Post1109 缓存回放

从原 `single_pass_t003_20261106_v1/coarse_003` 的缓存开始，核验源 STEP、库版本和阶段产物的哈希，加载固定工作网格、拓扑和顶点。重新执行直接距离曲线恢复、独立覆盖、曲面、STEP 验收和离线几何评分。原缓存位置可在另一工作树中。

```bash
old_group=/path/to/single_pass_t003_20261106_v1/coarse_003
python band_brep_reconstruction/benchmarks/replay_curve_recovery.py \
  --input-group "$old_group" --id 00009843 --output /path/to/new_replay --stop-after solid
```

`--stop-after curves` 运行到全局覆盖。为整批 20 例回放时，先冻结，再启动两个 CPU worker，每例总超时 10800 秒：

```bash
batch=/path/to/new_batch
python band_brep_reconstruction/benchmarks/run_curve_replay_batch.py \
  --initialize --input-group "$old_group" --output "$batch"
python band_brep_reconstruction/benchmarks/run_curve_replay_batch.py \
  --run --output "$batch"
```

逐例 `result.json` 记录曲线、全局覆盖及下游阶段计时。批次 `REPORT.md` 和 `summary.json` 报告完成、失败和通过数。面和图评分沿用原固定输入。运行中的进度见 `active/` 和 `state.json`。
