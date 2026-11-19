# 重建的独立几何评分

新实验使用同步后的生产 `veudf` v3。从源 STEP 运行 [边优先重建入口](../README.md)，或用新版资产的几何 condition 加连续 E/V 回调重建。每个样本写入独立目录。

`experiment_scoring.py` 保留三类离线评分。将本目录加入 Python 模块搜索路径后，可以在实验代码中调用：

| 函数 | 输入与指标 |
|---|---|
| `score(mesh, face_labels, graph, curves, source)` | 切割后的面标签及参数线框与源 CAD 实例对照 |
| `geometry_score(vertices, curves, source)` | 顶点及曲线相对于源 CAD 支撑的距离 |
| `fidelity_score(solid, source)` | 双向面积采样表面距离及体积相对误差 |

`offline_source(targets, raw, oracle, normalization)` 从源 STEP 的 CAD 目标、带面 ID 的三角化、焊接网格的 SDF oracle 和归一化元数据组装参考。真值只用于评分。旧批次的结果与实验口径见[归档实验记录](../../docs/Experiment.md)。

已安装两个模块且运行通过 STEP 验收后，可执行：

```bash
python band_brep_reconstruction/benchmarks/evaluate_step.py --step source.step --run trace_run
python band_brep_reconstruction/benchmarks/evaluate_step.py --step source.step --run mc_run
```

使用 `--extraction both --output paired_run` 时，评分的 `--run` 分别指向 `paired_run/trace` 和 `paired_run/mc`。父目录记录共享的粗细查询耗时。

`run_edge_first_experiment.py` 用于同源 20 例配对。它核验原 STEP 哈希、冻结当前代码和三个历史批次逐例对照，同一 worker 共享采样后分别运行两路。可用以下命令启动新批次：

```bash
OUT="$PWD/band_brep_reconstruction/outputs/new_edge_first_pair"
python band_brep_reconstruction/benchmarks/run_edge_first_experiment.py --initialize --output "$OUT"
python "$OUT/source_snapshot/band_brep_reconstruction/benchmarks/run_edge_first_experiment.py" --run --output "$OUT"
```

初始化支持 `--workers`。逐例结果在 `records/`，自动生成 `REPORT.md`、`metrics.csv` 和 `summary.json`，计时分离共享查询和两路下游阶段。`--report` 可刷新汇总。
