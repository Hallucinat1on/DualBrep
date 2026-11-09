# 网格起步的配对实验

`run_mesh_input_experiment.py` 使用当前主入口 `reconstruct(mesh, udf, config, output)`，同源生产 CAD 网格、精确 CAD E/V 和 oracle presence。CAD 实例与关联只供离线评分；不作为模型输入。它不复活旧 MC/CAD CLI。

默认读取 `outputs/mesh_density_20261028_v2/manifest.json`，核验全部 30 份源 STEP 的 SHA-256；只运行该批次相同的 20 个 dev，与旧 M0/L5 配对，10 个 check 保持未运行。旧 raw/prepared 训练点池不再存在时，只需同源 STEP 及已冻结的旧比较产物，无需重新生成完整训练池。

生产几何路径为 `read_normalized → check_solid_faces → collect_targets → validate_geometry → audit_shape → triangulate → weld`，写出 geometry-only condition。新增距离查询全部使用生产 `EdgeOracle/VertexOracle`。自2026-11-05起，当前runner只传入距离值，E/V梯度由重建库在邻近空间点查询UDF后统一中心差分，不读取最近实体ID计算解析梯度；这些邻点也计入 `evaluation_rows`，不再单列 `gradient_rows`。此前冻结批次仍使用原解析梯度，已有成绩未按新路径重跑，不能当作差分方案的验证结果。`offline_support.py` 的 `PolylineDistance` 保留旧实验的离线最近支撑评分方法，逐字复制自旧批次 `source_snapshot/benchmarks/experiment_fields.py` 的同名类，仅用于评分/覆盖，重建 UDF 不使用折线近似。

先不查场做全局粗加密至 0.02，再按E近场细化至 0.005，近边半径0.02。自2026-11-06起新初始化批次不再以V场扩展选区，旧冻结批次仍按原E/V选区解释。两级均只切超长边；不设轮数、三角形或 RSS 硬上限，达标后进入拓扑。`feature_sdf_tolerance=0.004`、batch=65536、关闭查询落盘与旧 L5 保持一致；其余使用主库默认值。每例总时间默认 10800 秒，包含源核验、准备、condition、加密、后处理和离线评分；worker 进程树超时终止。RSS 仅观测，并发 2，不使用 GPU。`--sample-timeout` 可在初始化时设置总秒数，`--coarse-edge` 设置粗目标；初始化后策略冻结。

runner 支持额外核验输入网格与上一批原生四轮实验的指纹一致，并同时对比旧 MC 和原生四轮成绩。旧 MC 成绩来自历史记录。此前仅限时单级批次已终止并删除；[两级加密方案](../../docs/BandLabelTopologyRecovery.md#refinement)已实现。新批次分别记录粗／细阶段的面数、轮数、耗时与中间进度；历史产物不使用新 runner 覆写。

```bash
# 在含 OCC 7.7、libigl 2.6.3 的环境，从仓库根目录运行。
python -m pip install -e './band_brep_reconstruction[benchmarks]'
export PYTHONPATH="$PWD/veudf/src:$PWD/band_brep_reconstruction"
experiment_output="band_brep_reconstruction/outputs/coarse_fine_new_experiment"
python band_brep_reconstruction/benchmarks/run_mesh_input_experiment.py \
  --initialize --output "$experiment_output"
python band_brep_reconstruction/benchmarks/run_mesh_input_experiment.py \
  --run --output "$experiment_output"
python band_brep_reconstruction/benchmarks/run_mesh_input_experiment.py \
  --report --output "$experiment_output"
```

`--initialize` 必须使用新目录，冻结代码、环境、配置、输入和旧对比记录；worker 从 `source_snapshot/` 导入代码，续跑只跳过已有最终 records。不允许静默覆盖未记录的半成品目录。检查 `active/` 可查看在运行的进程与资源，`logs/` 保存逐例标准输出。

批次结束后自动补充归一化回退审计、加密中间状态和耗时分析；也可手动重新生成报告：

```bash
python band_brep_reconstruction/benchmarks/analyze_mesh_input_experiment.py \
  --output "$experiment_output"
```

`REPORT.md`、`summary.json`、`metrics.csv` 按固定 20 例分母统计；运行中同时列出 completed/pending，不将 pending 算作已失败。每例 `source_audit.json` 报归一化差、目标数量和旧中心 32 点的 E/V 差异；目标语义变化、生产过滤拒绝和资源失败均需保留并解释。

只对比后处理流程，不能把精确 CAD 场的成绩当作 VAE/Flow 精度。新旧网格的几何、法向和采样不同，不同输入域的耗时也不能解释为同工作量的加速比。


`run_coarse_fine_comparison.py` 将粗目标 0.02 与 0.03 作为两个独立冻结子组，共用两个 worker，并按同一模型交错调度，共 40 个任务。初始化需要 `--preflight` 指向当前生产源码下的 20 例输入一致性预检 JSON；只接受两组其他配置一致、源码快照相同的比较。`--run` 自动写出顶层配对汇总和各组最终分析，`--report` 可刷新汇总；已完成组的附加审计不会被刷新覆盖。

新批次可在初始化时指定 `--fine-edge .003 --face-threshold1 .003 --no-face-second-pass`，分别设置细目标、第一阶段分割阈值与关闭第二阶段。当前细加密选区只使用E场（半径0.02），V场仍用于顶点定位；E/V梯度统一从距离采样中心差分。报告读取冻结参数，不使用硬编码旧阈值。可用 `--historical-comparison /path/to/coarse_fine_20261102_v2` 对同粗目标历史记录逐例统计面／图／图+STEP的新增成功和退化，并冻结所读历史文件哈希。

`preflight_mesh_inputs.py --output /path/to/new_preflight.json` 可在当前仓库依赖下重新核验全部20例的源哈希、输入网格、归一化及目标数量。预检记录依赖代码哈希，冻结时拒绝已变化的预检来源。上游生产目录变更时需记录差异；精确距离实验不能静默切换为截断UDF。

2026-11-02 的实际协议、证据和实时入口见[两级粗目标对照](../../docs/ThreeFieldReconstructionExperiments.md#coarse-fine-comparison)。
