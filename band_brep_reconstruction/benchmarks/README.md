# 网格起步的配对实验

手动复现 11-09 曲线到 STEP 的 20 例回放、11-16 区域修复五组 100 个任务，见[完整流程与复现手册](../../docs/Post1116.md)。手册区分历史冻结源码、当前代码重跑及从源网格重新生成前缀，并提供逐步命令。

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

`replay_curve_recovery.py` 从历史实验的已验哈希产物读取工作网格、边带、关联与顶点，跳过加密及拓扑恢复，只重做曲线和可选的曲面／STEP。使用全新输出目录，冻结当前代码，并在子进程中从快照运行；原依赖版本须与缓存观测一致。它不通过修改旧缓存身份强行 `--resume`。例如：

```bash
PYTHONPATH="$PWD/veudf/src:$PWD/band_brep_reconstruction" python \
  band_brep_reconstruction/benchmarks/replay_curve_recovery.py \
  --input-group band_brep_reconstruction/outputs/single_pass_t003_20261106_v1/coarse_003 \
  --id 00009843 --output /data/new_curve_replay \
  --method direct_distance --stop-after solid
```

`--method projected` 可生成旧投影算法的独立对照；`--stop-after curves` 只验收到曲线覆盖，`solid` 继续STEP导出读回及几何评分。`result.json` 分开记录曲线恢复与全局覆盖耗时，包含逐边距离拟合报告和查询计数；全局覆盖仍使用有限差分梯度。

`run_curve_replay_batch.py` 将同一历史组的20个dev冻结为 `direct_distance` 曲线到STEP回放，要求粗0.03／细0.003／单阶段阈值0.003。先用 `--initialize --input-group <历史coarse_003目录> --output <新目录>` 冻结，再用 `--run --output <新目录>` 运行；`--report` 刷新汇总。两个CPU worker，每例总超时10800秒，失败保留分母，`active/`、`runs/<id>/worker.log` 和 `state.json` 提供进度。固定拓扑的面／图评分只作为输入背景，新的成绩是STEP、几何及曲线耗时。11-09批次协议见[实验记录§5.12](../../docs/Experiment.md#512-11-09固定粗目标003网格重跑曲线与step)。

2026-11-02 的实际协议、证据和实时入口见[两级粗目标对照](../../docs/ThreeFieldReconstructionExperiments.md#coarse-fine-comparison)。

`run_region_repair_experiment.py` 固定11-06粗0.03组的原网格、细网格和中心场，从bands起重做五组：baseline、no_mean、small、single_face、joint。每组20dev，10check不运行。用 `--initialize --input-group <coarse_003> --comparison <11-09曲线批次> --output <新目录> --jobs 8` 冻结，再从 `source_snapshot/` 的脚本执行 `--run --output <新目录>`；`--report` 刷新汇总。初始化核验生产距离依赖、输入产物与源STEP哈希；各worker核验冻结源码及本例身份。每例每组10800秒包含全部回退和离线评分，父进程监督进程树。GT缓存只用于评分，重建仅接收输入网格与标量查询。

正式批次为 `outputs/region_repair_coarse003_20261116_v1/`，使用8个单线程CPU worker；预检目录独立，不计入100个正式任务。报告保留pending和全部失败，并分别记录请求模式、实际分支及成功回退。历史11-09仅从曲线开始，本轮包含分割、局部顶点域和定位，完整worker时间不可直接解释为曲线算法的性能回归。

`run_bucket_experiment.py` 按面数 10 面步长分桶（2–10、11–20……91–100），从通过的生产资产中每桶固定种子抽取；每例先跑到 `vertices` 的共同前缀，再执行一次 `direct_distance` 曲线回放和 `baseline/no_mean/small/single_face/joint` 五组区域修复。约 7 个重建任务/例，源码、输入、逐例 condition 和源 STEP 哈希在 `--initialize` 时冻结。小规模试跑示例：

```bash
PYTHONPATH="$PWD/veudf/src:$PWD/band_brep_reconstruction" \
  python band_brep_reconstruction/benchmarks/run_bucket_experiment.py \
  --initialize --dataset /path/to/veudf_dataset --output outputs/bucket_trial \
  --per-bucket 1 --buckets 2-10,41-50 --seed 20261117 --jobs 2
PYTHONPATH="$PWD/veudf/src:$PWD/band_brep_reconstruction" \
  python band_brep_reconstruction/benchmarks/run_bucket_experiment.py --run --output outputs/bucket_trial
python band_brep_reconstruction/benchmarks/run_bucket_experiment.py --report --output outputs/bucket_trial
```

正式 500 例需各桶至少 50 个通过资产；当前 `abc_ve_udf_neighborhood_pools_first50000_20261106` 的 91–100 桶只有 43 个，必须先补预处理或显式 `--allow-short` 做试跑。单例 worker 内部按“前缀→曲线→五组修复”顺序执行，前缀复用同一 CAD target/query，不计为独立批次加速比。

## 失败案例离线可视化

`visualize_reconstruction_failures.py` 为11-09曲线回放的8个失败案例导出离线HTML。当前环境需含OCC和Plotly；可安装可选依赖 `pip install -e './band_brep_reconstruction[visualization]'`。从仓库根目录运行：

```bash
/miniconda/envs/hunyuan/bin/python \
  band_brep_reconstruction/benchmarks/visualize_reconstruction_failures.py \
  --batch band_brep_reconstruction/outputs/curve_distance_coarse003_20261109_v1 \
  --output band_brep_reconstruction/outputs/my_failure_visualization
```

输出目录必须全新；`--cases 00005094 00008222` 可只导出选定案例。`--max-points 18000`、`--max-triangles 45000` 控制显示预算，不改变源实验。打开输出 `index.html` 即可使用，不需要服务或联网；移动产物时保留整个目录及本地 `assets/plotly.min.js`。

页面左右同步显示预测与真实CAD，支持全局/失败局部范围、图层开关、支撑E/V着色、按GT面映射着色、少数支撑过滤、点击实体查询及偏差连线放大。全局背景使用原始网格抽样，局部工作网格也受显示预算限制；数据计数、原始node ID和抽样说明单独保留，少数支撑不会因随机抽样消失。

00005094额外显示冻结函数诊断重放的开路径和遗漏支撑；00004627显示PCA-UV、控制预算与残差及未验收的诊断最小二乘曲面；00000602读取11-14诊断保存的接缝前后BRep；00008222读取拒绝面，比较三维边与pcurve的同参数位置，不修改SameParameter标志。后三者需要该批诊断BRep及冻结输入保持可用；缺失会明确记录 `export_failed`，不会生成虚构几何。

各例在独立进程中从原 `runs/<id>/source_snapshot` 导入重建模块，核验冻结输入、代码和归一化，并完整查询失败实例支撑的CAD边归属，再分层抽样。真实CAD信息只用于离线诊断。脚本仅写新可视化目录，不改实验records/缓存，不重新评定成功率。

每例包含 `index.html`、`data.json` 和 `identity.json`；顶层保存导出源码/模板快照、策略、逐例日志和最终 `state.json`。批次导出即使某例失败也会继续处理其他例，最终非零退出。实际生成页面见 [11-17可视化总览](../outputs/failure_visualization_20261117_v2/index.html)。
