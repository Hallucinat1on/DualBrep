# 独立三场 B-rep 重建模块

版本 0.3.0，按 `docs/BandLabelTopologyRecovery.md` 的多标签方案实现。复制本目录即可迁移；运行时不导入上级 DualBrep 或预处理项目的代码。复用代码位于 `bandbrep/vendor/`，来源见 [PROVENANCE.md](PROVENANCE.md)。

## 当前流程

1. SDF 提取固定表面网格，在三角形中心查询 `[sdf, dE, dV]`。
2. 复制 DualBrep 两阶段核心提取：第一阶段保留 `dE >= 0.005`，过滤小于 10 个三角形或平均 UDF 不大于 0.003 的分量；第二阶段在父分量内用 `dE >= 0.01` 拆分，仅当至少两个子分量满足大小条件时替换父核心。
3. 最终核心仅作标签来源。第一阶段低距离点、成功拆分时排除的低距离点组成 E；大小/均值过滤掉的分量单独标记，不传播、不作为边。`V = E & (dV < tau_vertex)`，普通带为 `E & ~V`。因此核心不再等于 `~E`，还存在排除区。
4. 普通带组件收集所有相邻面标签，得到独立 edge_id。相同邻面的一对孔口圆环仍是两条独立边。
5. 顶点域收集实际接触的 edge_id 和所有面标签，域内所有节点读取完整集合。禁止顶点域向普通带反向转发；后来标签不会丢失。
6. 输出无方向 E–F、V–E、V–F 关联；此时不要求端点使用次数、面环或方向。
7. v-UDF 距离中心拟合共享顶点；e-UDF 定位各边支持点，按几何组织曲线并固定共享端点拟合。
8. 几何后处理确定闭合/自环及边界方向和环；面内部从面核心采样，以 SDF 校验，在固定共享曲线约束下拟合曲面。
9. 补建 seam/pcurve，装配共享 OCC 实体，导出并读回 STEP。

`recommended` 预设在第 3、4、7、8、9 步启用了局部顶点域、短边推断、圆弧与回转面候选、周期面切口规划；各项见下文"配置与预设"。

多标签以组件引用存储：`BandTopology.node_labels(node)` 返回每个图节点完整的面/边集合，无需复制稠密 face×node 矩阵。`topology_labels.npz` 与 `incidence.json` 一起定义全部标签和成员支持。

已删除旧最短路竞争、等代价界线提取、前置端口实例化、近带反复细分和多阈值核心审计。旧模式与旧配置项不兼容，新配置中不再接受 `max_band_edge`、`max_refinements`、`band_refinement`、`face_seed_sampling`、`stability_factors`、`require_stability`、`verify_band_stability`。历史统计和成功结果保留，部分旧缓存已清理，不能作为新版 resume 缓存。

## 安装

```bash
cd band_brep_reconstruction
python -m pip install -e .
```

完整 CAD 装配需要 `pythonocc-core`；可使用包含它的 conda 环境。可选 `libigl` 用于 mesh_fast CAD SDF，`torch` 用于三场网络与 DualBrep 面初始化器。

```bash
conda install -c conda-forge pythonocc-core=7.7
python -m pip install -r requirements.txt
```

## 输入与运行

通道顺序固定 `[sdf, dE, dV]`。坐标和距离采用一致 metric 尺度；截断归一化由 provider 还原。SDF 内负外正。seam 从生成目标排除；不同面之间的相切/共面边仍保留。无生成顶点的闭合边是正常输入。

```bash
# 推荐：使用经独立集验证的配置预设
python reconstruct.py --provider cad --step /path/model.step \
  --sdf-backend mesh_fast --preset recommended --output outputs/model_solid

# 仅恢复无方向实例关联
python reconstruct.py --provider cad --step /path/model.step \
  --sdf-backend mesh_fast --preset recommended --output outputs/model_graph --stop-after graph

# 固定位置采样三场的插值输入
python reconstruct.py --provider sampled --sample-dir /path/fields \
  --output outputs/sampled_model --stop-after graph

# 解析场验证样例
python demo.py --shape sphere_bore --preset recommended --output outputs/sphere_bore
```

`--provider neural --field-model /path/model.pt` 接入三场网络；可通过 `--face-checkpoint` 和 `--face-config` 使用面网络初值。点云面网络仅提供初值，不预测边拓扑，也不替代共享边界约束。

`--stop-after` 支持 `seeds / graph / vertices / curves / surfaces / solid`。`--resume` 只接受相同输入、配置（除停止阶段外）与代码哈希，且验证已保存文件哈希。

### 配置与预设

**`Config` 的代码默认值仍是 v0.3.0 的历史配置**（分辨率 65、边带按共享边连通、下列改进全部关闭），保证历史实验可以原样复现。实际使用请加 `--preset recommended`。优先级为：代码默认值 < 预设 < `--config` 文件 < 命令行参数（`--resolution`、`--stop-after`）。

`recommended` 预设（`bandbrep/config.py` 中的 `PRESETS`）：

| 配置项 | 预设值 | 代码默认值 | 作用 |
|---|---|---|---|
| `resolution` / `max_resolution` | 257 / 257 | 65 / 129 | 与 DualBrep 两阶段阈值 0.005/0.01 相称的分辨率 |
| `query_batch_size` / `save_queries` | 1048576 / false | 8192 / true | 257³ 下的查询批量与存储 |
| `band_connectivity` | `vertex` | `edge` | 边带、顶点带按共享顶点求连通分量，避免窄带碎成大量片段 |
| `band_propagation` | `noncore` | `distance_only` | 全部非核心点参与标签传播；原因码 3/4/5 的点仍不作为几何支撑 |
| `tau_vertex` | 0.02 | 0.04 | 顶点带阈值；`vertex_domains='local'` 时为局部半径上限 |
| `vertex_domains` | `local` | `threshold` | 按 v-UDF 投票得到候选顶点，每个候选用局部半径，点归属最近候选；参数 `vertex_radius_fraction`=0.45、`vertex_cut_factor`=1.5、`vertex_min_votes`=3 |
| `infer_short_edges` | true | false | 两个顶点域直接相接时补出短边，需沿线 e-UDF/SDF 校验且恰好对应两个面 |
| `surface_candidates` | `revolution` | `basic` | 基础候选失败时再试圆柱、圆锥、圆环 |
| `curve_candidates` | `arcs` | `basic` | 两端有顶点的边额外尝试精确过两端点的圆弧 |
| `periodic_frame_planning` | true | false | 周期面参数切口放进边界空隙；同轴环面共用参考方向；双环圆柱/圆锥面显式构造 seam |

`surface_candidates='revolution'` / `curve_candidates='arcs'` 必须与 `periodic_frame_planning=true` 一起使用；单独打开时，新的周期 pcurve 会让 `FixMissingSeam` 失败，部分原本能成功的模型会失败。

预设在新独立集 H100b（100 个 ABC 模型）上 STEP 通过 38/100；关闭四项改进的同一配置为 19/100（详见 `../docs/ThreeFieldReconstructionExperiments.md` §7）。`project_face_samples` 选项已删除（实验表明无效且会让部分模型退化），含该键的旧配置文件需删掉这一项。

JSON 配置示例（在预设之上覆盖个别项）：

```json
{
  "tau_edge": 0.025,
  "face_threshold1": 0.005,
  "face_threshold2": 0.01,
  "face_filter_size": 10,
  "face_avg_udf_threshold": 0.003,
  "face_samples": 1200,
  "stop_after": "solid"
}
```

`tau_edge=0.025` 只保留给原有几何定位半径和覆盖探针检查，不再定义传播边带。分割单独由 `face_threshold1/2` 控制。两阶段是在 `bands` 阶段计算并缓存，`seeds` 阶段验收标签，避免重复分割。

阈值与网格分辨率需要共同评估，默认值不是任意模型的保证。`allow_surface_residual_exceedance=true` 允许最终 SDF 分辨率的残差告警，不绕过闭合与流形检查。仅执行上述两阶段过滤，不追加平面合并、标签回填或将零核心强制改成一个面：关联阶段保存多标签/无源组件和诊断；参数化遇到无法确认的邻面或顶点关联会明确报告，不伪造有效 STEP。

## 输出

| 文件 | 内容 |
|---|---|
| `sdf_surface.npz` | 固定 SDF 网格 |
| `bands.npz` | 三角形中心、三场值、嵌套 E/V 掩码、核心标签与 `exclusion_reasons` |
| `face_seeds.npz` | 面核心标签 |
| `topology_labels.npz` | 图节点到面/边/顶点实例的映射和原始区域接触 |
| `incidence.json` | 独立实例、完整标签集合、无方向关联、未解释区域 |
| `vertices.json` | v-UDF 定位结果 |
| `logical_graph.json` / `curves.json` | 几何阶段的共享顶点、端点使用与拟合曲线 |
| `cad_graph.json` | 后处理补齐的方向和面环 |
| `face_patches.json` | 固定边界的参数曲面 |
| `logical_to_cad.json` | 逻辑实例与 CAD 实体映射 |
| `reconstruction.step` | 通过内核和 STEP 读回验收的结果 |
| `run.json` / `quality.json` | 阶段、配置、耗时、缓存哈希和失败诊断 |

`logical_graph_ready` 仅表示无方向关联输出完成，可能包含 `unresolved_edges`；不是 CAD 有效性结论。只有 `solid_validated` 表示完整装配与读回检查通过，也不是对连续生成场全域正确性的数学证明。

## 测试与验证

```bash
PYTHONPATH=. python -m unittest discover -s tests -q
python benchmarks/validate_multilabel.py --output outputs/multilabel_validation
```

测试包含：顶点域完整吸收且不回传、后到标签、同面对多边、嵌套掩码、实际 edge_id 关联、固定端点/边界约束、断点续跑，以及方盒、圆柱、球体通孔、分割球面、空心球的 STEP 读回。验证脚本保存 STEP 和 `REPORT.md`。

测试区分分割一致性与下游几何回归：与原 DualBrep 函数比较 100 组随机图标签；验证拆分回退、排除区及缓存恢复。完整 CAD 回归显式使用较宽分割阈值 `0.025/0.05` 和最小分量 1，以保持解析样例的已解析边带；默认 `0.005/0.01` 在低分辨率样例上仍可能缺边，不能将 CAD 回归通过视为默认参数全通过。`tests/test_presets.py` 另用 `recommended` 预设（默认阈值、257³）对上述五个解析场跑完整流程，要求全部得到通过读回的 STEP，约 30 秒。理想解析场成功不代表 ABC 或生成场的批量成功率，仍需单独运行数据集评估。

旧 `segmentation_ablation`、`refinement_comparison` 运行脚本及空目录已删除；原始统计和诊断保留在 `outputs/`，部分中间数组已清理。

### 数据集实验脚本（`benchmarks/`，不属于重建模块）

| 文件 | 作用 |
|---|---|
| `run_reconstruction_experiments.py` | 主实验驱动：多进程、单模型限时、`--variant` 选择配置变体（`all` 即 `recommended` 预设，`base` 为关闭四项改进的 10-16 基线，另有逐项去除组）、逐模型落盘 |
| `experiment_fields.py` | 由 CAD 生成三场的查询接口、离线真值、可复现的相关扰动场 |
| `experiment_scoring.py` | 离线拓扑诊断（面/边/顶点实例与关联）和几何保真度评分 |
| `run_plan_20261017.sh` / `summarize_plan_20261017.py` | 10-18 批量实验与结果汇总 |
| `validate_plan_neural.py` | 单模型三通道 MLP 神经场拟合后的重建 |
| `validate_multilabel.py` / `evaluate_multilabel_batch.py` | v0.2 多标签验证与批次评估 |

这些脚本需要 `libigl` 和 `threadpoolctl`（`pip install -e .[benchmarks]`）。10-16 的一次性实验脚本（切口重试、曲面自由度、噪声补测、审计、报告生成、513³ 代理）已删除，原样副本在 `outputs/plan_20261016/source_snapshot/benchmarks/` 和 `outputs/plan_20261017/source_snapshot/benchmarks/`。

## 两阶段分割诊断

`bands.npz/exclusion_reasons`：0=保留核心，1=第一阶段低距离，2=已成功拆分父域中的第二阶段低距离，3=第一阶段大小过滤，4=第一阶段均值过滤，5=第二阶段子分量大小过滤。默认只有 1/2 进入边带；`band_propagation='noncore'` 时全部非核心点参与传播，但 3/4/5 仍不作为几何支撑点。`run.json` 的 bands 报告记录各原因计数和分割参数、成功拆分父分量；无核心仍明确失败，不把整张表面伪装成一个面。

## 实验结果

| 日期 | 配置 | S30 STEP | 独立集 STEP |
|---|---|---:|---|
| 10-15 | 代码默认值（65³） | 1/30 | — |
| 10-16 | 257³、共享顶点连通、τ_V=0.02 | 5/30 | H100：19/100 |
| 10-18 | `recommended` 预设 | 9/30 | H100b：38/100（同配置关闭四项改进为 19/100） |

剩余失败主要出在面分割（窄面被边带吞掉或碎成多个核心）。历次实验、逐模型结果与清理范围统一见 [三场重建实验报告](../docs/ThreeFieldReconstructionExperiments.md)。原始统计与成功 STEP 保留；10-15 归并前的报告原文压缩存放于 `outputs/experiment_archive_20261015/`，10-18 合并前的四份原文存放于 `outputs/experiment_archive_20261018/`。
