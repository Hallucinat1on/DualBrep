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
# 仅恢复无方向实例关联
python reconstruct.py --provider cad --step /path/model.step \
  --sdf-backend mesh_fast --output outputs/model_graph --stop-after graph

# 完整拟合与 STEP
python reconstruct.py --provider cad --step /path/model.step \
  --sdf-backend mesh_fast --output outputs/model_solid

# 固定位置采样三场的插值输入
python reconstruct.py --provider sampled --sample-dir /path/fields \
  --output outputs/sampled_model --stop-after graph

# 解析场验证样例
python demo.py --shape sphere_bore --output outputs/sphere_bore
```

`--provider neural --field-model /path/model.pt` 接入三场网络；可通过 `--face-checkpoint` 和 `--face-config` 使用面网络初值。点云面网络仅提供初值，不预测边拓扑，也不替代共享边界约束。

`--stop-after` 支持 `seeds / graph / vertices / curves / surfaces / solid`。`--resume` 只接受相同输入、配置（除停止阶段外）与代码哈希，且验证已保存文件哈希。

配置示例：

```json
{
  "resolution": 65,
  "max_resolution": 129,
  "tau_edge": 0.025,
  "tau_vertex": 0.04,
  "face_threshold1": 0.005,
  "face_threshold2": 0.01,
  "face_filter_size": 10,
  "face_avg_udf_threshold": 0.003,
  "face_samples": 1200,
  "save_queries": false,
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

测试区分分割一致性与下游几何回归：与原 DualBrep 函数比较 100 组随机图标签；验证拆分回退、排除区及缓存恢复。完整 CAD 回归显式使用较宽分割阈值 `0.025/0.05` 和最小分量 1，以保持解析样例的已解析边带；默认 `0.005/0.01` 在低分辨率样例上仍可能缺边，不能将 CAD 回归通过视为默认参数全通过。理想解析场成功不代表 ABC 或生成场的批量成功率，仍需单独运行数据集评估。

旧 `segmentation_ablation`、`refinement_comparison` 运行脚本及空目录已删除；原始统计和诊断保留在 `outputs/`，部分中间数组已清理。数据集三场 oracle 和 30 模型批处理设施仍在 `benchmarks/datasets30_20261013/`，核心重建模块不依赖它们。

## 两阶段分割诊断

`bands.npz/exclusion_reasons`：0=保留核心，1=第一阶段低距离，2=已成功拆分父域中的第二阶段低距离，3=第一阶段大小过滤，4=第一阶段均值过滤，5=第二阶段子分量大小过滤。仅 1/2 进入边带；3/4/5 均无面标签且不参与传播。`run.json` 的 bands 报告记录各原因计数和分割参数、成功拆分父分量；无核心仍明确失败，不把整张表面伪装成一个面。

2026-10-15 验证：42 项测试通过；默认阈值下固定初始分辨率 65 的解析样例 2/5 完成 STEP（圆柱、空心球），方盒/球面通孔缺边，分割球面在边定位阶段失败。详见 `../docs/ThreeFieldReconstructionExperiments.md`。这些失败不等于复制算法有偏差：测试确认标签与原函数一致，但窄带在粗网格上仍可能不可解析。完整 30 模型实验已重跑，结果为 1/30；详见统一实验报告。覆盖验证、传播及参数化方法保持原实现。

历次实验、逐模型结果与清理范围统一见 [三场重建实验汇总](../docs/ThreeFieldReconstructionExperiments.md)。原始统计与成功 STEP 保留；归并前报告原文压缩存放于 `outputs/experiment_archive_20261015/`。
