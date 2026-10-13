# 三场 B-rep 重建：现有数据审查与隔离带标签方案的代码实施设计

日期：2026-10-13。

本文先检查 [`edge_vertex_preprocessing`](../edge_vertex_preprocessing/README.md) 的实际处理过程及 `outputs/abc10_default_20261012` 的输出，再把 [BandLabelTopologyRecovery.md](BandLabelTopologyRecovery.md) 的隔离带、面标签传播、UDF 几何定位与参数化思路落实为模块、数据结构、算法和验收步骤。

**审查结论：四份成功输出在本次检查范围内具有正确的场定义、通道对应及数值精度，基本符合方案的数据目标；尚不能直接把这些文件交给现有 DualBrep 后处理就获得符合方案的 B-rep。** 需要补齐连续查询接口、用于分割的表面网格、可靠隔离带、传播与实例恢复，以及固定共享边界的参数化和 CAD 装配。

本文的 §1 是已执行的检查；其后是待实现的重建代码设计。本次没有修改预处理源码、重写已有输出或实现完整重建器，也没有把 CAD 真值中的面／边／顶点关联当作重建结果。

## 1. 数据处理与结果审查

### 1.1 已执行的检查范围

检查了 `evudf/cad.py`、`distance.py`、`sampling.py`、`pipeline.py`、`io.py`、配置、manifest、测试及四份输出。独立验证包括：

1. 在 Python 3.9.25、OCC 7.7.0、NumPy 1.26.4、SciPy 1.12.0 下重新执行现有 **24 项测试，全部通过**；包含临时依赖目录中的 libigl 2.6.2，耗时约 12.26 秒。
2. 重新计算四份输出的 artifact 哈希和源 STEP 哈希，全部一致；重新读取源 STEP、归一化并构建目标，目标拓扑与保存记录完全一致。
3. 使用不同于原质量检查的随机种子 `20261013`，每份抽取 512 个分层存储查询，共 2,048 个，重新查询 CAD SDF 和全量边距离。对非空顶点集在全部存储查询上重新计算最近顶点距离。
4. 每份新增 512 或 640 个空间／近表面／近边／近顶点查询，共 2,304 个，比较 BVH 与全量 CAD 边查询，并比较所用 SDF 后端与 CAD SDF。
5. 全量检查最近点回代距离和有效梯度长度；检查空集合占位语义。检查保存网格的连接、边界和焊接前后变化。
6. 用真实距离在保存网格的三角形中心进行三组隔离带诊断；这只是检验直接复用该网格的可行性，未运行完整标签传播重建。
7. 对六份被拒绝的源 STEP 重新执行导入和目标筛选，复现对应拒绝原因。

测试命令如下，临时依赖目录是本次环境路径，其他机器应按实际安装位置配置：

```bash
cd edge_vertex_preprocessing
PYTHONPATH=/tmp/evudf-test-deps:. /miniconda/envs/brdf/bin/python -m unittest discover -s tests -v
```

本次独立审计脚本和日志保存在 `/tmp/band-reconstruction-audit-20261013/`，关键结果均写入本文。原作者的检查记录见 [VALIDATION.md](../edge_vertex_preprocessing/VALIDATION.md) 和 [原批次报告](../edge_vertex_preprocessing/outputs/abc10_default_20261012/REPORT.md)。抽样检查与现有测试不构成整个 ABC 数据集或全域连续几何的正确性证明。

### 1.2 输出数量与数值检查

实际通道顺序始终是 **`[SDF, E-UDF, V-UDF]`**，不能按口头列举的 `SDF,V-UDF,E-UDF` 顺序读取。

| 模型 | Face / 目标 Edge / 必要 Vertex | 查询数量 | SDF 后端 | 本次 SDF 最大误差 | 本次 E-UDF 最大误差 | 本次 V-UDF 最大误差 |
|---|---|---:|---|---:|---:|---:|
| `00000003` | 25 / 40 / 18 | 200,138 | `mesh_fast` | $1.459\times10^{-4}$ | $2.945\times10^{-8}$ | $5.927\times10^{-8}$ |
| `00000007` | 3 / 2 / 0 | 200,006 | `cad` | $2.930\times10^{-8}$ | $2.976\times10^{-8}$ | 空集合 |
| `00000008` | 21 / 35 / 17 | 200,122 | `cad` | $2.967\times10^{-8}$ | $2.967\times10^{-8}$ | $5.960\times10^{-8}$ |
| `00000009` | 6 / 5 / 0 | 200,015 | `cad` | $2.757\times10^{-8}$ | $5.764\times10^{-8}$ | 空集合 |

表中 SDF／E-UDF 是每份 512 个存储查询的参考比较；V-UDF 为非空集合的全量比较。单位为归一化长度，非空字段未截断。四份共有 800,281 个共享查询；空顶点通道的 `0.02` 是占位值，不参与真实顶点误差统计。

新增查询中，BVH 与全量 CAD 的边距离差均为 0；`00000003` 的 SDF 最大差约 $1.478\times10^{-4}$，其余三个 CAD 后端为 0。存储查询的无歧义最近 edge id 没有发现不一致。符号统计排除 $\min(|s|,|s_{ref}|)\leq5.2\times10^{-4}$ 的近零点后，没有发现符号冲突；近零点仍纳入前述距离误差统计。这些边参考查询使用同一 OCC 距离内核，主要验证加速剪枝和存储；独立解析正确性由长方体、圆环、有限线段／圆弧等现有测试补充。

有效梯度单位长度的最大误差约 $5.97\times10^{-8}$。float32 最近点坐标回代产生的最大距离差约 $1.16\times10^{-7}$，与当前存储精度相容。

六份拒绝结果也复现：

| 模型 | 原因 | 本次复核 |
|---|---|---|
| `00000000` | 多 solid | 6 个 solid |
| `00000001` | 多 solid | 2 个 solid |
| `00000002` | 退化边 | source edge 5 |
| `00000004` | 退化边 | source edge 60 |
| `00000005` | 多 solid | 10 个 solid |
| `00000006` | CAD 容差超预算 | $2.12616\times10^{-5}>2\times10^{-5}$ |

这些属于初版支持范围及精度预算下的拒绝，不是“只成功提取部分字段”。不能把 4/10 推广为 ABC 全集接受率。

### 1.3 与目标文档的一致性

| 要求 | 实际实现 | 判断 |
|---|---|---|
| 同位置三通道、统一坐标尺度 | `query_points[Q,3]` 与 `query_fields[Q,3]`；先量化查询点再查询 | 符合 |
| SDF 内负外正，可作固定条件 | 边界 face 距离 + CAD 分类；03 使用有参考检查的 mesh fast 路径 | 符合，但 03 有约 $10^{-4}$ 表面近似误差 |
| 非 seam 面间边；保留相切／共面边 | `collect_targets()` 检查每次使用、两邻面和参数 seam；不按 sharp 删边 | 符合；共面分界已有回归测试 |
| 完整有限边距离，包括端点 | OCC 距离查询有限 edge；BVH 使用几何包围盒 | 符合，不是边点最近邻 |
| 从保留边构造必要顶点 | 真实二度点保留；圆／椭圆完整闭环人工锚点删除 | 符合保守初版，尚未完成一般闭环规范化 |
| 无必要顶点闭环 | 07 为 2 条闭环，09 为 5 条闭环；两者顶点集为空 | 符合，不能按“没有顶点”判为失败 |
| 高价顶点、多边身份 | 03 包含四度和二度顶点；每条 edge 有独立 id | 符合，不应改用固定三价顶点假设 |
| 退化／开放／非流形范围检查 | 先验证 CAD，再核查原始边使用和曲线 | 已实现主要检查；不等于全局无自交／重合认证 |
| 与后续连通分量分割匹配的表面 | 保存的是按 CAD face 拼接的三角化 | 需要另建或修复用于重建的表面连接 |
| 可查询连续场 | 当前输出是有限散点及附加真值；模块另有 CAD oracle | 文件本身尚不满足连续查询接口 |

实现位置：[cad.py](../edge_vertex_preprocessing/evudf/cad.py)、[distance.py](../edge_vertex_preprocessing/evudf/distance.py)、[sampling.py](../edge_vertex_preprocessing/evudf/sampling.py)、[pipeline.py](../edge_vertex_preprocessing/evudf/pipeline.py)。

### 1.4 必须处理的网格问题

`triangulate()` 为每张 CAD face 追加独立顶点，跨 face 的相同位置没有共享索引。直接在保存网格上运行 `face_adjacency`，其连通分量已经暴露原 CAD face 的分块，而且不能跨该边界传播。

| 模型 | 三角形数 | 原网格连通分量 | 原网格边界边数 | 诊断性坐标焊接后的分量／边界边 | 最大网格边长 |
|---|---:|---:|---:|---|---:|
| 03 | 31,436 | 25 | 4,586 | 1 / 0 | 1.4632 |
| 07 | 840 | 3 | 846 | 1 / 0 | 1.7999 |
| 08 | 6,464 | 21 | 4,764 | 1 / 0 | 1.2960 |
| 09 | 9,432 | 6 | 1,696 | 1 / 0 | 0.9658 |

诊断使用 `Trimesh(process=False)`，再执行 `merge_vertices(digits_vertex=7)`；此参数是坐标量化规则，**不是对任意 CAD 安全的焊接半径**。四份焊接网格均变为 watertight，未修改磁盘数据。生产路径不能无条件靠坐标焊接处理薄壁、接触或不匹配的边界采样。

此外，`mesh_deflection=0.0002` 控制曲面三角化偏差，**不限制平面上三角形的边长**。因此仍有接近形状尺度的长边。焊接后只在三角形中心查询真实 $d_E,d_V$，不做细分、不使用大小过滤，得到：

| 模型／真实 face 数 | $\tau_E=0.002$ 的核心数 | $\tau_E=0.005$ 的核心数 | $\tau_E=0.01$ 的核心数 |
|---|---:|---:|---:|
| 03 / 25 | 3 | 13 | 17 |
| 07 / 3 | 1 | 1 | 1 |
| 08 / 21 | 1 | 1 | 4 |
| 09 / 6 | 1 | 1 | 1 |

有顶点样本使用 $\tau_V=2\tau_E$；空顶点样本关闭顶点带。03 在最后一组丢失一个 face 的核心并分裂另一个 face；08 丢失五个 face 的核心。较小阈值下则存在多个 face 核心被错误连接的情况。

这验证了文档中关于“面核心必须非空且被完整隔开”的限制：**现有距离标签正确，不代表原始 CAD 三角化配合中心阈值就能正确分割。** §4–§5 的网格细化和带构造是本批数据上已有证据支持的必要工作。

### 1.5 其他限制与建议改进

- **连续查询缺口：**不能把 `query_fields` reshape 成三维体素，也不能用最近散点标签代替任意位置的真实场。需要 §3 的 provider。现有 `closest_E_points`、`vertex_points`、instance id 是额外真值，直接使用它们会改变“三个标量场重建”的实验设定。
- **表面样本有效数量：**32,000 个 `on_surface` query 从 10,000 个表面点重复抽取，四份实际唯一坐标仅约 9,592–9,595 个。距离标签有效，但覆盖不等于 32,000 个独立位置。后续可改为独立面积采样；评估训练拟合的留出集应先按坐标去重，防止重复点泄漏。
- **规范化范围：**`analytic_loop_anchor_v1` 仅删除明确的解析闭环锚点。一般 B-spline 闭环和人工二度分段保守保留，不能宣称已经得到与原始 CAD 参数切分完全无关的标准边图。训练、预测和评估必须使用同一版本。
- **全局几何检查：**`validate_geometry()` 只拒绝容差内的不同顶点；曲线内部重合、异常相交尚无全面认证。扩大 ABC 规模前应增加相应检查或明确数据范围。
- **误差预算：**原 `validate_fields()` 对 mesh SDF 使用约 $5.2\times10^{-4}$ 的复合容差；这不是可直接用于全部重建阶段的 CAD 装配容差。03 不能因为边／顶点标签约 $10^{-8}$ 的参考差值，就要求其 SDF 表面也达到同样精度。
- **代码版本：**03 的保存代码哈希与当前 `cc9b6c9e…` 一致；07–09 保存于 `41d8b9d8…`。本次重新查询和拓扑重建相符，但它们不是同一代码指纹的批次。正式冻结数据时统一版本或保留明确的混合版本清单；不要修改 metadata 来绕过 `--resume` 校验。

上述连续查询和网格接入问题不要求废弃现有三场标签；应先在重建侧补齐接口和网格处理。独立表面查询、更全面的几何检查及规范化扩展是下一版预处理的改进项，改变其规则时使用新的数据版本。

## 2. 重建范围、模块与数据对象

保持主路线：**SDF 表面 → 隔离带 → 面核心 → 受限多源传播 → 边／顶点实例与面环 → UDF 几何定位 → 共享曲线 → 边界约束曲面 → CAD 装配。** 几何定位可提前为 port 关联提供坐标，但不能用几何最近邻取代拓扑判断。

建议新建以下包，预处理模块保持独立。下列路径和接口是设计名称，尚未实现：

```text
band_brep_reconstruction/
  reconstruct.py           # 总入口、阶段缓存、失败返回
  field_provider.py        # metric 三场查询，梯度／误差／存在性
  surface_mesh.py          # SDF 等值面、定向、连通性、自适应网格
  bands.py                 # 边带、顶点带、带边界切分
  face_seeds.py            # 面核心连通分量及稳定性检查
  propagation.py           # 带内多标签到达代价
  interfaces.py            # 共同最小界面与一致拼接
  vertex_localization.py   # 顶点投票、距离中心拟合、分裂复核
  edge_graph.py            # 边实例、port、顶点关联
  geometry_refinement.py   # 分支内投影、追踪与精化
  boundary_cycles.py       # 有向面扇区、coedge、环遍历
  curve_fitting.py         # 固定端点曲线、闭环、分段组合
  surface_fitting.py       # 参数化初值、UV、固定边界求解
  occ_assembly.py          # 共享 OCC 实体、pcurve、seam、shell／solid
  evaluation.py           # 仅此模块读取目标拓扑作评分
  schemas.py              # 中间数据对象和序列化
```

中间对象至少区分：

| 对象 | 核心字段 |
|---|---|
| `SurfaceMesh` | 顶点、三角形、有向 halfedge、连通分量、局部分辨率、几何误差 |
| `BandState` | `edge_band`、`vertex_band`、`edge_region`、`core_mask`，各自独立 |
| `FaceSeed` | 临时 seed id、核心单元、候选 face id、稳定性状态 |
| `ArrivalState` | 带单元上的来源 face、有限到达代价、不可达 mask、局部竞争关系 |
| `VertexNode` | 共享位置、局部支撑、置信度、入射 ports；不先固定价数 |
| `EdgeInstance` | 独立 edge id、两侧 face、链／闭环、ports、端点 id、有序支持域 |
| `CoedgeUse` | face id、edge id、方向、wire id、局部扇区、pcurve |
| `BoundaryCycle` | 有序 coedge 使用；UV 前不强行确定内外环 |
| `CurveRecord` | 唯一三维曲线、逻辑 edge id、参数区间、内部拟合段 |
| `FacePatch` | 支撑曲面、有效 UV 域、边界使用、周期性与约束误差 |

edge id 不能由 face 对或顶点对代替。逻辑图允许多重边、必要二度点、高价点、self-loop，以及 `vertex_ids=[]` 的完整闭环。后处理新增的 seam、锚点或曲线切段单独记录来源，不增加模型应预测的顶点目标。

## 3. 从现有文件到可查询的三场接口

### 3.1 读取与尺度

现有读取器可直接复用：

```python
from evudf.io import load_fields

arrays, metadata = load_fields(sample_dir, normalize=False)
xyz = arrays["query_points"]
values = arrays["query_fields"]
assert metadata["channel_names"] == ["sdf", "dE", "dV"]
sdf, edge_udf, vertex_udf = values.T
```

几何运算统一使用上述 metric 值，即归一化物体坐标系中的实际长度。若 provider 接收训练尺度输出，应乘各自 `clip_values` 恢复长度，同时保留饱和 mask；被截断的远场距离不能恢复成真实未截断值。投影和梯度使用同一尺度，不能直接把 `load_fields(normalize=True)` 的梯度套进单位距离投影公式。

### 3.2 三种 provider，明确实验含义

| Provider | 实现 | 用途及限制 |
|---|---|---|
| `CadOracleProvider` | 复用 `read_normalized`、`collect_targets`、`EdgeOracle`、`VertexOracle`、`SDFOracle`，仅向重建器暴露三场 | 首先调通确定性算法；不返回目标 id、最近点、CAD 曲线或面标签 |
| `SampledFieldProvider` | 用 `query_points/query_fields` 拟合可查询函数，或在具有误差诊断的网格上插值 | 验证“只给当前 NPZ 能做到什么”；有新的近似误差，有限点不能保证唯一恢复全部细节 |
| `GeneratedFieldProvider` | 包装未来模型的 SDF 条件及两个结构通道 | 正式输入；遵循相同查询、梯度、尺度和置信接口 |

建议统一接口：

```python
class FieldProvider:
    channel_names = ("sdf", "dE", "dV")

    def evaluate(self, xyz):
        # values: [N,3] metric; valid/saturated: [N,3]
        # error: 误差估计及是否有严格界的标记
        raise NotImplementedError

    def gradient(self, xyz, channel):
        # [N,3] 梯度与有效性；可来自自动微分或受控有限差分
        raise NotImplementedError

    def presence(self):
        # 每个结构集合：present / absent / unknown
        raise NotImplementedError
```

`CadOracleProvider` 本身访问 CAD 是已声明的真值场实验；其余重建代码不应看到 CAD 拓扑。重新归一化时与 metadata 的矩阵核对，禁止对归一化后的 query 再变换一次。

`SampledFieldProvider` 初版可用三输出监督 MLP：按唯一坐标划分训练／验证点，按 query 来源及距离带平衡损失，对非空 UDF 强制非负；SDF 拟合完成后作为固定条件。仅在唯一最近点、非零、未饱和区域使用梯度或 Eikonal 辅助项；距离零集和等距切换处不可盲目约束单位梯度。必须报告场拟合误差再评估几何提取，不能把新增拟合误差记为预处理错误。

近邻插值可用于显示初值，不应作为默认精确重建后端：它会引入不连续或平滑偏差，密集查询不会补回原样本没约束的结构。对 CAD 真值实验可以追加独立验证查询；仅有 NPZ 时必须说明近零区域的覆盖及外推局限。

### 3.3 允许输入与评估真值分离

正式“三场重建”入口读取场值、空间坐标、尺度和 provider 能力；面、边、顶点的真值 id 不可进入种子、传播、拟合或装配。

以下字段只供评分或显式的额外监督对照：`topology.json`、`surface_face_id`、`edge_point_id`、`vertex_points`、`nearest_*_id`、`closest_*_points`。原始 `surface_mesh.npz` 只用于 CAD 表面辅助实验；正式路径从 SDF 重新提面，避免原网格分块泄漏分割答案。

真值场实验允许使用真实的 `has_vertices=False` 测试空集分支。生成场必须使用模型预测的存在性或有覆盖检查的场检测；“当前采样没找到极小值”应先记为 `unknown`，不能自动认定为空。

## 4. 阶段 A：SDF 表面与可用于传播的网格

### 4.1 从 SDF 提面

接口：`extract_surface(provider, cfg) -> SurfaceMesh`。

初版在 `[-1,1]^3` 规则格点分块查询 SDF，使用仓库已有的 PyMCubes 提取零等值面；把格点索引乘 `2/(resolution-1)` 并减 1，还原物体坐标。固定数组轴顺序，不能在构造 query 和映射 mesh 时交换轴。该步骤对 **SDF** 使用 Marching Cubes，不对两个 UDF 的零值使用普通 MC。

```python
# 设计片段：query_grid_sdf 分块查询，避免同时保存全部 xyz 和三个体素通道。
sdf_grid = query_grid_sdf(provider, bounds=(-1.0, 1.0), resolution=res)
vertices_grid, triangles = mcubes.marching_cubes(sdf_grid, 0.0)
vertices = vertices_grid * (2.0 / (res - 1)) - 1.0
mesh = orient_and_validate_surface(vertices, triangles, provider)
```

记录表面误差、体素尺寸和提取分辨率。不能因为格点角点同号就保证格内没有薄结构；在有有效 SDF 误差界时，用单元中心值和覆盖半径排除不可能含零集的格子，其余细化。没有严格误差界的生成场采用保守加密与多分辨率稳定性检查。

CAD oracle 的密集查询可能很慢：先测少量 query 吞吐量，再分块缓存；不要直接对数千万点逐点跑 CAD SDF。CPU 真值实验可显式采用已经验收的 mesh SDF 加速并记录误差；正式模型查询或 SampledFieldProvider 使用各自高效批处理。缓存键包含输入场、尺度、分辨率及后端版本。

### 4.2 网格有效性和后续细化

建立共享顶点、共享边的有向 halfedge 结构，检查重复／零面积三角形、非流形边、边界、法向方向和表面连通分量。根据局部两侧 SDF 值确认朝向，不只依赖有向体积的符号。合法内腔表面需要保留，不能沿用“只保留最大连通分量”。

对 CAD 辅助网格路径，先执行受控焊接与拓扑检查；不能直接把其原 25/3/21/6 个连通分量当作发现的 face。正式评分使用从 SDF 重新提取的网格，并比较不同分辨率结果。

`refine_surface_for_bands()` 在低 $d_E$、低 $d_V$、高曲率或掩码不确定区域细分，维护跨三角形共用的分裂点，禁止产生 T-junction 裂缝。远离结构的大片平面可以保持较粗；近带单元应满足独立的边长／覆盖半径限制，不能只使用 meshing deflection。

## 5. 阶段 B：隔离带与稳定面核心

### 5.1 掩码和阈值

接口：`build_bands(mesh, provider, cfg) -> (refined_mesh, BandState)`。

```python
d = provider.evaluate(sample_xyz).values
edge_band = d[:, 1] < tau_edge
vertex_band = d[:, 2] < tau_vertex if vertices_present else np.zeros(len(d), bool)
core_mask = ~(edge_band | vertex_band)
edge_region = edge_band & ~vertex_band
```

`vertices_present` 必须经过 §3.3 的存在性处理；`unknown` 时继续检测，不按假值跳过。顶点带独立于边带，允许 $\tau_V>\tau_E$；不先把顶点带裁剪进边带。空边集单独处理为按 SDF 表面区域建立 face 候选，之后仍需支持周期参数化。

所有阈值使用 metric 单位。初期可将 `(0.002,0.005,0.01)` 作为实验扫描值，但 §1.4 已表明它们在原网格上均不能直接解决问题。需要联合调节近带网格分辨率和局部阈值，而不是只扩大带。

### 5.2 不只看中心点：查询、认证与切分

对三角形查询顶点、边中点、重心，必要时追加查询。设某样本点 $p$ 到三角形全部位置的最大距离上界为 $r$，真实 UDF 为 1-Lipschitz，预测绝对误差不超过 $\epsilon$，则：

$$
\hat d(p)-r-\epsilon>\tau
\quad\Rightarrow\quad\text{整个三角形在带外},
$$

$$
\hat d(p)+r+\epsilon<\tau
\quad\Rightarrow\quad\text{整个三角形在带内}.
$$

介于两者之间的单元不能仅凭中心点保留连通通道，应继续细分，或构建经过验证的带边界切分。应用到近似 SDF 表面时，预算还包括表面几何误差。局部可变阈值需使用单元内阈值上下界；不能将单点阈值无条件用于整个三角形认证。

初版可在近带把单元边长降至 $\tau_E/2$ 量级作为起点，同时检验上下一级细分结果。这个经验比例不是拓扑保证；狭窄 face、短 edge 和近切分支需按局部结构继续调整。达到资源／精度下限仍不确定时记录失败，不把该单元自动设为带外。

带边界可在局部线性近似的 $g_E=d_E-\tau_E$、$g_V=d_V-\tau_V$ 上切分三角形，形成共享索引的 core / ordinary band / vertex region 子网格。切分内部仍需上述采样控制，避免线性插值漏掉 UDF 谷。对 band 构造的这些标量做阈值等值线，是提取有限宽邻域边界，不是把它当作 B-rep edge。

### 5.3 复用连通分量，修改外围判据

接口：`extract_face_seeds(mesh, bands, cfg) -> list[FaceSeed]`。

复用 [`clustering.py::hierarchical_segmentation`](../clustering.py) 的三角形共享边邻接、诱导子图及 `scipy.sparse.csgraph.connected_components` 思路。第一版只用一轮核心连通分量，不直接继承大小过滤、二次高阈值拆分、只保留最大网格分量或无结果时全设单 face 的兜底。

```python
def core_components(face_adjacency, active):
    ids = np.flatnonzero(active)
    labels = np.full(len(active), -1, dtype=np.int32)
    if len(ids) == 0:
        return labels
    remap = np.full(len(active), -1, dtype=np.int64)
    remap[ids] = np.arange(len(ids))
    pairs = face_adjacency[active[face_adjacency].all(axis=1)]
    graph = scipy.sparse.coo_matrix(
        (np.ones(len(pairs)), (remap[pairs[:, 0]], remap[pairs[:, 1]])),
        shape=(len(ids), len(ids)),
    ).tocsr()
    _, local = scipy.sparse.csgraph.connected_components(graph, directed=False)
    labels[ids] = local
    return labels
```

图的 shape 显式包含所有激活三角形，不能丢掉没有激活邻居的孤立节点。小核心保留并标记待判，不按大小直接删 face。

在相邻阈值和两级网格之间建立核心重叠对应：缺口导致跨面连接时补齐屏障；薄颈被带切断时降低局部阈值并检查该连接是否有真实边场低谷；整张窄面被吞掉时恢复其核心。不得只凭法向相近合并，因为相切／共面边界是保留目标。

不使用真值 face 数调参。调试评分可报告欠分割、过分割和缺种子；重建自身依靠覆盖、分辨率稳定性和距离证据决定是否接受。确认 seed 到 face 的映射后，再进入传播；映射改变必须重算相关传播。

## 6. 阶段 C：受限多源传播与共同最小界面

### 6.1 在带区域建立一致的离散域

接口：`build_band_complex(mesh, bands, face_seeds)`。

面核心仍可定义在三角形上；传播与界面插值建议统一转到**已切分的带三角网格顶点**上。必要时对原三角形做重心细分，引入原顶点、边中点和中心，使核心—带接触边界有显式公共节点。两种表示间保存映射，不把中心标签直接当作原网格顶点标签。

普通传播域为 $R_E=B_E\setminus B_V$。每个 face 在其核心与 $R_E$ 的接触边界产生来源 $S_f$。路径只能在该带连通区域内行走，不能进入 $B_V$、借道其他核心或通过空间 kNN 跳到薄壁另一侧。每个带连通区域记录与其接触的全部候选 face。

### 6.2 首版保留全部局部来源

接口：`propagate_arrivals(band_complex, seeds) -> ArrivalState`。

为带网格边赋正权重，初版取表面边长；从每个局部 face 来源执行多源 Dijkstra，得到 $T_f$。同一 face 的多个已确认核心合并为同一来源的多个起点。原型按带连通区域逐个计算，内存约为 $O(N_R K_R)$；来源很多时分块处理，先保证正确性，再研究候选裁剪。

```python
# 伪代码：每个来源是一个 face 的所有接触节点。
for region in band_components:
    for face_id in region.touching_faces:
        arrival[region.id, face_id] = dijkstra_from_sources(
            region.graph, contact_nodes[region.id, face_id]
        )
```

不要只给每个节点保存一个获胜标签，否则无法插值两来源的代价差；也不要预先强制最多两个来源，否则会掩盖漏检顶点或带粘连。不可达状态单独保存，禁止对 `inf-inf` 求根。没有接触来源的区域标记为未解释，并返回核心／网格检查。

### 6.3 在每个三角形内提取代价下包络

接口：`extract_interfaces(complex, arrivals, provider) -> InterfaceGraph`。

对一个传播三角形，令其重心坐标为 $\lambda$，将每个有效来源的三个顶点代价线性插值为 $T_f(\lambda)$。其获胜区域是：

$$
P_f=\{\lambda_i\geq0,\ \sum_i\lambda_i=1,\ T_f(\lambda)\leq T_g(\lambda)\ \forall g\}.
$$

用二维多边形半平面裁剪计算这些区域；两个区域共同边界是 $T_a=T_b=\min_fT_f$ 的候选段。这样处理三来源局部竞争，而不是把任意“前两名”直接宣布成 edge。

在共享网格边的端点 $u,v$ 上，若 $D=T_a-T_b$ 有有效异号值，交点参数为

$$
\alpha=\frac{D(u)}{D(u)-D(v)}.
$$

交点还必须处在共同最小部分；只满足两代价相等可能被第三个来源压住。缓存键包含共享网格边 id 和无序 face 对，邻接三角形复用同一交点。近零分母、整段相等、多来源交点采用统一容差与明确歧义状态，不能各三角形独立随机选解。

输出保存 face 对、界面段连接、两侧方向和支撑带。顶点带之外的三方／多方交汇先检查漏检顶点、混合核心或带粘连；不直接创建一个任意高价 CAD 顶点。

传播界面是拓扑候选。对每个候选链检查低 $d_E$ 支持及两侧核心来源；界面偏离真实边时，可在该链的支持带内定位 UDF 中心线。若两侧来源错误或没有对应低谷，应返回拓扑阶段，不吸附到最近的另一条真实边。

## 7. 阶段 D：必要顶点、边实例和 ports

### 7.1 从顶点场定位共享顶点

接口：`localize_vertices(provider, vertex_regions) -> list[VertexNode]`。

顶点带连通块只作候选区域，一个块可以包含多个真实顶点。对低 $d_V$ 的查询点，在梯度有效时使用有限步投票：

$$
v_i^{(0)}=x_i-\alpha\min(d_V(x_i),r_i)
\frac{\nabla d_V(x_i)}{\|\nabla d_V(x_i)\|+\eta}.
$$

用局部尺度和位置峰区分候选，不用固定大半径直接并成一个点。支持域内样本需与同一最近顶点假设相容；通过多中心竞争残差检查混合，必要时拆分。

距离中心可先线性初始化：对参考观测 $(x_0,d_0)$，由球面方程相减得到

$$
2(x_i-x_0)^Tv=\|x_i\|^2-\|x_0\|^2-(d_i^2-d_0^2).
$$

用 QR/SVD 求解并检查秩和条件数；再用 `scipy.optimize.least_squares` 的鲁棒损失精化

$$
\sum_i w_i\rho(\|x_i-v\|-d_V(x_i))
+\lambda_S s(v)^2+\lambda_E d_E(v)^2.
$$

SDF 与边场项按各自误差预算设权重。只有表面上的共面／单侧数据时，追加小范围不同方向的三维 query 改善条件数；不要将多个顶点附近的观测混入同一个球模型。检查多初值、支撑域收缩和追加查询后的稳定性。

### 7.2 边实例化

接口：`instantiate_edges(interface_graph, vertices, bands) -> LogicalGraph`。

按 **face 对 + 独立界面连通分支 + 必要顶点处分段** 建立 edge id。普通链内部应为二度结构；端点是顶点区域的 port 或暂时未解释的断口。完整二度循环建立一条无生成顶点闭环。

规则必须支持：同一 face 对的多个独立环；同一顶点对的多条曲线；有一个必要顶点的 self-loop；两侧 face 相同但中间存在必要二度顶点的两条边。拟合产生的内部曲线分段不参与这个实例划分。

如果已精化的顶点落在一条候选路径内部，先在其局部支撑处分裂路径，再创建两端 ports。不能只依赖粗顶点带是否已把路径完全切开。

### 7.3 顶点邻域的 ports 与短边

每个 port 保存 `edge_id`、face 对、进入点、切向、所属表面位置和置信度。围绕顶点选择小球面，查询 $d_E$ 的低值分支，与到达该区域的 ports 核对。半径应大于定位噪声、足够小以避开其他顶点；同一 self-loop 有两个不同 port，即使终点 id 相同也不能合并端口。

利用表面局部邻接恢复绕顶点的 face 扇区顺序，确认每个 port 两侧的 face 与扇区相容。只按最近顶点连接，或只按相同 face 集合合并顶点，都不可靠。

两个顶点邻域接触、短 edge 全部落入 $B_V$ 时，普通带内可能没有该边链。此时收缩已定位顶点的排除邻域，并在顶点间的局部表面补做 UDF 分支检测和标签传播；若无法分离，保留联合局部问题并继续细化。不能把缺少普通带样本当作短边不存在。

## 8. 阶段 E：在已确定实例内精化曲线几何

接口：`refine_edge_geometry(edge_instance, provider, graph) -> OrderedCurveSamples`。

对每条 edge 的支持带独立处理，保留弧向覆盖。低 UDF 点沿径向投影，步长受局部带宽和到其他分支的距离限制；每步检查 field 残差、所属表面区域及分支相容性。理想距离场有 $\pi_E(x)=x-d_E(x)\nabla d_E(x)$；预测梯度不是单位梯度时应归一化／阻尼并检查更新，而不是无条件迭代。

切向可来自投影点的局部 PCA，或三维扰动 query 的梯度矩阵最小右奇异向量。只在表面上采样可能只张成一个法向方向，应检查奇异值间隔，必要时追加三维 query。$\nabla d_E$ 本身是径向，不能直接当作曲线切向。

按标签分支的已有次序执行“切向预测 + 横截面校正”：

1. 从当前点沿连续定向切向预测步长 $h$。
2. 在垂直切向的小平面内最小化局部 $d_E^2+\lambda_Ss^2$，同时限制在所属支持域内。
3. 检查位移、切向变化、曲率和到其他分支的接近程度；减小 $h$ 直到稳定。
4. 到达 port 时接入对应共享顶点；闭环同时检查位置、切向和已走弧长后闭合，避免碰到附近环就提前停止。
5. 从另一方向或另一初值复核覆盖，必要时在稀疏区追加样本。

这一追踪仅补充**已实例化边**的有序几何，不以独立追踪取代面标签拓扑主流程。靠近交汇区时，可用各自 port 的方向和局部距离剖面约束接回共享顶点，不能将多条 incident edge 强制切向连续。

定位完成后冻结共享顶点坐标的当前版本。几何精化若发现端口关联、边数量或 face 支持矛盾，返回 §5–§7；不能在拟合器内静默合并 edge id。

## 9. 阶段 F：共享端点曲线与完整面环

### 9.1 固定端点的曲线拟合

接口：`fit_shared_curves(graph, ordered_samples, provider) -> dict[edge_id, CurveRecord]`。

对有端点的三次 Bézier，只优化内部控制点：

$$
C(t)=(1-t)^3v_0+3(1-t)^2tP_1+3(1-t)t^2P_2+t^3v_1.
$$

给定弧长初始化参数 $t_i$ 后，内部控制点通过最小二乘求解，端点不会因权重而漂移：

```python
def fit_cubic_fixed_ends(points, t, v0, v1, weights):
    b0 = (1.0 - t) ** 3
    b1 = 3.0 * (1.0 - t) ** 2 * t
    b2 = 3.0 * (1.0 - t) * t ** 2
    b3 = t ** 3
    A = np.column_stack((b1, b2))
    rhs = points - b0[:, None] * v0 - b3[:, None] * v1
    w = np.sqrt(weights)[:, None]
    interior, _, rank, _ = np.linalg.lstsq(w * A, w * rhs, rcond=None)
    if rank < 2:
        raise ValueError("insufficient curve samples")
    return np.vstack((v0, interior[0], interior[1], v1))
```

`weights` 非负且须有足够内部参数覆盖。曲线更新后再沿当前曲线求有序、单调的点参数，迭代参数和控制点；不允许最近点对应跳到不相干分支。超过误差时按局部残差／曲率自适应分段，内部段共享拼接点；光滑处施加按段参数长度缩放的 $C^1$ 或 $G^1$ 约束。

多个 Bézier 段组合成一条 piecewise B-spline／NURBS 作为一个逻辑 edge 的几何。内部拟合结点不自动创建 B-rep vertex；三次分段的重复 knot 表示和控制点转换需验证端点、导数及参数范围一致。

### 9.2 闭环、解析类型与全局几何验收

无必要顶点的闭环优先检测圆／椭圆等解析类型，再选择有理曲线或周期 B-spline；不把单条非有理三次 Bézier 当作精确圆。若用分段 Bézier，闭合辅助锚点与首尾控制约束独立管理，不受 $d_V$ 吸引。

直线／圆弧等解析候选必须同时满足分支覆盖、固定端点和场误差要求；拟合不相容时使用更一般曲线，而不是移动共享顶点迁就解析类型。self-loop 复用同一个必要顶点的两次端点使用，并保留两个不同 port。

拟合验收包括：到有序支持点的双向覆盖、$d_E(C(t))$、$|s(C(t))|$、端点一致性、误接近其他分支、自交及无关边相交。只优化 $d_E(C(t))$ 可能让整条曲线塌到另一条正确边上，必须保留实例支持与覆盖约束。

若进行一次全局共享顶点／曲线精化，顶点作为统一变量同时影响全部 incident edge；完成后冻结所有共享曲线及顶点版本。曲面拟合阶段不能独立移动其中任何一条边。

### 9.3 补全顶点区域并遍历面环

接口：`complete_face_regions_and_cycles(mesh, graph, curves) -> FaceRegions`。

先把精化边界嵌回其原支持区域：在局部三角形上追踪并切分曲线，所有相邻面引用同一组边界节点；允许网格边界在几何预算内贴合共享三维曲线，再对面内部进行 SDF 投影。不能无约束地把曲线搬到最近但不同的表面层。

顶点带中依据 ports 的环绕顺序补全 face 扇区，不能保留未归属洞口。为每次有向边使用创建 coedge，按“同一 face 位于一致一侧”的半边规则找下一条使用，直到闭合。`next` 来源于表面扇区顺序，不能通过三维最近端点排序猜测。

```python
for face_id in face_ids:
    uses = oriented_uses_owned_by(face_id)
    while exists_unvisited(uses):
        start = choose_unvisited(uses)
        cycle = walk_next_use_in_same_face_sector(start)
        require_closed_without_repeated_use(cycle)
        save_boundary_cycle(face_id, cycle)
```

无顶点单边闭环可单独形成一个 boundary cycle；不能像现有某些 wire 兜底路径那样要求至少两条边。一个 face 可有多个环；内外环在 UV 或有明确拓扑约定的参数域中判断，不能按三维包围盒或环长度排序替代。

若精化后边界穿越无关面、改变孔洞或与扇区顺序矛盾，返回拓扑／曲线阶段；不要通过重新枚举空间曲线闭环覆盖已有 face 身份。

## 10. 阶段 G：面点参数化与固定整条共享边界

### 10.1 从已恢复面区域获得初始化

接口：`initialize_face_surfaces(face_regions, curves, provider, model=None)`。

复用 [`rebuild.py::sample_faces`](../rebuild.py) 的 `[F,N,6]` 点／法向及 `[F,6]` bbox 格式，从重建面区域采样，而不是读取 `surface_face_id` 真值。保留窄面、高曲率和边界内侧覆盖；尖锐边附近使用各自 face 内侧法向。

已有网络仅用于曲面初值。直接调用面分支，避免 `model.inference()` 自动调用边分类／预测分支替换已恢复的拓扑：

```python
# 设计片段：模型加载、device、采样和字段检查由适配器负责。
data = {
    "face_sample_points": face_points,
    "face_input_bbox": face_bbox,
    "face_attn_mask": torch.zeros((num_faces, num_faces), dtype=torch.bool,
                                  device=face_points.device),
}
with torch.no_grad():
    encoded = model.encode(data, True)
    decoded = model.decode_face(encoded, None)
    grids = denormalize_coord0516(decoded["face_norm"], decoded["face_bbox"])[..., :3]
```

`denormalize_coord0516` 和网络约定来自 [`rebuild_model.py`](../rebuild_model.py)。可借鉴 `create_surface()` 将规则点网格拟合为支撑曲面，但这只是初始化，尚不保持共享边界。需另行支持单 face 输入，不能继承 `rebuild.py` 对 `<2 faces` 的跳过逻辑。

没有可用网络时，先对点云拟合可行的平面、圆柱等解析初值；一般自由曲面采用具有 UV 初值的样条支撑面。根据拟合残差、覆盖和拓扑域选择，不读取源 STEP 的支撑面。初始化失败作为独立状态返回。

### 10.2 建立 UV 与曲线参数对应

对内部采样点求 $\xi_k=(u_k,v_k)$；对每条共享边的主参数 $t$，求该 face 上的 pcurve 初值 $\gamma_{fe}(t)$。可以用 `GeomAPI_ProjectPointOnSurf` 枚举投影候选，再结合相邻样本的 UV 连续性、周期展开、所属区域和环顺序选择，不能逐点只取无约束的最近 UV。[OCC 曲面投影接口](https://dev.opencascade.org/doc/refman/html/class_geom_a_p_i___project_point_on_surf.html)。

投影后的三维点不能替换 $C_e(t)$；它只是曲面参数对应初值。二维 pcurve 在不同 face 上各不相同，但三维共享曲线只有一条。

判断面域是兼容四边盘、一般裁剪区域还是周期区域。含孔洞时保存多个 UV 环；周期面保存整数周期偏移和展开区间。若一个 face 的支撑面需要周期结构，在初始化阶段就规划，不能到最后随意调用 `SetUPeriodic()` 试图补救不相容控制网格。

### 10.3 兼容四边面：锁定全部边界控制点

对夹持的张量积 Bézier／B-spline 曲面，统一四条逻辑边界的方向、次数、结点和参数区间，直接固定四周控制点，仅优化内部控制点。若边界含有理圆弧，还需使权重和齐次控制点相容。

例如底边 $C_b(u)=\sum_iB_i^m(u)Q_i$，曲面 $S(u,v)=\sum_{i,j}B_i^m(u)B_j^n(v)P_{ij}$，令 $P_{i0}=Q_i$ 可恒等保持整条底边。只固定四个角点不能保证其他边界位置。

升阶和插结点可统一相对边基函数而不改变曲线；逻辑侧边包含多个 edge 时保留各自参数区间和必要顶点。四边构造只用于满足相容条件的面，不将任意多边、带孔或周期面强行填成一个四边面。

### 10.4 一般裁剪面：曲面控制点与 pcurve 的交替约束求解

固定共享曲线 $C_e$，未知量为曲面控制点 $P$、内部 UV 和 pcurve。给定 UV、结点及固定 NURBS 权重时，样条对控制点是线性的。令 $A$ 为内部样本矩阵，$B$ 为边界样本矩阵，$R$ 为平滑项，求解：

$$
\min_P\|AP-X\|_W^2+\lambda\|RP\|^2,
\qquad BP=C.
$$

以等式消元／零空间法或 KKT 求解，不能只把边界权重设得很大：

$$
\begin{bmatrix}A^TWA+\lambda R^TR&B^T\\B&0\end{bmatrix}
\begin{bmatrix}P\\\mu\end{bmatrix}
=\begin{bmatrix}A^TWX\\C\end{bmatrix}.
$$

使用秩揭示分解检查 $B$ 的冗余、冲突和病态；KKT 需合适的稀疏不定系统求解器，不手写矩阵求逆。固定权重时的线性求解之后，再以受限步长更新内部 UV 和 pcurve，检查环顺序、孔洞和周期跨越；每次 UV 更新后重新满足边界约束。

SDF、可靠法向及双向覆盖可作为内部拟合的非线性残差，用受约束 Gauss–Newton／SQP 或交替线性化优化。残差只在**有效裁剪域内**评估，不强迫孔洞和外环外的支撑面延伸部分也落在 SDF 零面上。

必须检查整个曲线的边—面残差：

$$
r_{fe}(t)=S_f(\gamma_{fe}(t))-C_e(t).
$$

初始边界采样通过不代表全曲线通过。在各 knot 区间、中点和高残差区追加检查与约束点；增加控制自由度或调整 UV 后重解。只有具有有效导数／区间界时才能给连续域保证；否则报告自适应采样的最大误差及覆盖尺度。

固定边界与某个指定解析曲面／低阶样条可能不相容。优先增加同一支撑面的表示能力；仍失败则报告 `surface_boundary_incompatible`。不静默放开共享边、不独立移动相邻面的边，也不把拆成多个 CAD face 当作无影响的修复。

### 10.5 冻结边图与面验收

03 的 SDF 本身有表面近似误差，因此面内部按相应预算拟合，不要求支撑面同时严格通过固定 CAD 精度边界、并在每个内部点精确满足有误差的 SDF。边—面误差、SDF 误差和面覆盖误差分别报告。

面验收至少检查：边界最大残差、端点、双向覆盖、面内法向、UV 环方向、局部雅可比、整体折叠和自交。局部雅可比非零不保证全局无自交；约束求解器收敛也不保证 B-rep 有效。

## 11. 阶段 H：共享 OCC 实体、seam 和 STEP 输出

### 11.1 创建一次，全局复用

接口：`assemble_brep(logical_graph, curves, face_patches, cfg)`。

先为全部必要顶点创建唯一 `TopoDS_Vertex`，再为每条共享曲线创建唯一 `TopoDS_Edge`。有端点的边使用同一 vertex 对象及正确的有限参数区间；相邻 face 引用相同 edge 的不同 orientation，而不是分别从点云重新拟合、重新建 edge。[OCC MakeEdge 接口](https://dev.opencascade.org/doc/refman/html/class_b_rep_builder_a_p_i___make_edge.html)。

无生成顶点的完整闭环按最终参数化选择辅助锚点；这些是 CAD 构造需要的点，不加入生成顶点计数。若相邻周期面需要不同的切分位置，统一规划必要的 CAD 子段，保存 `logical_edge_id → cad_edge_ids`，所有关联面同步使用相同切分，不能分别截断。

### 11.2 为每个 face 附着 pcurve、构建有序 wire

建议按以下顺序组织，具体 pythonocc 7.7 重载在实现时以本地绑定为准：

1. 用拟合支撑曲面创建空的 `TopoDS_Face`，避免无意保留自动生成的自然边界。
2. 给每条相关共享 edge 附着该 face 上的 `Geom2d_Curve`，设定与三维曲线一致的主参数范围。
3. 按已恢复的 coedge 顺序、方向和周期展开建立 wire，加入 face；根据有效 UV 域检查内外环和面方向。
4. 完成全部相关 pcurve 后，检查三维／二维参数对应并调用必要的 `SameParameter` 修复，再复核残差及实体容差。

`BRep_Builder.UpdateEdge(edge, pcurve, face, tolerance)` 可附着单条 pcurve；同一 face 上的 seam 需要对应的两条 pcurve 重载。[OCC Builder 接口](https://dev.opencascade.org/doc/refman/html/class_b_rep___builder.html)。本次已核对本地 pythonocc 7.7 绑定具有这两类重载。

`BRepLib.SameParameter` 用于协调边的三维曲线和二维表示参数，不会使一张不包含该曲线的支撑面自动经过它；也不能只设置 SameParameter 标志就当作校验通过。[OCC 参数协调接口](https://dev.opencascade.org/doc/refman/html/class_b_rep_lib.html)。修复后比较曲线、顶点、pcurve 与容差的变更，不允许超出预算。

### 11.3 seam 后处理的具体分工

周期 face 的 UV 域规划先确定周期方向和切缝位置，再生成 seam 曲线及两侧 pcurve。新增 seam 在同一 face 内正反使用两次，可关联已有必要顶点或新增辅助锚点；位置不必与原 STEP 相同。

以 07 为例，逻辑生成图应为 `F=3,E=2,V=0`。圆柱侧面的参数化会引入 seam，并为圆环选择相容起点；最终 CAD 顶点／边数量可以增加。验收时先过滤后处理 seam 和辅助锚点，再比较逻辑图；直接比较完整 STEP 的原始 V/E 数量会错误判失败。

可以借鉴 `FixMissingSeam()` 的使用，但只能在支撑面、周期域、边界和 pcurve 已相容时尝试受控修复。修复记录每个新增／替换实体及逻辑来源，不能借此删除原有面间边或任意放大容差。

### 11.4 shell、solid 与读回检查

根据共享 edge 使用和 face 朝向装配 shell，再区分外壳与内腔壳构造 solid。共享边已一致时优先直接装配；sewing 只作受控修复，前后比较逻辑关联、共享边几何及容差。

至少检查：普通面间 edge 两次反向使用、seam 同面双侧使用、每条 wire 闭合、shell 无意外自由边、实体方向和体积、CAD 有效性、再次三角化是否成功。使用 `BRepCheck_Analyzer` 并增加几何误差检查，不能仅以 `IsValid()` 作为场重建准确性指标。

导出 STEP 后重新读取，检查逻辑拓扑映射、几何误差、体积和单位；如需回到原始尺度，对整个实体使用 `normalized_to_world`，同时换算误差预算，不能分别缩放三种场或各面。

## 12. 现有代码的复用边界

| 现有代码 | 可以复用 | 必须调整或禁用 |
|---|---|---|
| `evudf/io.py::load_fields` | 三通道、尺度、空集合契约 | 增加 provider；重建不得读取隐藏真值作拓扑输入 |
| `evudf/distance.py` 的三个 oracle | 真值场实验、追加验证查询 | 包装为只返回场的接口；不返回原 edge id／最近点给主算法 |
| `clustering.py::hierarchical_segmentation` | 邻接、诱导子图和连通分量实现 | 双带、内部漏穿越细分、小核心保留、稳定性与显式失败 |
| `rebuild.py::sample_faces` | 重建面区域的点云／法向和 bbox | 单 face 支持；输入改为传播补全后的区域 |
| `rebuild_model.py::encode/decode_face` | 面支撑网格初值 | 不调用边分类／预测分支覆盖已有图；解码后另加边界约束 |
| `brep_post/utils.py::create_surface` | 规则点网格到支撑面的初值 | 不把普通拟合、周期启发式当作边界／周期有效性保证 |
| `create_edge` | 曲线拟合思路的参考 | 共享曲线已经拟合后不再次从点重建，以免移动端点和边 |
| `create_wire_from_ordered_edges` | 部分 OCC wire 组装调用 | 当前仍按最近端点贪心串接，并跳过单边环；用已知 coedge 顺序替换 |
| `create_trimmed_face_from_wire` | pcurve／seam 修复的参考 | 当前可修改几何／拓扑，且周期场景可能替换为自然 wire；不能原样接到冻结边界流程 |
| `construct_brep.py` 中的 `optimize` 调用 | 优化组织参考 | 当前会调整边或面的位置／尺度；冻结后改为只优化曲面和 pcurve |
| `get_solid`、STEP I/O | 内核操作和读写组织 | 增加共享实体、内腔、多重边及修复前后相容性检查 |

复用不等于直接调用旧总入口。尤其已有名称为 “ordered” 的 wire 函数内部仍会重新寻找最近端点；新的主流程必须使用自己恢复的顺序，才能避免改变拓扑。

## 13. 总入口、配置、缓存与可复现输出

### 13.1 总流程伪代码

```python
def reconstruct(provider, cfg):
    mesh = extract_surface(provider, cfg.surface)
    mesh, bands = build_bands(mesh, provider, cfg.bands)
    seeds = extract_face_seeds(mesh, bands, cfg.seeds)
    require_stable_nonmissing_seeds(mesh, bands, seeds, provider)

    complex_ = build_band_complex(mesh, bands, seeds)
    arrivals = propagate_arrivals(complex_, seeds)
    interfaces = extract_interfaces(complex_, arrivals, provider)

    vertices = localize_vertices(provider, bands.vertex_regions)
    graph = instantiate_edges(interfaces, vertices, bands)
    graph = resolve_ports_and_short_edges(graph, mesh, provider)
    validate_logical_graph(graph, mesh, provider)

    samples = refine_all_edge_geometry(graph, provider)
    curves = fit_shared_curves(graph, samples, provider)
    freeze_shared_geometry(graph, curves)
    regions = complete_face_regions_and_cycles(mesh, graph, curves)

    patches = initialize_face_surfaces(regions, curves, provider, cfg.face_model)
    patches = fit_surfaces_with_fixed_boundaries(patches, regions, curves, provider)
    require_surface_boundary_acceptance(patches, curves, provider)

    solid, cad_mapping = assemble_brep(graph, curves, patches, cfg.cad)
    verify_export_and_roundtrip(solid, graph, curves, patches, cad_mapping, cfg)
    return solid, graph, curves, patches, cad_mapping
```

所有 `require/validate` 都返回阶段化报告；失败时保留中间文件和回退原因。只重算受影响的局部区域和后继阶段，不能在异常后全设单 face 再继续标记成功。若局部调整达到资源上限，则输出阶段失败及诊断，不伪装成完整 B-rep。

### 13.2 配置按误差来源拆开

建议配置分组：

```text
field:
  provider, channel_order, metric_scale, presence_policy, saturation_policy
surface:
  initial_resolution, max_resolution, surface_error, orientation_checks
bands:
  tau_edge, tau_vertex, local_refinement, max_cell_radius, max_depth
seeds:
  threshold_sweep, stability_levels, keep_small_candidates
propagation:
  edge_length_weights, all_local_sources, ambiguity_tolerance
localization:
  trust_radius, query_shell_scales, gradient_validity, residual_budget
curves:
  fit_tolerance, max_segments, closed_curve_types, intersection_checks
surfaces:
  initialization, degree, knot_budget, boundary_tolerance, uv_constraints
cad:
  max_tolerance, repair_policy, seam_policy, roundtrip_checks
runtime:
  query_batch_size, cache_dir, stage_timeout, refinement_memory_limit
```

区分 $\epsilon_{field}$、$\epsilon_{surface}$、$\epsilon_{curve}$、$\epsilon_{edge-face}$ 和内核容差。候选阈值 $\tau$ 控制覆盖，不是最终精度。每项预算要能被对应 provider 的误差和网格尺度支持，不能在 03 的近似 SDF 上统一填一个 $10^{-7}$ 宣称高精度。

自适应过程保存实际 query 位置、尺度、阈值调整、网格细分级别及误差；同一个结果必须能从缓存复现。场变化、seed 合并／分裂或曲线重新协调后，使相应后继缓存失效。

### 13.3 输出契约

```text
reconstruction/<sample_id>/
  run.json                      # 输入模式、字段尺度、版本、配置、耗时
  field_validation.json         # 若拟合／插值过字段，单列其误差
  sdf_surface.npz               # 从 SDF 得到的连通有向表面
  bands.npz                     # 各掩码、带切分映射、局部尺度
  face_seeds.npz                # 暂定 seed 与最终 face 映射
  arrivals.npz                  # 可选分片；来源和有限性信息
  logical_graph.json            # 生成层 V/E/F、ports、coedges、cycles
  refined_geometry.npz          # 共享顶点和有序分支采样
  curves.brep                   # 唯一共享三维曲线／拓扑实体缓存
  face_patches.json             # 支撑面、pcurve、UV 域的参数化记录
  logical_to_cad.json           # seam、辅助锚点和 CAD 子段映射
  reconstruction.step
  quality.json                  # 每阶段接受条件与最终读回结果
```

辅助几何缓存格式按 OCC 序列化能力实现；JSON 不能只存内核对象地址。逻辑图保留稳定 id，导出后的 OCC id 可重新编号但须具有对应映射。

只有在全部 CAD 阶段通过时写 `status=solid_validated`；允许 `face_seeds_ready`、`logical_graph_ready`、`curves_ready` 等中间状态，避免把一组拟合曲线称为完整 B-rep。

## 14. 实施顺序与测试门槛

### 14.1 第一阶段：先证明面分割不靠真值分块

先使用 07：三个 face、两个闭环、没有必要顶点。目标是从 SDF 表面和两个 UDF 恢复三个稳定面核心及两条边，不能读取源网格的三分块或 `surface_face_id`。

交付 provider、SDF 提面、连通性检查、自适应双带和 seed 诊断。至少加入：

- 相同几何采用不同网格分辨率、三角剖分时结果稳定。
- 对大平面三角形内部穿越的边带能补查询、切分，不能再次出现 §1.4 的中心漏采。
- 空顶点标记关闭顶点提取而保留闭环。
- 狭窄 face／薄颈／内腔不会被面积过滤或最大分量策略删除。

### 14.2 第二阶段：传播、实例、ports 与环

在 07 通过后处理 09 的多个无顶点闭环，再进入 08 和 03 的必要顶点。额外构造同一 face 对对应两个闭环、同一顶点对的多边、必要二度点、四度点、self-loop、顶点带吞短边及多源竞争案例。

要求边身份来自共同最小界面的分支和必要顶点分段；每条边有两侧 face、完整 ports 和 cycle 使用。即使总边数正确，也要检查端点、face 对、环顺序和孔洞归属。真值图匹配需允许实例编号置换。

### 14.3 第三阶段：几何精化与固定边曲面

先完成直线／圆环／有限圆弧和分段 Bézier 的硬端点拟合，再扩展一般样条。验证同一逻辑顶点被所有相关边引用，无顶点环的辅助锚点不受 $d_V$ 约束，拟合内部结点不增加必要顶点数。

曲面阶段从平面、圆柱及兼容四边面开始，再加入带孔裁剪面和自由曲面。为“仅固定角点但边界偏离”的反例建立检查；要求所有关联面通过同一条三维边的残差验收。检查精化边界重新嵌入后，区域与 wire 顺序没有变化。

### 14.4 第四阶段：CAD 装配与真实输入替换

先验证 07 的非 seam 逻辑图和补 seam 后完整 CAD 图的对应，再验证多个环、高价顶点和空腔。加入修复器试图删除短边、增加容差或替换周期自然 wire 的检测，确保这些修改不会被静默接受。

最后依次替换 provider：CAD 真值场 → 可控噪声／偏差场 → 当前 NPZ 拟合场 → 生成场。每次分别报告场近似、网格提取、拓扑与几何误差，不将生成误差和后处理错误合并成一个成功率。

### 14.5 最终评价必须分层

| 层次 | 指标 |
|---|---|
| 输入场 | 与参考／留出点的分层误差、饱和范围、存在性、梯度有效率 |
| 表面与带 | SDF 表面误差、分辨率稳定性、窄面／短边覆盖 |
| 面核心 | 欠分割、过分割、缺种子及人工 GT 分块泄漏检查 |
| 逻辑拓扑 | V/E/F 匹配、edge–vertex、edge–face、coedge 顺序、闭环和孔洞 |
| 点／曲线 | 顶点位置、双向曲线覆盖、端点误差、自交与错误跨分支 |
| 参数曲面 | 整条边—面误差、面内覆盖、UV 有效性、折叠／自交 |
| CAD | 有效 face/shell/solid、共享实体、seam 映射、容差、STEP 读回 |

当前可以接受四份输出作为上述真值场实验的数据起点。最优先实现的是 **provider、SDF 表面连接、自适应隔离带和 seed 稳定性检查**；本次诊断已表明，直接将保存网格和中心阈值接到旧分割器，会在进入标签传播之前产生错误拓扑。



