# 隔离带多标签三场 B-rep 重建：算法与当前实现

文档更新：2026-11-01；重建代码核对：2026-10-31，模块版本 0.3.0。本文合并原 `postprocess.md`，按当前 `band_brep_reconstruction/bandbrep/` 解释输入、拓扑、几何、CAD 和缓存。安装、CLI 参数及完整预设表维护在[模块 README](../band_brep_reconstruction/README.md)，历史成绩维护在[实验主文档](ThreeFieldReconstructionExperiments.md)。

这是与原 DualBrep 学习式参数化并列的确定性重建路线。运行时不导入上级项目，复用源码放在 `vendor/`，来源见 [PROVENANCE](../band_brep_reconstruction/PROVENANCE.md)。算法只接收场值、梯度和存在性；CAD 实例 ID、最近点和真实邻接供离线评分，不能进入重建。旧多源竞争、等代价界线和前置 port 流程已移除。

## 1. 输入与几何目标

输入固定为同坐标、同 metric 尺度的三个通道：

$$
F(x)=[s(x),d_E(x),d_V(x)],\qquad
 d_E(x)=\inf_{y\in E}\|x-y\|,\quad d_V(x)=\min_{v\in V}\|x-v\|.
$$

`s` 内负外正；`E` 是非 seam 的有效面间边集合；`V` 是其必要顶点。不同面之间的相切、共面边仍保留，完整闭环可以没有生成顶点。seam 的参数切口和辅助锚点由后处理建立，不要求复原源 CAD 的编号或切口位置。当前预处理的闭环锚点规范及保守保留范围见[预处理规范](ABCEdgeVertexUDFPreprocessing.md)。

理想集合 `V ⊆ E ⊆ S` 给出 `|s| ≤ dE ≤ dV`，但混用近似条件 SDF、不同截断或学习误差后不能不带误差预算地应用。距离场也不能区分完全重合的不同实例；非流形、退化及重合多重结构不在普通闭合实体的支持范围。

### 查询接口

代码入口为 [field_provider.py](../band_brep_reconstruction/bandbrep/field_provider.py)：

- `evaluate(xyz)` 返回 `FieldEvaluation`，包括 `[N,3]` 场值、有效／饱和标记和误差信息。
- `sdf(xyz)` 可独立查询 SDF；`gradient(xyz, channel)` 返回梯度和有效标记，基础实现用中心差分。
- `presence()` 分别声明边和顶点为 `present / absent / unknown`。`build_bands()` 拒绝 `unknown`，阈值漏检不能作为不存在的证据。
- 截断归一化输入需先还原距离及梯度尺度；`reconstruct()` 对声明归一化的 provider 检查相关阈值是否到达截断平台。

| Provider | 当前行为 |
|---|---|
| `CadOracleProvider` | 从 STEP 构造 CAD 距离查询，提供 oracle 存在性；源拓扑封装在 provider 内 |
| `SampledFieldProvider` | 读取 `fields.npz` 和 `metadata.json`，使用移动局部仿射插值；留出位置报告误差，不提供连续场认证 |
| `CallableFieldProvider` | 接入自定义／解析查询函数，调用者声明存在性、尺度和输入指纹 |
| `NeuralFieldProvider` | 加载三通道 MLP，使用 autograd 梯度；拟合入口为 `fit_fields.py` |
| `RecordingProvider` | 按配置记录查询；不是新的场来源 |

当前生产数据由 [veudf](../veudf/README.md) 生成，schema 是 `edge_vertex_mesh_pools_v1`，字段为 `[s_mesh,dE,dV]` 与分池 `.npy`。上表的 `SampledFieldProvider` 仍服务历史 NPZ，`CadOracleProvider` 使用内部冻结副本；二者没有因数据模块替换自动变成生产点池输入。后续需按生产场接口单独接入。

`benchmarks/alignment_fields.py::AlignmentField` 是历史实验输入：精确 CAD E/V 加直接或原生网格插值 SDF，E/V 查询超出 `|sdf|≤0.02` 时失败。它没有注册到主 CLI 的 `--provider`，不能把实验 worker 等同于公开重建入口。

## 2. 阶段、代码与产物

编排器为 [reconstruct.py](../band_brep_reconstruction/bandbrep/reconstruct.py)。初始化检查后顺序执行：

| 阶段 | 主要模块 | 产物 |
|---|---|---|
| `surface` | `surface_mesh.py` | `sdf_surface.npz` |
| `bands` | `bands.py`、`vertex_domains.py`、复制的两阶段分割 | `bands.npz` |
| `seeds` | `face_seeds.py` | `face_seeds.npz` |
| `graph` | `topology.py` | `topology_labels.npz`、`incidence.json` |
| `vertices` | `vertex_localization.py` | `vertices.json` |
| `curves` | `edge_geometry.py`、`curve_fitting.py`、`geometry_refinement.py` | `logical_graph.json`、`curves.json`、`refined_geometry.npz` |
| `postprocess` | `postprocess.py`、`boundary_cycles.py` | `cad_graph.json` |
| `surfaces` | `surface_fitting.py` | `face_patches.json` |
| `cad` | `occ_assembly.py` | `logical_to_cad.json`、`reconstruction.step` |

`--stop-after` 只支持 `seeds / graph / vertices / curves / surfaces / solid`，其中 `solid` 包含最终 CAD 装配与 STEP 读回。不能直接指定 `bands` 或 `postprocess`。各阶段记录于 `run.json`，`quality.json` 保存完成或失败诊断。

`logical_graph_ready` 仅表示无方向关联已输出，可以含未解决的边。只有 `solid_validated` 表示 CAD 和读回验收完成；磁盘上存在 STEP 或 CLI 某个实验任务退出 0，都不能替代这个标记。STEP 有效性也不等于恢复了全部真实实例。

## 3. 固定表面与两阶段面核心

### 3.1 SDF 提面

`surface_mesh.py` 在 `bounds` 的均匀格点上查询 SDF，使用 PyMCubes；缺少它时尝试 `skimage.measure.marching_cubes`。生成网格合并舍入尺度重复顶点、去掉退化三角形，再检查三角形预算、闭合、共享边流形、半边方向与 SDF 顶点残差。

`surface_error / nonmanifold_surface / no_surface` 可触发 `r → min(max_resolution, 2r−1)` 重试；其他错误直接失败。`allow_surface_residual_exceedance` 只允许最后一档分辨率的残差告警，结构检查仍执行。`recommended` 的起止分辨率均为 257，因此没有更高档重试。

图节点始终对应三角形中心。面核心按三角形共享边连接；普通带及阈值顶点带的连通方式由 `band_connectivity` 决定。`recommended` 为共享顶点邻接，不是空间 kNN。局部顶点域按候选归属划分，不再按整个阈值带求一个连通分量。

### 3.2 两阶段分割与排除原因

`bands.py` 对最终网格的全部三角形中心查询三场，并只将 dE 送入 `vendor/dualbrep_segmentation.py::hierarchical_segmentation`：

1. 在 `dE≥face_threshold1` 的共享边图求连通分量；保留大小 `≥face_filter_size` 且平均距离 `>face_avg_udf_threshold` 的分量。
2. 对合格父分量，在 `dE≥face_threshold2` 上再次求分量。仅当至少两个子分量通过大小过滤时替换父分量；否则完整回退父分量。
3. 最终面标签连续编号。`seeds` 负责验收并落盘，不重复分割。

默认值依次为 `0.005 / 0.01 / 10 / 0.003`，与原分割函数签名默认值不同。复制源码增加了原因码和报告，标签决策有与原函数的随机图对照测试；这不证明新 dE 与原 GVD-UDF 有相同分割精度。

| 原因码 | 含义 | 几何支撑 |
|---:|---|---|
| 0 | 最终面核心 | 面内部采样来源 |
| 1 | 第一阶段低距离 | 边／顶点距离支撑 |
| 2 | 成功拆分父域中的第二阶段低距离 | 边／顶点距离支撑 |
| 3 | 第一阶段大小过滤 | 不作为边／顶点支撑 |
| 4 | 第一阶段均值过滤 | 不作为边／顶点支撑 |
| 5 | 第二阶段子分量大小过滤 | 不作为边／顶点支撑 |

当前分割不回填、不自动合面、不按 GT 面数重试。没有面核心时明确失败。

### 3.3 基础配置与推荐配置的掩码

面核心为 `labels≥0`，边带记为 `B_E`：

- `band_propagation='distance_only'`：`B_E = reason∈{1,2}`；原因 3/4/5 为不传播的排除区，此时核心不等于边带补集。
- `band_propagation='noncore'`：边存在时 `B_E = ~core`，原因 3/4/5 参与标签传播，但仍不作为边／顶点几何支撑。`recommended` 使用这一设置。
- 边不存在时边带为空；顶点不存在时顶点域为空。空集合由显式存在性区分，不能用全零距离表示。

基础顶点带 `B_V = B_E & (dV<tau_vertex)`；普通带为 `B_E & ~B_V`。局部模式按下一节替换 `B_V`，始终保持 `B_V⊆B_E`。

`tau_edge` 不定义分割边带。它仍用于边定位半径、覆盖探针及曲面可信内部点筛选，不能将它与 `face_threshold1/2` 混用。

## 4. 多标签关联与局部顶点域

### 4.1 完整收集与禁止回传

`topology.py::collect_labels()` 按组件汇总标签，与顶点域吸收的集合传播等价：

1. 每个普通带连通分量获得独立 `edge_id`，收集直接接触的全部面核心标签。两个独立圆环即使邻面相同，仍为两条边。
2. 每个顶点域收集直接接触的面核心，以及实际接触的 `edge_id`；再并入这些边的全部面标签。
3. 域内每个节点通过组件引用获得完整集合；顶点域不向普通边带回传，所以不同边不会经顶点交换标签。

实现不需要维护传播队列，也不会因为标签晚到而丢弃新标签。`BandTopology.node_labels(node)` 按实例引用返回集合，不复制稠密 face×node 矩阵。`topology_labels.npz` 和 `incidence.json` 共同保存成员、标签、跨域接触和无方向 E–F、V–E、V–F 关系。

普通边带收到恰好两个面标签时标为 `supported`，其余保留完整集合并标为 `unresolved`。关联阶段不要求两个端点、使用次数、方向或闭合面环；这些约束在几何阶段检查。边 ID 不能用面标签对或端点对代替。

### 4.2 推荐预设的局部顶点域

`vertex_domains.py` 在原因 1/2 且 `dV<tau_vertex` 的边带支撑上读取梯度，用 `x−dV·g/||g||` 投票。投票以 `max(2*localization_tolerance, 4*field_error)` 半径聚类，至少 `vertex_min_votes` 票的簇以中位数作为候选。

对候选 i，设最近其他候选间距为 gap，附近支撑点 dE 的 95% 分位数为 w，半径为：

$$
r_i=\min(\tau_V,\max(\text{vertex\_radius\_fraction}\cdot gap_i,
\text{vertex\_cut\_factor}\cdot w_i)).
$$

边带节点归属最近候选，距离小于该候选半径时进入顶点域。不同候选即使半径区域相接，也保留不同域 ID。该规则没有顶点夹角控制，不保证任意相邻结构都能分开。

### 4.3 短边推断

`infer_short_edges=True` 处理两个局部顶点域直接接触而无普通带的情形。已有边连接的顶点对跳过；候选必须有两邻面，并在两投票中心连线上 33 点满足 `dE≤2*localization_tolerance`、`|sdf|≤feature_sdf_limit`。

共享面多于两个时，代码比较各面核心到线段中部的最近距离；只有第三近距离至少为第二近的 1.5 倍，才保留最近两面，否则拒绝。通过者补建 `inferred_short` 实例，拒绝信息保存在报告中。它是有限采样的直段候选，不是通用短曲边追踪；之后仍需曲线验证。

## 5. 顶点与共享曲线

### 5.1 v-UDF 定位

`vertex_localization.py` 只使用顶点域原因 1/2 的支撑。梯度投票中位数提供初值，再在附近三维壳层追加距离查询，用 `distance_center()` 拟合：

$$
\min_v\sum_i\rho(\|x_i-v\|-d_V(x_i)).
$$

当前调用使用非加权 `soft_l1`；SDF 不进入这个优化目标，而是在中心拟合后与 dE/dV 一起验收。最多四轮缩小查询半径。距离残差超过 `localization_tolerance` 报 `vertex_mixed_support`；中心三场检查失败报 `vertex_geometry_invalid`。不能把域内全部混合顶点观测强行拟合成一个有效中心。

### 5.2 e-UDF 定位与实例内排序

`edge_geometry.py` 要求每条边恰好两个邻面、至多两个不同顶点域。正常边至少三个距离支撑点，沿 e-UDF 梯度做最多 32 次阻尼定位，位移受 `max(3*tau_edge, 4*surface_tolerance)` 限制；至少三个且不少于 80% 的点须通过 e-UDF/SDF 检查。

优先尝试直线与圆周排序，一般支持用局部 kNN 图的最小生成树主路径排序，再检查覆盖，分支或断裂会失败。这里的 kNN 用于单条已实例化曲线的几何排序，不用于表面拓扑传播。两个端点的推断短边可从端点直段初始化。

一个不同顶点在几何阶段变为两次端点使用；没有生成顶点的支持需要观测闭合，不能自动补齐大缺口。顶点域中的全部低 dE 点也不能同时当作每条入射边的支撑。

### 5.3 固定端点与拟合验收

`curve_fitting.py` 依次尝试直线、启用 `arcs` 后精确通过固定端点的圆弧、适用的闭合圆，最后回退分段三次 Bézier。分段拟合固定拼接点，并施加按参数长度缩放的内部 C1 约束；拼接点不是新增 B-rep 顶点。

单段固定端点约束为：

$$
C(t)=(1-t)^3v_a+3(1-t)^2tP_1+3(1-t)t^2P_2+t^3v_b.
$$

拟合后检查采样曲线的 e-UDF、SDF、支持覆盖及共享端点（位移小于 `1e-8`），并执行独立结构覆盖探针检查。只最小化全局 e-UDF 可能吸附到其他边，因此支持和端点约束必须同时保留。曲线一旦冻结，邻面引用同一条三维曲线。

## 6. 面环、曲面与 CAD

`postprocess.py::orient_boundaries()` 在曲线后处理顺序、方向、外环／孔环。闭合、自环及端点使用已在 `curves` 阶段开始确定；周期 seam/pcurve 在曲面可用后由装配阶段处理。

`surface_fitting.py` 按面核心三角形面积采样内部点。存在边时优先保留 `dE>tau_edge/2` 的可信内部样本（至少 20 个才启用该筛选），共享曲线提供边界；当前没有把所有多标签带点扩充为两侧面的内部样本。

基础候选包括平面、球面及适用的圆柱；`surface_candidates='revolution'` 在基础候选失败后增加圆柱／圆锥／圆环候选。没有合格解析候选时，用 B-spline 控制网格与交替 UV 拟合，在离散边界对应上解等式约束。可选 DualBrep 面网络仅提供初值。没有边界的非解析面当前仍明确失败。

曲面边界在每条冻结曲线的 129、257、513 点网格检查，受 `boundary_tolerance` 约束；内部点拟合及 SDF 残差受 `2*surface_tolerance` 检查。有限采样雅可比非退化不证明全域无折叠或自交。`fit_four_sided_bezier()` 提供相容四边控制网格的边界锁定工具，不代表所有裁剪面都走四边模板。

`occ_assembly.py` 构造共享 OCC 顶点、边、pcurve、wire、face 及实体。周期规划将参数切口安排在边界空隙，同轴环面协调参考方向，双环圆柱／圆锥面可显式建 seam。随后检查结构、容差、共享实体和导出读回，不能靠静默合并真实边或任意放宽 CAD 容差掩盖不相容。

## 7. 配置职责与误差预算

以 [config.py](../band_brep_reconstruction/bandbrep/config.py) 为准。默认配置为历史 `65/129` 分辨率及基础开关；`recommended` 是 `257/257`、共享顶点带邻接、非核心传播、局部顶点域、短边、圆弧／回转面和周期规划的组合。配置优先级为代码默认值 < 预设 < JSON < 显式 CLI 参数。

| 参数 | 当前职责 |
|---|---|
| `face_threshold1/2`、`face_filter_size`、`face_avg_udf_threshold` | 面核心分割 |
| `tau_vertex`、局部顶点域参数 | 顶点候选与截断范围 |
| `tau_edge` | 定位／探针半径和面内部可信样本筛选，不定义传播边带 |
| `localization_tolerance`、`curve_tolerance` | 顶点定位和曲线误差 |
| `feature_sdf_tolerance` | 顶点、边支持、曲线、覆盖探针、短边的 SDF 相容性 |
| `surface_tolerance` | MC 残差、方向探测及曲面内部拟合等；不是所有阶段共用的可放宽开关 |
| `boundary_tolerance`、`cad_tolerance` | 固定曲线与曲面相容性、CAD 内核验收 |

`feature_sdf_tolerance=None` 时 `feature_sdf_limit` 回退为 `surface_tolerance`；显式值必须有限且大于零。F30 候选 `0.004` 只由实验配置设置，主库默认和 `recommended` 没有改成 0.004。

回转面／圆弧开关应与周期规划配套；已有实验中单独开启会使部分模型失败，配置校验并不自动禁止这种组合。未知键被拒绝，旧 `max_refinements / max_band_edge / stability_factors / project_face_samples` 等参数已移除。

## 8. 网格加密属于独立实验路径

`mesh_densification.py::densify_surface_mesh()` 支持 `none / uniform / local`，采用共享边中点及一致过渡剖分，保持既有分片线性表面。它返回最终网格、原三角形父 ID 和诊断，检查闭合、方向、退化、壳数、欧拉特征及面积／有向体积不变性。

当前主 `reconstruct()` 不调用该函数，`Config` 也没有加密开关。`benchmarks/run_mesh_density_experiments.py` 在独立准备阶段调用加密，保存父三角形及中心重心坐标、重新查询最终中心三场，再接入实验重建。面积过滤也是该 worker 的显式实验变体，不是主库配置。参数、缓存和结果见[加密实验报告](MeshDensificationUDFExperimentReport.md#method)。

加密能改善查询覆盖和剖分，但不会增加 SDF 的几何信息。它与重新提取更密 SDF 零面不同，也与已删除的递归近带竞争流程不同。

<a id="implementation"></a>
## 9. 缓存、验证与实现边界

`cache.py` 的键包括 provider 指纹、配置（排除 `stop_after`）、`bandbrep` 包内 Python 源码哈希及可选面模型指纹。`--resume` 核验键与已记录产物哈希，从 JSON/NPZ 恢复阶段对象。输入、几何或源码身份变化时使用新输出；历史被清理的缓存不能跳过校验续跑。文档修改本身不改变主库代码哈希。

测试覆盖集合传播不回传、晚到标签、同面对多边、两阶段分割一致性、固定端点／边界、缓存、推荐预设解析 STEP，以及条件插值与网格加密。具体运行命令见[模块 README](../band_brep_reconstruction/README.md#测试与验证)；历史版本测试数和解析通过率不替代真实模型评估。

尚未解决的范围包括窄面核心找回、错误核心修复、相接顶点的多面歧义、通用短曲边追踪、无边界非解析面、复杂自由曲面及逐实例局部重试。加权距离中心、完整面域扩展、联合顶点／曲面优化和连续边界认证属于扩展设计，不能从公式或辅助函数推断为默认流程已实现。失效机制与候选改进见[面分割分析](ThreeFieldReconstructionExperiments.md#segmentation-failures)。

完整输出保留 `strict_topology_guarantee=False`、`continuous_boundary_guarantee=False` 等边界：有限查询、局部雅可比及 OCC 有效性检查不构成连续全域拓扑、自交或误差的数学证明。原方案与合并前代码讲解全文保存在[2026-10-31 文档归档](archive/20261031_document_sources.json.gz)。
