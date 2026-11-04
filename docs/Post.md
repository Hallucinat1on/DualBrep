# 网格与 E/V 距离场的隔离带 B-rep 重建

文档整理：2026-11-04，当前说明对应模块 0.5.0。已合并两级加密实施说明，验证记录集中到[实验主文档](ThreeFieldReconstructionExperiments.md#coarse-fine-validation)。当前输入改为预处理网格与连续 E/V，不再通过 SDF 提面。安装、模型接入和 CLI 见[模块 README](../band_brep_reconstruction/README.md)，历史成绩见[实验主文档](ThreeFieldReconstructionExperiments.md)，来源见 [PROVENANCE](../band_brep_reconstruction/PROVENANCE.md)。

## 1. 输入与几何目标

外部输入为闭合、定向、已归一化的三角网格，以及两个可连续查询的 metric 距离场 `dE(x), dV(x)`。`E` 为非 seam 的有效面间边，`V` 为其必要顶点；相切、共面面间边保留，完整闭环可以没有生成顶点。CAD 实例 ID、最近点和真实邻接不进入算法。

内部 `MeshFieldProvider` 按需组合 `[s_mesh,dE,dV]`：第一通道到**原始输入网格**，用 AABB 距离及绕数符号，内负外正。SDF 用于有效带、方向和特征/曲面相容性检查，无需模型预测，不用于 MC。CAD 边与三角网格有近似偏差，不能无误差预算地强制 `|s_mesh|≤dE`。

### 查询接口

[field_provider.py](../band_brep_reconstruction/bandbrep/field_provider.py) 提供：

- `CallableUDFProvider`：接收 `[N,3] -> [N,2]` metric E/V 函数，可显式提供梯度，否则中心差分。
- `ProductionUDFProvider`：接入生产 `FieldProviderAdapter` 的 E/V、有效/饱和 mask 和 autograd；验证 condition mesh 身份，要求其 SDF 来源为 condition。
- `TorchScriptUDFProvider`：加载已冻结条件与 latent 的两通道 decoder，直接 autograd。
- `MeshFieldProvider`：内部三通道、网格距离和查询域管理；`RecordingProvider` 记录 metric 查询与 mask。

E/V 梯度由模型连续场计算，不需要预测第三/第四个梯度头。外部梯度 channel 为 E=0、V=1，内部为 SDF=0、E=1、V=2。有效 mask 描述有限值和可查询范围，饱和 mask 描述截断平台/负值裁剪等不可作为几何零集的值；不要求独立预测这些 mask。

`presence` 可以接收模型适配器的场规则结果，也可在确定性表面样本上按低 UDF 支撑数推断；不能由全部无效的查询判定不存在。absent 通道使用有限哨兵并标记 invalid，必要查询仅要求 present 通道有效。该有限支撑规则可能漏掉微小特征，需随模型评估。

模型域为同坐标 `[-1,1]` 内、`|s_mesh|≤band_width` 的训练带。必要查询若无效或饱和即失败；边定位/覆盖探针的越带移动回退，顶点壳层缩小。保留真实几何误差，不把特征投影到网格后改写目标。固定网格顶点的预测数组无法替代连续函数。

## 2. 阶段、代码与产物

编排器为 [reconstruct.py](../band_brep_reconstruction/bandbrep/reconstruct.py)。初始化检查后顺序执行：

| 阶段 | 主要模块 | 产物 |
|---|---|---|
| `mesh` | `surface_mesh.py` | `input_mesh.npz` |
| `coarse_refinement` | `mesh_densification.py`，纯几何 | `coarse_mesh.npz`（含原父三角形）、`coarse_refinement_progress.json` |
| `fine_refinement` | `field_provider.py`、`mesh_densification.py` | `working_mesh.npz`（含组合后的原父三角形）、`center_fields.npz`、`fine_refinement_progress.json` |
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

### 3.1 直接网格与默认两级加密

网格可从生产condition或已归一化网格文件读取，不自动换坐标。验证有限坐标、索引、闭合、方向、退化面、正体积与多壳嵌套；condition另核验版本、dtype和mesh hash。以原网格构建一次AABB/winding加速结构，模型condition及内部SDF始终绑定原始网格。

随后先纯几何粗加密，再按连续E/V做近场细加密。新增位置查询场值，完成后的中心字段可直接交给分割复用；共享边与过渡剖分保持原分片线性表面，详见[第8节](#refinement)。这与历史四轮单级L5不同，不提高原网格几何精度。

图节点始终对应最终三角形中心。面核心按共享边连接，默认普通带按共享顶点连接；局部顶点域按候选归属，不按整个阈值带求一个分量。

### 3.2 两阶段分割与排除原因

`bands.py` 使用经身份/坐标/顺序核验的最终中心三场；没有缓存输入时自行查询，并只将 dE 送入 `vendor/dualbrep_segmentation.py::hierarchical_segmentation`：

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

### 3.3 核心与传播掩码

面核心为 `labels≥0`，边带记为 `B_E`：

- `band_propagation='distance_only'`：`B_E = reason∈{1,2}`；原因 3/4/5 为不传播的排除区，此时核心不等于边带补集。
- `band_propagation='noncore'`：边存在时 `B_E = ~core`，原因 3/4/5 参与标签传播，但仍不作为边／顶点几何支撑。默认使用这一设置。
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

### 4.2 默认局部顶点域

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

当前调用使用非加权 `soft_l1`；SDF 不进入这个优化目标，而是在中心拟合后与 dE/dV 一起验收。最多八次尝试缩小查询半径。距离残差超过 `localization_tolerance` 报 `vertex_mixed_support`；中心三场检查失败报 `vertex_geometry_invalid`。不能把域内全部混合顶点观测强行拟合成一个有效中心。

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

基础候选包括平面、球面及适用的圆柱；`surface_candidates='revolution'` 在基础候选失败后增加圆柱／圆锥／圆环候选。没有合格解析候选时，用 B-spline 控制网格与交替 UV 拟合，在离散边界对应上解等式约束。没有边界的非解析面当前仍明确失败。

曲面边界在每条冻结曲线的 129、257、513 点网格检查，受 `boundary_tolerance` 约束；内部点拟合及 SDF 残差受 `2*surface_tolerance` 检查。有限采样雅可比非退化不证明全域无折叠或自交。`fit_four_sided_bezier()` 提供相容四边控制网格的边界锁定工具，不代表所有裁剪面都走四边模板。

`occ_assembly.py` 构造共享 OCC 顶点、边、pcurve、wire、face 及实体。周期规划将参数切口安排在边界空隙，同轴环面协调参考方向，双环圆柱／圆锥面可显式建 seam。随后检查结构、容差、共享实体和导出读回，不能靠静默合并真实边或任意放宽 CAD 容差掩盖不相容。

## 7. 配置职责与误差预算

以 [config.py](../band_brep_reconstruction/bandbrep/config.py) 为准。默认启用 L5、共享顶点带邻接、非核心传播、局部顶点域、短边、圆弧/回转面和周期规划。优先级为代码默认值 < JSON < 显式 CLI 参数；不再提供 preset 或分辨率。

| 参数 | 当前职责 |
|---|---|
| `face_threshold1/2`、`face_filter_size`、`face_avg_udf_threshold` | 面核心分割 |
| `tau_vertex`、局部顶点域参数 | 顶点候选与截断范围 |
| `tau_edge` | 定位／探针半径和面内部可信样本筛选，不定义传播边带 |
| `localization_tolerance`、`curve_tolerance` | 顶点定位和曲线误差 |
| `feature_sdf_tolerance` | 顶点、边支持、曲线、覆盖探针、短边的 SDF 相容性 |
| `surface_tolerance` | 方向探测及曲面内部拟合等；不是所有阶段共用的可放宽开关 |
| `boundary_tolerance`、`cad_tolerance` | 固定曲线与曲面相容性、CAD 内核验收 |

`feature_sdf_tolerance=None` 时 `feature_sdf_limit` 回退为 `surface_tolerance`；显式值必须有限且大于零。F30 候选 `0.004` 只由实验配置设置，当前默认没有改成 0.004。

回转面／圆弧开关应与周期规划配套；已有实验中单独开启会使部分模型失败，配置校验并不自动禁止这种组合。未知键被拒绝，旧 `max_refinements / max_band_edge / stability_factors / project_face_samples` 等参数已移除。

<a id="refinement"></a>
## 8. 两级加密、字段复用与不变性

当前入口依次执行 `mesh → coarse_refinement → fine_refinement → bands`，默认粗目标0.02、细目标0.005、E/V近场半径0.02/0.03。原始condition网格继续决定模型输入和mesh SDF，工作网格只增加后处理采样，不修复mesh与CAD E/V的表示偏差。

### 粗阶段：纯几何长边细分

`coarse_refine_surface(mesh, target_edge)` 建立共享边索引，只标记 `edge_length > coarse_edge*(1+1e-10)`。现有red/green规则对1/2/3条切边一致剖分，邻面作必要过渡；新点是共享边中点，不移动已有顶点、不平滑或投影，也不查询SDF/E/V。迭代至全局最长边达标，细密曲面不被强制一分四。

### 细阶段：原近场规则与超长边剪枝

`fine_refine_surface(mesh, provider, cfg)` 首先查询粗网格顶点/中心。对每个present通道，最长边L、顶点最小距离m、中心距离d满足 `m−L≤rho` 或 `d≤rho` 时纳入近场；学习场须同时遵守有效性、饱和和误差语义。

```text
selected_face = near_field AND face_longest_edge > fine_edge
marked_edge = belongs_to_any(selected_face) AND edge_length > fine_edge
```

共享边全局唯一处理，未选邻面仍过渡剖分。每轮检查近场达标，已足够短的边不标记。当前没有附加更紧下界、永久远场排除、主动队列或共面重网格化；这些不能混入同批次成绩。共享边索引采用有范围检查的整数编码，本轮选择与剖分复用索引，避免重复排序。

### 字段、映射与缓存

新增顶点及子面中心重新查询，不复制父标签或插值父距离。原顶点字段复用，最终中心字段携带provider指纹、坐标及顺序交给 `build_bands`；独立调用分割仍有自行查询的fallback。这是相同查询复用，不是改变近场判断。

粗阶段保存原父面，细阶段合成为 `original_parent = coarse_parent[fine_parent]`。每个子面始终留在单一原三角形内部，不跨父面重三角化。检查闭合、半边方向、非零面积、壳数、欧拉特征和面积/有向体积不变性。

产物包括 `input_mesh.npz`、`coarse_mesh.npz`、带 `parent_triangles` 的 `working_mesh.npz`、带字段身份的 `center_fields.npz`，以及两级 `*_refinement_progress.json`。进度文件不是不可变阶段缓存；完成阶段网格和中心字段按SHA-256校验，`run.json`分别记录两级报告，细阶段失败后可复用已验收且同身份的粗阶段。旧单级缓存不能直接续用。

### 配置、时间与验证职责

`Config`和JSON校验 `0 < refinement_target_edge <= refinement_coarse_edge`，默认每例 `sample_timeout_seconds=10800`，两级共用总预算，不设轮数、三角形数量或RSS硬上限。批量runner的外部watchdog还覆盖输入准备、重建和评分，具体范围按冻结协议，不把粗阶段探针时间当完整加速比。

算法测试覆盖粗阶段不查场、两级达标、短边不重复切分、过渡闭合、原父面映射、字段复用与直接查询一致、condition/SDF身份、缓存及超时。真实模型耗时、回归数量与批量结果统一见[实施验证](ThreeFieldReconstructionExperiments.md#coarse-fine-validation)和[粗目标配对](ThreeFieldReconstructionExperiments.md#coarse-fine-comparison)，本页不复制运行状态。

<a id="implementation"></a>
## 9. 缓存、验证与实现边界

`cache.py` 的键包括原网格数组/归一化、E/V 模型指纹、有效带/presence、配置（排除 `stop_after`）、`bandbrep` 包内源码及生产 mesh oracle 实现哈希。`--resume` 核验键与已记录产物哈希，从 JSON/NPZ 恢复阶段对象。输入、几何或源码身份变化时使用新输出；历史被清理的缓存不能跳过校验续跑。文档修改本身不改变主库代码哈希。

测试覆盖集合传播不回传、晚到标签、同面对多边、两阶段分割一致性、固定端点／边界、缓存、网格起步的解析 STEP，以及连续查询有效域与网格加密。具体运行命令见[模块 README](../band_brep_reconstruction/README.md#测试与验证)；历史版本测试数和解析通过率不替代真实模型评估。

尚未解决的范围包括窄面核心找回、错误核心修复、相接顶点的多面歧义、通用短曲边追踪、无边界非解析面、复杂自由曲面及逐实例局部重试。加权距离中心、完整面域扩展、联合顶点／曲面优化和连续边界认证属于扩展设计，不能从公式或辅助函数推断为默认流程已实现。失效机制与候选改进见[面分割分析](ThreeFieldReconstructionExperiments.md#segmentation-failures)。

完整输出保留 `strict_topology_guarantee=False`、`continuous_boundary_guarantee=False` 等边界：有限查询、局部雅可比及 OCC 有效性检查不构成连续全域拓扑、自交或误差的数学证明。本次合并前算法说明与两级实施说明全文保存在[本次文档归档](archive/20261104_document_sources.json.gz)。
