# ABC 顶点／边 UDF 数据预处理：EMAP 仓库核查与实现计划

本文给出从 ABC 的 STEP B-rep 构建边距离场、顶点距离场和 SDF 几何条件的独立实施方案，供扩散模型训练及后续拓扑恢复使用。本文完整规定数据目标，不依赖其他设计文档才能实施。

**结论：可以参照 3D Neural Edge Reconstruction（EMAP）的场查询、梯度投影及提取评估代码，但不能直接复用它作为 STEP → 边／顶点 UDF 真值生成器。** 主预处理应建立在本仓库已有的 Open CASCADE / DualBrep 代码上：从 CAD 拓扑筛选目标实体，再查询完整有限曲线和必要顶点的距离。EMAP 用于辅助验证和后处理对照。

核查日期：2026-10-12。已克隆并静态检查官方 [`cvg/EMAP`](https://github.com/cvg/EMAP) 仓库，固定提交为 `652f24ecc3f3cbf538928f27cc6d55dbebb360c7`。§1–13保留2026-10-12的设计核查记录及示意接口，不将历史“待实现”描述当成当前模块状态。预处理模块现已实现，实际命令和参数见[模块README](../edge_vertex_preprocessing/README.md)；§14收录2026-10-13独立数据审查。原核查未运行EMAP训练。

## 1. EMAP 仓库实际提供了什么

### 1.1 学习 UDF、提取曲线与计算训练真值是三个不同环节

EMAP 的输入为多视角图像、相机参数和二维边缘图。`Dataset` 加载 `meta_data.json`、`color/`、`edge_DexiNed/` 或 `edge_PidiNet/`；训练时沿相机射线渲染边缘，用二维边缘误差及 Eikonal 项优化 UDF。其公开入口没有要求每个空间查询点带有 CAD 距离标签。[数据加载代码](https://github.com/cvg/EMAP/blob/652f24ecc3f3cbf538928f27cc6d55dbebb360c7/src/dataset/dataset.py#L50-L153)、[训练代码](https://github.com/cvg/EMAP/blob/652f24ecc3f3cbf538928f27cc6d55dbebb360c7/src/runner/runner_udf.py#L70-L166)。

`scripts/download_data.py` 下载已有数据包；`scripts/run_ABC.bash` 使用的是 `ABC-NEF_Edge` 多视角基准。检查该提交的脚本、数据集、模型、提取及评估代码后，未发现 STEP 读取、CAD seam 过滤、必要拓扑顶点筛选，或 STEP → 空间边／顶点距离监督的实现。[下载脚本](https://github.com/cvg/EMAP/blob/652f24ecc3f3cbf538928f27cc6d55dbebb360c7/scripts/download_data.py)、[ABC 配置](https://github.com/cvg/EMAP/blob/652f24ecc3f3cbf538928f27cc6d55dbebb360c7/confs/ABC.conf)。

因此，不需要先在 ABC 渲染图上训练 EMAP，再把它预测的场当作本方案真值。有 STEP 时，直接 CAD 查询更符合所需的边界语义，也避免把图像可见性、边缘检测及拟合误差写进训练标签。

### 1.2 最关键的差异：EMAP 的 ABC 参考边只选择 sharp 曲线

`src/eval/eval_util.py::get_gt_points()` 读取 `groundtruth/obj`、`chunk_0000_feats.json` 和 `chunk_0000_stats.json`，只在 `each_curve["sharp"]` 为真时收集该曲线的 `vert_indices`，再沿相邻 OBJ 点连线插值。这里得到的是**用于评估的尖锐边点云**，不是每条 CAD 曲线的精确距离查询，也不是完整的 B-rep 顶点集。[实际代码](https://github.com/cvg/EMAP/blob/652f24ecc3f3cbf538928f27cc6d55dbebb360c7/src/eval/eval_util.py#L194-L297)。

对本方案有四个直接影响：

- 不能沿用 `sharp=true` 过滤：两个不同 face 之间即使相切、共面，也可能存在需要恢复的拓扑边。
- `vert_indices` 指向沿曲线分布的**网格顶点**，不能全部当作 B-rep 顶点来构建 $d_V$。
- 不能把 `get_gt_points()` 输出的最近点距离称为连续 CAD 边距离；离散点间距及插值误差会影响标签。
- 该函数使用适配渲染设置的 bbox 缩放和平移，不能直接与 DualBrep 的 `[-0.9,0.9]` 坐标混用。

若需要对照 EMAP 的指标，可以另建“sharp 子集评估”；完整拓扑边标签必须使用本文的统一规范。

### 1.3 可复用部分及边界

| EMAP 位置 | 实际用途 | 本方案使用方式 |
|---|---|---|
| `src/models/udf_model.py::UDFNetwork` | 非负场输出及自动微分梯度 | 可作为小规模监督拟合诊断模型；不负责 CAD 真值 |
| `extract_pointcloud.py::get_udf_normals_grid` | 分块查询规则网格，阈值内查询梯度，局部梯度 SVD 估计线方向 | 借鉴为生成场的可视化／提取基线 |
| `extract_pointcloud.py::get_pointcloud_from_udf` | 阈值选候选点，再沿负梯度移动 | 借鉴验证距离场是否能回到参考曲线；不能替代预处理距离 oracle |
| `edge_fitting/`、`merging/` | 从提取点拟合、合并线段和 Bézier 曲线 | 作为后处理基线，不把拟合端点当作 CAD 顶点真值 |
| `eval/eval_util.py` | 点云距离与精确率／召回率等评估 | 复用指标组织，替换为本文规范的真值和尺度 |
| `scripts/download_data.py` | 下载打包的多视角数据 | 原始 ABC STEP 管线另建下载／索引入口 |

具体适配注意事项：

1. `get_pointcloud_from_udf()` 名称中的 `N_MC` 表示规则网格分辨率；其主体是阈值与投影，并没有按 Marching Cubes 表生成一维边拓扑。它使用 $x' = x-d(x)\widehat{\nabla d(x)}$；曲线附近的梯度 SVD 只适合局部单分支，交汇点和最近点不唯一处需单独处理。[提取代码](https://github.com/cvg/EMAP/blob/652f24ecc3f3cbf538928f27cc6d55dbebb360c7/src/edge_extraction/extract_pointcloud.py)。
2. 网格范围硬编码为 `[-1,1]^3`；回调期望 `func(x)` 返回三个对象，梯度使用特定张量维度。慢路径包含显式 `.cuda()`，移植时应适配设备、空候选集、批大小和返回类型。
3. `UDFNetwork` 提供 `abs`、`square` 等非负输出方式。这是参数化选择，不意味着训练目标是平方距离；本文仍监督欧氏距离，单位为长度。[模型代码](https://github.com/cvg/EMAP/blob/652f24ecc3f3cbf538928f27cc6d55dbebb360c7/src/models/udf_model.py#L82-L136)。
4. `bezier_fit()` 优化全部四个控制点；端点合并使用距离聚类。它们没有实现本方案所需的“先确定唯一共享顶点，再固定端点拟合”的约束。[Bézier 拟合](https://github.com/cvg/EMAP/blob/652f24ecc3f3cbf538928f27cc6d55dbebb360c7/src/edge_extraction/edge_fitting/bezier_fit.py)、[端点合并](https://github.com/cvg/EMAP/blob/652f24ecc3f3cbf538928f27cc6d55dbebb360c7/src/edge_extraction/merging/main.py#L222-L268)。

另外，静态核查发现一个移植时应修正的维度问题：该提交的 Runner 回调返回 `[N,1,3]` 梯度，而网格路径使用 `F.normalize(grad, dim=1)[:,0]`，归一化发生在长度为 1 的轴上，通常得到各坐标分量的符号，未保持所需的单位梯度方向。应先 `squeeze(1)`，再沿最后一个坐标轴归一化，并用解析距离场验证投影；慢路径也应统一接口。这进一步说明适合借鉴算法步骤，不能未经检查直接复制提取器。[网格路径](https://github.com/cvg/EMAP/blob/652f24ecc3f3cbf538928f27cc6d55dbebb360c7/src/edge_extraction/extract_pointcloud.py#L64-L85)、[Runner 回调](https://github.com/cvg/EMAP/blob/652f24ecc3f3cbf538928f27cc6d55dbebb360c7/src/runner/runner_udf.py#L520-L541)。

预处理初版无需引入 EMAP 运行依赖。若后续复制其中实质性代码，保留其许可证及来源说明，并核对所复制文件的第三方声明。[EMAP 许可证](https://github.com/cvg/EMAP/blob/652f24ecc3f3cbf538928f27cc6d55dbebb360c7/LICENSE.txt)。

## 2. 数据目标与适用范围

### 2.1 三类场

在同一归一化坐标系内，令 $M$ 为实体表面，$\mathcal E$ 为筛选和规范化后的目标 edge 实例，$V$ 为必要顶点位置集合：

$$
E=\bigcup_{e\in\mathcal E}\ \bigcup_{k\in\mathrm{pieces}(e)}C_k([a_k,b_k]),
\qquad
F(x)=[s(x),d_E(x),d_V(x)],
$$

$$
s(x)=\operatorname{SDF}_M(x),\quad
d_E(x)=\inf_{y\in E}\|x-y\|,\quad
d_V(x)=\min_{v\in V}\|x-v\|.
$$

SDF 采用实体内负、外正，作为固定几何条件；$d_E,d_V$ 是待学习的结构通道。所有曲线使用完整有限参数区间，**包含端点**；顶点坐标来自 CAD 拓扑。多条源曲线可以组成一条规范化后的目标边，不要求预处理时重新拟合。

边场不是 GVD 距离、到尖锐边的距离、到有限边点云的距离，也不是沿表面测量的测地距离。字段名称应显式使用 `dE`、`dV`，避免继续用语义不清的单个 `udf`。

### 2.2 统一的目标规范

- 只监督两个不同拓扑 face 共享的非退化、非 seam 边。相切／共面面间边及短边保留。
- seam 留给周期曲面参数化和 CAD 后处理，既不进入 $d_E$，也不作为近边采样目标。
- 从保留边重新构造必要顶点；不直接使用全部 STEP 顶点，不把所有二度顶点删除。
- 完整光滑闭环在没有必要分段时是一条目标 edge，可以没有生成顶点。仅作参数开口的人工锚点删除；真实交汇、转折及必要分段保留。
- 不通过坐标接近合并不同拓扑实体；不通过 face 标签对相同合并空间上不同的边。
- 初版支持有效的单个闭合流形 solid，可包含空腔。多 solid、开放壳、非流形、退化边、重合边内部或不支持的同面内部结构进入明确的拒绝／扩展分支。

“不支持的 CAD”与“合法的空集合”必须区分。$E\ne\varnothing,V=\varnothing$ 是常见合法情况；通过范围检查后 $E=V=\varnothing$ 也可以是合法的无面间边样本。集合为空时，用显式存在标记和截断值表达，不能伪造一个原点顶点或零边。

## 3. ABC 输入、索引与划分

### 3.1 选择原始 STEP 数据

以官方 ABC `v00` STEP chunks 为主输入，metadata 用于追溯和分组；OBJ / features / stats 作为可选核查材料。官方按格式提供压缩包及校验清单，部分辅助格式并非每个模型都有，不能把缺少 OBJ 当作 STEP 无效。[ABC 官方数据说明](https://deep-geometry.github.io/abc-dataset/)。

从官方 [STEP 清单](https://deep-geometry.github.io/abc-dataset/data/step_v00.txt) 选择 chunk。该清单首项对应 `abc_0000_step_v00.7z`；先列出归档内容并选择小批量样本，不预设每份归档内路径结构。下载后按官方 [校验清单](https://deep-geometry.github.io/abc-dataset/data/md5.yml) 验证完整性，并记录自己的文件哈希。

ABC features 中的网格索引与配套 OBJ 对应，索引基准也不同于 OBJ 文本表示；它们不能直接映射到重新导入 STEP 后的 OCC 拓扑 id。[官方格式说明](https://deep-geometry.github.io/abc-dataset/#download)。即使使用这些文件作交叉核验，目标边与顶点仍由 STEP 拓扑产生。

### 3.2 Manifest 和数据划分

建立 `manifest.jsonl`，每条记录至少包括：

```json
{
  "sample_id": "由完整归档成员路径稳定生成的标识",
  "abc_version": "v00",
  "chunk": "abc_0000_step_v00.7z",
  "archive_member": "归档中的原始相对路径",
  "step_path": "解压后的 STEP 路径",
  "source_sha256": "源文件内容哈希",
  "group_id": "原始 CAD 文档或模型分组",
  "split": "train"
}
```

标识中保留文件后缀之外的完整来源信息，不只取前八位编号，也不假设一个 CAD 文档只对应一份几何。压缩成员若还包含 gzip，再解压至工作缓存；保留来源链。初版遇到多个 solid 时拒绝该记录，后续若拆成多个样本，同源 parts 必须属于同一 split。

训练／验证／测试按源 CAD 文档分组；可解析 metadata 文档 id 时优先采用，否则使用稳定来源 id 并记录分组局限。先去除内容重复，再固定 split；增强样本不能跨 split。建议开发阶段先人工挑选 20 个覆盖特殊结构的样本，再扩展至约 1,000 个模型的预处理试验；规模和划分比例由项目数据量决定。

## 4. STEP 导入、有效性与统一归一化

### 4.1 复用入口，补全验证

复用 [`prepare_implicit.py`](../prepare_implicit.py) 的 `read_step_file`，记录导入状态、单位转换信息和内核版本。按实际 `TopAbs_SOLID` 拓扑枚举实体，不能用 `shape.NbChildren()==1` 代替 solid 数量检查。

执行 CAD 有效性检查，并检查壳闭合、面方向、边使用、曲线表示、有限坐标、体积和包围盒尺度。后续原始 edge 使用检查仍不可省略。模型若需要修复，默认隔离到修复流程；修复后重新建立映射和全部标签，记录修复前后差异，不能只修改网格却继续沿用旧拓扑。

### 4.2 一次变换作用于所有几何

优先复用 [`occ_utils.py::normalize_shape`](../brep_post/occ_utils.py)，将形状等比例缩放至约 `[-0.9,0.9]^3`，查询域为 `[-1,1]^3`：

$$
x_{\rm norm}=a(x_{\rm raw}-c),\qquad
d_{\rm norm}=a\,d_{\rm raw},\qquad a>0.
$$

最小调整是让归一化辅助接口同时返回实际 $c,a$ 或完整 `world_to_normalized` 及逆矩阵。现有函数返回的是**归一化后** bbox，不能从它直接得到原中心。保存 CAD 单位、距离尺度以及归一化前后的范围。

当前 Python 实现使用三角化 bbox，C++ Voronoi 路径使用 `BRepBndLib`；本流程只归一化一次，之后的曲线、顶点、面网格与查询共用该 shape。不要分别运行两套流程后假定坐标完全一致。检查曲线及表面的真实范围是否仍落在查询域；粗三角化 bbox 不是精确曲面包围盒保证。

`BRepAdaptor_Curve` 和曲面三角化的世界坐标应一致。现有 `get_triangulations()` 已应用 `TopLoc_Location` 并处理反向 face；新增曲线投影若使用局部 `Geom_Curve`，须相应转换查询点和返回最近点，避免遗漏或重复变换。

归一化后记录 CAD 实体容差、顶点到关联曲线端点的间隙及边—面误差。若源 CAD 的容差已大于标签精度预算，标记 `tolerance_exceeded`，不能通过提高采样密度消除这一误差。

## 5. 边与顶点目标的构建

### 5.1 先保留原始使用信息

新增 `collect_source_topology(shape)`：建立 source face、edge、vertex 表，以及每次 `face → wire → coedge` 使用记录。每次使用保存方向、wire 内顺序和相关原始 id；使用 `get_ordered_edges()` 的遍历思路，另核对是否遗漏不支持的 INTERNAL / EXTERNAL 使用。

实体身份按 OCC 的拓扑对象及 Location 去重，忽略 Orientation；方向保存在 coedge 中。使用拓扑 indexed map 或基于 `IsSame` 的受控映射，不以浮点坐标或 Python 对象地址作为持久 id。导入顺序产生的 id 只保证在固定源文件和内核版本下可追溯，不承诺跨 STEP 重导出保持不变。

每条源 edge 同时记录 `use_count` 和去重的 `incident_face_ids`。普通流形边要求两次合法使用且来自两个不同 face；仅检查不同 face 数量为 2 会漏掉额外重复使用。

### 5.2 边筛选的执行顺序

| 检查项 | 操作 | 说明 |
|---|---|---|
| 退化边、缺失／无效曲线、非有限区间 | 初版拒绝样本 | 在删除 seam 之前检查，避免隐藏异常 |
| 合法 seam | 排除该 edge 的生成目标 | 同一 face 内相容的重复使用及参数接缝关系 |
| seam 同时被其他 face 使用，或使用次数异常 | 拒绝样本 | 不能因“在任意 face 上是 seam”就直接跳过 |
| 非 seam 边两次使用、两个不同 face | 保留 | 不检查 `sharp`，不按法向夹角删边 |
| 开放边、非流形边、其他同面内部边 | 初版拒绝样本 | 不静默删边后继续沿用原面标签 |
| 短边、相切／共面边界、闭合面间边 | 保留 | 提高覆盖或精度；资源不足时记录失败 |

复用 `is_seam_edge()` 的 `BRep_Tool.IsClosed(edge, face)` 判据，并缓存 edge–face 关联。该重载检查 face 上的参数接缝，不能替换成仅检查 edge 首尾是否闭合。已有 `get_primitives(shape, TopAbs_EDGE, True, True)` 可给出候选，但不能替代前述使用关系验证。[OCC 接口](https://dev.opencascade.org/doc/refman/html/class_b_rep___tool.html)。

可复用 [`calculate_voronoi.cpp::sample_points_surface`](../Voronoi/calculate_voronoi/calculate_voronoi.cpp) 的边—面、顶点—边关联收集思路。该函数并未按 sharp 属性、相切或共面删除普通边；退化边及异常面使用会导致处理中止。其“generatrix”过滤依据同一拓扑 face，不能解释为删除某种几何曲线类型。

### 5.3 必要顶点与闭环规范化

筛选边后重新收集端点出现记录，分别统计不同保留 edge 数量和端点出现次数。单条闭环在同一 source vertex 上出现两次，不等价于两条不同边相交。

建议采用版本化的保守规范 `keep_necessary_v1`：

1. 没有保留边关联的 source vertex 不进入 $V$。
2. 真实分支交汇、转折、不同目标边之间应保留的分段点进入 $V$；二度点默认保留。
3. 若单条完整、正则、简单闭环只有一个重复首尾锚点，局部几何平滑延续，面侧关系无变化，且没有其他真实边连接，可移除该生成锚点。保留源参数起点，仅作为 CAD 来源信息。
4. 若 seam 造成多条源曲线的人工分段，只有能验证几何延续、无真实分支及面侧使用相容时，才消除分段点并合并目标 edge 身份。无法确定时保留或送复核，不能用“度数为 2”或“面标签对相同”自动代替判断。
5. 含必要顶点的闭合结构按必要顶点分段；仅含一个必要顶点时允许 self-loop，其起止 id 相同。无必要顶点闭环使用空端点列表。

对规则 3，圆／椭圆完整周期边可先实现明确判定；一般 B-spline 闭环还要验证闭合、接缝两侧切向、正则性及无自交，不因 `IsClosed` 单独成立就移除锚点。规则 4 属于第二阶段规范化；在实现前明确记录 `canonicalization_status`，不声称已消除所有原始人工分段。

合并目标边时保留有序 `source_curve_pieces`、参数范围及方向，距离仍查询原始几何并集。同步更新 edge–face、edge–vertex、各面边界使用映射及评估真值。原始 wire 中 seam 的位置保留在来源记录；去 seam 后不能简单删掉列表元素就宣称得到合法 CAD wire。

完整圆柱的标准预期是：轴向 seam 排除，上下两个面间圆环保留；若两个环的原顶点仅作参数开口，$V=\varnothing$。seam 与圆环交点仍属于 $E$，不能在保留圆环上挖孔。

## 6. 两个 UDF 的数值计算

### 6.1 先实现 CAD 参考查询器

`query_edge_distance_reference(points, targets)` 对每个查询点、每个保留的有限曲线 piece 求最小距离，再对所有 piece 取最小。首版可使用 `BRepBuilderAPI_MakeVertex` 构造查询点，以及 `BRepExtrema_DistShapeShape` 查询点到 edge 的距离；也可对仅含保留 edges 的 compound 查询，但必须正确映射最近支撑实体到目标 id。

该参考查询属于 CAD 内核数值计算，不是符号精确解。检查 `IsDone()`、解数、有限值、最近点位置及目标区间，记录内核容差与失败原因。用于 $d_E$ 的目标 compound **不包含 face 或 solid**，否则语义会变成到面／实体的距离。[OCC 距离接口](https://dev.opencascade.org/doc/refman/html/class_b_rep_extrema___dist_shape_shape.html)。

若采用已有 C++ 中的 `GeomAPI_ProjectPointOnCurve` 思路，完整的候选集合为：

$$
\{C(a),C(b)\}\ \cup
\{C(t):t\in(a,b),\ (C(t)-x)\cdot C'(t)=0\}.
$$

对 B-spline 非光滑分段的断点也要检查，或按相应 knot 区间拆分。只投影到无限支撑直线／圆或未裁剪曲线会错误降低距离；只用一处局部最优化可能漏掉其他局部最小值。投影接口 `NbPoints()==0` 也可能表示求解失败，不能直接认定最近点一定是端点，须重试、细分或用另一查询器核对。[OCC 投影接口](https://dev.opencascade.org/doc/refman/html/class_geom_a_p_i___project_point_on_curve.html)。

返回建议结构：

```text
distance, closest_xyz, target_edge_id, source_piece_id, curve_parameter,
tie_or_ambiguous, solver_status
```

最近点可以落在端点。不同边并列最近时，标量距离仍有定义，单一最近 edge id 则不代表唯一归属；保留歧义标记，id 仅用于检查和辅助监督。不要把空间最近 edge 的分类标签当作边—面拓扑。

### 6.2 规模化路径：保守包围盒 + CAD 查询

建议第一条加速路径为 `cad_bvh`，保持与参考查询相同的曲线几何：

1. 为每条有限曲线 piece 建立包含其全部几何的 AABB，构建 BVH；曲线较长或弯曲较大时划成多个参数区间。
2. 由曲线上的已知点得到一个当前距离上界 $U$，或先查询最可能的 piece。
3. 按点到 AABB 的距离下界 $L$，从近到远访问 BVH。
4. 对不能排除的叶节点执行有限区间 CAD 距离查询，并更新 $U$ 及最近点。
5. 只有当未访问节点的下界已不可能改善当前最优值时停止。考虑包围盒和求解器数值预算，临界情况不剪枝。

**包围盒必须包含完整曲线，而非仅包住少量采样点。** 可使用内核的保守几何包围盒，或对满足条件的样条使用控制多边形界；显式加入数值余量。若只能得到经验近似盒，则不能用它做保证不漏边的剪枝。

最近若干采样点可以帮助排列候选，却不能作为“只查固定 $k$ 条边”的依据：高曲率、短边和近邻边可能让这种截断漏掉真正最近曲线。实现先使用 Python 包装验证，再根据剖析结果把循环移到 C++；不要在正确性未确认时先引入复杂 GPU 投影。

### 6.3 可选的更快近似：自适应折线距离

为每条 CAD 曲线建立带 piece id 和参数范围的折线，用点到**线段**的距离构建 segment BVH。对线段 $[a,b]$：

$$
\lambda=\operatorname{clip}_{[0,1]}
\frac{(x-a)\cdot(b-a)}{\|b-a\|^2},\qquad
d_{[a,b]}(x)=\|x-a-\lambda(b-a)\|.
$$

重复端点形成的零长度段单独按点处理；不能让除零结果进入标签。闭环补齐最后到最初的段，保留每条目标 edge 的独立 id。

若折线集合 $P$ 与 CAD 曲线集合 $E$ 满足双向 Hausdorff 界 $d_H(P,E)\leq\varepsilon_P$，则有：

$$
|d_P(x)-d_E(x)|\leq\varepsilon_P\qquad\text{对所有 }x.
$$

这给出了采用折线标签的明确误差条件。可在 B-spline knot spans 内递归细分，用可验证的曲线界控制误差；中点／四分点偏差、切向转角与最大弧长作为细分指标。**少量探针或仅测中点不能证明整个区间的 Hausdorff 上界**。缺少可靠界时，将误差标记为经验验证值，并用 CAD reference 对照，不把配置的 deflection 自动当作已证明精度。

使用方式按优先级排列：

- 用折线最近点定位候选参数，再由 §6.2 的保守分支搜索完成 CAD 查询。
- 对已具备误差界的折线，可直接输出 `distance_backend=polyline_bounded` 标签及误差上界。
- 仅有经验误差的近似输出采用独立数据版本，记录验证分位数和最大误差；不与 CAD reference 标签无标记混合。

即使折线距离误差小，近邻两条边的最近 id 仍可能变化。实例辅助标签及紧邻顶点的歧义需要单独检查。不能用很小的标量平均误差代替拓扑监督正确性检查。

### 6.4 顶点 UDF

对筛选后的 `vertex_xyz` 直接执行批量欧氏最近邻或精确 KD-tree 查询：

$$
d_V(x)=\min_j\|x-v_j\|.
$$

顶点数少时按 query batch 广播计算即可；较多时使用精确最近邻模式，避免全量 $Q\times|V|$ 内存。保留 float64 CAD 坐标计算，最终训练数组存 float32。近邻但拓扑不同的顶点不合并；必要时记录最近两点距离差作为歧义诊断。

顶点 UDF 不需要从边点聚类、不需要曲线端点拟合，也不需要训练 EMAP 网络。没有必要顶点时跳过最近邻计算，并使用 §9 的空集合契约。

### 6.5 可选最近点方向与梯度

在最近点 $p^*(x)$ 唯一、$d(x)>0$ 的位置：

$$
\nabla d(x)=\frac{x-p^*(x)}{d(x)}.
$$

因此两个查询器都可提供距离梯度辅助标签。在零集、最近点不唯一或求解器歧义处设置 `gradient_valid=False`，不要用除零后填入的零向量作 Eikonal 监督。若训练通道经过截断，饱和区梯度不能沿用原始距离场的单位梯度；标量监督与梯度 mask 分开保存。

对真实距离，在单一最近点邻域的投影可以验证 $x-d(x)\nabla d(x)$ 是否回到目标集合。这适合检查实现中的坐标／尺度错误，不用于反过来定义 CAD 真值。

## 7. SDF 条件及表面数据

复用 `get_triangulations()`、表面采样以及 `igl.signed_distance()` 的组织方式，并补充下面的要求：

1. 从同一归一化 shape 生成网格；保存每个三角形的 `source_face_id`，不能在拼接后丢失面归属。现有辅助函数只返回顶点和三角形，需要扩展返回映射。
2. 当前 `get_triangulations()` 调用中 mesher 的 relative deflection 为真；`0.001` 不能未经验证就解释为全模型绝对误差 $10^{-3}$。正式管线显式保存绝对／相对模式，并检查三角化到 CAD 的偏差。
3. 按 face 拼接的网格可能包含边界重复顶点。若 SDF 符号算法或连通性检查需要焊接网格，根据 CAD 共享边建立相容关系，按受控容差处理；不能用过大的坐标焊接距离合并狭缝或薄壁两侧。保留焊接前后的 face 映射。
4. 对实体内部、外部、空腔及薄壁区域检查符号，必要时用 CAD solid classifier 交叉验证。无效、漏面或方向异常网格不输出正常 SDF 标签。
5. 网格距离是对 CAD 表面的近似。可抽查“查询点到 face 集合的距离 + solid 内外分类”作为 CAD 参考；不要用点到完整 solid 的距离直接替代 $|s|$，内点到 solid 的距离可能为零。

边／顶点 UDF 始终查询 CAD 目标，即使 SDF 用网格近似。面网格采样点也按实际位置查询字段，不为了整齐而强制令其 CAD SDF 或边距离为零。

对不存在空集合且数值相容的样本，应近似满足：

$$
V\subseteq E\subseteq M\quad\Rightarrow\quad |s(x)|\leq d_E(x)\leq d_V(x).
$$

CAD 容差、曲线查询和表面三角化误差都进入检查预算。先保存原始违反量并定位问题，不通过强制排序三个数值掩盖错误。空集合的有限占位值及不同通道的归一化输出不直接参与这一不等式比较。

## 8. 查询点与监督点采样

### 8.1 采样控制覆盖，距离查询决定真值

为每个查询点同时计算 `sdf`、`dE`、`dV`。所谓“近边点”只是采样来源，不表示最近实体一定就是产生它的那条边；扰动后必须对完整目标集合重新求最小距离。

建议初始配置为每个 shape 约 200,000 个随机查询点，另加入少量精确目标点。以下是待校准的起点，不是 EMAP 或 DualBrep 的已验证最优参数：

| query 来源 | 数量 | 目的 |
|---|---:|---|
| `[-1,1]^3` 内均匀空间点 | 32,000 | 全域距离与空区域覆盖 |
| 表面点加多尺度三维扰动 | 48,000 | SDF 邻域及面边界附近的场变化 |
| 表面上采样点 | 32,000 | 直接覆盖隔离带／面分割实际使用的位置 |
| 保留边上的点加多尺度三维扰动 | 64,000 | 细边及曲线法向管状邻域 |
| 必要顶点加多尺度三维扰动 | 24,000 | 小体积顶点邻域、端口和交汇区域 |
| CAD 边点及全部必要顶点 | 额外少量 | 零集定位与质量检查 |

可从标准差 `{0.0002, 0.001, 0.005, 0.02}` 的混合开始，均为归一化坐标长度；不同尺度配额显式记录。加入与局部短边长度／顶点间距相关的细尺度采样，避免固定噪声使相邻小特征的监督全部混在一起。

超出查询域的扰动点通过重采样处理，不逐坐标 clamp 到边界，以免制造边界点质量。$V=\varnothing$ 时，将近顶点配额按预定规则分给近边和表面查询；$E=\varnothing$ 时再分给空间和表面查询。记录实际配额，不对空数组调用随机采样。

阈值检测依赖低维集合附近的覆盖，仅均匀体采样效率较低。边附近与顶点附近都必须有独立配额；不能认为近边采样自然足以覆盖全部顶点。

### 8.2 监督点云与实例覆盖

复用 `sample_edge_points()` 中 CAD 求值和切向计算的框架，但改为：

- 输入筛选后的目标 edge pieces，输出 `xyz`、单位切向、目标 edge id、source piece id 和参数。
- 分配“每条 edge 最小数量 + 剩余按弧长比例”的预算；额外检查短边及完整闭环的覆盖。最低覆盖若超过预算则提升预算或记录资源限制，不让比例取整产生零样本边。
- 使用弧长采样或自适应细分后的弧长重采样；现有等参数采样在非均匀参数化曲线上可能严重失衡。
- 闭环按半开参数区间布点，避免首尾重复耗尽样本；曲线几何距离查询仍使用完整闭区间。
- 单独保留每条边的端点／必要顶点检查点；切向不可定义时设置 mask，不能调用归一化后产生 NaN。
- 顶点监督直接保存全部必要顶点，不通过随机采样丢弃少数顶点；批处理时以 padding mask 或变长结构表示。

表面点按面积采样并保证小 face 最低覆盖，保存 xyz、法向和 face id。法向、切向等方向值不作为顶点的伪“法向”；顶点输入若需要特征维度，使用独立的类型编码和 mask。

### 8.3 不继承 GVD 站点修正

原 C++ 的端点避让、随机微扰、删近顶点站点、面点避边服务于 Voronoi 构造。新距离目标不挖掉端点邻域，不要求生成 `voronoi.ply`，也不因 GVD 三角形数不足拒绝本来合法的样本。

通用空间和表面点可以靠近 seam；此时正常计算到保留 $E,V$ 的距离即可。只有查询点确实落在保留边上时才应有 $d_E\approx0$，不能因它位于 seam 就赋零。

## 9. 输出格式、空集合及数值精度

### 9.1 每个样本的输出

建议采用 `schema_version=edge_vertex_udf_v1`，目录结构如下；调试几何和加速缓存按需保存：

```text
processed/<sample_id>/
  fields.npz             # 数值数组，训练读取不依赖 pickle
  topology.json          # 目标实例、源使用关系、排除与合并映射
  metadata.json          # 来源、变换、版本、容差、采样配置、存在标记
  quality.json           # 检查结果、误差、耗时、状态
  normalized.brep        # 可选，复现数值查询的内核几何缓存
  surface_mesh.npz       # 可选，顶点、三角形及 source_face_id
  edge_samples.ply       # 可选，可视化
```

`fields.npz` 的基本契约：

| 字段 | 形状／类型 | 语义 |
|---|---|---|
| `query_points` | `[Q,3]` float32 | 统一坐标系的空间点 |
| `query_sdf` | `[Q]` float32 | 内负外正的几何条件，归一化长度单位 |
| `query_dE`, `query_dV` | 各 `[Q]` float32 | 非空集合的未截断距离；空集合占位规则见下文 |
| `query_source` | `[Q]` uint8 | 空间／表面／近边／近顶点等来源枚举 |
| `surface_points` | `[Ns,6]` float32 | xyz + 法向 |
| `surface_face_id` | `[Ns]` int32 | 与拓扑记录一致的 face id |
| `edge_points` | `[Ne,6]` float32 | 去 seam 的 xyz + 切向 |
| `edge_point_id` | `[Ne]` int32 | 每个监督点所属的目标 edge id |
| `edge_tangent_valid` | `[Ne]` bool | 切向是否可用 |
| `vertex_points` | `[Nv,3]` float32 | 全部必要顶点，行号对应 target vertex id |
| `has_edges`, `has_vertices` | 标量 bool | 集合是否非空 |
| `nearest_edge_id`, `nearest_vertex_id` | 可选，各 `[Q]` int32 | 最近实例的辅助标签；空集合为 `-1` |
| `nearest_tie_E`, `nearest_tie_V` | 可选，各 `[Q]` bool | 最近实例／位置是否有歧义 |
| `closest_edge_points`、距离梯度及其 mask | 可选 | 数值检查和辅助监督 |

`topology.json` 至少包含 source face/wire/coedge 使用表、目标 edge 的邻面和曲线 pieces、目标 vertex 的坐标与关联边、闭环／self-loop 标记、来源映射和每次删除／合并理由。无顶点闭环使用 `vertex_ids: []`；self-loop 使用 `[v,v]`，不可混淆。

保存源拓扑与目标边网络两层信息：源 CAD 的 seam 和原 wire 用于追溯，目标层明确标记不监督 seam。face 对、edge id 和 vertex id 不是同一类标签，不能只存 face 对省略 edge 身份。

### 9.2 截断、空集合与训练语义

对非空集合，基础数据保存未截断的归一化长度；训练读取时计算：

$$
\widetilde s=\operatorname{clip}(s,-\tau_S,\tau_S)/\tau_S,\quad
\widetilde d_E=\min(d_E,\tau_E)/\tau_E,\quad
\widetilde d_V=\min(d_V,\tau_V)/\tau_V.
$$

显式保存三个 `clip_*`。初步可统一用 `0.02` 检查流程，再按提取阈值及监督范围分别调节；该值是起始建议。阈值、扰动尺度、原始距离、截断后的网络输出需清楚区分。尤其生成后的边／顶点检测阈值必须小于对应饱和值。

空集合的数学距离是 $+\infty$；数值文件中相应通道填 `clip_E` 或 `clip_V`，并设存在标记为假。这是**有限占位值，不是实际未截断距离**；最近 id 为 `-1`，坐标和梯度辅助项无效。加载器依据存在标记产生全域饱和值，屏蔽定位和梯度损失，同时可保留“无目标”的标量监督。

改变训练截断值时，非空样本可由原始距离重新计算；空样本必须依据存在标记重新生成占位通道，不能机械重缩放旧占位值。推理时若需要存在标记，必须由模型预测或有独立检测规则，不能使用测试 CAD 的真值标记帮助重建。

数值查询失败与集合为空完全不同：失败样本重试／拒绝，不通过 `has_vertices=False` 或填饱和值掩盖。如果未来采用只计算窄带距离的优化，另用 `query_dE_clipped` 等明确字段及版本，不能把截断数据伪装成本文的未截断契约。

### 9.3 精度及增强

CAD、拓扑坐标及参考查询使用 float64，训练点与标量初版使用 float32。原 `prepare_implicit.py` 将全部数据转 float16；这不适合直接继承，因为单位尺度附近的坐标量化可到 $10^{-4}\sim10^{-3}$ 量级，可能掩盖短边和邻近顶点。压缩格式应通过实际量化误差检查后另行选择。

查询坐标先固定为最终存储的 float32 值，再提升至 float64 做 CAD 查询，可使标签严格对应磁盘中的查询位置；CAD oracle 仍保留高精度目标几何。零集检查另用原始 CAD 点执行，不要求量化后的边点在无限精度意义下仍严格位于曲线上。

旋转和平移对点、CAD、SDF 条件及方向同步作用，距离数值保持不变；等比例缩放还要同步缩放距离、截断及容差。非等比缩放会改变最近点和距离，不能只乘一个系数，需重新查询。所有增强属于原样本的 split。

## 10. 文件级实现计划与现有代码复用

### 10.1 新增模块

以下路径均为计划新增，不代表仓库已具备这些脚本：

| 模块 | 主要接口／责任 | 尽量复用的代码 |
|---|---|---|
| `preprocess/abc_manifest.py` | 归档索引、来源哈希、分组和 split | 新增 ABC 专用入口 |
| `preprocess/cad_targets.py` | `collect_source_topology`、`filter_edges`、`canonicalize_vertices` | `occ_utils.py` 遍历和 seam 检测；C++ 关联判据 |
| `preprocess/cad_distance.py` | CAD reference、保守 BVH、顶点最近邻 | 已有曲线 adaptor／投影调用；新增完整查询逻辑 |
| `preprocess/field_sampling.py` | 面／边／顶点采样及 query 配额 | `prepare_implicit.py` 采样框架，修正覆盖和空集 |
| `preprocess/field_quality.py` | 拓扑、距离、尺度、SDF 符号、序列化检查 | 新增跨字段检查 |
| `prepare_edge_vertex_implicit.py` | 单 shape 入口、批量调度、保存 | `prepare_implicit.py` 的读取和 NPZ 组织方式 |
| `configs/preprocess/abc_edge_vertex.yaml` | 版本化规范、采样、容差、后端 | 新增配置 |
| `tools/inspect_edge_vertex_fields.py` | 切片、目标边图及错误可视化 | 可借鉴 EMAP 的分块网格查询 |

优先小幅扩展共享辅助函数：`normalize_shape` 可选返回真实变换、`get_triangulations` 可选返回 face 映射和显式 meshing 模式。保持现有调用的默认返回格式，避免破坏旧推理流程。C++ 关联逻辑目前嵌在 GVD 构造内，应抽取或等价移植所需部分，不要求先运行 GVD。

### 10.2 单样本流程伪代码

```python
# 设计伪代码：接口有待实现，不能直接运行。
def build_sample(record, cfg):
    shape = read_step(record.step_path)
    validate_supported_solid(shape, cfg)
    shape, transform = normalize_once_with_transform(shape, cfg)

    source = collect_source_topology(shape)
    validate_source_uses_and_geometry(source, cfg)
    edges = filter_nonseam_interface_edges(source, cfg)
    targets = canonicalize_vertices_and_edges(source, edges, cfg)
    validate_target_incidence_and_tolerances(targets, cfg)

    mesh, triangle_face_id = triangulate_with_face_ids(shape, cfg)
    validate_surface_mesh_and_sign(shape, mesh, cfg)
    samples = sample_targets_and_surface(targets, mesh, triangle_face_id, cfg)
    query_xyz, query_source = sample_queries(samples, targets, cfg)

    edge_oracle = build_edge_oracle(targets, cfg)
    vertex_oracle = build_vertex_oracle(targets, cfg)
    fields = query_three_fields_in_batches(
        query_xyz, mesh, edge_oracle, vertex_oracle, cfg
    )  # 空集合显式分支；求解器错误不当作空集合

    quality = validate_fields_and_reference_subset(
        shape, targets, query_xyz, fields, cfg
    )
    require_quality_pass(quality)
    write_atomically(record, transform, targets, samples, fields, quality, cfg)
```

建议 CLI 设计：

```bash
# 以下命令在相应模块实现后才可使用。
python prepare_edge_vertex_implicit.py --step sample.step --out processed/sample --config configs/preprocess/abc_edge_vertex.yaml --distance-backend cad_reference
python prepare_edge_vertex_implicit.py --manifest manifests/abc_pilot.jsonl --out-dir processed/abc_pilot --config configs/preprocess/abc_edge_vertex.yaml --workers 4 --resume
```

### 10.3 与当前 DualBrep 数据／模型的接口

当前 [`prepare_implicit.py`](../prepare_implicit.py) 的 `query_edge_udf` 与 `query_surface_udf` 都是到 `voronoi.ply` 的距离；“edge”只描述查询点来源。其 `get_primitives(shape, TopAbs_EDGE, True)` 没有开启第四个 seam 过滤参数。因此旧 `edge_points` 和旧 UDF 均不能直接成为新真值。

[`dataset.py`](../dataset.py) 目前按旧字段读取 `surface_points`、`edge_points`、`voronoi_points`，[`model.py`](../model.py) 的归一化、增强和编码入口也依赖旧契约。接入新数据时应：

1. 新建或显式选择 `edge_vertex_udf_v1` 加载器，拒绝把旧 GVD 数据误读为新标签。
2. 支持 `vertex_points`、双结构通道、空点集和 mask；不复制顶点来冒充 Voronoi 点及其法向。
3. 为结构模型明确提供 SDF 条件以及 $d_E,d_V$ 监督，调整输出、损失、各通道尺度和存在性处理。是否联合解码 SDF 是模型设计选择，不改变这里的固定几何条件语义。
4. 保留旧加载分支服务现有权重；旧 checkpoint 的输出含义并不因新增文件而改变，不能未经适配当作新模型。

数据标签生成本身可以独立验收，不需要先完成扩散模型或完整 CAD 重建。

## 11. 批量运行、缓存与失败记录

以 shape 为进程级任务，每个 worker 自行读取 STEP 和构造内核对象，不在进程间传递 OCC 指针。距离查询和数组写出分块进行；记录每个阶段的时间、峰值内存、目标边／顶点数量及重试次数。CAD 内核、BLAS 与进程并发避免多层过度订阅。

缓存键包括 `source_sha256`、预处理代码版本、OCC/pythonocc 版本、过滤规范、归一化设置、查询配置和随机种子。`--resume` 只跳过键一致且质量检查通过的完整结果。先写临时目录，再原子提交；单 shape 超时或崩溃由父进程记录，不能留下被认为可训练的半份 NPZ。

建议错误码至少包括：

```text
read_failed / invalid_solid / unsupported_multi_solid / open_boundary
nonmanifold_use / unsupported_same_face_edge / degenerate_edge
missing_curve / invalid_parameter_range / ambiguous_canonicalization
tolerance_exceeded / unsupported_overlap / meshing_failed / sdf_sign_failed
distance_solver_failed / quality_failed / timeout / resource_limit
```

`seam_excluded` 和 `loop_anchor_removed` 属于目标规范化日志，不是样本失败；`no_target_vertices` 是合法状态。逐阶段统计接受率和拒绝原因，特别报告退化结构、短边及非尖锐面间边的比例，以便评估数据选择偏差。

先测小批量的 I/O、距离调用次数与文件体积，再确定正式规模和 worker 数。仅 `Q=200,000` 的 xyz 与三个 float32 标量就约 4.8 MB／shape，尚未计入点云、id、拓扑及缓存；百万级处理应按测得数据量规划分片，避免默认保存所有可选调试数据。

## 12. 质量检查和验收标准

### 12.1 先用可解释的 CAD 样例建立基准

| 样例 | 必须验证的结果 |
|---|---|
| 标准长方体 | 12 条边、8 个必要顶点；端点和内部段距离正确 |
| 单侧面完整圆柱 | 去轴向 seam，保留两圆环，无必要顶点；环锚点不进 $V$ |
| 两个半圆柱 face 拼成侧面 | 两条轴向面间边仍保留；不能因法向连续或同支撑圆柱而删除 |
| 两个共面 face 的真实分界 | 保留拓扑边，揭示 `sharp` 过滤的不适用性 |
| 必要二度分段及单顶点 self-loop | 保留必要顶点，端点映射正确 |
| 近邻顶点、短边、薄壁 | 无错误合并，采样覆盖及 SDF 符号正确 |
| 非均匀参数化／高曲率样条 | 自适应离散与 CAD reference 一致，不漏最近区间 |
| 仅有参数 seam、无面间边的合法形状 | 空边／空顶点契约可训练；须先通过非退化范围检查 |
| 含退化边、开放壳、额外重复使用、多 solid | 返回对应拒绝原因，不能通过删边伪装成功 |
| 含非平凡 Location 的同一几何 | 变换前后对应距离一致，排除重复或遗漏坐标变换 |

直线段可用解析点到段距离作独立基准；半径为 $R$、位于 $z=z_0$ 的完整圆环可用

$$
d(x,y,z)=\sqrt{(\sqrt{x^2+y^2}-R)^2+(z-z_0)^2}
$$

核对内核和加速器。有限圆弧另需比较裁剪端点，不能直接使用完整圆公式。

### 12.2 每份样本的检查

- **拓扑：**每条目标普通边对应两侧不同 face；使用记录、端点、源 piece 覆盖与目标映射相容。无顶点闭环、self-loop 分开检查；去 seam 不损失保留曲线几何。
- **零集：**在 CAD 保留边点上 $d_E\approx0$，在必要顶点上 $d_V\approx0$ 且 $d_E\approx0$。在不接触保留边的 seam 区域应有正的边距离，不断言所有 seam 点距离都大于某个统一常数。
- **查询：**随机空间点与近边／近顶点点分层抽查 CAD reference；重点抽查端点、高曲率、近邻曲线、参数接缝及并列最近点。
- **一致性：**检查非负、有限值、集合包含对应的距离不等式、最近点回代距离，以及未截断距离的 1-Lipschitz 性；均加入已知误差预算。
- **SDF：**检查内部、外部及空腔符号，面—边偏差与三角化误差分开统计。
- **存储：**写入后重新读取，校验 shape、dtype、字段版本、空集 mask、最大坐标／距离量化误差及数据哈希。

加速器验收不只报告平均误差，还报告最大值、P95/P99、按距离带及 edge 长度分组的误差、最近 id 歧义率和失败率。有限抽查不能证明全域误差界；只有具备保守几何界的部分才标记为有界保证。

### 12.3 建议的第一轮精度预算

以下值用于归一化坐标下的开发起点，需结合 ABC 容差分布调整并版本化：

| 项目 | 起始目标 | 验收含义 |
|---|---:|---|
| CAD reference 数值目标 | $10^{-7}\sim10^{-6}$ | 配置与交叉检查目标；不等于源 CAD 本身的几何精度 |
| 源几何相容误差预算 | $2\times10^{-5}$ | 超出时记录或进入明确的低精度配置，不能无提示放宽 |
| 可选折线几何误差预算 | $2\times10^{-5}$ | 区分有界保证与经验验证 |
| 边／顶点标签总误差目标 | $10^{-4}$ | 包括查询和存储；对细小特征可能还需更严格 |
| 表面网格到 CAD 误差目标 | $2\times10^{-4}$ | 单独验证，不直接拿 mesher 参数替代 |
| float32 序列化误差 | 小于上述预算的显著小部分 | 超出时改用更高精度保存相关数据 |

SDF 与 $d_E$ 的一致性允许误差包含表面网格预算；不能一边用较粗表面网格，一边按 $10^{-7}$ 严格检查 $|s|\leq d_E$。检测阈值应大于标签数值误差，并与局部特征间距、采样步长一起设置。

### 12.4 EMAP 只作为附加提取对照

基础数据通过 CAD 检查后，可在小批量上直接查询参考场，或训练一个监督 MLP 近似参考场，再套用经过适配的 EMAP 阈值／梯度投影提取器。分别记录“真实场的提取误差”与“网络近似后的提取误差”，避免把学习误差误认为预处理错误。

对照使用同一归一化尺度和本文目标边；若额外运行 EMAP 原始 sharp 评估，单列结果。评价既包括点到曲线误差，也包括边覆盖、必要顶点定位与实例连接正确性；点云 Chamfer 较小不自动表示 B-rep 拓扑正确。

## 13. 分阶段交付

1. **目标与参考查询：**完成单 shape 入口、源使用表、seam 过滤、必要顶点规则、空集契约，以及全量 CAD reference；通过长方体、圆柱及有限曲线解析对照。第一阶段就输出可追溯目标图。
2. **完整训练文件：**接入同坐标 SDF、三类目标采样、统一 query、NPZ/JSON 保存和读回验证，完成 20 个特殊样例与少量 ABC STEP 检查。
3. **加速与规范化扩展：**实现保守 BVH；需要时增加有误差标记的折线路径及人工二度分段规范化。对相同 query 与 reference 对照，通过后才用于扩大数据规模。
4. **ABC 试验批次：**处理约 1,000 个模型，报告接受率、全部失败类别、精度、速度和存储；人工复核被删除的 seam／闭环锚点以及保留的非尖锐边。
5. **训练接入与正式扩展：**实现新加载契约、空集合和双结构通道；小规模训练检查通过后冻结字段和规范版本，再处理更大 split。

第一版的关键交付物是：可审计的 CAD 目标图、可复核的 $s,d_E,d_V$ 数值标签、明确的空集合语义，以及不会混入旧 GVD／sharp-only 语义的数据加载接口。



## 14. 已生成数据的独立审查（2026-10-13）

### 14.1 已执行的检查范围

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

本次独立审计脚本和日志保存在 `/tmp/band-reconstruction-audit-20261013/`，关键结果均写入本文。原作者的检查记录见 [VALIDATION.md](../edge_vertex_preprocessing/VALIDATION.md) 和 [原批次报告](ThreeFieldReconstructionExperiments.md#41-预处理与早期验证)。抽样检查与现有测试不构成整个 ABC 数据集或全域连续几何的正确性证明。

### 14.2 输出数量与数值检查

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

### 14.3 与目标文档的一致性

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

### 14.4 必须处理的网格问题

`triangulate()` 为每张 CAD face 追加独立顶点，跨 face 的相同位置没有共享索引。直接在保存网格上运行 `face_adjacency`，其连通分量已经暴露原 CAD face 的分块，而且不能跨该边界传播。

| 模型 | 三角形数 | 原网格连通分量 | 原网格边界边数 | 诊断性坐标焊接后的分量／边界边 | 最大网格边长 |
|---|---:|---:|---:|---|---:|
| 03 | 31,436 | 25 | 4,586 | 1 / 0 | 1.4632 |
| 07 | 840 | 3 | 846 | 1 / 0 | 1.7999 |
| 08 | 6,464 | 21 | 4,764 | 1 / 0 | 1.2960 |
| 09 | 9,432 | 6 | 1,696 | 1 / 0 | 0.9658 |

诊断使用 `Trimesh(process=False)`，再执行 `merge_vertices(digits_vertex=7)`；此参数是坐标量化规则，**不是对任意 CAD 安全的焊接半径**。四份焊接网格均变为 watertight，未修改磁盘数据。生产路径不能无条件靠坐标焊接处理薄壁、接触或不匹配的边界采样。

此外，`mesh_deflection=0.0002` 控制曲面三角化偏差，**不限制平面上三角形的边长**。因此仍有接近形状尺度的长边。焊接后只在三角形中心查询真实 $d_E,d_V$，不做细分、不使用大小过滤，得到：

这里的参考 face 数是**原始 STEP face 数，也是当前预处理定义的逻辑目标 face 数**，没有排除相切／共面 face。`collect_targets()` 只从目标边集合中排除同一 face 的参数 seam，并删除部分不再必要的顶点；它保留相切／共面但属于不同 face 的面间边，也没有合并或删除任何 face。因此，去除 seam 不会使本表的目标 face 数减少。若将来决定把相切／共面 face 合并成一个几何区域，必须同时重新定义面、边、顶点真值并重新生成相应 UDF；不能只修改此处的参考面数。

| 模型／原始 STEP face 数＝当前目标 face 数 | $\tau_E=0.002$ 的核心连通分量数 | $\tau_E=0.005$ 的核心连通分量数 | $\tau_E=0.01$ 的核心连通分量数 |
|---|---:|---:|---:|
| 03 / 25 | 3 | 13 | 17 |
| 07 / 3 | 1 | 1 | 1 |
| 08 / 21 | 1 | 1 | 4 |
| 09 / 6 | 1 | 1 | 1 |

有顶点样本使用 $\tau_V=2\tau_E$；空顶点样本关闭顶点带。表中核心连通分量数仅是分割诊断，**不能单凭它与目标 face 数的差值推断错分了多少张面**：一个真实 face 可能被切成多个核心，不同 face 的核心也可能错误连通，另一些 face 可能没有核心。按来源 face 映射检查，03 在最后一组丢失一个 face 的核心并分裂另一个 face；08 丢失五个 face 的核心。较小阈值下则存在多个 face 核心被错误连接的情况。正式评价应把预测面区域与当前预处理保留的各个原始 face 做允许实例编号置换的一对一匹配，再分别报告无核心、合并和分裂；不能把该表当成最终面拓扑准确率。原网格按 CAD face 分块，来源 face id 只能用于离线诊断，不能输入重建算法。

这验证了文档中关于“面核心必须非空且被完整隔开”的限制：**现有距离标签正确，不代表原始 CAD 三角化配合中心阈值就能正确分割。** [隔离带方案](BandLabelTopologyRecovery.md)中粗表面与隔离带的处理是本批数据上已有证据支持的必要工作。

### 14.5 其他限制与建议改进

- **连续查询缺口：**不能把 `query_fields` reshape 成三维体素，也不能用最近散点标签代替任意位置的真实场。需要[隔离带实施接口](BandLabelTopologyRecovery.md#implementation)中的 provider。现有 `closest_E_points`、`vertex_points`、instance id 是额外真值，直接使用它们会改变“三个标量场重建”的实验设定。
- **表面样本有效数量：**32,000 个 `on_surface` query 从 10,000 个表面点重复抽取，四份实际唯一坐标仅约 9,592–9,595 个。距离标签有效，但覆盖不等于 32,000 个独立位置。后续可改为独立面积采样；评估训练拟合的留出集应先按坐标去重，防止重复点泄漏。
- **规范化范围：**`analytic_loop_anchor_v1` 仅删除明确的解析闭环锚点。一般 B-spline 闭环和人工二度分段保守保留，不能宣称已经得到与原始 CAD 参数切分完全无关的标准边图。训练、预测和评估必须使用同一版本。
- **全局几何检查：**`validate_geometry()` 只拒绝容差内的不同顶点；曲线内部重合、异常相交尚无全面认证。扩大 ABC 规模前应增加相应检查或明确数据范围。
- **误差预算：**原 `validate_fields()` 对 mesh SDF 使用约 $5.2\times10^{-4}$ 的复合容差；这不是可直接用于全部重建阶段的 CAD 装配容差。03 不能因为边／顶点标签约 $10^{-8}$ 的参考差值，就要求其 SDF 表面也达到同样精度。
- **代码版本：**03 的保存代码哈希与当前 `cc9b6c9e…` 一致；07–09 保存于 `41d8b9d8…`。本次重新查询和拓扑重建相符，但它们不是同一代码指纹的批次。正式冻结数据时统一版本或保留明确的混合版本清单；不要修改 metadata 来绕过 `--resume` 校验。

上述连续查询和网格接入问题不要求废弃现有三场标签；应先在重建侧补齐接口和网格处理。独立表面查询、更全面的几何检查及规范化扩展是下一版预处理的改进项，改变其规则时使用新的数据版本。


审查属于当时四份输出的有限验证；后续30模型与重建结果统一见[实验报告](ThreeFieldReconstructionExperiments.md)。重建输入/provider、共享边界及CAD实施细节已合入[隔离带方案](BandLabelTopologyRecovery.md#implementation)。
