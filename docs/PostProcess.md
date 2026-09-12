# DualBrep 后处理：从 SDF/UDF 到显式 B-rep，以及重建失败机制

本文结合当前仓库代码与 DualBrep 论文，说明解码后的 SDF/UDF 如何经过网格提取、面分割、神经参数化和 OpenCASCADE 装配，最终恢复为显式 B-rep；同时逐阶段分析哪些误差会使重建失败，以及失败在中间文件和最终结果中如何表现。

论文来源：[DualBrep: A Dual-Field Continuous Representation for B-rep Modelling](https://arxiv.org/abs/2606.31579)，以下称“论文”。代码行为以本仓库当前版本为准；论文中的概念性描述、补充材料参数与当前实现不完全一致时，本文会明确指出。

## 1. 先给出总体结论

SDF 和 UDF 并不直接组成 STEP/B-rep。它们只负责恢复一个“带面分区线索的三角网格”：

- SDF 的零等值面恢复物体的封闭几何外壳；
- UDF 在该外壳上的低值带近似 B-rep edge，用来把外壳切成 face patches；
- 神经 Parametrizer 再把每个离散 patch 转成规则 UV 曲面网格，并预测 patch 邻接关系及共享边；
- OpenCASCADE 将规则网格拟合为 B-spline 曲面/曲线，构造 wire、trimmed face、shell 和 solid，最后写出 STEP。

完整数据流如下：

```text
共享 latent
   │
   ├─ query(x)[0] → SDF 体 → Marching Cubes(SDF=0) → recon_sdf.ply
   │                                                        │
   └─ query(x)[1] → UDF ──在上述三角面中心直接查询────────────┘
                                      │
                                    udf_g.npy
                                      │
                         阈值掩码 + 两遍连通分量
                                      │
                                  cluster.ply
                         （每个三角面一个 patch label）
                                      │
                 每 patch 采样 100 个 xyz+normal 点
                                      │
                             Parametrizer 网络
                  ┌───────────────────┼──────────────────┐
                  │                   │                  │
           16×16 face grids      face adjacency      16点 trim edges
                  └───────────────────┼──────────────────┘
                                    post.npz
                                      │
              几何对齐优化 → B-spline 拟合 → wire/trim face
                                      │
                         sew shell → make/fix solid
                                      │
                         BRepCheck + STEP 回读检查
                                      │
                              <name>.step
```

因此，“场恢复成功”和“B-rep 恢复成功”是不同层次：一个视觉上很好的 `recon_sdf.ply` 或 `cluster.ply`，仍可能因为一条边未闭合而无法形成有效 solid。论文也将主要瓶颈归因于连续 patch 到严格离散 B-rep 的最后一段转换。

## 2. B-rep 最终需要恢复什么

论文将 B-rep 写为几何与拓扑的组合。当前流水线实际恢复的核心对象是：

1. **Face geometry**：每个面对应一张参数曲面；
2. **Edge geometry**：相邻面之间的共享裁剪曲线；
3. **Face-edge incidence**：每条边属于哪两个面；
4. **Wire/trim**：每个面的边按闭合环组织，用来裁剪无限或未裁剪参数曲面；
5. **Shell/solid**：所有裁剪面以一致方向无缝拼合成封闭壳，再成为实体。

顶点没有由网络单独回归。论文认为在曲线和 face-edge connectivity 已知后，顶点以及 edge-vertex connectivity 可由曲线端点/相交关系确定。当前代码也是先预测边曲线，再从相邻边端点推断三面交汇的顶点约束（`brep_post/utils.py:267-363`）。

## 3. 阶段一：从共享 latent 得到 SDF/UDF

### 3.1 连续查询与距离反归一化

`DualVAE.query(points, latent)` 对任意三维坐标输出两个通道：

$$
F(\mathbf{x})=[\hat{s}(\mathbf{x}),\hat{u}(\mathbf{x})].
$$

其中通道 0 是截断归一化 SDF，通道 1 是截断归一化 UDF（`model.py:388-390`）。`ae_reconstruct.py:98-100` 将它们裁剪并乘回 `clip_value=0.015`：

$$
s=\operatorname{clip}(\hat{s},-1,1)\cdot0.015,
\qquad
u=\operatorname{clip}(\hat{u},0,1)\cdot0.015.
$$

这里的长度都在归一化坐标空间中，而非 STEP 原始单位。SDF 表示到实体表面的有符号距离；UDF 表示到 GVD/Voronoi 分界面的无符号距离。二者的几何分工是：

$$
\mathcal S=\{\mathbf{x}\mid s(\mathbf{x})=0\},
\qquad
\mathcal E\approx\mathcal S\cap\{\mathbf{x}\mid u(\mathbf{x})\approx0\}.
$$

即 SDF 决定“表面在哪里”，UDF 决定“表面身份在哪里改变”。UDF 的零集在三维空间中是二维 GVD sheet；它与 SDF 表面的交线才近似一维 B-rep edge。更完整的 GVD 定义见 `docs/UDF.md`。

### 3.2 稠密与加速网格查询

普通路径 `_inference` 在 $[-1,1]^3$ 的 $R^3$ 规则网格上查询网络（`model.py:173-196`）。默认加速路径 `_inference_acc` 则：

1. 先完整查询 $128^3$；
2. 根据粗 SDF 的 $|\hat{s}|<0.010/0.015$ 区域生成并膨胀近表面 mask；
3. 在 mask 内重新查询 $256^3$，其余位置三线性上采样；
4. 请求 `test_res=512` 时再重复一次局部细化。

相关实现位于 `model.py:200-274`。SDF 和 UDF 使用同一个、由 **SDF** 决定的细化 mask。这样做符合下游只关心物体表面附近 UDF 的目标，但也意味着粗 SDF 没有捕获的区域不会在高分辨率阶段得到真正重算。

## 4. 阶段二：SDF 恢复三角外壳

`mesh_utils.py:sdf2mesh` 对物理尺度的 SDF 体执行：

```python
v, f = mcubes.marching_cubes(v_sdf, 0.0)
v = v / (res - 1) * 2 - 1
```

也就是提取 $s=0$ 等值面，再把体素坐标映射回 $[-1,1]^3$（`mesh_utils.py:12-18`），得到 `recon_sdf.ply`。

`ae_reconstruct.py:105-110` 若发现 Marching Cubes 没有产生三角形，会输出 `empty SDF mesh, skipping`，该样本不会进入后续步骤。

### 4.1 最大连通分量过滤

分割前，`clustering.py:1177-1187` 在 `recon_sdf.ply` 的三角面邻接图上求连通分量，只保留三角形数最多的一个，并用相同 mask 过滤 `udf_g.npy`。

这一操作会清除 SDF 噪声造成的小浮岛，但它也明确假设目标是单个连通实体。论文的数据过滤同样排除了 multiple disjoint solids。若输入本来包含多个独立实体，较小实体会被无条件删除。

## 5. 阶段三：UDF 把 SDF 外壳切成 face patches

### 5.1 默认分割不依赖 `recon_udf.ply`

`ae_reconstruct.py:111-121` 同时做了两件容易混淆的事：

1. 对稠密 UDF 以 `udf_threshold=0.01` 做 Marching Cubes，得到 `recon_udf.ply`；它是 $u=0.01$ 的偏置壳，主要用于可视化；
2. 对 `recon_sdf.ply` 每个三角形中心 $\mathbf c_i$ 直接调用网络：

   $$u_i=\operatorname{clip}(\hat u(\mathbf c_i),0,1)\cdot0.015,$$

   并保存为 `udf_g.npy`。

默认 `hierarchical` 分割读取第二项，而不是从 `recon_udf.ply` 求交。直接查询避免了先将 UDF 离散成体素再插值的误差。只有缺少 `udf_g.npy` 而存在旧格式 `recon_udf.npy` 时，`clustering.py:1129-1143` 才在三角面中心对 UDF 体做三线性采样。

### 5.2 第一遍：挖掉低 UDF 边界带

令 SDF 网格的每个三角面为一个图节点，相邻三角面共享图边。当前默认参数定义在 `clustering.py:38-48`：

```text
threshold1 = 0.005
threshold2 = 0.01
filter_size = 10
avg_udf_threshold = 0.003
mode = hierarchical
```

第一遍只激活：

$$
A_1=\{i\mid |u_i|\ge\tau_1\},\qquad \tau_1=0.005.
$$

低 UDF 三角形被标为 `-1`，相当于从表面挖掉沿 B-rep edge 的窄带。在激活三角面诱导的邻接子图上求连通分量，每个连通分量就是一个候选 face patch（`clustering.py:467-493`）。然后删除：

- 三角形数少于 `filter_size=10` 的分量；
- 平均 UDF 不大于 `0.003` 的分量。

### 5.3 第二遍：用更严格阈值拆开欠分割 patch

对于第一遍保留下来、最大 UDF 达到 $\tau_2=0.01$ 的 patch，只保留其中：

$$
A_2=\{i\mid |u_i|\ge\tau_2\}
$$

的三角形，再求一次连通分量。只有产生至少两个、且各自不少于 10 个三角形的有效子块时，才用它们替换原 patch；否则保留第一遍结果（`clustering.py:508-588`）。这就是论文所谓 coarse pass 加 stricter revisiting。

论文补充材料给出的 $\tau_1=0.005$、$\tau_2=0.01$、平均 UDF 阈值 `0.003` 与当前代码一致，但论文写最小 component size 为 5；当前运行路径将模块级 `filter_size=10` 显式传入函数，因此实际是 10。函数签名自身的默认值仍是 5。

### 5.4 分割输出

`process_item` 把标签存为 `cluster.ply` 的 face attribute `label`；`-1` 面显示为黑色（`clustering.py:1120-1126`）。若所有 patch 都被过滤，代码不是宣告失败，而是将整个最大连通分量强制设为 cluster 0（`clustering.py:1207-1210`）。这让分割阶段能输出文件，但下一阶段因为少于两个 face 而跳过该候选。

## 6. 阶段四：从离散 patch 预测参数曲面、边和拓扑

入口是 `rebuild.py`，论文称该网络为 learned B-rep rebuilder；当前仓库又把其输出之后的 OCC 阶段称为 post-processing/rebuilder。为避免名称歧义，本文将神经网络部分称为 **Parametrizer**，将 OCC 部分称为 **assembler**。

### 6.1 每个 patch 的输入

`rebuild.py:83-107` 对每个非 `-1` label：

1. 提取该 label 的三角子网格；
2. 计算 patch 包围盒中心和三轴 extent；
3. 在其三角面上均匀采样 100 个点；
4. 使用 triangle normals，形成 `(100,6)` 的 `xyz+normal` 输入。

如果有效 patch 少于两个，`rebuild.py:208-211` 直接打印 `<2 faces, skipping`，不会生成 `post.npz`。

### 6.2 预测规则曲面网格

`Parametrizer` 用点编码器把每个 patch 压成 $4\times256$ token，再拼接一个包围盒 token；所有 patch 经 8 层 self-attention 交换全局拓扑上下文（`rebuild_model.py:301-319,373-413`）。

face decoder 从 $2\times2$ feature map 逐步反卷积到 $16\times16$，输出每个 UV 格点的 xyz 与 normal，并回归包围盒修正量（`rebuild_model.py:415-429`）。反归一化后得到：

```text
pred_face: (N_face, 16, 16, 6)
```

其中后处理只使用 xyz；normal 是网络训练和几何约束的一部分。

### 6.3 预测 face adjacency 和共享边

推理时网络枚举全部有序 face pairs $(i,j)$，拼接两面的特征并做二分类：

$$
\hat a_{ij}=\mathbf 1[\sigma(l_{ij})>0.5].
$$

对判为相邻的 pair，1D decoder 输出 16 个五维样本，其中前两维是第一个面的 UV 坐标（`rebuild_model.py:436-510`）。`hermite_sample` 用双线性 `grid_sample` 在第一个面的预测 $16\times16$ 网格上采样，将 UV 曲线变成 16 个三维点（`rebuild_model.py:652-665`）：

```text
pred_edge: (N_edge, 16, 3)
pred_edge_face_connectivity: (N_edge, 3)
                         每行 = [edge_id, face_i, face_j]
```

`rebuild.py:128-155` 将 face grids、3D edge samples 和 connectivity 写入 `post.npz`。这一阶段才显式预测离散 face-face topology；SDF/UDF 本身并没有直接输出 OCC 拓扑图。

### 6.4 24 个旋转候选

UV parameterization 存在方向/起点歧义，网络也并非完全旋转不变。`rebuild.py:54-55,167-220` 可将 `cluster.ply` 分别施加正八面体群的 24 个旋转，为同一形状产生 24 份 `post.npz`。assembler 最后应用逆旋转恢复原坐标。

这不是几何上的必要步骤，而是一种 test-time search：不同姿态造成略有不同的 surface/edge/topology 预测，只需其中一个候选成功闭合即可。

## 7. 阶段五：从 `post.npz` 装配 OpenCASCADE B-rep

### 7.1 半边去重与初始拓扑整理

`construct_brep_from_datanpz` 读取 `pred_face`、`pred_edge` 和 `pred_edge_face_connectivity`，构造 `Shape`（`brep_post/construct_brep.py:122-153`）。

因为网络枚举有序 pair，$(i,j)$ 与 $(j,i)$ 可能各预测一条方向相反的“半边”。`Shape.remove_half_edges` 按无序面 pair 分组：

- 两条都存在时，选择平均距离两个相邻面更近的一条；
- 只存在一条且足够靠近两面时保留；
- 边到两面的平均距离超过 `0.3` 时删除；
- 如果所有边都被删除，将 `have_data=False`。

见 `brep_post/utils.py:155-242`。随后 `build_fe` 建立每个 face 的 edge id 列表；`build_vertices` 在三面构成三角邻接环时，将三条边最近的端点视为同一候选 B-rep vertex，只保留平均端点距离不超过 `0.1` 的组合（`brep_post/utils.py:255-363`）。

### 7.2 几何对齐优化

默认 assembler 调用 `optimize`（`brep_post/construct_brep.py:252-267`）。优化变量是：

- 每条边的三轴缩放和三轴平移；
- 每张 face grid 的三轴平移。

损失包括：

1. edge 到两个 incident face 的 Chamfer 距离；
2. 三条边形成候选顶点时的端点闭合误差；
3. 同一 face 上各边端点的最近邻连接误差；
4. 很小的变换正则项。

实现位于 `brep_post/utils.py:504-635`。若优化被判断发散，会返回原始预测，而不是继续使用发散后的结果。

这一步只能做低自由度对齐，不能改正错误的 face 数、错误邻接，也不能将形状错误的曲面重新建模。

### 7.3 B-spline 曲面与曲线拟合

每个 $16\times16$ `pred_face` xyz grid 通过 `GeomAPI_PointsToBSplineSurface` 拟合为 C2 B-spline surface；代码依次尝试 `0.001, 0.01, 0.03, 0.05, 0.08` 拟合精度，选择采样误差最小者，全部失败则用 `0.1` 粗略回退（`brep_post/utils.py:70-76,762-829`）。

每条 16 点 `pred_edge` 类似地通过 `GeomAPI_PointsToBSpline` 拟合为 C2 B-spline curve，尝试 `0.001, 0.005, 0.008, 0.05`，失败则以 `0.1` 回退（`brep_post/utils.py:832-878`）。

当前实现不会分类并恢复精确 plane、cylinder、circle 等解析 primitive；这与论文的限制说明一致。即使原模型是精确平面和圆，输出通常仍是近似 B-spline。

### 7.4 edge → wire → trimmed face

对每个面，assembler 收集预测拓扑中所有 incident edges，并依次尝试连接容差：

```text
0.002, 0.006, 0.01, 0.015, 0.02, 0.025, 0.05, 0.08
```

首选 `ShapeAnalysis_FreeBounds.ConnectEdgesToWires` 将无序边集合连接为 wire；每个容差最多随机打乱重试 3 次，并优先选 wire 数更少的结果（`brep_post/utils.py:881-927`）。

若无序连接不能构造 trimmed face，当前仓库增加了 ordered fallback：按最近端点贪心排序边，允许最大跳跃 `0.15`，随后用 `ShapeFix_Wire` 重排、连接、修补 3D gap 并闭环（`brep_post/utils.py:940-1011`）。

`create_trimmed_face_from_wire` 把 wire 添加到拟合曲面，通过 `ShapeFix_Face` 修复方向、缺失 seam 和相交 wire，最后只接受 `BRepCheck_Analyzer(face).IsValid()` 的 face（`brep_post/utils.py:1025-1122`）。

### 7.5 trimmed faces → shell → solid

若有效 trimmed face 少于预测 face 数的 80%，当前候选直接放弃构造 solid（`brep_post/construct_brep.py:373-385`）。否则在上述容差序列上逐一尝试：

1. `BRepBuilderAPI_Sewing` 缝合所有 trimmed faces；
2. 若结果为 `COMPOUND`，判为失败；
3. 无效 shell 用 `ShapeFix_Shell` 修复 face 和方向；
4. `BRepBuilderAPI_MakeSolid` 从 shell 构造 solid；
5. `ShapeFix_Solid` 修复 shell 与方向，但禁止生成 open solid；
6. 只有结果类型为 `TopAbs_SOLID` 且通过 `BRepCheck_Analyzer` 才返回。

见 `brep_post/utils.py:1259-1321`。

成功 solid 会先撤销 test-time rotation，写为 `recon_brep.step`，再重新读入并检查：

- STEP 能够读入；
- 根 ShapeType 必须是 `TopAbs_SOLID`；
- 设置 `0.1` tolerance 后通过 `BRepCheck_Analyzer`。

只有这些条件全部成立，才写 `success.txt` 和 `recon_brep.stl`（`brep_post/construct_brep.py:387-399`，`brep_post/construct_brep.py:56-75`）。

如果无法形成 solid，代码还可能把 trimmed face 和未裁剪整面混合缝成 compound，并写一个候选 `recon_brep.step`（`brep_post/construct_brep.py:403-479`）。但它不会写 `success.txt`，所以 `postprocess.py` 不把这个文件视为有效结果，也不会提升为最终 `<name>.step`。

### 7.6 多候选选择

`postprocess.py` 对所有旋转候选执行上述过程，默认每个候选最多实际运行 1200 秒。每个 shape 按候选目录名排序，选择第一个带 `success.txt` 的候选复制为：

```text
<name>.step
<name>.ply
```

见 `postprocess.py:241-257`。这里优化目标是“找到一个能闭合的候选”，并不会在所有有效候选中选几何误差最小者。

## 8. 重建会在哪些步骤失败

下表先汇总，再在后文解释因果链。

| 阶段 | 主要失败原因 | 中间结果中的表现 | 最终影响 |
|---|---|---|---|
| 双场预测 | SDF/UDF 回归误差或彼此不一致 | 表面变形；UDF 低值线偏离表面 edge | 错误几何或错误 face partition |
| 网格采样 | 分辨率有限、粗网格漏检、Marching Cubes 混叠 | 薄壁消失/粘连，小孔封死，细槽变钝，浮岛 | 错误壳体；后续无法补回细节 |
| 最大分量过滤 | 多实体或小的有效独立组件 | 小实体整个消失 | 不完整模型 |
| UDF 分割 | 阈值、噪声、窄 patch、三角面中心采样 | 欠分割、过分割、黑色 `label=-1` 带/块 | face 数与拓扑错误 |
| patch 采样 | 每面仅 100 点、错误 patch 非单一参数面 | 小面/高曲率区域采样不足 | UV grid 拟合变形 |
| Parametrizer | surface grid、adjacency、trim curve 预测错误 | `recon_faces.ply` 错位；`recon_edges.ply` 缺边/多边/离面 | wire 不闭合或面不相交 |
| 几何优化 | 错拓扑无法优化；可用变换自由度有限 | edge 仍不同时贴合两面，角点分裂 | 大 gap、重叠或扭曲 |
| B-spline 拟合 | 规则网格折叠、噪声、周期性判断错误 | 曲面起皱、自交；曲线过冲 | 无效 trimmed face |
| wire/trim | 边端点 gap、顺序错误、假邻接、缺邻接 | 多个 open wires、自交环、错误内外环 | `trimmed_face=None` |
| sewing/solid | 面间缝隙、方向不一致、缺面、非流形边 | `COMPOUND`、open shell、多个 shell | 无 `success.txt`，最终无 STEP |
| 候选执行 | OCC 卡死、异常、内存问题、超时 | candidate error/timeout | 该旋转失败；全部失败则整形状失败 |

### 8.1 SDF 误差：几何首先不可逆地丢失

SDF 决定后续所有步骤工作的载体。如果 SDF 有偏差，会出现：

- 零等值面整体收缩、膨胀或局部波动；
- 两片本应分开的薄壁粘成一个连通表面；
- 薄片、窄槽、小孔、小圆角或尖锐细节消失；
- 本应连接的部分断开，产生多个连通分量；
- SDF 全正或全负，没有零穿越，Marching Cubes 输出空网格。

论文明确指出体素/采样分辨率会使极薄结构和高频细节 alias 或消失。当前加速实现还有一个具体放大机制：高分辨率 mask 来自 $128^3$ 粗 SDF。如果细薄特征在粗网格上已经漏掉，它所在区域就可能只被插值，而不会在 $256^3/512^3$ 真正调用网络重新查询。因此调高 `test_res` 不保证找回粗阶段已经漏检的结构。

外在表现包括 `empty SDF mesh, skipping`、`recon_sdf.ply` 中孔洞被封、薄壁融合、部件缺失或噪声浮岛。浮岛会在最大连通分量过滤时消失，但错误融合不会被修复。

### 8.2 UDF 与 SDF 不一致：表面正确但切分错误

共享 latent 旨在让几何和拓扑联合生成，但不保证两场完全一致。典型情况是 SDF 表面视觉正确，而 UDF 的低值 sheet 与它相交在错误位置：

- **边界 UDF 过高**：低值带未切断三角面邻接，两个真实 B-rep faces 被合成一个 patch，形成欠分割；
- **面内部 UDF 虚假过低**：同一真实面被切成多个 patch，形成过分割；
- **边界带断裂**：局部仍有激活三角面跨过真实边界，两个 patch 通过细桥连接；
- **边界带过宽**：窄 face 的所有三角形都低于阈值，整个 face 变成 `label=-1`；
- **UDF 平滑掉小拓扑**：小孔、短边和相邻窄面合并。

论文 Figure 5/限制部分特别指出薄壁可能在 GVD segmentation 中消失或与邻区合并。数据集还主动过滤了归一化后长度小于 $10^{-3}$ 的 edge、任一 face 超过 50 条 trimming edges 的模型，以及 10–100 faces 范围外的模型；这说明超出这些尺度/复杂度的输入并非其主要训练分布。

当前代码的固定阈值也带来尺度敏感性。虽然整体对象被归一化到约 $[-0.9,0.9]^3$，局部 face 宽度仍可能远小于 `0.005`。此外 UDF 只在三角形中心采样：比三角形更窄的低值边界可能没有被中心命中，而粗三角形中心落入边界带时也可能一次移除过大区域。

外在表现是 `cluster.ply` face 数明显过少或过多、同一颜色跨过几何 edge、平滑面上出现多种颜色、黑色 `label=-1` 区域吞掉整个窄面。全被过滤时虽然代码强制生成单 cluster，`rebuild.py` 随后会输出 `<2 faces, skipping`。

### 8.3 最大连通分量过滤：设计假设导致的“成功删除”

`process_item` 只保留最大的 SDF 网格连通分量。这对论文训练分布中的单 solid 合理，但对以下输入会产生确定性信息损失：

- assembly 或多个互不接触实体；
- 本应独立存在的小实体；
- 因 SDF 误差断开的、但仍属于目标的组件。

其表现不是报错，而是 `cluster.ply` 相比 `recon_sdf.ply` 少了一整个组件，因此很容易被误认为网络没有生成该部件。

### 8.4 错误 segmentation 传入 Parametrizer 后通常无法恢复

Parametrizer 把每个 label 当作一个 B-rep face。它没有回到 UDF 或原始点云重新修正 patch 数量：

- 欠分割时，一个输入 patch 可能横跨两个相交解析面，不存在单张平滑 $16\times16$ UV surface 能忠实表示它；网络常输出跨边界的圆滑过渡或折叠网格；
- 过分割时，一个真实面变成多个预测面，网络还必须预测它们之间的假边，显著增加闭合难度；
- `label=-1` 三角形完全不参与 face 采样，其覆盖区域不会直接成为参数面；
- 小 patch 只有 100 个均匀样本，高曲率或边界细节容易漏采；triangle normal 也会携带 Marching Cubes 离散噪声。

可通过 `input_faces.ply` 检查送入网络的点，通过 `recon_faces.ply` 检查 $16\times16$ 曲面网格是否覆盖原 patch，通过 `recon_edges.ply` 检查边是否落在两相邻面交界处。

### 8.5 邻接分类错误是离散拓扑的关键失败点

网络以 0.5 固定阈值独立判断所有有序 face pairs。它没有在分类输出层强制：

- 邻接矩阵严格对称；
- 每条开边恰好被两个面使用；
- 每个面上的 incident edges 构成一个或多个闭环；
- 全体 faces 构成单个闭合二维流形。

因此可能出现：

- **漏边**：两个相邻面没有共享 edge，面边界留下缺口；
- **假边**：不相邻面被连接，产生跨空间曲线或错误裁剪；
- **单侧半边**：$(i,j)$ 被预测、$(j,i)$ 未预测；代码会尝试保留它，但几何可能只贴合第一个面；
- **重复边**：同一 pair 产生两条不一致半边；`remove_half_edges` 只能择一，无法保证选择后的边能闭合两面的 wire；
- **孤立面**：没有 incident edge，`construct_brep.py` 将该 face 的 trimmed result 设为 `None`。

这些错误对 Chamfer/F1 的平均影响可能很小，却足以破坏 watertightness。论文补充材料强调：一个 missing 或 misaligned face 就能让 closed-shell constraint 失败；生成版的微小随机扰动会被脆弱的 rebuilder 放大。

### 8.6 surface/edge 几何不一致

推理时 edge 的 3D 点最初是在有序 pair 的第一个 face grid 上通过 UV 采样得到，因此天然贴合 face (i)，但不保证贴合 face (j)。几何优化尝试让 edge 靠近两面，却只允许 edge 三轴缩放/平移和 face 整体平移，无法改变曲线局部形状或曲面形状。

若两个 face grids 本身没有在应有位置相交，或者预测 edge 弯曲形状错误，就会留下：

- edge 与一张 incident surface 分离；
- 相邻边端点不重合，wire 有 gap；
- 三条边应交于一个 vertex，却形成三个相近但不同的端点；
- 两个 surface 相互穿透、重叠或留缝。

优化若发散会退回未经优化的几何；它不会将该候选直接标为失败。真正失败通常延迟到 wire、trim 或 sewing 阶段才显现。

### 8.7 B-spline 拟合与 UV 参数化失败

论文和当前实现统一把所有面、边拟合为 B-spline，而不是恢复精确 primitive type。这样支持自由曲面，但会造成：

- 原平面或圆柱只能近似，数值误差在共享边处累积；
- $16\times16$ grid 若行列翻转、折叠或自交，拟合曲面也可能扭曲；
- 周期面由网格首末边距离启发式判定 U/V periodic，判断错误会产生 seam 问题；
- 高阶 C2 拟合可能在稀疏/噪声点之间过冲；
- 粗回退 tolerance 可生成可拟合但偏离原几何较大的曲面/曲线。

其表现是 `recon_faces.ply` 尚可，但 OCC 的 `separate_faces.ply` 起皱或范围异常；或者曲面能生成，却无法用预测 wire 得到有效 trimmed face。

### 8.8 wire 构造的容差两难

对一个 face 的无序边集合，单一全局连接容差存在经典冲突：

- 容差太小，真实连续边之间的预测 gap 无法跨越，留下多个 open wires；
- 容差太大，附近但不相连的顶点被错误合并，wire 顺序错误或自交。

当前代码通过容差阶梯、随机排序和 ordered fallback 缓解，但不能保证正确。ordered fallback 也是贪心最近端点策略：在高密度顶点、多个内环、相近孔洞或错误拓扑下，最近端点不一定是真正的下一条边；允许到 `0.15` 的 jump 还可能把相距较远的错误边强行焊接。

失败表现包括 `wire_list is None`、`face_fixer.Face().IsNull()` 或 `BRepCheck_Analyzer(face)` 不通过，最终该面对应 `trimmed_face=None`。若失败面达到 20%，候选不会尝试构造 solid；即使少于 20%，缺少关键面也通常使 sewing 失败。

### 8.9 sewing 和 solid validity 的“全局脆弱性”

所有局部 face 看起来正确，不代表能构成 solid。一个有效封闭 B-rep 要求共享边在拓扑和几何容差内一致、shell 无洞、方向一致、边的使用次数合法且没有非流形连接。

当前代码中常见失败结果是：

- sewing 返回 `TopAbs_COMPOUND`：faces 没有缝成一个 shell；
- 得到 open/invalid shell，`ShapeFix_Shell` 无法修复；
- `MakeSolid` 结果不是有效 `TopAbs_SOLID`；
- 内存中的 solid 通过，但 STEP 写出再读入后检查失败，代码删除 `recon_brep.step`；
-只能写出 fallback compound，没有 `success.txt`；
- OCC 运算卡在复杂自交/修复中，超过 1200 秒后进程被强杀。

论文报告 point-cloud reverse engineering 的单次结果 validity 为 76.34%，生成式版本为 69.49%；这意味着高质量平均几何并不等价于 100% 可用 CAD solid。论文的瓶颈分析进一步显示，segmented triangle patches 的 surface F1 高于重建后的 B-rep，而 validity 在重建后下降得更明显。

### 8.10 当前“成功”判据也不是全部 CAD 质量判据

最终候选成功依赖 `TopAbs_SOLID + BRepCheck_Analyzer.IsValid()`，并做一次 STEP 回读检查。`brep_post/utils.py:1491-1542` 还定义了更细的 `solid_valid_check`：检查每面可三角化、wire 顺序、wire 自交和 shell bad edges，但当前主流水线没有调用它。

因此存在两类值得区分的输出：

1. **硬失败**：无 `success.txt`，所有旋转都失败，最终没有 `<name>.step`；
2. **通过当前检查但质量较差**：有 STEP 且 OCC 判 valid，但可能几何偏差较大、解析 primitive 丢失、存在不理想的 trim、过分割/欠分割，或在更严格 CAD 工具中暴露问题。

另外，`--drop_num>0` 会搜索删除最多约 20% faces 后能闭合的子 solid（默认是 0）。这可能提高“有效 solid”率，但得到的实体可能缺面、缺特征或改变拓扑，不能把 seal 成功直接等同于忠实重建。

## 9. 如何根据输出文件定位失败阶段

建议按以下顺序检查，而不是只看最终是否有 STEP：

| 文件/日志 | 检查内容 | 可定位的问题 |
|---|---|---|
| `recon_sdf.ply` | 外壳是否完整、孔/薄壁是否存在、是否有浮岛 | SDF、网格分辨率、Marching Cubes |
| `udf_g.npy` | 真正边界附近是否低值，面内部是否高值 | UDF/SDF 对齐与阈值可分性 |
| `recon_udf.ply` | 仅辅助观察 UDF 偏置壳，不应作为默认分割真值 | UDF 全局形态 |
| `cluster.ply` | 颜色是否一一对应真实 faces，是否有黑块 | 欠分割、过分割、被过滤 patch |
| `input_faces.ply` | 每个 patch 的 100 点输入是否覆盖合理 | patch 提取/采样 |
| `recon_faces.ply` | 规则曲面网格是否平滑、覆盖对应 patch | face parameterization |
| `recon_edges.ply` | 缺边、多边、边是否同时落在两面交界 | adjacency/trim curve |
| `post.npz` | face/edge 数量与 connectivity 是否合理 | 神经 Parametrizer 输出 |
| `pp/optimized_edge.obj` | 优化后端点是否闭合、edge 是否贴面 | 几何对齐优化 |
| `pp/separate_faces.ply` | 未裁剪拟合曲面的形状 | B-spline surface fitting |
| `pp/mixed_faces.ply` | 能生成的 trimmed faces 是否缺失 | wire/trim 失败 |
| `pp/success.txt` | 是否形成并回读为有效 solid | OCC sewing/solid 总判据 |
| `postprocess.py` 日志 | exception、timeout、各 rotation seal 数 | OCC 异常、超时、姿态敏感性 |

一个实用的判别顺序是：

```text
recon_sdf 错
  → 场/分辨率问题，后处理无法恢复

recon_sdf 对但 cluster 错
  → UDF 或分割阈值问题

cluster 对但 recon_faces/recon_edges 错
  → Parametrizer 问题

post.npz 看似对但无 success.txt
  → 几何对齐、wire/trim、sewing 或 OCC validity 问题

有 success.txt 但形状不忠实
  → 当前 validity 判据通过，但几何/primitive/topology 质量不足
```

## 10. 论文结论与当前代码的对应关系

### 10.1 论文主张得到代码支持的部分

- SDF 通过零等值面提供整体几何外壳；
- GVD-derived UDF 通过表面低值边界实现 primitive-free segmentation；
- 分割采用 UDF 约束的两遍 hierarchical region growing；
- rebuilder 从 patches 预测规则 surface grids、adjacency 和 trim curves；
- CAD kernel 最终装配 trimmed surfaces 为 solid；
- 微小局部误差可能几乎不影响平均几何指标，却破坏严格 watertightness；
- 极薄结构受体素/采样分辨率限制，复杂相交处的 stitching 仍会失败。

### 10.2 阅读论文时需注意的当前实现差异

1. 论文补充材料的 segmentation 最小 component size 为 5，当前默认运行路径为 10。
2. 论文把 edge 描述为预测在 surface UV 中并投影到曲面；当前代码确实预测 UV，但推理时先在有序 pair 的第一个 `pred_face` 网格上双线性采样成 3D edge，之后再由优化把它对齐两个面。
3. 论文概念上称输出 $16\times16\times3$ surface grid；当前 `pred_face` 包含 xyz+normal 共 6 通道，但 OCC 只消费前三个 xyz。
4. 当前代码包含 ordered wire fallback 和 24-rotation candidate search，这些工程措施显著影响实际 seal rate。
5. 当前主流程的 success 判据没有调用代码中定义的全部严格 wire/shell 检查。

## 11. 最核心的失败因果链

DualBrep 后处理不是简单地把两个等值面相交后写成 STEP，而是一条误差会逐级放大的链：

$$
\begin{aligned}
&\text{SDF 几何误差}
\Rightarrow \text{错误三角外壳}\\
&\text{UDF 拓扑误差}
\Rightarrow \text{错误 patch 数量/边界}\\
&\text{错误 patches}
\Rightarrow \text{错误 UV grids、adjacency、trim curves}\\
&\text{局部曲面/边误差}
\Rightarrow \text{wire gap、自交或错误 trim}\\
&\text{任一关键 face/edge 不一致}
\Rightarrow \text{shell 不闭合}\\
&\text{shell 不闭合}
\Rightarrow \text{无有效 B-rep solid}.
\end{aligned}
$$

其中最值得强调的是最后一步的非线性：前面 99% 的 patches 都正确，仍可能因为剩余 1% 的局部错误导致整个 solid validity 从成功变成失败。这正是论文所说的 B-rep modeling bottleneck，也是为什么仓库用 24 个旋转候选、几何优化、容差阶梯和多种 wire 修复策略来提高最终闭合率。

## 12. 主要代码索引

- 双场查询与多分辨率推理：`model.py:164-274,388-390`
- SDF/UDF 等值面：`mesh_utils.py:12-27`
- 场反归一化、三角面中心查询：`ae_reconstruct.py:90-139`
- UDF hierarchical segmentation：`clustering.py:38-48,440-588`
- 最大连通分量、分割文件输出：`clustering.py:1129-1235`
- patch 采样与旋转候选：`rebuild.py:83-107,167-220`
- Parametrizer surface/adjacency/edge inference：`rebuild_model.py:269-510,652-667`
- 半边整理、顶点关系和几何优化：`brep_post/utils.py:134-363,504-635`
- B-spline surface/curve fitting：`brep_post/utils.py:762-878`
- wire 与 trimmed face：`brep_post/utils.py:881-1122`
- shell/solid sewing 与修复：`brep_post/utils.py:1259-1321`
- STEP 检查和候选装配：`brep_post/construct_brep.py:47-75,179-480`
- 多旋转候选选择、异常和超时：`postprocess.py:69-105,128-267`

## 13. 参考资料

- [DualBrep 论文主页](https://arxiv.org/abs/2606.31579)
- [DualBrep HTML 全文](https://arxiv.org/html/2606.31579v1)
- 论文 Sec. 3.3：Mesh Extraction and Segmentation
- 论文 Sec. 3.4：Learned B-rep Rebuilder
- 论文 Sec. 4.4：Limitations and Failure Cases
- 论文补充材料：Hierarchical Region Growing、Dataset Filtering、Understanding the B-rep Modeling Bottleneck
