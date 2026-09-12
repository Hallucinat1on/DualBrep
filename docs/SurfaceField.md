# 从 SDF 表面预测 Voronoi-UDF 表面场，并复用 DualBrep 重建 B-rep

核对日期：2026-09-12。依据 DualBrep 论文及官方仓库当前 `main` 分支。

本文讨论的方案是：给定 SDF 或其提取的 mesh，将 DualBrep 的 Voronoi-UDF 限制到物体表面；以完整 mesh 几何、法向及可选单视图等作为条件，直接使用 DDPM 或 Flow Matching 生成表面标量场，随后恢复 B-rep。当前计划不以回归模型为前置步骤，也不使用 VAE，不重新生成已给定的几何。

**结论：这一方案在表示和后处理接口上可行，适合作为 mesh/SDF → B-rep 的研究路线。默认面分割只需要 mesh 上的 UDF 标量，可以不预测完整体 UDF；但“表面场足以支持分割”“输入几何能唯一确定表面场”“后处理能得到有效 B-rep”是三个不同命题，只有第一个可以在明确条件下直接论证。**

本文区分三类内容：论文与代码中可核实的事实、根据定义得到的数学推导，以及建议实施但尚未验证的实验方案。没有运行模型训练、官方权重推理或 OpenCascade 装配，文中的代码片段是接口示例，不代表已经完成端到端测试。

## 1. 目标表面场的准确含义

### 1.1 定义

设物体的 SDF 为 $s(x)$，其零水平集为：

$$
M=\{x\in\mathbb R^3\mid s(x)=0\}.
$$

令 $\Gamma$ 为以 B-rep faces 为几何站点构造的三维 Voronoi 分割面集合。DualBrep 的体 UDF 是：

$$
u(x)=d_{\mathbb R^3}(x,\Gamma)
=\inf_{y\in\Gamma}\|x-y\|_2.
$$

本方案预测其在物体表面上的限制，也称取迹：

$$
\boxed{g=u|_M,\qquad g(p)=d_{\mathbb R^3}(p,\Gamma),\quad p\in M.}
$$

这里仅限制查询点 $p$ 在表面，最近点 $y$ 仍可以处于物体内外的三维空间。它不是到最近 face 的距离，也不是沿物体表面到最近 edge 的距离。论文 §3.1 给出了这一场定义；官方标签生成代码对查询点计算的是到 Voronoi 三角网格的无符号距离。[论文](https://arxiv.org/html/2606.31579v1#S3.SS1)、[prepare_implicit.py](https://github.com/AutodeskAILab/DualBrep/blob/main/prepare_implicit.py)

为避免混淆，本文称它为 **Voronoi-UDF 表面场**。

### 1.2 与表面 edge-distance 的区别

记 $E$ 为不同 face 之间的边界曲线集合。

| 表示 | 定义 | 几何含义 |
| --- | --- | --- |
| 本文目标 $g(p)$ | $d_{\mathbb R^3}(p,\Gamma)$ | 到三维 Voronoi 分割面的直线距离 |
| 欧氏 edge-distance | $d_{\mathbb R^3}(p,E)$ | 到边界曲线的直线距离 |
| 测地 edge-distance | $d_M(p,E)$ | 沿物体表面到边界曲线的最短路径长度 |

以两个平面构成的直角棱边为例，若表面点到棱边的距离为 $t$，且附近没有其他 Voronoi 分割面参与最近距离竞争，则：

$$
g(p)=t/\sqrt{2},\qquad d_{\mathbb R^3}(p,E)=t.
$$

这个差别还可能来自非局部几何：对于尺寸为 $10\times10\times0.2$ 的薄长方体，上表面中心到周边棱线的距离为 $5$，到上下两个 face 之间的 Voronoi 中间面的距离却只有 $0.1$。以上是按场定义构造的几何例子，不是论文中的实测失败案例。

**本方案保持 DualBrep 的标签定义，只把预测域从体域改为表面。** 若日后改成 edge-distance，应作为另外一个表示消融实验，不能混用标签和分割阈值。

## 2. 为什么仅预测表面场在表示上可行

### 2.1 理想连续情形：保留 face 分割边界

考虑满足以下假设的理想 B-rep：

1. $M$ 是已知且正确的闭合表面；各 face 的内部互不重叠。
2. 不同 face 的公共边界构成 $E$，各 face interior 连通。
3. 使用与这些 faces 一致的理想 Voronoi 分区，且 $M\cap\Gamma=E$。
4. 此处讨论普通 face 间边界，不把周期 face 内部的 seam、退化边及拓扑重数归入这一分割定理。

根据距离函数的定义：

$$
g(p)=0
\iff p\in M\cap\Gamma
\iff p\in E.
$$

因此，精确连续场的零集恢复 $E$，而：

$$
\operatorname{CC}(M\setminus E)
$$

恢复各个 face interior，其中 $\operatorname{CC}$ 表示连通分量。

**这证明：在这些假设下，已知 $M$ 时，$g$ 足以保留面分割所需的边界信息。** 它没有证明可以唯一恢复所有 CAD 参数、seam、edge identity、顶点的拓扑重数或原始建模历史。

还有一个容易忽略的区别：完整连续场和有限表面点上的若干数值并不等价。有限采样可能完全漏掉一条短边或一个小面。因此工程方案应保留 mesh 邻接关系，并允许在更多表面位置查询场，而不能只保存一份很稀疏的带值点云就认为信息完整。

### 2.2 对现有实现更直接的论证：后处理只读取有限表面值

官方 `ae_reconstruct.py` 先提取 SDF mesh，再在三角面中心查询场，保存逐三角面的 `udf_g.npy`。默认 `hierarchical` 分割随后根据这些数值与 mesh 的邻接关系进行处理。[ae_reconstruct.py](https://github.com/AutodeskAILab/DualBrep/blob/main/ae_reconstruct.py)、[clustering.py](https://github.com/AutodeskAILab/DualBrep/blob/main/clustering.py)

将这一分割过程抽象为：

$$
L=\mathcal A\bigl(M_h,\{u(c_j)\}_{j=1}^{N_T};\eta\bigr),
$$

其中 $M_h$ 是离散 mesh，$c_j$ 是其第 $j$ 个三角面的中心，$\eta$ 是阈值与过滤参数。

如果新模型输出：

$$
\hat g(c_j)=u(c_j),\quad j=1,\ldots,N_T,
$$

并保持 mesh、三角面顺序和参数一致，则两种输入得到同样的分割结果：

$$
\mathcal A(M_h,\hat g;\eta)=\mathcal A(M_h,u|_{\{c_j\}};\eta).
$$

这个接口等价性不要求重建整个 $\Gamma$，也不依赖“连续零集能否完美提取”的理想化假设。它是本方案可以接入官方后处理的最直接依据。

等价范围应限定为这里核对的标量分割路径。仓库还有依赖额外方向信息或 UDF mesh 的可选分割模式，不能由一个标量输出无条件替代。

### 2.3 数值连续不意味着拓扑自动稳定

距离函数满足 1-Lipschitz 性质：

$$
|u(p)-u(q)|\le\|p-q\|_2.
$$

若 Voronoi 网格近似误差为 $\delta_\Gamma$、对应表面点偏差为 $\delta_M$，则在相应 Hausdorff 距离与点对应条件成立时：

$$
|d(p,\Gamma)-d(\tilde p,\tilde\Gamma)|
\le\delta_M+\delta_\Gamma.
$$

再加上网络预测和插值误差，就得到表面数值误差的预算。这说明目标是可连续近似的，适合作为回归或生成信号。

但阈值分割包含离散决策：一处断裂可能合并两个 face，一条伪边可能把一个 face 切开。只有当所有相关阈值比较都有足够误差裕量，并且采样、邻接关系保持一致时，才能进一步讨论离散分割稳定性。不能用低平均 L1 误差替代拓扑验证。

## 3. 几何条件下生成这个场，是否可学习

### 3.1 网络实际学习的映射

设输入 mesh 为 $M_h=(V,T)$，可从中提取条件点云：

$$
P=\{(p_i,n_i)\}_{i=1}^{N},
$$

可选条件 $C$ 可以包含单视图图像、相机、类别或上游几何 latent。对固定表面查询点集 $Q$，目标是学习条件分布：

$$
p_\theta\bigl((g(q_i))_{i=1}^{N_Q}\mid M_h,Q,C\bigr).
$$

网络需要整件物体的几何上下文。只把单点的 xyz 和法向输入独立 MLP，通常不足以区分薄壁、相邻曲面或不同整体形状下的 Voronoi 距离。编码器应把局部几何与全局结构一起编码；去噪/速度网络以时间、整组带噪场值和几何特征为输入，联合输出每个查询点的一个标量，而不是让各点独立生成。

该任务可以理解为学习一个几何条件下的面分割先验；输出用连续标量编码边界，避免直接给每个点预测固定数量的 face ID。

### 3.2 存在不可辨识性，但不否定统计学习

同一个平面区域可以在 CAD 中是一个 face，也可以人为分成两个共面的 faces。两种 B-rep 可以对应相同的 SDF 和完全相同的无标签表面点云，却具有不同的 $\Gamma$ 和 $g$。因此：

$$
P\not\Rightarrow \text{唯一原始 CAD 面划分}.
$$

可学习性依赖数据中的分割惯例和条件信息，而不是几何唯一性定理。合理目标是：

- 在统一标注规则的数据集上，预测该数据分布中常见、可重建的面划分；
- 对无法由输入区分的多种合法划分，允许输出候选，而不宣称唯一恢复原始 CAD；
- 对同一个模型的多视图、重网格版本、退化版本采用按 CAD 模型划分的训练/测试集，避免泄漏。

### 3.3 单视图条件能提供什么

图像可以提供形状语义、遮挡前的可见细节和对象类别信息；相机已知时，也可以把可见位置的图像特征与表面查询点对齐。图像不能凭空确定不可见区域的 CAD 分割，材质或颜色边界也不必然是 B-rep 边界。

建议先训练纯几何版本，再增加图像条件做消融。使用几何条件作为主要依据，图像作为补充。若按投影采样图像特征，应处理相机坐标、深度可见性和遮挡；缺少相机时，可先采用全局图像 token 的交叉注意力。

### 3.4 直接使用条件生成模型

第一版直接在离散表面场值上训练生成模型。固定输入 mesh 和本次生成使用的查询位置，只更新场值，不更新顶点位置或 mesh 连接关系。

采用条件 DDPM 或 Flow Matching 学习：

$$
p_\theta(g\mid M_h,Q,C).
$$

多峰目标直接用回归损失可能产生中间场，导致边界变弱；生成模型可以输出不同候选，但不保证候选具有有效拓扑，也不保证单一标注数据足以学到完整的多解分布。候选质量与多样性需要分别评估。

VAE 不属于当前最小实现。只有当表面查询规模或多步生成成本成为瓶颈时，再评估压缩是否保留短边、窄面和边界连续性。GT labels/GT 场的后处理对照仍需保留，用于定位失败，但不要求先训练回归模型。

## 4. 如何提取得到训练用表面场

### 4.1 首先区分手中已有的数据

| 已有信息 | 能得到什么标签 | 建议用途 |
| --- | --- | --- |
| GT STEP/B-rep，保留 face 划分 | 可构造 GT Voronoi 网格，再提取几何标签 | 首选监督来源 |
| 具有可靠 CAD face labels 的 mesh | 可用各标签对应的有限面片作为 Voronoi 站点，得到近似标签 | 没有 STEP 时的替代来源 |
| 已有 DualBrep UDF 模型或 UDF 网格 | 可在指定表面查询或插值得到预测标签 | 教师蒸馏或接口验证 |
| 只有无标签 SDF/mesh | 不能唯一提取原始 CAD 对应的 GT 场 | 需要额外监督、分割先验或教师 |

“不预测完整 UDF”不等于“训练标签完全不需要 CAD 拓扑”。推理阶段可以只输入几何；制作 GT 标签阶段仍需要知道 face 如何划分。

### 4.2 首选路线：GT B-rep → Voronoi 网格 → 表面距离

**步骤 A：统一坐标系。**

让 STEP、从它构造的 SDF、SDF 提取的 mesh、Voronoi 网格和相机使用一致的变换。官方流程将 STEP 规范化到 `[-0.9, 0.9]` 范围，并让 Voronoi 生成与隐式采样流程采用一致的规范化约定。[官方数据处理说明](https://github.com/AutodeskAILab/DualBrep#b-from-your-own-step-compute-the-samples)

实现时必须保存实际归一化变换，不要分别按几份近似 mesh 的包围盒重新归一化。独立归一化会把同一条边的位置和距离标签错开。若使用统一相似变换，距离与缩放系数线性变化；若包含各轴不同的缩放，应在变换后的坐标系重新计算欧氏距离，不能用一个标量换算。

**步骤 B：构造 Voronoi 网格。**

优先复用官方 `Voronoi/` 下的工具。在按仓库说明完成依赖和编译后，可运行：

```bash
Voronoi/build/calculate_voronoi/calculate_voronoi \
  input/00000164.step work/00000164/
```

得到 `voronoi.ply` 等中间文件。这里的目标是不同 CAD faces 之间的空间分区；不能把每个三角形都当作独立 CAD face，否则会把三角剖分边界也编码进去。[官方仓库](https://github.com/AutodeskAILab/DualBrep)

若自行实现 Voronoi 构造，应使用有限、已裁剪的 face 作为站点，并与目标分区定义保持一致。直接用无限延伸平面/曲面的距离，会改变裁剪边界附近的分区。为建立第一版基线，不建议同时重写 Voronoi 构造算法。

**步骤 C：从 SDF 提取推理时会使用的表面 mesh。**

使用 Marching Cubes 或上游已有的等值面提取结果，记作 $M_h=(V,T)$。保留顶点、三角面、法向和邻接关系。条件点云从 $M_h$ 上采样，而不是只在体网格中筛选 $|s(x)|<\epsilon$：后者得到的是有厚度的窄带点，不是严格的表面点。

三角形内部一般也不是解析 SDF 的精确零集，但它属于后处理真正使用的离散表面。因此，想严格复用官方接口时，应优先在这些三角形上标注。若另外采用 SDF 投影细化查询点，应记录该变化，并用相同约定训练与推理。

**步骤 D：准备两组点。**

| 点集 | 作用 | 建议 |
| --- | --- | --- |
| 条件点 $P$ | 编码器观察输入几何 | 面积采样，辅以小面/局部覆盖策略 |
| 监督查询点 $Q$ | 计算目标场和预测损失 | 表面随机点、靠边点及三角面中心 |

两组点不必相同。可以用较少条件点编码整个物体，但生成状态应覆盖需要联合预测的查询点。第一版优先使用全部三角面中心；若必须分块，应保留跨块共享状态或上下文，不能把各块独立生成后直接拼接并假定边界一致。不得把 GT edge 点集、face ID 或 UDF 数值作为推理时不存在的条件输入；GT 引导的查询布局也可能泄漏边界，主生成任务应使用推理时可获得的查询布局，通过损失加权和分组评估关注边界精度。

**步骤 E：计算点到 Voronoi 三角网格的距离。**

对每个 $q_i\in Q$：

$$
g_i=d(q_i,\Gamma_h),
$$

其中 $\Gamma_h$ 是 `voronoi.ply` 的三角网格。应计算点到三角形集合的最近距离，不能只计算到 Voronoi 网格顶点或稀疏点云的最近距离；后者会引入依赖采样密度的偏大误差。

官方 `prepare_implicit.py` 使用 libigl 的 unsigned distance 模式完成查询。可以复用其距离计算方式，但将查询点替换为所需的表面采样点，无需生成全部体域监督点。[prepare_implicit.py](https://github.com/AutodeskAILab/DualBrep/blob/main/prepare_implicit.py)

### 4.3 截断与单位：区分训练数值和后处理数值

建议同时保存：

$$
g_{\rm raw}=d(q,\Gamma_h),
\qquad
g_\tau=\min(g_{\rm raw},\tau),
\qquad
y=g_\tau/\tau.
$$

| 数值 | 范围 | 用途 |
| --- | --- | --- |
| `udf_raw` | 非负，未截断 | 标签检查、改变阈值或截断尺度 |
| `udf_metric` | $[0,\tau]$ | 后处理，在规范化几何坐标的距离单位中 |
| `udf_target` | $[0,1]$ | 网络训练目标 |

官方配置的 `clip_value` 为 `0.015`；输出到 `udf_g.npy` 前会乘回这个数值。它是在规范化坐标中的距离，不是固定的毫米或米。[config.yaml](https://github.com/AutodeskAILab/DualBrep/blob/main/config.yaml)、[ae_reconstruct.py](https://github.com/AutodeskAILab/DualBrep/blob/main/ae_reconstruct.py)

因此，若预测的是 $\hat y\in[0,1]$，写给后处理的应是：

$$
\hat g_{\rm metric}=\tau\hat y.
$$

不要直接将 `[0,1]` 的输出写给采用距离阈值的分割器。第一版可以使用官方尺度作为对齐起点，但阈值是否适用于自己的网格分辨率和形状分布，需要验证。

### 4.4 已有 UDF 时，如何提取

**已有隐式 UDF 网络：** 在其所用坐标系内，对自己的 mesh 三角面中心或其他表面点直接查询 UDF 头，然后按训练约定反归一化。无需先生成完整 UDF 体网格。若输入几何和教师内部重建的几何不同，应明确这是教师 UDF 在另一个表面上的取值，不能视为该输入几何的 GT。

**已有稠密 UDF 网格：** 将查询点映射到体网格索引，进行三线性插值。必须核对轴顺序、坐标范围、体素中心/角点约定、截断尺度及坐标对齐。近零谷和窄边界可能被有限分辨率抬高或抹平，应优先使用直接网络查询或 GT Voronoi 网格查询。

**已有 `udf_g.npy`：** 先检查其形状。官方该文件通常是与 `recon_sdf.ply` 三角面顺序对应的一维数组，不是可以在任意位置查询的三维体网格。换了 mesh、重排三角面或重新网格化后，不能继续按原索引使用。

**已有预测的 `recon_udf.ply`：** 不要把它默认当作 GT 的 `voronoi.ply`。预测流程可能提取正阈值等值面，它与 UDF 零集的几何位置不同；重新计算到该等值面的距离一般不等于原始 UDF。应优先查询原场或读取匹配的表面值。

### 4.5 关于几何退化和输入分布

为了接近“生成 mesh → B-rep”的实际使用，应在 GT 对应的 SDF 上模拟重采样、低分辨率、有限平滑和小幅扰动，并在退化后的实际查询点重新计算 $d(q,\Gamma_{\rm GT})$。

不过，这种监督只是固定 GT Voronoi 场在近似表面上的限制。若退化已删除薄壁、封死孔洞或改变连通性，它的零集不一定还能形成合理分割；此时不能盲目保留标签。应过滤严重不对应样本，或把几何修复作为额外任务。不能把错误 mesh 上的一组“精确距离值”等同于有效 B-rep 监督。

## 5. 当前最小生成模型与参考项目

### 5.1 固定几何，联合生成场值

一个足够清楚的第一版结构是：

$$
M_h\xrightarrow{\mathrm{MeshEncoder}}Z_{\rm geom},
\qquad
I\xrightarrow{\mathrm{ImageEncoder}}Z_{\rm img}\quad\text{（可选）},
$$

$$
F_\theta(y_t,t;Q,n_Q,Z_{\rm geom},Z_{\rm img})\in\mathbb R^{N_Q}.
$$

其中 $y_t$ 是带噪表面场，$F_\theta$ 在 DDPM 中预测噪声，在 Flow Matching 中预测速度。噪声/速度可为负，不对其输出头施加非负或 `[0,1]` 约束。生成结束后才按目标归一化约定恢复场值，在导出接口处裁剪至合法区间并乘回 $\tau$，同时记录裁剪前越界比例。

第一版优先令 $Q$ 为后处理 mesh 的全部三角面中心，避免顶点场到面中心的额外插值误差。若复用顶点网络，应显式实现面/顶点特征映射，而不只是改通道数。完整 mesh 提供几何与邻接，采样点可以承担几何条件。空间邻域交互可用于建模薄壁两侧的影响，但不能直接替代表面分割邻接，否则会把薄壁两侧连接起来。

表面网络应结合局部表面算子与三维全局条件，例如 xyz、法向、全局几何 token 或空间邻域交互。单纯依靠内蕴热扩散不能保证区分表面度量相似、空间嵌入不同的形状。不同样本可有不同查询数量，采用逐样本、打包或带 mask 的 batch；一次采样轨迹内保持查询集不变。

### 5.2 DDPM 与 Flow Matching 的训练目标

记 $y=\min(g,\tau)/\tau$。DDPM 可以使用：

$$
y_t=\sqrt{\bar\alpha_t}y+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \mathcal L_{\rm DDPM}=\mathbb E\|\epsilon_\theta(y_t,t;M_h,Q,C)-\epsilon\|_2^2.
$$

Flow Matching 的最小选择是噪声到数据的线性路径：

$$
y_t=(1-t)\epsilon+ty,\qquad
\mathcal L_{\rm FM}=\mathbb E\|v_\theta(y_t,t;M_h,Q,C)-(y-\epsilon)\|_2^2,
\quad\epsilon\sim\mathcal N(0,I).
$$

推理从 $y_0\sim\mathcal N(0,I)$ 出发，积分 $dy_t/dt=v_\theta$ 到 $t=1$。生成状态是 $\mathbb R^{N_Q}$ 中的标量向量，不是在曲面上移动粒子。DDPM 与 FM 需要各自训练，不能只替换采样器。上述公式是适配建议，不代表已经实现。

### 5.3 边界精度与训练检查

主损失采用噪声或速度预测损失。可对估计的干净场增加边界加权 L1/Huber 辅助项，但应单独消融权重与时间范围，不将其作为生成模型的唯一目标。薄壁处低 UDF 不一定意味着近边，应依据 GT face 边界单独评价。

注意以下训练问题：

- 大多数点可能落在截断饱和区；全局平均损失很低，也可能完全漏掉边界。
- 查询点可以按“边界附近、过渡区、面内部”分组采样，并分别记录误差。
- 只给标量增加强平滑正则可能抹平边界零谷，不应无条件使用。
- $g$ 是空间距离函数的表面限制，通常不满足表面 Eikonal 等式 $\|\nabla_Mg\|=1$。其可微处的切向梯度模一般不超过 1，不能把它按测地距离场施加强制单位梯度约束。

若出现稳定的断边或伪边，可增加相邻三角面的 same-face affinity 辅助监督。但这是后续增强项，不是复用官方标量分割接口的必要条件。

### 5.4 参考项目与复用优先级

以下依据公开论文、仓库说明与代码入口筛选，未在本项目数据上复现；没有一个是开箱即用的 DualBrep 表面场生成器。

| 项目 | 与本方案的对应关系 | 建议用途 |
| --- | --- | --- |
| [DoubleDiffusion](https://github.com/Wxyxixixi/DoubleDiffusion_3D_Mesh) | 在输入 mesh 上直接去噪生成表面纹理 | 第一版 DDPM 基础，RGB 改为单通道场 |
| [SQuadGen](https://github.com/microsoft/SQuadGen) | 生成表面距离场，再恢复 quad 布局 | 结构场、几何条件与拓扑评价参考 |
| [UV3-TeD](https://github.com/simofoti/UV3-TeD) | 在物体表面生成带颜色的点云信号 | mesh 提供结构、采样点承载场值 |
| [Functional Diffusion](https://github.com/1zb/functional-diffusion) | 将生成对象视为连续定义域上的函数 | 后续连续查询与采样一致性研究 |
| [Meta Flow Matching](https://github.com/facebookresearch/flow_matching) | 提供概率路径、求解器和训练示例 | 与表面网络组合，实现直接 FM |

**DoubleDiffusion：第一版实现参考。** 官方代码直接在 mesh 上进行去噪扩散，利用热扩散传播表面特征。阅读 `src/ssdm`、`lib/models/diffusion_net`、`train.py`、`infer.py`。RGB 改成 UDF 之外，还需修改数据加载、距离归一化、几何条件、输出位置和评价指标。热扩散是特征传播机制，去噪扩散才是生成机制。公开 bunny 纹理示例不能证明预训练模型能泛化到任意 CAD，应在自己的场标签上训练。[论文](https://arxiv.org/abs/2501.03397)、[官方代码](https://github.com/Wxyxixixi/DoubleDiffusion_3D_Mesh)

**SQuadGen：结构恢复参考。** 通过 CDF/DCDF 表面标量场表达 quad 布局，再提取离散结构，与“连续场 → face 分割 → B-rep”思路相近。重点看 `data_tools/README.md`、`infer_sqdiffuse.py`、`train_sqdiffuse.sh`，借鉴边界编码和提取后结构评价。CDF/DCDF 不等于 Voronoi-UDF，quad 约束不是 CAD face 约束；它使用 Geom-AE、SQ-VAE 和扩散流程，比当前方案更重，不整体照搬。[官方项目与论文入口](https://github.com/microsoft/SQuadGen)

**UV3-TeD：表面采样路线。** 在物体表面进行 DDPM，生成带颜色的点云，并使用几何热扩散，无须 UV 参数化。阅读 `data_loading.py`、`network.py`、`model_manager.py`。可借鉴“完整 mesh 提供结构、采样点承载生成值”，但必须验证向全部三角面中心传递场值时的边界误差，不能把稀疏点生成等同于完整连续场。[论文](https://arxiv.org/abs/2408.16762)、[官方代码](https://github.com/simofoti/UV3-TeD)

**Functional Diffusion：连续函数路线。** 将扩散对象扩展为连续定义域上的函数，采用 Transformer，展示 SDF 与表面形变函数生成。阅读 `models_ae.py`、`engine_ae.py`、`sample_class_cond.py`；不能仅凭文件名中的 `ae` 判断机制。若需要不同查询位置对应同一份随机生成结果，应研究其函数表示与采样一致性，而不是每换一批查询点就独立采样。公开 README 较简略，适配成本更高，不作为第一版首选。[项目与论文](https://1zb.github.io/functional-diffusion/)、[官方代码](https://github.com/1zb/functional-diffusion)

**Meta Flow Matching：生成流程组件。** 复用路径、求解器和训练示例，配合表面速度网络；它不提供现成 mesh-UDF 模型。第一版 FM 可采用 §5.2 的线性路径，几何编码可以缓存，采样过程中只更新场值。[官方代码](https://github.com/facebookresearch/flow_matching)

当前组合：**DDPM 以 DoubleDiffusion 为基础；FM 复用其表面网络思路并接入 Meta Flow Matching；结构场与后处理评价参考 SQuadGen。** UV3-TeD 用于大 mesh 的采样方案，Functional Diffusion 留作连续查询增强。VAE、几何生成和图像条件都不是第一版必要项。

## 6. 从预测表面场到 B-rep 的流程

### 6.1 第一步：在用于后处理的 mesh 上得到逐三角面数值

保留输入 SDF 提取的 mesh $M_h=(V,T)$，计算：

$$
c_j=(v_{j1}+v_{j2}+v_{j3})/3.
$$

固定这些中心作为查询集，从噪声开始联合生成一份表面场 $\hat y$，结束后裁剪到 `[0,1]`，转为 $\hat g(c_j)=\tau\hat y(c_j)$。输出长度必须等于三角面数量；每份候选单独保存并执行后处理。

如果网络只对一组固定点输出值，可通过表面局部插值映射到三角面中心，但应限制在正确的表面邻域中，避免跨薄壁或窄缝插值。更直接的方案是让查询解码器直接在中心处预测。

### 6.2 第二步：复用 UDF 引导的 mesh 分割

在低 UDF 区域屏蔽三角面，再在剩余邻接图中寻找连通分量。官方 hierarchical 函数还在较高阈值下尝试拆分组件，并过滤过小组件。未归属区域可以保留 `-1`，不必为了填满标签而强行扩张边界。[clustering.py](https://github.com/AutodeskAILab/DualBrep/blob/main/clustering.py)

输出是 mesh 上的 face labels。注意这里每个三角面的标签表示其归属的 **CAD face**，并非每个三角面一个独立 CAD face。

### 6.3 第三步：复用 learned parametrizer

将分割写入 `cluster.ply` 的逐三角面 `label` 属性。`rebuild.py` 会按标签采样点和法向，送入预训练 `Parametrizer`；默认每个分组采样 100 个点，`-1` 分组被排除。[rebuild.py](https://github.com/AutodeskAILab/DualBrep/blob/main/rebuild.py)

从论文方法看，学习模块预测曲面网格、面邻接和对应的 UV 裁剪曲线；它利用各面片的整体上下文补全连续参数结构。[论文 §3.4](https://arxiv.org/html/2606.31579v1#S3.SS4)

**它不是直接沿表面场零集描边后就完成 CAD 转换。** 因此边界带保留少量空隙仍有可能被重建，但面片丢失、错误合并、错误拆分和显著分布变化都可能导致失败。

### 6.4 第四步：拟合、裁剪与装配

`postprocess.py` 使用 learned parametrizer 的输出进行曲面和边界处理，组织 wire、构造裁剪面，再缝合与检查 solid，最终保存 STEP。官方实现以 B-spline 曲面进行拟合；它不保证识别并恢复原始 plane、cylinder 等解析类型。[postprocess.py](https://github.com/AutodeskAILab/DualBrep/blob/main/postprocess.py)、[论文 §3.4](https://arxiv.org/html/2606.31579v1#S3.SS4)

最终 B-rep 的曲面经过重新拟合，几何可能偏离输入 mesh。应同时检查有效性和几何误差，不能只判断是否产生了 `.step` 文件。

### 6.5 哪些部分可以直接复用

| 部分 | 复用结论 | 条件与限制 |
| --- | --- | --- |
| 上游 SDF/mesh 生成器 | 可以保留 | 几何已经给定时无须重复生成 |
| DualBrep 原始双场编码器、VAE、Flow 模型 | 本方案不需要依赖 | 新的表面预测器需自行训练或适配 |
| 默认标量 UDF 分割逻辑 | 可以复用 | mesh 邻接、逐三角面值、单位与阈值一致 |
| 依赖 UDF mesh 或方向的可选模式 | 不能仅靠标量直接替代 | 需要额外输出或更换分割模式 |
| `cluster.ply` → `rebuild.py` | 可以沿用接口与已有权重 | 合法标签、点/法向质量、坐标与数据分布需匹配 |
| `post.npz` → `postprocess.py` | 可以复用 | 保持中间文件结构、坐标与旋转约定 |
| 原论文的有效率 | 不能直接继承 | 必须在自己的输入几何与预测场上重新评估 |

推荐首先冻结 parametrizer，测试直接复用的效果。如果 GT labels 或高质量表面场在自己的数据上也无法良好重建，再考虑微调 parametrizer。

当前 `rebuild.py` 会跳过少于两个有效 face 分组的输入。因此，单 face 的周期曲面等情况需要额外处理，不能把“接口支持变长面片集合”理解为所有 B-rep 特例都已支持。其他复杂边界也应以实际模型配置和权重能力为准。[rebuild.py](https://github.com/AutodeskAILab/DualBrep/blob/main/rebuild.py)

**最低改动量是：新表面场预测器 + 一个导出适配层。** 不必为了接入而重新训练整套 DualBrep。

## 7. 对接文件与运行方式

### 7.1 数据对应关系

建议一个样本目录至少保存以下内容：

| 文件 | 内容 |
| --- | --- |
| `recon_sdf.ply` | 真正用于预测和分割的三角 mesh |
| `udf_g.npy` | 逐三角面中心的距离值，形状 `(N_T,)` |
| `surface_field.npz` | 自定义训练点、法向、距离标签、采样三角面索引 |
| `transform.json` | 自定义保存的坐标变换及距离单位说明 |
| `cluster.ply` | 分割结果，三角面带整数 `label` 属性 |

`surface_field.npz` 建议保存 `points`、`normals`、`triangle_id`、`udf_raw`、`udf_target`、`tau`。这些键是本方案建议，不是 DualBrep 原始 dataset loader 的现成输入格式。

不要只导出彩色点云替代 `cluster.ply`。官方读取的是 PLY 三角面属性中的 `label`，兼容 `cluster` 字段；颜色本身不构成标签。[rebuild.py](https://github.com/AutodeskAILab/DualBrep/blob/main/rebuild.py)

### 7.2 复用官方命令

以下命令假设位于 DualBrep 仓库根目录，已按官方说明配置依赖并取得 `parametrizer.ckpt`，且样本使用一致的规范化坐标。先用当前分割入口处理自己的输入：

```bash
python clustering.py surface_predictions/
```

`surface_predictions/` 下每个样本目录应具有匹配的 mesh 与 `udf_g.npy`。如果当前检出的脚本入口存在额外路径或元数据要求，可以直接调用其 `hierarchical_segmentation`，按上述 PLY 属性约定输出 `cluster.ply`；不要为满足入口而伪造体 UDF。

然后运行：

```bash
python rebuild.py \
  --input surface_predictions/ \
  --out rebuilt_brep/ \
  --rotations 3

python postprocess.py --input rebuilt_brep/ --serial
```

先用官方约定的 identity rotation `3` 完成固定预算基线，再按需要测试 `--rotations all` 的 24 个候选。旋转候选会增加成本，也会改变成功率统计；比较实验应统一候选数。[官方使用说明](https://github.com/AutodeskAILab/DualBrep#parametrization-segmented-triangle-soup--uv-grids--trimmed-assembled-b-rep)

形状目录建议使用不带下划线的唯一 ID，例如 `00000164`，以避免与官方“形状 ID + 旋转编号”的命名解析冲突。若自行封装，应保留它的旋转还原约定。

### 7.3 导出前的必要检查

1. `len(udf_g) == len(mesh.faces)`，所有值非负且有限。
2. 保存/加载 mesh 没有改变三角面顺序；完成重网格、去重或修复后必须重新查询场。
3. 字段值与 mesh 使用同一距离单位，不能把 `[0,1]` 训练数值误当距离。
4. 所有需要分割的面片已在 mesh 中，法向与朝向一致，邻接图不存在明显的跨表面连接。
5. `cluster.ply` 中确实存在整数标签属性，而不仅是 RGB。
6. 自行增加的坐标还原逻辑只执行一次，不与官方已有变换重复。

## 8. 表面场提取核心代码示例

下面示例假定两份输入已经对齐：`recon_sdf.ply` 是 SDF 提取的 mesh，`voronoi.ply` 是同一 GT B-rep 的 Voronoi 网格。代码只展示标签采样和距离查询，不包含 SDF 构建、Voronoi 工具编译及网络训练。

```python
from pathlib import Path

import igl
import numpy as np
import trimesh


def require_triangle_mesh(path):
    obj = trimesh.load(path, process=False)
    if not isinstance(obj, trimesh.Trimesh) or len(obj.faces) == 0:
        raise ValueError(f"Expected a nonempty triangle mesh: {path}")
    return obj


def query_voronoi_distance(query_xyz, voronoi_mesh):
    vertices = np.asarray(voronoi_mesh.vertices, dtype=np.float64)
    triangles = np.asarray(voronoi_mesh.faces, dtype=np.int32)
    values = igl.signed_distance(
        np.asarray(query_xyz, dtype=np.float64),
        vertices,
        triangles,
        sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED,
    )[0]
    values = np.maximum(np.asarray(values).reshape(-1), 0.0)
    if not np.isfinite(values).all():
        raise ValueError("Distance query returned nonfinite values")
    return values.astype(np.float32)


def sample_surface(mesh, count, rng):
    area = np.asarray(mesh.area_faces, dtype=np.float64)
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError("Invalid mesh area")
    tri_id = rng.choice(len(area), size=count, p=area / area.sum())
    tri_xyz = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[tri_id]]
    r1, r2 = rng.random((2, count))
    root = np.sqrt(r1)
    bary = np.stack((1 - root, root * (1 - r2), root * r2), axis=1)
    points = np.einsum("ni,nij->nj", bary, tri_xyz)
    normals = np.asarray(mesh.face_normals)[tri_id]
    return points, normals, tri_id


sample_dir = Path("surface_predictions/00000164")
mesh = require_triangle_mesh(sample_dir / "recon_sdf.ply")
voro = require_triangle_mesh("work/00000164/voronoi.ply")
tau = 0.015

# 与后处理真正使用的三角面一一对应。
centers = np.asarray(mesh.vertices)[np.asarray(mesh.faces)].mean(axis=1)
center_raw = query_voronoi_distance(centers, voro)
np.save(sample_dir / "udf_centers_raw.npy", center_raw)
np.save(sample_dir / "udf_g.npy", np.minimum(center_raw, tau))

# 基础面积采样；正式训练应另外补充边界附近与小面采样。
points, normals, tri_id = sample_surface(mesh, 32768, np.random.default_rng(0))
point_raw = query_voronoi_distance(points, voro)
np.savez_compressed(
    sample_dir / "surface_field.npz",
    points=points.astype(np.float32),
    normals=normals.astype(np.float32),
    triangle_id=tri_id.astype(np.int32),
    udf_raw=point_raw,
    udf_target=(np.minimum(point_raw, tau) / tau).astype(np.float32),
    tau=np.float32(tau),
)
```

这里输出到 `udf_g.npy` 的是 GT 场，用于首先验证后处理。当网络训练完成后，只需将 `center_raw` 的来源替换为模型对这些中心的预测，再按相同单位保存。

若查询规模很大，可以分批查询并缓存 Voronoi 网格的空间加速结构。示例使用 float32 保存标签，避免在边界误差尚未检查前引入过低精度量化。

## 9. 必须明确的限制与相应对策

| 问题 | 对本方案的影响 | 对策 |
| --- | --- | --- |
| 同一几何对应不同 face 划分 | GT 不一定由点云唯一决定 | 统一标签惯例，评估候选或等价划分 |
| seam 和退化边 | face 间分割场不编码全部 CAD 拓扑 | 由周期曲面参数化和重建阶段单独处理 |
| 薄壁导致低值平台 | 低 UDF 不总是近边，固定阈值可能删除面内部 | 用 GT 场检查阈值失败，再评估自适应分割或 affinity |
| 小面、短边采样不足 | 连续场有信息，离散样本却可能漏掉 | 更密查询、边界采样和分辨率分组评估 |
| mesh 缺面、错连或拓扑已损坏 | 场不能在不存在的表面上生成面片 | 改善几何输入，或增加独立几何修复阶段 |
| 参数化模型存在域偏移 | 分割正确仍可能重建失败 | 先测 GT labels，再决定是否微调 parametrizer |
| 只优化平均标量损失 | 小范围断边可导致全局失败 | 同时评价边界、分割与有效 solid |

特别地，**只把体域预测改为表面预测，并不会自动消除原 UDF 的薄壁低值问题**。它改变的是模型预测范围；若保留原标签定义，其几何性质也会保留。

相对于完整体场，表面场可以减少目标采样和查询范围，但不能直接声称获得固定的 $R^3\to R^2$ 倍数加速：官方模型采用隐式查询，实际成本还取决于 token 数、采样策略和网络结构。训练阶段构造 GT Voronoi 网格的成本也仍然存在。

## 10. 最小验证实验与判定标准

建议先选取覆盖棱柱、圆柱、圆角、薄壁、孔洞及自由曲面的少量 GT CAD，跑通以下对照。所有方案使用相同 mesh、相同 parametrizer 权重、相同旋转候选数与装配预算。

| 实验 | 输入到分割/重建阶段的内容 | 能定位的问题 |
| --- | --- | --- |
| A：GT 面标签 | GT labels → parametrizer → 装配 | 后处理在当前数据上的参考表现，不是理论上界 |
| B：GT 表面场 | 精确查询的 $g$ → 分割 → 同一后处理 | 场采样、阈值与分割是否丢失面片 |
| C：官方预测场的表面值 | 原模型查询值 → 相同后处理 | 建立教师/原模型参考 |
| D：直接表面场生成器 | 无 VAE 的 DDPM/FM 采样 $\hat g$ → 相同后处理 | 比较直接生成场的质量与采样成本 |
| E：几何加单视图 | 在 D 中加入图像条件 | 图像是否提供额外有效信息 |

GT 标签只用于参考实验和评估。不得在 D/E 的推理中用 GT 数量挑阈值、用 GT 边修补断边或根据 GT 选择候选。

D 直接训练生成模型，不以回归预训练为前置条件。DDPM 与 FM 使用匹配的数据划分、几何条件和网络容量，报告采样步数/函数评估次数。分别报告单候选和固定 $K$ 候选预算的结果；候选排序只能使用推理时可获得的内核有效性、几何一致性等信息。若另报基于 GT 的 oracle best-of-$K$，必须明确其仅为分析指标。A/B 是无需训练新预测器的接口对照，可以与生成模型开发并行开展。

建议记录三层指标：

1. **场值层：** 未饱和区 MAE、GT 边界附近误差、不同厚度和面尺度下的误差；全局 MAE 只作辅助。
2. **分割层：** 边界 precision/recall/F1、face 匹配指标、过分割/欠分割率、小面召回率和未归属区域比例。
3. **B-rep 层：** CAD 内核有效性、预期 solid 数和闭合性、相对输入 mesh 的几何误差、face-edge/edge-vertex 拓扑指标，以及耗时、显存和失败样本比例。

诊断顺序是：

- A 已经较差：优先检查 parametrizer、输入分布和 CAD 装配，不应先归咎于表面场。
- A 较好而 B 较差：优先检查 GT 场定义、对齐、分辨率、阈值和薄壁问题。
- B 较好而 D 较差：重点改进预测器、采样和边界监督。
- D/E 能保持重建质量，同时减少表面场模块的训练/推理成本：才支持这项改造的工程收益。

本方案第一阶段目标是：**固定上游几何与后处理，不使用 VAE，直接训练条件 DDPM 或 Flow Matching，验证“生成表面 Voronoi-UDF”能否替代“预测体 UDF 后再在表面查询”。** 它保留了原方法可变数量面片的连续边界表达，并将新增学习任务集中到已有几何表面上。

## 参考来源

- [DualBrep 论文](https://arxiv.org/abs/2606.31579)：场定义、生成框架及 learned rebuilder。
- [DualBrep 官方仓库](https://github.com/AutodeskAILab/DualBrep)：数据工具、运行入口和依赖说明。
- [prepare_implicit.py](https://github.com/AutodeskAILab/DualBrep/blob/main/prepare_implicit.py)：GT Voronoi 网格与距离标签查询。
- [ae_reconstruct.py](https://github.com/AutodeskAILab/DualBrep/blob/main/ae_reconstruct.py)：三角面中心查询及数值反归一化。
- [config.yaml](https://github.com/AutodeskAILab/DualBrep/blob/main/config.yaml)：当前截断尺度和重建设置。
- [clustering.py](https://github.com/AutodeskAILab/DualBrep/blob/main/clustering.py)：标量阈值与邻接图分割。
- [rebuild.py](https://github.com/AutodeskAILab/DualBrep/blob/main/rebuild.py)：标签 mesh 的读取、面片采样与 parametrizer 接口。
- [postprocess.py](https://github.com/AutodeskAILab/DualBrep/blob/main/postprocess.py)：CAD 拟合、装配与候选输出。

以上链接指向公开版本；实际实施时应记录自己的仓库 commit、权重版本和后处理配置，以确保实验可复现。
