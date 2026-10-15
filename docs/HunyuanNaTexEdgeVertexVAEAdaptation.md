# 将 Hunyuan3D / NaTex VAE 改造为边、顶点距离场 VAE 的方案

日期：2026-10-14。状态：代码检查后的设计方案，尚未实施或训练。本次仅新增本文档。

目标代码目录：`/opt/data/private/yihengxu/Hunyuan3D/hy3dshape`。本文中的 `H/` 指该目录，其 Python 包位于 `H/hy3dshape/natex/`。检查的是本地 NaTex 贴图实现及当前工作区，不将其等同于 NaTex 作者的官方实现；仓库 HEAD 为 `08ae0e4`，工作区另有实验相关修改。

数据来源：本仓库 `edge_vertex_preprocessing` 生成的 ABC 三场样本。重建目标遵循 [BandLabelTopologyRecovery.md](BandLabelTopologyRecovery.md)，数据契约参考 [ABCEdgeVertexUDFPreprocessing.md](ABCEdgeVertexUDFPreprocessing.md)。

## 1. 推荐的整体改造

保留“几何条件 latent + 待生成属性 latent + 坐标查询 decoder”的双分支结构，将颜色属性改成联合的边／顶点距离场：

$$
\text{已知 SDF }s\longrightarrow (Z_g,A),\qquad
(s,d_E,d_V;Z_g,A)\xrightarrow{\text{field encoder}}q_\phi(Z_f),
$$

$$
(Z_g,Z_f,x,s(x))\xrightarrow{\text{field decoder}}
(\widehat d_E(x),\widehat d_V(x)).
$$

生成阶段只运行几何 encoder，Flow 生成 $Z_f$，再查询两个场。**SDF 仍是外部给定的几何条件，最终重建直接使用输入 SDF，不要求 Flow 重新生成 SDF。**

第一版采用以下明确选择：

1. 使用一个联合 field posterior，而不是两个相互独立的 VAE；共享 latent、分开的 E/V 输出头和损失。
2. 保留表面锚点与 token 一一对应关系；锚点只能由已知几何产生，不能从 GT 顶点或边采样。
3. 几何 encoder 读取几何表面点和 SDF 空间样本；field encoder 读取另一批混合空间点及其 SDF/E/V 值。两批输入点不必相同。
4. 对 E/V 使用保留全距离信息的可逆 log 编码，停止把 `.02` 硬截断值作为主重建目标。
5. 连续坐标 decoder 保留；不先改成固定分辨率的三维体素 VAE。
6. 设置 E/V 存在性输出，正确覆盖“有闭合边、没有必要顶点”等样本。
7. 先验证 VAE 能重建可用的场和拓扑，再缓存 latent 训练 Flow。

这一路径复用现有 cross-attention、latent Transformer、Gaussian posterior、几何引导查询、缓存及 latent Flow 的主要机制。改造重点是输入语义、数据采样、场编码、decoder 约束和验收方法。

## 2. 代码检查：哪些可以复用，哪些必须改

### 2.1 本地实际架构

项目根 README 主要介绍原始 Shape 流程；用户当前的贴图工作实际位于 `natex/`。已检查的关键实现如下。

| 当前模块 | 检查到的行为 | 新方案处理 |
|---|---|---|
| `models/autoencoders/model.py::GeometryAwareColorVAE` | 几何输入为 `Fourier(pos)+normal`；颜色 KV 为 `Fourier(pos)+normal+RGB` | 复用双分支主体，替换输入投影和目标语义 |
| `encode_geometry` | 从表面点选择 anchors，cross-attention 后生成 $Z_g$ | 保留几何锚点，增加独立 SDF KV 输入 |
| `encode_color` | Q 来自 $Z_g$；要求与几何 encoder 使用相同 `pos` | 改为 field encoder，保持 token 对齐，允许不同 KV 位置 |
| `TokenDiagonalGaussian` | FP32 posterior 矩、logvar clamp、KL 对 token 与 channel 取 mean | 复用；不能照搬使用 KL sum 的权重 |
| `prepare_decode` | 分别解码两组 latent memory，供反复查询 | 复用缓存方式，`memory_c` 改为 `memory_f` |
| `FeatureQueryDecoder` | 连续坐标的 cross-attention 查询，线性输出头 | 复用；不受输出体素格点数量限制 |
| `CoupledColorQueryDecoder` | 几何 feature 构成颜色 query；构造时严格要求颜色输出 3 通道 | 改成场查询 decoder，不能只改 `out_channels` 而忽略检查 |
| `training/losses.py::vae_objective` | 表面／近表面 RGB MSE、表面 UDF MSE、两支 KL | 重写为 SDF、E/V 分层距离、存在性及可选几何约束 |
| `data/surface_color.py` | 读取表面 RGB 池与两类 queries，输入和监督分开抽样 | 复用 LRU／确定性验证／分割管理；替换数据 schema 和采样规则 |
| `models/anchors.py` | random 或 FPS，显式共享 anchor plan | 保留；几何分支负责唯一一次选择 |
| `data/latent_cache.py` | 几何 mean、颜色 posterior、anchor_xyz、统计和 provenance | 改为 field posterior 语义并重新缓存 |
| `models/diffusion/flow_matching.py` | $z_t=(1-t)\epsilon+t z_c$，速度目标 $z_c-\epsilon$ | 数学流程不变，$z_c$ 换成 $z_f$ |
| `configs/natex_flow_unconditional.yaml` | 已有关闭图像条件、保留几何条件的 Flow 配置 | 可作为 geometry-only 结构生成入口 |
| `pipeline.py` | 颜色导出时 clamp 到 `[0,1]`，整体使用 `no_grad` | 新增距离场查询入口；不能复用 RGB 导出和梯度限制 |

这里的旧 `pred_udf` 是**到物体表面的距离**，不是到 CAD 边的距离。新方案中该分支应改成 SDF 辅助监督，E/V 由属性分支输出。

`latent_decoder_layers` 控制的是 latent memory 的 Transformer 深度，不是逐坐标 query decoder 的深度。后续容量实验必须区分这两个位置。

### 2.2 为什么不能只做 RGB → 两通道替换

只修改最后一层会遗漏五个实质问题：

- 原编码器主要观察表面颜色；新的场需要在三维空间定义并接受监督。
- 边、顶点附近的低距离区域远小于整片表面，均匀 MSE 容易忽略它们。
- `.02` 截断后的大片常数区域会掩盖结构缺失。
- 空顶点集不是一张数值恒为 `.02` 的真实距离场。
- 旧的“输入点一致”检查与新的体空间 field KV 不兼容；直接取消所有对齐检查又会破坏 Flow 条件对应关系。

正确做法是保留 **latent token 的几何对齐**，调整输入点集的契约。

### 2.3 现有贴图实验能提供什么证据

本地 `H/../docs/Overfit_Sampling_Tokens_20261013.md` 与 `Overfit_Sampling_Capacity_20261013.md` 已记录输入点数、query 数、token 数、宽度和 latent decoder 深度的独立实验。这种分开定位瓶颈的实验方法可以复用。

贴图重建指标、单物体实验、`latent_dim=256` 或 KL=0 的设置，不能直接证明新的 UDF VAE 适合多物体 latent 生成。本文不把这些实验参数当作已经验证的 UDF 配置。

## 3. 场的定义、坐标和数据契约

### 3.1 三场语义保持不变

设物体边界为 $M$，过滤后的目标边集合为 $E$，必要顶点集合为 $V$：

$$
s(x)=\operatorname{SDF}_M(x),\quad
d_E(x)=\min_{y\in E}\|x-y\|,\quad
d_V(x)=\min_{v\in V}\|x-v\|.
$$

SDF 沿用现有内负外正约定。边是完整有限曲线，包含端点。seam 不进入训练目标；光滑完整闭环可以有边而无必要顶点。保留相切、共面 face 之间的真实目标边；不改为仅锐边检测。

当目标非空且 $V\subseteq E\subseteq M$ 时，有：

$$
|s(x)|\le d_E(x)\le d_V(x).
$$

这可以用于软约束和诊断，但不能替代真实距离监督，也不能直接保证重建出正确拓扑。

### 3.2 读取原始数据，不走旧截断分支

现有 schema 为 `edge_vertex_udf_v1`，**固定通道顺序 `[sdf,dE,dV]`**。输入点与三个值同位置保存：

```text
query_points       [Q,3]
query_fields       [Q,3] = [sdf,dE,dV]
surface_points     [Ns,6] = [xyz,normal]
edge_points        [Ne,6] = [xyz,tangent]
vertex_points      [Nv,3]
has_edges / has_vertices
gradient_E/V, gradient_valid_E/V, nearest_tie_E/V
nearest_E/V_id, closest_E/V_points, query_source
```

使用 `evudf.io.load_fields(..., normalize=False)` 或等价的 schema 检查。非空场原始距离没有截断；`normalize=True` 才按 `.02` 截断并归一化，不能作为本方案的 loader 主路径。

此次复查 ABC10 已有结果，非空 V 的两个样本在 `.02` 截断下会分别有 **89.03% 与 88.91%** 的 query 饱和。该比例来自现有混合采样分布，不代表整个空间的体积比例。原始距离场本身是稠密标量函数，稀疏的是零集及硬截断后仍有变化的区域。

### 3.3 统一坐标：避免再次归一化

当前 ABC 预处理默认 `bounding=.9`，将 CAD 包围盒最长边归一化为约 `1.8`；查询域是 `[-1,1]^3`。NaTex 的 `geometry.py` 默认 `COORD_SPAN=1.9`。

新数据入口必须以 ABC 的 `world_to_normalized` 为准，不再次调用 NaTex 的 `make_normalization`。点、SDF、E/V、anchor 和重建阈值共用同一个尺度。若以后主动变更尺度，距离值必须同步缩放，并创建新的数据／模型版本。

旋转增强也必须保证查询域约定成立。第一版不加尺度增强；可先使用保持立方域的轴置换与适当旋转，任意旋转需重新处理域外点。法向和梯度同步变换，不能旋转坐标后保留旧方向标签。

### 3.4 训练 batch 的建议接口

以下是拟新增的接口，不是现有可运行 API。

| 字段 | 形状 | 用途 |
|---|---|---|
| `geo_pos / geo_normal` | `[B,Ng,3]` | 从已知几何获得的表面条件 |
| `geo_query_xyz / geo_sdf` | `[B,Ns,3]` / `[B,Ns,1]` | 仅依赖 SDF 的体空间条件 |
| `field_input_xyz` | `[B,Nf,3]` | field posterior 的混合空间输入 |
| `field_input_values` | `[B,Nf,3]` | 原始三场值，进入网络前执行指定编码 |
| `field_presence` | `[B,2]` | GT E/V 存在性，只供 posterior 输入和监督 |
| `query_xyz / query_values` | `[B,K,3]` / `[B,K,3]` | decoder 的共同查询坐标和三场监督 |
| `distance_valid` | `[B,K,3]` | 排除未定义的空集距离等 |
| `gradient_valid / gradient_gt` | 按 E/V 分别保存 | 选用的导数监督 |
| `anchor_idx / anchor_xyz` | `[B,M]` / `[B,M,3]` | 来自 `geo_pos` 的唯一 anchor plan |
| `source / nearest_id / zero_kind` | 训练辅助 metadata | 采样、分组损失和评测，不作为生成条件 |

**数据同位置保存三通道，不意味着两个 encoder 必须观察同一批位置。** Dataset 从同一个合法数据池中抽取不同角色的子集即可。

## 4. 距离表示：推荐可逆压缩，不做窄带硬截断

### 4.1 E/V 主表示

对非空通道采用：

$$
u_c(x)=f_c(d_c(x))=
\frac{\log(1+d_c(x)/\sigma_c)}{L_c},\qquad
L_c=\log(1+D/\sigma_c),\quad c\in\{E,V\}.
$$

逆变换：

$$
d_c(x)=\sigma_c\operatorname{expm1}(L_cu_c(x)).
$$

首轮建议固定 $D=2\sqrt3$，$\sigma_E=\sigma_V=.02$，再消融 `.05`。在查询和目标均位于 `[-1,1]^3` 的前提下，$D$ 是统一的保守距离上界；它不是已有数据的最大值，也不是每个实例单独统计的最大值。

优势是零点不变、排序不变、近零误差得到更高权重、远处仍保留变化。阈值可精确换算为 $u_\tau=f(\tau)$。例如上述参数下 `.002`、`.01`、`.02` 约对应 `.0185`、`.0786`、`.1343`，不会像旧方案那样从 `.02` 开始全部变成 1。

仅有 log 编码仍不足以保证小顶点被保留：还必须做近场和逐实例的采样／损失平衡。

### 4.2 SDF 表示

几何分支建议使用有符号 log 编码：

$$
u_s=\operatorname{sign}(s)\frac{\log(1+|s|/\sigma_s)}{\log(1+D/\sigma_s)},
\qquad \sigma_s=.05\text{ 作为起点}.
$$

这是几何条件特征和辅助监督的编码；最终重建仍查询原始输入 SDF。SDF 损失同时包含近表面的物理距离误差，防止全局拟合掩盖表面偏差。

### 4.3 输出激活与近零精度

主方案输出无激活的 $\widehat u_E,\widehat u_V$，用线性／小 MLP 头回归编码值，并对负值施加软惩罚。不直接使用 RGB sigmoid：sigmoid 难以达到严格零值，且近端容易梯度饱和。

训练中的逆变换使用未裁剪输出，以便对越界预测持续产生梯度。对过大输出监测并处理数值异常；不能静默 clamp 后宣称距离准确。对外的物理 UDF 接口可以返回 `max(decoded_distance,0)`，但必须同时保留未裁剪 residual 与梯度有效性信息。

负值裁剪会形成假零平台，因此几何细化不能只对裁剪后的值求梯度。负值率、平台大小和零集位置都应进入验证。Softplus 非负距离头可作为对照，但同样需要验证其近零偏置，不能仅以“非负”认定更好。

### 4.4 其他表示的定位

| 表示 | 适合的角色 | 不作为首版主方案的原因 |
|---|---|---|
| 原始距离 + 距离分层损失 | 必做基线 | 远近量级差异较大，需仔细平衡 |
| `.02` 截断距离 | 对照／局部辅助目标 | 丢失远场，V 通道饱和比例高 |
| Gaussian 热度 $\exp[-d^2/(2\sigma^2)]$ | 可选粗候选定位辅助头 | 窄核仍稀疏，宽核不能保证近邻顶点分离，远处逆变换不稳定 |
| 多尺度截断／热度 | 后续改善候选召回 | 增加输出与权重，需和主距离通道分开评价 |
| 三维体素 VAE | 需要规则体积输出时的备选 | 引入空间离散瓶颈，改动范围明显大于当前连续查询结构 |

若使用热度，辅助目标应由最近距离直接计算，等价于相同宽度的各顶点 Gaussian 取 max；不能未经说明改成求和，否则目标语义及峰位置性质都会改变。

## 5. 几何分支：SDF 条件如何进入 VAE

### 5.1 保留几何 token，引入两个条件来源

表面点提供局部形状和法向；空间 SDF 值提供内外及离表面距离。推荐分别投影后拼成 geometry KV：

$$
K_g^\mathrm{surf}=P_\mathrm{surf}[\gamma(p),n]+e_\mathrm{surf},\qquad
K_g^\mathrm{vol}=P_\mathrm{vol}[\gamma(x),u_s(x)]+e_\mathrm{vol}.
$$

$$
K_g=[K_g^\mathrm{surf};K_g^\mathrm{vol}],\qquad
Q_g=\operatorname{gather}(K_g^\mathrm{surf},A).
$$

后续复用 `geo_cross → geo_latent → pre_kl_g`。几何查询 decoder 的输出头从 surface UDF 改为 signed SDF 编码。

这里的 `e_surf/e_vol` 仅区分两种几何输入类型，不包含 face、edge、vertex ID。

若要先做最小实现，可保留原来的 `pos+normal` geometry encoder，仅替换成 SDF 监督、由 SDF 提取输入表面，并在 field decoder 输入 $s(x)$。此版本作为简化消融；主方案增加 SDF KV，确保空间 SDF 信息确实进入几何 latent。

### 5.2 防止条件信息泄漏

几何 encoder、anchors 和 Flow 条件都只能使用生成时可获得的信息：

- 可使用：输入 SDF、其提取表面、可靠的表面法向、仅由几何确定的采样。
- 不可使用：GT E/V 值、GT 必要顶点坐标、face/edge ID、GT presence、按 GT 顶点聚集得到的 anchors。
- 不要直接把混合 `query_points` 全部当几何 KV：其中 near-edge/near-vertex 的**坐标分布本身**已经包含目标结构信息。

原始 `surface_points` 使用 CAD 分面网格并含每面最小采样配额。快速基线可以读取位置和法向，但正式条件管线应从输入 SDF 提取的表面统一面积采样，不使用 GT 面的配额或边界。这样训练与实际生成时条件来源一致，也减少 CAD 分面信息从采样分布间接进入几何分支。

`geo_query_xyz` 采用固定比例的均匀空间点与几何近表面点；选择规则不依赖 E/V 是否存在。已有三场样本可用于初期监督池，正式版本应单独固定条件采样规范。

应为条件单独定义 `SDFCondition.value(x)`、`gradient(x)`、`sample_surface(seed)` 和 `fingerprint` 契约。若生产输入是体素插值 SDF，训练也应评测相同插值和分辨率；若是神经 SDF，则记录其模型和误差。不能训练时始终使用精确 CAD 条件、测试时改成低分辨率 SDF，却把全部性能下降归因于 field VAE。GT 三场监督与实际可用的条件 SDF 可以分开保存，必须共用 normalization，并记录两者的偏差。

### 5.3 几何分支的训练和冻结

推荐先训练 SDF 几何分支，再冻结，随后训练 field VAE：

1. 几何预训练：重建 SDF；可以保留原 Gaussian/KL 结构，明确其 mean reduction。
2. field 训练：几何条件使用 $Z_g=\mu_g$，与现有 latent cache 和 Flow 推理行为一致。
3. 若后续发现几何特征限制 UDF 重建，再低学习率联合微调；仍不向几何输入加入 GT 拓扑信息。
4. 任何几何权重变化后，重建全部 geometry/field latent 缓存与统计。

冻结 geometry decoder 参数不等于禁止对查询坐标求导。梯度监督时不能把整个几何查询路径包进 `no_grad`，否则会丢失 $h_g(x)$ 对 $x$ 的导数。

## 6. Field encoder：将两类距离联合压缩

### 6.1 输入与 query

推荐 field KV 输入：

$$
K_f=P_f[\gamma(x),u_s(x),u_E(x),u_V(x),p_E,p_V],
$$

其中 $p_E,p_V$ 是训练样本的存在性。体空间点没有唯一“表面法向”，因此不能继续把 `normal` 字段机械搬入所有 field KV。第一版不输入梯度、最近点或实例 ID；这些只用于监督或采样。

field query 保持与 geometry token 一一对应：

$$
Q_f=W_g Z_g+W_A\gamma(A),\qquad
H_f=\operatorname{Transformer}(\operatorname{CrossAttn}(Q_f,K_f)),
$$

$$
(\mu_f,\log\sigma_f^2)=\operatorname{preKL}_f(H_f),
\qquad Z_f\in\mathbb R^{B\times M\times C_f}.
$$

位置项 $W_A\gamma(A)$ 是建议新增的显式对应信息，可与仅使用 $W_gZ_g$ 的现有方式做消融。`C_f=64` 为首版起点，与现有 Flow target channels 一致。

### 6.2 修改对齐检查的正确边界

现有 `assert_aligned_encodings(geo,pos)` 约束输入点相同。新接口应替换成：

- `geo.anchor_idx` 仍索引 `geo_pos`，`anchor_xyz` 必须与 gather 一致。
- $Z_g$、$Z_f$ 必须共享 batch、token 数、token 顺序和 anchor plan。
- `field_input_xyz` 允许与 `geo_pos` 不同，数量也可不同。
- decoder 只接收 latent、共享 anchors、已知 SDF，不读取 field 输入点的隐藏缓存。

允许 posterior 观察 GT 边／顶点附近的点是合理的，因为它是在编码目标。将这些 GT 位置直接传给生成阶段 decoder 或 RoPE 则不合理：生成时并不知道它们。

### 6.3 为什么优先联合 E/V latent

E/V 描述同一个边网络，联合 latent 能表达“顶点落在边上”“一条闭环没有顶点”等关系。两个独立 VAE/Flow 分别采样会增加不一致的自由度。

第一版使用共享 field memory，加两个小的独立输出头，并分别归一化 E/V 损失。若 V 明显受 E 主导，依次尝试：

1. 加强逐顶点采样和 V 的近场损失。
2. 给 E/V 各加一层轻量 decoder 分支。
3. encoder 中按 near-E/near-V 分两次 attention，再在同一组 anchors 上融合。
4. 仍不足时再考虑 $Z_f=[Z_E,Z_V]$ 的联合生成；此时 latent channel 及 Flow 输入输出配置可能需要变更。

不要预先把“每个 token 对应一个顶点或一条边”写进契约。当前 token 是几何条件锚点上的表示单元，拓扑实体数量可以随样本变化。

### 6.4 为什么暂时不把 latent 改成三维网格

E/V 的目标集合都位于已知物体表面，三维空间的距离由这些集合决定。因此可以让表面上的 latent anchors 承载结构，再由带空间坐标的 decoder 表达整个体空间距离。field KV 本身仍覆盖体空间，并不要求每个空间位置都有独立 latent token。

这一点是选择表面 anchors 的建模依据，不是有限 token 一定足够的保证。若远场或深凹区域持续失败，可增加仅由几何确定的体空间 anchors，与 geometry/field/Flow 同步对齐；这属于后续结构消融。直接改成 voxel/triplane 会增加另一套编码与插值设计，首版没有必要同时引入。

## 7. Field decoder：连续空间查询和两个距离头

### 7.1 保留几何引导，并加入坐标与 SDF 的直接通路

复用两个 latent decoder 得到 $M_g,M_f$。每个坐标 $x$ 的查询为：

$$
h_g(x)=\operatorname{GeoQuery}(\gamma(x),M_g),
$$

$$
q_f(x)=W_hh_g(x)+W_x\gamma(x)+W_su_s(x),
\qquad h_f(x)=\operatorname{FieldQuery}(q_f(x),M_f),
$$

$$
\widehat u_E(x)=\operatorname{Head}_E(h_f(x)),\qquad
\widehat u_V(x)=\operatorname{Head}_V(h_f(x)).
$$

直接坐标通路避免所有空间高频信息都经过 geometry feature 的单一路径压缩。显式 $s(x)$ 通路利用已知条件；它仍不足以独立确定 B-rep 的划分。

E/V 头可先各使用 `Linear(width,width/2) → SiLU → Linear(...,1)`；再与原单 Linear 头比较。输出头必须能处理共享任意查询坐标，不再区分 RGB surface/near 接口。

### 7.2 SDF 条件接口需要支持导数

若 decoder 使用 $s(x)$，在计算 $\nabla_x\widehat d$ 时必须包含这一路的导数。选择之一：

- 使用可微的 SDF 网络／网格插值器；
- 为外部 SDF oracle 提供正确的梯度接口或自定义 autograd；
- 后处理对**完整复合查询函数**做有限差分；
- 第一版不启用解析梯度损失，先用相邻点差分验证，再接入完整导数。

把预先查询的 `sdf_values` 当常数送入 decoder，再对 `xyz` 求 autograd，会漏掉这一路的链式导数。该结果不能当完整的三维 UDF 梯度。

### 7.3 Presence 输出与空集合

增加一个全局 presence head，例如对 $M_f$ 与 $M_g$ 的 pooled feature 回归两个 logits。它由 latent 解码得到，不需要向 Flow 增加全局 token 或额外噪声维度。

GT presence 在 field encoder 中是目标的一部分；生成时由生成的 $Z_f$ 预测，不能从测试 metadata 偷取。

空通道处理规范：

1. 读取 `has_edges/has_vertices` 后，先标记对应真实距离无定义。
2. 不把磁盘 `.02` 占位值进行 log 编码后作为真实标签。
3. field encoder 对空通道使用固定 sentinel，例如 `u=1`，并显式输入 presence=0。
4. 物理距离、梯度、零集、包含关系损失对该通道屏蔽。
5. 使用 presence BCE；可另加低权重的空通道 `u=1` 规范化损失，但明确它是约定，不是距离。
6. 导出时返回存在状态与有限占位数值，重建按存在状态跳过该通道。

含义相近但不同的“空集”和“有效场远离查询位置”由 presence 区分。$p_V=1,p_E=0$ 不符合目标定义，应通过训练惩罚／推理一致性检查处理。

### 7.4 精度与缓存

可复用 `decode_cache_scope` 缓存同一 decode state 的 latent KV；换物体、换 latent 后必须清理。不能复用贴图导出端的 RGB clamp、颜色空间转换或 UV 烘焙。

骨干可用 BF16。Fourier 坐标、距离编码／逆变换、posterior、loss reduction 建议 FP32；接近重建阈值的查询头和梯度细化需实测 FP32 与 BF16 差异。不能仅把 BF16 输出转换成 FP32，就认为前面的数值误差消失了。

连续 decoder 可以在任意坐标查询，但表示能力仍受输入点覆盖、latent 容量和网络频率限制。“没有固定 voxel 分辨率”不等于“有无限几何精度”。

## 8. 采样：分别解决输入覆盖和输出监督

### 8.1 四个数量必须分开配置

| 数量 | 作用 | 增大后的主要成本 |
|---|---|---|
| $N_g+N_s$ | 几何 encoder 看到多少条件信息 | 几何 cross-attention |
| $N_f$ | posterior 能观察多少 E/V 结构 | field cross-attention |
| $M$ | 多少 latent tokens 保存整个场 | 两支 latent attention 与 Flow 成本 |
| $K$ | 每步监督多少坐标 | decoder 查询与损失 |

仅增加 $K$ 无法补回 encoder 从未观察到的顶点；仅增加 $M$ 也无法修复缺失的近顶点监督。field encoder 输入与 query 应独立抽样，验证 query 尽量不与 encoder 输入重合。

### 8.2 第一版混合比例

field encoder 与 decoder 各从合法三场池独立抽样，可先用下面的比例，再按结果调整：

| 来源 | 比例起点 | 目的 |
|---|---:|---|
| 均匀空间 | 20% | 保留全局距离，约束虚假零集 |
| 表面点 | 20% | 服务面隔离带和标签恢复 |
| 近表面空间点 | 10% | 表面附近三维变化 |
| 近边点 | 25% | 边曲线距离和方向变化 |
| 近顶点点 | 25% | 防止 V 通道被忽略 |

这些是训练子集比例，不要求立刻重做已有 200k 数据池。已有 near-V 池可以支持每步更高比例的抽样，但它的独立点数量和每个顶点的覆盖必须统计。

目标不存在时，相应 field 输入／监督配额按固定规则转移给 uniform 和 near-surface；不要同时改变几何条件的采样规则。所有点仍查询／监督所有存在的通道，而不是“near-E 点只看 E”。

对 `query_source=5/6` 的已有 edge/vertex 零集检查点，另设小批监督配额。具体 source 数值需按 metadata 名称映射，不把未经 schema 校验的数值硬编码。

### 8.3 用实际距离分层，而不只看来源标签

近某个顶点采样的点，最近目标未必还是这个顶点；near-edge 点也可能落在另一条边附近。因此损失分层以真实 $d_E,d_V$ 为准，例如：

```text
[0, .005), [.005, .02), [.02, .1), [.1, D]
```

每个通道分别分层。V 近场和 E 近场不能合并成一个 `min(dE,dV)<tau` 的大组，否则大多数 E 近场会再次压倒 V 近场。

上述阈值是归一化坐标中的实验起点，后续应围绕实际重建的 $\tau_E,\tau_V$ 增加窄的验证区间。

### 8.4 逐实体覆盖和多尺度扰动

在训练采样器中可以读取 `nearest_E_id/nearest_V_id`，均衡长短边和各顶点的监督数量；这些 ID 不进入神经网络和重建接口。

- 近顶点：逐顶点抽取多半径、各方向的点，而非只使用表面附近的某个方向。
- 近边：按弧长覆盖各边，再抽多个扰动尺度；保留端点附近和短边的最小配额。
- 高频近邻结构：单独记录最小点距／边距，评估是否被网络合并。
- 近顶点同样使用最近完整目标集的距离，不能以扰动半径代替真实 $d_V$。

已有噪声尺度为 `.0002/.001/.005/.02`。若要增加 `.05/.1` 等中场样本，必须对新坐标重新查询正确标签；不能给旧坐标加噪后沿用旧值。CAD edge 标签应继续使用已有有限曲线 oracle，不能退化成到稀疏 `edge_points` 的最近邻距离。

### 8.5 不让重复点伪装成覆盖提升

现有 `surface_points` 池为 10000 点，约 32000 个 on-surface query 从中重复抽样。将 encoder 配成 32768 点或将 surface query 加倍，可能主要增加重复行。

第一版 `Ng=8192` 可以使用当前池；正式条件采样从 SDF 表面独立生成。训练／验证应按坐标去重后分割，不能把相同坐标的复制行随机拆到两边。更可靠的 field 评测使用新坐标和新 oracle 标签。

## 9. 损失函数设计

### 9.1 主距离损失：通道、物体、距离层分别归一化

记 $B_{icr}$ 为物体 $i$、通道 $c$、距离层 $r$ 中的有效查询集合：

$$
L_{icr}=\frac{1}{|B_{icr}|}\sum_{x\in B_{icr}}
\rho(\widehat u_c(x)-u_c(x)),
$$

$$
L_c=\operatorname{mean}_{i:p_{ic}=1}
\frac{\sum_{r:B_{icr}\ne\emptyset}\alpha_r L_{icr}}
     {\sum_{r:B_{icr}\ne\emptyset}\alpha_r}.
$$

$\rho$ 可先使用 Smooth L1；原 MSE 作为对照。初始 $\alpha=[4,2,1,1]$，E/V 各有独立的 $L_c$，总权重先均为 1。不能把所有通道和所有 query 合并后只做一次 mean。

某物体的某层为空时跳过该层并重新归一化；整个 batch 没有 V 时，V 距离损失合法为零，presence 损失仍启用。现有 `masked_mse` 对全空集合直接抛错，不能不作修改地用于这些合法空通道。

### 9.2 近场物理距离项

编码空间的好结果不自动等于物理阈值可靠。对 $d_c<\tau_\mathrm{near}$ 再加：

$$
L_\mathrm{metric,c}=
\operatorname{mean}\rho\left(
\frac{\widehat d_c-d_c}{\tau_\mathrm{near}}
\right),\qquad \tau_\mathrm{near}=.02\text{ 起步}.
$$

物理距离在 FP32 中从输出逆变换，训练不先 clamp。该项在各物体上归一化，再平均。

### 9.3 零集位置项

对真实目标顶点 $v$ 和真实边上的点 $e$：

$$
L_0=\operatorname{mean}_{v\in V}|\widehat u_V(v)|
+\operatorname{mean}_{e\in E_\mathrm{sample}}|\widehat u_E(e)|
+\operatorname{mean}_{v\in V}|\widehat u_E(v)|.
$$

实际使用原始保存坐标的 oracle 值作为标签，接受 float32 坐标量化和 CAD 容差内的小非零值，不强行改写所有标签为数学零。mesh 表面点也不自动拥有严格 `sdf=0`。

该项提升召回，却不能排除其它地方的伪零；uniform 和中远场损失必须保留。E 零集按边均衡，V 零集按顶点均衡。

### 9.4 梯度监督与 Eikonal：第二阶段再加

在距离可微且最近目标唯一的位置，使用已有 `gradient_valid` 和 tie 标签监督：

$$
L_\nabla=\operatorname{mean}\|
\nabla_x\widehat d_c-g_c^*\|^2.
$$

对 log 编码必须执行链式变换：

$$
\nabla_x\widehat d_c
=\sigma_c L_c\exp(L_c\widehat u_c)\nabla_x\widehat u_c
=(\sigma_c+\widehat d_c)L_c\nabla_x\widehat u_c.
$$

不能对 $u_c$ 直接施加单位梯度 Eikonal。E/V 的零集、最近对象切换处和某些对称位置本来不可微；这些位置不施加单位梯度约束。即使监督 mask 有效，距离过小、数值噪声较大的点也应排除。

若缺少可信梯度标签，可在有效子集上用 $\bigl(\|\nabla\widehat d_c\|-1\bigr)^2$ 作弱正则；已有真实梯度时，Eikonal 多半是辅助诊断，不必两项同时使用大权重。

训练该项涉及对坐标导数再反传到模型参数。第一版只取 256–1024 个 gradient query，检查 attention 后端是否支持所需高阶 autograd，必要时对该路径切换到兼容实现；不能假定 BF16 fused attention 的二阶反传可直接工作。

### 9.5 场之间的包含关系

针对存在的通道，以物理单位约束：

$$
L_\mathrm{inc}=\operatorname{mean}
\operatorname{ReLU}(|s|-\widehat d_E-\epsilon_s)^2
+\operatorname{mean}
\operatorname{ReLU}(\widehat d_E-\widehat d_V-\epsilon_{EV})^2.
$$

两项分别按 E 有效、E/V 同时有效屏蔽。可以除以固定参考距离的平方，避免数值权重依赖单位。$\epsilon$ 根据 SDF 近似误差、CAD 数值误差和训练标签精度设置，不能对近似 SDF 强制完全零容差。

这只能保证标量包含关系趋于合理，不能由此推断每条边具有正确端点或每个 face wire 闭合。初期低权重，避免把预测偏差从一通道传播到另一通道。

### 9.6 Presence、范围及 KL

$$
L_\mathrm{pres}=\operatorname{BCEWithLogits}(\ell_E,p_E)
+\operatorname{BCEWithLogits}(\ell_V,p_V).
$$

记录类别比例，必要时分层抽取含／不含顶点样本。可加 $\operatorname{ReLU}(\widehat p_V-\widehat p_E)^2$；推理仍需处理不确定状态。不能删除全部空 V 样本来规避该问题。

输出负值惩罚使用 $L_\mathrm{neg}=\operatorname{mean}\operatorname{ReLU}(-\widehat u_c)^2$。空通道的可选规范项另命名为 $L_\mathrm{empty}$，与物理距离损失分开记录。

field KL 沿用：

$$
L_\mathrm{KL,f}=\operatorname{mean}_{B,M,C_f}
\frac12(\mu_f^2+\exp(\log v_f)-1-\log v_f).
$$

正式 VAE 的 KL beta 可从 `1e-6` warmup 起步，比较 `1e-7/1e-6/1e-5`；这些只是沿用现有量级的实验起点。KL=0 适合先检验确定性重建上限，不应直接作为已经验证的 Flow-ready VAE。

### 9.7 分阶段总损失

第一阶段 field 训练：

$$
L=L_E+L_V+\lambda_m L_\mathrm{metric}
+\lambda_0L_0+\lambda_pL_\mathrm{pres}
+\lambda_nL_\mathrm{neg}+\beta L_\mathrm{KL,f}.
$$

可从 `lambda_metric=.5, lambda_zero=.1, lambda_presence=.1, lambda_negative=.01` 起步。各项按上述定义归一化后才有比较意义。先观察梯度与近场误差，再加入小权重的 `L_grad/L_inc`；不要一次开启所有正则导致无法定位问题。

几何分支单独预训练时使用 SDF 编码损失、近表面物理损失及可选 geometry KL；冻结后不再重复为不可更新的几何参数计算训练损失。联合微调时再恢复这些项。

## 10. 训练阶段、首轮配置和验收门槛

### 10.1 建议的阶段顺序

| 阶段 | 内容 | 进入下一阶段前检查 |
|---|---|---|
| A：数据适配 | 原始三场读取、编码／逆编码、空集、条件采样 | 单位与通道正确；训练条件无 GT 拓扑泄漏 |
| B：几何条件 | SDF geometry encoder/decoder 预训练 | signed SDF、表面重建与输入条件一致 |
| C：单物体 AE | 关闭 field KL，固定条件和 anchors，动态 query | 能定位全部目标顶点与各段边；独立 query 误差可接受 |
| D：小集合 AE/VAE | 多物体、空 V、复杂局部结构；逐渐启用 KL | sampled posterior 与 mean 都能重建；没有大规模 latent collapse |
| E：正式 VAE | ABC 按源模型／相近变体分组拆分 | 在未见物体上保持近场、presence 和拓扑指标 |
| F：缓存与 Flow | 冻结最终 VAE，重新缓存、统计并训练 | 生成场通过同一套独立评测与后处理 |

现有 ABC10 目录只有 4 个成功处理的物体，可用于接口与过拟合测试，不足以验证泛化。正式训练必须扩大数据集，并按 CAD 原始模型家族拆分，避免同一设计的相近零件进入不同 split。

### 10.2 结构起点

以下是新任务的中等规模 pilot，**不是已经实测可用的训练配置**：

| 项目 | 起点 | 扩容条件 |
|---|---:|---|
| `Ng` 表面条件 | 8192 | SDF 小特征条件不足 |
| `Ns` SDF 条件 | 2048 | 内外／整体几何表达不足 |
| `Nf` field 输入 | 8192 | encoder 漏掉短边／近邻顶点 |
| `K` decoder 查询 | 8192 + 小批零集点 | 输入已覆盖，但查询监督不足 |
| `M` latent tokens | 1024 | 单物体覆盖充分，仍无法保存结构 |
| `Cg / Cf` | 64 / 64 | 固定 M 下信息瓶颈明确 |
| width / heads | 384 / 6 | 容量消融表明需要增加 |
| encoder layers | 4 | 与宽度／M 分开消融 |
| latent decoder layers | 6 | 与 query decoder 深度分开 |
| Fourier frequencies | 8 | 比较 6/8，警惕振荡伪零 |
| anchors | random 基线，FPS 对照 | 以几何覆盖与稳定性选择 |
| optimizer | AdamW，LR `1e-4` | 根据实测收敛调整 |
| microbatch | 1–2 起测 | 以真实峰值显存和吞吐增加 |

改用 `M=1024` 需要两支 VAE、缓存和 Flow 同时采用该 token 数；它不改变 Flow 算法，但不是复用现有缓存。若必须保持某个既定 Flow 配置，则直接使用其 M，并独立调整 `Nf/K`。

不能从贴图实验显存直接推断本方案显存，尤其加入 gradient loss 后。主要计算增长包括 encoder 的 $M N$、latent 的 $M^2$、query 的 $K M$。

### 10.3 模型精度门槛与后处理阈值关联

以实际使用的隔离带阈值 $\tau_E,\tau_V$ 为准，建议将相关近场距离误差 p95 小于阈值的 `1/4` 作为初步目标，再检查真实分割是否成功。它是经验验收目标，不是拓扑正确性的数学保证，也不是可信的全域误差上界。

最小边间距、顶点间距也要参与验收：即使总体 p95 低，少数紧邻结构仍可能合并。位置容差不应低于预处理和条件 SDF 的实际精度预算。

## 11. Flow 可以保留到什么程度

### 11.1 保留的部分

目标仍是固定形状的连续 latent：

$$
Z_f\in\mathbb R^{B\times M\times64},\qquad
Z_g\in\mathbb R^{B\times M\times64}.
$$

标准化后的 $Z_f$ 进入现有线性路径、velocity regression、DiT 与 Euler 采样。RoPE 继续读取同一组几何 `anchor_xyz`。生成两个物理场并不要求 Flow target channels 变成 2；64 是 latent 宽度，2 是 decoder 输出数。

只保留几何条件时，使用现有关闭图像的路径，不需要 DINO／RGB 渲染／光照一致性。仓库中的 `unconditional` 命名实际仍含几何条件。若以后要对 SDF 做 CFG，需额外训练几何条件 dropout，不能直接套用当前只对图像条件做 dropout 的 CFG 语义。

### 11.2 仍然必须改的工程接口

- 目标 posterior 从颜色改成 field，生成入口不再加载 RGB 数据。
- cache、statistics、bundle 明确 `target_kind=edge_vertex_udf`，字段编码、坐标和 presence 版本一致。
- 老 Flow 权重与旧颜色 latent 缓存不能视作等价的新 UDF 模型；即使张量形状相同也必须重新训练／验证。
- 冻结 VAE 后用 $Z_g=\mu_g$ 编码 field posterior，缓存 `mu_f/logvar_f`。
- 若 Flow 用 posterior sample 作为目标，统计方差包括 $\operatorname{Var}(\mu_f)+\mathbb E[\sigma_f^2]$；若使用 mean，则只统计 mean 分布。可复用现有统计实现，但必须重算。
- VAE 的训练采样、确定性缓存采样和实际推理采样共享几何条件规范，确保 token 对应关系一致。

可在过渡 adapter 中把 `z_f_mu/logvar` 映射到旧 `z_c_mu/logvar` 参数名以复用 Flow 代码；磁盘元信息必须标明新语义，不能只换路径沿用旧 schema 身份。

### 11.3 SDF 不唯一确定 B-rep 划分

同一个实体边界可对应不同的共面 face 分割或参数分段。即使已排除 seam，SDF 仍未必唯一决定需要生成的全部 E/V。

因此本任务是学习 $p(d_E,d_V\mid s)$ 的合理目标规范，而不是假设可以从 SDF 唯一推导任意 CAD 的原始拓扑。VAE 应能保存 GT 的不同合法分割，Flow 应学习这种条件分布；预处理需保持一致的规范化规则。

若 VAE 总是忽略 $Z_f$、仅从几何分支回归一个平均划分，重建 loss 可能下降，生成多样性和拓扑保真却会受损。可通过固定 $Z_g$、交换或扰动 $Z_f$，检查其是否确实承载目标结构信息。

## 12. 推理与现有 B-rep 重建接口

### 12.1 推理顺序

```text
输入 SDF 与 normalization
  → 统一几何采样（仅由 SDF 得到表面、法向、空间 SDF 点）
  → 固定 anchor plan
  → geometry encoder mean：Zg
  → latent Flow：生成 Zf
  → prepare_decode(Zg,Zf,A)
  → presence head
  → 任意坐标查询：SDF_condition(x), decode_metric_E/V(x)
  → SDF 表面网格、隔离带、面核心与标签传播
  → 顶点／边实例、几何定位、共享曲线与约束曲面、B-rep 装配
```

VAE 自编码评测中，用 GT field encoder 的 posterior mean/sample 代替 Flow 生成的 $Z_f$；其余查询和后处理保持一致。

### 12.2 对接已存在的 FieldProvider

本仓库已有 `band_brep_reconstruction/bandbrep/field_provider.py`，接口返回 `[sdf,dE,dV]`，并提供 presence、gradient、误差等信息。适配器应做到：

1. `sdf(x)` 返回输入 SDF 的值，避免由 VAE 辅助几何头再次重建后引入额外几何漂移。
2. E/V 编码输出先逆变换为归一化物理距离，再按通道顺序组装。
3. presence 使用模型预测；GT presence 只允许在明确的 oracle 消融中使用。
4. 低置信度 presence 可返回 `unknown` 并进入一致性检查，阈值用验证集校准。
5. 误差报告从独立验证统计得到，`strict_error_bound=False`；不能把样本 p95 声称为全域严格界。
6. 缓存标识包含模型权重、latent、条件 SDF、编码版本与 normalization。

当前 `CallableFieldProvider(normalized=True)` 只是乘 `clip_values` 的线性反归一化，**不能处理 log 编码**。应在 callable 内完成 log 逆变换后以 metric 模式接入，或新增专门 adapter。

场值接口要求非负，因此要区分对外 value 与用于局部拟合的未裁剪 residual。若发生负值裁剪，该点不能再无条件返回“裁剪值有效、梯度也有效”；需设置有效性／裁剪标记，并对细化路径作相应适配。

### 12.3 梯度定位必须使用正确的距离

局部投影／最小化需要物理距离及完整导数。不能将 log 编码直接代入 $x-d\nabla d$，也不能把正数阈值面当最终边或顶点。

对梯度不可靠、最近目标切换或存在裁剪的平台，采用受限步长、多个候选和残差验证；必要时对完整查询函数有限差分。VAE 表达改变后，后处理阈值仍以物理单位设置。

## 13. 需要改动的文件与实施顺序

建议以独立的 field 模块／任务类型实现，复用已有底层 attention 和工具，保留原贴图实验入口。下表都是待实施工作。

| 优先级 | 现有位置／拟新增位置 | 具体工作 |
|---|---|---|
| P0 | 新 `natex/data/edge_vertex_fields.py` | 读取 ABC 三场、分组 split、采样、空集和合法 mask |
| P0 | 新 `natex/field_encoding.py` | log／signed-log 编解码、导数和版本契约 |
| P0 | `natex/contracts.py` 或新 field contracts | 分离 geometry 输入和 field 输入；保持 token／anchor 校验 |
| P0 | `models/autoencoders/` 新 field model | geometry SDF KV、field KV、联合 posterior、双输出头、presence |
| P0 | 新 `CoupledFieldQueryDecoder` | 坐标/SDF/geometry feature 融合，距离输出及查询接口 |
| P0 | 新 field losses | 分通道／距离层／实体损失，空通道处理，KL |
| P0 | 新 `train_field_vae.py`、配置 schema | geometry pretrain、AE/VAE 模式、确定性验证 |
| P1 | 新 field reconstruction pipeline | 任意坐标 metric 查询、分块、梯度、可视化和 FieldProvider adapter |
| P1 | 新 field evaluation | 近场、零集、presence、拓扑恢复与压缩指标 |
| P1 | `config.py / checkpoints.py` | 新架构键、构造器、bundle class/task、编码元信息 |
| P2 | `tools/natex/cache_latents.py` 对应 field 入口 | 非 RGB 数据准备、field posterior、重新计算统计 |
| P2 | Flow 数据 adapter／配置 | 继续使用原 Flow 核心，加载新 latent 和几何条件 |

不要只新增 YAML 字段而不更新 `config.py`：现有 schema 会拒绝未知键。`checkpoints.py` 当前也检查 bundle class 为 `GeometryAwareColorVAE`，必须注册新类／任务。

checkpoint 与 cache 至少保存：

```text
target_kind: edge_vertex_udf
raw_channel_names: [sdf, dE, dV]
output_channel_names: [dE, dV]
encoding: log_distance_v1
encoding_parameters: D, sigma_E, sigma_V, sigma_S
empty_set_policy, presence_head_version
normalization_policy, query_domain
target_filter_version, geometry_condition_sampling_version
Ng, Ns, Nf, M, Cg, Cf, anchor_method
geometry_posterior_mode, field_posterior_mode
dataset/split identity, model fingerprint, cache/statistics identity
```

旧颜色权重仅作为可选初始化实验：按名称和 shape 筛选通用 Transformer 权重；输入投影、SDF/field 头、presence 和语义变化模块重新初始化。shape 相同也不代表 RGB 与 UDF 输出头可以直接迁移。新任务必须有从头训练基线。

## 14. 必须做的测试和实验

### 14.1 契约与数值测试

- log 编码的零值、单调性、逆变换、阈值换算、链式梯度。
- `[sdf,dE,dV]` 不换序；原始非空距离不被静默截断。
- 空 V、空 E/V、闭环无顶点、高度相邻顶点，各自 mask 和 presence 正确。
- 改变 GT E/V 或标签采样而固定几何条件时，`Zg/anchors` 不变化。
- field KV 点数／位置改变不破坏 anchor 对齐；随机打乱 token 对齐时明确报错。
- decoder 输入移除 posterior 观察点后仍可查询；不存在未声明的 GT 最近点／ID 通路。
- query 分块与整批预测一致；不同物体之间 KV cache 不串用。
- 带显式 SDF 路径的完整坐标导数与有限差分一致；检查高阶反传。
- 新 bundle 与旧颜色 cache 拒绝混用；编码或权重变化使 cache 失效。
- 若做独立 query 划分，重复坐标不跨训练／验证集合。

### 14.2 不能只报全局 MSE

| 层级 | 应报告的指标 |
|---|---|
| 距离场 | 各通道、各距离层 metric MAE/p95；uniform 全域误差；负值率 |
| 结构存在性 | E/V presence 混淆矩阵、校准、空 V 误报 |
| 顶点 | 一对一匹配 precision/recall、定位误差、合并／分裂数量 |
| 边 | 双向几何距离、短边召回、闭环召回、断裂／桥接数量 |
| 隔离带 | 阈值前后带覆盖、面核心丢失／混合／过分割 |
| 拓扑 | face/edge/vertex 数及匹配，E–V/E–F incidence、wire 闭合 |
| B-rep | 有效 shell/solid 比例、曲线／曲面误差、装配失败原因 |
| latent | KL、各维统计、mean/sample 重建差距、扰动鲁棒性 |
| 资源 | encoder、query 和 Flow 分别的显存／耗时 |

边和顶点几何指标允许 GT 用于匹配；重建过程中仍禁止读取 GT 实例关系。

固定模型同时评估：GT 三场后处理、VAE mean 后处理、VAE sampled posterior 后处理、Flow 生成后处理。这样可以区分后处理本身失败、VAE 压缩损失和 Flow 生成误差。

### 14.3 最小消融矩阵

| 编号 | 改变因素 | 回答的问题 |
|---|---|---|
| A0 | 原始距离 + 分层损失 | 无非线性编码时的基线 |
| A1 | log 编码，其余同 A0 | 全局压缩是否改善近场与 latent 训练 |
| A2 | `.02` 硬截断 | 大片常数是否导致顶点召回下降 |
| A3 | 去掉 near-V／逐顶点均衡 | V 的问题是否主要来自采样 |
| A4 | field encoder 只看表面点 | 体空间 posterior 输入是否必要 |
| A5 | 去掉 geometry 的 SDF KV | 显式 SDF 体信息的贡献 |
| A6 | 仅 geometry feature query，与坐标/SDF 直连比较 | 是否存在查询特征瓶颈 |
| A7 | 固定采样，单独增加 M；再单独增加 Nf/K | 区分 latent、输入和监督瓶颈 |
| A8 | KL=0 与不同 KL beta | 压缩正则对近零结构和 posterior 稳定性的影响 |
| A9 | 增加梯度／包含关系／热度辅助头，各自单独做 | 判断辅助损失收益，避免混合归因 |

至少覆盖箱体、圆柱／多闭环无顶点、短边、近邻顶点、相切面边界、同一 face pair 的多条边、高 valence 顶点。增加一个相同几何、不同合法面划分的对照，用于检查 geometry 条件泄漏和 field latent 是否被利用。

## 15. 尚需实验决定的事项

本文确定接口与主线，不提前断言以下选择已经最优：

- `sigma=.02` 还是 `.05`，以及 E/V 是否采用不同 sigma。
- M=1024 是否足够，还是需要 2048/6144；不能仅按 CAD 顶点数量选 M。
- geometry 预训练后冻结是否优于后续联合微调。
- 单共享 memory + 双头是否足够，还是需要 E/V 各自的轻量分支。
- gradient loss 是否在相同计算预算下改善最终几何，而不只是改善梯度数值。
- SDF 条件分辨率和误差对边界拟合的限制；模型输出不能凭空恢复条件中已经丢失的几何细节。

优先按“数据语义和条件来源 → 单物体重建上限 → 多物体 VAE → latent Flow”的顺序推进。只有 VAE 在独立查询及后处理指标上成立，才有依据把后续误差归因于 Flow。

## 16. 核查依据与参考范围

本方案的模块映射来自2026-10-14的本地代码检查。以下保留核查位置；两处目前已不存在的外部文件以历史路径记录。主要位置：

- [双分支 VAE](../../Hunyuan3D/hy3dshape/hy3dshape/natex/models/autoencoders/model.py)、[query decoder](../../Hunyuan3D/hy3dshape/hy3dshape/natex/models/autoencoders/query_decoders.py)、[契约](../../Hunyuan3D/hy3dshape/hy3dshape/natex/contracts.py)。
- [现有损失](../../Hunyuan3D/hy3dshape/hy3dshape/natex/training/losses.py)、表面颜色 Dataset（历史路径 `H/hy3dshape/natex/data/surface_color.py`）、[anchors](../../Hunyuan3D/hy3dshape/hy3dshape/natex/models/anchors.py)。
- [Flow](../../Hunyuan3D/hy3dshape/hy3dshape/natex/models/diffusion/flow_matching.py)、[latent cache](../../Hunyuan3D/hy3dshape/hy3dshape/natex/data/latent_cache.py)、几何条件 Flow 配置（历史路径 `H/configs/natex_flow_unconditional.yaml`）。
- [ABC 数据读取](../edge_vertex_preprocessing/evudf/io.py)、[采样](../edge_vertex_preprocessing/evudf/sampling.py)、[现有重建 FieldProvider](../band_brep_reconstruction/bandbrep/field_provider.py)。

概念参考：NaTex 提供几何条件下连续颜色场的双分支 latent 生成思路；3DShape2VecSet 提供以向量集合和 attention 表达神经场的相关基础。[NaTex 原文](https://arxiv.org/html/2511.16317v1)、[3DShape2VecSet 原文](https://arxiv.org/abs/2301.11445)。

本文的 log 距离编码、采样比例、空集设计、损失配方和训练阶段属于针对当前 E/V 任务提出的方案，不宣称是上述论文已经采用或验证的设置。本次没有修改训练／推理代码，没有启动训练，也没有新增实测性能结论。
