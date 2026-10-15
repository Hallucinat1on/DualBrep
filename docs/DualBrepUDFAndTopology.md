# DualBrep：UDF 表示、面分割与 B-rep 拓扑恢复

更新日期：2026-10-15。本说明合并原 UDF 表示说明与拓扑/seam 说明，介绍原 DualBrep 的 GVD-UDF、训练与推理、面分割、学习式参数化及 CAD 后处理。它与本仓库新设计的最近真实边/顶点三场方案区分阅读。

- [A. UDF 表示与处理流程](#udf-flow)
- [B. 拓扑恢复、seam 与特殊边](#topology-seam)

两部分分别保留各自章节号；代码位置与论文依据沿用原审查记录，不表示本次重新运行了论文实验。

<a id="udf-flow"></a>
## A. UDF 表示与处理流程

本文完全依据当前仓库代码说明 UDF（Unsigned Distance Field，无符号距离场）在 DualBrep 中的几何含义、数据表示、生成方式、网络表示、推理后处理以及如何用于 B-rep 面分割。

### 1. 先给出结论

DualBrep 不是只用 SDF 表示一个 CAD 模型，而是联合使用两个连续隐式场：

$$
F(\mathbf{x})=
\left[
s(\mathbf{x}),\ u(\mathbf{x})
\right],\qquad \mathbf{x}\in[-1,1]^3
$$

- $s(\mathbf{x})$ 是到实体表面的有符号距离 SDF。它主要回答“物体表面在哪里”，零等值面 $s=0$ 用于重建三角网格。
- $u(\mathbf{x})$ 是到 **Voronoi 分界网格** 的无符号欧氏距离 UDF。它主要回答“不同 B-rep 面的分界在哪里”，在重建表面上可近似理解为到 B-rep 边界的距离。

代码中最严格的 UDF 定义是：

$$
u(\mathbf{x})=d(\mathbf{x},\mathcal V)
=\min_{\mathbf{y}\in\mathcal V}\|\mathbf{x}-\mathbf{y}\|_2\ge 0
$$

其中 $\mathcal V$ 是 `Voronoi/calculate_voronoi` 生成的 `voronoi.ply` 三角网格。这个定义直接体现在 `prepare_implicit.py:141-147`，而不是直接计算查询点到原始 OCC B-rep edge 集合的距离。

之所以 README、`dataset.py` 和 `clustering.py` 会把它简称为“到 B-rep edge 的距离”，是因为在实体表面 $\mathcal S$ 上，Voronoi 分界场与表面的交线 $\mathcal V\cap\mathcal S$ 对应或逼近相邻 B-rep 面的边界。因此：

- 在三维空间中：UDF 是到 Voronoi 分界面的距离；
- 限制到重建表面上：低 UDF 区域形成面与面之间的边界带，可作为“到 B-rep 边的距离”使用。

这一区分很重要，否则会误以为 `prepare_implicit.py` 调用了点到 OCC 曲线的距离计算；实际标签是通过 libigl 对 `voronoi.ply` 求距离得到的。

### 2. UDF 为什么能补充 SDF

单独的 SDF 只能可靠恢复几何外壳：

$$
\mathcal S=\{\mathbf{x}\mid s(\mathbf{x})=0\}
$$

但同一张连续表面可以有不同的 B-rep 面划分。例如，一个平面可能在拓扑上是一个面，也可能被边切分为多个面。只看 SDF，二者的几何外形可以完全相同，无法恢复这种面划分信息。

UDF 在边界附近取小值、在一个 B-rep 面内部取较大值。把低值区域从 SDF 重建出的三角网格上暂时去掉后，各个面内部通常断开成不同连通分量。因此本项目的核心分工是：

```text
SDF = 几何外壳：在哪里有表面
UDF = 拓扑分界：表面应在哪里被切成多个 B-rep face
```

### 3. UDF 参考几何 `voronoi.ply` 是怎样得到的

#### 3.1 坐标归一化

`Voronoi/calculate_voronoi/calculate_voronoi.cpp:510-547` 先读取 STEP 包围盒，将中心平移到原点，并按最长轴做等比例缩放，使模型落入大约 $[-0.9,0.9]^3$。`prepare_implicit.py` 使用同样的 `BOUNDING = 0.9` 和 `normalize_shape`，保证 STEP 三角网格与 `voronoi.ply` 位于同一坐标系（`prepare_implicit.py:98-103`）。

设原始包围盒中心为 $\mathbf c$，最长边尺寸为 $L_{\max}$，则可以把归一化理解为：

$$
\mathbf{x}_{norm}=(\mathbf{x}_{orig}-\mathbf c)\frac{1.8}{L_{\max}}
$$

因此配置中的 `0.015`、`0.01`、`0.005` 等距离阈值都是归一化空间中的长度，而不是原始 STEP 文件的毫米或米。

#### 3.2 按 B-rep face 标记采样点

`sample_points_surface` 为采样点维护 `primitive_index`，用它记录点来自哪个 B-rep face（`calculate_voronoi.cpp:102-119`）。在共享 edge 两侧，代码沿曲线采样，然后利用面法向和边切向的叉积，把采样点向两个相邻面的内部轻微平移：

$$
\mathbf d=\mathbf n\times\mathbf t,\qquad
\mathbf p'=\mathbf p+\epsilon\mathbf d
$$

这里 `voronoi_distance = 0.0001`，相关实现位于 `calculate_voronoi.cpp:227-250` 和 `calculate_voronoi.cpp:312-369`。这样同一条边两侧会产生分别带有两个 face id 的采样点，使后续 Voronoi 分界能够穿过真实面边界。

#### 3.3 构造不同 face 样本之间的 Voronoi 分界面

`compute_voronoi` 对全部采样点建立三维 Delaunay/Voronoi 结构，并把 Voronoi cell 的 facet 三角化（`geom_voronoi.cpp:11-90`）。随后只保留相邻采样点所属 `primitive_index` 不同的公共 facet（`geom_voronoi.cpp:147-180`），最终输出 `voronoi.ply`（`geom_voronoi.cpp:213-220`）。

因此，这里编码的不是语义类别，而是“离不同 B-rep face 样本等距”的几何分界。它是一张三维三角网格，并可能从 CAD 边界向实体内部或外部延伸。

### 4. UDF 监督数据如何生成

入口是 `prepare_implicit.py::build_implicit_npz`。

#### 4.1 输入给编码器的三类点云

每个样本包含三类带方向的点：

| 字段 | 典型形状 | 内容 | 代码位置 |
|---|---:|---|---|
| `surface_points` | `(100000, 6)` | 表面 xyz + normal | `prepare_implicit.py:109-111` |
| `edge_points` | `(100000, 6)` | B-rep 边 xyz + tangent | `prepare_implicit.py:113-117` |
| `voronoi_points` | `(Nv, 6)` | Voronoi 网格 xyz + normal | `prepare_implicit.py:130-135` |

注意：`edge_points` 是编码器的几何输入之一；它本身不是 UDF 标签。UDF 标签仍然是查询点到 `voronoi.ply` 的距离。

Voronoi 点还会按照“到实体表面的无符号距离小于 0.2”进行筛选，使编码器关注靠近物体、与表面分割更相关的 Voronoi 区域。

#### 4.2 两组查询点

代码针对不同区域构造两组查询：

1. `query_edge_points`

   从稠密 B-rep edge 采样点出发，分别添加标准差为 `0.001`、`0.005`、`0.007`、`0.01` 的高斯噪声（`prepare_implicit.py:119-122`）。它强化边界附近的监督。

2. `query_surface_points`

   由三部分拼接（`prepare_implicit.py:124-139`）：

   - $[-1,1]^3$ 内 200000 个均匀随机点，用于覆盖全局空间；
   - 表面采样点加 `0.001`、`0.005` 高斯噪声，用于强化 SDF 零面附近；
   - Voronoi 采样点加 `0.001`、`0.005` 高斯噪声，用于强化 UDF 零集附近。

这种混合采样避免了均匀体采样几乎碰不到薄的表面/分界区域，同时又保留全局场监督。

#### 4.3 用 libigl 计算标量标签

`prepare_implicit.py:141-147` 分别计算：

```python
surface_udf = igl.signed_distance(
    q_surface_points, voro_v, voro_f,
    sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED)

edge_udf = igl.signed_distance(
    q_edge_points, voro_v, voro_f,
    sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED)
```

虽然 API 名称叫 `signed_distance`，传入 `SIGNED_DISTANCE_TYPE_UNSIGNED` 后结果没有正负号，理论范围为 $[0,+\infty)$。两组 UDF 最终分别保存为：

- `query_surface_udf: (Qs,)`
- `query_edge_udf: (Qe,)`

`.npz` 中统一转为 `float16` 以节省空间（`prepare_implicit.py:149-160`）；`VAEDataset.__getitem__` 读取后再转为 `float32`，并各随机抽取 `n_supervision` 个查询（`dataset.py:117-150`）。

### 5. 截断与归一化：网络真正看到的 UDF

原始距离不是直接送入网络。`model.py::augment_submitted` 使用 `clip_value` 截断并归一化：

$$
\tilde u(\mathbf{x})=
\frac{\min(\max(u(\mathbf{x}),0),c)}{c}
\in[0,1]
$$

默认 $c=0.015$，对应 `config.yaml:14` 和 `model.py:124-128`。所以：

- `0`：点就在预测的 Voronoi 分界上；
- `0.5`：原距离约为 `0.0075`；
- `1`：原距离大于等于 `0.015`，所有更远距离都饱和为同一个值。

截断的作用是把学习能力集中在分界附近。代价是模型无法区分 `0.02` 和 `0.2` 这样的远场距离，但下游分割并不需要这种区别。

SDF 同时被截断到 $[-c,c]$ 并缩放到 $[-1,1]$。因此两个输出通道的数值域不同：

| 输出通道 | 含义 | 网络尺度 | 恢复距离 |
|---|---|---:|---|
| `[..., 0]` | SDF | `[-1, 1]` | `clip(x,-1,1) * c` |
| `[..., 1]` | UDF | `[0, 1]` | `clip(x,0,1) * c` |

旋转增强会同时旋转输入点云和查询坐标，但距离标量不变，这符合欧氏距离的旋转不变性（`model.py:150-159`）。需要注意，`is_aug == 3` 中也可能做非均匀缩放，但代码仍保持距离标签不变；默认重建配置 `is_aug: 0` 不走这条训练增强路径。

### 6. 网络内部怎样表示 UDF

#### 6.1 输入几何变成 latent set

`DualVAE.encode` 先对三类点的 xyz 做 Fourier 特征编码，再与 normal/tangent 拼接并投影到宽度 768（`model.py:337-353`）。随后：

- 对表面点和边点分别用 FPS 选择 token；
- 用 cross-attention 分别聚合 surface、edge、Voronoi 特征；
- 把三个分支相加并经过 self-attention；
- 用 `pre_kl` 将每个 token 压缩为 32 维 latent。

对应代码为 `model.py:355-383`。这里没有一个单独的“UDF latent”；SDF 和 UDF 共享同一个几何/拓扑 latent set。

#### 6.2 查询解码器输出两个标量

`DualVAE.query(points, latent)` 对任意三维坐标 $\mathbf{x}$ 做 Fourier 编码，以 latent set 为 key/value 做 cross-attention，最后由：

```python
self.output_proj1 = nn.Linear(width, 2)
```

输出两个通道（`model.py:312-314, 388-390`）：

```text
[B, Q, 3] 查询坐标
        │ Fourier + cross-attention(latent)
        ▼
[B, Q, 2]
        ├── channel 0: 归一化 SDF
        └── channel 1: 归一化 UDF
```

最后一层代码中没有 `sigmoid` 或 `tanh`，所以网络原始输出在数学上不受限；范围限制是在推理后通过 `np.clip` 完成的。

#### 6.3 关于训练损失的代码边界

当前仓库的 `model.py` 是推理版模型，保留了编码、解码和 query 逻辑，但没有提供利用 `query_surface_udf` / `query_edge_udf` 计算重建损失的 VAE 训练 `forward`/`loss`。仓库中能确认的是标签准备、归一化和已训练 checkpoint 的推理接口；不能仅依据当前代码断言训练时 UDF 使用的是 L1、L2，或两类查询的具体损失权重。

另外，在当前 `ae_reconstruct.py` 的推理路径里，`.npz` 中的查询标签只会触发 `augment` 并完成标准化/姿态处理，真正编码使用的是 surface/edge/Voronoi 三类点云；推理并不会用 ground-truth UDF 去修正预测结果。

### 7. 从 latent 得到稠密 UDF 体

#### 7.1 普通稠密查询

`model.py::_inference` 在每个轴上生成 `linspace(-1,1,res)`，形成 $R^3$ 个查询点，并以每批 100000 点调用 `model.query`（`model.py:172-197`）。返回：

$$
\text{grid shape}=[B,R,R,R,2]
$$

其中 `grid[..., 1]` 就是归一化 UDF 体。

`ae_reconstruct.py:97-100` 将它转回截断距离：

```python
udf = np.clip(grid[..., 1], 0, 1) * clip_value
```

#### 7.2 加速多分辨率查询

默认 `runtime.acc: true` 使用 `_inference_acc`：

1. 在 $128^3$ 网格上完整计算 SDF 和 UDF；
2. 根据 `abs(SDF) < 0.010/0.015` 找出近表面体素，并做一次 3×3×3 膨胀；
3. 对其他区域三线性上采样，只在近 SDF 表面的掩码中重新查询 256 或 512 分辨率；
4. SDF、UDF 使用完全相同的掩码更新。

关键实现位于 `model.py:225-274`。这说明高分辨率 UDF 的精确求值区域是由 **SDF 近表面带** 决定的，而不是由 UDF 自己的低值区域决定。这样设计是合理的，因为 UDF 的主要下游用途是在 SDF 表面上分割；远离物体表面的 Voronoi 细节并不重要。

### 8. UDF 得到后有两条处理路径

#### 8.1 路径 A：UDF 等值面可视化/边线管道

UDF 没有正负号，其零集通常非常薄，数值预测也很难恰好等于零，因此不能像 SDF 一样直接稳定地对 `iso=0` 做 marching cubes。项目使用一个正阈值：

$$
\mathcal M_u(\tau)=\{\mathbf{x}\mid u(\mathbf{x})=\tau\}
$$

默认 `udf_threshold = 0.01`。`mesh_utils.py::udf2mesh` 对该等值面运行 marching cubes，再把体素坐标映射回 $[-1,1]^3$（`mesh_utils.py:21-27`），输出 `recon_udf.ply`。

从严格几何意义看，`recon_udf.ply` 是距离 Voronoi 场为 $\tau$ 的偏置壳层；代码和 README 把它称为 edge/wireframe iso-surface，是因为它在目标表面附近刻画 B-rep 分界带。它不是 OCC 中一维解析曲线的直接重建结果。

#### 8.2 路径 B：在 SDF 表面上采样 UDF，用于面分割

这是 UDF 最关键的用途。`ae_reconstruct.py:109-121` 先对 SDF 的零等值面做 marching cubes，得到顶点 `v` 和三角形 `f`；然后计算每个三角形中心：

$$
\mathbf c_i=\frac{\mathbf v_{i0}+\mathbf v_{i1}+\mathbf v_{i2}}{3}
$$

再直接调用隐式解码器查询 $u(\mathbf c_i)$，得到：

```text
udf_g.npy: (F,)
```

其中 $F$ 是 `recon_sdf.ply` 的三角形数量。`udf_g.npy` 保存的是已经乘回 `clip_value` 的归一化空间距离。直接 query 比从低分辨率体素网格采样更准确。

若没有 `udf_g.npy` 而只有历史格式的 `recon_udf.npy` 稠密体，`clustering.py::_perface_udf_from_volume` 会把面中心从 $[-1,1]$ 映射到体素坐标，并做三线性插值（`clustering.py:1129-1143`）。

### 9. UDF 如何驱动 B-rep face 分割

`clustering.py::process_item` 的默认模式是 `hierarchical`，参数为：

```python
threshold1 = 0.005
threshold2 = 0.01
filter_size = 10
mode = "hierarchical"
```

这些值位于 `clustering.py:38-48`，单位均为归一化坐标长度。

#### 9.1 先只保留最大网格连通分量

`process_item` 读取 `recon_sdf.ply`，根据三角面邻接关系求连通分量，只保留三角形数量最大的分量，并用同一个 mask 过滤 `udf_g.npy`（`clustering.py:1177-1187`）。这会去掉 SDF marching cubes 产生的游离小壳层。

#### 9.2 第一遍：低 UDF 作为边界带

令每个三角形的 UDF 为 $u_i$，第一遍只激活：

$$
A_1=\{i\mid |u_i|\ge 0.005\}
$$

低于阈值的三角形接近 Voronoi/B-rep 分界，被标为 `-1`，相当于从表面图上切掉边界带。然后只连接“两个端点都激活”的相邻三角形，并求无向图连通分量（`clustering.py:464-493`）。

直观上：

```text
一个 B-rep face 内部：UDF 大，三角形仍连通
B-rep edge 附近：    UDF 小，被挖掉形成断带
不同 B-rep face：    因断带而落入不同连通分量
```

随后删除三角形数小于 `filter_size`，或平均 UDF 不超过默认 `0.003` 的分量（`clustering.py:495-504`），减少噪声小块。

#### 9.3 第二遍：更高阈值重新切分

对于最大 UDF 达到 `threshold2=0.01` 的第一遍 cluster，再仅保留：

$$
A_2=\{i\mid u_i\ge 0.01\}
$$

并在原 cluster 内重新求连通分量（`clustering.py:508-540`）。若得到至少两个足够大的子分量，就用它们替换原 cluster；否则保留第一遍结果。这一遍能够切开第一遍低阈值下仍通过较宽通道粘连的区域。

最终标签写入 `cluster.ply` 的 per-face `label` 属性。这里得到的是与 CAD 曲面片对应的几何 face cluster，不是“轮毂、螺栓、孔”等语义零件标签。

### 10. AE 重建与条件生成中的处理差异

#### 10.1 AE 重建

`ae_reconstruct.py` 的链路是：

```text
surface/edge/Voronoi 点云
  -> DualVAE.encode
  -> 32 维 latent tokens
  -> DualVAE.decode
  -> query([-1,1]^3)
  -> [SDF, UDF]
  -> recon_sdf.ply + recon_udf.ply + udf_g.npy
  -> clustering.py
  -> cluster.ply
```

点云模式 `DualVAE_PC` 可以只根据 surface point cloud 编码；即使输入没有 edge/Voronoi 点，已训练模型仍会同时解码 SDF 和 UDF。这里的 UDF 是模型从表面几何中推断出的潜在拓扑分界，而不是从输入点云直接测量出来的。

#### 10.2 点云/图像条件生成

`diffusion_model.py::FusedModelFlow.inference` 从随机噪声出发，通过 rectified-flow ODE 得到 VAE latent，再由冻结的 DualVAE decoder 解码（`diffusion_model.py:128-145`）。`vae_generate.py` 后续采用和 AE 基本相同的稠密查询、反归一化、marching cubes、面中心 UDF 查询与分割流程。

因此生成模式中也不是 diffusion 模型直接输出体素 UDF；flow 生成的是 latent set，连续 UDF 仍由 `DualVAE.query(x, latent)` 按需计算。

### 11. 各种 UDF 文件/张量不要混淆

| 名称 | 形状 | 所处阶段 | 数值尺度 | 含义 |
|---|---:|---|---|---|
| `query_surface_udf` | `(Qs,)` | 训练数据 `.npz` | 原始归一化空间距离，保存为 float16 | surface/global/Voronoi 查询点到 `voronoi.ply` 的距离 |
| `query_edge_udf` | `(Qe,)` | 训练数据 `.npz` | 同上 | edge 邻域查询点到 `voronoi.ply` 的距离 |
| `grid[...,1]` | `(B,R,R,R)` | 网络推理 | 约 `[0,1]` | 截断归一化 UDF |
| `udf` | `(B,R,R,R)` | 推理后处理 | `[0,0.015]` | 乘回 `clip_value` 后的稠密 UDF |
| `recon_udf.ply` | 网格 | 可视化输出 | 坐标在 `[-1,1]^3` | UDF=`udf_threshold` 的等值面 |
| `udf_g.npy` | `(F,)` | 分割输入 | `[0,0.015]` | SDF 表面每个三角形中心处的 UDF |
| `recon_udf.npy` | `(R,R,R)` | 兼容输入，可选 | 应为距离尺度 | 历史/外部稠密 UDF；本推理脚本默认不保存它 |

### 12. 阈值之间的关系与常见误区

#### 12.1 `clip_value` 和 `udf_threshold` 不是一回事

- `clip_value=0.015`：定义网络 UDF 的截断/缩放范围；大于它的距离都映射为 1。
- `udf_threshold=0.01`：从稠密 UDF 中提取 `recon_udf.ply` 时的等值面。
- `threshold1=0.005`、`threshold2=0.01`：对 `udf_g.npy` 做三角面分割时使用的门限。

虽然两个地方都出现 `0.01`，它们服务于不同步骤。

#### 12.2 UDF 的零值不是“实体表面”

`SDF=0` 是实体表面；`UDF=0` 是 Voronoi 分界场。两者的交集才与表面上的 B-rep 边界有关。不能对 UDF 做 `iso=0` 后把结果当作实体表面。

#### 12.3 `recon_udf.ply` 不参与默认标量分割

默认 `hierarchical` 分割读取的是 `udf_g.npy`，不是 `recon_udf.ply`。后者主要用于可视化或可选的 `udf_mesh` 模式。`process_item` 的标量回退输入也是 `recon_udf.npy`，不是 PLY（`clustering.py:1150-1198`）。

#### 12.4 阈值依赖归一化

所有默认阈值都假设最长轴被缩放到约 1.8。如果绕开标准数据加载/归一化流程，直接把原始尺寸网格送进分割，`0.005` 和 `0.01` 将失去原有几何意义。

### 13. 一段对应代码的整体伪代码

```python
# 1. 由 STEP 的 face-labelled samples 构造 Voronoi 分界网格 V
V = calculate_voronoi(normalize(step))

# 2. 构造全局、表面附近、边附近、Voronoi 附近的查询点
Q_surface = concat(random_volume, noisy_surface, noisy_voronoi)
Q_edge = noisy_brep_edges

# 3. 制作 UDF 标签
u_surface = unsigned_distance(Q_surface, V)
u_edge = unsigned_distance(Q_edge, V)
save_npz(..., query_surface_udf=u_surface, query_edge_udf=u_edge)

# 4. 网络尺度
u_target = clip(u, 0, 0.015) / 0.015

# 5. 编码几何，并对任意坐标连续查询两个场
latent = vae.encode(surface_points, edge_points, voronoi_points)
sdf_udf = vae.query(x, vae.decode(latent))  # last dim == 2

# 6. 稠密体与等值面
grid = query_regular_grid([-1, 1] ** 3)
sdf = clip(grid[..., 0], -1, 1) * 0.015
udf = clip(grid[..., 1],  0, 1) * 0.015
surface_mesh = marching_cubes(sdf, iso=0)
udf_shell = marching_cubes(udf, iso=0.01)

# 7. 在重建表面上获得真正用于分割的每面 UDF
centers = mean(surface_mesh.vertices[surface_mesh.faces], axis=1)
face_udf = clip(vae.query(centers)[..., 1], 0, 1) * 0.015

# 8. 删除低 UDF 边界带，对剩余三角面求连通分量
labels = hierarchical_connected_components(
    surface_mesh, face_udf, threshold1=0.005, threshold2=0.01)
```

### 14. 最终理解

DualBrep 的 UDF 可以概括为一个“连续、可查询、经过截断的 Voronoi 分界距离场”。其表示经历了五个层级：

1. **参考几何层**：由带 B-rep face id 的样本构造 `voronoi.ply`；
2. **监督标量层**：查询点到 Voronoi 网格的无符号距离，保存到 `.npz`；
3. **神经隐式层**：与 SDF 共享 latent，由 query decoder 的第二通道连续预测；
4. **离散输出层**：采样为 $R^3$ UDF 体，或提取正阈值等值面 `recon_udf.ply`；
5. **拓扑处理层**：在 SDF 表面三角形中心查询为 `udf_g.npy`，把低值带作为 B-rep 边界，再通过连通分量得到 `cluster.ply`。

所以，UDF 的最终目标不是代替 SDF 重建物体，而是给 SDF 恢复出的几何外壳补上“在哪里切分成 B-rep faces”的拓扑线索。

<a id="topology-seam"></a>
## B. 拓扑恢复、seam 与特殊边

本文整理两组问题：

1. SDF 与 UDF 的零集既然能够给出 B-rep 边界，为什么还需要网络预测拓扑？能否直接从曲线恢复拓扑？
2. GVD 构造过滤 seam 等非面间边之后，后处理怎样建立 seam？其他特殊边又如何处理？

说明依据为 [DualBrep 原论文](https://arxiv.org/html/2606.31579v1)及当前仓库代码。下文区分理想几何条件、论文设计和实际实现；UDF 的完整定义与处理流程另见 [A部分：UDF表示与处理流程](#udf-flow)。

### 1. 核心结论

- **数学上并非必须使用拓扑预测网络。** 如果已经获得准确表面和完整、可靠的嵌入曲线图，可以通过确定性算法推导面、边、顶点及其关联。
- **当前代码没有先提取完整交线图，再重复预测一次拓扑。** 它利用 UDF 分割面内部，然后从面片采样点重新预测参数曲面、面对邻接和裁剪曲线。
- **seam 不是从 GVD 中找回的。** 它依赖预测曲面的闭合性和参数化，由 OCC 自然建面或缺缝修复重新建立。
- **恢复有效 B-rep 不等于无损还原原始 B-rep。** seam 的位置、边的分段、拓扑编号，以及部分特殊边，不保证与原始模型一致。

### 2. 为什么双场零集相交会得到面间边界？

#### 2.1 UDF 是到 GVD 分界面的距离

严格来说，本项目的 UDF 不是直接到 B-rep 边的距离。记实体表面为 $M$，广义 Voronoi 分界面为 $\mathcal D$，则：

$$
M=\{\mathbf x:s(\mathbf x)=0\},\qquad
u(\mathbf x)=d(\mathbf x,\mathcal D).
$$

这里 $s$ 是 SDF，$u$ 是 UDF。GVD 将空间划分为“最近 B-rep face 不同”的区域，$\mathcal D$ 是这些区域的分界。

在理想、准确的面划分下：

- 面内部的点到自身所属面的距离为零，到其他面的距离为正。
- 共享边界上的点同时属于两个或多个面的边界，到这些面的距离均为零，因此位于 Voronoi 分界上。

针对不同面之间的边界 $E$，有：

$$
E=M\cap\mathcal D
=\{\mathbf x:s(\mathbf x)=0,\ u(\mathbf x)=0\}.
$$

这对应论文 §3.1 的解释：**SDF 决定表面在哪里，GVD 决定面身份在哪里改变。**

#### 2.2 理想零集关系与实际近似不能混淆

[`prepare_implicit.py`](../prepare_implicit.py) 对 `voronoi.ply` 求无符号距离；[`Voronoi/calculate_voronoi/`](../Voronoi/calculate_voronoi/) 使用带 face 标签的采样点构造分界网格。实际流程还存在有限采样、网格化和神经场预测误差，所以双场交线通常只是目标边界的近似。

即使限制在实体表面上，到 GVD 分界面的距离也不必等于到真实边界曲线的距离：最近的 GVD 点可能并不位于实体表面。需要区分“零集对应”与“距离值相同”。

此外，GVD 编码的是面身份的变化，并不只编码尖锐棱线。两个不同 face 即使相切或共面，其分界也可以被编码；同一面的周期 seam 则不属于这种面间分界。

### 3. 如果交线足够准确，能否直接推导拓扑？

可以。假设已经得到可靠外壳 $M$ 和完整嵌入其中的边界曲线图 $E$，可以采用以下确定性流程：

1. 沿曲线切分表面，得到面区域。
2. 根据每段曲线两侧的区域，建立 edge–face 关联。
3. 在曲线汇合处建立顶点，得到 edge–vertex 关联。
4. 沿各面边界排序，确定外环、内环及方向。

因此问题不是“曲线信息原则上无法给出拓扑”，而是**怎样从近似连续场中稳定提取完整曲线图，以及怎样把它转换为可装配的参数 B-rep**。

论文 §3.1 也明确指出：曲面、曲线及面—边关联建立后，顶点和边—顶点关联可以确定性推导，不需要所有拓扑元素都由网络单独预测。

不过，这通常得到的是与几何和面划分相容的 B-rep，不保证还原原始 CAD 的每个表示细节。周期 seam 的位置、闭合曲线的分段、人为插入的拓扑顶点等，可能需要额外约定。

### 4. 当前代码为什么仍然预测邻接和边？

#### 4.1 分割阶段交付的是面内部，不是完整曲线图

[`clustering.py`](../clustering.py) 的 `hierarchical_segmentation()` 将低 UDF 三角形视为边界带，对剩余三角形求连通分量。边界带保留为 `label=-1`，默认不回填。

随后，[`rebuild.py`](../rebuild.py) 的 `sample_faces()` 排除 `-1` 区域，从每个保留面片采样点和法线。

以立方体为例，交给后续网络的更接近于六块边缘被削去的面片，而不是六个共享同一套精确边界的面。

`recon_udf.ply` 也不能视为完整边图：[`mesh_utils.py`](../mesh_utils.py) 的 `udf2mesh()` 提取的是 UDF 正阈值等值面，属于二维偏置壳层，而非一维 CAD 曲线。默认标量分割路径不使用这个 PLY 文件。

#### 4.2 参数化网络没有直接接收 UDF 曲线或已知邻接

每个保留面默认采样 100 个点及法线，并附带包围盒。实际推理输入如下：

```214:216:rebuild.py
            data = {"face_sample_points": fsp, "face_input_bbox": fbb,
                    "face_attn_mask": torch.zeros((n, n), dtype=torch.bool, device=device)}
            results = model.inference(data)
```

这里没有 UDF 数值、边界曲线、原网格连接关系或已知 face adjacency。全零的 attention mask 也不是从网格邻接构造的限制。

网络获得了面数、面分组及其几何位置，但仍需根据面片形状、位置和全局上下文判断哪些面对共享边。

#### 4.3 网络联合预测曲面、面对邻接与裁剪曲线

[`rebuild_model.py`](../rebuild_model.py) 中的 `Parametrizer`：

1. 编码每个面片，并通过跨面 attention 汇聚上下文。
2. 为每个面预测 `16×16` 参数曲面网格。
3. 在 `decode_edge()` 中枚举面对，分类判断连接关系。
4. 对判定相邻的面对预测 UV 裁剪曲线。
5. 在预测曲面网格上采样，将 UV 曲线映射为三维边点。

面对分类的阈值判断为：

```463:468:rebuild_model.py
            intersected_edge_feature = torch.cat(intersected_edge_feature, dim=0)
            pred = torch.cat(pred, dim=0)
            pred_labels = torch.sigmoid(pred) > 0.5

            intersected_edge_feature = intersected_edge_feature[pred_labels]
            decoding_results["pred_face_adj"] = pred_labels.reshape(-1)
```

实际推理中的边点由面对中第一张预测曲面网格采样得到，并非 OCC 对两张参数曲面做精确求交，也没有在这一阶段保证边同时精确落在第二张面上。后续仍需几何优化和装配修复。

因此，这个模块不是“对已提取交线再做一次拓扑识别”，而是：

**从分割面片联合补出参数曲面、邻接和边界。**

#### 4.4 为什么选择网络方案？

论文 §3.4 明确给出的动机是：相较启发式拟合，提高抗噪能力，并支持自由曲面。结合实现，可以分为三个问题。

##### 近似几何误差会转化为离散拓扑错误

实际使用的是预测场和有限分辨率采样，而不是精确零集：

- 边界断开一点，两个面可能被合并。
- 两条近邻边界粘连，薄面可能消失。
- 顶点附近几条曲线错开，边界环可能无法闭合。

这些误差在几何距离上很小，但对拓扑是“连接／不连接”的突变。UDF 又没有跨零的正负变化，代码选择了低值带分割，而不是直接追踪精确零线。

##### 网格曲线不等于可装配的参数 B-rep

即使曲线图正确，仍需建立曲面参数域、拟合曲面、得到 UV 裁剪曲线、协调共享边，并处理闭环、孔洞和周期接缝。

独立拟合后的曲面也未必继续精确经过原网格边界。论文预测 UV 曲线，是为了加强边与曲面的几何一致性；当前实现仍需优化边面贴合，再由 OCC 修复和缝合。

##### 网络提供先验，但不保证正确

跨面 attention 可以利用整体形状推断缺失边界，但不能由此断言网络必然优于确定性提取。

论文补充材料 §6 反而将 rebuilder 列为主要瓶颈：较好的分割经过重建后，仍可能因局部误差而无法闭合。网络不是数学上的必需条件，也不是拓扑正确性的保证。

#### 4.5 能否去掉拓扑分类器？

可以作为替代路线：

**保留完整网格和边界带 → 恢复边界曲线及两侧面标签 → 确定邻接 → 曲面参数化与曲线拟合 → OCC 装配。**

当前模型的 `decode_edge_given_topo()` 允许给定面对关系及特征后预测边 UV，但默认流水线没有采用这条路径。真正替换邻接分类器，还需要可靠的边界提取、归属判定及容差处理。

因此，论文中“不需要直接预测离散邻接”的优势应理解为连续表示和生成主干层面的设计：**DualBrep 将离散拓扑处理推迟到显式重建阶段，并没有彻底消除整个流水线中的拓扑预测。**

### 5. GVD 不保存 seam，为什么仍能建立 seam？

#### 5.1 seam 是参数域接缝，不是面间分界

以完整圆柱侧面的参数化 $P$ 为例：

$$
P(0,v)=P(2\pi,v).
$$

三维空间中这是同一条母线，展开到矩形 UV 域后，却分别对应 $u=0$ 和 $u=2\pi$ 两条边界。

B-rep 通常用同一条拓扑 `edge`，在同一个 `face` 的边界中反向使用两次，并在该面上保存两条 `pcurve`，即二维参数曲线。

因此 seam：

- 不分隔两个不同的面。
- 不是实体表面的裂缝。
- 是闭合曲面参数域表示所需的接缝。

圆柱 seam 放在哪个角度，实体外形和面间 GVD 都可以不变，所以仅凭 SDF/GVD 无法唯一反推原始 seam 位置。

对于普通闭合流形实体，除退化等特殊情况外，两侧属于同一面的常规边主要就是这类 seam。

#### 5.2 论文仍通过参数网格表达闭合结构

论文 §3.1 将周期曲面沿 seam 展开为矩形 UV 网格，首末行或列重合。因此，虽然 seam 被排除在 GVD 面间分界之外，曲面的闭合性仍体现在参数网格表示中。

参数化网络需要学习这种结构；但预测网格并不天然保证首尾完全重合，也不保证原始 seam 的位置不变。

### 6. 当前代码如何重新构造 seam？

#### 6.1 拟合曲面后，尝试设置周期性

[`brep_post/utils.py`](../brep_post/utils.py) 的 `create_surface()` 先拟合 B-spline 曲面，再由内部函数 `set_face_uv_periodic()` 根据参数边界端点距离与网格平均间距，尝试设置 U/V 周期性。例如 U 方向：

```776:780:brep_post/utils.py
        u_min, u_max, v_min, v_max = geom_face.Bounds()
        us = geom_face.Value(u_min, v_min)
        ue = geom_face.Value(u_max, v_min)
        if us.Distance(ue) < np.mean(u_intervals):
            geom_face.SetUPeriodic()
```

V 方向采用类似处理。

这是启发式判断，不是逐点验证整条对边，也没有强制预测网格首尾相同。端部接近、几何闭合与成功建立周期 B-spline 并不是完全等价的条件，因此周期性识别和拟合可能失败。

#### 6.2 自然建面或缺缝修复提供 seam

`create_trimmed_face_from_wire()` 有两种相关机制。

**自然边界建面：**

调用 `BRepBuilderAPI_MakeFace()`。对覆盖完整周期的参数域，生成的自然边界可以包含 seam。代码还有一个在周期面、自然 wire 数量和两条预测边长度等条件满足时，直接采用自然 wire 的替换分支。

**缺缝修复：**

代码显式开启 `SetFixMissingSeamMode(True)`，执行 `Perform()` 后还再次调用：

```1099:1100:brep_post/utils.py
        face_fixer.FixOrientation()
        face_fixer.FixMissingSeam()
```

OCC 对 `FixMissingSeam()` 的定义是：闭合曲面上的两个 wire 在三维中闭合，但在二维参数域存在缺口时，加入 seam，将它们组成新的边界 wire。

典型情况是圆柱侧面的上下两个圆环：三维中各自闭合，展开到 UV 域却还需要左右接缝，才能形成完整边界。

这不是无条件补全任意缺边的算法。拟合曲面、已有 wire 和容差必须满足修复条件，随后还需要通过面及实体有效性检查。

#### 6.3 网络预测的同面边没有直接进入最终装配

[`brep_post/construct_brep.py`](../brep_post/construct_brep.py) 在构建面—边关系时，遇到 `face1 == face2` 会直接跳过。将该边保留为 seam 的相关代码被注释了。

实际分工是：

**参数化网络提供曲面网格与闭合线索 → OCC 建立或修复 seam → 校验面和实体。**

这里建立的是满足当前参数化和边界条件的可用 seam，而不是找回被 GVD 删除的原始边。其位置、拓扑身份和分段方式均不保证与原始 CAD 一致。

### 7. 除 seam 外，还有哪些不由两个不同面共享的边？

应区分一般 B-rep 中可能存在的结构，以及本项目实际支持的范围。

| 类型 | 几何／拓扑含义 | 当前项目的处理 |
|---|---|---|
| **退化边** | UV 域中是边，映射到三维却缩成一个点，例如某些球面极点、圆锥顶点的参数边界；不是普通短边 | GVD 预处理通过 `BRep_Tool::Degenerated()` 检测后报错拒绝处理，不是简单删除。后处理中专门的 `FixPeriodicDegenerated` 模式也被关闭，不能声称统一补回 |
| **自由边／开边界边** | 只被一个面使用一次，例如开放薄片的外轮廓，另一侧没有面 | 不属于目标闭合实体的常规边。预处理要求保留边有两次面关联，不满足会拒绝 |
| **同面内部边、裂缝／切口边** | 一般拓扑数据中可能存在的不分隔两个独立 face 的附加结构，不一定是周期 seam | 代码没有专门区分这些语义：同面正反向重复边会跳过，残留的两侧同面边也会删除；没有通用恢复机制。若只出现一次，也可能被面关联数检查拒绝 |

相关过滤逻辑位于 [`Voronoi/calculate_voronoi/calculate_voronoi.cpp`](../Voronoi/calculate_voronoi/calculate_voronoi.cpp) 的 `sample_points_surface()`。

“退化”“自由”“seam”不是完全同一维度的分类：退化描述几何映射性质，自由边和 seam 更多描述边在面边界中的使用方式。

#### 7.1 闭合曲线不等于 seam

圆柱上下圆环的首尾闭合，但通常分别连接侧面与端面，是正常的不同面共享边，不会仅因曲线闭合而被 GVD 排除。

反过来，圆柱轴向 seam 的两个端点通常不重合。判断 seam 要看它相对于面的参数曲线和拓扑使用方式，而不是只看三维曲线是否闭合。

#### 7.2 光滑连接边也不等于 seam

两个不同 face 即使相切或共面，没有明显棱角，其分界仍可进入 GVD。判断依据是 face 身份是否不同，而非法向是否突变。

### 8. 总结

双场包含面划分信息，但将近似场变为可靠的显式曲线图和参数 B-rep，仍是独立的重建问题。当前项目选择学习式参数化、邻接预测和 UV 裁剪曲线预测，再由几何优化与 OCC 完成装配；确定性拓扑提取是可行的替代方向，并非数学上不可能。

GVD 负责面间分界；seam 由曲面参数化和 CAD 内核补建；其他非面间边没有统一恢复机制。因此，整个流程的目标是得到可用的参数 B-rep，而非无损还原原始 B-rep 的全部拓扑细节。

### 参考资料

- [DualBrep 原论文](https://arxiv.org/html/2606.31579v1)：§3.1 双场与 B-rep 表示；§3.3 网格分割；§3.4 学习式重建；补充材料 §1 GVD 构造、§6 重建瓶颈。
- [OCC `ShapeFix_Face` 官方参考](https://occt3d.com/dev/doc/refman/html/class_shape_fix___face.html)：`FixMissingSeam()`、`FixPeriodicDegenerated()` 等接口的适用条件。
- [OCC `BRep_Tool` 官方参考](https://occt3d.com/dev/doc/refman/html/class_b_rep___tool.html)：`IsClosed(edge, face)` 的双 pcurve 定义及退化边查询。
- [A部分：UDF表示与处理流程](#udf-flow)：本仓库 UDF 定义、采样、预测与分割流程。
