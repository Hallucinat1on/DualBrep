# DualBrep 中 UDF 的表示与处理流程

本文完全依据当前仓库代码说明 UDF（Unsigned Distance Field，无符号距离场）在 DualBrep 中的几何含义、数据表示、生成方式、网络表示、推理后处理以及如何用于 B-rep 面分割。

## 1. 先给出结论

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

## 2. UDF 为什么能补充 SDF

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

## 3. UDF 参考几何 `voronoi.ply` 是怎样得到的

### 3.1 坐标归一化

`Voronoi/calculate_voronoi/calculate_voronoi.cpp:510-547` 先读取 STEP 包围盒，将中心平移到原点，并按最长轴做等比例缩放，使模型落入大约 $[-0.9,0.9]^3$。`prepare_implicit.py` 使用同样的 `BOUNDING = 0.9` 和 `normalize_shape`，保证 STEP 三角网格与 `voronoi.ply` 位于同一坐标系（`prepare_implicit.py:98-103`）。

设原始包围盒中心为 $\mathbf c$，最长边尺寸为 $L_{\max}$，则可以把归一化理解为：

$$
\mathbf{x}_{norm}=(\mathbf{x}_{orig}-\mathbf c)\frac{1.8}{L_{\max}}
$$

因此配置中的 `0.015`、`0.01`、`0.005` 等距离阈值都是归一化空间中的长度，而不是原始 STEP 文件的毫米或米。

### 3.2 按 B-rep face 标记采样点

`sample_points_surface` 为采样点维护 `primitive_index`，用它记录点来自哪个 B-rep face（`calculate_voronoi.cpp:102-119`）。在共享 edge 两侧，代码沿曲线采样，然后利用面法向和边切向的叉积，把采样点向两个相邻面的内部轻微平移：

$$
\mathbf d=\mathbf n\times\mathbf t,\qquad
\mathbf p'=\mathbf p+\epsilon\mathbf d
$$

这里 `voronoi_distance = 0.0001`，相关实现位于 `calculate_voronoi.cpp:227-250` 和 `calculate_voronoi.cpp:312-369`。这样同一条边两侧会产生分别带有两个 face id 的采样点，使后续 Voronoi 分界能够穿过真实面边界。

### 3.3 构造不同 face 样本之间的 Voronoi 分界面

`compute_voronoi` 对全部采样点建立三维 Delaunay/Voronoi 结构，并把 Voronoi cell 的 facet 三角化（`geom_voronoi.cpp:11-90`）。随后只保留相邻采样点所属 `primitive_index` 不同的公共 facet（`geom_voronoi.cpp:147-180`），最终输出 `voronoi.ply`（`geom_voronoi.cpp:213-220`）。

因此，这里编码的不是语义类别，而是“离不同 B-rep face 样本等距”的几何分界。它是一张三维三角网格，并可能从 CAD 边界向实体内部或外部延伸。

## 4. UDF 监督数据如何生成

入口是 `prepare_implicit.py::build_implicit_npz`。

### 4.1 输入给编码器的三类点云

每个样本包含三类带方向的点：

| 字段 | 典型形状 | 内容 | 代码位置 |
|---|---:|---|---|
| `surface_points` | `(100000, 6)` | 表面 xyz + normal | `prepare_implicit.py:109-111` |
| `edge_points` | `(100000, 6)` | B-rep 边 xyz + tangent | `prepare_implicit.py:113-117` |
| `voronoi_points` | `(Nv, 6)` | Voronoi 网格 xyz + normal | `prepare_implicit.py:130-135` |

注意：`edge_points` 是编码器的几何输入之一；它本身不是 UDF 标签。UDF 标签仍然是查询点到 `voronoi.ply` 的距离。

Voronoi 点还会按照“到实体表面的无符号距离小于 0.2”进行筛选，使编码器关注靠近物体、与表面分割更相关的 Voronoi 区域。

### 4.2 两组查询点

代码针对不同区域构造两组查询：

1. `query_edge_points`

   从稠密 B-rep edge 采样点出发，分别添加标准差为 `0.001`、`0.005`、`0.007`、`0.01` 的高斯噪声（`prepare_implicit.py:119-122`）。它强化边界附近的监督。

2. `query_surface_points`

   由三部分拼接（`prepare_implicit.py:124-139`）：

   - $[-1,1]^3$ 内 200000 个均匀随机点，用于覆盖全局空间；
   - 表面采样点加 `0.001`、`0.005` 高斯噪声，用于强化 SDF 零面附近；
   - Voronoi 采样点加 `0.001`、`0.005` 高斯噪声，用于强化 UDF 零集附近。

这种混合采样避免了均匀体采样几乎碰不到薄的表面/分界区域，同时又保留全局场监督。

### 4.3 用 libigl 计算标量标签

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

## 5. 截断与归一化：网络真正看到的 UDF

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

## 6. 网络内部怎样表示 UDF

### 6.1 输入几何变成 latent set

`DualVAE.encode` 先对三类点的 xyz 做 Fourier 特征编码，再与 normal/tangent 拼接并投影到宽度 768（`model.py:337-353`）。随后：

- 对表面点和边点分别用 FPS 选择 token；
- 用 cross-attention 分别聚合 surface、edge、Voronoi 特征；
- 把三个分支相加并经过 self-attention；
- 用 `pre_kl` 将每个 token 压缩为 32 维 latent。

对应代码为 `model.py:355-383`。这里没有一个单独的“UDF latent”；SDF 和 UDF 共享同一个几何/拓扑 latent set。

### 6.2 查询解码器输出两个标量

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

### 6.3 关于训练损失的代码边界

当前仓库的 `model.py` 是推理版模型，保留了编码、解码和 query 逻辑，但没有提供利用 `query_surface_udf` / `query_edge_udf` 计算重建损失的 VAE 训练 `forward`/`loss`。仓库中能确认的是标签准备、归一化和已训练 checkpoint 的推理接口；不能仅依据当前代码断言训练时 UDF 使用的是 L1、L2，或两类查询的具体损失权重。

另外，在当前 `ae_reconstruct.py` 的推理路径里，`.npz` 中的查询标签只会触发 `augment` 并完成标准化/姿态处理，真正编码使用的是 surface/edge/Voronoi 三类点云；推理并不会用 ground-truth UDF 去修正预测结果。

## 7. 从 latent 得到稠密 UDF 体

### 7.1 普通稠密查询

`model.py::_inference` 在每个轴上生成 `linspace(-1,1,res)`，形成 $R^3$ 个查询点，并以每批 100000 点调用 `model.query`（`model.py:172-197`）。返回：

$$
\text{grid shape}=[B,R,R,R,2]
$$

其中 `grid[..., 1]` 就是归一化 UDF 体。

`ae_reconstruct.py:97-100` 将它转回截断距离：

```python
udf = np.clip(grid[..., 1], 0, 1) * clip_value
```

### 7.2 加速多分辨率查询

默认 `runtime.acc: true` 使用 `_inference_acc`：

1. 在 $128^3$ 网格上完整计算 SDF 和 UDF；
2. 根据 `abs(SDF) < 0.010/0.015` 找出近表面体素，并做一次 3×3×3 膨胀；
3. 对其他区域三线性上采样，只在近 SDF 表面的掩码中重新查询 256 或 512 分辨率；
4. SDF、UDF 使用完全相同的掩码更新。

关键实现位于 `model.py:225-274`。这说明高分辨率 UDF 的精确求值区域是由 **SDF 近表面带** 决定的，而不是由 UDF 自己的低值区域决定。这样设计是合理的，因为 UDF 的主要下游用途是在 SDF 表面上分割；远离物体表面的 Voronoi 细节并不重要。

## 8. UDF 得到后有两条处理路径

### 8.1 路径 A：UDF 等值面可视化/边线管道

UDF 没有正负号，其零集通常非常薄，数值预测也很难恰好等于零，因此不能像 SDF 一样直接稳定地对 `iso=0` 做 marching cubes。项目使用一个正阈值：

$$
\mathcal M_u(\tau)=\{\mathbf{x}\mid u(\mathbf{x})=\tau\}
$$

默认 `udf_threshold = 0.01`。`mesh_utils.py::udf2mesh` 对该等值面运行 marching cubes，再把体素坐标映射回 $[-1,1]^3$（`mesh_utils.py:21-27`），输出 `recon_udf.ply`。

从严格几何意义看，`recon_udf.ply` 是距离 Voronoi 场为 $\tau$ 的偏置壳层；代码和 README 把它称为 edge/wireframe iso-surface，是因为它在目标表面附近刻画 B-rep 分界带。它不是 OCC 中一维解析曲线的直接重建结果。

### 8.2 路径 B：在 SDF 表面上采样 UDF，用于面分割

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

## 9. UDF 如何驱动 B-rep face 分割

`clustering.py::process_item` 的默认模式是 `hierarchical`，参数为：

```python
threshold1 = 0.005
threshold2 = 0.01
filter_size = 10
mode = "hierarchical"
```

这些值位于 `clustering.py:38-48`，单位均为归一化坐标长度。

### 9.1 先只保留最大网格连通分量

`process_item` 读取 `recon_sdf.ply`，根据三角面邻接关系求连通分量，只保留三角形数量最大的分量，并用同一个 mask 过滤 `udf_g.npy`（`clustering.py:1177-1187`）。这会去掉 SDF marching cubes 产生的游离小壳层。

### 9.2 第一遍：低 UDF 作为边界带

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

### 9.3 第二遍：更高阈值重新切分

对于最大 UDF 达到 `threshold2=0.01` 的第一遍 cluster，再仅保留：

$$
A_2=\{i\mid u_i\ge 0.01\}
$$

并在原 cluster 内重新求连通分量（`clustering.py:508-540`）。若得到至少两个足够大的子分量，就用它们替换原 cluster；否则保留第一遍结果。这一遍能够切开第一遍低阈值下仍通过较宽通道粘连的区域。

最终标签写入 `cluster.ply` 的 per-face `label` 属性。这里得到的是与 CAD 曲面片对应的几何 face cluster，不是“轮毂、螺栓、孔”等语义零件标签。

## 10. AE 重建与条件生成中的处理差异

### 10.1 AE 重建

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

### 10.2 点云/图像条件生成

`diffusion_model.py::FusedModelFlow.inference` 从随机噪声出发，通过 rectified-flow ODE 得到 VAE latent，再由冻结的 DualVAE decoder 解码（`diffusion_model.py:128-145`）。`vae_generate.py` 后续采用和 AE 基本相同的稠密查询、反归一化、marching cubes、面中心 UDF 查询与分割流程。

因此生成模式中也不是 diffusion 模型直接输出体素 UDF；flow 生成的是 latent set，连续 UDF 仍由 `DualVAE.query(x, latent)` 按需计算。

## 11. 各种 UDF 文件/张量不要混淆

| 名称 | 形状 | 所处阶段 | 数值尺度 | 含义 |
|---|---:|---|---|---|
| `query_surface_udf` | `(Qs,)` | 训练数据 `.npz` | 原始归一化空间距离，保存为 float16 | surface/global/Voronoi 查询点到 `voronoi.ply` 的距离 |
| `query_edge_udf` | `(Qe,)` | 训练数据 `.npz` | 同上 | edge 邻域查询点到 `voronoi.ply` 的距离 |
| `grid[...,1]` | `(B,R,R,R)` | 网络推理 | 约 `[0,1]` | 截断归一化 UDF |
| `udf` | `(B,R,R,R)` | 推理后处理 | `[0,0.015]` | 乘回 `clip_value` 后的稠密 UDF |
| `recon_udf.ply` | 网格 | 可视化输出 | 坐标在 `[-1,1]^3` | UDF=`udf_threshold` 的等值面 |
| `udf_g.npy` | `(F,)` | 分割输入 | `[0,0.015]` | SDF 表面每个三角形中心处的 UDF |
| `recon_udf.npy` | `(R,R,R)` | 兼容输入，可选 | 应为距离尺度 | 历史/外部稠密 UDF；本推理脚本默认不保存它 |

## 12. 阈值之间的关系与常见误区

### 12.1 `clip_value` 和 `udf_threshold` 不是一回事

- `clip_value=0.015`：定义网络 UDF 的截断/缩放范围；大于它的距离都映射为 1。
- `udf_threshold=0.01`：从稠密 UDF 中提取 `recon_udf.ply` 时的等值面。
- `threshold1=0.005`、`threshold2=0.01`：对 `udf_g.npy` 做三角面分割时使用的门限。

虽然两个地方都出现 `0.01`，它们服务于不同步骤。

### 12.2 UDF 的零值不是“实体表面”

`SDF=0` 是实体表面；`UDF=0` 是 Voronoi 分界场。两者的交集才与表面上的 B-rep 边界有关。不能对 UDF 做 `iso=0` 后把结果当作实体表面。

### 12.3 `recon_udf.ply` 不参与默认标量分割

默认 `hierarchical` 分割读取的是 `udf_g.npy`，不是 `recon_udf.ply`。后者主要用于可视化或可选的 `udf_mesh` 模式。`process_item` 的标量回退输入也是 `recon_udf.npy`，不是 PLY（`clustering.py:1150-1198`）。

### 12.4 阈值依赖归一化

所有默认阈值都假设最长轴被缩放到约 1.8。如果绕开标准数据加载/归一化流程，直接把原始尺寸网格送进分割，`0.005` 和 `0.01` 将失去原有几何意义。

## 13. 一段对应代码的整体伪代码

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

## 14. 最终理解

DualBrep 的 UDF 可以概括为一个“连续、可查询、经过截断的 Voronoi 分界距离场”。其表示经历了五个层级：

1. **参考几何层**：由带 B-rep face id 的样本构造 `voronoi.ply`；
2. **监督标量层**：查询点到 Voronoi 网格的无符号距离，保存到 `.npz`；
3. **神经隐式层**：与 SDF 共享 latent，由 query decoder 的第二通道连续预测；
4. **离散输出层**：采样为 $R^3$ UDF 体，或提取正阈值等值面 `recon_udf.ply`；
5. **拓扑处理层**：在 SDF 表面三角形中心查询为 `udf_g.npy`，把低值带作为 B-rep 边界，再通过连通分量得到 `cluster.ply`。

所以，UDF 的最终目标不是代替 SDF 重建物体，而是给 SDF 恢复出的几何外壳补上“在哪里切分成 B-rep faces”的拓扑线索。

## 15. 从 SDF/UDF 到最终 B-rep 的完整还原流程

前面的内容解释到了 `cluster.ply`。但 `cluster.ply` 仍然只是带面标签的三角网格，并不是严格意义上的 B-rep。B-rep 除了几何形状，还必须显式包含：

- face 对应的参数曲面；
- edge 对应的参数曲线；
- 每条 edge 邻接哪些 face；
- edge 如何组成闭合 wire；
- wire 如何裁剪无限参数曲面；
- 多个 trimmed face 如何缝合为闭合 shell 和 solid。

本仓库通过“隐式场重建 → 网格分割 → 参数化网络 → OpenCASCADE 组装”完成这一转换。总入口 `run_pipeline.sh` 将它划分为三步：

```text
输入点云/隐式样本
  │
  ├─ ae_reconstruct.py
  │    SDF/UDF → recon_sdf.ply + udf_g.npy → cluster.ply
  │
  ├─ rebuild.py
  │    cluster.ply → 参数曲面网格 + 交线 + 拓扑 → post.npz
  │
  └─ postprocess.py + brep_post/
       post.npz → B-spline 曲面/曲线 → trimmed faces
                → sewn shell → solid → STEP
```

### 15.1 第一阶段：SDF 恢复几何外壳

`ae_reconstruct.py:97-100` 先把网络的两个输出通道恢复为归一化空间中的截断距离：

```python
sdf = np.clip(grid[..., 0], -1, 1) * clip_value
udf = np.clip(grid[..., 1],  0, 1) * clip_value
```

然后 `mesh_utils.py::sdf2mesh` 在：

$$
\mathcal S=\{\mathbf x\mid s(\mathbf x)=0\}
$$

上执行 marching cubes：

```python
v, f = mcubes.marching_cubes(v_sdf, 0.0)
v = v / (res - 1) * 2 - 1
```

结果 `recon_sdf.ply` 是位于 $[-1,1]^3$ 中的三角网格。它提供了后续 B-rep 的整体几何外形，但此时没有参数曲面、解析曲线或显式 face-edge 拓扑。

`recon_udf.ply` 是在 `UDF=udf_threshold` 上提取的等值壳，仅用于显示 UDF 分界场或可选的 `udf_mesh` 分割模式。默认 B-rep 还原并不直接把它拟合成 OCC edge。

### 15.2 第二阶段：UDF 将外壳切分成候选 B-rep faces

`ae_reconstruct.py:115-121` 在 `recon_sdf.ply` 每个三角形中心直接查询网络 UDF，保存为 `udf_g.npy`。`clustering.py::process_item` 随后：

1. 只保留 SDF 网格的最大连通分量；
2. 将 `UDF < 0.005` 的三角形视为边界带并标为 `-1`；
3. 对剩余相邻三角形求连通分量；
4. 过滤过小分量；
5. 用 `UDF >= 0.01` 做第二遍细分；
6. 将 cluster id 写入 `cluster.ply` 的 per-face `label` 属性。

因此 `cluster.ply` 中的一个 label 是一个候选 B-rep face 的离散支撑区域：

```text
recon_sdf.ply 中的一组三角形
              +
同一个 per-face label
              ↓
一个候选 B-rep 参数曲面
```

这一步建立的是“哪些三角形属于同一个面”，还没有恢复面之间的准确交线。低 UDF 的边界三角形被标成 `-1`，后续参数化时不会作为任何 face 的输入。

### 15.3 第三阶段：把每个三角网格分块变成参数化网络输入

入口是 `rebuild.py`。`read_cluster` 从 PLY face 属性中读取 `label`；`sample_faces` 遍历除 `-1` 外的所有唯一标签（`rebuild.py:76-101`）。对每个标签：

1. 提取属于该 label 的子网格；
2. 用 Open3D 均匀采样默认 100 个点；
3. 每个点拼接三角面法向，形成 `(x,y,z,nx,ny,nz)`；
4. 计算该分块的中心和三轴 extent，形成 6 维 bbox。

模型输入因而是：

| 字段 | 形状 | 含义 |
|---|---:|---|
| `face_sample_points` | `(N,100,6)` | $N$ 个候选 face 的位置与法向采样 |
| `face_input_bbox` | `(N,6)` | 每个 face 的中心 `(cx,cy,cz)` 与 extent `(sx,sy,sz)` |
| `face_attn_mask` | `(N,N)` | face 之间的 attention mask；推理时全为 `False` |

这里 `normalize_coord0516` 计算：

$$
\mathbf c_i=\operatorname{mean}(P_i),\qquad
\mathbf e_i=\max(P_i)-\min(P_i)
$$

薄轴 extent 小于 `0.01` 时，在归一化/反归一化计算里暂时用 1 替代，避免除零（`rebuild_model.py:66-114`）。

若一个 `cluster.ply` 少于两个有效 label，`rebuild.py:208-211` 会跳过该候选，因为后续模型需要从多个 face 中预测相交关系。

### 15.4 测试时旋转：为几何组装生成多个候选

`rebuild.py` 使用八面体旋转群的 24 个旋转矩阵。对每个旋转 $M_k$：

```python
m = mesh.copy()
T[:3, :3] = ROT[k]
m.apply_transform(T)
```

随后重新采样并独立运行参数化网络，候选保存到：

```text
<out>/tmp/<shape>_<k>/post.npz
```

旋转没有改变目标拓扑，但会改变神经网络看到的姿态，因此 24 个候选的曲面、交线和连接误差会略有不同。某个姿态组装失败不意味着所有姿态都会失败。默认 `run_pipeline.sh` 使用 `ROTATIONS=all`。

### 15.5 Parametrizer 如何预测参数曲面

`rebuild_model.py::Parametrizer.encode` 对每个 face 的点云使用 `PointEncoder3` 编码，并把 bbox 特征拼接进去（`rebuild_model.py:373-413`）。所有 face 特征再经过 Transformer，使一个面的表示能参考模型中其他面。

`decode_face` 从一个 $2\times2$ face feature map 连续上采样三次，最终输出：

```text
face_norm: (N,16,16,6)
```

最后 6 个通道表示归一化 xyz 和 normal。网络还预测 bbox 修正量：

```python
delta_bbox = self.face_center_scale_decoder(...)
face_bbox = delta_bbox + face_input_bbox
```

随后 `denormalize_coord0516` 将规则网格恢复到全局坐标，得到：

```text
pred_face: (N,16,16,6)
```

对每个 face，`pred_face[i,:,:,0:3]` 可以看成一个离散参数曲面：

$$
\mathbf S_i(u_p,v_q),\qquad p,q=0,\ldots,15
$$

它与分割三角片的区别是：三角片只有不规则离散表面，而 `16×16` 网格已经建立了规则的二维 $(u,v)$ 参数域，为后续 B-spline 曲面拟合提供输入。

### 15.6 Parametrizer 如何预测 face 邻接和交线

`Parametrizer.decode_edge` 枚举全部有序 face 对，包括 `(i,j)`、`(j,i)` 和 `(i,i)`（`rebuild_model.py:436-450`）。每对 face feature 经过：

```python
self.inter       # 构造相交特征
self.classifier  # 判断两个 face 是否相交
```

分类概率经 sigmoid 后使用 `0.5` 阈值：

```python
pred_labels = torch.sigmoid(pred) > 0.5
```

对判定相交的 face 对，输出：

```text
pred_edge_face_connectivity[e] = [edge_id, face_id_1, face_id_2]
```

这就是 B-rep 中最重要的显式拓扑关系：一条 edge 由哪两个 face 共享。

edge decoder 还为每条边输出 16 个参数点，其中前两个通道是第一个相邻 face 参数域中的 $(u,v)$。`hermite_sample` 把它们从 `[0,1]` 转到 `grid_sample` 使用的 `[-1,1]`，再在预测的 `16×16` face 网格上做双线性采样（`rebuild_model.py:13-24, 652-665`）：

$$
\mathbf C_e(t_k)=mathbf S_{f_1}(u_k,v_k),qquad k=0,\ldots,15
$$

因此得到：

```text
pred_edge: (E,16,3)
```

这种构造保证预测 edge 点落在第一个相邻预测曲面上；后处理中的几何优化再让它同时靠近第二个相邻曲面。

`rebuild.py::save_outputs` 最终保存：

| `post.npz` 字段 | 形状 | 用途 |
|---|---:|---|
| `pred_face` | `(N,16,16,6)` | 拟合 OCC 参数曲面的规则采样网格 |
| `pred_edge` | `(E,16,3)` | 拟合 OCC 参数曲线的有序采样点 |
| `pred_edge_face_connectivity` | `(E,3)` | `[edge_id, face1, face2]` 拓扑关系 |

`recon_faces.ply` 和 `recon_edges.ply` 只是这些预测的可视化；真正交给 B-rep 组装器的是 `post.npz`。

### 15.7 清理重复 half-edge 并建立局部拓扑

`postprocess.py::build_one` 调用 `brep_post.construct_brep.construct_brep_from_datanpz`。`get_data` 读取 `post.npz` 并建立 `Shape`（`construct_brep.py:122-153`）。

由于模型枚举的是有序 face 对，`(i,j)` 和 `(j,i)` 可能各预测一条几何上重复的 edge。`Shape.remove_half_edges` 会把正反 face 对归到一起，并根据预测 edge 到两个相邻 face 的 Chamfer 距离选择更合适的一条；明显远离两个面的候选会被删除（`brep_post/utils.py:155-241`）。

随后：

- `check_openness` 根据曲线首尾方向判断 edge 是否近似闭合；
- `build_fe` 建立每个 face 对应的 edge 列表 `face_edge_adj`；
- `build_vertices` 搜索三个两两相邻 face 构成的环，并确定三条 edge 中应当汇聚的端点。

此时已经有了近似的 face-edge-vertex 组合关系，但不同网络输出之间通常还有小间隙。

### 15.8 几何优化：让交线真正贴合两个相邻曲面

默认 `postprocess.py` 开启几何优化，最大迭代数默认为 200。`brep_post/utils.py::optimize` 为：

- 每条 edge 学习逐轴缩放和平移参数；
- 每个 face 学习一个平移量。

主要损失包括：

1. **edge-face 贴合损失**：每条边同时靠近两个相邻 face，并惩罚到两侧距离不平衡；
2. **corner 损失**：三个 face 交汇处对应的三条 edge 端点应汇聚到同一点；
3. **wire 连通损失**：同一个 face 周围各 edge 的端点应能两两连接；
4. **正则项**：限制 edge 变换和 face 平移不要偏离原预测太远。

核心形式可概括为：

$$
L=L_{edge\leftrightarrow face}+L_{corner}+L_{wire}+L_{reg}
$$

对应实现位于 `brep_post/utils.py:504-635`。如果优化被判断为发散，代码会退回原始 face/edge 预测，而不是使用发散结果。

### 15.9 离散规则网格拟合为 OCC B-spline 几何

几何优化后，`construct_brep.py:295-303` 分别调用：

```python
recon_geom_faces = [create_surface(points) ...]
recon_geom_curves = [create_edge(points) ...]
```

#### 曲面拟合

`create_surface` 将 `16×16` xyz 网格填入 `TColgp_Array2OfPnt`，调用：

```python
GeomAPI_PointsToBSplineSurface(...).Surface()
```

生成 `Geom_BSplineSurface`。代码依次尝试：

```text
FACE_FITTING_TOLERANCE = [0.001, 0.01, 0.03, 0.05, 0.08]
```

以采样点到拟合曲面的 `RMSE + max_error` 选取较好结果；目标连续性为 `GeomAbs_C2`，曲面次数范围为 3 到 8。若规则网格首尾足够接近，还会把曲面设置为 U 或 V 周期曲面（`brep_post/utils.py:762-829`）。

#### 曲线拟合

`create_edge` 将 16 个 edge 点填入 `TColgp_Array1OfPnt`，调用：

```python
GeomAPI_PointsToBSpline(...).Curve()
```

依次尝试：

```text
EDGE_FITTING_TOLERANCE = [0.001, 0.005, 0.008, 0.05]
```

同样按拟合误差选取曲线，连续性为 C2，次数最大为 8（`brep_post/utils.py:832-878`）。

然后用 `BRepBuilderAPI_MakeEdge` 把几何曲线变成拓扑 edge。注意这里的 B-spline 是对神经网络预测点的近似拟合，不是在识别平面、圆柱、圆或直线等解析 primitive。

### 15.10 用 edge 构造 wire，并裁剪参数曲面

一个无限延伸的 `Geom_BSplineSurface` 还不是有限 B-rep face。对于 face $i$，代码利用 `pred_edge_face_connectivity` 收集所有相邻 edge（`construct_brep.py:324-345`），然后尝试把无序 edge 连接成闭合 wire：

```python
ShapeAnalysis_FreeBounds.ConnectEdgesToWires(
    edges, connected_tolerance, False)
```

连接容差按以下序列从严到松尝试：

```text
[0.002, 0.006, 0.01, 0.015, 0.02, 0.025, 0.05, 0.08]
```

如果无序连接无法闭合，代码会回退到 `create_wire_from_ordered_edges`：按照最近端点顺序串联 edge，并使用 `ShapeFix_Wire` 修复局部间隙（`brep_post/utils.py:940-1011`）。

得到 wire 后，`create_trimmed_face_from_wire` 使用 `ShapeFix_Face` 将一个或多个 wire 添加到参数曲面上，修复三维 gap、方向和缺失 seam，并通过 `BRepCheck_Analyzer` 验证（`brep_post/utils.py:1025-1122`）。

这里：

- 最大 wire 通常是外边界；
- 其他闭合 wire 可以表示孔洞的内边界；
- 参数曲面提供 face 的几何；
- wire 提供 face 的有限拓扑边界。

只有成功裁剪的结果才成为 OCC `TopoDS_Face`。

### 15.11 从 trimmed faces 缝合 shell 和 solid

当成功裁剪的 face 数量超过候选 face 数量的 80% 时，代码调用 `get_solid`（`construct_brep.py:373-385`）：

1. 将所有 trimmed face 加入 `BRepBuilderAPI_Sewing`；
2. 按逐渐增大的连接容差缝合共享边；
3. 若结果是多个不相连部分组成的 `COMPOUND`，直接判定当前 solid 失败；
4. 用 `ShapeFix_Shell` 修复 face 和 shell 方向；
5. 用 `BRepBuilderAPI_MakeSolid` 从闭合 shell 构造 solid；
6. 用 `ShapeFix_Solid` 修复 shell/solid；
7. 仅当结果类型是 `TopAbs_SOLID` 且 `BRepCheck_Analyzer` 有效时返回。

实现位于 `brep_post/utils.py:1259-1321`。

成功后，`construct_brep.py:387-398` 先应用 $M_k^{-1}$ 撤销 `rebuild.py` 中的测试时旋转，再写出 `recon_brep.step`。代码会重新读取 STEP，并同时检查：

```text
shape.ShapeType() == TopAbs_SOLID
BRepCheck_Analyzer(shape).IsValid() == True
```

只有两项都通过，才生成 `success.txt` 和 `recon_brep.stl`。如果无法形成有效 solid，代码可能额外输出由全部面组成的 compound STEP 作为调试结果，但它没有 `success.txt`，不会被当作成功 B-rep 提升到最终输出。

### 15.12 多旋转候选的选择与最终文件

`postprocess.py` 可用 Ray 并行组装每个 `<shape>_<rotation>` 候选。它按 shape 分组，选择排序后第一个存在 `success.txt` 的候选，将：

```text
tmp/<shape>_<k>/pp/recon_brep.step
```

复制为：

```text
<out>/<shape>.step
```

并将成功 solid 的 STL 三角化结果转为 `<shape>.ply`。因此最终 `.step` 的判据不是“成功写出了文件”，而是候选确实被 OCC 识别为有效 `SOLID`。

### 15.13 SDF 和 UDF 在 B-rep 还原中的职责边界

完整链路中两种隐式场的作用可以精确区分为：

| 信息来源 | 直接贡献 | 不直接负责 |
|---|---|---|
| SDF | `SDF=0` 生成整体三角外壳，为各 face 提供几何采样 | 不提供显式面邻接和精确参数曲线 |
| UDF | 在 SDF 表面定位低值分界带，产生 face cluster | 默认流程不直接把 `recon_udf.ply` 当 OCC edge |
| Parametrizer | 将 face cluster 重拟合为规则参数网格，预测 edge 和 face-edge 拓扑 | 不再查询原始 SDF/UDF |
| OCC 后处理 | B-spline 拟合、wire、trim、sew、solid、STEP 验证 | 不重新推断缺失的全局语义拓扑 |

换句话说，从 `rebuild.py` 开始，后续代码不再使用稠密 SDF/UDF 数值；它只使用由二者共同产生的 `cluster.ply`。因此早期误差会沿链路传播：

```text
SDF 几何误差
  → cluster 的表面位置不准
  → 参数曲面拟合偏移

UDF 分界误差
  → face 过分割/欠分割
  → face 数量与邻接预测错误
  → wire 无法闭合或 shell 无法密封
```

### 15.14 坐标系与尺寸

测试时八面体旋转会在成功组装后撤销，因此不会改变最终方向。但是否恢复到原始输入尺寸取决于上游数据路径：

- 隐式 STEP `.npz` 路径中的 SDF/UDF 位于归一化空间，当前 B-rep 流程没有再次读取 `calculate_voronoi` 的原始 STEP 归一化参数，所以最终 B-rep 通常仍在归一化坐标系；
- 点云路径若启用 `per_axis_norm`，`ae_reconstruct.py` 会保存 `norm_params.npz`，`clustering.py` 在写 `cluster.ply` 时执行 `v_orig = v_norm * scale + center`，因此参数化和最终 STEP 使用反归一化后的点云坐标；
- 默认点云配置 `per_axis_norm: false` 时，输入仍按最长轴归一化，最终结果也保留该尺度。

### 15.15 运行方式和关键中间结果

完整命令为：

```bash
./run_pipeline.sh
```

等价于依次执行：

```bash
python ae_reconstruct.py \
    config=config_pc.yaml \
    runtime.compute_clustering=true \
    runtime.output_dir=output_pipeline/recon

python rebuild.py \
    --input output_pipeline/recon \
    --out output_pipeline/brep \
    --rotations all

python postprocess.py \
    --input output_pipeline/brep
```

建议排查 B-rep 失败时按以下顺序检查：

| 文件 | 检查内容 |
|---|---|
| `recon_sdf.ply` | 整体表面是否完整、封闭、无明显浮壳 |
| `udf_g.npy` / `cluster.ply` | B-rep 面是否过分割、欠分割，边界是否合理 |
| `recon_faces.ply` | `16×16` 参数曲面是否贴合各 cluster |
| `recon_edges.ply` | 交线是否落在两个相邻面附近，端点能否闭合 |
| `post.npz` | face 数、edge 数和 `[edge,face1,face2]` 是否合理 |
| `pp/optimized_edge.obj` | 几何优化后边界是否连续 |
| `pp/separate_faces.ply` | OCC 拟合后的独立曲面是否正确 |
| `pp/success.txt` | 是否最终形成通过 BRepCheck 的有效 solid |

最终可以把“由 SDF/UDF 还原 B-rep”概括为：

$$
[SDF,UDF]
\rightarrow
\text{segmented triangle surface}
\rightarrow
\text{parametric face/edge samples + topology}
\rightarrow
\text{B-spline geometry + trimmed topology}
\rightarrow
\text{watertight OCC solid}
$$

其中 SDF/UDF 解决连续隐式几何和分界，Parametrizer 解决参数化及显式邻接预测，OpenCASCADE 解决满足 CAD 数据结构要求的几何拟合、拓扑裁剪、缝合和有效性验证。
