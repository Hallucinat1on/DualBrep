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
