# ABC → Voronoi-UDF 表面场：数据处理计划

核对日期：2026-09-13。代码基线：`f37eecfa61e1cc856a67bc5c59763a8e0fc66f61` 加本文所述工作区修改，正式处理需记录实际脚本哈希。本文依据本地数据目录、当前处理脚本和 [SurfaceField.md](SurfaceField.md) 制定。

**目标：以给定 mesh 为条件，直接用 DDPM 或 Flow Matching 生成表面 Voronoi-UDF 标量；不生成几何或完整体 UDF，不以回归模型、VAE 为前置步骤。** 主数据处理器已经实现，并以两个真实 STEP/OBJ 完成实验后端的端到端 smoke test；全量正式标签尚未生成，C++ Voronoi 后端也尚未构建测试。

## 1. 本地数据与首版范围

只读数据源：`/opt/data/private/yihengxu/Datasets/abc`。

```text
abc/
├── step/abc_0000_step_v00.7z
├── step/0000/<shape_id>/*_step_*.step
├── obj/abc_0000_obj_v00.7z
├── obj/<shape_id>/*_trimesh_*.obj
├── step_v00.txt
├── obj_v00.txt
└── download.md
```

本次扫描实际文件、按文件名前缀 ID 配对得到：

| 项目 | 数量 |
| --- | ---: |
| STEP 文件 | 10,000 |
| OBJ 文件 | 7,168 |
| 同时具有 STEP 与 OBJ 的 ID | 7,168 |
| 只有 STEP 的 ID | 2,832 |
| 只有 OBJ 的 ID | 0 |
| 任一格式有多个文件的 ID | 0 |

这是文件级统计，不表示全部配对都通过几何检查。例如 `00000001` 有 STEP，但 OBJ 子目录为空，不能按目录数计算配对数。已核对的一组实际文件是：

```text
step/0000/00009867/00009867_086229ea796640d29acd68ff_step_002.step
obj/00009867/00009867_086229ea796640d29acd68ff_trimesh_002.obj
```

配对记录 `chunk + shape_id + 中间标识串 + 末尾编号`。先按 ID 找候选，再核对其余标识与几何，不盲目拼路径；多匹配和冲突进入隔离清单。

首版以 7,168 组为候选池，质量筛选后才形成训练集合。只有 STEP 的样本后续可通过重新三角化补充，记录不同 `mesh_source`，不伪称为原始 OBJ 配对。ABC 的 OBJ 不是全覆盖，patch/curve 索引映射属于独立 `feat` 格式；本地当前未发现这种特征文件，不能假设 OBJ 自带 CAD face labels。[ABC 官方格式说明](https://deep-geometry.github.io/abc-dataset/)

## 2. 最终标签定义

从 STEP 的 CAD faces 构造三维 Voronoi 分界面网格 $\Gamma_h$。对最终输入 mesh $M_h=(V,T)$ 的每个三角形中心：

$$
q_j=(v_{T_{j0}}+v_{T_{j1}}+v_{T_{j2}})/3,\qquad
g_j=d_{\mathbb R^3}(q_j,\Gamma_h).
$$

保存：

$$
g_j^{\rm raw}=g_j,\qquad
g_j^{\rm metric}=\min(g_j,\tau),\qquad
y_j=g_j^{\rm metric}/\tau,\quad\tau=0.015.
$$

`0.015` 是规范化几何坐标中的距离尺度，不是毫米。`y` 是干净训练目标；`g_metric` 是写入 `udf_g.npy` 的后处理数值。保留 raw 标签，以便改变截断值而不重新计算距离。

它不是到 mesh 自身的距离、不是到 CAD edge 的距离，也不是测地距离。查询对象必须是 Voronoi 三角形集合，而不是其顶点或点云。标签包含 $\Gamma_h$ 的离散构造误差，不能把“距离查询准确”当成“理想 Voronoi 场准确”。查询点取自后处理实际使用的 mesh，不用有厚度的三维窄带代替表面。

## 3. 现有脚本：复用与新增边界

| 文件 | 已有能力 | 本方案用途 |
| --- | --- | --- |
| [calculate_voronoi.cpp](../Voronoi/calculate_voronoi/calculate_voronoi.cpp) | STEP 读取、归一化、按 CAD faces 采样与 Voronoi 调用 | 标签构造核心；先验证坐标与拓扑限制 |
| [geom_voronoi.cpp](../Voronoi/calculate_voronoi/geom_voronoi.cpp) | 输出 `voronoi.ply` | 复用算法，独立验证结果 |
| [prepare_implicit.py](../prepare_implicit.py) | STEP/Voronoi 采样和 libigl 距离查询 | 复用无符号距离计算，不运行完整体域数据流程 |
| [occ_utils.py](../brep_post/occ_utils.py) | OCC 拓扑遍历、三角化、location/朝向处理 | 复用几何操作，新增逐三角形 face ID 导出 |
| [dataset.py](../dataset.py) | 原双场 VAE 推理数据读取 | 不直接使用 `VAEDataset`；新增表面场 loader |
| [ae_reconstruct.py](../ae_reconstruct.py) | 面中心查询、UDF 反归一化导出 | 复用接口约定，不必运行原 VAE |
| [clustering.py](../clustering.py) | UDF 引导的 mesh 分割 | GT 场对照和生成结果后处理 |
| [rebuild.py](../rebuild.py)、[postprocess.py](../postprocess.py) | 参数化、裁剪与 STEP 装配 | 验证下游可重建性，不是标签计算步骤 |
| [eval_seg.py](../eval_seg.py) | 元素几何与拓扑评价 | 需新增符合格式的本地 GT 导出器 |

`prepare_implicit.py` 的 `query_surface_points` 混合体域随机点、扰动表面点、扰动 Voronoi 点，不是纯表面点；它还计算 SDF、采样大量 edge/Voronoi 点并存为 float16。当前主任务不需要这些。其 batch 接口要求平铺的 `<name>.step` 与 `<name>.ply`，不能直接传 ABC 根目录。

`dataset.py` 顶部部分注释把 UDF 描述成 nearest edge，不能据此更改定义；以实际对 `voronoi.ply` 的无符号距离查询为准。GT edge、Voronoi 点和 CAD face ID 只能作为监督或评价，不得作为纯 mesh 条件模型的输入。

## 4. 完整流程

```text
STEP + OBJ 清单
  → 配对、去重分组、train/val/test 划分
  → STEP 拓扑预检 + OBJ 基础质量检查
  → STEP 构造 Voronoi，取得权威归一化变换
  → OBJ、STEP 真值统一坐标，验证对齐
  → 固定输入 mesh、面顺序与邻接
  → 全部三角面中心查询 Voronoi 距离
  → 保存几何条件、场标签与独立评价真值
  → 数值/拓扑质量检查、GT 场后处理对照
  → 发布版本化表面场数据
  → DDPM/FM 固定几何和查询位置，只生成场值
```

### 4.1 Manifest、去重与划分

逐样本保存：ID、chunk、STEP/OBJ 绝对路径、原文件名、大小、内容哈希、配对状态、`mesh_source`、`group_id`、split。后续派生数据附带源文件与算法版本。

建议初始 train/val/test 比例 80/10/10、种子 `20260912`；这是实施建议，不是官方划分。先按重复/同源组划分，再做多分辨率、旋转和退化增强，同一 CAD 的所有派生数据继承同一 split。

文件名中间串可以作为保守的同源线索，但其语义不能只凭名称确定。结合文件哈希、规范化几何指纹与近重复检查建立分组；近重复几何即使面划分不同，也不能轻易散入不同 split。分组变更发布新版本。若日后与论文测试集比较，还要检查训练 ID 与其测试清单重叠情况。

### 4.2 STEP/OBJ 预检与过滤

STEP 使用 OCC 遍历实际 solids/shells/faces/edges/vertices，检查读取成功、形状非空、有限包围盒、非零体积、闭合性与拓扑有效性。不要用 `shape.NbChildren()` 代替 solid 数量。

首版主集合要求单个有效闭合 solid；装配体、多 solid、开放壳体、非流形和退化情况单独分层。不自动只保留最大 solid，不合并共面 faces，否则会改变监督拓扑。

C++ 工具的实际限制需要提前筛查：

- 发现退化 edge 会进入错误路径。
- seam 等处理后，普通 edge 的邻接 face 数必须为 2。
- `geom_voronoi.cpp` 在输出三角形数小于 10 时抛异常。
- 部分 C++ 错误路径使用无活动异常的 `throw;`，可能直接终止进程。逐样本子进程隔离，不能只靠 Python 异常捕获保护整个批次。
- `rebuild.py` 跳过少于两个有效 face 分组的输入。单 face 周期曲面单列，不复制标签来绕过检查。

OBJ 检查索引范围、有限坐标、零面积三角形、重复面、连通分量、边界边、非流形边、朝向和自交风险。清理需记录前后统计和索引映射；改变几何或拓扑语义的修复不静默接受。

### 4.3 Voronoi 构造与统一坐标：批量前必过项

C++ `main` 用 `BRepBndLib::Add` 计算 STEP 包围盒，按最长轴缩放到约 `[-0.9,0.9]`。`normalized_params.txt` 的四个数为平移向量 $b$ 与正尺度 $a$：

$$
b=-\tfrac12(p_{\min}+p_{\max}),\quad
a=\frac{1.8}{\max(p_{\max}-p_{\min})},\quad
x_n=a(x_o+b),\quad x_o=x_n/a-b.
$$

**以 C++ 构造 Voronoi 时的变换为唯一来源，不能另外按 OBJ 包围盒归一化。** Python `normalize_shape` 使用三角化顶点 bbox，而 C++ 使用 OCC bbox，结果可能不同；Python 返回值中的 bbox 还是变换后的 bbox，不能用来推回原始平移。

实施步骤：

1. 导出 `transform.json`：平移、尺度、正/逆 4×4 矩阵、单位约定、源文件哈希。当前 C++ 文本导出使用默认精度，批量前建议改成 17 位双精度输出；暂不修改时必须量化读回误差。
2. OBJ 和评价用 STEP/edge/face 几何全部施加同一变换。先核对 STEP 读取器的单位转换与 OBJ 原始单位；额外比例或偏移需要查明，不能用独立归一化或自动 ICP 隐藏。
3. 当前 C++ `write_topp_shape` 与面采样读取三角化节点时取得了 `TopLoc_Location`，但没有显式应用到节点；Python `get_triangulations` 已应用 `loc.Transformation()`。这是源码风险，不代表已实测所有样本错误。用带非零 placement、旋转和平移的测试 STEP 验证；失败时先修正 C++ 全局坐标处理，再生成标签。
4. 对比变换后的 OBJ、施加同一变换并正确三角化的 STEP、C++ `normalized_mesh.ply`：检查 bbox、双向表面距离、代表性边界位置。误差预算应明显小于分割阈值，记录绝对量和相对 $\tau$ 的比例。

工具输出包括 `normalized_mesh.ply`、`normalized_params.txt`、`sampled_points.ply`、`vertices.ply`、`voronoi.ply`。`sampled_points.ply` 包含构造用偏移/辅助点及 primitive 属性，不是逐三角面 GT 标签；`normalized_mesh.ply` 经 polygon soup repair，不能假设保留 CAD face 顺序。

### 4.4 固定输入 mesh 与索引

保留两种来源，分别记录：

- `abc_obj`：主路线，配对 OBJ 经统一变换与受控清理后的完整 mesh。
- `step_tessellation`：OBJ 不可用时的后续补充或诊断对照；逐 STEP face 三角化，附带 CAD face ID。

STEP 三角化要处理 location 和反向 face winding。逐 face 拼接常产生重复边界顶点，需按 CAD 共享边或经验证的小容差建立共享顶点与邻接。不能使各 CAD face 成为互不连接的 triangle soup，也不能用大焊接容差连接薄壁两侧。

保存权威 `vertices/triangles`、面顺序哈希和清理映射，导出后读回验证。沿用 `recon_sdf.ply` 文件名仅为兼容后处理，即使来自 OBJ 也须在 manifest 明示。

重网格、简化、焊接、删除面等操作先于标签计算。每个分辨率版本独立保存 mesh 哈希、查询点和标签，禁止沿用旧面索引。高复杂度样本先按三角面数分桶，在模型侧用打包/mask；不任意截断前 N 个面来凑固定长度。

面向后续生成 mesh 的域偏移，可另增 `sdf_mc` 版本：从 GT solid 构造 SDF、提取 mesh，再查询固定 GT Voronoi 场。该步骤不是首版必要项。退化封孔、删除薄壁或改变连通性时，应隔离，不把“距离还能计算”当成“标签仍可形成合理分割”。

### 4.5 查询点、条件点与距离计算

主生成任务使用全部面中心，保存 `query_xyz (F,3)`、`query_normal (F,3)`、`query_triangle_id=arange(F)`、面积和邻接。法向从最终输入 mesh 计算。

可额外按面积采样 32,768 个条件点及法向，数量可配置，不限制完整 mesh 的面数；保存种子、三角形索引和重心坐标。条件采样只依赖输入几何，不用 GT face 数量、edge 或 UDF 决定布局。

复用 `prepare_implicit.py` 的核心查询方式：

```python
distance, _, _, _ = igl.signed_distance(
    query_xyz.astype(np.float64),
    voronoi_vertices.astype(np.float64),
    voronoi_triangles.astype(np.int32),
    sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED,
)
```

大样本分批处理；空间加速结构缓存需使用实际支持的 API，不假定这个接口自动跨调用缓存。Voronoi 分界面无需闭合，不因其非 watertight 就补洞或拒收。

先验证距离有限、非负，异常负值报错，不用裁剪掩盖；再保存 float32 raw/metric/target，索引用 int32/int64。不要直接沿用原脚本 float16 标签。

边界附近额外查询放在诊断/辅助监督字段，不改变主生成查询布局。GT 引导的查询位置可能泄漏边界；训练主任务使用推理可取得的位置，通过损失加权和分组评价关注边界精度。原脚本的三维高斯扰动查询不属于严格表面点。

### 4.6 独立的 face/edge 真值

距离标签只要求 STEP Voronoi 与输入 mesh 对齐，不必先知道 OBJ 每个三角面的 CAD face ID。但分割评价和边界诊断需要额外真值：

1. STEP 三角化路线在 face 遍历时记录 `triangle_cad_face_id`，并随索引变换保留映射。
2. OBJ 路线可计算到各有限、已裁剪 STEP face 的距离，或从带 face ID 的高精度参考三角化转移标签。保存最近距离、次近间隔、法向一致性和歧义标记，不能仅对无限延伸曲面求距离。
3. 跨越 CAD 边界的三角形不应由中心标签假装整面归属确定。保存 mixed/ambiguous mask，评价时明确排除或细采样规则；必要时细化输入后重新计算全套标签。
4. 普通边界只指不同 CAD faces 的公共边；seam、退化 edge 与拓扑重数另外保存。
5. `eval_seg.py` 需要 `<id>.ply` 表面点标签、`<id>_edge.ply`、`<id>_vertex.ply`、`<id>_adj.npz`（`face_edge/edge_vertex`）。其注释提到的 `prepare_seg_gt.py` 当前仓库未提供，需新增导出器，核对采样密度、ID 顺序、坐标与 seam 约定。

真值放入独立监督文件，禁止条件 loader 默认读取。只观测到一份 CAD 划分时，不宣称数据已覆盖同一几何的全部合法划分。

### 4.7 数据质量与 GT 场后处理对照

每个样本检查：

- mesh、查询点与标签数量一致，索引合法，重算中心一致。
- 所有数值有限，raw/metric/target 转换一致，float32 保存读回误差受控。
- 相邻/随机点对满足 $|g_i-g_j|\le\|q_i-q_j\|+\epsilon$。这是距离场必要自检，不是语义正确性的充分证明。
- 普通 GT 边界附近的低值谷、饱和比例、薄壁低值平台、小面与短边覆盖。有限面中心未出现恰好零值不代表失败。
- 代表性样本用另一种查询或小规模 brute-force 点到三角形计算交叉验证。
- 分别运行 GT labels → rebuilder 与 GT 表面场 → 分割 → rebuilder，区分参数化问题和场/阈值问题，不要求先训练回归模型。

后处理必须记录真实入口参数与行为：

- `hierarchical_segmentation` 函数默认值与 CLI 使用的全局值不同。当前 `process_item` 路径为 `threshold1=0.005`、`threshold2=0.01`、`filter_size=10`、`mode=hierarchical`、`is_merge_small=False`，不能误用函数签名的 `0.003/5`。
- `process_item` 只保留最大连通分量，并对零 cluster 回退为单 cluster。两者都可能掩盖失败。单 solid 也可能含多个壳体或内腔，需单独检查，不能静默删掉表面。
- 入口 `trimesh.load` 默认处理可能改变索引。验证读入前后面顺序与标签；必要时后续实现改为 `process=False` 加显式清理映射。长度相等不能证明一一对应。

分别保存 `field_valid`、`segmentation_status`、`brep_status`。GT 场的阈值分割失败不一定表示距离标签错误，保留诊断集合；若使用可重建子集训练，应报告筛选比例与偏差，不在测试集静默删除困难样本。

## 5. 产物目录与字段规范

实际派生数据根目录为 `/opt/data/private/yihengxu/Datasets/surface`，不改原始 ABC 目录：

```text
/opt/data/private/yihengxu/Datasets/surface/
├── config.json
├── manifest.jsonl
├── splits/{train,val,test}.txt
├── reports/inventory.json
├── reports/process_results.jsonl
├── reports/items/<id>.json
├── work/<id>/                 # Voronoi 中间产物与日志
├── samples/<id>/
│   ├── recon_sdf.ply
│   ├── geometry.npz
│   ├── surface_field.npz
│   ├── transform.json
│   └── quality.json
└── validation/                # 后续 GT 分割/B-rep 对照
```

清单、报告和两个 smoke 样本已经按该布局生成。`supervision.npz`、`gt_seg/` 和 B-rep 对照属于下一阶段，当前处理器没有伪造这些文件。不同 mesh variant 应使用不同输出根目录；每个根目录只放一种 variant，用八位 ID 避免旋转后缀解析冲突。

| 文件 | 字段 | 类型与用途 |
| --- | --- | --- |
| `geometry.npz` | `vertices`, `triangles` | `(V,3)` float32、`(F,3)` 整数，权威 mesh 索引 |
| 同上 | `face_adjacency`, `face_area` | `(E,2)` 整数、`(F,)` float32 |
| 同上 | `condition_points`, `condition_normals` | `(N,3)` float32，仅来自输入几何 |
| 同上 | `condition_triangle_id`, `condition_barycentric` | 采样溯源 |
| `surface_field.npz` | `query_xyz`, `query_normal`, `query_triangle_id` | `(F,3)`、`(F,3)`、`(F,)` |
| 同上 | `udf_raw`, `udf_metric`, `udf_target`, `tau` | 三个 `(F,)` float32 数组和标量 |
| `supervision.npz` | `cad_face_id`, `ambiguous_mask`, `boundary_mask` | 监督与评价专用，不作为条件 |
| 同上 | 可选边界查询、拓扑数据 | 不与主查询数组混用 |
| `transform.json` | 正/逆矩阵、尺度、平移、单位约定 | 双向恢复坐标 |
| `quality.json` | 统计、误差、阶段状态、耗时 | 验证与失败分析 |

manifest/config 还需记录 schema 版本、源文件/mesh/标签哈希、实际脚本哈希、C++ 与 Python OCC 版本、三角化与焊接参数、种子、Voronoi 参数和阶段状态。NPZ 使用普通数值数组，不引入必须 `allow_pickle=True` 的对象字段。

loader 读取几何与 `udf_target`，在线采样噪声与时间，无需预存 DDPM/FM 带噪状态。一次采样轨迹内不重采样位置，多候选只改变随机种子。可选 Laplacian、质量矩阵、特征分解缓存绑定 mesh 哈希和参数，不是计算标签的必要步骤。

## 6. 已实现处理器、运行方式与后续模块

[prepare_abc_surface.py](../scripts/prepare_abc_surface.py) 已实现：递归扫描与精确配对、稳定 split、STEP/OBJ 预检、C++ 子进程隔离、坐标变换、OBJ/STEP bbox 对齐、全部三角面中心距离查询、Open3D/libigl 后端、条件点采样、原子导出、缓存溯源和失败报告。它只处理 `pair_status=paired` 的样本，因此本阶段不会从 2,832 个缺 OBJ 的 STEP 自动三角化补入训练集。

先建立或刷新清单：

```bash
python scripts/prepare_abc_surface.py inventory \
  --abc-root /opt/data/private/yihengxu/Datasets/abc \
  --output-root /opt/data/private/yihengxu/Datasets/surface
```

正式处理默认使用项目 C++ Voronoi：

```bash
/miniconda/envs/HYCAD/bin/python scripts/prepare_abc_surface.py process \
  --output-root /opt/data/private/yihengxu/Datasets/surface \
  --voronoi-exe Voronoi/build/calculate_voronoi/calculate_voronoi \
  --limit 20
```

也可使用 `run` 在一次调用中刷新清单后处理。默认 `--resume` 只复用处理配置和 Voronoi 来源均一致的成功结果；设置变化时必须显式 `--overwrite`，旧结果会先重命名归档，不会直接覆盖。

仍待实现的是 CAD face/edge/vertex 监督导出器、`eval_seg.py` 所需 GT 格式、GT labels/B-rep 对照和训练 Dataset 类。下一步先用正式 C++ 后端处理 10–20 个分层 smoke 样本，再扩至 100–200 个，最后决定是否全量处理。数量是预算建议，不承诺筛选后规模与耗时。

当前默认路径下未发现编译好的 `Voronoi/build/calculate_voronoi/calculate_voronoi` 和 `checkpoints/parametrizer.ckpt`，但未检查机器上其他环境/路径。表面标签生成无需 parametrizer 权重，B-rep 对照需要。

为验证除 C++ 外的完整链路，处理器提供显式启用的 `--voronoi-backend scipy --allow-experimental-voronoi`。它从规范化 STEP 的逐 face 三角化中采样带 CAD face ID 的点，用 SciPy 三维 Voronoi 提取不同 face 样本站点之间的有限 ridge，再生成分界面网格。其采样与项目 C++/Geogram 实现不同，只用于 smoke test；输出的 `quality.json` 和 `voronoi_backend.json` 均标记 `experimental=true`，正式训练前必须用 C++ 后端重建，缓存不会跨后端复用。

2026-09-13 的真实数据 smoke 结果：

| ID | 结果 | 输入 OBJ 三角面 | 说明 |
| --- | --- | ---: | --- |
| `00000003` | 成功 | 41,238 | 逐面查询、Open3D 距离、PLY 回读和 NPZ 一致性通过 |
| `00009867` | 成功 | 46,634 | 同上 |
| `00000002`、`00000004` | 预检隔离 | — | STEP 含退化 edge，与当前 C++ 限制一致 |
| `00000005` | 预检隔离 | — | 含 10 个 solids，超出首版单 solid 范围 |

两个成功样本位于 `samples/<id>/`，其 `len(triangles) == len(query_xyz) == len(udf_raw) == len(udf_g)`，raw/metric/target 换算和非负有限性断言均通过。该结果证明处理器链路能运行，不构成 SciPy 标签与正式 C++ 标签等价或可用于训练的结论。复现实验命令为：

```bash
/miniconda/envs/HYCAD/bin/python scripts/prepare_abc_surface.py process \
  --output-root /opt/data/private/yihengxu/Datasets/surface \
  --ids 00000003,00009867 \
  --voronoi-backend scipy --allow-experimental-voronoi \
  --scipy-voronoi-points 2000 --scipy-min-face-points 8 \
  --condition-points 2048 --alignment-samples 512 \
  --distance-backend open3d
```

构建依据 [Voronoi/install.sh](../Voronoi/install.sh) 和 [README](../README.md)。安装脚本含 sudo/apt、网络下载和依赖构建，属于独立环境准备步骤，不能在数据扫描时自动执行。C++ 采样半径、边采样密度等当前写在源码中，记录源码哈希，不编造 CLI 参数。

构建完成后，也可直接测试 C++ 单样本接口：

```bash
Voronoi/build/calculate_voronoi/calculate_voronoi \
  /opt/data/private/yihengxu/Datasets/abc/step/0000/00009867/00009867_086229ea796640d29acd68ff_step_002.step \
  /opt/data/private/yihengxu/Datasets/surface/work_cpp_test/00009867/
```

实际批处理应由 `prepare_abc_surface.py` 调用该程序，以获得超时、日志、来源哈希、对齐检查和失败记录；不运行完整 `prepare_implicit.py` 制造无关体数据。

GT 场对照目录准备好后可复用：

```bash
python clustering.py /opt/data/private/yihengxu/Datasets/surface/validation/gt_field
python rebuild.py \
  --input /opt/data/private/yihengxu/Datasets/surface/validation/gt_field \
  --out /opt/data/private/yihengxu/Datasets/surface/validation/brep_gt_field \
  --rotations 3
python postprocess.py \
  --input /opt/data/private/yihengxu/Datasets/surface/validation/brep_gt_field --serial
```

全程保持规范化坐标，不在这些目录混入旧 `norm_params.npz`：分割入口发现它会反归一化，且其 scale 是逆变换的除数，与 C++ 的乘数不是同一约定。最终 STEP 若需恢复原坐标，在规范化评价完成后明确执行一次逆变换并验证 round-trip。

`eval_seg.py` 显式传本地 `--gt`、`--list`、`--pred`，其默认远程路径不适用。GT exporter 完成前不能声称已跑通这种评价。

## 7. 失败记录、缓存与资源控制

每个样本阶段：`paired → prechecked → voronoi_ready → aligned → mesh_frozen → field_ready → validated`。B-rep 状态另列，不混同标签生成成功。

建议失败码包括：`missing_obj`、`pair_conflict`、`step_read_failed`、`invalid_solid`、`unsupported_topology`、`mesh_invalid`、`placement_mismatch`、`normalization_mismatch`、`voronoi_crash`、`voronoi_timeout`、`voronoi_empty`、`distance_nonfinite`、`face_index_mismatch`、`gt_transfer_ambiguous`、`segmentation_failed`、`brep_invalid`。

缓存 key 包含源文件、变换、mesh、参数与版本哈希。验证后才原子写成功标记，文件存在不代表成功。新版本写新目录，不覆盖旧标签，不改写或删除源 STEP/OBJ。保留 Voronoi 与日志以支持重算；归档另做容量计划。

C++ 从单进程测量峰值内存、时长和失败类型，再设置受限并发与逐任务超时。距离计算主要是 CPU 几何工作，不默认要求 GPU；GPU 主要用于后续网络和参数化。记录各阶段 p50/p95 耗时和实际磁盘用量，由 smoke 结果估算全量成本，不一次启动全部重任务。

## 8. 验收标准

1. 每个发布样本可追溯至 STEP/OBJ、mesh variant、变换和算法版本。
2. split 在 CAD/重复组层面隔离，所有增强继承划分。
3. 每个目标与最终 mesh 面中心一一对应，保存/读取/分割入口均已验证索引一致。
4. 验证 Python/C++ 的 placement、bbox 与变换精度，不仅观察可视化大致重合。
5. raw/metric/target 转换可核验，标签以 float32 保存。
6. 条件字段只来自推理可得几何，GT face/edge/Voronoi 与条件隔离。
7. 代表性样本完成 GT labels 与 GT 场对照，报告薄壁、小面、短边、seam 等问题，而不只汇报平均场误差。
8. 交付 inventory、阶段成功率、失败原因和资源报告；未验证样本留在隔离清单。

最终产物为：**版本化清单 + 冻结输入 mesh + 逐面中心 Voronoi-UDF + 几何条件 + 独立评价真值 + 质量报告**。同一套几何标签同时服务 DDPM 与 Flow Matching。

## 参考

- [SurfaceField.md](SurfaceField.md)：场定义、直接生成方案与后处理接口。
- [PostProcess.md](PostProcess.md)：分割、参数化与 B-rep 失败分析。
- [ABC 官方项目与格式说明](https://deep-geometry.github.io/abc-dataset/)：STEP/OBJ/feat 用途与非全覆盖关系。
- [DualBrep README](../README.md)、[prepare_implicit.py](../prepare_implicit.py)、[C++ Voronoi 入口](../Voronoi/calculate_voronoi/calculate_voronoi.cpp)：本计划的实现依据。
