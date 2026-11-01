# ABC 三场数据处理：生产规范与本仓库同步

更新：2026-11-01。后续数据处理以 `/opt/data/private/yihengxu/Hunyuan3D/veudf` 为唯一生产来源，本仓库使用同步的 [veudf/](../veudf/README.md)。旧 `edge_vertex_preprocessing/evudf` 代码、配置和 CLI 已移除，目录只保留迁移说明及既有实验产物。

源码、测试、配置和打包文件按 [SOURCE.json](../veudf/SOURCE.json) 记录 SHA-256。同步命令为 `python scripts/sync_veudf.py --apply`，只读检查为 `--check`；新功能在生产目录开发后再同步。安装与命令行示例维护在[模块 README](../veudf/README.md)。

本文描述当前 `edge_vertex_mesh_pools_v1`，不将历史 `edge_vertex_udf_v1` 或 F30 的体素／MC 条件输入当作当前生产格式。历史实验保留当时版本和分母，见[实验主文档](ThreeFieldReconstructionExperiments.md)。

## 1. 三场定义与几何来源

每个查询位置保存 `[s_mesh,dE,dV]`：

- `s_mesh`：到最终闭合有向 CAD 三角网格的最近距离，绕数判断内外，内负外正。
- `dE`：到所有保留的有限 CAD 边的欧氏距离，包含端点。
- `dV`：到必要 CAD 顶点集合的最近距离。

STEP 用 OCC 几何包围盒居中并等比例缩放，默认最长边 1.8，位于 `[-0.9,0.9]^3`；查询域为 `[-1,1]^3`。曲线、顶点、网格及标签共用一次变换，距离单位是归一化长度。

条件网格从 CAD 三角化得到，在固定精度焊接重复顶点，不经体素 SDF 或 MC。检查闭合、方向、正体积及退化三角形；不能静默修面后继续沿用原 provenance。`MeshSDFOracle` 为每个物体复用 libigl AABB 与 winding BVH，底层查询以最多 512 点分批，避免额外线程膨胀。[网格查询实现](../veudf/src/veudf/geometry/mesh.py)

E/V 仍对应 CAD，网格表面不一定精确经过 CAD 边，因此不能无误差预算地强制 `abs(s_mesh)≤dE`。审计用 CAD SDF 检查 CAD 包含关系，并独立记录 mesh/CAD 差异；`dE≤dV` 在非空有效目标上仍检查。参考审计和有限采样不是连续全域误差或自交证明。

## 2. 目标筛选与默认数据政策

`preprocess/cad.py` 保留旧实现的统一归一化和目标语义：排除合法 seam，保留两个不同面的非 seam 边，包括相切／共面边；必要顶点在去 seam 后构造。仅移除符合条件的完整圆／椭圆孤立闭环锚点，一般 B-spline 锚点和人工分段仍保守保留。

普通目标边要求两次合法使用、两个不同面；非流形、退化、开放、异常同面内部结构、多 solid 及过大源 CAD 容差明确失败。距离目标不是 sharp-only 边、GVD、边点云或测地距离。

当前 [Config](../veudf/src/veudf/preprocess/config.py) 与 [filtering.py](../veudf/src/veudf/preprocess/filtering.py) 还规定：

| 检查 | 默认政策 |
|---|---|
| 面数 | 2–100 |
| 每面唯一 source edge 数 | 不超过 50，累积全部 wire，包含 seam |
| 所有 source edge 的归一化弧长 | 至少 0.001，包含 seam；不合格拒绝整模型 |
| 几何容差 | `geometry_tolerance=2e-5` |
| 标签容差 | `label_tolerance=1e-4` |
| 三角化参数 | `mesh_deflection=0.0002`、`mesh_angle=0.15` |

不按长度删掉短边再继续沿用原面拓扑。无必要顶点的圆环是正常目标，空通道独立处理。特定批次使用自己的冻结配置；历史 F30 的最低 10 面是旧数据政策，不能代替当前默认 2 面。

## 3. 查询与梯度

E-UDF 用 `cad_bvh` 对保守几何包围盒剪枝，候选再由 OCC 求有限 CAD 曲线距离；`cad_reference` 用全部保留边作抽查。V-UDF 使用精确 KD-tree。最近位置由 CAD 求解得到，不能用特征采样候选的近似距离替换标签。

CAD SDF 只作审计，生产第一通道固定 `s_mesh`，没有旧 `--sdf-backend cad/mesh/mesh_fast` 切换。[距离实现](../veudf/src/veudf/preprocess/distance.py)

当前 E/V 标签输出包含距离、最近 ID 和边歧义标记，不保存旧 `gradient_E/V`、最近点及梯度 mask 数组。后续梯度监督应在生产目录扩展并重新记录版本；模型推理时仍可对连续 decoder 的实际距离输出用 autograd 求预测梯度。网格条件 SDF 查询另有梯度和不可微有效性接口。

## 4. 普通点池、零集与独立评估

默认 `pool_count=1572864`，四池比例及固定预算：

| 池 | 点数 | 位置 |
|---|---:|---|
| `surface` / S | 393216 | 最终 CAD 网格面积采样 |
| `near_surface` / N | 550502 | 表面附近，实际 `abs(s_mesh)≤0.03` |
| `edge_surface` / E_S | 393216 | CAD 边邻域内的网格表面点 |
| `vertex_surface` / V_S | 235930 | 必要顶点邻域内的网格表面点 |

S 的固定前 `geometry_surface_pool_count=393216` 点为 S0，仅依赖网格、点数和几何随机种子，法向取对应三角面。缺 E/V 时配额回退到 S 的扩展部分，S0 不变；不再按每个 CAD face 分配固定最低面点数。

E_S/V_S 在候选三角形内部连续采样，不只挑网格顶点，也不是把 CAD 曲线零集直接当 encoder 表面点。每个可达实例最少 32 点，边剩余配额按弧长平方根分配，顶点均分；邻域半径为 `0.002/0.005/0.01/0.02/0.03`，初始权重 `20/30/30/20/0`，最后一层供不足回退。候选局部细分不改变最终条件网格，最终 E/V 和距离层由 CAD 查询决定。

N 的基点按 S/E_S/V_S=50/30/20 取样，80% 法向偏移、20% 各向同性扰动；每种来源、模式和层独立按接受点填足配额。越界或超过真实 SDF 带宽时重采，不 clamp。接受时的 SDF 复用为标签。

Z 单独保存每边 3 个弧长检查点和所有必要顶点，具有两个通道独立零集 mask；不因普通带宽筛选删除。默认 `evaluation/` 的普通池每组 4096 点，使用独立随机流；Z 按同一几何规则构造，可能与训练 Z 重合。不能把 evaluation 普通池说成训练池行子集。

默认 `volume_pool_count=0`，没有全域体积池。显式开启后仅为均匀空间点保存 SDF，并另建独立 volume evaluation，不计入四池预算、不计算 E/V。训练采样预算与方案 A/B 属于训练工程，不能反写成预处理固定配置。

## 5. 数值、空集与标签身份

所有非空距离不截断，查询标签在最终存储的 float32 坐标上求值，不硬把表面点 SDF 设为零。空 E/V 通道保存 float32 `2√3`，最近实例 ID 为 `-1`，由 `has_edges / has_vertices` 区分占位与真实距离。旧 `.02` 占位和 `normalize=True` 读取规则不适用。

source instance 是提出采样的目标，不一定等于该点最近的实例。source face、target edge、target vertex、condition triangle ID 分开保存；不能用最近 ID 替代完整关联，也不能把相同邻面的一对独立边合并。

## 6. 当前文件与字段

```text
root/
  config.json / manifest.jsonl / identity.json
  assets/<id>/
    condition/{vertices.npy,triangles.npy,meta.json}
    pools/{surface,near_surface,edge_surface,vertex_surface,zero}/
    evaluation/<对应点池>/
    audit/ / topology.json / meta.json
  results/ / logs/ / index.json / report.json / status.json
```

每池的数组按需存在：

| 字段 | 形状和用途 |
|---|---|
| `points.npy` | float32 `[N,3]` 查询坐标 |
| `fields.npy` | float32 `[N,3]`，列序 `[s_mesh,dE,dV]` |
| `nearest_ids.npy` | int32 `[N,2]`，最近 E/V 实例 |
| `source_instance.npy` | int32 `[N]`，采样来源实例 |
| `strata.npy` | uint8 `[N,1]`，N 为基点来源，特征池为真实距离层 |
| `flags.npy` | uint8 `[N]`，bit0 为最近边歧义 |
| `normals.npy`、`triangle_ids.npy` | 三个表面池的法向及 condition 三角形映射 |
| `perturbation_mode.npy`、`perturbation_layer.npy` | N 的扰动模式与层 |
| `zero_mask.npy` | Z 的 bool `[N,2]` 零集 mask |

`meta.json` 记录目标存在性、归一化、配置／依赖／代码身份、逐池统计、可达性与审计；`topology.json` 保留 CAD source 使用及目标图；condition 顶点 float64、三角形 int64，几何元数据足够重新生成 S0。`audit/` 的独立去重特征和 triangle→CAD face ID 不作为模型几何条件。

只读取发布的索引及按需 mmap 数组：

```python
from pathlib import Path
import numpy as np
from veudf.io.dataset import discover_assets

assets = discover_assets('/data/new_mesh_pools')
asset = Path(assets[0]['directory'])
points = np.load(asset / 'pools/surface/points.npy', mmap_mode='r', allow_pickle=False)
fields = np.load(asset / 'pools/surface/fields.npy', mmap_mode='r', allow_pickle=False)
```

`veudf.io.condition.read_geometry()` 只读几何 condition、不读 GT E/V。预测交换则使用 `veudf.io.predictions` 的独立 `edge_vertex_prediction_v2`，不是原始数据 schema。

## 7. 发布、去重和续跑

manifest 按 ABC 源文档分组、处理字节重复并固定 split。构建逐资产子进程隔离、记录失败和超时；独立几何／边／顶点特征及有界拓扑同构检查用于几何近重复审核，跨 split 冲突明确拒绝。几何去重是项目阈值下的检查，不作旋转／ICP 对齐或全局几何同一证明。

只有通过标签审计、去重与同源保护的资产发布到 `index.json`；资产目录数量不等于最终样本数。`--target` 按发布量继续取候选。首次 root 冻结配置、清单、代码和依赖身份；重复相同命令续跑，身份变化使用新 root，不能在旧 NPZ 根目录原地升级。

`prepare_remaining_run` 冻结增量候选与独立 `run/code/veudf`；`run_guarded_build` 保留 CPU／内存余量并在资源压力下软停止，可在原身份下续跑。同步仓库不更改已经冻结或运行中的生产任务。

## 8. 历史流程与重建边界

- 原 `edge_vertex_preprocessing/evudf` 的 `fields.npz / metadata.json / quality.json`、配置和命令已退役；数据不原地转换，历史输出保留。
- 根目录 `prepare_implicit.py` 的 Voronoi UDF 仅服务原 DualBrep 模型复现，不用于新 E/V 数据。
- 后处理现直接接收生产 condition 网格与连续预测 E/V，内部网格 SDF 复用生产 `MeshSDFOracle`，默认 L5 加密。`vendor/oracle`、旧 NPZ 插值与 MC 输入已删除，见[后处理入口](../band_brep_reconstruction/README.md)。
- `ProductionUDFProvider` 接入外部生产模型适配器；点池和离散预测交换文件用于训练/评估，不能替代任意新位置的连续 E/V 查询。
回归与使用入口见[模块 README](../veudf/README.md#入口与验证)。历史数据和实验范围继续见[实验主文档](ThreeFieldReconstructionExperiments.md#data-and-metrics)，旧设计与 EMAP 核查原文仍可从[文档归档](archive/20261031_document_sources.json.gz)追溯。
