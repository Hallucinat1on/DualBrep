# ABC 三场数据处理：生产规范与本仓库同步

更新：2026-11-19。后续数据处理以 `/opt/data/private/yihengxu/Hunyuan3D/veudf` 为唯一生产来源，本仓库使用同步的 [veudf/](../veudf/README.md)。

源码、测试、配置、打包文件和生产工具按 [SOURCE.json](../veudf/SOURCE.json) 记录 SHA-256。同步命令为 `python sync_veudf.py --apply`，只读检查为 `--check`；新功能在生产目录开发后再同步。安装与命令行示例维护在[模块 README](../veudf/README.md)。

当前 schema 为 `edge_vertex_neighborhood_pools_v3`。历史 v1 和 F30 数据保留在原批次，见[实验记录](Experiment.md)。

## 1. 三场定义与几何来源

原始三场定义为 `[s_mesh,dE,dV]`。训练落盘时截断 E/V，独立评估保存原始距离。

- `s_mesh`：到最终闭合有向 CAD 三角网格的最近距离，绕数判断内外，内负外正。
- `dE`：到所有保留的有限 CAD 边的欧氏距离，包含端点。
- `dV`：到必要 CAD 顶点集合的最近距离。

STEP 先选择体积最大的有效正体积 solid，体积相同按源遍历顺序确定；记录 `source_solid_count`、`selected_solid_index` 和选择策略。仅有一个 solid 时仍检查其外部游离拓扑；多 solid 输入只处理被选中的实体。随后用 OCC 几何包围盒居中并等比例缩放，默认最长边 1.8，位于 `[-0.9,0.9]^3`；查询域为 `[-1,1]^3`。曲线、顶点、网格及标签共用一次变换，距离单位是归一化长度。

条件网格从 CAD 三角化得到，在固定精度焊接重复顶点，不经体素 SDF 或 MC。检查闭合、方向、正体积及退化三角形；不能静默修面后继续沿用原 provenance。`MeshSDFOracle` 为每个物体复用 libigl AABB 与 winding BVH，底层查询以最多 512 点分批，避免额外线程膨胀。[网格查询实现](../veudf/src/veudf/geometry/mesh.py)

E/V 仍对应 CAD，网格表面不一定精确经过 CAD 边，因此不能无误差预算地强制 `abs(s_mesh)≤dE`。审计用 CAD SDF 检查 CAD 包含关系，并独立记录 mesh/CAD 差异；`dE≤dV` 在非空有效目标上仍检查。参考审计和有限采样不是连续全域误差或自交证明。

## 2. 目标筛选与默认数据政策

`preprocess/cad.py` 保留旧实现的统一归一化和目标语义：排除合法 seam，保留两个不同面的非 seam 边，包括相切／共面边；必要顶点在去 seam 后构造。仅移除符合条件的完整圆／椭圆孤立闭环锚点，一般 B-spline 锚点和人工分段仍保守保留。

普通目标边要求两次合法使用、两个不同面；非流形、退化、开放、异常同面内部结构及过大源 CAD 容差明确失败。距离目标不是 sharp-only 边、GVD、边点云或测地距离。

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

训练由 `FeatureUDFOracle` 统一查询 E/V。每批查询点共用 KD-tree，边的采样覆盖球和 CAD 包围盒负责筛选候选，有限边距离仍由 OCC 求解。顶点候选使用精确欧氏距离。截断外的点可以跳过远距离求解。[训练距离实现](../veudf/src/veudf/preprocess/distance.py)

独立评估调用 `ReferenceEdgeOracle` 和 `ReferenceVertexOracle` 查询完整距离。审计还会对有限采样用穷举 CAD 边核验标签。生产第一通道始终取 `s_mesh`，CAD SDF 用于参考审计。[精确距离实现](../veudf/src/veudf/preprocess/reference.py)

产物保存标量距离和有效标签信息。边优先重建通过空间邻点查询连续 E/V，再用中心差分计算梯度。[后处理流程](ScanEdgeFirst.md)

## 4. 普通点池、零集与独立评估

| 普通池 | 默认点数 | 来源 |
|---|---:|---|
| `surface` | 393216 | 网格表面面积采样，固定几何前缀 S0 |
| `near_surface` | 275251 | 以 `surface` 为基点的空间扰动 |
| `edge_neighborhood` | 558367 | 393216 个边附近表面点及 165151 个空间点 |
| `vertex_neighborhood` | 346030 | 235930 个顶点附近表面点及 110100 个空间点 |

S0 由网格、点数和几何随机种子确定。特征池缺失时，表面和空间配额分别回到 `surface` 扩展段和 `near_surface`。特征表面点在候选三角形内采样。空间点按 80% 法向扰动和 20% 各向同性扰动生成，在 `[-1,1]^3` 内且 `|s_mesh|≤0.03` 才接受。

`edge_zero` 默认有 65536 个 CAD 边点，给每条边留基础配额，其余按弧长分配。样本取等弧长分段中点。`vertex_zero` 包含全部必要顶点。独立评估用自己的普通采样流和边零集预算 4096，顶点零集仍是必要顶点坐标。可选 volume 池只保存 SDF，默认关闭。

## 5. 数值与实例身份

训练保存 `[s_mesh,min(dE,0.05),min(dV,0.05)]`。E/V 到达截断上限时标为饱和，`nearest_ids` 置为 `-1`。目标缺失时，训练通道也填 0.05，由 `has_edges/has_vertices` 声明缺失，饱和标记为 false。独立评估使用完整 E/V，空通道填 `2√3`。

采样来源实例和最近实例分别保存。目标边、顶点及三角形索引属于不同的 ID 空间。

## 6. 当前文件与字段

```text
root/
  config.json / manifest.jsonl / identity.json
  assets/<id>/
    condition/{vertices.npy,triangles.npy,meta.json}
    pools/{surface,near_surface,edge_neighborhood,vertex_neighborhood,edge_zero,vertex_zero}/
    evaluation/<对应点池>/
    audit/ / topology.json / meta.json
  results/ / logs/ / index.json / report.json / status.json
```

每池的数组按需存在：

| 字段 | 形状和用途 |
|---|---|
| `points.npy` | float32 `[N,3]` 查询坐标 |
| `fields.npy` | float32 `[N,3]`，列序为 SDF、E、V |
| `saturated.npy` | bool `[N,2]`，E/V 截断饱和标记 |
| `sample_type.npy` | uint8 `[N]`，0 表面、1 空间、2 零集 |
| `nearest_ids.npy` | int32 `[N,2]`，最近 E/V 实例 |
| `source_instance.npy` | int32 `[N]`，采样来源实例 |
| `strata.npy` | uint8 `[N,1]` 距离分层 |
| `flags.npy` | uint8 `[N]`，bit0 为最近边歧义，bit1 为最近顶点歧义 |
| `normals.npy`、`triangle_ids.npy` | 表面采样的法向和 condition 三角形索引，混合池空间行分别填 0 和 -1 |
| `perturbation_mode.npy`、`perturbation_layer.npy` | 空间样本的扰动模式和层 |
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

## 8. 版本与重建兼容性

v3 构建使用新数据根目录。发布索引会校验 schema，续跑会校验配置、代码、清单和依赖身份。历史 v1 数据作为归档保留。几何 condition 的版本和数组契约保持一致。

重建的 STEP 入口调用生产版的精确 reference 距离。新实验从源 STEP 或 v3 几何 condition 起跑，使用当前代码重新计算前缀。训练点池用于离散监督，重建时需提供能在任意查询位置求值的连续 E/V。预测适配器应保留 E/V 存在性和饱和掩码。若必需的查询点落在饱和区，模板会拒绝该查询。[重建入口](../band_brep_reconstruction/README.md)
