# 可迁移的 STEP 边／顶点 UDF 预处理

将整个 `edge_vertex_preprocessing/` 目录复制到其他项目即可使用。脚本不导入 DualBrep、PyTorch、EMAP，也不需要 `voronoi.ply`。代码基于 DualBrep 的拓扑遍历、seam 检测、曲线采样和三角化思路独立实现，输出同一查询位置的三个标量：

```text
query_points[Q, 3]     = [x, y, z]
query_fields[Q, 3]     = [SDF, E-UDF, V-UDF]
```

通道顺序固定为 `sdf, dE, dV`，写入 `metadata.json`。SDF 可作为几何条件；两个 UDF 可作为结构生成目标，三通道保存不要求模型同时生成 SDF。

## 1. 安装与快速开始

代码已在 Python 3.9、pythonocc-core 7.7、NumPy 1.26、SciPy 1.12 环境验证；提供的环境配置选择 Python 3.10。Open CASCADE 通过 conda 安装：

```bash
cd edge_vertex_preprocessing
conda env create -f environment.yml
conda activate edge-vertex-udf

# 先用少量查询跑通整个流程。
python preprocess.py --step /data/example.step --out outputs/example --config configs/smoke.json

# 正式配置：约 20 万随机查询 + 边／顶点检查点。
python preprocess.py --step /data/example.step --out outputs/example_full --config configs/default.json
```

已有 pythonocc 环境可直接执行，无需安装本目录为 Python 包。需要项目内导入或命令行入口时，可以安装：

```bash
python -m pip install .
evudf-preprocess --help
```

`requirements.txt` 仅列 pip 可安装依赖；pythonocc-core 仍由 conda 提供。CLI 的 `--help` 与 manifest 索引不需要导入 OCC。

## 2. ABC 批量处理

从 ABC 官方下载并校验 STEP chunk，在脚本之外解压 `.7z`。本工具读取已经解压的 `.step`、`.stp`，也支持其中仍为 gzip 的 `.step.gz`、`.stp.gz`；gzip 在临时目录展开，不覆盖输入文件。

```bash
python make_manifest.py --step-dir /data/abc/step --out manifests/abc.jsonl --seed 0

python preprocess.py --manifest manifests/abc.jsonl --out-dir outputs/abc --config configs/smoke.json --limit 20 --workers 2

python preprocess.py --manifest manifests/abc.jsonl --out-dir outputs/abc_full --config configs/default.json --workers 4 --timeout 3600

# 输入、配置、运行版本和代码哈希一致，且输出校验通过时才续跑。
python preprocess.py --manifest manifests/abc.jsonl --out-dir outputs/abc_full --config configs/default.json --workers 4 --resume
```

也可直接使用 `--step-dir /data/abc/step --out-dir outputs/abc`。`--split train`、`--split val` 或 `--split test` 可筛选 manifest。

自动索引从 ABC 文件名中的 24 位文档 id 分组；普通文件按内容哈希分组。若两个文档含字节相同的 STEP 文件，会先合并整个文档分组，再划分 split，避免它们的其他 parts 泄漏到不同 split。重复文件保留 `duplicate_of` 记录，批量处理跳过重复项。gzip 文件按压缩后的源文件字节计算 SHA256；不同压缩结果未必被视为内容重复。

自动识别不了源 CAD 文档的非标准文件名时，可提供自定义 manifest，明确 `group_id`、`split`。STEP 内多个 solid 初版拒绝；如在外部拆分 parts，须保持同源分组。

Manifest 采用 JSONL：

```json
{"sample_id":"00000050_part0","step_path":"../raw/example.step","source_sha256":"源文件SHA256","group_id":"原CAD文档ID","split":"train"}
```

其中 `source_sha256` 可省略，提供时必须与实际文件一致。相对 `step_path` 相对于 manifest 所在目录解析；`sample_id` 只允许安全的单层目录名，不允许 `../` 或重复 id。

## 3. 已实现的几何目标规范

- 支持有效的单个闭合流形 solid，包括内部空腔。导入后检查 solid 数量、CAD 有效性和游离拓扑。
- 所有几何只执行一次平移与等比例缩放，约束在 `[-0.9,0.9]^3`；保存真实变换及其逆矩阵。
- 建立 source face、wire、coedge、edge、vertex 表，保留每次使用的方向及顺序。
- 排除 `BRep_Tool.IsClosed(edge, face)` 指示的合法 seam。额外面使用、异常使用次数及非 seam 同面内部边拒绝。
- 普通目标边必须有两次使用、两个不同 face；保留相切、共面及非 sharp 面间边。
- 距离查询完整有限 CAD edge，包含端点。短边不按长度删除，采样预算包含每条边的最低数量。
- 从保留边重新构造必要顶点，保留真实二度点及分段点。
- 单条完整圆／椭圆闭环，若唯一顶点仅作重复首尾锚点且无其他保留边连接，则移除该生成顶点。圆柱通常得到两个圆环、零必要顶点。
- 支持没有必要顶点的闭环、自环以及合法的空边／空顶点集合。

当前规范化版本为 `analytic_loop_anchor_v1`：一般 B-spline 的闭环锚点和 seam 引入的多 edge 二度人工分段采用保守保留，来源及状态写入拓扑文件。复杂人工分段合并还需扩展几何延续与面侧使用检查，不能仅根据度数或面标签对自动合并。

含退化边、开放壳、多 solid、非流形关联的样本返回失败原因。不同目标顶点在几何容差内重合时拒绝，避免距离场无法区分。有效 CAD 检查与这些局部检查不构成全局曲线自交／重合认证；需要处理异常 CAD 时应增加对应几何检查。

## 4. 距离后端与速度

`--edge-backend cad_reference` 对所有保留 edge 逐一执行 CAD 距离查询，用于小规模基准。

`--edge-backend cad_bvh` 是默认后端。为完整有限 edge 建立不依赖采样点的 OCC 几何包围盒，加入容差余量，再通过 BVH 距离下界剪枝。最终标签仍来自 `BRepExtrema_DistShapeShape`，没有用最近边点替代曲线距离。每个样本分层抽查全量 reference，并记录误差和调用次数。

顶点 UDF 使用精确模式的 `scipy.spatial.cKDTree`。空集合独立分支，不执行最近邻查询。

SDF 默认 `--sdf-backend cad`：查询到边界 face compound 的距离，用 CAD solid classifier 判断内外，内负外正。它无需 libigl，适合基准和小批量；20 万查询的复杂 CAD 可能较慢，应先测量样本耗时。

可选的 `mesh` 后端需要额外安装 libigl：

```bash
python -m pip install 'libigl==2.6.2'
python preprocess.py --step /data/example.step --out outputs/example_mesh --config configs/default.json --sdf-backend mesh
```

该后端用 libigl 计算到同一 CAD 三角化的无符号距离，再用 CAD classifier 判定符号，避免直接依赖拼接网格的 winding sign。保留三角形到 source face 的映射，采用绝对 meshing deflection；对一组分层查询以 CAD SDF 验证。质量报告中的 mesh 偏差属于抽样验证，mesher 参数本身不是全域误差证明。

当复杂曲面的 CAD classifier 也成为瓶颈时，可以选择 `--sdf-backend mesh_fast`。它使用 libigl 的 fast winding number 为完整 CAD 三角化计算有符号距离，不逐点调用 CAD classifier；要求正的网格有向体积，并仍以 CAD reference 对照分层查询的符号与距离。该后端只改变 SDF 条件的近似计算，E-UDF 与 V-UDF 后端及查询数量保持原配置。它依赖闭合、方向正确的 CAD 三角化；质量报告中的抽查不构成全域符号认证。默认后端仍为 `cad`。

距离大小由 libigl 返回的最近三角形位置重新计算，避免近似 winding 值对距离幅值的缩放；winding 仅用于选择符号。接近分类阈值的少数歧义点回退到 CAD classifier，零距离邻域显式处理。空腔及反向网格包含在回归测试中。

```bash
python preprocess.py --step /data/example.step --out outputs/example_fast --config configs/default.json --sdf-backend mesh_fast
```

## 5. 采样与精度

`query_weights` 按以下顺序设置：

```text
uniform, near_surface, on_surface, near_edge, near_vertex
```

先混合这些来源构成统一查询点集，再对每个位置计算所有三个通道。近顶点／近边不存在时，配额按固定规则转移到表面邻域。额外加入每条目标边的检查点及全部必要顶点，因此最终 `Q` 略大于 `query_count`。

表面点按面积分配，保证每个 face 最小数量。边点按弧长分配，使用 OCC 的等弧长采样，每条 edge 至少 `min_edge_samples` 个点，闭环避免重复首尾占用配额。最低覆盖超过请求预算时会提升实际点数。

查询点先量化为最终 float32 坐标，再提升至 float64 计算距离，使磁盘标签对应相同位置。CAD 目标及数值查询使用 float64，输出训练数组使用 float32。噪声越界采用重采样，避免 clamp 到查询域边界。

`geometry_tolerance` 为归一化长度单位，过大的源 CAD 容差会拒绝；调整该值时也需要调整精度预算。通道截断仅用于训练归一化，非空集合的磁盘距离保持未截断。

## 6. 输出与训练读取

每个通过检查的样本输出：

```text
<output>/
  fields.npz
  topology.json
  metadata.json
  quality.json
  surface_mesh.npz       # save_mesh=true 时保存
```

`fields.npz` 不含 pickle 对象，主要字段如下：

| 字段 | 形状 | 含义 |
|---|---|---|
| `query_points` | `[Q,3]` | 所有通道共享的查询坐标 |
| `query_fields` | `[Q,3]` | `[SDF,E-UDF,V-UDF]`，归一化长度单位 |
| `query_source` | `[Q]` | 来源编号，定义见 metadata |
| `surface_points` | `[Ns,6]` | xyz + 法向 |
| `surface_face_id` | `[Ns]` | source face id |
| `edge_points` | `[Ne,6]` | xyz + 切向 |
| `edge_point_id` | `[Ne]` | target edge id |
| `edge_tangent_valid` | `[Ne]` | 切向有效性 |
| `vertex_points` | `[Nv,3]` | 全部必要顶点 |
| `has_edges`, `has_vertices` | 标量 bool | 目标集合存在性 |
| `nearest_E_id`, `nearest_V_id` | `[Q]` | 最近 target 实例，空集合为 `-1` |
| `nearest_tie_E`, `nearest_tie_V` | `[Q]` | 并列／不唯一最近解标记 |
| `closest_E_points`, `closest_V_points` | `[Q,3]` | 最近位置，空集合占位为零且无效 |
| `gradient_E`, `gradient_V` | `[Q,3]` | 未截断距离的梯度 |
| `gradient_valid_E`, `gradient_valid_V` | `[Q]` | 非零且未检测出最近解歧义的梯度标记 |

source edge id、target edge id、source face id、target vertex id 分开记录。最近 edge id 不代表完整拓扑关系；`topology.json` 保存邻面、端点、源曲线区间及源使用记录。无顶点闭环的端点列表为空，self-loop 的两个端点 id 相同。

梯度 mask 屏蔽零集、内核返回的多解、不同 edge／vertex 并列最近，以及圆弧轴线上的连续多解。一般曲线的全部不可微位置未作符号认证，梯度辅助监督应配合 mask 使用；标量距离不受这一辅助标记限制。

空边／空顶点通道存对应 `clip_edge` / `clip_vertex` 作为有限占位，同时标记集合不存在。这些值不解释为真实有限距离；模型读入时生成全域饱和通道，屏蔽几何定位和梯度项。查询失败则拒绝样本，不能冒充空集合。

Python 读取示例（把本目录放到 `PYTHONPATH`，或安装为包）：

```python
from evudf.io import load_fields

arrays, metadata = load_fields("outputs/example", normalize=True)
xyz = arrays["query_points"]
sdf_condition = arrays["query_fields"][:, :1]
structure_target = arrays["query_fields"][:, 1:]  # [E-UDF, V-UDF]
has_vertices = bool(arrays["has_vertices"])
```

`normalize=False` 返回归一化长度单位的原始标签；`normalize=True` 根据三个 `clip_values` 截断并缩放，SDF 在 `[-1,1]`，UDF 在 `[0,1]`。梯度随距离缩放，饱和区的梯度 mask 被关闭。训练时仍需按任务处理存在性预测；推理不能使用测试 CAD 的真值标记。

## 7. 失败、续跑及验证

每个样本由独立进程处理，批量最多同时运行 `--workers` 个进程；超时后终止对应 worker。`run.jsonl` 追加每次结果／失败详情，任一失败使 CLI 返回非零退出码，其余样本继续处理。

通过检查后先写隐藏临时目录，读回并验证哈希和数组契约，再提交至最终输出。中断的临时目录不被当作完成样本。`--resume` 检查输入内容、配置、代码、依赖版本及 manifest 身份，并核对文件哈希。参数变化应写入新输出目录，或使用 `--overwrite`；后者只替换带有本工具 schema 标记的样本目录，并在新结果验收通过后替换。

检查覆盖零集、非负／有限值、最近点回代、距离包含关系、随机对 Lipschitz 检查、CAD reference 对照，以及存储读回。网格误差和曲线重合的检查范围写入 `quality.json`；未运行全 ABC 数据集验证。

运行回归测试：

```bash
# 在本目录执行。
PYTHONPATH=. python -m unittest discover -s tests -v
```

测试涵盖长方体解析距离、圆柱 seam 与圆环、空顶点／空边集、共面拓扑边、有限圆弧端点、BVH 与全量查询对照、梯度及歧义、退化结构拒绝、短边覆盖、变换与 gzip、三通道和哈希、复制目录后的并行 CLI 与续跑。

实际验证环境、ABC 示例结果及检查范围见 [VALIDATION.md](VALIDATION.md)。

## 8. E-UDF / V-UDF 三维可视化

`visualize_fields.py` 读取已有 `fields.npz` 和 `metadata.json`，分别导出带 RGB 颜色的二进制 PLY 点云。只需要 NumPy，不需要 OCC、GPU 或重新计算距离。安装本目录后也可使用 `evudf-visualize` 命令。

在本目录执行，批量导出已有四份成功结果：

```bash
python visualize_fields.py --input outputs/abc10_default_20261012 --output visualizations/abc10_all --references
```

每份样本对应一个输出子目录：

```text
e_udf.ply                  # 查询位置，按到目标边的距离着色
v_udf.ply                  # 查询位置，按到必要顶点的距离着色；空顶点集不导出
reference_edges.ply        # --references 时保存，白色 CAD 目标边采样
reference_vertices.ply     # --references 时保存，白色必要顶点；非空时保存
visualization.json         # 色标、单位、筛选参数、点数及空集合状态
```

可用 CloudCompare、MeshLab 等查看 PLY 的 RGB 点云；PLY 同时保存未截断的 `distance` 标量和 `query_source`，支持自定义标量显示的查看器可进一步按距离查看。参考点云用于叠加比较 CAD 目标，不表示重建结果。输出保持与预处理相同的归一化坐标，可与保存的表面几何叠加。

颜色由小距离到大距离依次为 **深蓝 → 青 → 绿 → 黄 → 红**。默认线性色标上限来自各通道的 `clip_values`（本批数据为 `0.02`）；超过上限的点显示红色，但 PLY 中仍保存其真实距离。通过 `--color-max` 改变色标上限。`visualization.json` 记录每个色标节点的距离及 RGB。

全域点云可能遮挡内部细节，以下两种筛选适合观察几何结构。第一种仅显示带 `on_surface` 来源标记的查询点，第二种显示表面附近且靠近目标结构的查询点：

```bash
python visualize_fields.py --input outputs/abc10_default_20261012 --output visualizations/abc10_surface --selection surface --references
python visualize_fields.py --input outputs/abc10_default_20261012 --output visualizations/abc10_near --surface-band 0.002 --max-distance 0.01 --color-max 0.01 --references
```

所有筛选阈值使用归一化长度。`--surface-band` 使用 `abs(SDF)` 同时筛选两类点云；`--max-distance` 分别依据 E-UDF 和 V-UDF 筛选，两个文件的点数因而可以不同。正阈值得到的是边／顶点附近的候选点带，不是精确提取的曲线／顶点。

若希望分辨接近零值的距离变化，可使用对数色标：

```bash
python visualize_fields.py --input outputs/abc10_default_20261012 --output visualizations/abc10_log --surface-band 0.002 --scale log --log-offset 0.00001 --references
```

单份样本目录或其 `fields.npz` 路径也可以作为 `--input`。默认每个 UDF 最多导出 200,000 个点，以固定 `--seed` 随机抽样；`--max-points 0` 导出全部筛选点。没有通过筛选的通道记录 `no_selected_points`，空目标集合记录 `absent` 并跳过占位值可视化。

脚本默认输出到输入目录的 `visualizations/`；已有同名导出文件时要求使用新的 `--output`，避免将不同筛选配置混在同一目录。

## 9. 文件组织与迁移

```text
preprocess.py / make_manifest.py    命令行入口
visualize_fields.py                两类 UDF 的彩色 PLY 导出入口
evudf/config.py                    配置与参数验证
evudf/cad.py                       STEP、拓扑、归一化、三角化
evudf/distance.py                  CAD/BVH 边距离、顶点距离、SDF
evudf/sampling.py                  实例覆盖、混合查询、空集配额
evudf/pipeline.py                  三通道输出、质量检查、缓存与提交
evudf/manifest.py                  ABC 索引、去重分组、split
evudf/io.py                        训练读取，仅需要 NumPy
evudf/visualize.py                 距离着色、筛选和三维点云导出
configs/                          完整与小规模配置
tests/                            CAD 回归及迁移测试
```

复制本目录即可保留所有运行模块、配置和测试；输入 STEP、输出数据及环境需在新项目配置。无需迁移原仓库的 GVD 构造、模型权重或训练代码。
