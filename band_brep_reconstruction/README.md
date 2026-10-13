# 独立三场 B-rep 重建模块

根据 `ThreeFieldBRepReconstructionImplementation.md` 实现隔离带标签方案。**复制本目录即可迁移**；运行时不导入上级项目的 `rebuild_model.py`、`clustering.py`、`brep_post` 或 `edge_vertex_preprocessing`。复用源码均已复制到 `bandbrep/vendor/`，来源见 [PROVENANCE.md](PROVENANCE.md)。

流程为：metric 三场 → SDF Marching Cubes → 近带共形细分 → 双隔离带／面核心 → 全局部来源 Dijkstra → 共同最小界面 → 顶点／边实例 → UDF 定位 → 共享端点曲线 → 固定边界曲面 → 共享 OCC 实体／seam → STEP 读回验收。

本模块实际运行全部阶段，并提供明确的中间状态与失败报告；不保证任意 ABC 模型或任意生成场都能成功。`solid_validated` 表示通过配置误差预算下的 CAD 与读回检查，不是连续全域正确性证明。

## 安装和迁移

Python 3.9+。基础依赖安装：

```bash
cd band_brep_reconstruction
python -m pip install -e .
```

完整 STEP 重建需要 `pythonocc-core`，本项目实际测试版本为 7.7。可在 conda 环境中安装：

```bash
conda install -c conda-forge pythonocc-core=7.7
python -m pip install -r requirements.txt
```

可选依赖：`torch` 用于三场 MLP 和 DualBrep 面网络；`libigl` 用于 CAD oracle 的批量 `mesh_fast` SDF。CPU CAD SDF 按点查询较慢。基础拓扑与曲线模块不要求 OCC；CAD oracle／装配阶段才导入 OCC。

迁移时复制本目录的源码、配置、测试和文档即可。`outputs/` 是实验缓存，可以按需要另行复制；其中的数据路径不会成为代码导入依赖。

## 输入模式

| 模式 | 输入 | 含义 |
|---|---|---|
| `cad` | STEP，可选已有样本目录校验归一化 | 明确的真值场实验；CAD 只在 provider 内查询三场，不把实例 id 或原始曲面交给算法 |
| `sampled` | `metadata.json` + `fields.npz` | 局部加权仿射插值实验；先按唯一坐标划分留出集，报告近似误差。它不等于原始连续场 |
| `neural` | `fit_fields.py` 生成的 checkpoint | 三通道监督 MLP，可连续查询及自动微分；必须检查独立场拟合误差 |
| Python API | `CallableFieldProvider` | 接入未来生成模型；明确存在性、metric 尺度、误差和唯一输入指纹 |

通道顺序固定为 **`[sdf, dE, dV]`**，SDF 内负外正。生成阶段排除 seam，保留不同 face 之间的相切／共面分界。完整闭环无需生成顶点；必要二度点、高价点、self-loop 和同一面标签对下的多条独立边均有独立数据表示。

正式散点入口只读取 `query_points`、`query_fields`、存在性和尺度。**不会读取** `surface_face_id`、`edge_point_id`、`vertex_points`、最近点或 `topology.json` 来恢复拓扑。离线评分在 `evaluation.py` 中隔离。

## 运行现有样本 07

从本目录执行，路径按迁移后的实际位置替换：

```bash
python reconstruct.py \
  --provider cad \
  --sample-dir ../edge_vertex_preprocessing/outputs/abc10_default_20261012/00000007_b33a147f86da49879455d286_step_000 \
  --sdf-backend mesh_fast \
  --output outputs/abc07
```

`--sample-dir` 在 oracle 模式中用于获取源路径与检查坐标变换；也可以只传 `--step /path/input.step`。`mesh_fast` 的几何误差由 `field_error` 预算声明，它不是精确 CAD SDF。

只运行面分割或逻辑图：

```bash
python reconstruct.py --provider cad --step /path/input.step \
  --sdf-backend mesh_fast --output outputs/example --stop-after seeds
```

继续同一输入／配置／代码的缓存，允许把 `stop_after` 从 `seeds` 改为 `solid`：

```bash
python reconstruct.py --provider cad --step /path/input.step \
  --sdf-backend mesh_fast --output outputs/example --stop-after solid --resume
```

缓存验证源指纹、配置、包内源码和所有阶段文件哈希。改变场、阈值、误差预算或代码时使用新的输出目录；不会静默复用旧拓扑。

## 用已有散点拟合可查询场

直接插值实验：

```bash
python reconstruct.py --provider sampled --sample-dir /path/sample \
  --output outputs/interpolation --stop-after seeds
```

MLP 训练及重建：

```bash
python fit_fields.py --sample-dir /path/sample --output outputs/field.pt \
  --steps 5000 --device cuda
python reconstruct.py --provider neural --field-model outputs/field.pt \
  --output outputs/neural --config configs/default.json --device cuda
```

训练会生成 `.validation.json`，按唯一坐标划分训练／验证，避免重复表面点泄漏。场拟合误差需单独计入预算；示例训练步数不构成精度保证。重建时模型固定，几何拟合不会改变 SDF 或两个 UDF。

生成模型接入：

```python
import numpy as np
from bandbrep import Config, CallableFieldProvider
from bandbrep.reconstruct import reconstruct

provider = CallableFieldProvider(
    lambda xyz: model_query(xyz),    # [N,3], metric [sdf,dE,dV]
    presence=("present", "absent"), # 模型预测或已核实的结构存在性
    error=2e-4,
    fingerprint="my-model-checkpoint-sha256",
)
result = reconstruct(provider, Config(), "reconstruction/example")
```

`unknown` 不会被当作空集；当前入口要求调用者提供已核实／预测的存在性，否则报告 `presence_unknown`。若返回训练归一化距离，设置 `normalized=True, clip_values=[...]`；阈值必须低于相应截断平台。饱和区域无法恢复真实远场距离。

## 可选 DualBrep 面初始化

提供 checkpoint 与模型构造配置：

```bash
python reconstruct.py --provider neural --field-model outputs/field.pt \
  --face-checkpoint /path/parametrizer.ckpt --face-config /path/model_config.json \
  --device cuda --output outputs/with_face_model
```

网络代码是 `vendor/dualbrep_model.py` 的本地副本。适配器只调用 `encode()`、`decode_face()`，不会调用边预测分支覆盖拓扑。解析支撑面能通过验收时优先采用解析面；一般曲面可用网络网格初始化 UV，再求固定边界样条。没有 checkpoint 时使用解析面或 PCA UV 初始化。

## 配置与算法细节

配置为 JSON，键与 `bandbrep/config.py::Config` 对应，未知键直接报错。

- `resolution/max_resolution`：SDF 初始／最大格点分辨率。表面误差超预算或网格无效时加密，不只保留最大分量。
- `tau_edge/tau_vertex`：独立的隔离带阈值，默认 `0.025/0.04`；它们不是最终几何精度。小特征需要更小阈值及更细网格。
- `max_band_edge`：近带最大边长，要求不大于 `tau_edge/2`。使用 UDF 距离与单元覆盖半径发现潜在内部穿越，共用中点细分，避免 T-junction。
- 核心采用局部线性带切分的连通逻辑：混合三角形可以保留带外部分，仅沿共享边的带外区间连接。没有核心大小过滤。
- `stability_factors`：相邻阈值的核心重叠匹配；`verify_band_stability` 默认开启，进一步将近带边长减半，检查两级网格之间是否出现合并／分裂。
- 到达代价保留全部局部来源；界面需要两来源共同最小，第三来源可以阻断候选段。
- 顶点先用梯度投票区分候选，再用独立三维距离观测拟合共享坐标。曲线固定端点；分段三次曲线施加参数缩放后的 C1 约束，内部 knot 不增加逻辑顶点。
- 面优先拟合平面／圆柱／球面。一般样条采用秩揭示零空间法满足边界采样等式，并交替更新 UV。四边 Bézier 提供整条边界控制点锁定工具；不只锁四角。
- 闭环圆边在周期 CAD 装配时统一规划参数锚点。圆柱环带、球面环带和球冠使用显式 seam；后处理锚点、pole 退化边和 seam 不计入逻辑生成图。
- 共享边一次创建，全局复用；`SameParameter` 后复核。自动修复若替换原有逻辑边或超出容差，直接拒绝。
- `max_triangles/max_refinements/max_segments/max_spline_controls`：明确资源上限；达到上限保留失败阶段，不伪装成功。

## 输出与验收

每次运行保存 `run.json`、`quality.json`，以及 SDF 网格、隔离带、核心、到达代价、界面、逻辑图、精化点、曲线参数、曲面参数和逻辑到 CAD 映射。`queries/` 保存实际查询；仅查询 SDF 的记录中未查询的 UDF 为 NaN，不能当作训练三通道标签。`save_queries=false` 可关闭。

状态包括 `face_seeds_ready`、`logical_graph_ready`、`curves_ready`、`surfaces_ready`、`solid_validated` 和 `failed`。失败报告包含阶段、错误代码与细节；只在 STEP 读回通过后发布 `reconstruction.step`。CAD 输出在归一化坐标中，原始尺度变换保存在输入 metadata；需要世界尺度时对整个实体统一变换。

验收检查核心稳定性、顶点与边场残差、独立边场覆盖探针、固定端点、整条边的加密边—面残差、内部 SDF、局部雅可比、共享边两侧使用、CAD 容差、正体积、三角化、STEP 读回和读回曲线几何。体积读回预算同时参考表面积与边界误差，不能用几何容差已达 `1e-3` 的结果要求无条件 `1e-7` 体积一致。

## 当前边界

- 任意有限格点和散点都不能保证找出未采到的薄结构。报告明确标记没有严格连续域／全局拓扑保证。
- 核心／标签界面使用经过细化的离散线性近似；不是精确符号场裁剪器。阈值与加密检查仍不足时返回失败。
- 多个顶点或短边被带吞掉、高价扇区无法唯一关联、复杂周期域未能补缝时会报告对应失败。当前没有自动的任意局部重新播种和短边搜索调度器；可调整配置重跑，不能声称复杂 ABC 已全面支持。
- 样条边界约束可能与指定基函数不相容；增加自由度后仍冲突会报告 `surface_boundary_incompatible`。当前未实现任意 rational 多边裁剪域的连续精确边界证明。
- 已实现局部雅可比和 OCC 有效性检查；**尚无独立的全局曲面自交／折叠认证**。不是通用 CAD 修复器。
- 原仓库面模型已经复制且接口可用，但未提供／虚构训练权重；实际预训练 checkpoint 的推理需使用者提供相容配置。

## 测试

```bash
python -m unittest discover -s tests -v
python demo.py --shape cylinder --output outputs/demo_cylinder
python demo.py --shape box --output outputs/demo_box
python demo.py --shape sphere_bore --output outputs/demo_bore
```

测试包含三来源竞争、同一 face 对两闭环、self-loop 双端口、距离中心定位、硬端点／整边约束、冲突拒绝、空结构、带细分、隐藏真值隔离、缓存和失败报告。完整集成测试覆盖立方体、圆柱、球体通孔、球内腔、平滑球面分界，实际运行 STEP 导出读回。详见 [VALIDATION.md](VALIDATION.md)。
