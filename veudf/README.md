# 网格表面 E/V-UDF 数据处理

当前 `surface-udf` 分支直接维护本仓库 `veudf/`，版本 3.0.0。生产目录 `/opt/data/private/yihengxu/Hunyuan3D/veudf` 不随本次改造修改。来源记录见 [SOURCE.json](SOURCE.json)，字段与算法见[数据规范](../docs/ABCEdgeVertexUDFPreprocessing.md)。

schema 为 `edge_vertex_surface_udf_v1`，仅保存 `[dE,dV]`：在最终 condition 网格表面采样，计算到**全模型保留 CAD 边／必要顶点**的欧氏距离。不是到所属面的边界、沿表面的测地距离或到三角网格全部边／顶点的距离。所有 CAD 筛选、归一化、分组 split 和几何去重规则继续适用。

## 运行

需要 Python 3.10+、OCC/OCCT 7.7.0、libigl 2.6.3；依赖见 [environment.yml](environment.yml)。可复用已有环境执行本地源码，无需更改生产 editable 安装：

```bash
export PYTHONPATH="$PWD/veudf/src"
python -m veudf.cli.build_dataset manifest \
  --step-dir /data/abc/step/0000 --out /data/surface_manifest.jsonl
python -m veudf.cli.build_dataset build --root /data/surface_udf_v1 \
  --manifest /data/surface_manifest.jsonl --config veudf/configs/surface_pools.json \
  --workers 4 --threads 1 --timeout 7200
python -m pytest veudf/tests -q
```

使用新的输出 root。旧三通道点池、离面池、体积池和旧预测交换格式全部拒绝，不提供兼容读取。相同 root 的续跑仍核验清单、配置、源码和运行环境身份；已经冻结的旧任务保持其自身代码和环境。

## 采样与距离

默认三池合计 1,022,362 点：普通表面 393,216，近边表面 393,216，近顶点表面 235,930。S0 固定几何前缀仍为 393,216，特征配额变化不改变它。目标不存在或无法到达时，按原有规则回流普通表面池，存在性和覆盖审计显式记录。

每个点保存三角形 ID 和 float64 重心坐标；float32 点位的重建误差不超过 2e-7。普通池与特征池全部来自最终网格，没有法向噪声／各向同性噪声。CAD 零集不写入训练或 evaluation；曲边的离散网格点到 CAD 曲线可能具有正距离，不能硬写为零。训练与 evaluation 使用独立随机流。

E 距离引擎按三角形缓存保守候选，按候选边批量剪枝；直线段和有限圆弧走解析距离，复杂曲线使用 OCC。所属面和来源边只是搜索提示，全局包围盒筛选保证薄壁背面的更近边不会丢失。V 距离使用批量精确 KDTree。E/V 分别保存等距标记；解析曲线的多解按几何判断，避免把 OCC 返回的重复数值极值当作真正多解。

采样器用三角形包围盒索引筛选特征邻域；该邻域依据特征半径，不能用最近边归属代替。`cad_reference` 仅作独立的全边 OCC 标量审计。

## 输出与读取

```text
root/
  config.json / manifest.jsonl / identity.json
  assets/<id>/
    condition/{vertices.npy,triangles.npy,meta.json}
    pools/{surface,edge_surface,vertex_surface}/
    evaluation/{surface,edge_surface,vertex_surface}/
    audit/ / topology.json / meta.json
  results/ / logs/ / index.json / report.json / status.json
```

每池含 `points[N,3]`、`normals[N,3]`、`triangle_ids[N]`、`barycentric[N,2]`、`fields[N,2]`、`nearest_ids[N,2]`、`flags[N,2]`、`source_instance[N]`、`strata[N,1]`。重心坐标为顶点 1/2 的权重，顶点 0 权重为 `1-u-v`。

`flags` 每通道 bit 0 表示有效，bit 1 表示等距歧义。空通道距离为 float32 `2√3`、最近 ID 为 -1、有效位为 0。距离不截断、不做标签归一化。`source_instance` 只表示采样来源，不能当作最近实例。

```python
from veudf.io.dataset import discover_assets, load_surface_pool
assets = discover_assets('/data/surface_udf_v1')
bank = load_surface_pool(assets[0]['directory'], 'edge_surface')
# bank['fields'] 的通道顺序为 [dE,dV]，数组以 mmap 读取。
```

`condition` 只提供几何。CAD 实例 ID、拓扑、三角形到 CAD face 的映射是监督／审计数据，不能送入模型条件。网格距离工具保留给几何验证及独立后处理，表面 E/V 标签不再执行网格 SDF 查询。几何 VAE 如需要体积 SDF，应使用独立数据流程。

外部训练工程仍需迁移三通道、`near_surface`、`zero` 和损失 mask 的读取逻辑；本仓库本次不修改生产训练代码。

## 性能验证

[配对基准](benchmarks/benchmark_surface_distance.py) 在完全相同的表面点上比较距离，不把删除 35% 离面点算作算法加速：

```bash
python veudf/benchmarks/benchmark_surface_distance.py \
  --manifest /path/to/frozen_manifest.json --output /data/surface_distance_benchmark
```

清单为已有配对实验 JSON 数组，含 `id / step / source_sha256`；可用 `--ids`、`--count` 选择样本与查询数。`--baseline /path/to/frozen_distance.py` 可加入旧冻结引擎，仅用于基准，不作为生产后端。报告记录冷／热查询时间、输入和源码哈希、OCC 调用数以及独立参考误差；冷查询包含候选缓存建立。
