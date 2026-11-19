# Post1109 B-rep 重建模板

从闭合三角网格及连续 E/V 距离场恢复 STEP。流程依据 [Post1109](../docs/Post1109.md)。所有阈值使用归一化坐标。

| 阶段 | 默认设置 |
|---|---|
| 网格 | 几何粗加密目标 0.03，E 近区半径 0.02 与细目标 0.003 |
| 面与拓扑 | 单阶段面阈值 0.003，非核心区域传播，共顶点邻接，局部顶点域和短边推断 |
| 顶点 | V-UDF 中心差分投票与空间球壳距离中心拟合 |
| 曲线 | 直接距离拟合依次尝试直线、圆或圆弧、分段三次 Bézier，随后独立检查 E 场覆盖 |
| 曲面与 STEP | 筛选解析曲面，必要时拟合固定边界的 B 样条，再做周期规划、共享边装配和 STEP 读回 |

在含 Python 3.10+、pythonocc-core 7.7、libigl 2.6.3 的环境中，从仓库根目录运行：

```bash
python -m pip install -e ./veudf -e './band_brep_reconstruction[test]'
band-brep-reconstruct --step /path/to/source.step --output /path/to/new_run
```

运行时要求从本仓库安装的 `veudf` 提供 `edge_vertex_neighborhood_pools_v3` schema。

命令行从源 STEP 生成归一化网格，调用生产 veudf 的 `ReferenceEdgeOracle` 和 `ReferenceVertexOracle` 查询完整 CAD E/V。v3 训练标签在 0.05 截断，几何 condition 沿用现行版本和数组契约。Python 接口接受已归一化的网格和连续距离回调，便于接入已解码的距离值：

```python
from bandbrep import CallableUDFProvider, Config
from bandbrep.reconstruct import reconstruct
from bandbrep.surface_mesh import load_mesh

mesh = load_mesh("/data/new_mesh_pools_v3/assets/<id>/condition")
udf = CallableUDFProvider(
    evaluate_ev,  # [N, 3] -> metric [N, 2], columns dE and dV
    fingerprint=field_identity,
    presence=("present", "present"),
)
result = reconstruct(mesh, udf, Config(), output_dir)
```

E/V 是否存在由调用方显式给出。接入有截断的预测器时，通过 `FieldEvaluation` 传递 valid 和 saturated 掩码，必需通道的无效或饱和值会被拒绝。梯度统一通过空间采样中心差分计算，步长 1e-5。默认查询批大小 65536，每例预算 10800 秒。通过 [default.json](configs/default.json) 查看参数，使用 `--config` 覆盖数值或 `--stop-after` 查看中间阶段。使用 `--resume` 续跑时，输入、数值配置、代码及已登记产物必须匹配缓存身份。

结果写入 `reconstruction.step`（验收后）、`run.json`、`quality.json` 和各阶段产物。新实验从源 STEP 或 v3 几何 condition 重新执行网格加密到 STEP 验收。评分函数见[独立评分](benchmarks/README.md)，旧批次成绩见[历史实验记录](../docs/Experiment.md)。
