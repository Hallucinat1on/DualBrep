# 重建的独立几何评分

新实验使用同步后的生产 `veudf` v3。从源 STEP 运行 [Post1109 重建入口](../README.md)，或用新版资产的几何 condition 加连续 E/V 回调重建。每个样本写入独立目录。

`experiment_scoring.py` 保留三类离线评分。将本目录加入 Python 模块搜索路径后，可以在实验代码中调用：

| 函数 | 输入与指标 |
|---|---|
| `score(topology, bands, source)` | 工作网格的面、边、顶点标签与源 CAD 实例对照 |
| `geometry_score(vertices, curves, source)` | 顶点及曲线相对于源 CAD 支撑的距离 |
| `fidelity_score(solid, source)` | 双向面积采样表面距离及体积相对误差 |

`offline_source(targets, raw, oracle, normalization)` 从源 STEP 的 CAD 目标、带面 ID 的三角化、焊接网格的 SDF oracle 和归一化元数据组装参考。真值只用于评分。旧批次的结果与实验口径见[归档实验记录](../../docs/Experiment.md)。
