# 文档导航

更新：2026-11-02。当前主目录保留 11 份文档，按背景、数据、重建、生成设计和实验组织。安装与命令行维护在各模块 README，算法与字段规范维护在技术文档，成绩与样本口径维护在实验文档。

| 文档 | 内容与状态 |
|---|---|
| [DualBrep UDF 与拓扑](DualBrepUDFAndTopology.md) | 原 DualBrep 的 GVD-UDF、网络、两阶段分割及学习式 CAD 重建；与三场路线分开阅读 |
| [ABC 表面双 UDF 预处理](ABCEdgeVertexUDFPreprocessing.md) | 当前分支 `veudf` 的表面 E/V、三点池、批量距离与 NPY 字段；生产旧格式单独保留 |
| [隔离带算法与实现](BandLabelTopologyRecovery.md) | 当前 `bandbrep` 的多标签关联、局部顶点域、共享几何、曲面、CAD、配置和缓存；已合并原重建代码详解 |
| [直接边图提取备选方案](EdgeVertexDistanceFields.md) | 分支追踪／网格局部提取的研究设计；完整方案未作为主库模式实现 |
| [Hunyuan／NaTex 场 VAE 设计](HunyuanNaTexEdgeVertexVAEAdaptation.md) | 2026-10-14 的外部模型改造设计快照，涵盖场编码、联合 E/V latent、Flow 与验收；不代表本次已实施训练 |
| [实验主文档](ThreeFieldReconstructionExperiments.md) | 历史验证、消融、F30 条件场对照、面分割失败分析和待验证改进 |
| [网格加密方法与报告](MeshDensificationUDFExperimentReport.md) | 10-28—10-29 的 20 例×7组结果、方法、预注册预算和选型门槛；已合并原计划 |
| [原生网格 L5 配对实验](NativeMeshL5Experiment.md) | 同源 20 dev 已完成：四轮 L5 全部未达标；与旧 M0/L5 配对，解释原生网格起始尺度差异 |
| [网格粗细加密实施与验收](CoarseFineMeshRefinementPlan.md) | 已实现：纯几何粗加密至 0.02、近场细化至 0.005、超长边剪枝及字段复用；完整新批次已完成，见粗目标对照 |
| [两级加密粗目标对照](CoarseFineComparison20261102.md) | 40/40已完成；两组图+STEP均11/20，粗0.03耗时中位数下降28.2%，含分阶段耗时 |

使用入口：[重建模块 README](../band_brep_reconstruction/README.md)、[生产数据模块 README](../veudf/README.md)。

`surface-udf` 分支直接开发本仓库 `veudf/`，schema 为 `edge_vertex_surface_udf_v1`；外部生产目录保持原实现，历史数据使用各自冻结版本。新数据只有表面查询和 E/V 两通道，模型读取端需配套迁移。

建议先读[当前重建阶段](BandLabelTopologyRecovery.md#2-阶段代码与产物)与[当前数据字段](ABCEdgeVertexUDFPreprocessing.md#6-当前文件与字段)，再按需查[F30 结果](ThreeFieldReconstructionExperiments.md#f30-results)、[网格加密结果](MeshDensificationUDFExperimentReport.md)、[面分割分析](ThreeFieldReconstructionExperiments.md#segmentation-failures)和[后续实验](ThreeFieldReconstructionExperiments.md#next-experiments)。原 GVD-UDF、真实有限边/顶点距离、直接 CAD SDF 和当前 mesh SDF 的语义不能互换。

2026-10-31 整理将原重建详解合并到隔离带文档，原面分割分析合并到实验主文档，原加密计划合并到加密报告。合并前 11 份 docs 文档及两个模块 README 的完整原文保存在[归档](archive/20261031_document_sources.json.gz)，[清单](archive/20261031_document_sources.manifest.json)记录路径、字节数和 SHA-256。归档中的日期、计划和旧接口保留历史语义，不作为当前使用指南。

读取归档（在仓库根目录运行）：

```python
import gzip
import json
from pathlib import Path

path = Path("docs/archive/20261031_document_sources.json.gz")
records = json.loads(gzip.decompress(path.read_bytes()))["files"]
original_plan = next(r["text"] for r in records
                     if r["path"] == "docs/MeshDensificationUDFExperimentPlan.md")
```

10-19 及更早的原始结果、源码快照和文档归档入口仍见[实验证据](ThreeFieldReconstructionExperiments.md#artifacts)。历史成绩保留原批次版本、配置及分母，不能据文档更新视为新代码已经重跑通过。
