# 文档导航

更新日期：2026-10-18。按原理、数据、重建路线、生成模型和实验阅读。

| 文档 | 范围 |
|---|---|
| [DualBrep UDF与拓扑](DualBrepUDFAndTopology.md) | GVD-UDF、监督/网络/推理、两阶段面分割、学习式拓扑及seam |
| [ABC三场预处理](ABCEdgeVertexUDFPreprocessing.md) | 目标过滤、同位置三通道、采样/查询/存储、EMAP参考和独立数据审查 |
| [直接边图提取方案](EdgeVertexDistanceFields.md) | 距离场定位、分支追踪或网格局部提取、共享几何约束；独立备选路线 |
| [隔离带重建方案与实施](BandLabelTopologyRecovery.md) | 当前v0.3.0两阶段核心、多标签吸收、实例化、固定边界参数化和CAD |
| [Hunyuan/NaTex场VAE方案](HunyuanNaTexEdgeVertexVAEAdaptation.md) | 生成侧双分支、场编码、采样和损失；与确定性后处理分开 |
| [实验报告](ThreeFieldReconstructionExperiments.md) | 10-12—10-18 全部实验：早期批次与消融、失败归因、257 基线与阶段隔离、独立评估、场误差、局部顶点域/回转面/周期面 CAD 改进 |

直接边图提取与隔离带恢复保留为两个可独立阅读的计划，共用物理目标契约。原DualBrep的GVD-UDF与新到真实有限边/顶点的距离场不混用。实施状态与一般数学约束目标分开阅读。

本次将UDF/拓扑两份背景说明合一；旧实施设计的现行接口和几何约束并入隔离带文档，实测数据审查移到预处理§14。旧竞争代价/端口总流程仅为历史设计，不作为当前实现说明。合并前8份原文压缩保存在`band_brep_reconstruction/outputs/experiment_archive_20261015/document_merge_sources.json.gz`，迁移记录为同目录`document_merge.json`。

2026-10-18 将实验汇总、失败原因与改进建议、后续实验报告、改进实验报告四份合并为一份实验报告；原文压缩保存在 `band_brep_reconstruction/outputs/experiment_archive_20261018/report_merge_sources.json.gz`，哈希记录为同目录 `report_merge.json`。
