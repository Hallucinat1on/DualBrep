# 两级加密：粗目标 0.02 / 0.03 配对实验

2026-11-02 启动。状态及成绩以[实时汇总](../band_brep_reconstruction/outputs/coarse_fine_20261102_v2/REPORT.md)、[summary.json](../band_brep_reconstruction/outputs/coarse_fine_20261102_v2/summary.json) 和 [state.json](../band_brep_reconstruction/outputs/coarse_fine_20261102_v2/state.json) 为准。本批使用原 F30 中相同的 20 dev，每例运行两组，共 40 个任务；10 check 不运行。

## 固定协议

| 设置 | 基线组 | 成本候选组 |
|---|---|---|
| 粗目标 | 0.02 | 0.03 |
| 细目标 | 0.005 | 0.005 |
| E/V 近场半径 | 0.02 / 0.03 | 0.02 / 0.03 |
| 两阶段分割阈值 | 0.005 / 0.01 | 0.005 / 0.01 |
| 最小分量 | 10 个三角形 | 10 个三角形 |
| 字段 | 生产精确 CAD E/V，oracle presence | 同左 |
| 剪枝 | 仅切超长边 | 同左 |
| 总超时 | 每例 10800 秒 | 每例 10800 秒 |

其余配置完全一致，并在初始化时逐键核验。粗阶段不查场，细阶段及后续计算继续使用原始 condition 作为模型/SDF 基准，复用最终中心字段。无轮数、三角形数或 RSS 硬上限；并发为两组共用两个 CPU worker，按样本交错调度，不使用 GPU。

`feature_sdf_tolerance=0.004`、查询 batch=65536、关闭查询落盘，沿用历史比较口径。总超时从 worker 启动起包含输入核验、源准备、condition、两级加密、拓扑、几何、STEP 和离线评分。旧 MC 与原生四轮数字读取历史记录，不重跑或覆盖旧结果。

## 生产版本与输入一致性

按生产目录同步了 36 个受管理文件。生产 STEP 读取规则更新为选取最大的有效正体积实体；本批逐例预检确认原 20 dev 均只有一个实体，原网格指纹、归一化矩阵和 E/V 目标数量全部与历史一致。详见[preflight.json](../band_brep_reconstruction/outputs/coarse_fine_20261102_v2/preflight.json)、生产源码记录和各组的 source_audit.json。

本次重建及生产预处理 84 项测试通过。对控制器的临时模拟测试覆盖 40 个任务、双 worker 总并发、两组参数隔离、汇总／最终分析和附加审计保留；模拟结果仅供测试，未进入本批数据。

首次试启动在产生任何样本结果前发现汇总刷新会覆盖附加审计，已停止、清理并修正后重新冻结为当前 v2；该试启动不构成结果组。

## 运行与验收

控制器为 `benchmarks/run_coarse_fine_comparison.py`，单组 worker 使用 `run_mesh_input_experiment.py`。父控制器和 workers 都从冻结的 source_snapshot 运行。各组单独保存 policy、identity、active、logs、runs、records、REPORT、metrics.csv 和 analysis.json；控制器在每个任务完成后更新顶层配对汇总，组完成后自动追加输入／耗时审计。

[粗0.02明细](../band_brep_reconstruction/outputs/coarse_fine_20261102_v2/coarse_002/REPORT.md)、[粗0.03明细](../band_brep_reconstruction/outputs/coarse_fine_20261102_v2/coarse_003/REPORT.md)。每组始终使用 20 例分母，pending 单列，失败和超时保留。完整配对后比较面／图／STEP、粗细面数、总查询量、总耗时、峰值 RSS 和失败原因；运行中的已完成子集不能替代完整成绩。

本批检验精确场下的后处理与粗阈值取舍，不能解释为真实 VAE/Flow 或 decoder 的精度、吞吐。实施与字段契约见[两级加密文档](CoarseFineMeshRefinementPlan.md)。
