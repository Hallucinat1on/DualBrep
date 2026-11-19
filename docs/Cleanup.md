# 目录归属与清理清单

本清单按 2026-11-19 的工作区和 Git 分叉前基线 `3f27607` 核对。容量是当时的本地磁盘占用估计。迁移已完成，清单中的可删项尚未执行删除。

| 路径 | 归属与用途 |
|---|---|
| [veudf/](../veudf/README.md) | 现行生产预处理镜像，含 v3 代码、配置、工具及测试 |
| [scripts/sync_veudf.py](../scripts/sync_veudf.py) | 生产镜像同步及哈希检查 |
| [band_brep_reconstruction/](../band_brep_reconstruction/README.md) | 现行 Post1109 重建、独立评分、配置及测试 |
| [legacy_dualbrep/](../legacy_dualbrep/README.md) | 原 DualBrep 源码、原版 README、原理解读、配套样例和本地资产 |
| [docs/](README.md) | 现行说明、E/V 历史实验和设计备忘 |

`legacy_dualbrep/` 有原版 `Voronoi/`、`brep_post/`、训练、推理、分割和参数化脚本、配置、样例与 11-11 圆锥实验。`visualize_sdf_udf.py` 是本仓库后加的原版 GVD-UDF 可视化工具，也随旧流程归拢。新重建内复用的源码出处见 [PROVENANCE.md](../band_brep_reconstruction/PROVENANCE.md)。

## 可再生成的文件

以下条目只在非活跃任务目录清理，运行中的批次和冻结快照整目录留存。

| 路径或匹配项 | 清理条件 |
|---|---|
| 项目中的 `__pycache__/`、`*.pyc`、`*.pyo`、`.pytest_cache/` | Python 与测试缓存，按需重新生成 |
| 项目中的 `*.egg-info/`、Python `build/` 和 `dist/` | 打包产物。清理后如需 editable 安装，重新安装相应模块 |
| `legacy_dualbrep/Voronoi/build/` | 原版 CMake 构建产物。迁移前的 CMake 缓存带旧路径，重新编译时使用新构建目录 |
| `.cache/huggingface/` | 本地缓存。确认未被下载过程占用时可清理 |
| `.vscode/` | 本地编辑器设置。仅在无需保留个人设置时清理 |

## 原版与文档可选清理

| 路径 | 清理条件 |
|---|---|
| `legacy_dualbrep/` 整体 | 只维护现行 E/V 路线时可整体归档后移出工作树。同时调整 README 和文档导航，保留源码归属与基线提交记录。Git 可恢复跟踪源码，不能恢复其中被忽略的本地资产 |
| `legacy_dualbrep/checkpoints/` | 约 14 GB。仅原版使用。确认已有可靠备份或可重新下载对应权重后清理 |
| `legacy_dualbrep/Voronoi/external/` | 约 7.7 GB。确认原版外部依赖可重装且无原版任务使用后清理 |
| `legacy_dualbrep/demo/`、`legacy_dualbrep/output_pipeline/`、`legacy_dualbrep/output_udf_demo/` | 原版演示和推理产物。先保存需要的模型、输入及最终 STEP |
| `legacy_dualbrep/output_cone_test_20261111/` | 原版圆锥实验，含独立源 STEP、包装脚本与观察结果。先保留需复现的输入、报告、关键输出和依赖版本，再清理体素场和中间产物 |
| `docs/Post1104.md` | 被现行流程取代。若不再维护历史过程说明，可移入归档或删除，并移除导航链接 |
| `docs/Para.html` | 聊天导出，和 `Para.md` 的内容不完全相同。确认其独有的参数化讨论已保留或不再需要后可删除 |
| `docs/Pipeline.md`、`docs/Para.md`、`docs/Wireframe.md` | 研究备忘，可合并或移入归档。当前代码和测试不依赖这些设计文档 |
| `docs/Post1116.md` | 旧实验复现手册。仅在把仍需保留的协议、命令和结果入口并入 `Experiment.md` 后可移除 |
| `docs/archive/20261104_document_sources.json.gz` 及配套 manifest | 原文和来源记录。只有在整组归档另有完整副本并迁移相关引用后才可移除 |

`README.md`、`docs/README.md`、`docs/Pre.md`、`docs/Post1109.md`、`docs/Experiment.md` 以及模块 README 是现行入口与实验索引。生产镜像及其 `SOURCE.json`、测试和同步工具也应保留。

## 独立实验产物

- **保留活跃批次** `band_brep_reconstruction/outputs/bucket500_20261117_v1/`。最近一次检查约 291 GB，确认有 8 个活跃 worker。其目录下的 `active/`、`runs/`、`logs/`、`records/`、`source_snapshot/`、清单及身份文件都属于同一运行资产。批次结算后再评估裁剪。
- **保全关键证据** `single_pass_t003_20261106_v1/`、`curve_distance_coarse003_20261109_v1/`、`region_repair_coarse003_20261116_v1/`、`alignment30_20261019_v3/`。它们支撑历史成绩、对比组和共同输入。保留清单、策略、冻结代码、逐例记录、报告、必要输入和成功 STEP。若只保留统计而不再复现，先另存归档，再裁剪重复网格、三角形中心场、重复快照或失败中间件。裁剪后应在归档标记可复现范围。
- **仅在保留需要的证据后可清理** `band_brep_reconstruction/benchmarks/datasets30_20261013/`。这是旧 S30 本地输入资产，约 318 MB。清理前保存 manifest、必要输入及来源信息。

下列条目均在 `band_brep_reconstruction/outputs/`。它们不参与现行重建入口。退出对应历史用途以后，可按组清理：

| 对象 | 目录或文件名 |
|---|---|
| 试验和预检 | `bucket_trial_20261117_v1/`、`bucket_trial_20261117_v2/`、`bucket_trial_20261117_v3/`、`bucket_trial_high_20261117_v1/`、`region_repair_preflight_20261116_v1/` |
| 可视化 | `failure_visualization_20261117_v1/`、`failure_visualization_20261117_v2/`。保留仍引用的 HTML、截图或诊断时，应连同其资源保留 |
| 基线与版本验证 | `abc07_final/`、`alignment30_20261019/`、`alignment30_20261019_v2/`、`native_mesh_L5_20261101_v1/`、`coarse_fine_20261102_v2/`、`coarse_fine_validation_20261101_v1/`、`coarse_threshold_validation_20261102_v1/`、`curve_distance_validation_20261109_v1/`、`curve_distance_validation_20261109_v2/` |
| 网格、分割和关联消融 | `mesh_density_20261028_v2/`、`mesh_density_analytic_20261028/`、`refinement_comparison_20261014_145926/`、`segmentation_ablation_20261014_004422/`、`two_pass_validation_20261015/`、`multilabel_validation_20261014_172636/`、`datasets30_continue_20261013_225540/`、`datasets30_multilabel_20261014_182251/`、`datasets30_two_pass_20261015_004354/` |
| 便利索引及历史归档 | `*_latest.txt` 可随对应目标调整或清理。`cleanup_redundant_outputs_20261101.json`、`document_consolidation_20261019/`、`experiment_archive_20261015/` 是清理与文档来源记录，先另存归档 |

`veudf/outputs/` 下的 `surface_asset_validation_20261102_v1/`、`surface_distance_20261102_v1/`、`surface_distance_20261102_v2/`、`surface_distance_20261102_final/` 属于旧版预处理验证，合计约 4.3 MB。保留所需审计记录之后可清理。生产数据实际存放在外部数据目录，清理本仓库时保留外部数据和源 STEP。
