# 文档导航

| 用途 | 文档 |
|---|---|
| 总体方案：几何先验条件下的线框优先生成式 B-Rep 重建、干净 v4 预处理与退化边处理 | [Pipeline.md](Pipeline.md) |
| 8D 联合目标表示（UDF + 3D 最近点位移向量场）与中场零集修正方案 | [VectorFieldTarget.md](VectorFieldTarget.md) |
| v5 数据集格式与全链路改造方案（8D 联合场 + 编解码池彻底解耦，现行） | [Dataset.md](Dataset.md) |
| 面向扫描点云与生成式 Mesh 的退化数据构造与抗噪训练方案 | [DegradedTraining.md](DegradedTraining.md) |
| 历史 STEP 到 v3 数据说明 | [Pre.md](Pre.md) |
| 当前网格到 STEP 入口 | [边优先重建 README](../band_brep_reconstruction/README.md) |
| 后处理文档版本关系与归档口径（最新） | [Post1125.md](Post1125.md) |
| 上一代边优先方案（MC 球壳 + 体素 thinning 前端，已被 8D 向量拉回取代） | [Post1121.md](Post1121.md) |
| 历史面优先流程 | [archive/Post1109.md](archive/Post1109.md) |
| 原 DualBrep 源码和原理解读 | [原版 README](../legacy_dualbrep/README.md) · [DualBrep.md](../legacy_dualbrep/DualBrep.md) |
| 历史成绩、协议和冻结命令 | [Experiment.md](Experiment.md) · [archive/Post1116.md](archive/Post1116.md) |
| 旧处理说明 | [archive/Post1104.md](archive/Post1104.md) |
| 研究备忘 | [Wireframe.md](Wireframe.md) |

现行使用以 `Pipeline.md`、`Pre.md`、`VectorFieldTarget.md`、[边优先重建 README](../band_brep_reconstruction/README.md) 和各模块 README 为准；`Post1121.md` 的线框提取前端（阶段 1/2）已被 8D 向量拉回取代，版本关系见 `Post1125.md`。历史实验保留原配置、数据版本和成绩口径，`Post1104/1109/1116` 已归档至 `archive/`。
