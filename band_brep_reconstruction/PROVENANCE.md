# 复用源码来源

更新：2026-11-01，模块 0.5.0。

`bandbrep/vendor/dualbrep_segmentation.py` 的面核心提取改编自归档原版 [clustering.py](../legacy_dualbrep/clustering.py) 的 `hierarchical_segmentation`，保留第一阶段的连通分量与过滤逻辑，并记录排除原因。原 `clustering.py` SHA-256 为 `c76ce8b8022691fc003b1bbb1229aa9c175d4d23a10267da59865d9f79267432`。

`surface_fitting.py` 的采样格式改编自 [原版 rebuild.py](../legacy_dualbrep/rebuild.py) 的 `sample_faces`，改用本地 NumPy 面积采样并支持单面。OCC 曲面构造组织借鉴 [原版 brep_post/utils.py](../legacy_dualbrep/brep_post/utils.py) 的 `create_surface`。原项目权利与许可仍适用。

网格 SDF、几何身份和表面采样直接依赖 [veudf](../veudf/README.md)，与生产预处理使用同一实现。生产来源与同步哈希见 [SOURCE.json](../veudf/SOURCE.json)。

旧 `vendor/oracle`、`vendor/dualbrep_model.py` 及其适配器随旧输入流程删除，不再维护冻结副本。旧文件来源与哈希可在 Git 历史查看；旧说明引用的 `docs/archive/20261031_document_sources.json.gz` 是历史位置，本次核对时已不在工作区；来源与旧源码身份请从 Git 历史或对应批次快照核验。本说明的本次整理前副本见[2026-11-04 文档归档](../docs/archive/20261104_document_sources.json.gz)。历史实验输出未修改。
