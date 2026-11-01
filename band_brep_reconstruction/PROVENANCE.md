# 复用源码来源

更新：2026-11-01，模块 0.4.0。

`bandbrep/vendor/dualbrep_segmentation.py` 复制自根目录 `clustering.py::hierarchical_segmentation`，增加排除原因、成功拆分父域和统计输出，标签决策保持一致。原 `clustering.py` SHA-256 为 `c76ce8b8022691fc003b1bbb1229aa9c175d4d23a10267da59865d9f79267432`；测试通过 AST 提取原函数进行随机图对照。运行时不导入根目录模块。

`surface_fitting.py` 的采样格式改编自 `rebuild.py::sample_faces`，改用本地 NumPy 面积采样并支持单面。OCC 曲面构造组织借鉴 `brep_post/utils.py::create_surface`。原项目权利与许可仍适用。

网格 SDF、几何身份和表面采样直接依赖 [veudf](../veudf/README.md)，与生产预处理使用同一实现。生产来源与同步哈希见 [SOURCE.json](../veudf/SOURCE.json)。

旧 `vendor/oracle`、`vendor/dualbrep_model.py` 及其适配器随旧输入流程删除，不再维护冻结副本。旧文件来源与哈希可在 Git 历史查看；整理前说明也保存在 [2026-10-31 文档归档](../docs/archive/20261031_document_sources.json.gz)中。历史实验输出未修改。
