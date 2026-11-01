# 旧预处理入口已迁移

2026-11-01 起，后续数据处理以生产目录 `/opt/data/private/yihengxu/Hunyuan3D/veudf` 为准，本仓库同步副本位于 [veudf/](../veudf/README.md)。旧 `evudf` 代码、CLI、配置和测试已移除，避免维护两套生成流程。

在仓库根目录运行：

```bash
python scripts/sync_veudf.py --check
python -m pip install -e './veudf[jobs,postprocess,test]'
python -m veudf.cli.build_dataset --help
```

新 schema 为 `edge_vertex_mesh_pools_v1`，使用最终 CAD 网格 SDF、精确 CAD E/V 与 `.npy` 点池。旧 `edge_vertex_udf_v1` 的 `fields.npz / metadata.json`、旧配置和 `evudf-*` 命令不兼容，不能继续用于新数据生产。规范与迁移边界见[三场数据处理](../docs/ABCEdgeVertexUDFPreprocessing.md)。

本目录中的历史 `outputs/` 和 `visualizations/` 保留作实验溯源，不是新批次输入规范。重建模块 `vendor/oracle` 保留旧版本冻结查询代码，以复现实验；它不生成新的生产数据。
