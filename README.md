# E/V-UDF 数据处理与 B-rep 重建

本分支包含 [veudf v3 数据预处理](veudf/README.md) 和 [Post1109 重建模板](band_brep_reconstruction/README.md)。原 DualBrep 的 GVD-UDF、模型和脚本集中在 [legacy_dualbrep/](legacy_dualbrep/README.md)。

```bash
python scripts/sync_veudf.py --check
python -m pip install -e './veudf[jobs,postprocess]' -e ./band_brep_reconstruction
band-brep-reconstruct --step /path/to/source.step --output /path/to/new_run
```

预处理从 STEP 生成 `edge_vertex_neighborhood_pools_v3`，训练 E/V 截断到 0.05，独立评估保留完整 CAD 距离。重建可从源 STEP 开始，也可用 v3 几何 condition 和连续 E/V 查询。环境和数据构建命令见各模块 README。

- [文档导航](docs/README.md)
- [历史实验](docs/Experiment.md)
- [目录归属与清理清单](docs/Cleanup.md)
