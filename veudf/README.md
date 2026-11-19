# E/V-UDF 数据预处理

本目录镜像生产 [Hunyuan3D/veudf](/opt/data/private/yihengxu/Hunyuan3D/veudf)。当前数据格式为 `edge_vertex_neighborhood_pools_v3`。数据契约、采样和旧产物处理见[预处理说明](../docs/Pre.md)，同步文件的 SHA-256 见 [SOURCE.json](SOURCE.json)。

在 Python 3.10+、OCC 7.7.0、libigl 2.6.3 环境下，从仓库根目录运行：

```bash
python -m pip install -e './veudf[jobs,postprocess,test]'
python -m veudf.cli.build_dataset manifest \
  --step-dir /data/abc/step/0000 --out /data/new_manifest.jsonl
python -m veudf.cli.build_dataset build --root /data/new_mesh_pools_v3 \
  --manifest /data/new_manifest.jsonl --config veudf/configs/surface_pools.json \
  --workers 4 --threads 4 --timeout 7200
```

训练通道为 `[s_mesh, min(dE,0.05), min(dV,0.05)]`，独立评估用完整 CAD E/V。默认普通池共 1,572,864 点，`surface / near_surface / edge_neighborhood / vertex_neighborhood` 的占比为 25%／17.5%／35.5%／22%。另有 `edge_zero` 和 `vertex_zero`。构建 v3 数据使用新根目录，旧批次使用原冻结代码与配置。

| 入口 | 用途 |
|---|---|
| `veudf.cli.build_dataset` | 清单与数据构建 |
| `veudf.cli.prepare_condition` | 生成几何条件 |
| `veudf.cli.prepare_remaining_run` | 冻结增量构建输入与源码 |
| `veudf.cli.run_guarded_build` | 资源受限时软停止，保持身份后可续跑 |
| `adaptive_build.py` | 生产自适应构建执行器 |
| [benchmarks/](benchmarks/README.md) | 预处理基准 |
| `scripts/visualize_assets.py` | 数据资产可视化 |

```bash
python sync_veudf.py --check
python -m pytest veudf/tests -q
```

[边优先重建](../band_brep_reconstruction/README.md)使用几何 condition 和连续 E/V 查询。源 STEP 重建调用精确 CAD reference 接口。新实验使用 v3 数据根目录和当前重建入口。
