# 从 E/V 管壳到 B-rep

实现 [边优先优化方案](../docs/MCShellCenterline.md)。当前输入为闭合、方向一致、已归一化的三角网格与连续 E/V 查询。STEP 入口用一次归一化得到网格和生产 v3 的精确 CAD E/V 查询，真值拓扑只用于评分。

在含 Python 3.10+、OCC 7.7 和 libigl 2.6.3 的环境，从仓库根目录运行：

```bash
python -m pip install -e ./veudf -e './band_brep_reconstruction[test]'
band-brep-reconstruct --step source.step --output run
```

| 阶段 | 方法与产物 |
|---|---|
| `scout` | 按 `0.03 → 0.015 → 0.003` 分层筛选近表面块，整数格点去重并记录各层查询量。输出 `scout.npz` |
| `wireframe` | 在角值均有效的细格上分别提取 E 管壳和 V 球壳。从管壳几何截面估计中心线，从球壳拟合球心和半径，再做球面切断及固定端点的参数边拟合。输出 `edge_shell.npz`、`vertex_shell.npz`、`wireframe.json` 和 `curves.json` |
| 面拓扑及拟合 | 将线框粗投影到原网格的边，按屏障划分面域，在面域内按面积采点做平面 RANSAC。通过边界和装配验收时写 `face_regions.npz`，其中保存原网格及面标签。其余样本用 `split_mesh.npz`，自由面拟合失败时尝试在已有边界顶点之间增加内部 patch 接缝 |
| `solid` | 共享边和 pcurve、周期面接缝及 STEP 导出读回。裁剪结果写 `logical_graph.json`。若增加内部 patch，另存装配图及 CAD patch 到逻辑面映射 |

`mc_udf_refine` 默认 `off`，出壳后提取、拟合及验收线框时只用几何和缓存 E/V。启用 `sparse` 可在有限中心点补查标量 E，按横向对称性接受位置修正。可用 `fast_faces=false` 直接走精确切面，`patch_budget=1` 禁用内部接缝回退。其他参数见 [default.json](configs/default.json)。

`--stop-after scout|wireframe|split_mesh|solid` 控制实验终点，`split_mesh` 强制走切面路径。每次使用空输出目录。通过时写出 `reconstruction.step`。失败保留上游几何和 `run.json`、`quality.json`。Python 入口为 `bandbrep.edge_first.reconstruct(mesh, udf, Config(), output_dir)`。

实现的截面提取以三轴切片覆盖不同方向，接受稳定的闭合截面。无可信截面、管壳融合、自由面缺少可切内部路径等情况会保留失败诊断。开放扫描和带模型噪声的完整重建仍需独立验证。离线评分见 [benchmarks/](benchmarks/README.md)。
