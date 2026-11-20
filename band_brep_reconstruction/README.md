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
| `wireframe` | 从有效 E 细格构建管内体素，做按距离排序的三维保拓扑细化。拟合 V 球壳得到顶点，用骨架与切球的交点确定边的端点，然后固定端点拟合曲线。输出 `edge_skeleton.npz`、`vertex_shell.npz`、`wireframe.json` 和 `curves.json` |
| 面拓扑及拟合 | 将线框粗投影到原网格的边，按屏障划分面域，在面域内按面积采点做平面 RANSAC。通过边界和装配验收时写 `face_regions.npz`，其中保存原网格及面标签。其余样本用 `split_mesh.npz`，自由面拟合失败时尝试在已有边界顶点之间增加内部 patch 接缝 |
| `solid` | 共享边和 pcurve、周期面接缝及 STEP 导出读回。裁剪结果写 `logical_graph.json`。若增加内部 patch，另存装配图及 CAD patch 到逻辑面映射 |

`centerline_method` 默认 `thinning`，可设 `sections` 复现管壳截面方案并输出 `edge_shell.npz`。`mc_udf_refine` 默认 `off`，线框提取、拟合及验收只用几何和缓存 E/V。启用 `sparse` 可在有限中心点补查标量 E，按横向对称性接受位置修正。可用 `fast_faces=false` 直接走精确切面，`patch_budget=1` 禁用内部接缝回退。其他参数见 [default.json](configs/default.json)。

`--stop-after scout|wireframe|split_mesh|solid` 控制实验终点，`split_mesh` 强制走切面路径。每次使用空输出目录。通过时写出 `reconstruction.step`。失败保留上游几何和 `run.json`、`quality.json`。Python 入口为 `bandbrep.edge_first.reconstruct(mesh, udf, Config(), output_dir)`。

骨架细化保留已观测体素的连通性。未绑定的分支、有效带截断和自由面缺少可切内部路径等情况会保留诊断。开放扫描和带模型噪声的完整重建仍需独立验证。离线评分见 [benchmarks/](benchmarks/README.md)。
