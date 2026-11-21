# 从 E/V 管壳到 B-rep

实现 [边优先重建流程（Post1121）](../docs/Post1121.md) 与 [管壳与中心线分析](../docs/MCShellCenterline.md)。输入为闭合、方向一致、已归一化的三角网格与近表面带（默认 `0.02`）内的连续 E/V 查询（模型不单独生成 SDF）。STEP 入口用一次归一化得到网格和生产 v3 的精确 CAD E/V 查询，真值拓扑只用于评分。

在含 Python 3.10+、OCC 7.7 和 libigl 2.6.3 的环境，从仓库根目录运行：

```bash
python -m pip install -e ./veudf -e './band_brep_reconstruction[test]'
band-brep-reconstruct --step source.step --output run
```

| 阶段 | 方法与产物 |
|---|---|
| `scout` | 按全程二分 `0.024 → 0.012 → 0.006 → 0.003` 分层筛选近表面块，整数格点去重并记录各层查询量。输出 `scout.npz` |
| `wireframe` | 从有效 E 细格做三维保拓扑细化（综合边界深度与 E 谷值序）；从 V 球壳拟合顶点，对粘连球壳用缓存局部极小值及局部细化盒拆分（仍不可分时合并为超级顶点）；对锐角并管自适应扩展切球，对窄双链气泡压扁，对中段并管做稀疏局部管道 `(h/2, r/2)` 细化与分叉点提升，并恢复切球内短边；按精修置信度鲁棒拟合固定端点曲线（含 Bézier 拟合后表面贴合复核与弦长补点保形回退）。输出 `edge_skeleton.npz`、`vertex_shell.npz`、`wireframe.json` 和 `curves.json` |
| 面拓扑及拟合 | 优先尝试粗投影平面快速路径（`fast_faces=true`），否则走精确切面（`split_mesh.npz`，含回退尖刺清理、薄壁法向惩罚、Steiner 内部节点避让、共享顶点近邻跳步与重叠合并重试）。面拟合采用边界腐蚀核心采样与 `soft_l1` 鲁棒优化，自由面支持 LSCM 内蕴参数化与薄板平滑样条；双连通自由面支持跨环成对内部 patch 切分；合并边由两侧解析面精确求交重定位（`GeomAPI_IntSS`） |
| `solid` | 共享边和 pcurve、全周期面显式 seam（含异母线 UV 线性螺旋 seam）及 STEP 导出读回。裁剪结果写 `logical_graph.json`，若增加内部 patch 则另存装配图及逻辑面映射 |

关键配置见 [default.json](configs/default.json)：`scan_offset_tolerance` 与 `boundary_tolerance`/`cad_tolerance` 解耦；`local_refine_levels` 控制粘连顶点与并管分叉的局部细化层数；`allow_feature_collapse`（默认 `true`，`collapse_tolerance=0.005`）启用超短边/极近边的受控合并与两侧解析面求交重定位。

`--stop-after scout|wireframe|split_mesh|solid` 控制实验终点，`split_mesh` 强制走切面路径。每次使用空输出目录。通过时写出 `reconstruction.step`，失败保留上游几何和 `run.json`、`quality.json`。Python 入口为 `bandbrep.edge_first.reconstruct(mesh, udf, Config(), output_dir)`。离线评分见 [benchmarks/](benchmarks/README.md)。
