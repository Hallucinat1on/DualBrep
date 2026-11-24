# 从 8D 联合 E/V 场到 B-rep

实现 [8D 联合目标与中场向量拉回方案](../docs/VectorFieldTarget.md) 与 [线框优先重建主方案](../docs/Pipeline.md)。输入为闭合、方向一致、已归一化的三角网格与近表面带（默认 `0.02`）内的连续 8D 联合场查询 `[dE, dV, uE(3), uV(3)]`（标量无符号距离 + 3D 最近点位移向量；模型不单独生成 SDF）。STEP 入口用一次归一化得到网格和 `veudf` v5 精确 8D CAD 场查询，真值拓扑只用于评分。

在含 Python 3.10+、OCC 7.7 和 libigl 2.6.3 的环境，从仓库根目录运行：

```bash
python -m pip install -e ./veudf -e './band_brep_reconstruction[test]'
band-brep-reconstruct --step source.step --output run
```

| 阶段 | 方法与产物 |
|---|---|
| `scout` | 按全程二分 `0.024 → 0.012 → 0.006 → 0.003` 分层筛选近表面块，整数格点去重并同步缓存 8D 标量与位移向量。输出 `scout.npz` |
| `wireframe` | `vector_wireframe.py` 执行 8D 中场向量拉回（`extraction='vector_pull'`）：① E/V 通道并行中场单步拉回（带标量-向量一致性与表面贴合软权重）+ Weiszfeld 加权 L1-median 聚类候选顶点；② E/V 双向互证剔除无边支撑或 `dE` 过高的假顶点，构造自适应解析切断球 $r_i = \min(r_{\text{cut}}, 0.35\,d_{\min})$；③ V 主导切断交点脏区后，经微体素多点共识、1D 高斯加权 PCA 中心线瘦身、位置+切向+横向偏移联合分簇、EMST 直径排序与段级 `dE` 回查、切向对齐挂接顶点、弱边断口桥接及极短边兜底；④ 固定端点鲁棒拟合 Line / Circle / C1 Bézier。输出 `edge_support.npz`、`wireframe.json` 和 `curves.json` |
| 面拓扑及拟合 | 优先尝试粗投影平面快速路径（`fast_faces=true`），否则走精确切面（`split_mesh.npz`）。面拟合采用边界腐蚀核心采样与 `soft_l1` 鲁棒优化，自由面支持 LSCM 内蕴参数化与薄板平滑样条；双连通自由面支持跨环成对内部 patch 切分；合并边由两侧解析面精确求交重定位（`GeomAPI_IntSS`） |
| `solid` | 共享边和 pcurve、全周期面显式 seam、圆锥/球冠极点退化边（`make_conical_cap` / `make_spherical_cap` / 部分旋转面极点退化边）及 STEP 导出读回。裁剪结果写 `logical_graph.json`，若增加内部 patch 则另存装配图及逻辑面映射 |

关键配置见 [default.json](configs/default.json)：`scout_margin`、`edge_mid_min/max`、`vertex_mid_min/max`、`edge_consistency_tol`、`vertex_consistency_tol`、`vertex_cluster_radius`、`vertex_cut_radius`、`edge_link_radius`、`edge_lateral_tol`、`pca_thin_radius/iterations` 控制 8D 中场拉回与切向追踪；`allow_feature_collapse`（默认 `true`，`collapse_tolerance=0.005`）控制极近顶点/边受控合并。

`--stop-after scout|wireframe|split_mesh|solid` 控制实验终点，`split_mesh` 强制走切面路径。每次使用空输出目录。通过时写出 `reconstruction.step`，失败保留上游几何和 `run.json`、`quality.json`。Python 入口为 `bandbrep.edge_first.reconstruct(mesh, udf, Config(), output_dir)`。离线评分见 [benchmarks/](benchmarks/README.md)。
