# 从 8D 联合 E/V 场到 B-rep（网格 & 点云统一后处理）

实现 [8D 联合目标与中场向量拉回方案](../docs/VectorFieldTarget.md) 与 [现行统一后处理规范（Post1125）](../docs/Post1125.md)。支持**闭合三角网格**与**纯点云**双输入，配合近表面带（默认 `0.02`）内的连续 8D 联合场查询 `[dE, dV, uE(3), uV(3)]`（标量无符号距离 + 3D 最近点位移向量），端到端重建通过 OCC 读回校验的水密 STEP。

在含 Python 3.10+、OCC 7.7、libigl 2.6.3、shapely 2.1+ 与 pymeshlab 的环境，从仓库根目录运行：

```bash
python -m pip install -e ./veudf -e './band_brep_reconstruction[test]'
band-brep-reconstruct --step source.step --output run
```

| 阶段 | 方法与产物 |
|---|---|
| 输入适配 | 网格直通 `reconstruct(mesh, udf, cfg, out)`；纯点云走 `reconstruct_point_cloud(points, udf, cfg, out, normals=..., proxy_mesh=...)`（`point_cloud_source.py` 自动 PCA+MST 定向与 Screened Poisson 构造水密代理网格定拓扑，同时保留原始点云用于曲面拟合） |
| `scout` | 按全程二分 `0.024 → 0.012 → 0.006 → 0.003`（`scout_margin=0.007`）分层筛选近表面块，整数格点去重并缓存 8D 标量与位移向量。输出 `scout.npz` |
| `wireframe` | `vector_wireframe.py` 执行 8D 中场向量拉回（`extraction='vector_pull'`）：E/V 并行中场拉回 → 双向互证过滤假顶点与自适应解析切断球 → V 主导切断 → 微体素共识 + 1D 加权 PCA 瘦身 + 切向/横向偏移聚类 + EMST 排序与断口桥接 → 固定端点拟合 Line / Circle / C1 Bézier。输出 `edge_support.npz`、`wireframe.json`、`curves.json` |
| `split_mesh` | 网格输入优先尝试平面快路径（`fast_faces`），否则走精确流形切面（`edge_first_embedding.py` + `boundary_cycles.py`），确定性恢复面标签、有向邻接与外环/内孔环（含重叠边受控合并与解析求交重定位）。输出 `split_mesh.npz`、`logical_graph.json` |
| `surfaces` | 网格采样或原始点云映射 + 边界腐蚀核心点；极窄/削圆面（核心点 `< min_core_points=15`）自动启用 `boundary_prior.py`（Newell + 2D PCA 局部基平面/二次曲面先验兜底）；依次拟合解析面 → 回转面 → LSCM/PCA 薄板 B-spline（含 `face_patches.py` 内部切缝重试）。输出 `face_patches.json` |
| `solid` | 共享边和 pcurve、全周期面显式 seam、圆锥/球冠极点退化边（`make_conical_cap` / `make_spherical_cap`）及 STEP 导出读回校验。输出 `reconstruction.step`、`logical_to_cad.json`、`quality.json` |

关键配置见 [default.json](configs/default.json)。`--stop-after scout|wireframe|split_mesh|solid` 控制实验终点，每次使用空输出目录。离线评分见 [benchmarks/](benchmarks/README.md)。
