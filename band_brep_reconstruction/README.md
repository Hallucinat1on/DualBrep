# 从网格和 E/V-UDF 恢复边、面及 STEP

当前分支实现 [边优先方案](../docs/ScanEdgeFirst.md)。首版输入为闭合、方向一致、已归一化的三角网格与可连续查询的 E/V。历史面优先流程见 [Post1109](../docs/Post1109.md)。

在具备 Python 3.10+、OCC 7.7 和 libigl 2.6.3 的环境从仓库根目录安装：

```bash
python -m pip install -e ./veudf -e './band_brep_reconstruction[test]'
band-brep-reconstruct --step source.step --extraction trace --output trace_run
band-brep-reconstruct --step source.step --extraction mc --output mc_run
# 共用一份粗细查询，分别写出两路产物
band-brep-reconstruct --step source.step --extraction both --output paired_run
```

STEP 入口用同一次归一化得到输入网格和生产 v3 的精确 CAD E/V 查询。全直边模型用批量有限线段距离，其他曲线沿用生产参考 oracle。重建器只接收网格、标量场及其有效性与存在性。CAD 实例 ID 和关联保留在离线评分侧。

| 阶段 | 方法与产物 |
|---|---|
| `scout` | 近表面稀疏粗块 (H=0.012) 与候选细格 (h=0.003)，E/V 查询位置按整型格点去重。距输入网格 0.015 内优先，外层仍查询，中心无效时探查角点。有效带最多 0.02。输出 `scout.npz` |
| `wireframe` | `trace` 投影带内低 E 样本，沿局部切向双向连接。 `mc` 提取有效单元中的 (E=0.005) 管壳，再将壳顶点向 E 低谷投影和去重，形成骨架支撑。两者通过 V 定位、验证短边候选及固定端点距离拟合得到 `wireframe.json`、`curves.json`。MC 另存 `edge_shell.npz` |
| `split_mesh` | 将三维曲线嵌入网格副本，沿局部路径插入边界约束。由切口的有向两侧确定面标签、E–F 关系和有向环。输出 `split_mesh.npz` 和 `logical_graph.json` |
| `solid` | 解析面或有约束的 B 样条、共享边装配及 STEP 读回。通过时输出 `reconstruction.step` |

`--stop-after` 可指定上述阶段。数值配置见 [default.json](configs/default.json)，`--config` 可覆盖参数。`scout_point_budget` 限制粗块数量和细格去重前的角点分配量。每次使用空输出目录。 `run.json`、`quality.json` 记录输入身份、阶段用时及失败诊断，失败时保留上游已生成的几何。使用 `both` 时，父目录保留通用查询数据，`trace/` 和 `mc/` 分别记录重建结果及耗时。任一路失败时，配对命令返回失败。

Python 入口为 `bandbrep.edge_first.reconstruct(mesh, udf, Config(), output_dir)`。可通过 `CallableUDFProvider` 接入已归一化坐标中的连续 E/V，明确通道是否存在。首版针对同源 STEP 网格和精确场。开放点云、预测噪声与扫描纠偏需要另行验证。MC 的管壳投影是近似骨架方法，近邻分支合并、截断区和缺失支撑需查看诊断。离线评分见 [benchmarks/](benchmarks/README.md)。
