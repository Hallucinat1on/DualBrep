# 验证记录

日期：2026-10-13。环境：Python 3.9.25、pythonocc 7.7.0、NumPy 1.26.4、SciPy 1.12.0；PyMCubes、Torch，以及 CAD `mesh_fast` 实验所需的 libigl 可用。

## 独立迁移验证

将整个模块目录复制到 `/tmp/bandbrep-migration-e5vjd5bo/module`，排除输出与 `__pycache__`，在复制后的目录执行：

```bash
/miniconda/envs/brdf/bin/python -m unittest discover -s tests -v
```

**31 项测试全部通过，耗时约 121.83 秒。** 测试日志已复制到 [validation/tests.log](validation/tests.log)。静态导入审查未发现任何对上级 DualBrep 模块的运行时导入。五份复用源码副本逐字节核对与原文件相同。另已成功构建 wheel，并安装到 `/tmp/bandbrep-installed`，从仓库外成功导入主包、OCC 装配和复制的面模型。现有样本 07 的 `--resume` 指纹与文件哈希检查也通过。

完整 STEP 集成案例：

| 输入 | 逻辑 Face / Edge / Vertex | 验证要点 |
|---|---|---|
| 解析立方体 | 6 / 12 / 8 | 顶点由 v-UDF 定位；12 条直线共用 8 个硬端点；六张平面共享边 |
| 解析圆柱 | 3 / 2 / 0 | 两条圆环，无必要顶点；周期侧面显式补 seam |
| 球体通孔 | 2 / 2 / 0 | 两条独立环具有同一面标签对；球面与圆柱孔壁共用同一三维边 |
| 球内腔 | 2 / 0 / 0 | 空边／顶点集；内外表面均保留；分开装配外壳和内腔壳 |
| 平滑球面分界 | 2 / 1 / 0 | 同一球面几何被结构场分成两张面；保留平滑边，球冠补 seam／pole |

其他测试覆盖共形细分、孤立核心保留、第三来源阻断、三来源下包络、距离中心定位与秩亏拒绝、硬端点、整条四边控制边界、边界约束冲突、self-loop 的双 port、隐藏真值隔离、场尺度、存在性未知的拒绝、失败报告、查询记录、缓存哈希与续跑、面实例编号置换、MLP 去重训练／checkpoint／自动微分以及面网络适配器不调用边预测分支。

MLP 测试只验证训练与接口，不宣称两步训练得到准确距离场。面网络适配器使用测试替身验证调用协议；未提供真实预训练权重，因此未宣称已验证真实 checkpoint 的 CAD 重建效果。

## 现有 ABC 样本 07

命令从本模块目录执行：

```bash
PYTHONPATH=/tmp/evudf-test-deps /miniconda/envs/brdf/bin/python reconstruct.py \
  --provider cad \
  --sample-dir ../edge_vertex_preprocessing/outputs/abc10_default_20261012/00000007_b33a147f86da49879455d286_step_000 \
  --sdf-backend mesh_fast \
  --output outputs/abc07_final
```

临时依赖路径仅是本机 libigl 安装位置。其他项目应安装依赖并替换路径。重建从查询 SDF 提取表面，没有读取原 `surface_mesh.npz` 的三块划分，也没有使用源面／边／顶点 id。

结果：**`solid_validated`，耗时约 155.38 秒**，运行时另一个迁移测试进程也在占用 CPU。实际数值保存于 [validation/report.json](validation/report.json)，完整阶段日志见 [validation/abc07.log](validation/abc07.log)。

- SDF 初始表面：65³ 格点，14,504 个三角形，一个封闭连通分量；最大表面 SDF 残差约 `5.37e-4`。
- 第一层近带网格 57,822 个三角形；将最大近带边长从 `0.0125` 减半到 `0.00625` 后得到 169,194 个三角形。两级均为 3 个面核心，核心重叠一一对应。
- 三种阈值倍率 `0.85/1/1.15` 均保持 3 个核心且一一对应；没有按真实面数调参。
- 逻辑图：**3 面、2 边、0 必要顶点**。两条边均拟合为完整圆环，最大支持点拟合误差约 `2.15e-5`。
- 2,048 个独立边场覆盖探针均定位成功，到恢复曲线的最大距离约 `6.90e-4`；这是有限采样检查，不是连续覆盖证明。
- 补缝后的 CAD：3 面、3 条边，其中一条为后处理 seam；逻辑边仍为 2 条。
- OCC 有效性、共享边两侧反向使用、正体积和三角化通过；最大实体容差 `0.001`，未超过配置的 `0.002`。
- 归一化体积约 `0.63881731`；STEP 读回体积差约 `1.37e-6`；读回共享曲线采样的最大误差约 `4.24e-13`。

可查看 [STEP 输出](outputs/abc07_final/reconstruction.step)、[逻辑图](outputs/abc07_final/logical_graph.json) 和 [验收报告](outputs/abc07_final/quality.json)。`outputs/` 已加入本模块 `.gitignore`；它们保留在本机，不会自动作为源码提交。

## 限制

没有完成其他三个现有 ABC 样本的完整重建验证，没有据此宣称 ABC 通用成功率。生成场、噪声场、训练后的真实 NPZ MLP 和真实面网络 checkpoint 的重建精度需要分别评估。

离散稳定性、有限边界残差、OCC 有效性和 STEP 读回通过，不能证明所有未采样薄结构、短边或全局曲面自交都被正确处理。复杂情况下模块返回明确的阶段失败；当前支持边界见 [README.md](README.md)。
