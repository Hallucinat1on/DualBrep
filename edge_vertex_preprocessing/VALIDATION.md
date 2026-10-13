# 实现验证记录

日期：2026-10-12。该记录说明当前实现的实际验证范围，后续代码变化应重新执行测试。

## 回归测试

在 Python 3.9.25、pythonocc-core 7.7.0、NumPy 1.26.4、SciPy 1.12.0 下运行 22 项测试，全部通过。可选 libigl 2.6.2 安装到临时测试依赖目录，并包含在本次测试中；没有修改项目环境。

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

验证项目包括：

- 长方体边／顶点及 SDF 解析距离。
- 圆柱 seam 排除、完整圆环及空顶点集。
- 合法空边／空顶点集、空腔 SDF 内外符号。
- 共面 face 分界保留、短边最低覆盖。
- 有限圆弧／直线段的端点距离、高曲率 Bézier 查询。
- BVH 与全量 CAD 距离的一致性及剪枝。
- 零集、并列最近点、圆弧轴线连续多解及梯度 mask。
- 退化边与多 solid 拒绝。
- 归一化逆变换、gzip 输入、三通道对齐和空集读取。
- 保存哈希、配置变化时拒绝续跑、保护无关目录。
- 将整个目录复制至临时位置后独立运行并行 CLI 和续跑。
- worker 超时记录、manifest 文档分组与重复内容合并。
- 可选 mesh SDF 与 CAD reference 的抽查。

未安装 libigl 时，对应测试显示为 skipped；其余基准使用 CAD SDF，无需 libigl。

## ABC 官方示例

使用 ABC 官网提供的 [`00000050` STEP 示例](https://deep-geometry.github.io/abc-dataset/data/00000050_80d90bfdd2e74e709956122a_step_000.step)，源文件约 99 KB。该 STEP 含 3 个 solid；直接输入时返回 `unsupported_multi_solid`，符合单 solid 范围约定。

源文件 SHA256：`579e4b3e18709f6888a4179f189926e40c35a35c156392ec593105cf1650a60c`。

在临时目录中将三个 solid 分别导出为 STEP，采用 `configs/smoke.json`、`cad_bvh` 边查询、`mesh` SDF、2 个 worker 进行标注。三个 parts 均通过检查：

| Part | Face | 目标 edge | 必要 vertex | 查询点数 | SDF 最大抽查误差 | E-UDF 最大抽查误差 | V-UDF 最大抽查误差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5 | 6 | 0 | 274 | $9.70\times10^{-5}$ | $2.24\times10^{-8}$ | 空集合 |
| 1 | 10 | 17 | 10 | 317 | $6.35\times10^{-5}$ | $2.22\times10^{-8}$ | $2.02\times10^{-8}$ |
| 2 | 28 | 76 | 48 | 532 | $1.62\times10^{-5}$ | $2.77\times10^{-8}$ | $1.27\times10^{-8}$ |

误差以归一化长度计，是分层查询子集对 CAD reference 的差值；不是全域误差证明。无歧义的最近 edge id 抽查没有发现不一致。Part 0 同时验证了真实数据的“有边、无必要顶点”状态。

这是一份官网模型的三个 parts，不代表 ABC 全集的接受率、分布或吞吐量。原始／派生 STEP 和测试输出仅保存在临时目录，不随源码分发。

## 迁移与构建

已验证源码目录复制后不依赖 DualBrep 根目录运行。CLI `--help` 可在没有 OCC / NumPy 的 Python 环境执行；实际距离标注需要安装 README 中的依赖。

`pip wheel --no-deps --no-build-isolation .` 成功生成 `evudf_preprocessing-0.1.0-py3-none-any.whl`。源码和 wheel 均为 Python 实现；OCC 内核需在目标环境另行安装。

一般 B-spline 闭环锚点和人工二度分段采用保守保留，尚未实现全局自交／重合认证。每份输出的 `topology.json` 和 `quality.json` 明确记录规范化状态及检查范围。

## 本地 ABC 10 个模型测试补充

本地数据位于 `/opt/data/private/yihengxu/Datasets/abc/step/0000`。选取前十个模型目录 `00000000`–`00000009`，使用默认 200,000 混合查询点，不用 smoke 配额，也不替换被拒绝的模型。

模型 `00000003` 的逐点 CAD 内外分类耗时较大：128 个查询的 CAD SDF 约 2.2 秒，现有 `mesh` 后端仍调用 classifier，2048 点约 25.9 秒。因此补充 `mesh_fast` 后端，通过 fast winding number 计算 SDF，保留 CAD/BVH E-UDF 和精确顶点最近邻。原 CAD 任务主动终止，日志保留这一操作；不能把退出信号误解为代码崩溃。

新增 fast mesh 端到端、空腔符号及反向网格拒绝测试。完整十个模型的结果、实际采样数与运行后端记录在 `outputs/abc10_default_20261012/REPORT.md` 和 `summary.json`。

补充后共 24 项回归测试全部通过。十个本地数据的最终结果为 **4 个通过、6 个按规则拒绝**：3 个多 solid，2 个含退化边，1 个归一化 CAD 容差超出预算。未用其他模型替换被拒绝的样本。

| 模型 | 共享查询数 | 必要顶点数 | 完成耗时（秒） | SDF 后端 |
|---|---:|---:|---:|---|
| 00000003 | 200,138 | 18 | 120.87 | mesh_fast |
| 00000007 | 200,006 | 0 | 109.42 | cad |
| 00000008 | 200,122 | 17 | 192.80 | cad |
| 00000009 | 200,015 | 0 | 136.09 | cad |

成功样本均保存 10,000 个表面点与 10,000 个边点，合计保存 800,281 个三通道查询。E-UDF／V-UDF 最大 reference 抽查差值不超过约 $5.70\times10^{-8}$；模型 `00000003` 的 fast mesh SDF 最大抽查差值约 $1.52\times10^{-4}$，单位均为归一化长度。

表中完成耗时不含 `00000003` 原 CAD 任务主动终止前已运行的约 1002 秒。原始日志、重跑日志及后端切换记录均保留，不能将该次主动取消解释为处理器自然超时或代码崩溃。
