# 三场 B-rep 重建实验报告（2026-10-12—2026-10-18）

本文合并了此前四份文档：实验汇总（10-12—10-15）、失败原因与改进建议（10-15）、后续实验报告（10-16）、改进实验报告（10-18）。四份原文压缩保存在 `band_brep_reconstruction/outputs/experiment_archive_20261018/report_merge_sources.json.gz`。方案定义见 [隔离带重建方案与实施](BandLabelTopologyRecovery.md)。

全文按时间分四个阶段：

| 阶段 | 日期 | 内容 | 章节 |
|---|---|---|---|
| 一 | 10-12—10-15 | 预处理、v0.1–v0.3 四轮 30 模型批次、带宽/细分消融、解析场 | §4 |
| 二 | 10-15 | 失败归因、改进建议、解析场分割补充验证 | §5 |
| 三 | 10-16 | 257 分辨率基线、阶段隔离、分割/顶点消融、独立 H100、场误差 | §6 |
| 四 | 10-18 | 局部顶点域、回转面、周期面 CAD 规划；逐项消融与新独立集 H100b | §7 |

## 1. 总体结论

| 批次 | 日期 | 主要配置 | S30 STEP | 独立集 STEP |
|---|---|---|---:|---|
| R1 | 10-13 | 旧细分、竞争界线、端口；300 秒限时 | 3/30 | — |
| R2 | 10-13 | 同上，取消超时、部分质量错误改告警 | 3/30 | — |
| R3 | 10-14 | v0.2 全标签传播、单阶段核心，65³ | 2/30 | — |
| R4 | 10-15 | v0.3 DualBrep 两阶段核心，65³ | 1/30 | — |
| 10-16 基线 | 10-16 | 257³、边带共享顶点邻接、τ_V=0.02、严格表面检查 | 5/30 | H100：19/100 |
| 10-18 改进 | 10-18 | 基线 + 局部顶点域 + 回转面 + 周期面 CAD 规划 | **9/30** | H100：32/100（开发集）；**H100b：38/100**（原方法 19/100） |

- **成功率的主要提升来自两步**：
  - 10-16：把分辨率提到与 DualBrep 两阶段阈值相称的 257³，并按共享顶点邻接划分边带；
  - 10-18：补齐顶点、曲面、CAD 三处的具体短板。

  在调参时没用过的新独立集 H100b 上，最终成功率为 38%，原方法为 19%；没有一个原本成功的模型变成失败。10-18 的配置已固化为库中的 `recommended` 预设（`--preset recommended`）；代码默认值仍是历史配置（65³，相当于 R4），直接使用默认值得不到上述结果。
- **各轮的数字不能简单相加或直接比较**：
  - 所有批次共用同一批 S30 模型，不是独立样本；
  - R1–R4 使用 65/129 分辨率和 continuation 告警策略，10-16 起使用 257³ 和严格表面检查；
  - 独立集结论以 H100b 为准。
- **当前主要瓶颈是面分割**：H100b 剩下的 62 个失败中，46 个出错在面分割（窄面被边带吞掉、窄面碎成多个核心）。这一项至今没有改动，是下一步的重点。
- **几何保真**：10-18 所有通过 STEP 的模型都核对了几何：体积相对误差中位数小于 1e-4，与真值表面的最大双向距离不超过 0.0019（坐标归一化到 [-1,1]，最长边 1.8）。
- **只用了精确场**：主线实验全部使用由 CAD 生成的精确三场。10-16 的扰动实验表明，场误差达到 0.001 时 CAD 阶段就大量失败；神经场两例都没有完成 STEP。

## 2. 数据、场与评估口径

### 2.1 模型集

| 名称 | 规模 | 来源 | 用途 |
|---|---:|---|---|
| S30 | 30 | `abc/step/0000` 按目录顺序取每目录第一份 STEP，通过范围检查且源哈希不重复的前 30 个（预筛 120 个），来自 13 个源文档组 | 全部阶段的开发集 |
| S8 | 8 | S30 中 10-16 基线拓扑诊断正确的模型：07、09、22、61、68、70、93、102 | 10-16 完整流程、扰动、几何消融 |
| H100 | 100 | 种子 20261016；面数 ≤8、9–25、26–80、81–250 四层各 25；每个源文档组最多一份，排除 S30 的源文档组 | 10-16 独立评估；10-18 调试时用到，降为开发集 |
| H100b | 100 | 种子 20261017；分层方式同上；排除 S30 与 H100 的全部源文档组（交集为 0） | 10-18 独立评估，抽取前已冻结配置 |

S30 不是 ABC 的随机样本；H100、H100b 都限定为受支持的单闭合实体，是分层工程测试，不是 ABC 总体成功率估计。原始 STEP 位于外部 Datasets 目录，实验中没有改动。

### 2.2 三场

- 三个通道共用查询位置，通道顺序为 `[SDF, e-UDF, v-UDF]`，坐标域 `[-1,1]^3`，SDF 内负外正，距离按归一化长度计。
- 目标边不含生成阶段的 seam（seam 在 CAD 后处理补建），保留不同面之间的相切/共面边及必要顶点。
- 阶段一：30 模型共保存 **6,006,802** 个三场查询位置，每模型 200,006–200,793 个。重建时通过连续 CAD oracle 在新位置查询场值，并与保存数据和 OCC reference 抽查一致性。抽查最大参考误差：SDF `0.00019115607`、e-UDF `5.84838438e-08`、v-UDF `5.89968214e-08`。
- 阶段三起：SDF 用 `mesh_fast`；e-UDF 由稠密 CAD 边折线计算，与 OCC reference 比较的最大误差为 `1.42485413e-07`（门槛 1e-5）；v-UDF 精确查询。SDF 无符号距离的有限参考最大误差为 `0.000186780805`，**不声称 SDF 达到 1e-5**。
- 以上都是有限抽查结果，不是全域误差或符号认证。普通重建只接收标量、梯度和存在性；真值只用于离线评分和显式标注的真值注入组。

### 2.3 判定口径

- **STEP 通过（`solid_validated`）**：OCC 内核检查、容差、方向/使用关系和 STEP 读回全部通过。进程退出码为 0 只表示任务结束，不表示 STEP 成功；预处理失败、进程失败和超时都计入模型级失败，不从分母中剔除。
- **面诊断 / 拓扑诊断（`face_good` / `graph_good`）**：阶段三起的离线最近支撑评分。
  - 所有三角形中心按最近真值面三角形、真值边密集采样、真值顶点归属；第二来源至少 3 点且占比至少 5% 视为混合。
  - 面诊断要求每个真值面恰好对应一个核心，没有缺失、重复或混合。
  - 拓扑诊断在面诊断基础上，还要求边、顶点一一对应，且 E–F、V–E、E–V 关联全部一致。
  - 这是诊断，不是数学认证，不检查坐标精度、曲线/曲面质量和后处理。

  阶段一使用较弱的面核心采样诊断：每核心最多 128 点；域混合按每域最多 512 点判断。
- **几何保真（10-18 新增）**：对通过 STEP 的模型，计算重建实体相对真值的体积误差，以及双方各取 2 万个面积加权表面采样点的双向距离。用的是三角化采样距离，不是精确 Hausdorff 距离。
- **STEP 通过不等于拓扑正确**：00008951 通过了 STEP 且几何忠实，但一个约 0.006×0.006 的小平面被相邻核心吞掉了（§7.5）。所以本文同时报告 STEP 通过数和拓扑诊断通过数。
- **失败归因**：阶段一只统计每模型首个阻断错误。阶段三、四另外用离线诊断找出最早出错的环节；拓扑诊断不拦截后续步骤，很多拓扑已错的模型要到 `curves` 阶段才报错。

## 3. 实验总表

| 编号 | 日期 | 实验 | 样本/任务 | 结果 |
|---|---|---|---:|---|
| P0 | 10-12 | 最初十模型预处理 | 10 | 4 通过、6 范围拒绝 |
| S0 | 10-13 | 00000007 单模型完整验证 | 1 | 1 成功，约 155.38 秒 |
| R0 | 10-13 | 严格 SDF 检查初轮（已被 R1 覆盖） | 30 | 2/30，仅存汇总 |
| R1 | 10-13 | 最大分辨率残差告警、300 秒限时 | 30 | 3/30，10 个超时 |
| R2 | 10-13 | 取消超时、部分质量错误告警继续 | 30 | 3/30 |
| A1 | 10-14 | 带宽/分辨率、单轮/局部第二轮 | 12 | 10 组分割诊断，5 个构带失败；无 STEP |
| A2 | 10-14 | fixed/adaptive/legacy 细分策略 | 36 | 32 构带完成、14 探针 completed；无 STEP |
| V2 | 10-14 | v0.2 解析三场 | 5 | 5/5 |
| R3 | 10-14 | v0.2 全标签传播、单阶段核心 | 30 | 2/30 |
| V3 | 10-15 | v0.3 默认两阶段参数解析三场 | 5 | 2/5 |
| R4 | 10-15 | v0.3 DualBrep 两阶段核心 | 30 | 1/30 |
| V4 | 10-15 | 解析场分割/实例化：65 vs 257、边邻接 vs 顶点邻接 | 4 | 257 下面核心全部正确；顶点邻接下边数全部正确 |
| E0 | 10-16 | 工程与输入校验 | 30 | e-UDF 参考误差 ≤1.4e-7 |
| E1/E4/E5 | 10-16 | S30 257 基线；边带邻接、传播、τ_V、过滤、阈值、顶点自适应消融 | 30 × 15 组 | 基线拓扑诊断 8/30 |
| E2 | 10-16 | S8 完整流程及几何消融；解析场 | 8 + 5 | S8 5/8；解析场 5/5 |
| E3 | 10-16 | 真值注入阶段隔离 | 30 × 3 组 | 普通 5、真值顶点 5、真值几何 4 |
| — | 10-16 | 全局 513³ 窄面代理 | 4 | 4 例都未得到正确关联 |
| E7 | 10-16 | 独立 H100 | 100 | 拓扑诊断 30/100、STEP 19/100 |
| E8 | 10-16 | 相关扰动；单模型 MLP 神经场 | 8 × 4 幅度；2 | 0.001 时 STEP 3/8；神经场 0/2 |
| I1 | 10-18 | 三项改进；开发集 5 组、H100b 2 组 | 130 × 5、100 × 2 | H100b：38/100（原方法 19/100） |

## 4. 阶段一（10-12—10-15）：早期批次与消融

### 4.1 预处理与早期验证

P0 首十个模型不补选：4 通过、6 拒绝（多实体 3、退化边 2、容差超限 1）。四份通过的输出共 800,281 个查询位置。00000003 因 CAD SDF 吞吐改用验证后的 `mesh_fast`，没有改变采样或两个 UDF 后端。这些是输入范围拒绝，不能算作拓扑重建错误。

S0 证明一个简单真实 CAD 能完成全流程。但该样例与后续 30 模型重叠，不是额外的独立成功。P0 四份原始数据也保留着；它们与 30 模型数据的采样种子、后端不同，不能仅按模型 ID 当成可删除的重复数据。

阶段一的初始 SDF 网格为 65³，必要时重提到 129³（分别为 274,625 与 2,146,689 个空间位置）。新版分割在三角形中心图上进行，表面节点 5,560–172,528 个，中位数 43,590；不能把约 20 万预处理点等同于分割节点数。R3、R4 两轮的 30 份 SDF 网格逐顶点、逐三角形一致。

### 4.2 四轮 30 模型批次

| 批次 | 核心/传播 | 质量和预算策略 | 成功 | 墙钟分钟 | 成功模型 |
|---|---|---|---:|---:|---|
| R1 | 旧细分、竞争界线、端口 | 300 秒限时，最大分辨率 SDF 残差告警 | 3/30 | 22.09 | 07、61、102 |
| R2 | 旧细分、竞争界线、端口 | 无超时，部分几何质量改告警，结构约束保留 | 3/30 | 60.67 | 07、22、102 |
| R3 | 固定中心图、全标签集合、单阶段核心 | 边带 0.025、顶点带 0.04，无旧端口阶段 | 2/30 | 27.64 | 07、102 |
| R4 | 固定中心图、全标签集合、两阶段核心 | 0.005/0.01，最小分量 10，均值下限 0.003 | 1/30 | 26.61 | 102 |

- **R0** 只保留 2/30 与失败计数：20 个 `surface_error`，另有方向、细分、超时、端口/稳定性/曲线错误。完整结果被 R1 覆盖，无法追补逐样本轨迹或运行时间。
- **R1** 放行轻微残差后，00000102 新增成功，说明轻微残差不必阻断重建，但不意味着所有几何偏差都无害。
- **R1 与 R2** 成功数都是 3，但样例不同：R1 的 61 在 R2 失败，22 则在 R2 成功。R2 同时改变了超时、稳定性阻断和部分质量策略，R1→R2 不是单变量改进。
- **R3、R4** 沿用 continuation 告警策略，成功模型仍需查看 `quality_warnings.json`。
- **耗时**：各轮耗时包含 oracle 提面和机器负载差异，终止阶段也不同，不能直接当作同等质量下的加速比。

首个阻断错误（每模型只统计第一个，不代表其他错误不存在）：

| 批次 | 错误 | 数量 |
|---|---|---:|
| R0 | `surface_error` | 20 |
| R0 | `orientation_ambiguous` / `timeout` | 2 / 2 |
| R0 | `band_resolution_limit` / `port_ambiguous` / `unstable_face_seeds` / `curve_field_failed` | 各 1 |
| R1 | `timeout` | 10 |
| R1 | `unstable_face_seeds` | 9 |
| R1 | `band_resolution_limit` | 3 |
| R1 | `orientation_ambiguous` | 2 |
| R1 | `port_ambiguous` / `curve_field_failed` / `port_unmatched` | 各 1 |
| R2 | `port_unmatched` | 7 |
| R2 | `unexplained_band` | 5 |
| R2 | `port_ambiguous` | 4 |
| R2 | `band_resolution_limit` | 3 |
| R2 | `face_sector_ambiguous` / `interface_branch_ambiguous` | 2 / 2 |
| R2 | `curve_fit_failed` / `surface_boundary_incompatible` / `vertex_without_ports` / `mesh_resource_limit` | 各 1 |
| R3 | `edge_neighbors_unresolved` | 17 |
| R3 | `curve_support_disconnected` / `curve_support_branching` | 4 / 4 |
| R3 | `edge_coverage_unobserved` / `curve_closure_unobserved` / `edge_vertices_unresolved` | 各 1 |
| R4 | `edge_neighbors_unresolved` | 13 |
| R4 | `curve_closure_unobserved` | 6 |
| R4 | `edge_localization_failed` | 4 |
| R4 | `edge_coverage_unobserved` / `coincident_vertex_candidates` | 2 / 2 |
| R4 | `edge_vertices_unresolved` / `curve_support_branching` | 各 1 |

R3 的 28 个失败和 R4 的 29 个失败全部记在 curves 阶段。该阶段包含边身份检查、支撑定位/排序、拟合和覆盖检查，不能解释为此前拓扑都已正确。

### 4.3 A1：带宽与分辨率消融

- **设置**：四个模型（49、63、72、106），三个尺度 1/0.5/0.25，同步缩小 E/V 阈值和近带目标边长（原尺度为 0.025/0.04/0.0125）。小尺度比较单轮与局部第二轮：第二轮用第一轮两倍的阈值，并检查 1.8/2/2.2 倍下的稳定性，**这与 R4 的 DualBrep 两阶段过滤不同**。最多 14 轮、300 万三角形，不删除小核心。
- **结果**：耗时约 23.72 分钟。12 个构带任务中 7 个成功，共产出 10 组诊断；原先设想的 20 组单/双轮诊断没有全部产生。失败为 4 个 `band_resolution_limit`、1 个 `mesh_resource_limit`。10 组诊断中 6 组界线探针未抛异常。
- 63 在半尺度达到 6/6 且无采样漏面，72 在四分之一尺度达到 28/28。它们没有做曲面拟合或 STEP，不能算完整成功。

### 4.4 A2：细分策略消融

| 策略 | 构带完成 | 界线探针 completed | 主要限制 |
|---|---:|---:|---|
| fixed | 12/12 | 4/12 | 中心核心到旧顶点单所有者映射冲突、漏带 |
| adaptive | 12/12 | 5/12 | 在同一折面上细分，不能修正原几何 |
| legacy | 8/12 | 5/12 | 4 个细分停滞，部分区域无源 |

- 36 个任务全部结束，耗时约 15.87 分钟。fixed/adaptive 虽降低了构带工作量，但没有一组同时做到核心数达标、采样无漏面、无混合核心。legacy 的 63/0.5、63/0.25、72/0.25 满足这些条件，仍未做完整 STEP。三种模式还改变了投影和核心定义，质量差异不能全部归因于细分数量。
- **关键几何证据**：
  - 63 的 65³ 网格顶点最大 |SDF| 为 0.001216，三角形中心最大却是 0.013274。其薄层厚度约 0.039823，仅 1.27 个体素。
  - 离线直线边采样并扣除间距误差后，真边到旧折面网格的最小距离下界约 0.0133042，大于半尺度边阈值 0.0125，所以原折面不进入边带。**单纯细分原折面找不回这条低谷**。（真值只用于评估。）
  - 106 四分之一尺度 adaptive 的 467 个核心中，464 个少于 10 个三角形，而最大核心覆盖 12 个真值面。可见删除小分量和拆开混合大区域是两个独立问题：只删除小于 10 的分量会剩 3 个核心，仍恢复不了 16 个面。
- 旧 `core_contact_conflict` 来源于把中心核心转换成顶点单所有者，与后来的全中心图、多标签传播不是同一约束。

### 4.5 多标签单阶段（R3）与两阶段核心（R4）对照

- **R3**：普通带收集所有面标签，整个顶点域吸收面标签和实际 edge_id，禁止向普通带回传；按"面→边→顶点"实例化，按"顶点→边→面"参数化。
- **R4**：沿用 R3 的传播和几何代码，只更换核心提取和传播掩码。
  - 第一阶段保留 dE≥0.005 的区域，并做大小/均值过滤；第二阶段在父核心内保留 dE≥0.01，只有出现至少两个合格子核心才替换父核心。
  - 最终保留的核心与传播边带互斥。只有因距离阈值被排除的点进入边带，大小/均值过滤区单独记录且不传播。
  - `tau_edge=0.025` 仍用于既有的几何半径和覆盖检查。27/30 个模型实际触发了父域拆分。

| 指标 | R3 | R4 |
|---|---:|---:|
| STEP 成功 | 2/30 | 1/30 |
| 面核心采样未检出缺失/拆分/混合 | 6/30 | 5/30 |
| F/E/V 数量全部一致 | 4/30 | 1/30 |
| 边实例数中位数 | 34 | 118.5 |
| 顶点实例数中位数 | 14 | 35.5 |
| 单模型总秒数中位数 | 83.18 | 85.69 |

- 面核心采样可靠的样例：R3 为 07/22/61/77/93/102，R4 为 09/61/68/70/102。
- 顶点按 0.003 容差一对一匹配（目标共 1117 个）：R3 预测 683、匹配 649；R4 预测 1669、匹配 878。位置召回从约 58.1% 升到 78.6%，匹配精度从 95.0% 降到 52.6%，实例碎裂和重复严重。

### 4.6 关键案例

| 模型 | 证据 | 结论 |
|---|---|---|
| 00000003 | R3 208 个核心→R4 24 个，真值 25；R4 仍有 86 条边，真值 40 | 核心改善不能自动解决边实例 |
| 00000009 | R4 六个核心采样可靠，但 54 条边，真值 5 | 面核心可靠后，仍需修复带连通和闭环支撑 |
| 00000022 | R3 6/12/8 且关联全部匹配；原中心最小 dE≈0.01863，覆盖门槛 0.015；R4 变为 1/0/0 | R3 被验证采样阻断；R4 更早就漏带，不能只跳过检查 |
| 00000061 | R3 7/22/10，异常边只有 1–5 点；R4 7/16/17，报 localized=2,total=2 | 两个点都定位成功，但达不到最少三点的要求 |
| 00000077 | R3 某边 979 点靠近三条真边，收到三个面标签 | 标签收集正确，也可能来自混合边域 |
| 00000068 | R4 22 个面核心可靠，预测 92 顶点；32 个真顶点均匹配，但有重合候选 | 顶点残差小不能证明身份划分正确 |
| 00000070 | R4 14 个面核心可靠，48 边/27 顶点，真值 36/24；边只有 2 个支撑点 | 实例划分和支撑不足可以分开检验 |
| 00000035 | R4 84/881/242，真值 82/188/112 | 窄带碎片增加后续定位的工作量 |
| 00000102 | R4 第一阶段 1 核心，第二阶段拆成 3 个，恢复 2 个闭环并成功 STEP | 两阶段局部拆分有实际成功案例 |
| 00000039 | SDF 网格有 371 个壳；R3 998 个核心，R4 242 个，真值 16 | 最大残差小不能保证离散表面的拓扑正确 |

另有两个旧实现的观察（不能直接当作新版有同样错误的证明）：

- R2 的 68 曾出现支撑点拟合残差很小、段内 e-UDF 却约 0.18476 的异常，说明欠定空段和支撑排序都需要检查。
- R2 的 39 曾因保守的 `F+6S` 预算被过早拒绝；共形模板对应 `F+2S`。修正预算可消除这次过早拒绝，但不保证细分收敛。

### 4.7 解析场与测试

| 解析样例 | V2 单阶段 | V3 默认两阶段（65³） |
|---|---|---|
| box | solid_validated | edge_coverage_missing |
| cylinder | solid_validated | solid_validated |
| sphere_bore | solid_validated | edge_coverage_missing |
| split_sphere | solid_validated | edge_localization_failed |
| hollow_sphere | solid_validated | solid_validated |

- v0.2 有 36 项测试通过；v0.3 有 42 项通过，其中包括与 DualBrep 原函数比较的 100 组随机图。
- 完整 CAD 回归测试显式使用较宽的 0.025/0.05 和最小分量 1，以隔离后续几何约束；不能把 42 项通过理解为默认参数下五例全部通过。

### 4.8 阶段一结论

- 输入三场的提取在受支持的 30 模型子集上已经通过，但完整重建成功率很低。
- 两阶段过滤在部分模型上减少了碎核心、恢复了缺失核心，但在固定网格上，窄带同时让边域和顶点域更碎。正确传播全部标签不等于正确实例化边和顶点。
- 细分失败不全是算力问题：带宽过大吞面、表面几何跨过边低谷、离散图断裂、顶点域碎裂/混合都有证据。仅加点或统一调阈值不能一并解决。
- 旧版的界线探针 completed 只表示探针没抛异常，不是完整 B-rep 或 STEP 成功。
- 几何参数化能力在这一阶段几乎没有被检验，因为绝大多数模型在曲线阶段之前就停止了。

## 5. 阶段二（10-15）：失败归因与解析场补充验证

### 5.1 R4 成功率低的原因

R4 的 29 个失败**全部停在 curves 阶段**（错误分布见 §4.2），曲面拟合与 CAD 装配几乎没有被检验。这些错误反映的是**实例划分错误**，不是拟合算法本身的问题。

**主因：阈值与网格尺度失配，离散隔离带分不开相邻面。**

隔离带方案在离散图上成立，需要两个条件：

1. 边带在三角形邻接图上是**无缺口的分隔带**；
2. 每个真顶点**恰好对应一块**顶点带。

真边经过的三角形，其中心到真边的距离可达"三角形半径 \(r\) + MC 在锐边处的削角偏移 \(\delta\)"，所以边带阈值至少需要 \(w_E \gtrsim r+\delta\)。

- **65³ 网格的需求**：体素约 0.031，\(r\approx0.02\)，实测 \(\delta\ge0.0133\)（§4.4，00000063），需要 \(w_E\approx0.03\)。
- **R4 的取值**：\(w_E\) 只有 0.005 或 0.01。
- **与 DualBrep 对比**：DualBrep 的 0.005/0.01 用在 256³（体素约 0.0078）上，约为 0.64 和 1.28 个体素；这里只有 0.16 和 0.32 个体素，相对尺度小了约 4 倍。

不满足条件时出现四种离散现象，与报错一一对应：

| 离散现象 | 后果 | 对应报错与证据 |
|---|---|---|
| 边带完全为空 | 只剩 1 个面、0 条边 | `edge_coverage_unobserved`；22、63 都得到 1/0/0 |
| 边带有缺口 | 相邻面粘成一个核心，边带碎片只接触一个面 | `edge_neighbors_unresolved`；07 得到 1 面 / 64 边 |
| 一条真边碎成多段，段间没有顶点带 | 每段被当作闭环，或只有 1–2 个支撑点 | `curve_closure_unobserved`、`edge_localization_failed`；61、70 |
| 顶点带随边带一起碎裂 | 一个真顶点对应多个顶点候选 | `coincident_vertex_candidates`；68 预测 92 个顶点，真值 32 |

佐证：

- R3→R4：边带阈值从 0.025 改为 0.005/0.01 后，边实例中位数从 34 增到 118.5，顶点匹配精度从 95.0% 降到 52.6%。
- 解析场：单阶段 0.025 时 5/5，默认 0.005/0.01 时 2/5。
- 自然对照：07 与 102 拓扑相同（3 面 / 2 边 / 0 顶点）。102 走第二阶段拆分，带宽 0.01，成功；07 未拆开，带宽停在 0.005，碎成 64 段，失败。

**次因：**

1. **均匀粗网格上找不到满足所有约束的全局阈值。**
   - \(w_E\) 要小于最窄面宽度的一半，否则圆角、倒角面被吞掉。
   - \(\tau_V\) 要大于 \(w_E/\sin(\theta/2)\) 才能切断边带（\(\theta\) 为相交两边在面扇区内的夹角），又要小于最短边和最近两顶点间距的一半。

   65³ 下的下界约 0.03，已超过很多 ABC 模型的上界。A2 已证明，只细分同一张网格而不重新贴合 SDF，\(\delta\) 不会变。
2. **小分量过滤在粗网格上会删掉真实小面。** `face_filter_size=10` 是按 256³ 的三角形数设定的；在 65³ 下，约 0.07×0.07 以下的面都会被删掉。R4 中排除点最多的 39（1258）、08（908）、72（685）、57（655）、24（589）全部失败。
3. **没有实例层的检查和局部修复。** 任一实例出错整个模型就失败，失败也不会触发局部重新采样。

### 5.2 解析场补充验证（V4）

在 4 个解析场上运行同一套两阶段分割（默认 0.005/0.01、最小分量 10），统计面核心数和普通边带连通分量数，分别按"共享边邻接 / 共享顶点邻接"计算。只运行到分割与实例化，所用临时脚本未保留。

| 样例（真值 F/E） | 初始 65³（括号内为实际提面分辨率） | 257³ |
|---|---|---|
| box（6/12，V=8） | 1 个核心；三角形中心最小 dE=0.014，完全没有边带 | 6 个核心；边带分量 12 / 12；顶点带分量 8 / 8 |
| sphere_bore（2/2） | （129）1 个核心；最小 dE=0.008，没有边带 | 2 个核心（第二阶段拆分）；边带分量 2 / 2 |
| cylinder（3/2） | （129）3 个核心（第二阶段拆分，带宽 0.01）；边带分量 2 / 2 | 3 个核心（第一阶段直接分开，带宽 0.005）；边带分量 **152** / 2 |
| split_sphere（2/1） | 2 个核心（第一阶段直接分开）；边带分量 **140** / 1 | 2 个核心；边带分量 1 / 1 |

结论：

1. **复用 DualBrep 分割需要与之一致的分辨率。** 257³ 下 4 个样例的面核心全部正确；65³ 下 box 与 sphere_bore 的面分割本身就失败。
2. **两阶段分割让带宽不一致。** 一条边的带宽取决于两侧面在哪个阶段分开：第一阶段为 `dE<0.005`，第二阶段为 `dE<0.01`。
3. **边实例化用错了连通性。** 边带只要能把面隔开，其三角形之间只需共享顶点。按共享边邻接求分量，窄带会碎成大量一两个点的片段，这解释了"面核心可靠、边却多出很多"（如 09：6 个面、54 条边，真值 5 条）。改用共享顶点邻接后，上表各样例的边数和顶点数全部正确。
4. **排除区改变了 DualBrep 的语义。** DualBrep 中原因码 1–5 都是 `-1`；新流程把 3/4/5 单独作为阻断传播的排除区。

据此确定 10-16 基线：分辨率 257；面分割照搬 DualBrep；全部非核心点参与传播；边带与顶点带的连通分量按共享顶点邻接计算。

### 5.3 改进建议及后续执行情况

| 建议 | 执行情况 |
|---|---|
| 1. 默认阈值下把分辨率提到 257，做判定实验 | 完成：§5.2 与 10-16 基线（§6） |
| 2. 在边带附近局部加密，并把新顶点重新贴合 SDF | **未实现**；10-16 只做了全局 513³ 代理（§6.6），4 例都未得到正确关联 |
| 3. 单阈值边带；全部非核心点参与传播；小分量过滤按面积 | 全部非核心传播已实现（`band_propagation='noncore'`），但没有额外收益（§6.2）；按面积过滤未实现；过滤大小消融无整体改善 |
| 4. 按几何位置实例化顶点，每个顶点用自己的半径 | 10-16 的简单投票自适应无改善；10-18 实现局部顶点域与短边推断（§7），有效，但未加夹角约束 |
| 5. 用 e-UDF 追踪合并/拆分边实例 | **未实现** |
| 6. 拓扑预检和局部重试 | **未实现** |
| 7. 几何阶段：面采样 SDF 投影、边端部追踪、圆锥/圆环与 seam | 面采样投影 10-16 消融无改善（§6.3），选项已删除；边端部追踪未实现；圆锥/圆环与 seam 在 10-18 实现（§7） |

## 6. 阶段三（10-16）：257 基线、阶段隔离与鲁棒性

### 6.1 设置

- **主基线**：DualBrep 两阶段 `0.005/0.01`、最小分量 10、均值下限 0.003；MC 257³；顶点阈值 τ_V=0.02。
  - 边带、顶点带按共享顶点邻接；全部非核心节点参与传播，但原因码 3/4/5 不作为几何支撑；面分割仍按共享边邻接。
  - 新配置项的代码默认值保持旧行为，实验配置中显式开启新模式。
- **严格表面检查**：不沿用 R4 的 continuation 告警策略，所以本轮与历史成功率不是只改变一个变量的比较。
- **环境**：Python 3.12.14、OCC 7.9.0、NumPy 2.5.3、SciPy 1.16.2；单模型限时 300 秒，并发 2。
- **E0 工程修复**：B 样条 UV 更新在 NumPy 2 下批量 `solve` 的右端维度出错，兼容修复不改变数学目标和边界约束，修复前的结果另存。
- **进程结算**：

| 阶段 | 计划 | 结算 | 状态 |
|---|---:|---:|---|
| graph | 30 | 30 | complete 29、process_failed 1 |
| full | 8 | 8 | complete 8 |
| isolate | 30 | 30 | complete 29、process_failed 1 |
| holdout | 100 | 100 | complete 92、process_failed 7、timeout 1 |
| noise | 8 | 8 | complete 6、timeout 2 |

### 6.2 E1/E4/E5：S30 基线与消融

| 组别 | 有图结果 | 面诊断通过 | 图诊断通过 | 正确真边 / 目标 | 正确真顶点 / 目标 | 丢失面 | 拆分面 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 29 | 15 | 8 | 1185 / 1787 | 583 / 1098 | 120 | 19 |
| `edge_distance_tau04` | 29 | 15 | 1 | 737 / 1787 | 287 / 1098 | 120 | 19 |
| `edge_noncore_tau04` | 29 | 15 | 1 | 734 / 1787 | 289 / 1098 | 120 | 19 |
| `vertex_distance_tau04` | 29 | 15 | 4 | 1070 / 1787 | 428 / 1098 | 120 | 19 |
| `vertex_noncore_tau04` | 29 | 15 | 4 | 1070 / 1787 | 428 / 1098 | 120 | 19 |
| `vertex_distance_tau02` | 29 | 15 | 8 | 1185 / 1787 | 583 / 1098 | 120 | 19 |
| `filter3` | 29 | 14 | 8 | 1185 / 1787 | 583 / 1098 | 110 | 30 |
| `filter5` | 29 | 15 | 8 | 1185 / 1787 | 583 / 1098 | 115 | 25 |
| `filter1` | 29 | 5 | 4 | 1150 / 1787 | 569 / 1098 | 93 | 226 |
| `threshold_half` | 29 | 12 | 7 | 1030 / 1787 | 547 / 1098 | 183 | 8 |
| `no_second_pass` | 29 | 12 | 7 | 1021 / 1787 | 547 / 1098 | 189 | 7 |
| `tau01` | 29 | 15 | 3 | 760 / 1787 | 328 / 1098 | 120 | 19 |
| `tau03` | 29 | 15 | 7 | 1129 / 1787 | 510 / 1098 | 120 | 19 |
| `adaptive_vertices` | 29 | 15 | 8 | 1143 / 1787 | 580 / 1098 | 120 | 19 |
| `gt_core_labels` | 29 | 20 | 8 | 1188 / 1787 | 583 / 1098 | 116 | 0 |

组名含义：`edge`/`vertex` 为边带邻接方式，`distance`/`noncore` 为传播范围，`tau0x` 为顶点阈值；`filterN` 为最小分量；`threshold_half` 同时减半两个面阈值；`no_second_pass` 关闭第二阶段。

- **共享顶点邻接有效**：在 τ_V=0.04 下，图诊断通过数从 1 升到 4；再把 τ_V 改为 0.02，升到 8。
- **全部非核心传播没有额外收益。**
- 降低过滤大小、减半阈值、关闭第二阶段都**没有整体改善**；最小分量改为 1 时，面诊断通过数从 15 降到 5。
- `adaptive_vertices` 只实现了 v-UDF 投票聚类和"最近候选间距"半径上限，没有夹角约束，也没有提高图诊断通过数。它已被 10-18 的局部顶点域（§7.1）取代，实验脚本中的这一对照组已删除。
- `gt_core_labels` 保持原核心/隔离带掩码，只把核心标签换成最近真值面，属于真值注入诊断。原始评分存在空编号风险，表中为去除空编号后的重评分。
- 固定清单为 30 个，表内目标总数只覆盖有图结果的 29 个模型；00000116 提面失败（`surface_error`）。逐模型结果见附录 A。

### 6.3 E2：S8 完整流程与几何消融

| 模型 | 兼容修复前 | 兼容修复后（基线） | 面采样 SDF 投影 | 参数切口 CAD-only 重试 |
|---|---|---|---|---|
| 00000007 | STEP 通过 | STEP 通过 | STEP 通过 | 未运行 |
| 00000009 | surfaces / `ValueError` | surfaces / `surface_boundary_incompatible` | surfaces / `surface_boundary_incompatible` | 未运行 |
| 00000022 | STEP 通过 | STEP 通过 | STEP 通过 | 未运行 |
| 00000061 | STEP 通过 | STEP 通过 | STEP 通过 | 未运行 |
| 00000068 | STEP 通过 | STEP 通过 | cad / `cad_face_invalid` | 未运行 |
| 00000070 | cad / `cad_face_invalid` | cad / `cad_face_invalid` | cad / `cad_face_invalid` | STEP 通过 |
| 00000093 | cad / `cad_face_invalid` | cad / `cad_face_invalid` | cad / `cad_face_invalid` | `cad_face_invalid` |
| 00000102 | STEP 通过 | STEP 通过 | STEP 通过 | 未运行 |

- S8 基线 **5/8**。曲线阶段全部通过，剩下的问题转到曲面边界相容性（09）和 CAD 参数化（70、93）。
- **参数切口重试**：只旋转周期面的参数坐标系，把切口放到边界角度空隙的中间；三维顶点、曲线和支撑曲面不变，也没有放宽内核检查。它只测试 CAD 阶段，输入来自修复前的结果，不算全流程成功。这一做法在 10-18 并入了正式流程。
- **面采样 SDF 投影**：没有改善，还让 68 退化。该选项（`project_face_samples`）已于 10-18 从库中删除。
- **曲面自由度**：09 固定已恢复曲线和 0.001 边界容差，把 B 样条每方向控制点从 8 改为 12、16、24，全部失败（`surface_boundary_incompatible` 或 `surface_fit_failed`）。说明问题在参数化而不在自由度；10-18 用圆锥候选解决了该模型。
- **解析场**：5 个解析场在默认两阶段阈值、MC 257³ 下全流程 **5/5**（对比 §4.7 中 65³ 下的 2/5）。

### 6.4 E3：真值注入阶段隔离

| 组别 | 有结果模型 | STEP 通过 | 首个失败分布 |
|---|---:|---:|---|
| `full`（普通） | 29 | 5 | curves：`edge_neighbors_unresolved` 9、`curve_field_failed` 6、`curve_support_disconnected` 2、其他 3；surfaces 1；cad 2 |
| `gt_vertices` | 29 | 5 | 与普通组基本相同（`curve_field_failed` 5、`curve_support_branching` 2） |
| `gt_geometry` | 29 | 4 | postprocess：`boundary_orientation_ambiguous` 13、其他 2；surfaces：`surface_fit_failed` 5、`surface_boundary_incompatible` 3；真值构造 1；cad 1 |

- `gt_vertices` 只替换候选顶点的位置，不修复候选数量、混合域或错误拓扑。它没有提升成功率，说明失败**不在顶点坐标精度**，而在实例划分。
- `gt_geometry` 注入真值拓扑、顶点和解析线/圆，其他边由真值支撑点拟合，仍接受场检查；它是几何上界的代理，不是全部精确 CAD 曲线注入。
  - 真值构造阶段的 1 个错误（64，`curve_endpoint_moved`）在原始 JSON 中记为 vertices 阶段，这里重新归类。
  - 49 的 `IndexError` 是某个真值面没有 MC 核心支撑，不能算作拟合器的普遍失败。

### 6.5 E7：独立 H100

- 面诊断 46/100、图诊断 30/100、STEP **19/100**。8 个模型没有图结果（7 个进程失败、1 个超时），仍计入分母。配置在抽取 H100 前已冻结，没有按 H100 调参。
- **按面数分层的失败**：
  - 1–8 面层 14/25 成功，失败主要是薄板短边丢失和非平面曲面拟合失败；
  - 26 面以上两层 0/50，几乎全部失败在面分割。
- **少面数模型也会失败的原因**：难度取决于最小特征尺寸和曲面类型，而不是面数。
  - **薄板**（63、3952、7255、7332、1130）：板厚 0.011–0.040，小于顶点带直径 2τ_V=0.04，两个顶点域连成一片，厚度方向的短边消失；稳定地从 6/12/8 恢复成 6/8/4。
  - **非平面曲面**（09、2265、1812、2623、2801、5890）：含圆锥、圆环或自由曲面，当时只有平面/球/圆柱三种解析候选，平面投影 UV 的 B 样条又表示不了绕轴卷起的面。
  - 任一面或边失败，整个模型就失败；面少的模型更依赖周期面的 seam 处理。

### 6.6 全局 513³ 窄面代理

| 模型 | 513 同阈值：面/图通过，丢失面 | 513 半阈值：面/图通过，丢失面 |
|---|---|---|
| 00000049 | 否/否，19 | 否/否，19 |
| 00000054 | 否/否，70 | 否/否，62 |
| 00000106 | 否/否，0 | 否/否，0 |
| 00000116 | 否/否，24 | 否/否，0 |

只是全局提高分辨率的代理试验，**不是局部 SDF 重提面或局部重试**。部分核心被找回，但 4 例都没有得到正确的完整关联：单靠提高全局分辨率不够。

### 6.7 E8：场误差与神经拟合

| 相关扰动幅度上界 | 有图结果 | 面诊断通过 | 图诊断通过 | STEP 通过 |
|---|---:|---:|---:|---:|
| 0.0001 | 8/8 | 8/8 | 8/8 | 4/8 |
| 0.0005 | 8/8 | 8/8 | 8/8 | 4/8 |
| 0.001 | 7/8 | 7/8 | 6/8 | 3/8 |
| 0.002 | 7/8 | 7/8 | 6/8 | 0/8 |

![相关扰动实验曲线](../band_brep_reconstruction/outputs/plan_20261016/noise_robustness.svg)

- **扰动方式**：固定坐标相关的有界正弦叠加，同一位置重复查询结果一致，不是逐查询的高斯白噪声。单种子、单谱，不能给出普适的误差上限。
- **结果统计**：图评分来自独立的图级补测；STEP 成功数来自 300 秒限时批次，没跑完的幅度按失败计。09 另做了 900 秒补测，四个幅度分别失败于曲面（两次）和非流形表面（两次）。
- **结论**：扰动下拓扑诊断大体保持，几何和 CAD 阶段先失败。所以评估预测场时，不能只看面数或 MAE。
- **神经场**：07、22 各做 5000 步三通道监督 MLP 拟合（单模型拟合，不是跨模型泛化或生成模型验证）。
  - 07：MAE `[0.00179, 0.00201, 0.000093]`，失败于 curves / `curve_support_disconnected`；
  - 22：MAE `[0.00380, 0.00439, 0.00498]`，失败于 vertices / `vertex_geometry_invalid`。

### 6.8 阶段三结论与里程碑

- 保留 DualBrep 两阶段分割；共享顶点邻接和 τ_V=0.02 有明确收益。
- 统一降低过滤大小、减半阈值、关闭第二阶段、简单的顶点投票自适应、面采样投影都没有整体改善，不作为默认。
- 下一步应先解决参数化和 CAD 切缝，再处理窄面/短边的局部修复。
- **里程碑**（本阶段都未达到）：
  - M1：S8 完整流程 ≥6/8，本阶段 5/8；
  - M2/M3：S30 图诊断 ≥12/30、STEP ≥10/30，本阶段 8/30、5/30。

## 7. 阶段四（10-18）：局部顶点域、回转面与周期面 CAD 规划

### 7.1 改动内容

新行为都由配置开关控制，代码默认值关闭，原基线可原样复现。

| 改进项 | 开关 | 解决的问题 |
|---|---|---|
| 局部顶点域与短边推断 | `vertex_domains='local'`、`infer_short_edges=True` | 顶点间距小于 2τ_V 时两个顶点域合并、短边消失（典型是薄板） |
| 回转面与定端点圆弧 | `surface_candidates='revolution'`、`curve_candidates='arcs'` | 圆锥、圆环面只能退回平面投影 B 样条而拟合失败 |
| 周期面 CAD 规划 | `periodic_frame_planning=True` | 周期面参数切口落在边界上；`FixMissingSeam` 替换共享边 |

- **局部顶点域**：用 v-UDF 梯度投票得到候选顶点。每个候选的半径取下面两者中较大的一个，但不超过 τ_V：
  - 到最近候选距离的 0.45 倍；
  - 当地边带半宽的 1.5 倍，以保证能切断边带。

  带上的点只归属**最近**的候选，两个顶点域即使相接也不会连成一片。没有实现夹角约束。
- **短边推断**：两个顶点域直接相接、中间没有边带时，在两个中心之间连一条候选边。只有同时满足下面三条才接受，否则记录后放弃：
  - 沿线 e-UDF ≤ 0.002；
  - 沿线 |SDF| ≤ 0.003；
  - 对应的面恰好是两个。

  共享面超过两个时，取核心最接近线段中段的两个面，且要和第三个面明显拉开距离。
- **回转面**：基础候选（平面、球、圆柱）都不合格时，再用"轴线 + 母线截面"联合最小二乘拟合圆柱、圆锥、圆环。接受门槛不变（内部误差 ≤0.006，边界误差 ≤0.001），按"圆柱 → 圆锥 → 圆环"由简到繁选择。
- **定端点圆弧**：两端有顶点的边，额外尝试一段精确经过两端顶点的圆弧，避免 Bézier 近似带来的参数偏差。
- **CAD 规划**：只改变参数坐标系和环的起点，不改变任何三维几何。
  - 边界没绕满一周的周期面：把参数切口转到边界的角度空隙里，不再需要 seam；
  - 同轴的一串环面（圆柱、圆锥、圆环、球）：共用一个参考方向，seam 落在同一条母线上；
  - 由两条各绕一周的环围成的圆柱/圆锥面：显式构造直母线 seam，不再调用 `FixMissingSeam`。

### 7.2 数据与对照

- **开发集**：S30 + H100，共 130 个模型。调试时看过其中部分失败模型，开发集结果可能偏乐观。
- **独立集 H100b**：在配置冻结后才抽取（§2.1），H100b 上没有做任何调整。
- **对照组**：
  - `base`：新代码、全部开关关闭，结果与 10-16 完全相同（S30 图诊断 8、STEP 5；H100 图诊断 30、STEP 19）；
  - `all`：三项全开；
  - 开发集另做逐项去除：`no_vertex`、`no_surface`、`no_cad`。
- **运行条件**：并发 8、单模型限时 600 秒，所有组都没有超时；单模型耗时中位数约 60 秒，`all` 与 `base` 基本相同。

### 7.3 主要结果

| 数据集 | 原方法 STEP | 改进后 STEP | 原方法拓扑诊断 | 改进后拓扑诊断 |
|---|---:|---:|---:|---:|
| S30（开发） | 5/30 | **9/30** | 8/30 | 9/30 |
| H100（开发） | 19/100 | **32/100** | 30/100 | 35/100 |
| **H100b（独立）** | **19/100** | **38/100** | 34/100 | 42/100 |

- **没有退化**：开发集新增成功 17 个，H100b 新增 19 个，都没有原本成功、现在失败的模型。
- **几何保真**：所有成功模型的体积相对误差中位数小于 1e-4，最大双向表面距离不超过 0.0019，没有模型超过 1% 体积误差或 0.01 距离的警戒线。
- **里程碑**：S8 的 8 个模型全部通过 STEP（8/8，达到 M1 的 ≥6/8）；S30 的 STEP 为 9/30、图诊断为 9/30，仍未达到 M3 的 10/30 和 M2 的 12/30。

### 7.4 逐项去除消融（开发集）

| 组别 | S30 STEP | H100 STEP | 比 `all` 少的模型 |
|---|---:|---:|---:|
| `all` | 9 | 32 | — |
| `no_vertex` | 8 | 28 | 5 |
| `no_surface` | 7 | 27 | 7 |
| `no_cad` | 6 | 21 | 14 |
| `base` | 5 | 19 | 17 |

- 每项改进都有独立贡献，没有任何去除组能通过 `all` 失败的模型。
- **顶点改进**只影响薄板类：63、3367、3952、7255、9554。去掉它，这 5 个就恢复成 6/8/4 形态。
- **曲面改进**负责圆锥/圆环面：09、93、2265、2623、2801、8162、9242。
- **CAD 规划**贡献最大，`no_cad` 少了 14 个。其中 3641、8413 在 `base` 下原本能通过，在 `no_cad` 下失败，因为新增的圆弧和回转面产生了 `FixMissingSeam` 处理不了的周期 pcurve。所以**曲面改进必须和 CAD 规划一起启用**。

### 7.5 按面数分层（H100b）

| 面数层 | 原方法 STEP | 改进后 STEP | 原方法拓扑诊断 | 改进后拓扑诊断 |
|---|---:|---:|---:|---:|
| 1–8 | 10/25 | 15/25 | 14/25 | 17/25 |
| 9–25 | 9/25 | 17/25 | 16/25 | 16/25 |
| 26–80 | 0/25 | 4/25 | 2/25 | 7/25 |
| 81–250 | 0/25 | 2/25 | 2/25 | 2/25 |

- 9–25 面层接近翻倍。这一层原来大量模型是"拓扑对了，但卡在曲面或 CAD"，现在大多能完成。
- 26 面以上首次有了成功（6 个），但这两层仍有 37/50 失败在面分割。

**特例**：00008951 在 `base` 和 `all` 下都通过了 STEP，几何也忠实（最大表面距离 0.0008），但拓扑诊断失败：一个面积 4×10⁻⁵（约 0.006×0.006）的小平面被相邻核心吞掉了。

## 8. 当前失败原因（10-18 改进后）

按最早出错的环节归类（离线诊断判断，不是首个报错代码）：

| 最早出错环节 | S30 | H100 | H100b |
|---|---:|---:|---:|
| 预处理（提面失败、方向不确定、进程失败） | 1 | 8 | 5 |
| 面分割错误 | 14 | 46 | 46 |
| 面对、边/顶点实例错 | 6 | 11 | 6 |
| 拓扑对、曲线失败 | 0 | 1 | 3 |
| 拓扑对、曲面失败 | 0 | 2 | 1 |
| 拓扑对、CAD 失败 | 0 | 0 | 1 |
| **STEP 通过** | 9 | 32 | 38 |

- **面分割错误**（窄面被边带吞掉、窄面碎成多个核心）占剩余失败的 70% 以上，至今没有改动。在流程中表现为 `edge_neighbors_unresolved`（H100b 37 例）和 `edge_vertices_unresolved`（8 例）。
- **边/顶点实例错误**：H100b 从 14 降到 6。剩下的主要是相接顶点之间共享面多于两个、无法判断的情况（如 S30 的 78、105，H100 的 7332），按设计放弃推断。
- **拓扑正确却未完成**的 8 个（开发集 H100 3 个，H100b 5 个）：
  - 4 个卡在曲线阶段：3 个边带支撑点不足（7021、4069、9274），1 个曲线 e-UDF 校验失败（2967）；
  - 2 个是没有任何边界的单个非解析面（5890 为圆环面，691），走不到回转面候选；
  - 1 个是自由曲面（1812）；
  - 1 个是 CAD 面无效（871）。

## 9. 后续工作

1. **面分割（最高优先级）**：用 e-UDF 确认第二阶段的拆分是否真对应一条边，并在边带内部找回被吞掉的窄面核心；需要时在局部重提 SDF。全局 513³ 代理已表明，单靠提高全局分辨率不够。
2. **相接顶点共享面多于两个的情况**：现在的规则是"一律放弃"，可以用两个顶点各自的入射边方向判断，而不是只比较核心距离。
3. **曲线与曲面的剩余问题**：
   - 边端部补采样或沿 e-UDF 追踪，解决支撑点不足；
   - 单个无边界回转面和自由曲面，需要不依赖边界的初始化，或引入学习得到的面模型；
   - 一般网格 UV 与周期样条。
4. **实例层的拓扑预检与局部重试**：避免复杂模型因单个实例出错而整体失败。
5. **场鲁棒性**：在改进后的流程上重新评估扰动场和神经场；先提高场精度，并检验零集和梯度。
6. **评估**：H100b 已用于报告 10-18 结果；后续再调参前，应另外留出新的独立集。

## 10. 局限

- 主线实验都使用 CAD 生成的精确三场，改进后的流程没有重新评估扰动场和神经场。
- 拓扑诊断是最近支撑评分，不是数学认证；保真度只在通过 STEP 的模型上计算，用的是三角化采样距离；没有全域自交证明，也没有核对 STEP 读回后的全部 E–F/V–E 映射。
- 短边推断、回转面候选的阈值是固定值，没有在更大集合上做敏感性分析；局部顶点域没有夹角约束。
- 开发集参与了调试，独立结论以 H100b 为准；H100b 每层只有 25 个模型，分层比例的置信区间较宽。
- 耗时、内存和查询数是 worker 累计值，包含预处理、参考校验、评分等，不用于声称纯算法加速。
- 阶段一的部分历史输出已清理，旧实现无法用当前代码原样复现（§12）。

## 11. 附录

### 附录 A：S30 逐模型结果

"10-16 基线"与"10-18 改进"两列分别对应 10-18 的 `base` 和 `all` 组（`base` 与 10-16 主批次结果一致）。单元格为 STEP 通过或首个阻断错误；"图诊断"一列为"10-16 基线 / 10-18 改进"。

| 模型 | 真值 F/E/V | R1 限时 | R2 旧方案 | R3 多标签 | R4 两阶段 65³ | 图诊断 | 10-16 基线 | 10-18 改进 |
|---|---|---|---|---|---|---|---|---|
| 00000003 | 25/40/18 | timeout | port_unmatched | edge_neighbors_unresolved | edge_neighbors_unresolved | ✗/✗ | edge_neighbors_unresolved | edge_neighbors_unresolved |
| 00000007 | 3/2/0 | STEP | STEP | STEP | edge_neighbors_unresolved | ✓/✓ | STEP | STEP |
| 00000008 | 21/35/17 | timeout | port_unmatched | curve_support_disconnected | edge_neighbors_unresolved | ✗/✗ | edge_neighbors_unresolved | edge_neighbors_unresolved |
| 00000009 | 6/5/0 | unstable_face_seeds | curve_fit_failed | curve_support_disconnected | curve_closure_unobserved | ✓/✓ | surface_boundary_incompatible | **STEP** |
| 00000022 | 6/12/8 | band_resolution_limit | STEP | edge_coverage_unobserved | edge_coverage_unobserved | ✓/✓ | STEP | STEP |
| 00000024 | 30/73/43 | orientation_ambiguous | band_resolution_limit | edge_neighbors_unresolved | edge_neighbors_unresolved | ✗/✗ | curve_support_disconnected | edge_neighbors_unresolved |
| 00000031 | 27/51/27 | timeout | port_unmatched | curve_closure_unobserved | coincident_vertex_candidates | ✗/✗ | curve_closure_unobserved | curve_closure_unobserved |
| 00000035 | 82/188/112 | timeout | port_ambiguous | edge_neighbors_unresolved | curve_closure_unobserved | ✗/✗ | insufficient_edge_support | insufficient_edge_support |
| 00000039 | 16/16/0 | orientation_ambiguous | mesh_resource_limit | edge_neighbors_unresolved | curve_closure_unobserved | ✗/✗ | edge_neighbors_unresolved | edge_neighbors_unresolved |
| 00000049 | 76/135/98 | unstable_face_seeds | unexplained_band | curve_support_branching | edge_neighbors_unresolved | ✗/✗ | curve_field_failed | edge_neighbors_unresolved |
| 00000054 | 72/210/140 | unstable_face_seeds | unexplained_band | edge_neighbors_unresolved | edge_neighbors_unresolved | ✗/✗ | curve_support_branching | vertex_support_insufficient |
| 00000057 | 49/139/88 | timeout | unexplained_band | curve_support_branching | edge_neighbors_unresolved | ✗/✗ | curve_field_failed | edge_neighbors_unresolved |
| 00000060 | 76/217/142 | timeout | port_unmatched | edge_neighbors_unresolved | edge_vertices_unresolved | ✗/✗ | edge_neighbors_unresolved | edge_neighbors_unresolved |
| 00000061 | 7/15/10 | STEP | face_sector_ambiguous | edge_neighbors_unresolved | edge_localization_failed | ✓/✓ | STEP | STEP |
| 00000063 | 6/12/8 | port_ambiguous | port_ambiguous | curve_support_branching | edge_coverage_unobserved | ✗/✓ | curve_field_failed | **STEP** |
| 00000064 | 27/59/35 | timeout | vertex_without_ports | edge_neighbors_unresolved | curve_closure_unobserved | ✗/✗ | edge_neighbors_unresolved | edge_neighbors_unresolved |
| 00000065 | 55/139/90 | unstable_face_seeds | port_ambiguous | edge_neighbors_unresolved | edge_neighbors_unresolved | ✗/✗ | edge_neighbors_unresolved | edge_neighbors_unresolved |
| 00000068 | 22/56/32 | curve_field_failed | surface_boundary_incompatible | edge_neighbors_unresolved | coincident_vertex_candidates | ✓/✓ | STEP | STEP |
| 00000070 | 14/36/24 | unstable_face_seeds | port_unmatched | curve_support_branching | edge_localization_failed | ✓/✓ | cad_face_invalid | **STEP** |
| 00000072 | 28/78/52 | unstable_face_seeds | unexplained_band | edge_neighbors_unresolved | edge_neighbors_unresolved | ✗/✗ | curve_field_failed | edge_neighbors_unresolved |
| 00000077 | 16/36/24 | port_unmatched | port_unmatched | edge_neighbors_unresolved | edge_neighbors_unresolved | ✗/✗ | edge_neighbors_unresolved | edge_neighbors_unresolved |
| 00000078 | 21/48/29 | timeout | unexplained_band | edge_neighbors_unresolved | edge_localization_failed | ✗/✗ | edge_neighbors_unresolved | edge_neighbors_unresolved |
| 00000079 | 19/48/32 | unstable_face_seeds | port_ambiguous | edge_neighbors_unresolved | edge_localization_failed | ✗/✗ | curve_field_failed | edge_neighbors_unresolved |
| 00000093 | 18/44/24 | unstable_face_seeds | face_sector_ambiguous | edge_neighbors_unresolved | curve_closure_unobserved | ✓/✓ | cad_face_invalid | **STEP** |
| 00000102 | 3/2/0 | STEP | STEP | STEP | STEP | ✓/✓ | STEP | STEP |
| 00000103 | 13/12/0 | band_resolution_limit | band_resolution_limit | curve_support_disconnected | edge_neighbors_unresolved | ✗/✗ | curve_support_disconnected | curve_support_disconnected |
| 00000104 | 11/11/0 | timeout | band_resolution_limit | edge_neighbors_unresolved | edge_neighbors_unresolved | ✗/✗ | edge_neighbors_unresolved | edge_neighbors_unresolved |
| 00000105 | 11/26/17 | timeout | interface_branch_ambiguous | edge_neighbors_unresolved | edge_neighbors_unresolved | ✗/✗ | curve_field_failed | edge_neighbors_unresolved |
| 00000106 | 16/42/28 | band_resolution_limit | interface_branch_ambiguous | edge_vertices_unresolved | curve_support_branching | ✗/✗ | edge_vertices_unresolved | edge_vertices_unresolved |
| 00000116 | 92/108/19 | unstable_face_seeds | port_unmatched | curve_support_disconnected | curve_closure_unobserved | ✗/✗ | surface_error（提面） | surface_error（提面） |

7 个模型（24、49、54、57、72、79、105）在改进后首个报错发生了变化，多数变成 `edge_neighbors_unresolved`。原因是局部顶点域改变了边带的切分方式：去掉顶点改动（`no_vertex`）后，这 7 个都恢复为原来的报错。它们在两种配置下的拓扑诊断都未通过，失败的根源仍是面分割或实例错误。

### 附录 B：R3 与 R4 候选数量

| 模型 | R3 F/E/V | R4 F/E/V | R4 排除点数 |
|---|---|---|---:|
| 00000003 | 208/52/18 | 24/86/19 | 132 |
| 00000007 | 3/2/0 | 1/64/0 | 0 |
| 00000008 | 29/25/15 | 27/155/37 | 908 |
| 00000009 | 4/3/0 | 6/54/0 | 0 |
| 00000022 | 6/12/8 | 1/0/0 | 0 |
| 00000024 | 31/54/12 | 23/536/32 | 589 |
| 00000031 | 62/49/28 | 19/121/35 | 50 |
| 00000035 | 80/130/84 | 84/881/242 | 40 |
| 00000039 | 998/376/0 | 242/1201/0 | 1258 |
| 00000049 | 10/29/16 | 30/259/101 | 98 |
| 00000054 | 28/49/42 | 2/78/52 | 70 |
| 00000057 | 74/49/26 | 27/382/131 | 655 |
| 00000060 | 129/262/141 | 76/989/265 | 104 |
| 00000061 | 7/22/10 | 7/16/17 | 1 |
| 00000063 | 2/4/4 | 1/0/0 | 0 |
| 00000064 | 18/35/20 | 45/193/53 | 455 |
| 00000065 | 163/187/84 | 37/1017/245 | 11 |
| 00000068 | 36/74/34 | 22/100/92 | 8 |
| 00000070 | 39/34/12 | 14/48/27 | 4 |
| 00000072 | 8/15/14 | 12/196/51 | 685 |
| 00000077 | 16/34/24 | 12/88/44 | 0 |
| 00000078 | 19/46/22 | 21/286/52 | 16 |
| 00000079 | 28/27/20 | 14/487/53 | 93 |
| 00000093 | 18/56/12 | 16/273/36 | 1 |
| 00000102 | 3/2/0 | 3/2/0 | 0 |
| 00000103 | 13/12/0 | 7/58/0 | 73 |
| 00000104 | 121/73/0 | 7/116/0 | 174 |
| 00000105 | 18/22/10 | 10/72/15 | 3 |
| 00000106 | 2/20/13 | 2/37/15 | 103 |
| 00000116 | 22/27/14 | 59/772/55 | 6 |

### 附录 C：A1/A2 逐组记录

| 实验 | 模型 | 尺度 | 策略 | 核心/目标 | 漏面 | 混合核心 | 界线诊断 |
|---|---|---:|---|---|---:|---:|---|
| A1 | 00000049 | 0.25 | 构带 | — | — | — | mesh_resource_limit |
| A1 | 00000049 | 0.5 | 构带 | — | — | — | band_resolution_limit |
| A1 | 00000049 | 1.0 | single | 10/76 | 66 | 0 | unexplained_band |
| A1 | 00000063 | 0.25 | 构带 | — | — | — | band_resolution_limit |
| A1 | 00000063 | 0.5 | single | 6/6 | 0 | 0 | completed |
| A1 | 00000063 | 0.5 | two_pass | 6/6 | 0 | 0 | completed |
| A1 | 00000063 | 1.0 | single | 2/6 | 4 | 0 | completed |
| A1 | 00000072 | 0.25 | single | 28/28 | 0 | 0 | completed |
| A1 | 00000072 | 0.25 | two_pass | 28/28 | 0 | 0 | completed |
| A1 | 00000072 | 0.5 | single | 22/28 | 6 | 0 | unexplained_band |
| A1 | 00000072 | 0.5 | two_pass | 22/28 | 6 | 0 | unexplained_band |
| A1 | 00000072 | 1.0 | single | 2/28 | 26 | 0 | unexplained_band |
| A1 | 00000106 | 0.25 | 构带 | — | — | — | band_resolution_limit |
| A1 | 00000106 | 0.5 | 构带 | — | — | — | band_resolution_limit |
| A1 | 00000106 | 1.0 | single | 2/16 | 14 | 0 | completed |
| A2 | 00000049 | 0.25 | adaptive | 14/76 | 19 | 11 | completed |
| A2 | 00000049 | 0.25 | fixed | 39/76 | 19 | 11 | core_contact_conflict |
| A2 | 00000049 | 0.25 | legacy | — | — | — | band_resolution_limit |
| A2 | 00000049 | 0.5 | adaptive | 57/76 | 23 | 7 | core_contact_conflict |
| A2 | 00000049 | 0.5 | fixed | 83/76 | 23 | 0 | core_contact_conflict |
| A2 | 00000049 | 0.5 | legacy | — | — | — | band_resolution_limit |
| A2 | 00000049 | 1.0 | adaptive | 10/76 | 66 | 0 | completed |
| A2 | 00000049 | 1.0 | fixed | 10/76 | 66 | 0 | completed |
| A2 | 00000049 | 1.0 | legacy | 10/76 | 66 | 0 | unexplained_band |
| A2 | 00000063 | 0.25 | adaptive | 1/6 | 0 | 1 | no_interfaces |
| A2 | 00000063 | 0.25 | fixed | 1/6 | 1 | 1 | no_interfaces |
| A2 | 00000063 | 0.25 | legacy | 6/6 | 0 | 0 | completed |
| A2 | 00000063 | 0.5 | adaptive | 1/6 | 0 | 1 | no_interfaces |
| A2 | 00000063 | 0.5 | fixed | 1/6 | 1 | 1 | no_interfaces |
| A2 | 00000063 | 0.5 | legacy | 6/6 | 0 | 0 | completed |
| A2 | 00000063 | 1.0 | adaptive | 2/6 | 4 | 0 | completed |
| A2 | 00000063 | 1.0 | fixed | 2/6 | 4 | 0 | completed |
| A2 | 00000063 | 1.0 | legacy | 2/6 | 4 | 0 | completed |
| A2 | 00000072 | 0.25 | adaptive | 19/28 | 2 | 1 | core_contact_conflict |
| A2 | 00000072 | 0.25 | fixed | 10/28 | 7 | 1 | core_contact_conflict |
| A2 | 00000072 | 0.25 | legacy | 28/28 | 0 | 0 | completed |
| A2 | 00000072 | 0.5 | adaptive | 22/28 | 6 | 0 | unexplained_band |
| A2 | 00000072 | 0.5 | fixed | 26/28 | 26 | 0 | core_contact_conflict |
| A2 | 00000072 | 0.5 | legacy | 22/28 | 6 | 0 | unexplained_band |
| A2 | 00000072 | 1.0 | adaptive | 2/28 | 26 | 0 | unexplained_band |
| A2 | 00000072 | 1.0 | fixed | 8/28 | 26 | 0 | core_contact_conflict |
| A2 | 00000072 | 1.0 | legacy | 2/28 | 26 | 0 | unexplained_band |
| A2 | 00000106 | 0.25 | adaptive | 467/16 | 0 | 1 | core_contact_conflict |
| A2 | 00000106 | 0.25 | fixed | 21/16 | 7 | 1 | core_contact_conflict |
| A2 | 00000106 | 0.25 | legacy | — | — | — | band_resolution_limit |
| A2 | 00000106 | 0.5 | adaptive | 2/16 | 14 | 0 | completed |
| A2 | 00000106 | 0.5 | fixed | 2/16 | 14 | 0 | completed |
| A2 | 00000106 | 0.5 | legacy | — | — | — | band_resolution_limit |
| A2 | 00000106 | 1.0 | adaptive | 2/16 | 14 | 0 | completed |
| A2 | 00000106 | 1.0 | fixed | 2/16 | 14 | 0 | completed |
| A2 | 00000106 | 1.0 | legacy | 2/16 | 14 | 0 | completed |

## 12. 数据位置、复现与清理

### 12.1 阶段一（10-12—10-15）

阶段一在另一台机器上运行（数据根目录为 `/opt/data/private/yihengxu/...`）。下面的阶段一路径、链接都是当时的相对位置，当前工作区中不存在，也未纳入 git；需要时从原机器取回。

- 固定 30 模型清单：[manifest.jsonl](../band_brep_reconstruction/benchmarks/datasets30_20261013/manifest.jsonl)；原始三场：`band_brep_reconstruction/benchmarks/datasets30_20261013/fields`。
- 实验注册表：[registry.json](../band_brep_reconstruction/outputs/experiment_archive_20261015/registry.json)；清理记录：[cleanup_manifest.json](../band_brep_reconstruction/outputs/experiment_archive_20261015/cleanup_manifest.json)。
- 各批次的完整配置、时间、错误和旧源码指纹保留在 `progress.json` / `evaluation.json` / `source_sha256.json`。指纹不是旧版本的完整源码快照，不承诺当前代码能原样复现已删除实现的历史实验。

| 批次 | 成功 STEP |
|---|---|
| R1 / 00000007 | [reconstruction.step](../band_brep_reconstruction/benchmarks/datasets30_20261013/reconstruction/00000007_b33a147f86da49879455d286_step_000/reconstruction.step) |
| R1 / 00000061 | [reconstruction.step](../band_brep_reconstruction/benchmarks/datasets30_20261013/reconstruction/00000061_767e4372b5f94a88a7a17d90_step_002/reconstruction.step) |
| R1 / 00000102 | [reconstruction.step](../band_brep_reconstruction/benchmarks/datasets30_20261013/reconstruction/00000102_5ed74bccca6f4e89829bcb5e_step_002/reconstruction.step) |
| R2 / 00000007 | [reconstruction.step](../band_brep_reconstruction/outputs/datasets30_continue_20261013_225540/reconstruction/00000007_b33a147f86da49879455d286_step_000/reconstruction.step) |
| R2 / 00000022 | [reconstruction.step](../band_brep_reconstruction/outputs/datasets30_continue_20261013_225540/reconstruction/00000022_ad34a3f60c4a4caa99646600_step_002/reconstruction.step) |
| R2 / 00000102 | [reconstruction.step](../band_brep_reconstruction/outputs/datasets30_continue_20261013_225540/reconstruction/00000102_5ed74bccca6f4e89829bcb5e_step_002/reconstruction.step) |
| R3 / 00000007 | [reconstruction.step](../band_brep_reconstruction/outputs/datasets30_multilabel_20261014_182251/reconstruction/00000007_b33a147f86da49879455d286_step_000/reconstruction.step) |
| R3 / 00000102 | [reconstruction.step](../band_brep_reconstruction/outputs/datasets30_multilabel_20261014_182251/reconstruction/00000102_5ed74bccca6f4e89829bcb5e_step_002/reconstruction.step) |
| R4 / 00000102 | [reconstruction.step](../band_brep_reconstruction/outputs/datasets30_two_pass_20261015_004354/reconstruction/00000102_5ed74bccca6f4e89829bcb5e_step_002/reconstruction.step) |

阶段一当时版本的重跑与查询命令：

```bash
/miniconda/envs/brdf/bin/python band_brep_reconstruction/benchmarks/datasets30_20261013/continuous_batch.py --output /path/to/new_empty_output
/miniconda/envs/brdf/bin/python band_brep_reconstruction/benchmarks/datasets30_20261013/continuous_batch.py --output /path/to/output --status
```

**10-15 清理**：保留原始三场、筛选记录、模型清单、所有成功 STEP 及其几何结果、原始统计/错误/配置、旧消融诊断 JSON 与 CSV；v0.3.0 的全部中间数据和 v0.2.0 的 09/22/61/77/93 关键失败中间数据也保留。共删除 995 个文件（内容合计 231.53 MiB），225 个保留文件的 SHA-256 核验全部一致。删除的是早期查询录制、旧失败批次和已废弃消融的可重算数组、重复结果、Python 字节码和失效脚本等。清理后的历史输出属于分析归档，部分缓存条目对应的文件已删除，不能按完整缓存 resume；重建时应使用保留的输入并新建输出目录。

### 12.2 阶段三（10-16）

- 结果目录：`band_brep_reconstruction/outputs/plan_20261016/`
  - `main/manifest.json`：主清单与环境；
  - `main/holdout_manifest.json`：H100 清单；
  - 各阶段模型目录下的 `result.json`：逐模型配置与评分；
  - `summary.json`：结构化汇总；
  - `audit.json`：源码指纹一致性检查、STEP 存在性与 SHA-256、空编号重评分；
  - `source_snapshot/`：运行时的源码快照。
- 脚本：主驱动 `benchmarks/run_reconstruction_experiments.py` 仍在工作区。本阶段的一次性脚本已于 10-18 从工作区删除，原样副本保存在 `outputs/plan_20261016/source_snapshot/benchmarks/`（10-18 快照中也有）：
  - `validate_plan_analytic.py`（解析场全流程，已改为单元测试 `tests/test_presets.py`）、`validate_plan_refinement.py`（513³ 代理）；
  - `retry_periodic_frames.py`（切口重试，已并入 `periodic_frame_planning`）、`retry_surface_budget.py`（曲面自由度）、`complete_noise_scores.py`（扰动补测）；
  - `audit_plan_results.py`（审计）、`summarize_plan.py`（10-16 报告生成，另有副本 `outputs/plan_20261016/report_generation.py`）。

  `validate_plan_neural.py` 保留在工作区，供在改进后的流程上重测神经场。
- 主命令参数：`--data /opt/yihengxu/Datasets/abc/step --phase all --jobs 2 --timeout 300`。复现本轮须使用源码快照与记录的环境：工作区中的主驱动已删去 `adaptive_vertices` 对照组和面采样 SDF 投影组（`project_face_samples` 选项已从库中删除）。

### 12.3 阶段四（10-18）

- 结果目录：`band_brep_reconstruction/outputs/plan_20261017/`
  - `dev_{all,base,no_vertex,no_surface,no_cad}/`：开发集各组，每个模型目录下有 `result.json`、`process.json` 和成功时的 STEP；
  - `h100b_{all,base}/`：独立集；
  - `holdout_b_manifest.json`：H100b 清单；
  - `summary.json`：结构化汇总，含逐模型结果与成功/退化翻转列表；
  - `source_snapshot/`：运行时的源码快照。
- 批量脚本：`benchmarks/run_plan_20261017.sh`；汇总脚本：`benchmarks/summarize_plan_20261017.py`（运行后才写成，不在源码快照里）。
- `all` 组的配置已固化为库中的 `recommended` 预设（`bandbrep/config.py::PRESETS`，命令行 `--preset recommended`）；实验脚本的五个变体由该预设派生，逐项核对与实际运行的配置完全一致。
- 测试：运行时 52 项通过。10-18 整理后为 54 项：新增 `tests/test_presets.py`（预设加载规则，以及预设下五个解析场完整 STEP，约 30 秒）。
- 调试过程中的探针输出 `probe_all/`、`probe_all2/`（约 308 MB）和 `plan_20261016/final_driver_smoke07/` 未被报告引用，已删除。
- 复现时使用相同数据根目录和新的空输出目录，例如：

```bash
python benchmarks/run_reconstruction_experiments.py --data /opt/yihengxu/Datasets/abc/step \
  --output <新目录> --phase pipeline --variant all \
  --holdout-manifest outputs/plan_20261017/holdout_b_manifest.json --sets holdout --jobs 8 --timeout 600
```
