# 三场 B-rep 重建实验汇总（2026-10-12—2026-10-15）

本报告统一整理三场预处理、早期单模型验证、四轮有记录的 30 模型重建、两组四模型消融及两轮解析场验证。覆盖前的严格表面检查实验仅保留汇总，单独列出，不补造逐模型结果。当前实现为 v0.3.0。

## 1. 结论

- 输入三场的数据提取已经在受支持的 30 模型子集通过，但完整重建成功率仍低：限时重跑 3/30、取消超时的旧方案 3/30、多标签单阶段 2/30、DualBrep 两阶段核心方案 1/30。多轮共享同一批模型，不能相加成独立样本。
- 两阶段过滤在部分模型上减少碎核心、恢复缺失核心；固定网格上的窄带同时增加了边和顶点域碎裂。正确传播全部标签不等于正确实例化边/顶点。
- 细分失败不全是算力问题。带宽过大吞面、表面几何跨过边低谷、离散图断裂、顶点域碎裂/混合均有证据；仅加点或统一调阈值不能一并解决。
- 旧版界线提取、端口匹配与新版完整集合传播语义不同。旧消融中的 completed 只是探针未抛异常，不是完整 B-rep 或 STEP 成功。
- 几何参数化的通用能力尚未得到充分评估：新版绝大多数模型在曲线阶段前置条件或支撑点处理时停止。不能据这些结果宣称复杂曲面拟合已验证。

## 2. 数据、场和评估口径

30 模型来自 `/opt/data/private/yihengxu/Datasets/abc/step/0000`，按目录顺序、每目录第一份 STEP，取通过范围检查且源哈希不重复的前 30 个；预筛 120 个候选。它们来自 13 个源 CAD 文档分组，不是 ABC 随机样本或整体覆盖率估计。原始 STEP 位于外部 Datasets，本次没有改动。

三场共用 query_points，通道为 `[SDF,E-UDF,V-UDF]`，坐标域 `[-1,1]^3`；SDF 内负外正。目标边不含生成阶段 seam，保留不同面之间的相切/共面边及必要顶点；seam 在 CAD 后处理补建。距离以归一化长度计。

30 模型共保存 **6,006,802** 个三场查询位置；每模型 200,006–200,793 个。重建通过连续 CAD oracle 在新增位置查询场值，并与保存数据及 OCC reference 抽查一致性，不是仅对 NPZ 插值，也不是生成场噪声实验。拓扑 ID 仅参与离线评分。抽查最大参考误差：SDF `0.00019115607`、E-UDF `5.84838438e-08`、V-UDF `5.89968214e-08`；这是有限抽查结果，不是全域保证。

初始 SDF 网格 65³，必要时重提到 129³，分别为 274,625 与 2,146,689 个空间位置。新版分割在三角形中心图上进行，表面节点为 5,560–172,528 个，中位数 43,590；不能将约 20 万预处理点等同于分割节点数。两轮新版的 30 份 SDF 网格逐顶点、逐三角形一致。

只有 solid_validated 表示 CAD 装配和 STEP 读回校验通过；仍需检查场保真警告和离线关联。候选图完成、面数一致、局部低残差、界线探针 completed 均不等同于拓扑正确。面匹配每核心最多 128 点；域混合最多每域 512 点的最近真值支撑诊断；结果受采样和薄层最近面误配影响。超时按超时记录，不认定为算法必然失败。

## 3. 实验总表

| 编号 | 日期 | 实验 | 样本/任务 | 结果 | STEP 范围 |
|---|---|---|---:|---|---|
| P0 | 10-12 | 最初十模型预处理 | 10 | 4 通过、6 范围拒绝 | 未做 |
| S0 | 10-13 | 00000007 单模型完整验证 | 1 | 1 成功，约 155.38 秒 | 完整 |
| R0 | 10-13 | 严格 SDF 检查初轮（已被覆盖） | 30 | 2/30，仅余汇总 | 完整尝试 |
| R1 | 10-13 | 最大分辨率残差告警、300 秒限时重跑 | 30 | 3/30，10 超时 | 完整尝试 |
| R2 | 10-13 | 取消超时、部分质量错误告警继续 | 30 | 3/30 | 完整尝试 |
| A1 | 10-14 | 带宽/分辨率、单轮/局部第二轮 | 12 | 10 组分割诊断、5 个构带失败 | 无 |
| A2 | 10-14 | fixed/adaptive/legacy 细分策略 | 36 | 32 构带完成、14 探针 completed | 无 |
| V2 | 10-14 | v0.2 解析三场 | 5 | 5/5 | 完整 |
| R3 | 10-14 | v0.2 完整多标签传播、单阶段核心 | 30 | 2/30 | 完整尝试 |
| V3 | 10-15 | v0.3 默认两阶段参数解析三场 | 5 | 2/5 | 完整尝试 |
| R4 | 10-15 | v0.3 DualBrep 两阶段核心 | 30 | 1/30 | 完整尝试 |

## 4. 预处理与早期验证

P0 首十个模型不补选：4 通过、6 拒绝；多实体 3、退化边 2、容差超限 1。四份通过输出共 800,281 个查询位置。00000003 因 CAD SDF 吞吐改用验证后的 mesh_fast，未改变采样或两个 UDF 后端。这些是输入范围拒绝，不能归入拓扑重建错误。

S0 证明过一个简单真实 CAD 可完成全流程，但样例与后续 30 模型重叠，不是额外独立成功。P0 四份原始数据也保留，与 30 模型数据的采样种子/后端不同，不能仅按模型 ID 当成可删除重复数据。

## 5. 四轮完整批次与覆盖前汇总

| 批次 | 核心/传播 | 质量和预算策略 | 成功 | 墙钟分钟 | 成功模型 |
|---|---|---|---:|---:|---|
| R1 | 旧细分、竞争界线、端口 | 300 秒限制，最大分辨率 SDF 残差告警 | 3/30 | 22.09 | 00000007, 00000061, 00000102 |
| R2 | 旧细分、竞争界线、端口 | 无超时，部分几何质量改告警，结构约束保留 | 3/30 | 60.67 | 00000007, 00000022, 00000102 |
| R3 | 固定中心图、全标签集合、单阶段核心 | 边带 0.025、顶点带 0.04，无旧端口阶段 | 2/30 | 27.64 | 00000007, 00000102 |
| R4 | 固定中心图、全标签集合、两阶段核心 | 0.005/0.01，最小分量 10，均值下限 0.003 | 1/30 | 26.61 | 00000102 |

R0 仅保留 2/30 与失败计数：20 个 surface_error，另有朝向、细分、超时、端口/稳定性/曲线错误。完整旧结果被 R1 覆盖，不能追补逐样本轨迹或运行时间。R1 放行残差后新增 00000102 成功，说明轻微残差不必阻断重建；不意味着所有几何偏差均无害。

R1 与 R2 的成功数都为 3，但样例不同：R1 的 00000061 在 R2 失败，00000022 则在 R2 成功，不能将相同成功数解释为行为相同。

R2 同时改变超时、稳定性阻断与部分质量策略，不能将 R1→R2 视为单变量改进。R3、R4 沿用 continuation 告警策略；成功仍须查看 quality_warnings.json。多轮耗时包含 oracle 提面及机器负载差异、且终止阶段不同，不能直接作为同等质量加速比。

### 首个失败原因

| 批次 | 错误 | 数量 |
|---|---|---:|
| R0 | `surface_error` | 20 |
| R0 | `orientation_ambiguous` | 2 |
| R0 | `band_resolution_limit` | 1 |
| R0 | `timeout` | 2 |
| R0 | `port_ambiguous` | 1 |
| R0 | `unstable_face_seeds` | 1 |
| R0 | `curve_field_failed` | 1 |
| R1 | `reconstruct:timeout` | 10 |
| R1 | `reconstruct:unstable_face_seeds` | 9 |
| R1 | `reconstruct:band_resolution_limit` | 3 |
| R1 | `reconstruct:orientation_ambiguous` | 2 |
| R1 | `reconstruct:port_ambiguous` | 1 |
| R1 | `reconstruct:curve_field_failed` | 1 |
| R1 | `reconstruct:port_unmatched` | 1 |
| R2 | `curve_fit_failed` | 1 |
| R2 | `port_unmatched` | 7 |
| R2 | `unexplained_band` | 5 |
| R2 | `port_ambiguous` | 4 |
| R2 | `band_resolution_limit` | 3 |
| R2 | `face_sector_ambiguous` | 2 |
| R2 | `surface_boundary_incompatible` | 1 |
| R2 | `vertex_without_ports` | 1 |
| R2 | `interface_branch_ambiguous` | 2 |
| R2 | `mesh_resource_limit` | 1 |
| R3 | `edge_coverage_unobserved` | 1 |
| R3 | `curve_support_disconnected` | 4 |
| R3 | `edge_neighbors_unresolved` | 17 |
| R3 | `curve_closure_unobserved` | 1 |
| R3 | `curve_support_branching` | 4 |
| R3 | `edge_vertices_unresolved` | 1 |
| R4 | `edge_neighbors_unresolved` | 13 |
| R4 | `edge_coverage_unobserved` | 2 |
| R4 | `curve_closure_unobserved` | 6 |
| R4 | `coincident_vertex_candidates` | 2 |
| R4 | `edge_localization_failed` | 4 |
| R4 | `edge_vertices_unresolved` | 1 |
| R4 | `curve_support_branching` | 1 |

每模型只统计首个阻断错误，不代表其他错误不存在。R3 的 28 个失败及 R4 的 29 个失败全部记在 curves 阶段；该阶段包含边身份检查、支撑定位/排序、拟合及覆盖检查，不能解释为所有拓扑先已正确。

## 6. A1 带宽与分辨率消融

四模型 49/63/72/106，三个尺度 1/0.5/0.25，同步缩小 E/V 阈值和近带目标边长；原尺度参数为 0.025/0.04/0.0125。小尺度比较单轮与局部第二轮。最多 14 轮、300 万三角形；不删除小核心。第二轮用第一轮两倍阈值并检查 1.8/2/2.2 倍稳定性，**不同于 R4 的 DualBrep 两阶段过滤**。

A1 批次耗时约 23.72 分钟。12 个构带任务中 7 个成功，产出 10 组诊断；原先设想的 20 组单/双轮诊断并未全部产生。4 个 band_resolution_limit、1 个 mesh_resource_limit；10 组诊断中 6 组界线探针未抛异常。63 半尺度达到 6/6 且无采样漏面；72 四分之一尺度达到 28/28。它们没有执行曲面拟合或 STEP，不能写成完整成功。

## 7. A2 细分策略消融

| 策略 | 构带完成 | 界线探针 completed | 主要限制 |
|---|---:|---:|---|
| fixed | 12/12 | 4/12 | 中心核心到旧顶点单所有者映射冲突、漏带 |
| adaptive | 12/12 | 5/12 | 在同一折面上细分不能修正原几何 |
| legacy | 8/12 | 5/12 | 4 个细分停滞，部分无源区域 |

36 个任务均结束，耗时约 15.87 分钟，固定/自适应虽降低构带工作量，却没有一组同时达到目标核心数、采样无漏面、无混合核心。legacy 的 63/0.5、63/0.25、72/0.25 满足这些核心诊断条件，仍未做完整 STEP。三模式还改变投影和核心定义，质量差异不能全部归因于细分数量。

关键几何证据：63 的 65³ 网格顶点最大 |SDF| 为 0.001216，但三角形中心最大为 0.013274。其薄层厚度约 0.039823，仅 1.27 个体素。离线直线边采样及 Lipschitz 扣除间距误差后，真边到旧折面网格的最小距离下界约 0.0133042，大于半尺度边阈值 0.0125；在该点三角距离数值精度下，原折面不进入边带。单纯切细原折面无法找回这条低谷。该诊断使用真值只用于评估，未给重建引入真边引导。

106 四分之一尺度 adaptive 的 467 个核心中 464 个少于 10 个三角形，而最大核心采样覆盖 12 个真值面：删小分量和拆混合大区域是两个问题。只做 `<10` 删除会剩 3 个核心，仍不能恢复 16 个面。旧 core_contact_conflict 来源于中心核心转成顶点单所有者，与后来的全中心图、多标签传播并非同一约束。

## 8. 多标签与两阶段核心的直接对照

R3 普通带收集所有面标签，整个顶点域吸收面标签和实际 edge_id，禁止向普通带回传；面→边→顶点实例化，顶点→边→面参数化。R4 沿用该传播及几何代码，只更换核心提取和传播掩码。

R4 第一阶段保留 dE≥0.005 并做大小/均值过滤；第二阶段在父核心内保留 dE≥0.01，仅出现至少两个合格子核心才替换父核心。最终保留核心与传播边带互斥。只有因距离阈值排除的点进入边带，大小/均值过滤区单独记录且不传播。tau_edge=0.025 仍供既有几何半径/覆盖检查使用，不再定义 R4 的传播带。27/30 模型实际触发父域拆分。

| 指标 | R3 | R4 |
|---|---:|---:|
| STEP 成功 | 2/30 | 1/30 |
| 面核心采样未检出缺失/拆分/混合 | 6/30 | 5/30 |
| F/E/V 数量全部一致 | 4/30 | 1/30 |
| 边实例数中位数 | 34 | 118.5 |
| 顶点实例数中位数 | 14 | 35.5 |
| 单模型总秒数中位数 | 83.18 | 85.69 |

R3 核心采样可靠样例为 07/22/61/77/93/102；R4 为 09/61/68/70/102。顶点 0.003 容差一对一匹配：R3 预测 683、匹配 649；R4 预测 1669、匹配 878；目标总数 1117。位置召回从约 58.1% 升到 78.6%，匹配精度从 95.0% 降到 52.6%；实例碎裂和重复仍然严重。

## 9. 跨实验关键案例与归因

| 模型 | 证据 | 可支持的结论 |
|---|---|---|
| 00000003 | R3 208 核心→R4 24，真值25；R4仍有86边，真值40 | 核心改善不能自动解决边实例 |
| 00000009 | R4六个核心采样可靠，但54条边，真值5 | 面核心可靠后仍需修复带连通与闭环支持 |
| 00000022 | R3 6/12/8，关联均匹配；原中心最小dE≈0.01863，覆盖门槛0.015；R4变1/0/0 | R3是验证采样阻断；R4已更早漏带，不能只跳检查 |
| 00000061 | R3 7/22/10，异常边仅1–5点；R4 7/16/17，定位报localized=2,total=2 | 后者两个点都定位成功，实际不足最小三点要求 |
| 00000077 | R3某边979点靠近三条真边，收到三个面标签 | 标签收集正确也可能来自混合边域 |
| 00000068 | R4面核心22个可靠，预测92顶点；32真顶点均匹配，仍有重合候选 | 顶点残差小不能证明身份划分正确 |
| 00000070 | R4面核心14个可靠，48边/27顶点，真值36/24；边只有2支撑点 | 实例划分和支撑不足可单独测 |
| 00000035 | R4 84/881/242，真值82/188/112 | 窄带碎片增加后继定位工作量 |
| 00000102 | R4第一阶段1核心，第二阶段拆成3，恢复2闭环并成功STEP | 两阶段局部拆分有实际成功案例 |
| 00000039 | 371壳的SDF网格；R3 998核心，R4 242，真值16 | 轻微最大残差不能保证离散表面拓扑正确 |

R2 的 68 曲线曾出现支持点拟合残差很小、段内 E-UDF 约0.18476的异常；欠定空段与支持排序都需检查。该失败属于旧几何实现，不能直接当作新版同样错误的证明。R2 的39曾因保守 `F+6S` 预算过早拒绝；当前共形模板对应 `F+2S`。修正预算可消除该次过早拒绝，却不保证细分收敛。

## 10. 解析场与测试记录

| 解析样例 | V2单阶段 | V3默认两阶段 |
|---|---|---|
| box | solid_validated | edge_coverage_missing |
| cylinder | solid_validated | solid_validated |
| sphere_bore | solid_validated | edge_coverage_missing |
| split_sphere | solid_validated | edge_localization_failed |
| hollow_sphere | solid_validated | solid_validated |

V2五例5/5；V3默认0.005/0.01在65初始分辨率下2/5（圆柱、空心球）。v0.2有36项测试通过；v0.3有42项通过，其中对原DualBrep函数比较100组随机图。完整CAD回归测试显式使用较宽0.025/0.05及最小分量1来隔离后续几何约束，不能将42项通过写成默认参数五例全通过。全部成功解析STEP保留。

## 11. 后续实验优先级

1. 在09/61/68/70等核心可靠样例上，保持全标签传播语义，独立检查边域碎裂、顶点域重复与混合。先诊断原图接触位置、场投影票和实际支撑，避免无条件合并邻近实例。
2. 联合评估表面几何精度、局部采样间距与带宽。先检查旧折面是否进入低谷；需要时局部重提SDF，再加密阈值不确定区域。区分几何修正和同一折面的细分。
3. 从原始带域分层选独立覆盖探针，投影后检查双向覆盖；分别报告支撑不足、投影失败和排序失败。对少量支撑先补采样，不能通过放松闭合/邻面约束伪造STEP。
4. 用继承的网格邻接及经场验证的局部连接改善曲线排序；固定k近邻图断裂不必然证明真边断裂，增大k也不能无条件连接独立闭环。
5. 增加逐实例诊断模式，前置条件可靠的边/面可独立评估；整体拓扑和CAD不完整仍按模型失败。之后再做带噪/生成场测试和复杂曲面评估。

## 12. 全部逐模型完整批次结果

| 模型 | 真值F/E/V | R1限时 | R2旧方案 | R3多标签 | R4两阶段 |
|---|---|---|---|---|---|
| 00000003 | 25/40/18 | timeout | port_unmatched | edge_neighbors_unresolved | edge_neighbors_unresolved |
| 00000007 | 3/2/0 | STEP通过 | STEP通过 | STEP通过 | edge_neighbors_unresolved |
| 00000008 | 21/35/17 | timeout | port_unmatched | curve_support_disconnected | edge_neighbors_unresolved |
| 00000009 | 6/5/0 | unstable_face_seeds | curve_fit_failed | curve_support_disconnected | curve_closure_unobserved |
| 00000022 | 6/12/8 | band_resolution_limit | STEP通过 | edge_coverage_unobserved | edge_coverage_unobserved |
| 00000024 | 30/73/43 | orientation_ambiguous | band_resolution_limit | edge_neighbors_unresolved | edge_neighbors_unresolved |
| 00000031 | 27/51/27 | timeout | port_unmatched | curve_closure_unobserved | coincident_vertex_candidates |
| 00000035 | 82/188/112 | timeout | port_ambiguous | edge_neighbors_unresolved | curve_closure_unobserved |
| 00000039 | 16/16/0 | orientation_ambiguous | mesh_resource_limit | edge_neighbors_unresolved | curve_closure_unobserved |
| 00000049 | 76/135/98 | unstable_face_seeds | unexplained_band | curve_support_branching | edge_neighbors_unresolved |
| 00000054 | 72/210/140 | unstable_face_seeds | unexplained_band | edge_neighbors_unresolved | edge_neighbors_unresolved |
| 00000057 | 49/139/88 | timeout | unexplained_band | curve_support_branching | edge_neighbors_unresolved |
| 00000060 | 76/217/142 | timeout | port_unmatched | edge_neighbors_unresolved | edge_vertices_unresolved |
| 00000061 | 7/15/10 | STEP通过 | face_sector_ambiguous | edge_neighbors_unresolved | edge_localization_failed |
| 00000063 | 6/12/8 | port_ambiguous | port_ambiguous | curve_support_branching | edge_coverage_unobserved |
| 00000064 | 27/59/35 | timeout | vertex_without_ports | edge_neighbors_unresolved | curve_closure_unobserved |
| 00000065 | 55/139/90 | unstable_face_seeds | port_ambiguous | edge_neighbors_unresolved | edge_neighbors_unresolved |
| 00000068 | 22/56/32 | curve_field_failed | surface_boundary_incompatible | edge_neighbors_unresolved | coincident_vertex_candidates |
| 00000070 | 14/36/24 | unstable_face_seeds | port_unmatched | curve_support_branching | edge_localization_failed |
| 00000072 | 28/78/52 | unstable_face_seeds | unexplained_band | edge_neighbors_unresolved | edge_neighbors_unresolved |
| 00000077 | 16/36/24 | port_unmatched | port_unmatched | edge_neighbors_unresolved | edge_neighbors_unresolved |
| 00000078 | 21/48/29 | timeout | unexplained_band | edge_neighbors_unresolved | edge_localization_failed |
| 00000079 | 19/48/32 | unstable_face_seeds | port_ambiguous | edge_neighbors_unresolved | edge_localization_failed |
| 00000093 | 18/44/24 | unstable_face_seeds | face_sector_ambiguous | edge_neighbors_unresolved | curve_closure_unobserved |
| 00000102 | 3/2/0 | STEP通过 | STEP通过 | STEP通过 | STEP通过 |
| 00000103 | 13/12/0 | band_resolution_limit | band_resolution_limit | curve_support_disconnected | edge_neighbors_unresolved |
| 00000104 | 11/11/0 | timeout | band_resolution_limit | edge_neighbors_unresolved | edge_neighbors_unresolved |
| 00000105 | 11/26/17 | timeout | interface_branch_ambiguous | edge_neighbors_unresolved | edge_neighbors_unresolved |
| 00000106 | 16/42/28 | band_resolution_limit | interface_branch_ambiguous | edge_vertices_unresolved | curve_support_branching |
| 00000116 | 92/108/19 | unstable_face_seeds | port_unmatched | curve_support_disconnected | curve_closure_unobserved |

### R3与R4候选数量

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

## 13. 消融逐组记录

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

## 14. 数据位置、成功STEP和复现

- 固定30模型清单：[manifest.jsonl](../band_brep_reconstruction/benchmarks/datasets30_20261013/manifest.jsonl)；原始三场：`band_brep_reconstruction/benchmarks/datasets30_20261013/fields`。
- 实验注册表：[registry.json](../band_brep_reconstruction/outputs/experiment_archive_20261015/registry.json)；完整清理记录：[cleanup_manifest.json](../band_brep_reconstruction/outputs/experiment_archive_20261015/cleanup_manifest.json)。
- 完整配置、时间、错误和旧源码指纹保留在各批次 progress.json / evaluation.json / source_sha256.json。指纹不是旧版本完整源码快照，不承诺当前代码可原样复现已经删除实现的历史实验。

| 批次 | 成功STEP |
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

当前版本重跑（新输出目录）与查询：

```bash
/miniconda/envs/brdf/bin/python band_brep_reconstruction/benchmarks/datasets30_20261013/continuous_batch.py --output /path/to/new_empty_output
/miniconda/envs/brdf/bin/python band_brep_reconstruction/benchmarks/datasets30_20261013/continuous_batch.py --output /path/to/output --status
```

## 15. 文件清理

保留原始三场数据、筛选记录、模型清单、所有成功STEP及其几何结果、原始统计/错误/配置、旧消融诊断JSON与CSV。v0.3.0全部中间数据和v0.2.0的09/22/61/77/93关键失败中间数据保留。

已删除 995 个文件，文件内容合计 231.53 MiB，回收已分配文件块约 239.61 MiB。225 个保留数据/成功STEP文件的 SHA-256 核验全部一致；执行记录见 cleanup_manifest.json。

删除早期查询录制、旧失败批次/已废弃消融的可重算数组、完成任务的PID/作业副本、重复结果和评分副本、Python字节码、一次性等待脚本与失效诊断脚本。重复Markdown报告归并到本文，原文压缩归档在historical_report_texts.json.gz；完成作业配置另有压缩归档。

清理后的历史输出属于分析归档，部分artifacts哈希条目对应文件已删除，不能按完整缓存resume；原run.json不改写，以保留当时证据。需要重建时应使用保留输入和当前版本新建输出。最新完整实验目录可以按其原代码/配置缓存规则继续检查。
