# 非精确扫描件条件下的配对 VAE / Flow 训练、退化边处理与后处理协同方案

设计稿，2026-11-21。本文档针对“非精确扫描件 / 粗网格重建 B-Rep（重建结果不强贴输入表面，在近表面三维空间恢复干净 CAD）”这一核心目标，系统分析“仅在 Flow 训练阶段对条件点云加噪”的架构与几何局限，论证为什么不能直接用近表面点池 `N` 替换输入点云，给出**严格 NaTex 对齐 + 离线预处理多档退化变体 + `Q00Zero` 零集补全 + 配对 VAE / Flow + 后处理协同**的完整实施方案，并统一明确 **退化边（Degenerate Edge）** 的 V-UDF 采样策略及预处理/后处理代码级改造清单。

关联文档：
- 边优先扫描件重建方案：[Post1121.md](Post1121.md)
- 边优先优化方案：[MCShellCenterline.md](MCShellCenterline.md)
- Pipeline 层优化分析（结合前沿工作）：[Opt.md](Opt.md)
- 数据生产规范：[Pre.md](Pre.md)
- 上游 ShapeVAE 与零集编码：[Hunyuan3D VAE.md](../../Hunyuan3D/docs/VAE.md)

---

## 1. 背景与问题定义

当前两套仓库的分工如下：
- **上游生成网络（`Hunyuan3D/hy3dshape`）**：基于 `EdgeVertexShapeVAE`（Q00 / Q00Zero）与 `FieldDiT` 条件 Flow，从输入几何条件编码 $z_g$，生成并解码近表面带内的连续 E/V UDF（$\hat d_E, \hat d_V$）。
- **下游后处理（`DualBrep/band_brep_reconstruction`）**：采用边优先流水线（`scout → wireframe → split_mesh / fast_faces → solid`），在近表面三维带内提取边骨架、顶点球壳与参数曲线，再投影到输入网格划分面域、拟合曲面并装配为 STEP。

为了让生成网络学会从**含噪、偏移或圆钝的扫描件 / 粗网格** $C_{\text{noisy}}$ 恢复出**干净、锐利、易装配为有效 B-Rep 的 CAD E/V 场** $B^*$，需要打通从条件退化、VAE 编解码、Flow 生成到下游面拟合的全链路一致性。

---

## 2. 为什么不能“只改 Flow 条件加噪、冻结干净 VAE”

若仅在 Flow 训练阶段对控制条件点云施加 $\le 0.02$ 的扰动，同时沿用干净点云训练的冻结 VAE 与干净目标潜变量 $z_f^{\text{clean}}$，会在当前代码架构下触发五个硬性瓶颈：

### 2.1 VAE 解码器几何强耦合导致「训练-推理失配」（最核心）
在 [shapevae.py](../../Hunyuan3D/hy3dshape/hy3dshape/models/autoencoders/edge_vertex/shapevae.py#L193-L200) 中，联合解码查询路径为：
```python
h = self.geo_query_decoder.forward_features(queries=x.float(), latents=memory_g)
q = self.geo_feature_to_field_query(h)
field = self.field_query_decoder(query_embeddings=q, latents=memory_f)
```
- `field_query_decoder` **不直接接收坐标 `x`**，所有空间定位信息均经由 `memory_g` 生成的几何特征 $h_g(x)$ 传递。
- 在 [pipeline.py](../../Hunyuan3D/hy3dshape/hy3dshape/inference/pipeline.py#L325-L327) 推理时，`prepare_decode(geo.z_g, z_f_pred, geo.anchors)` 喂入的是**退化条件编码出的 `memory_g_noisy`**。
- 干净 VAE 中的 $z_f^{\text{clean}}$ 仅与 `memory_g_clean` 协同工作。即便 Flow 完美预测出 $z_f^{\text{clean}}$，将 `(memory_g_noisy, z_f_clean)` 配对解码仍会因 $h_g(x)$ 分布偏移导致 E/V 场扭曲、模糊或被拉回噪声表面；而 Flow 在 [flow_matching.py](../../Hunyuan3D/hy3dshape/hy3dshape/models/diffusion/flow_matching.py#L54-L62) 中仅在 latent 空间做速度匹配，无法感知或补偿解码器失配。

### 2.2 Token / Anchor 空间对齐被破坏（`FieldDiT` 硬约束）
- [field_dit.py](../../Hunyuan3D/hy3dshape/hy3dshape/models/denoisers/field_dit.py#L168-L188) 通过**逐 token 通道拼接** `torch.cat((z_t, z_g), -1)` 注入几何条件，并对两者共享同一组 `SurfaceRoPE(anchor_xyz)`（`rope_coord_scale = 100.0`）。
- [shapevae.py](../../Hunyuan3D/hy3dshape/hy3dshape/models/autoencoders/edge_vertex/shapevae.py#L124-L138) 中，目标 $z_f^{\text{clean}}$ 的 4,096 个 token 严格绑定在干净点云经 FPS 采出的 `anchors_clean` 顺序与位置上。
- 若对加噪点云重新跑 FPS 编码 `z_g_noisy`，4,096 个 anchor 的顺序和 3D 位置与 $z_f^{\text{clean}}$ 完全错位，逐 token 拼接直接失效；若训练时复用干净 `anchor_idx`，推理面对真实扫描件时仍须独立跑 FPS（且 FPS 天然偏好向外凸出的噪点），同样存在严重的训练-推理分布落差。

### 2.3 偏移上限 `0.02` 与查询带宽 `0.02` 的几何冲突及双重外推
- **管壳截断**：按 [Post1121.md §1](Post1121.md) 的几何预算，默认带宽 $w = 0.02$、管壳/球壳半径 $r = 0.005$、细格步长 $h = 0.003$ 时，保证完整管壳和一层细格角点不越界的保守条件为：
  $$\delta + r + \sqrt{3}h \le w \implies \delta \lesssim 0.0098$$
  偏移达到 `0.01 ~ 0.02` 时，真实边/顶点管壳会被带边界裁断（`valid=False`），导致漏边、断链或球壳残缺。
- **解码器 OOD 外推**：现有 `veudf` 空间监督点仅覆盖距**干净表面** `0.03` 以内（VAE 训练默认截到 `0.02`，见 [Pre.md §4](Pre.md#L59)）。若输入点云偏离干净表面 `0.02`，推理在退化点云 `0.02` 带内查询时，查询点距干净表面可达 `0.04`，落入解码器未训练区。
- **`presence` 漏判**：[pipeline.py](../../Hunyuan3D/hy3dshape/hy3dshape/inference/pipeline.py#L73-L77) 仅在输入表面 `S0` 上采样估计存在性，大偏移下小顶点可能被误判为 `absent`。

### 2.4 点级白噪声的法向陷阱与 Domain Gap
- `encode_geometry` 同时接收 3D 坐标 `pos` 与法向 `normal`。
- 纯逐点 i.i.d. 随机加噪存在两难：**不改法向**会泄露精确干净 CAD 法向导致过拟合作弊；**对抖动点云重算 PCA 法向**（当抖动幅度接近 `0.02`，远大于 81,920 点的局部点距 `~0.005`）会使法向退化为纯噪声，破坏预训练 ShapeVAE 几何主干。
- 真实扫描件/粗网格的核心退化是**空间相关的低频偏移**（壁厚偏差、锐边圆化、网格简化收缩、局部缺测），而非高频白噪声。

### 2.5 下游面拟合的隐藏硬门限
- 在 [surface_fitting.py](file:///opt/data/private/yihengxu/DualBrep/band_brep_reconstruction/bandbrep/surface_fitting.py#L493-L561) 中，尽管 `scan_offset_tolerance = 0.02` 放宽了 `.max()` 上限，但 **90%/95% 分位点内点误差和输入网格 `provider.sdf(q)` 仍硬卡在 `2 * surface_tolerance = 0.006`**。当干净边界曲线与系统性偏移的扫描内点相差超过 `0.006` 时，会触发 `surface_fit_failed` 或 `surface_boundary_incompatible`。

---

## 3. 为什么不能直接用近表面点池 `N` 替换输入点云

在重训路径 A 的 VAE 时，**不建议**直接拿磁盘上已有的 `near_surface` 点池（`N`）随机替换 `S0` 作为编码器输入：

1. **`N` 是双侧对称的三维厚点雾，存在零均值捷径**：
   - 见 [sampling.py](../../Hunyuan3D/veudf/src/veudf/preprocess/sampling.py#L314-L343)：`near_surface` 由 80% 沿法向**等概率 `±1` 双侧扰动**（`[0, 0.002]` 30%、`[0.002, 0.01]` 40%、`[0.01, 0.03]` 30%）与 20% 各向同性高斯扰动组成，厚度达 `0.04 ~ 0.06`。
   - 其局部空间均值精确落在干净 CAD 表面 `S0` 上，`geo_cross`（20:1 聚合）极易通过局部平均直接滤回干净表面，无法学会处理真实扫描件的**单侧系统性偏移**。
2. **FPS Anchor 退化为双层体壳**：对三维厚雾跑 FPS 会优先选出内外最外缘极值点，与推理期单层二维扫描表面的 anchor 分布完全不同。
3. **补法向陷入两难**：`N` 未保存 `normals` 与 `triangle_ids`；从最近干净面补法向会泄露干净 CAD 法向，对三维点雾做 PCA 估算法向则极不稳定。

---

## 4. 推荐方案：路径 A（离线预处理配对退化资产 + 严格 NaTex 对齐 + `Q00Zero`）

核心思路：
1. **严格保留 NaTex 对齐契约**（与 [Post1121.md](Post1121.md) 和 [shapevae.py](../../Hunyuan3D/hy3dshape/hy3dshape/models/autoencoders/edge_vertex/shapevae.py#L124-L138) 一致）：随机采样主路中，`encode_geometry` 与 `encode_field` 共享同一批退化二维表面采样点 `(pos_noisy, normal_noisy)` 和同一 `SamplingPlan`，而 `field_udf` 是在 `pos_noisy` 坐标处针对**干净 CAD $B^*$** 用 `FeatureUDFOracle` 预先算好的精确 OCC 真值距离。
2. **统一甜点档阈值（全量重处理数据集）**：选定 **$\delta_{\max}=0.015$、推理带宽 $w_{\text{infer}}=0.026$、预处理监督带 `band_max=0.040`、截断半径 `udf_truncation=0.05`（不变）**——兼顾 90%+ 真实扫描与 AI 粗网格误差、管壳零截断（$0.015 + 0.005 + 0.0052 \le 0.026$）、解码器零 OOD 外推（$0.026 + 0.015 \approx 0.040$），且不稀释近零集数值精度。
3. **同域规范化 + 精简三维监督池 + `Q00Zero` 零集补全**：预处理先用 `ShapeUpgrade_UnifySameDomain` 消除无意义的同曲面拼缝（如两块半圆锥/半圆柱拼成的整面）；Decoder 监督池瘦身为**三维近表面分层池 + 三维边管/顶点球近零集池 + 精确零集池**；`Q00Zero` 零集支路补足偏移表面缺失的 `dE ≈ 0` 后验观测。

### 4.1 整体数据流

```mermaid
flowchart TB
    subgraph PRE["离线数据预处理（全量重建 v4 资产，mmap 落盘）"]
        STEP["源 STEP 读取与归一化 [-0.9, 0.9]^3<br/>ShapeUpgrade_UnifySameDomain 同域合并<br/>放行合法退化边（排除出 E-UDF）"]
        DEG["连续参数随机组合生成 K 个二维退化表面变体<br/>v00 .. v0(K-1)（低频位移 + 平滑 + 简化/MC + 微配准偏置）"]
        SAMP["每变体在退化表面采样 M=163,840 点<br/>(pos_noisy, normal_noisy) + FeatureUDFOracle 精确 [s_clean, dE*, dV*]"]
        POOL["共享三维监督池（精简 3 池，零表面细分）<br/>near_surface_band (<=0.04) + 3D edge_tube/vertex_ball + edge/vertex_zero"]
        STEP --> DEG --> SAMP
        STEP --> POOL
    end

    subgraph TRAIN["VAE 训练与 Latent 缓存（纯 mmap 抽取 + 零重标在线增强）"]
        DRAW["条件抽取（20% 干净 S0 + 80% 退化 v_k）<br/>无放回抽取 81,920 点 + 在线非均匀/遮挡权重 + 微法向抖动 / Normal Dropout"]
        GE["encode_geometry<br/>（浅层 LoRA / 低 LR 微调）"]
        ZG["z_g & anchors [4096]"]
        MG["memory_g"]

        ZERO["Q00Zero 零集支路（仅后验）<br/>干净 vertex_zero + edge_zero"]
        FE["encode_field（严格共享 SamplingPlan）<br/>Q = field_query_proj(z_g)<br/>K/V = 同批 (pos, normal, udf*) + 零集增量"]
        ZF["z_f [4096, 64]<br/>与 z_g / anchors 100% 逐 token 对齐"]
        MF["memory_f"]

        DEC["联合解码器（新增零初始化坐标残差）<br/>q = geo_feature_to_field_query(h_g(x)) + field_coord_proj(PE(x))"]
        SUP["共享三维监督池（|s_clean| <= 0.04）<br/>+ 10% 未进 Encoder 的 S0 / 退化表面剩余点"]
    end

    FLOW["FieldDiT Flow 训练<br/>缓存多变体 (z_g, anchors, z_f_mu, z_f_logvar)<br/>条件: (z_g, anchors) → 目标: z_f"]

    SAMP -.-> DRAW
    POOL -.-> SUP
    POOL -.-> ZERO
    DRAW --> GE --> ZG --> MG --> DEC
    DRAW --> FE
    ZG --> FE
    ZERO --> FE --> ZF --> MF --> DEC
    DEC --> SUP
    ZG -.-> FLOW
    ZF -.-> FLOW
```

### 4.2 离线预处理、阈值甜点与精简点池设计

#### (1) 已敲定的甜点档核心阈值
| 参数 | 取值 | 作用与约束依据 |
|---|---|---|
| 退化偏移上限 $\delta_{\max}$ | **`0.015`**（180 mm 零件约 1.5 mm；局部极值容许至 `0.018`） | 覆盖 90%+ 工业/消费级扫描与 AI 生成粗网格的圆化、漂移误差，同时避免薄壁穿带 |
| 推理近表面查询带 $w_{\text{infer}}$ | **`0.026`**（`0.025 ~ 0.026`） | 严格满足管壳不截断条件 $\delta_{\max} + r + \sqrt{3}h = 0.015 + 0.005 + 0.0052 = 0.0252 \le 0.026$；查询量相对原 `0.02` 仅增 `~1.25×` |
| 预处理监督带 `band_max` | **`0.040`** | 覆盖推理极限距离 $w_{\text{infer}} + \delta_{\max} \approx 0.041$，确保解码器几乎零 OOD 外推，且满足 `bounding(0.9) + band_max = 0.94 < 1.0` |
| UDF 截断半径 `udf_truncation` | **`0.05`**（保持不变） | 满足 `udf_truncation > band_max + label_tolerance`，**完全不稀释近零集数值分辨率**，沿用原距离编码与 loss 分桶 `[0.001, 0.005, 0.02]` |
| 后处理容差联动 | `band_width = 0.026`，`scan_offset_tolerance = 0.018` | 为局部尾部极值留 `0.003` 余量，并与 `surface_fitting.py` 分位门限联动 |

#### (2) 资产目录与精简点池结构（新数据集根 v4）
```text
assets/<asset_id>/
  condition/                  # 干净参考网格（经 ShapeUpgrade_UnifySameDomain 规范化后生成）
    vertices.npy / triangles.npy / meta.json
  encoder_surfaces/           # 专供 Encoder 抽取的二维表面池（严格 NaTex 对齐）
    clean/                    # 干净 S0（M_clean = 262,144 或 393,216）
      points.npy / normals.npy / fields.npy / saturated.npy
    v00/ ... v0(K-1)/         # K 个预生成组合退化二维表面变体（每变体 M = 163,840）
      points.npy              # float32 [M, 3]，退化二维表面采样坐标 pos_noisy
      normals.npy             # float32 [M, 3]，退化网格自身真实面/顶点法向（绝不用干净法向）
      fields.npy              # float32 [M, 3]，FeatureUDFOracle 精确标注的 [s_clean, min(dE*, 0.05), min(dV*, 0.05)]
      saturated.npy           # bool    [M, 2]，E/V 截断饱和标记
      mesh/{vertices.npy, triangles.npy}   # val 必存（用于下游闭环评测），train 可选
      meta.json               # 退化组合参数、随机种子、实测偏移统计（rms/p95/max）、闭合检查
  pools/                      # 共享三维 Decoder 监督池（砍掉耗时的表面 proposal_fragments 细分）
    near_surface/             # 三维近表面 4 层分层池（|s_clean| <= 0.040）
    edge_neighborhood/        # 纯三维边管近零集池（直接绕干净边采三维距离分层，覆盖 E=0.005 管壳两侧）
    vertex_neighborhood/      # 纯三维顶点球近零集池（直接绕干净顶点采三维距离分层，覆盖 V=0.005 球壳两侧）
    edge_zero/                # 精确边零集（65,536 等弧长点，供 Q00Zero + Decoder）
    vertex_zero/              # 全部必要顶点零集（供 Q00Zero + Decoder）
  evaluation/ / audit/ / topology.json / meta.json
```

* **精简 3 池采样配额建议（Decoder 共享监督池）**：
  1. **`near_surface`（约 350k–400k 点，覆盖 `|s_clean| <= 0.040`）**：
     - 80% 法向双向扰动拆为 **4 层**：`[0, 0.002, 0.01, 0.025, 0.04]`，层权重 **`[0.25, 0.35, 0.25, 0.15]`**（60% 守住 `[0, 0.01]` 核心区，15% 守住 `[0.025, 0.04]` 防 OOD 缓冲区，避免扩带稀释内层密度）；
     - 20% 各向同性高斯扰动（`std ∈ [0.001, 0.005, 0.012, 0.025]`，过滤 `|s_clean| <= 0.040`）。
  2. **`edge_neighborhood` / `vertex_neighborhood`（纯三维管/球近零集池，各约 350k / 200k 点）**：
     - **彻底去掉原先占 ~70% 预算、耗时的二维表面三角细分 `proposal_fragments`**；
     - 直接以 `edge_zero` / `vertex_zero` 为中心，按三维法截面/球壳半径分层采样（`feature_radii = (0.002, 0.005, 0.01, 0.02, 0.04)`，权重 `(0.20, 0.28, 0.27, 0.17, 0.08)`），接受 `|s_clean| <= 0.040` 且位于 `[-1, 1]^3` 内的点，把 `E=0.005` / `V=0.005` 等值管壳与球壳两侧打透。
  3. **`edge_zero` / `vertex_zero`（精确零集）**：沿用现行配额（`edge_zero = 65,536`，`vertex_zero = 全部必要顶点`）。
  4. **少量 `s=0` 表面监督**：不另开重型表面特征池，直接把训练期未进入 Encoder 的 `clean/surface` 或当前变体 `v_k` 剩余点按约 `10%` 配额混入 Decoder 查询即可。

#### (3) 组合式表面级退化与防单一性设计
为避免离线变体过于单一，每个变体**不是从 4 种固定退化里单选其一，而是从连续参数区间随机采样、组合叠加以下退化算子**，并将总偏移严格控制在 $\delta_{\max} \le 0.015$（局部极端尾部 $\le 0.018$）：

| 退化算子模块 | 参数连续采样范围 | 物理对应误差 |
|---|---|---|
| **低频光滑 3D 位移场** | 随机 3–8 个 RBF / 低频正弦基，幅值 `U(0.001, 0.012)` | 扫描标定漂移、热/结构形变 |
| **法向系统性偏置** | 局部低频调制法向膨胀/收缩 `U(-0.006, +0.006)` | 薄壁壁厚测量误差、表面反射偏置 |
| **Taubin / Laplacian 平滑** | 随机迭代 `0 ~ 8` 步（或按曲率加权平滑） | 扫描仪 PSF 与泊松重建导致的**锐边圆化** |
| **网格简化 / 多分辨率 MC** | 随机二选一或组合：QEM 简化保留 `10% ~ 60%` 面片，或体素化 + MC（分辨率连续抽 `72³ ~ 144³`） | 粗网格三角化、3D 生成大模型（Hunyuan3D 等）体素阶梯与收缩 |
| **微小刚体/缩放配准残差** | 极小旋转 `U(0°, 0.4°)` + 各向异性微缩放 `U(0.992, 1.008)`（受总偏移 `<= 0.015` 约束） | 扫描件配准残差与标尺微误差 |
| **曲率相关高频微扰** | 沿法向微抖动 `σ ∈ [0.0003, 0.0015]`（高曲率/锐边附近略大） | 真实扫描仪掠射角与棱边高噪声特性 |

* **变体数 $K$ 与防单一的“零 OCC 重标”在线多样性放大**：
  * **离线数量**：万级大数据集 `train` 每资产 **`K = 6 ~ 8`**（小数据集 `< 2,000` 资产可提至 `K = 10 ~ 12`），`val` 每资产固定 **`K = 3`** 档确定性变体（`low: <=0.005` / `mid: 0.005~0.010` / `high: 0.010~0.015`）+ `clean`。
  * **每变体点数**：`M = 163,840`（`2 × 81,920`），单变体约 `6.5 MB`，6 个变体合计约 `~40 MB / 资产`。
  * **训练期 4 重零重标在线多样性放大**（不改变点坐标，因此**完全不需要重算 OCC UDF**）：
    1. **随机子集与随机 FPS**：每轮从 `163,840` 点中无放回抽不同 `81,920` 子集，FPS 产生不同 4,096 anchor 集合；
    2. **在线非均匀密度 / 局部视角遮挡抽样**：按随机空间权重（或随机剔除 1–2 个小锥角/凹槽区域）加权抽取 `81,920` 点，模拟真实扫描密度不均与局部遮挡缺测；
    3. **在线法向微扰 + 15% Normal Dropout**：对已加载的 `normal_noisy` 施加 `1° ~ 5°` 随机锥角偏转或 15% 概率置零（法向不影响标量 UDF 真值）；
    4. **全局随机刚体旋转增强（可选）**：对输入点、法向、零集与所有查询点同步施加同一随机 SO(3) 旋转（欧氏距离完全旋转不变）。

#### (4) 真值 `[s_clean, dE*, dV*]` 统一标注方式
* 全部直接复用 [veudf/src/veudf/preprocess/distance.py](../../Hunyuan3D/veudf/src/veudf/preprocess/distance.py#L32-L117) 的 **`FeatureUDFOracle`**（`udf_truncation = 0.05`）：仅在边覆盖球与顶点 `0.05` 球内调用 OCC 精确求解，远场直接赋值 `0.05` 并标 `saturated=True`；`s_clean` 由干净网格 `MeshSDFOracle` 精确求解。
* 整个预处理链路 100% 精确一致，无近似误差。

### 4.3 模型侧与训练抽样改动

#### (1) `SurfacePoolDataset` 抽样策略（零在线退化开销）
- **编码器输入抽取**：
  - `train`：以 `20%` 概率抽 `encoder_surfaces/clean`，以 `80%` 概率随机抽 `encoder_surfaces/v00..v0(K-1)`；从 `M` 点中按均匀或非均匀/遮挡权重无放回抽取 `81,920` 个 `(pos, normal, udf*)`，**同时送入 `encode_geometry` 和 `encode_field`**（严格维持 `geo_pos == field_pos` 与共享 `SamplingPlan`）。
  - 在线施加轻微法向锥角微扰（`1° ~ 5°`）与 `15%` Normal Dropout（同步作用于 `geo_normal` 和 `field_normal`）。
  - **`Q00Zero` 零集支路**：始终从 `pools/edge_zero` 与 `pools/vertex_zero` 抽取精确干净零集（预算 20,480），不随退化变体改变。
- **Decoder 监督点抽取**：
  - 从精简后的 3 类共享三维监督池抽取（`supervision_band_width = 0.040`，覆盖 `|s_clean| <= 0.040`），并按约 `10%` 配额混入当前抽中表面池（`clean` 或 `v_k`）中**未进 Encoder 的剩余点**。

#### (2) VAE 解码器坐标残差与几何主干微调
- **坐标直连残差**：在 [shapevae.py](../../Hunyuan3D/hy3dshape/hy3dshape/models/autoencoders/edge_vertex/shapevae.py#L199-L200) 中，将
  ```python
  q = self.geo_feature_to_field_query(h)
  ```
  升级为
  ```python
  q = self.geo_feature_to_field_query(h) + self.field_coord_proj(self.pe(x.float()))
  ```
  新投影层 `field_coord_proj` 采用**零初始化**，可无缝从现有 Q00 / Q00Zero checkpoint 热启，防止查询坐标定位过度受限于含噪 `memory_g`。
- **几何编码器轻量微调**：对 `geo_input_proj` 与浅层 `geo_cross` 开放 LoRA 或低学习率微调（例如主学习率的 `0.1×`），使预训练 ShapeVAE 几何编码器适应退化表面与圆化法向。

### 4.4 Latent 缓存、Flow 训练与下游后处理协同

1. **多变体 Latent 缓存**：扩展 [latent_cache.py](../../Hunyuan3D/hy3dshape/hy3dshape/data/latent_cache.py#L59-L125)，遍历每个资产的 `[clean, v00, ..., v0(K-1)]` 分别缓存 `(z_g_raw, anchor_xyz, z_f_mu, z_f_logvar, variant_id, offset_tier)`。`stats_f` / `stats_g` 仍仅在 `train` split 上拟合。
2. **拓扑敏感加权 Flow Loss**：按每个 `anchor` 处对应输入点的最小 `dE*` / `dV*` 分桶加权 Flow Matching 损失，把生成容量集中在决定拓扑成败的管壳与交汇区。
3. **重建导向选优与微调（Best-of-N / DPO）**：推理期对同一扫描输入采样 $K = 4\text{--}8$ 组 $z_f$，以 `band_brep_reconstruction` 的快速 `wireframe` 指标（`unresolved` 数量、闭环完整度、骨架连通性）选优；进一步可将后处理验收通过/失败构造为偏好对，对 Flow 做轻量 DPO / RFT 微调。
4. **下游 `band_brep_reconstruction` 协同适配**：
   - 将默认查询带调至 `band_width = 0.026`、`scan_offset_tolerance = 0.018`；
   - 在 [surface_fitting.py](../band_brep_reconstruction/bandbrep/surface_fitting.py#L493-L561) 中，将硬编码的 `2 * surface_tolerance = 0.006` 内点 90%/95% 分位阈值与 `scan_offset_tolerance` 联动，避免偏移 `0.006 ~ 0.018` 的有效干净面被误拒；
   - 对偏移较大的扫描面，提高干净三维边界曲线对平面/柱面/锥面参数的约束优先级，扫描内点主要用于法向定向与离群过滤；
   - 在 `scout` 阶段结合三维体素格点最小值辅助判定 E/V `presence`，避免偏移下在输入表面 `S0` 误判顶点缺失；
   - 中长期可结合 [Opt.md](Opt.md) 同步引入解析面求交反哺线框、边界驱动曲面实例化以及双场干净 SDF 输出。

> **备选消融（方案 A2：解耦 `encode_field` K/V）**：若仅做概念验证、不想为退化变体标注 UDF，可在 `shapevae.py` 中临时放开 `encode_field` 坐标一致断言，让 `encode_field` 的 K/V 直接读干净 `S` 池、Query 仍用 `geo.z_g`。正式重处理数据集仍以本节的**离线组合退化 + `FeatureUDFOracle` 精确标注 + 严格 NaTex 对齐**为主方案。

---

## 5. 退化边（Degenerate Edge）处理与 V-UDF 采样策略

### 5.1 概念厘清：什么是退化边？

在 B-Rep（尤其是 OpenCASCADE）中，**退化边**（`BRep_Tool.Degenerated(edge) == True`）是指在二维 UV 参数域中具有非零参数长度的 pcurve，但映射到三维空间后**缩成单个点（三维长度为 0、无 3D 曲线）**的拓扑边。工业 CAD 中常见三类退化边：

1. **完整解析旋转面极点**：如完整圆锥的尖顶（$v = v_{\text{apex}}, u \in [0, 2\pi]$）、球冠极点（$v = \pm\pi/2, u \in [0, 2\pi]$）、完整球体南北两极。
2. **部分解析旋转面极点**：如被平面切开的半圆锥尖顶、切分球面过极点（此时极点连着两条及以上普通面间边，但在 UV 域中仍跨 $u \in [u_1, u_2]$）。
3. **退化 B-spline / 扫掠片收缩边**：四边张量曲面将一条参数边界的所有控制点重合为同一 3D 顶点（如三边面用四边 B-spline 表示）。

### 5.2 核心问题：是否应把 seam 加入 E-UDF？圆锥顶点 / 球面极点是否纳入 V-UDF？

#### (1) 强烈不建议将 seam 作为普通边加入 E-UDF
1. **几何不可辨识（无限多解），破坏生成网络训练**：如 [DualBrep.md §5.1](../legacy_dualbrep/DualBrep.md#L637) 所述，完整圆柱、圆锥、球面和圆环的 seam 位于 $u \in [0, 2\pi)$ 的哪条母线/经线上，三维外形完全相同。将 seam 加入 E-UDF 会迫使 VAE/Flow 在光滑无特征的旋转面上猜测随机母线，导致光滑曲面上的预测 E-UDF 出现模糊低谷或虚假断续骨架。
2. **引入大量伪顶点，污染 V-UDF**：seam 另一端落在底面圆环上的交点是没有几何特征的任意圆环锚点（现行 [cad.py:312-320](../../Hunyuan3D/veudf/src/veudf/preprocess/cad.py#L312-L320) 已规范化移除），强行保留会在光滑圆环上引入随机三度伪顶点。
3. **破坏下游拓扑假设**：[boundary_cycles.py](../band_brep_reconstruction/bandbrep/boundary_cycles.py)、[edge_first_embedding.py](../band_brep_reconstruction/bandbrep/edge_first_embedding.py) 与 [occ_assembly.py:552](../band_brep_reconstruction/bandbrep/occ_assembly.py#L552) 均假设逻辑边分隔**两个不同面**（`len(uses)==2 and uses[0][0]!=uses[1][0]`）。
4. **技术上无需借助 seam**：[edge_first_grid.py:185](../band_brep_reconstruction/bandbrep/edge_first_grid.py#L185) 的 `scout` 筛块本就是 `E` / `V` 通道取并集（`.any(1)`），且圆锥尖顶与球面极点完全由单环解析面拟合闭式反推。

#### (2) 圆锥顶点、球面极点是否纳入 V-UDF 采样？
按**“是否有普通面间边连接到该点”**统一判定，不引入孤立顶点特判：

| 几何情形 | 是否纳入 `V-UDF` | 原因 |
|---|---|---|
| **球冠极点 / 完整球南北极** | **否** | 三维上是光滑球面内部的普通点，无几何棱角，纯属参数原点；纳入 `V-UDF` 模型无法从几何定位。 |
| **完整圆锥尖顶**（仅连退化边与 seam，无普通面间边） | **否** | ① [veudf/preprocess/pipeline.py:146-149](../../Hunyuan3D/veudf/src/veudf/preprocess/pipeline.py#L146-L149) 强制校验全局不变量 **`dE <= dV + label_tolerance`**，孤立尖顶处 $d_V = 0, d_E > 0$ 会直接报 `quality_failed: dE > dV`；② 下游 `edge_first_graph.py` 不允许 degree=0 孤立顶点进入逻辑线框图；③ 单环圆锥面在 `surface_fitting.py` 解析拟合后，尖顶直接由闭式公式 $P_{\text{apex}} = O - \frac{R}{\tan\alpha}\mathbf{a}$ 反推，精度更高且抗扫描磨圆。 |
| **半圆锥尖顶 / 切分球极点 / 退化 B-spline 收缩点**（连着 $\ge 2$ 条普通面间边） | **是（自动纳入）** | 本身就是真实普通边（如半圆锥的两条母线边）的交汇端点，天然满足 $d_E \le d_V$，现行 [cad.py:297-322](../../Hunyuan3D/veudf/src/veudf/preprocess/cad.py#L297-L322) 顶点筛选逻辑**自动保留**。 |

> 排除退化边（与 seam 同级）后，全圆锥的目标语义与圆柱一致（`has_edges=True, has_vertices=False`），现有 `veudf` 与 `hy3dshape` 已原生支持空 V 通道。

### 5.3 分场景恢复方案（含斜切圆锥与斜圆锥）

#### 场景 A：完整圆锥尖顶 / 斜切正圆锥 / 球冠极点 / 完整球体（无普通边连到极点）
* **拓扑切分**：底面闭合环将网格切分为 `Face 0（底面）` 与 `Face 1（单环圆锥/球冠侧面）`。
* **几何恢复**：[surface_fitting.py:338-423](../band_brep_reconstruction/bandbrep/surface_fitting.py#L338-L423) 的 `revolution_candidates` 已支持任意三维轴向的单环正圆锥拟合（输出 `origin` $O$、`axis` $\mathbf{a}$、`angle` $\alpha$、`radius` $R$），尖顶解析解为：
  $$v_{\text{apex}} = -\frac{R}{\sin\alpha}, \qquad P_{\text{apex}} = O - \frac{R}{\tan\alpha}\,\mathbf{a}$$
* **OCC 装配**：仿照 [occ_assembly.py:267-301](../band_brep_reconstruction/bandbrep/occ_assembly.py#L267-L301) 已实现的 `make_spherical_cap()`，对称新增 **`make_conical_cap()`**：
  1. 触发条件：`patch.kind == 'cone' and len(incident) == 1 and incident[0].closed`（**不硬限底环必须是圆**，改用通用 `make_pcurve` 生成底环 2D 曲线 $(u(t), v(t))$，从而同时支持水平底圆与斜平面/曲面斜切得到的椭圆/闭合自由底环！）；
  2. 求尖顶参数 $v_{\text{apex}} = -R / \sin\alpha$（OCC 中 $\alpha \in (-\pi/2, \pi/2)\setminus\{0\}$ 可正可负，故需以 `apex_above = v_apex > v_ring_anchor` 确定 `(low, high)`）与 3D 顶点 `apex = BRepBuilderAPI_MakeVertex(surface.Value(0, v_apex)).Vertex()`；
  3. 用 1 次 `Geom_BSplineCurve`（底环锚点 $(u_0, v(0))$ 到尖顶 $(u_0, v_{\text{apex}})$）构造直线母线 `seam`，挂两条 UV pcurve（$u = u_0$ 与 $u = u_0 + 2\pi$）；
  4. 构造 `apex` 处的退化边 `degen`（`builder.Degenerated(degen, True)`，两端绑定 `apex`，挂 pcurve $(u_0, v_{\text{apex}}) \to (u_0 + 2\pi, v_{\text{apex}})$）；
  5. 按 `apex_above` 组装闭合 wire 并按 `orientation_point` 与 `patch.normal` 校验面朝向。
  * *扩展*：若全圆锥侧面带有侧孔（`len(wraps) == 1` + 若干 `holes`），可在 `plan_periodic_band` 中同样把孔环附着到 `make_conical_cap` 建出的面上。

#### 场景 B：半圆锥 / 切分球等部分旋转面（极点连着普通面间边）
* **同域规范化说明**：两个半圆锥/半圆柱拼成的整面，在预处理阶段已由 `ShapeUpgrade_UnifySameDomain` 规范化合并为场景 A 的单个整面；只有真正的半圆锥（另一侧为切割平面而非同源圆锥）会进入场景 B。
* **拓扑与几何**：极点是多条普通面间边的交汇点，天然存在于 E-UDF 与 V-UDF 中，被正常恢复为共享顶点 `occ_vertices[vid]`。
* **OCC 装配**：在组装 `cone` / `sphere` 的普通 wire 时，检查环上相邻两条 coedge `(u_prev, u_next)`：若共享 3D 顶点落在解析极点（$|v - v_{\text{pole}}| < \text{tol}$）且前后 pcurve 方位角存在跳变（$|u_{\text{prev,end}} - u_{\text{next,start}}| > 10^{-5}$），在两者之间自动插入一条绑定到该共享 `occ_vertices[vid]` 的退化边（`Degenerated=True`，pcurve 为直线段 $(u_{\text{prev,end}}, v_{\text{pole}}) \to (u_{\text{next,start}}, v_{\text{pole}})$）。

#### 场景 C：一般 B-spline 自由曲面上的退化边与真正斜圆锥（Oblique Cone）
* **普通三边退化 B-spline 片**（退化顶点连着 $\ge 2$ 条普通面间边）：预处理排除退化边出 E-UDF、自动保留收缩顶点进 V-UDF；后处理 [surface_fitting.py:434-497](../band_brep_reconstruction/bandbrep/surface_fitting.py#L434-L497) 直接拟合裁剪张量 B-spline 曲面。
* **真正斜圆锥（非旋转面的孤立尖点退化 B-spline）**：
  * **首版预处理策略（覆盖 ~99% 退化边数据）**：在 `cad.py` 中按退化边所属面的曲面类型 `BRepAdaptor_Surface(face).GetType()` 精准放行——放行 `GeomAbs_Cone`、`GeomAbs_Sphere` 退化边及有 $\ge 2$ 条普通边相连的退化边；暂拒极罕见的“非解析曲面 + 无普通边相连的孤立退化尖点”（记为 `unsupported_isolated_degenerate_freeform`）。
  * **后续扩展支持（仍无需把尖顶加入 V-UDF）**：任意斜圆锥/椭圆锥的所有切平面均严格过尖顶 $P_{\text{apex}}$（$\mathbf{n}_i^\top P_{\text{apex}} = \mathbf{n}_i^\top \mathbf{p}_i$），可在 `surface_fitting.py` 中对单环非解析面通过 3 变量鲁棒线性最小二乘直接由 `p_core` 闭式求出 $P_{\text{apex}}$，再构造 degree-(3, 1) 直纹退化 B-spline 锥面 $S(u, v) = (1 - v) C(u) + v P_{\text{apex}}$ 精确恢复。

---

## 6. 详细分仓库改造计划（可直接照表实施）

按执行依赖顺序分为四个阶段，跨 `Hunyuan3D/veudf` → `Hunyuan3D/hy3dshape` → `DualBrep/band_brep_reconstruction` 三个包推进：

### Phase 1：数据预处理改造（`Hunyuan3D/veudf` → 同步至 `DualBrep/veudf`）

| 任务 | 目标文件与位置 | 具体改动内容 |
|---|---|---|
| **1.1 甜点档配置升级** | [veudf/preprocess/config.py](../../Hunyuan3D/veudf/src/veudf/preprocess/config.py) & [configs/surface_pools.json](../../Hunyuan3D/veudf/configs/surface_pools.json) | 设 `band_max = 0.04`、`udf_truncation = 0.05`、`feature_radii = (0.002, 0.005, 0.01, 0.02, 0.04)`、`feature_weights = (0.20, 0.28, 0.27, 0.17, 0.08)`；新增退化变体配置：`num_train_variants = 6`、`num_val_variants = 3`、`variant_surface_points = 163840`、`max_degradation_offset = 0.015`、`unify_same_domain = True` |
| **1.2 同域规范化合并** | [veudf/preprocess/cad.py](../../Hunyuan3D/veudf/src/veudf/preprocess/cad.py#L139-L166)（`read_normalized` 后） | 引入 `ShapeUpgrade_UnifySameDomain(normalized, True, True, False)`（`SetLinearTolerance(cfg.geometry_tolerance)`、`SetAngularTolerance(1e-4)`），合并后检查 `BRepCheck_Analyzer` 有效且相对体积变化 `< 1e-5` 才采纳，并在 `metadata` 记录合并前后 face/edge 数量 |
| **1.3 放行合法退化边** | [veudf/preprocess/cad.py:231-274](../../Hunyuan3D/veudf/src/veudf/preprocess/cad.py#L231-L274) & [filtering.py:71-81](../../Hunyuan3D/veudf/src/veudf/preprocess/filtering.py#L71-L81) | `cad.py` 在 `BRepAdaptor_Curve` 之前拦截 `BRep_Tool.Degenerated(edge)`：校验 `len(use_list)==1, len(fids)==1, vids[0]==vids[1]`；检查所属面为 `Cone`/`Sphere` 或该顶点连有 $\ge 2$ 条普通边后记入 `dropped(reason="degenerate_excluded")`（其余孤立非解析退化尖点报 `unsupported_isolated_degenerate_freeform`）；`filtering.py` 的 `audit_shape` 遇到退化边直接跳过弧长积分 |
| **1.4 精简三维 Decoder 监督池** | [veudf/preprocess/sampling.py:310-447](../../Hunyuan3D/veudf/src/veudf/preprocess/sampling.py#L310-L447) | ① `near_surface` 法向扰动改为 4 层 `[0, 0.002, 0.01, 0.025, 0.04]`（权重 `[0.25, 0.35, 0.25, 0.15]`）+ 20% 各向同性；② 新增 `sample_feature_3d_pool()` 直接绕 `edge_zero`/`vertex_zero` 在三维近表面带内按 `feature_radii` 分层采样（接受 `|s_clean| <= 0.04`），替代耗时的表面 `proposal_fragments` 细分；③ `edge_zero` / `vertex_zero` 逻辑保持不变 |
| **1.5 组合式退化二维表面变体构建** | 新增 `veudf/preprocess/degradation.py`，接入 [veudf/preprocess/pipeline.py](../../Hunyuan3D/veudf/src/veudf/preprocess/pipeline.py) | 实现组合式表面退化管线（低频 RBF/正弦位移 + 法向微偏置 + Taubin 平滑 + QEM/MC + 微配准残差 + 曲率加权微扰），实测距干净面最大偏移控制在 `<= 0.015`（`val` 固定 `low/mid/high` 三档并保留 `mesh/`）；在退化网格上采样 `M = 163,840` 点 `(pos_noisy, normal_noisy)` 并**统一调用 `FeatureUDFOracle` + `MeshSDFOracle`** 写入 `encoder_surfaces/v00..v0(K-1)/` |
| **1.6 同步与回归测试** | `DualBrep/sync_veudf.py` & `veudf/tests/` | 补充 3 个合成 STEP 单测（全圆锥、倾斜半圆锥、两半圆锥合并），跑通 `veudf` 单测后运行 `python sync_veudf.py --apply` 同步到 `DualBrep/veudf` |

### Phase 2：配对 VAE 重训与多变体 Latent 缓存（`Hunyuan3D/hy3dshape`）

| 任务 | 目标文件与位置 | 具体改动内容 |
|---|---|---|
| **2.1 数据集读取与零重标在线增强** | [hy3dshape/data/surface_pools.py](../../Hunyuan3D/hy3dshape/hy3dshape/data/surface_pools.py) | ① 支持读取 `encoder_surfaces/{clean, v00..}`；`train` 按 20% `clean` + 80% 随机 `v_k` 抽取 `81,920` 点同时喂 `geo_*` 与 `field_*`（严格 NaTex 对齐）；② 新增在线零重标增强：非均匀密度/局部遮挡加权抽点 + 法向 `1°~5°` 锥角微扰 + 15% Normal Dropout + 可选全局随机 SO(3) 旋转；③ 将 Decoder 监督过滤解耦为 `supervision_band_width = 0.040`，并按 10% 混入当前表面池未进 Encoder 的剩余点 |
| **2.2 VAE 坐标残差与浅层几何微调** | [hy3dshape/models/autoencoders/edge_vertex/shapevae.py](../../Hunyuan3D/hy3dshape/hy3dshape/models/autoencoders/edge_vertex/shapevae.py#L66-L79,L193-L200) | ① 解码器新增零初始化 `self.field_coord_proj`，将 `q = self.geo_feature_to_field_query(h) + self.field_coord_proj(self.pe(x.float()))`；② `pretrained_geometry` 模式下增加配置项开放 `geo_input_proj` 与浅层 `geo_cross` 的 LoRA 或 `0.1×` 低学习率微调；③ 保持 `Q00Zero` 零集支路启用 |
| **2.3 分档评估与多变体缓存** | [hy3dshape/data/latent_cache.py](../../Hunyuan3D/hy3dshape/hy3dshape/data/latent_cache.py) & `evaluation/` | `build_cache` 遍历每资产的 `[clean, v00..v0(K-1)]` 分别缓存 `(z_g_raw, anchor_xyz, z_f_mu, z_f_logvar, variant_id, offset_tier)`；验证脚本按 `clean / low / mid / high` 四档分桶报告近边/近顶点 MAE |

### Phase 3：配对 Flow 训练与推理适配（`Hunyuan3D/hy3dshape`）

| 任务 | 目标文件与位置 | 具体改动内容 |
|---|---|---|
| **3.1 特征区加权 Flow Loss** | [hy3dshape/models/diffusion/flow_matching.py](../../Hunyuan3D/hy3dshape/hy3dshape/models/diffusion/flow_matching.py) & `training/losses.py` | 缓存每个 anchor 的近边/近顶点距离 `anchor_udf`，在 `flow_objective` 中对近边/近顶点 token 按距离分桶加权，聚焦拓扑关键区 |
| **3.2 推理带宽与 `presence` 适配** | [hy3dshape/inference/pipeline.py](../../Hunyuan3D/hy3dshape/hy3dshape/inference/pipeline.py#L71-L78) | 推理默认 `band_width = 0.026`；放宽/改进 `infer_presence`（避免偏移扫描面在 `S0` 上误判小顶点缺失，支持交由下游 `scout` 三维体素复核）；支持多 seed 批量采样供 Best-of-N 选优 |

### Phase 4：后处理退化边支持与扫描容差协同（`DualBrep/band_brep_reconstruction`）

| 任务 | 目标文件与位置 | 具体改动内容 |
|---|---|---|
| **4.1 默认配置更新** | [configs/default.json](../band_brep_reconstruction/configs/default.json) | 设 `band_width = 0.026`、`scan_offset_tolerance = 0.018` |
| **4.2 `surface_fitting.py` 4 处改造** | [bandbrep/surface_fitting.py](../band_brep_reconstruction/bandbrep/surface_fitting.py) | ① 第 73-87 行：`FacePatch.uv(..., ordered=True)` 在 `cone`/`sphere` 极点（$\rho < 10^{-7}$）沿有序曲线继承相邻 `u` 修复方位角跳变；② 第 290-292 行：单/双圆环边界球面将 `axis` 对齐到圆环法向；③ 第 493, 525, 559-560 行：内点 90%/95% 分位门限与 `scan_offset_tolerance` 联动，大偏移面提高干净边界对解析面参数的约束优先级；④ 第 562-566 行：`cone`/`sphere` 折叠检查排除极点 `< 1e-4` 邻域 |
| **4.3 `occ_assembly.py` 3 处改造** | [bandbrep/occ_assembly.py](../band_brep_reconstruction/bandbrep/occ_assembly.py) | ① 第 267-301, 338, 471-475 行：扩展单环周期面支持到 `cone`（含斜切闭合底环通用 pcurve），对称新增 `make_conical_cap()`；② 第 480-485 行：非单环 `cone`/`sphere` wire 增加极点 UV 跳变检测并自动插入绑定到共享顶点的退化边；③ 第 513-517 行：`SameParameter` 循环跳过 `BRep_Tool.Degenerated(te)` |
| **4.4 端到端闭环验收** | `band_brep_reconstruction/tests/` & `benchmarks/evaluate_step.py` | ① 先跑 3 个合成退化边 STEP 单测（全圆锥、倾斜半圆锥、倾斜半球冠）；② 在 `val` 集 4 档退化网格（`clean / low / mid / high`）上联调 `Mesh → Flow → band-brep-reconstruct → STEP`，验收各档 STEP 通过率、V–E–F 拓扑 F1 与相对输入网格的几何纠偏改善量 |
