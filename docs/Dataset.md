# v5 数据集格式与全链路改造方案（8D 联合场 + 编解码池彻底解耦）

本文档确立面向 **8D 联合目标表示 `[dE, dV, uE(3), uV(3)]`**（见 [VectorFieldTarget.md](VectorFieldTarget.md)）的 **v5 数据集格式（`edge_vertex_neighborhood_pools_v5`）** 以及预处理（`veudf`）、训练加载与损失（`hy3dshape`）、下游契约（`band_brep_reconstruction`）的完整工程改造方案。

---

## 1. 改造目标与核心设计原则

相对 v4（纯标量 `[s_mesh, dE, dV]`），v5 的核心优化目标是在引入 6 维位移向量目标 `[uE(3), uV(3)]` 的同时，**将离线预处理增量与在线训练额外开销压到最低**：

1. **编码器输入池与解码器查询池彻底解耦**
   - `pools/surface/` 变为**编码器专用池**（仅服务 `geo_encoder` 与 `encode_field`），训练解码器不再专门在 `surface` 上抽查询点，原 `10%` 配额转移至 `near_surface`；
   - **收益 1（精度与泛化）**：精确表面 $s_{\text{mesh}} = 0$ 严格位于双侧 `near_surface`（第一层 `[0, 0.005]`）与 360° 特征邻域监督的正中央，内插无损；同时避免与 `anchor_xyz` 共面的“表面捷径过拟合”，提升对削圆/偏移网格的泛化能力；
   - **收益 2（消除 CPU 热点）**：彻底删除 [surface_pools.py](../../Hunyuan3D/hy3dshape/hy3dshape/data/surface_pools.py#L401-L404) 中每步对 `262,144` 行与 `81,920` 行执行的 `np.setdiff1d` 互斥剔除逻辑；
   - **收益 3（省磁盘）**：`pools/surface/` 只需存 2 通道 `[dE, dV]`（无需存 `s_mesh`、6D 位移与 `flags`），单资产直接省下约 `7.3 MB`。
2. **面向 DataLoader 随机访存的连续内存布局**
   - 解码查询池将全部 9 个标量/向量通道合并为单个连续文件 **`fields.npy (N, 9) float32`**（每行 `36 bytes`，落在单个 64-byte CPU 缓存行内）。训练每步随机行索引只需 **1 次连续切片**即可拿齐全部 8D 目标，避免拆分多个 `.npy` 带来的多路随机 page fault 以及在线查 `vertex_zero` 做减法的 CPU 开销；
   - 删掉纯 3D 池 `edge_neighborhood` / `vertex_neighborhood` 中无用的全零 `normals.npy`（再省约 `6.3 MB/asset`）。
3. **训练期零冗余文件读取**
   - 由于契约保证所有 `saturated` 点和缺失通道的距离严格等于 `udf_truncation (0.05)`，只要向量监督阈值满足 `d_vec_max < 0.05`（如 `0.035`），**`d < d_vec_max` 在数学上自动等价排除了所有饱和点与 `nearest_id == -1` 点**；
   - 因此训练 `sample()` 仅需额外读 `1 byte/行` 的 `flags.npy`（取 bit 0 = `e_tie`、bit 1 = `v_tie`），**完全不读 `nearest_ids.npy` 和 `saturated.npy`**。
4. **预处理零额外 OCC 求解开销**
   - [distance.py](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/distance.py#L100) 中的 `solve()` 原本就已算出精确 3D 最近点 `projected`，v5 只需将其同步记录，**OCC 求解次数与 v4 完全一致**。

---

## 2. v5 数据集存储布局与契约规范

- **Schema 标识**：`SCHEMA = "edge_vertex_neighborhood_pools_v5"`
- **通道定义**：
  - 编码器专用池（`pools/surface`）：`ENCODER_CHANNELS = ("dE", "dV")`
  - 解码器查询池 & 评测池：`CHANNELS = ("s_mesh", "dE", "dV", "uEx", "uEy", "uEz", "uVx", "uVy", "uVz")`
- **核心阈值**（沿用 v4 黄金配置，采样策略不变）：`band_max = 0.04`、`udf_truncation = 0.05`、`feature_radii = (0.002, 0.005, 0.01, 0.02, 0.04)`、`feature_weights = (0.20, 0.28, 0.27, 0.17, 0.08)`。

### 2.1 资产目录与各池文件清单

| 角色 | 池目录 | 点数（默认） | 存储的核心 `.npy` 文件 | 说明 |
|---|---|---|---|---|
| **编码器专用池** | `pools/surface/` | `262,144` | `points.npy (N, 3) f32`<br>`normals.npy (N, 3) f32`<br>`fields.npy (N, 2) f32` (`[dE, dV]`)<br>`nearest_ids.npy (N, 2) i32`<br>`saturated.npy (N, 2) bool` | `encode_field` 仅用 `[dE, dV]`；不存 `s_mesh`、6D 位移与 `flags`；前 `262,144` 行 `points/normals` 继续在 `condition/meta.json` 做 SHA-256 锁定 |
| **解码器 3D 空间池** | `pools/near_surface/`<br>`pools/edge_neighborhood/`<br>`pools/vertex_neighborhood/` | `393,216`<br>`327,680`<br>`196,608` | `points.npy (N, 3) f32`<br>`fields.npy (N, 9) f32`<br>`flags.npy (N,) u8`<br>`nearest_ids.npy (N, 2) i32`<br>`saturated.npy (N, 2) bool`<br>`sample_type.npy` / `strata.npy` / 特征实例元信息 | **彻底删除 `edge/vertex_neighborhood` 的全零 `normals.npy`**；训练 `sample()` 只读 `points`、`fields`、`flags` |
| **零集池** | `pools/edge_zero/`<br>`pools/vertex_zero/` | `65,536`<br>精确顶点数 | `points.npy (N, 3) f32`<br>`fields.npy (N, 9) f32`<br>`flags.npy (N,) u8`<br>`zero_mask.npy (N, 2) bool`<br>`source_instance.npy` 等 | 保持 `(N, 9)` 与 3D 空间池格式统一，便于 `append_rows` 零分支拼接；`Q00Zero` 仅切 `fields[:, 1:3]` |
| **独立评测池** | `evaluation/{surface, near_surface, edge_neighborhood, vertex_neighborhood, edge_zero, vertex_zero}/` | 每池 `4,096`（`vertex_zero` 为全量顶点） | 全部统一保存完整 **`fields.npy (N, 9) f32`** + `flags.npy` 等无截断精确 CAD 真值 | **保留 `evaluation/surface/` 完整 9 通道**，用于客观检验不在 `surface` 上做解码监督时的精确表面内插精度 |

### 2.2 数值与掩码契约

- **位移向量定义**：$\mathbf{u}_E = \mathbf{p}_E^* - \mathbf{x}$，$\mathbf{u}_V = \mathbf{p}_V^* - \mathbf{x}$（由查询点指向最近 CAD 边/顶点，各分量可正可负）。
- **饱和与缺失通道**：
  - `saturated[:, 0] == True`（或 `has_edges == False`）时：`dE == 0.05`、`nearest_ids[:, 0] == -1`、`uE == (0, 0, 0)`、`e_tie == False`；
  - `saturated[:, 1] == True`（或 `has_vertices == False`）时：`dV == 0.05`、`nearest_ids[:, 1] == -1`、`uV == (0, 0, 0)`、`v_tie == False`。
  - **防跳变原则**：磁盘在饱和区存入 `(0, 0, 0)` 仅作为防 `NaN/Inf` 占位符，**绝不参与 Loss 监督**；训练期仅在 `d < d_vec_max`（如 `0.035 < 0.05`）内施加向量监督，`d >= d_vec_max` 保持 0 梯度自由外推，彻底避免边界断崖跳变。
- **切迹（`tie`）点**：保留 `reduce_min` 胜出特征的位移值（不清零，便于离线质检校验 `||u|| == d`），同时置位 `flags`（bit 0 = `e_tie`，bit 1 = `v_tie`），由训练 Loss 掩码屏蔽。
- **零集点**：`zero_mask[:, 0]` 处满足 `dE <= label_tolerance` 且 `||uE|| <= label_tolerance`；`zero_mask[:, 1]` 处同理。

---

## 3. 预处理端（`veudf` v4 → v5）逐文件改造清单

> 按项目规范：先在 `Hunyuan3D/veudf` 完成修改并跑通全部单元测试，再执行 `sync_veudf.py --apply` 同步到 `DualBrep/veudf` 并用 `--check` 校验一致。`cad.py`、`filtering.py`、`condition.py`、`config.py` **均无需修改**。

### 3.1 [schema.py](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/schema.py)
- `SCHEMA = "edge_vertex_neighborhood_pools_v5"`
- `ENCODER_CHANNELS = ("dE", "dV")`
- `CHANNELS = ("s_mesh", "dE", "dV", "uEx", "uEy", "uEz", "uVx", "uVy", "uVz")`

### 3.2 [distance.py](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/distance.py)
- `_empty(self, count)`（[L59-L64](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/distance.py#L59-L64)）：新增 `offset=np.zeros((count, 3), dtype=np.float64)`。
- `reduce_min(result, rows, distances, instance, local_ties, tolerance, offsets)`（[L16-L29](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/distance.py#L16-L29)）：在 `changed = rows[improved]` 时写入 `result["offset"][changed] = offsets[improved]`。
- `_finish(self, result, present)`（[L66-L73](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/distance.py#L66-L73)）：新增 `result["offset"][saturated] = 0.0`。
- `query(self, points)`（[L96-L116](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/distance.py#L96-L116)）：
  - 边循环：预分配 `offsets = np.empty((len(rows), 3), dtype=np.float64)`，在 [L100](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/distance.py#L100) `solve()` 后记录 `offsets[j] = projected - q[row]` 并传入 `reduce_min`；
  - 顶点循环：将 `point - q[rows]` 作为 `offsets` 传入 `reduce_min`。

### 3.3 [reference.py](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/reference.py)
- `empty_result(count)`（[L80-L86](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/reference.py#L80-L86)）：新增 `offset=np.zeros((count, 3), dtype=np.float64)`。
- `ReferenceEdgeOracle.query()`（[L116-L163](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/reference.py#L116-L163)）：维护 `best_offset`，在 `if distance < best:` 时更新 `best_offset = projected - point`，循环结束后写入 `result["offset"][row] = best_offset`。
- `ReferenceVertexOracle.query()`（[L172-L184](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/reference.py#L172-L184)）：写入 `result["offset"] = self.vertices[index] - points`。

### 3.4 [sampling.py](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/sampling.py)
- `sample_feature_3d_pool()`（[L341-L349](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/sampling.py#L341-L349)）：删除全零字段 `normals=np.zeros((total, 3), np.float32)`（与 `near_surface` 保持一致，纯 3D 池不存 `normals.npy`）。

### 3.5 [pipeline.py](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/pipeline.py)
- `label_pool(..., encoder_only=False)`（[L79-L173](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/pipeline.py#L79-L173)）：
  - 当 `encoder_only=True`（即训练 `pools/surface`）：
    - `fields.npy` 为 `shape=(n, 2), dtype=np.float32`，仅写入 `np.column_stack((e["distance"], v["distance"]))`；
    - 不需要调 `oracle.query(q)` 算 `sdf`，也不需要写 `flags.npy`（或可保留 `flags.npy` 仅 256 KB 供 audit 复用，推荐仅训练不读、audit 复用更简洁）。
  - 当 `encoder_only=False`（其余 5 个训练池 + 全部 6 个 `evaluation/` 池）：
    - `fields.npy` 为 `shape=(n, 9), dtype=np.float32`，写入 `np.column_stack((sdf, e["distance"], v["distance"], e["offset"], v["offset"]))`。
  - **必改的 3 处距离切片**（向量分量可为负，不可再用 `[:, 1:]`）：
    1. 非负校验（[L129](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/pipeline.py#L129)）：仅对标量距离通道检查 `< 0`（`encoder_only` 为 `values < 0`，9 通道为 `values[:, 1:3] < 0`）；
    2. 零集残差校验（[L153](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/pipeline.py#L153)）：改为 `values[:, 1:3][pool["zero_mask"]] > cfg.label_tolerance`；
    3. 独立审计误差（[L248](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/pipeline.py#L248)）：改为 `expected - stored_distances`。
  - **新增 4 项向量硬质检**（对所有 9 通道池执行）：
    1. 有限性：`np.isfinite(values).all()`；
    2. 模长一致性：对非饱和且 `nearest[:, c] >= 0` 的点，`| ||u|| - d | <= cfg.label_tolerance`；
    3. 饱和清零：`saturated[:, c]` 处 `u == 0`；
    4. 零集向量：`zero_mask[:, c]` 处 `||u|| <= cfg.label_tolerance`。
- `audit()`（[L220-L271](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/pipeline.py#L220-L271)）：
  - 在非 `tie`、非饱和的唯一最近特征抽检点 `unique` 上，校验落盘 `uE, uV` 与 `ReferenceEdgeOracle / ReferenceVertexOracle` 的向量误差 `<= cfg.label_tolerance + cfg.geometry_tolerance`。
- `meta.json`（[L397-L400](file:///opt/data/private/yihengxu/DualBrep/veudf/src/veudf/preprocess/pipeline.py#L397-L400)）：同时记录 `channel_names=list(CHANNELS)` 与 `encoder_channel_names=list(ENCODER_CHANNELS)`。

---

## 4. 训练端（`Hunyuan3D/hy3dshape`）与下游契约改造清单

### 4.1 DataLoader（[surface_pools.py](file:///opt/data/private/yihengxu/Hunyuan3D/hy3dshape/hy3dshape/data/surface_pools.py)）
1. **契约校验（`load()`，[L252-L352](file:///opt/data/private/yihengxu/Hunyuan3D/hy3dshape/hy3dshape/data/surface_pools.py#L252-L352)）**：
   - `pools/surface`：校验 `fields.shape == (n, 2)`、`normals.shape == (n, 3)`；
   - 其余 5 个解码池：校验 `fields.shape == (n, 9)`、`flags.shape == (n,)`，且不再要求 `edge_neighborhood / vertex_neighborhood` 存在 `normals.npy`；
   - 距离与饱和一致性校验改为读取标量列（`surface` 为 `fields[:, c]`，其余池为 `fields[:, c + 1]`）；
   - `data["indices"]["surface"]` 直接取 `np.arange(len(sp["points"]))`（表面点恒满足 `|s_mesh| == 0 <= band_width`）。
2. **删除 `exclude` / `np.setdiff1d`（`_rows()`，[L398-L406](file:///opt/data/private/yihengxu/Hunyuan3D/hy3dshape/hy3dshape/data/surface_pools.py#L398-L406)）**：
   - 由于解码器不再从 `surface` 抽 `field_queries`，彻底移除 `exclude` 参数与 `np.setdiff1d`。
3. **5 池解码采样（`sample()`，[L472-L508](file:///opt/data/private/yihengxu/Hunyuan3D/hy3dshape/hy3dshape/data/surface_pools.py#L472-L508)）**：
   - 编码端：`fu = pools["surface"]["fields"][fr["surface"]]`（`(N_enc, 2)`），`sample_zero_encoder` 取 `pool["fields"][pick, 1:3]`；
   - 解码端：从 `("near_surface", "edge_neighborhood", "vertex_neighborhood", "edge_zero", "vertex_zero")` 5 个池按比例 `[0.35, 0.375, 0.25, 0.02, 0.005]` 采样；
   - `append_rows(name, rows)` 同步采集：
     - `qparts.append(pools[name]["points"][rows])`
     - `vparts.append(pools[name]["fields"][rows, 1:])` → `(M, 8)` 即 `[dE, dV, uE(3), uV(3)]`
     - `fparts.append(pools[name]["flags"][rows])` → `(M,)` `uint8`
   - 返回新增 `field_query_flags`。
4. **几何查询兜底（`sample_geometry_queries()`，[L165-L190](file:///opt/data/private/yihengxu/Hunyuan3D/hy3dshape/hy3dshape/data/surface_pools.py#L165-L190)）**：
   - `surface` 部分的 `s_mesh` 直接填充 `np.zeros((len(rows), 1), dtype=np.float32)`，其余 3D 池仍切 `fields[rows, :1]`。
5. **旋转增强等变性（`apply_training_augmentation()`，[L45-L60](file:///opt/data/private/yihengxu/Hunyuan3D/hy3dshape/hy3dshape/data/surface_pools.py#L45-L60)）**：
   - 当 `cube_rotation_aug=True` 时，对 `field_query_values` 中的 `uE`（`[:, 2:5]`）与 `uV`（`[:, 5:8]`）同步右乘 `rot_t`：
     ```python
     vals = result["field_query_values"].copy()
     vals[:, 2:5] = vals[:, 2:5] @ rot_t
     vals[:, 5:8] = vals[:, 5:8] @ rot_t
     result["field_query_values"] = vals
     ```

### 4.2 配置、模型输出头与损失函数
- **YAML 配置（`hy3dshape/configs/*.yaml`）**：
  - `field_query_ratios` 更新为 5 池比例：`[0.35, 0.375, 0.25, 0.02, 0.005]`；
  - 新增向量场损失超参：`vec_truncation: 0.035`、`vec_loss_weight: 1.0`、`vec_consistency_weight: 0.05`、`vec_smooth_l1_beta: 0.003`。
- **模型架构（与 M4 无冲突的标量/向量解耦设计）**：
  - `geo_encoder`、`encode_field`、`Q00Zero`、潜空间 `z_g / z_f ([4096, 64])`、`FieldDiT` **零结构修改**；
  - **标量通道 `[dE, dV]`（2 维）**：完全保留现有 `field_query_decoder.output_proj(2) + M4 [edge_head(1), vertex_head(1)]` 结构不变，支持直接从既有 Run 9 权重热启动；
  - **向量通道 `[uE(3), uV(3)]`（6 维）**：在 M4 `LocalKnnQueryDecoder` 中新增两个独立轻量 MLP 头（`edge_vec_head(3)`、`vertex_vec_head(3)`），直接基于含 3D 有向相对坐标与 `S0` 局部几何 patch 的局部特征 `feat` 输出归一化位移 `[uE / 0.05, uV / 0.05]`（不走全局低频 `output_proj`，避免远场无监督区噪声污染）；未启用 M4 时回退至 `field_query_decoder.output_proj` 输出 8 维；
  - 最终拼接输出 `[B, Nq, 8]`（`[dE, dV, uE_norm(3), uV_norm(3)]`），尺度与标量 `[0, 1]` 严格对齐。
- **掩码损失（`losses.py`）**：
  - 保留原有 M2 标量损失作用于 `pred[:, 0:2]` vs `target[:, 0:2]`；
  - 向量掩码：
    - `mask_uE = (dE_gt < vec_truncation) & ((flags & 1) == 0)`
    - `mask_uV = (dV_gt < vec_truncation) & ((flags & 2) == 0)`
  - 仅在掩码内施加 `SmoothL1`（禁止 L2）与弱一致性项 `| ||u_pred|| - d_pred.detach() |`。

### 4.3 下游契约同步（`DualBrep/band_brep_reconstruction`）
- `veudf_contract.py` / `cad_source.py` / `cache.py` 升级为 `require_veudf_v5()`，评测读取 `evaluation/*/fields.npy[:, 1:3]` 与 `[:, 3:9]`。

---

## 5. 开销对比与验证清单

### 5.1 相对 v4 的系统开销评估

| 维度 | v4 基线 | v5 解耦 8D 方案 | 变化说明 |
|---|---|---|---|
| **离线预处理耗时** | 1.0× | **≈ 1.0×** | OCC 求解次数零增加，`pools/surface` 还省去了 `sdf` 计算 |
| **单资产磁盘占用** | ~55 MB | **~70 MB**（`+15 MB`） | 5 个解码池 `fields` 增加 ~28 MB；解耦 `surface` 省 7.3 MB + 删全零 `normals` 省 6.3 MB |
| **训练 DataLoader CPU / I/O** | 含 `setdiff1d(262k, 82k)` | **持平甚至更快** | 消灭 `setdiff1d` 排序开销；`(N, 9)` 单缓存行切片 + 1B `flags`，不碰 `nearest_ids / saturated` |
| **训练 GPU 显存与单步耗时** | 1.0× | **< 1.01×** | 仅 Decoder 末层多 6 通道（~3K 参数）+ 掩码 SmoothL1，无 autograd / 有限差分 |

### 5.2 改造验收检查项

1. **预处理单测与真实 OCC 校验**：跑通 `veudf` 全部单元测试（重点验证 `||uE|| == dE`、`||uV|| == dV`、饱和点 `u == 0`、`tie` 标志位与独立 `ReferenceEdgeOracle / ReferenceVertexOracle` 向量误差）；
2. **双仓同步检查**：运行 `sync_veudf.py --apply` 后执行 `sync_veudf.py --check`，确认 `provenance_matches: true`；
3. **训练数据管线测试**：验证 `cube_rotation_aug` 旋转后 `||uE'|| == dE` 且 `(x' + uE')` 仍精确落在旋转后的边上；验证 `surface` 与解码 5 池完全解耦；
4. **端到端评测对比**：在 `evaluation/surface`（未在训练解码器中直接监督）与 `evaluation/near_surface` 上核验 8D 场的零集无偏性与中场单步拉回精度。
