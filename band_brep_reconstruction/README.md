# 网格与 E/V 场的 B-rep 后处理

版本 0.4.0，2026-11-01。输入为**预处理得到的闭合网格，以及可连续查询的两个 metric UDF**。默认先在低 UDF 区域执行 L5 加密，然后恢复面核心、拓扑关联、共享顶点和曲线，拟合曲面并装配 STEP。[算法与产物](../docs/BandLabelTopologyRecovery.md)、[代码来源](PROVENANCE.md)。

## 安装

在仓库根目录、Python 3.10+ 环境中执行：

```bash
python -m pip install -e ./veudf -e './band_brep_reconstruction[models,test]'
# 完整 STEP 装配还需要 OCC；图阶段不需要。
conda install -c conda-forge pythonocc-core=7.7
```

`veudf` 是[生产源码同步副本](../veudf/README.md)，提供与预处理相同的网格 AABB 距离和绕数符号。无需 PyMCubes。生产模型 bundle 推理另外需要生产环境中的 `hy3dshape`。

## 输入与运行

网格可传 `condition/`、包含它的资产目录，或已处于模型坐标系的三角网格文件。condition 包含 `vertices.npy`、`triangles.npy`、`meta.json`；读取时检查版本、dtype 和 mesh hash。坐标范围为 `[-1,1]`，不自动重新归一化；网格必须闭合、方向一致、没有零面积面，腔体内壳方向相反。

模型应在同一坐标系及距离尺度上查询 `[dE,dV]`。固定顶点的预测数组不足以支持加密、顶点壳层和曲线连续检查。截断归一化的模型需要在 decoder 中还原 metric 距离；负值、无效值和饱和平台不能作为有效零距离。

```python
from bandbrep import Config, CallableUDFProvider
from bandbrep.surface_mesh import load_mesh
from bandbrep.reconstruct import reconstruct

mesh = load_mesh('/data/asset/condition')
udf = CallableUDFProvider(
    decode_metric_ev,                  # [N,3] -> [N,2]，也可返回带有效/饱和 mask 的对象
    fingerprint='weights-and-latent-sha256',
    gradient=metric_ev_gradient,        # 可省略；省略后采用中心差分
    band_width=.02,                    # 必须与模型训练带一致
)
result = reconstruct(mesh, udf, Config(), '/data/new_output')
```

`gradient(xyz, channel)` 接收 `0=E, 1=V`，返回 `[N,3]` 梯度和 `[N]` 有效 mask。调用者负责让指纹覆盖权重、latent、条件几何及编码策略。省略 presence 时，在默认 393216 个确定性网格表面样本上，以距离低于 0.005 的支撑数至少 16 推断；它不需要独立预测头，也不读取 GT 拓扑。

生产 `hy3dshape.inference.pipeline.FieldProviderAdapter` 可用 `ProductionUDFProvider(adapter)` 接入；只使用其 E/V 及梯度，验证其 condition 与输入网格完全一致。adapter 必须选择 `sdf_source='condition'`。

CLI 支持两种连续模型来源：

```bash
# 已冻结 latent/条件的 TorchScript decoder：[N,3] -> metric [N,2]
python band_brep_reconstruction/reconstruct.py \
  --mesh /data/asset/condition --udf-model /models/ev_decoder.pt \
  --band-width .02 --output /data/run --stop-after graph

# 生产 Flow：只读取 condition 几何，不读取 E/V 真值
python band_brep_reconstruction/reconstruct.py \
  --mesh /data/asset/condition --vae-bundle /models/vae --flow-bundle /models/flow \
  --device cuda --flow-steps 5 --output /data/generated_brep

# 显式 VAE 重建评估：编码已知资产的 E/V 真值，再后处理解码结果
python band_brep_reconstruction/reconstruct.py \
  --mesh /data/pools/assets/id/condition --vae-bundle /models/vae \
  --asset /data/pools/assets/id --output /data/vae_reconstruction
```

后两个入口调用外部生产 pipeline；本仓库测试覆盖 adapter 契约，未使用真实训练权重进行端到端验证。VAE 单独不能从网格确定 E/V latent，condition-only 生成需要 Flow 或调用者提供 latent 的 decoder。

## 默认流程与配置

1. 验证原始网格，建立网格距离查询；输入网格始终是几何基准。
2. 在现有顶点与三角形中心查询 E/V，近边/顶点区域做共享边一致剖分。新增顶点、最终中心重新查询模型，不插值旧 UDF 或复制父标签。
3. 最终三角形中心执行 DualBrep 两阶段面核心提取；非核心传播、局部顶点域恢复 E–F、V–E、V–F。
4. 由两个 UDF 的梯度定位顶点和边，拟合固定共享端点的曲线，再拟合曲面并构建面环、seam 和 pcurve。
5. OCC 装配、导出并读回 STEP；仅 `solid_validated` 表示全部验收完成。

L5 默认：目标最长边 `0.005`，近边/顶点选区半径 `0.02/0.03`，最多四轮，最多 150 万三角形。线性细分保持原表面、面积和有向体积，不恢复网格缺失的曲率。细长扇形三角形也可能导致细分超过预算。**非常粗的原生 CAD 网格可能四轮仍未达标**，此时报告 `refinement_target_unmet`；可通过 JSON 增加轮数，例如 `{"refinement_max_rounds": 12}`，三角形预算仍生效。加密默认必经，没有关闭入口。

配置见 [default.json](configs/default.json) 和 [Config](bandbrep/config.py)。优先级为默认值 < `--config` JSON < 显式 CLI 参数。已删除 `--preset`、分辨率、MC、旧 CAD/离散 NPZ/三场 MLP 输入和面网络初始化接口；旧配置和缓存不能续用。旧实验脚本随旧入口删除，历史报告和输出保留。

SDF 无需模型预测：内部按需计算到原始网格的距离，用于特征与曲面相容性检查、查询带判定和方向探测。E/V 在训练带与坐标域内查询；带外梯度更新回退，顶点壳层缩小，仍无有效支撑则失败。

## 输出与续跑

`input_mesh.npz` 保存原网格；`working_mesh.npz` 保存加密网格及原始父三角形；其余阶段产物见[阶段表](../docs/BandLabelTopologyRecovery.md#2-阶段代码与产物)。`run.json` 保存配置、输入身份、加密历史、阶段与失败信息，`quality.json` 保存完成状态。默认记录查询点、metric 场值及有效/饱和 mask。

`--stop-after` 可选 `seeds / graph / vertices / curves / surfaces / solid`，默认 `solid`。相同输入、配置和源码可用 `--resume` 从早期阶段继续，仅 `stop_after` 可变化。缓存键覆盖原网格数组与归一化信息、模型指纹、查询带与 presence、配置及实现哈希；每个阶段产物另验 SHA-256。

有限查询、几何采样和内核有效性不证明所有真实边均已恢复。窄面核心缺失、无边界非解析面和复杂自由曲面仍可能失败。

## 测试与验证

```bash
PYTHONPATH="$PWD/veudf/src:$PWD/band_brep_reconstruction" \
  python -m pytest band_brep_reconstruction/tests -q
python scripts/sync_veudf.py --check
```

测试包含闭合剖分、域与 mask、梯度、生产 adapter、输入/缓存身份、分割/拓扑及网格起步的立方体、圆柱、腔体球与球体贯通孔 STEP 读回。外部真实权重和真实生产批次需单独评估，历史实验成绩不作为新流程成绩。
