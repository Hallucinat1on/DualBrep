# 网格与 E/V 场的 B-rep 后处理

版本 0.5.0。输入为**预处理得到的闭合网格，以及可连续查询的两个 metric UDF**。默认先纯几何粗加密至 0.02，再按 E 近场细化至 0.005，然后恢复面核心、拓扑关联、共享顶点和曲线，拟合曲面并装配 STEP。[算法与产物](../docs/Post.md)、[代码来源](PROVENANCE.md)。

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
    band_width=.02,                    # 必须与模型训练带一致
)
result = reconstruct(mesh, udf, Config(), '/data/new_output')
```

`gradient(xyz, channel, step=1e-5)` 接收 `0=E, 1=V`，返回 `[N,3]` 梯度和 `[N]` 有效 mask。自2026-11-05起，E/V梯度统一从UDF采样值做中心差分：固定latent，在当前位置及三个坐标轴正负方向的6个空间位置查询距离，不使用autograd或解析梯度回调。原 `gradient=` 构造参数已移除，调用方应只提供距离查询函数。重建入口还检查中心及全部邻点的有效域、valid和saturated；无效时不调用替代梯度。步长使用模型坐标单位，过小会放大数值误差。详细公式见[查询接口](../docs/Post.md#查询接口)。

调用者负责让指纹覆盖权重、latent、条件几何及编码策略。省略 presence 时，在默认 393216 个确定性网格表面样本上，以距离低于 0.005 的支撑数至少 16 推断；它不需要独立预测头，也不读取 GT 拓扑。

生产 `hy3dshape.inference.pipeline.FieldProviderAdapter` 可用 `ProductionUDFProvider(adapter)` 接入；只使用其 E/V 距离和mask，验证其 condition 与输入网格完全一致。adapter 必须选择 `sdf_source='condition'`。

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

1. 验证并保存原始网格，保留为模型 condition 和 SDF 基准。
2. 不查询任何场，仅对超过 `0.02` 的共享边做粗加密，直到全局达标。
3. 基于原始网格建立场接口；在粗网格顶点与中心查询字段，仅依据 E 场选近场，对其中仍超过 `0.005` 的边做细加密。新增点重新查询，不插值 UDF 或复制父标签。
4. 将最终中心字段直接交给 DualBrep 两阶段面核心提取；非核心传播、局部顶点域恢复 E–F、V–E、V–F。
5. 由两个 UDF 的梯度定位顶点和边，拟合固定共享端点的曲线，再拟合曲面并构建面环、seam 和 pcurve。
6. OCC 装配、导出并读回 STEP；仅 `solid_validated` 表示全部验收完成。

默认 `refinement_coarse_edge=0.02`、`refinement_target_edge=0.005`，校验 `0 < fine <= coarse`；近边选区半径为 `0.02`。自2026-11-06起细加密只按E场选区；旧 `refinement_rho_vertex` 为配置兼容保留，不再影响选区。现有decoder仍联合返回E/V，本改动不代表已支持仅在边带内解码V。两级均只切超长边，邻面作必要共享边过渡。不设轮数、三角形数量或 RSS 硬上限；每例 `sample_timeout_seconds` 默认 `10800` 秒，超时报 `sample_timeout`。两级加密保持原分片线性表面、面积与有向体积，不恢复网格缺失的曲率。

库接口在重建和场查询检查点执行时间预算；实验 runner 另用独立进程监控，从 worker 启动起覆盖源准备、condition、两级加密、重建及离线评分，也能终止阻塞的原生几何调用。`coarse_refinement_progress.json`、`fine_refinement_progress.json` 分别保存两级进度。

配置见 [default.json](configs/default.json) 和 [Config](bandbrep/config.py)。优先级为默认值 < `--config` JSON < 显式 CLI 参数。已删除 `--preset`、分辨率、MC、旧 CAD/离散 NPZ/三场 MLP 输入和面网络初始化接口；旧配置和缓存不能续用。旧实验脚本随旧入口删除，历史报告和输出保留。

面分割默认 `face_second_pass=true`，执行两阶段核心提取。只做第一阶段时，在 Python 中使用 `Config(face_second_pass=False)`，在配置 JSON 中设置 `"face_second_pass": false`，或在原运行命令后加 `--no-face-second-pass`；`--face-second-pass` 可显式重新开启。关闭后仍执行第一阶段距离阈值、连通分量、大小和均值过滤，保留通过过滤的完整父核心，跳过第二阶段拆分；面标签继续重编号为连续整数。`bands`／`seeds` 阶段报告中的 `segmentation.second_pass_enabled` 记录开关状态。切换开关会改变缓存身份，应使用新的输出目录。

SDF 无需模型预测：内部按需计算到原始网格的距离，用于特征与曲面相容性检查、查询带判定和方向探测。E/V 在训练带与坐标域内查询；带外梯度更新回退，顶点壳层缩小，仍无有效支撑则失败。

## 输出与续跑

`input_mesh.npz` 保存原网格；`coarse_mesh.npz` 保存粗网格及原父三角形；`working_mesh.npz` 保存最终细网格及组合后的原父三角形；`center_fields.npz` 保存最终中心坐标、字段与 provider 身份，复用前严格校验坐标顺序和身份；其余阶段产物见[阶段表](../docs/BandLabelTopologyRecovery.md#2-阶段代码与产物)。`run.json` 保存配置、输入身份、加密历史、阶段与失败信息，`quality.json` 保存完成状态。默认记录查询点、metric 场值及有效/饱和 mask。

`--stop-after` 可选 `seeds / graph / vertices / curves / surfaces / solid`，默认 `solid`。相同输入、配置和源码可用 `--resume` 从早期阶段继续，仅 `stop_after` 可变化。缓存键覆盖原网格数组与归一化信息、模型指纹、查询带与 presence、配置及实现哈希；每个阶段产物另验 SHA-256。

有限查询、几何采样和内核有效性不证明所有真实边均已恢复。窄面核心缺失、无边界非解析面和复杂自由曲面仍可能失败。

## 测试与验证

```bash
PYTHONPATH="$PWD/veudf/src:$PWD/band_brep_reconstruction" \
  python -m pytest band_brep_reconstruction/tests -q
python scripts/sync_veudf.py --check
```

测试包含闭合剖分、域与 mask、梯度、生产 adapter、输入/缓存身份、分割/拓扑及网格起步的立方体、圆柱、腔体球与球体贯通孔 STEP 读回。外部真实权重和真实生产批次需单独评估，历史实验成绩不作为新流程成绩。

2026-11-01 已用同源 20 个 dev 完成[原生网格 L5 配对实验](../docs/ThreeFieldReconstructionExperiments.md#native-mesh-l5)：精确 CAD E/V 下，当时的四轮配置均未达到 0.005，尚未进入拓扑/STEP；旧 MC 的 M0/L5 图+STEP 分别为 5/20 和 7/20。该批原生网格的起始边长远大于 MC，该限制促成后续两级方案；不能用解析回归测试代替真实输入验证。

仅限时的单级重跑已终止，相关产物和专用文档已清理。[粗加密＋近场细加密方案](../docs/BandLabelTopologyRecovery.md#refinement)已实现并通过回归；2026-11-02 的同源 20 dev × 粗目标 0.02/0.03 [配对协议与状态入口](../docs/ThreeFieldReconstructionExperiments.md#coarse-fine-comparison)集中维护在实验主文档，最新进度和完整成绩以冻结批次汇总为准。
