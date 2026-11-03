# ABC 网格表面双 UDF 数据处理

更新：2026-11-02。本页描述 `surface-udf` 分支的本仓库 [veudf](../veudf/README.md)，版本 3.0.0，schema `edge_vertex_surface_udf_v1`。外部生产目录保持原实现；旧三场生产数据和历史实验保留原版本语义，新分支不兼容其读取。

## 1. 场定义与几何来源

查询点仅位于最终闭合 condition 网格 M。dE 是到全模型保留有限 CAD 边的最小欧氏距离，dV 是到全模型必要 CAD 顶点的最小欧氏距离。网格三角化／焊接、CAD 筛选、seam 排除和必要顶点判定沿用 `collect_targets()`。不是测地距离，也不限制为所属面的边／顶点。

网格近似 CAD，表面样本在曲边附近也可能有正距离。所有距离依据实际存储坐标重新计算，不把来源边／来源顶点当作最近实例，不强制曲边标签为零。

## 2. 三个表面池

- surface：面积均匀采样，包含独立可复现的几何前缀 S0。
- edge_surface：CAD 边附近、位于 M 上的特征采样。
- vertex_surface：必要顶点附近、位于 M 上的特征采样。

默认点数分别为 393216、393216、235930，合计 1022362。删除 near_surface、噪声层、CAD zero 训练池和 volume 路径。evaluation 使用独立随机流，不能从训练点池抽取冒充独立评估。

每个点记录所属三角形及重心坐标，保留 float64 重心权重，点坐标和法向使用 float32。归一化坐标下，存储点与重心重建位置误差上限为 2e-7；这是浮点存储预算，不是 CAD 三角化误差界。

## 3. 距离引擎

[distance.py](../veudf/src/veudf/preprocess/distance.py) 对直线段、有限圆弧批量解析求解，复杂曲线继续调用 OCC。处理有限参数、跨周期圆弧、完整圆、端点和轴上多解。解析多解按几何判定，不把重复数值极值当作不同最近点。

每个网格三角形建立保守候选边缓存。中心 c、半径 r、任意种子边在中心的距离 Uc 给出整个三角形的上界 Uc+r。只有当边盒到三角形盒的距离下界大于该上界及容差时，才排除该边。所属 face／采样来源只作初始提示，薄壁跨面最近目标仍可被找到。

查询按三角形组织，边批量求距离并继续逐点包围盒剪枝。V-UDF 使用精确批量 cKDTree。特征采样共用三角形空间索引思想，以特征半径筛邻域；不能用最近边候选表替代完整采样邻域。

## 4. 几何条件

保留 condition/vertices.npy、triangles.npy、meta.json 及独立 S0。元数据包含网格／点／法向哈希、归一化和几何采样版本。条件读取不接触 E/V 标签、CAD face ID 或拓扑。

网格距离工具仍用于验证和后处理，E/V 点标注不生成 SDF。几何 VAE 的体积 SDF 监督如有需要，独立生成，不混入此 schema。

## 5. 审计与发布

- 所有点均通过三角形／重心重建验证。
- 抽样使用全边 OCC 参考查询核对距离及唯一最近实例；E/V 等距标记独立校验。
- 维持 dE≤dV、全局欧氏 1-Lipschitz、特征半径及空集约定。
- 资产原子发布；清单、环境、配置和源码哈希固定后才允许续跑。
- 保留内容去重、几何去重及 family split 隔离。
- 性能分别记录 sampling、labels、audit、evaluation、总时间和距离引擎计数；候选建立时间包含在查询时间内，不能重复加总。

## 6. 当前文件与字段

每个训练／evaluation 表面池均使用独立 NPY 数组，可 mmap：

| 文件 | 形状、类型 | 含义 |
|---|---|---|
| points.npy | N×3 float32 | 网格表面查询位置 |
| normals.npy | N×3 float32 | 所属网格三角形法向 |
| triangle_ids.npy | N int32 | 最终网格三角形 ID |
| barycentric.npy | N×2 float64 | 顶点 1/2 权重，顶点 0 权重为 1-u-v |
| fields.npy | N×2 float32 | 完整距离 dE、dV |
| nearest_ids.npy | N×2 int32 | 全局最近 CAD 边／必要顶点 |
| flags.npy | N×2 uint8 | bit0 有效，bit1 等距歧义 |
| source_instance.npy | N int32 | 采样来源，不是最近实例 |
| strata.npy | N×1 uint8 | 按实际全局距离得到的半径层 |

空通道以 2√3 占位、nearest_id=-1、有效位关闭；不会误当作真实监督。CAD face 对应关系保存在 audit，仅用于标注／审计。

读取入口为 `veudf.io.dataset.discover_assets / load_surface_pool`。预测交换使用新 `edge_vertex_surface_prediction_v1` 两通道格式；旧三通道 schema 明确拒绝。

## 7. 迁移边界

新分支直接维护仓库内 veudf；旧生产镜像导入脚本移除，历史来源见 SOURCE.json。生产训练端仍使用旧三通道／离面池／zero mask 时，需要单独迁移数据读取、损失与评估协议。本次仓库预处理改造不修改外部生产文件。

命令与配对性能基准见[模块 README](../veudf/README.md)，历史重建成绩见[实验主文档](ThreeFieldReconstructionExperiments.md)。历史成绩不代表此数据契约已经完成训练或后处理验证。
