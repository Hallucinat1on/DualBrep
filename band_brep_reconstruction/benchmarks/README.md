# 离线评分

安装 veudf 和重建模块后，用原始 STEP 对照已经通过读回的结果：

```bash
python band_brep_reconstruction/benchmarks/evaluate_step.py --step source.step --run run
```

评分校验源 STEP 与输出 STEP 哈希，统计边、顶点、面域、关联及采样曲面误差。粗投影路径用原三角形及其面域标签做对照，精确切面路径用切割网格。patch 回退时按原逻辑面和线框评分，内部接缝及 STEP 曲面保留在装配记录。真值仅在此步骤使用。

历史双向追踪／MC 配对实验及脚本见 [Experiment.md](../../docs/Experiment.md) 所列冻结批次。
