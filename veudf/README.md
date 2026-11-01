# 生产 E/V-UDF 数据处理

本仓库后续数据处理以 `/opt/data/private/yihengxu/Hunyuan3D/veudf` 为唯一生产来源。`veudf/` 是其源码、配置、测试和打包文件的同步副本；本地运行不需要导入外部 Hunyuan3D。同步记录见 [SOURCE.json](SOURCE.json)，数据规范见[三场数据处理](../docs/ABCEdgeVertexUDFPreprocessing.md)。

当前版本 2.0.0，schema 为 `edge_vertex_mesh_pools_v1`，通道为 `[s_mesh,dE,dV]`。SDF 来自最终闭合 CAD 三角网格，E/V 来自有限 CAD 曲线和必要顶点。默认不生成体素 SDF，也不经过 MC。原 `edge_vertex_preprocessing/evudf` 已退出新数据处理流程。

## 同步生产实现

在仓库根目录执行：

```bash
# 只读核对生产文件、本地文件和同步哈希。
python scripts/sync_veudf.py --check
# 生产目录有更新后，覆盖本地受管理的源码、配置、测试与打包文件。
python scripts/sync_veudf.py --apply
```

自定义来源可加 `--source /path/to/veudf`。脚本不复制数据、运行任务、缓存或训练工程，不修改生产目录。本地 README 和集成文档独立维护；数据处理功能先在生产目录开发，再同步，避免形成第二套实现。`SOURCE.json` 记录生产文件 SHA-256，源码和测试保持逐字节一致。

## 环境与运行

需要 Python 3.10+、OCC/OCCT 7.7.0 和 libigl 2.6.3。完整依赖见 [environment.yml](environment.yml)、[pyproject.toml](pyproject.toml)。在仓库根目录：

```bash
conda env update -n hunyuan -f veudf/environment.yml
conda activate hunyuan
python -m pip install -e './veudf[jobs,postprocess,test]'

python -m veudf.cli.build_dataset manifest \
  --step-dir /data/abc/step/0000 --out /data/new_manifest.jsonl
python -m veudf.cli.build_dataset build --root /data/new_mesh_pools \
  --manifest /data/new_manifest.jsonl --config veudf/configs/surface_pools.json \
  --workers 4 --threads 4 --timeout 7200
```

请使用新输出 root；旧 `edge_vertex_udf_v1` NPZ、历史体素／MC 条件场与当前数据不能混用。续跑相同 root 时，清单、配置、源码和依赖身份必须一致。已运行的冻结任务继续使用其自身的源码与环境，不能因同步仓库而替换运行中代码。

如果生产目录已安装同名 `veudf` 包，在本仓库测试请使用独立虚拟环境，避免改变生产环境的 editable 安装。同步副本没有修改生产环境的安装位置。

## 点池与输出

默认普通池 1,572,864 点，S/N/E_S/V_S 比例 25/35/25/15；S0 的固定几何前缀为 393,216 点。特征池在 CAD 边／顶点附近的最终网格表面采样，N 池按实际 `abs(s_mesh)≤0.03` 接受。另有 CAD 零集与独立 evaluation 池，普通池不包含全域体积监督；可选 volume 池仅保存 SDF。

```text
root/
  config.json / manifest.jsonl / identity.json
  assets/<id>/
    condition/{vertices.npy,triangles.npy,meta.json}
    pools/{surface,near_surface,edge_surface,vertex_surface,zero}/
    evaluation/<对应点池>/
    audit/ / topology.json / meta.json
  results/ / logs/ / index.json / report.json / status.json
```

点池使用独立 `.npy` 数组并支持 mmap；只有标签审计、去重及 split 检查通过的资产发布到 `index.json`。空通道保存 float32 `2√3` 并标记不存在，不使用旧 `.02` 占位或截断归一化读取接口。

## 入口与验证

| 模块入口 | 安装后的命令 |
|---|---|
| `python -m veudf.cli.build_dataset` | `veudf-build-dataset` |
| `python -m veudf.cli.prepare_condition` | `veudf-prepare-condition` |
| `python -m veudf.cli.prepare_remaining_run` | `veudf-prepare-remaining` |
| `python -m veudf.cli.run_guarded_build` | `veudf-guarded-build` |

增量候选冻结仅复制 `veudf` 包，保留 family split 和既有排除；资源保护是软停止，不是硬内存隔离。使用各命令 `--help` 查看参数。

```bash
python -m pytest veudf/tests -q
python scripts/sync_veudf.py --check
```

测试包含距离、筛选、网格条件、采样、资产构建、几何去重、发布／续跑、包依赖边界及脱离仓库的冻结任务。完整曲线拟合和 B-rep 装配仍由独立 `band_brep_reconstruction` 负责；当前生产点池没有自动注册到该模块的旧 `sampled` provider，不能直接替换其历史 NPZ 输入。
