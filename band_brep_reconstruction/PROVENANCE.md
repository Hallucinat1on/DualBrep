# 复用源码来源

本目录不依赖上级项目的 Python 导入。下列文件直接复制，便于迁移。日期：2026-10-13。

| 本地副本 | 原仓库来源 | SHA-256 |
|---|---|---|
| `bandbrep/vendor/oracle/__init__.py` | `../edge_vertex_preprocessing/evudf/__init__.py` | `7e5b48636d3e8be2d82e13fee5f4d21409bbd77ffbeb6355f95994adc1d57bcd` |
| `bandbrep/vendor/oracle/config.py` | `../edge_vertex_preprocessing/evudf/config.py` | `878daf15fe8ddb3d6abe453486d526d0bac64193503a730d5c47766719934c88` |
| `bandbrep/vendor/oracle/cad.py` | `../edge_vertex_preprocessing/evudf/cad.py` | `a09ada89f1ce1f228caf6263a19aa8ffda817ca21b1f07b2bf65526cba53d1fd` |
| `bandbrep/vendor/oracle/distance.py` | `../edge_vertex_preprocessing/evudf/distance.py` | `db3ba2722e414fbaa0b1203746ccc16a279b825499a2956da8d87eeea083e33d` |
| `bandbrep/vendor/dualbrep_model.py` | `../rebuild_model.py` | `ca6839907f905d5ee1ff0a2ffe6cfa9b25b131c758dfacbabb8e89faaf75c760` |

其余代码为本次重建实现。`face_seeds.py` 的共享边诱导子图与连通分量思路改编自 `clustering.py::hierarchical_segmentation`；`surface_fitting.py` 的采样格式改编自 `rebuild.py::sample_faces`，替换 Open3D 为本地 NumPy 面积采样并支持单面；规则曲面到 OCC 的构造组织借鉴 `brep_post/utils.py::create_surface`，没有引入其重新预测边、贪心串 wire 或无约束几何修复入口。

复制源码的原项目权利与许可仍适用；这里没有为原源码另行指定或改写许可。

## 2026-10-15 两阶段分割副本

`bandbrep/vendor/dualbrep_segmentation.py` 直接复制 `clustering.py::hierarchical_segmentation`，仅增加排除原因、成功拆分父域和统计输出，标签决策保持一致。原文件 SHA-256：`c76ce8b8022691fc003b1bbb1229aa9c175d4d23a10267da59865d9f79267432`。实际运行参数采用原脚本调用值 0.005/0.01、最小分量 10、平均 UDF 下限 0.003，不采用函数签名里不同的默认值。运行时不导入原仓库；仅可选测试通过 AST 提取原函数核对标签。
