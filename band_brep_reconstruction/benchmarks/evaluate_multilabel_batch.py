"""Offline evaluation of a completed datasets30 multi-label batch; no reconstruction edits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'datasets30_20261013'))
from worker import score


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline', required=True)
    args = parser.parse_args()
    out, baseline = Path(args.output).resolve(), Path(args.baseline).resolve()
    progress, previous = read(out / 'progress.json'), read(baseline / 'progress.json')
    if progress['state'] != 'complete':
        raise RuntimeError('Evaluate only after batch completes')
    fields = Path(progress['source_fields'])
    rows = []
    for record in progress['selected']:
        sid = record['sample_id']; folder = out / 'reconstruction' / sid
        result = progress['results'][sid]
        run = read(folder / 'run.json') if (folder / 'run.json').exists() else {}
        row = dict(model_id=record['model_id'], sample_id=sid, target=record['counts'],
                   status=result['status'], code=result.get('code'),
                   failed_stage=run.get('failed_stage'), error=run.get('error'),
                   wall_seconds=result['wall_seconds'], stages=run.get('stages', []),
                   config=run.get('config', {}),
                   warnings=result.get('quality_warnings', []),
                   previous={k: previous['results'][sid].get(k) for k in ('status', 'code', 'wall_seconds')})
        try:
            row['evaluation'] = score(fields / sid, folder)
        except Exception:
            row['evaluation_error'] = traceback.format_exc()
        if (folder / 'incidence.json').exists():
            incidence = read(folder / 'incidence.json')
            row['edge_face_cardinality'] = dict(Counter(len(e['faces']) for e in incidence['edges']))
            row['edge_vertex_cardinality'] = dict(Counter(len(e['vertices']) for e in incidence['edges']))
        if (folder / 'bands.npz').exists():
            with np.load(folder / 'bands.npz') as f:
                band_points = f['points']
                row['band_sampling'] = dict(nodes=len(f['points']), edge_nodes=int(f['edge_band'].sum()),
                    vertex_nodes=int(f['vertex_band'].sum()), min_edge_udf=float(f['values'][:, 1].min()),
                    coverage_probe_count=int((f['values'][:, 1] < run['config']['tau_edge']*.6).sum()))
                if 'exclusion_reasons' in f:
                    row['band_sampling']['exclusion_counts'] = dict(Counter(map(int, f['exclusion_reasons'])))
            if (folder / 'incidence.json').exists():
                from scipy.spatial import cKDTree
                with np.load(fields / sid / 'fields.npz') as f:
                    edge_points, edge_ids, vertex_points = f['edge_points'][:, :3], f['edge_point_id'], f['vertex_points']
                probes = {}
                for kind, domains, truth_points, truth_ids in (
                    ('edges', incidence['edges'], edge_points, edge_ids),
                    ('vertices', incidence['vertices'], vertex_points, np.arange(len(vertex_points)))):
                    entries = []
                    tree = cKDTree(truth_points) if len(truth_points) else None
                    for domain in domains:
                        nodes = np.asarray(domain['nodes'])
                        chosen = nodes[np.linspace(0, len(nodes)-1, min(512, len(nodes))).astype(int)]
                        labels = truth_ids[tree.query(band_points[chosen])[1]] if tree else []
                        histogram = Counter(map(int, labels))
                        substantial = [k for k, n in histogram.items() if n >= max(3, .05*len(labels))]
                        entries.append(dict(id=domain['id'], nodes=len(nodes), target_histogram=dict(histogram),
                                            substantial_targets=substantial))
                    probes[kind] = entries
                row['domain_truth_diagnostics'] = probes
                row['domain_truth_method'] = 'Offline nearest GT edge samples / vertex positions, at most 512 probes per domain; >=3 probes and >=5% defines substantial support. Diagnostic only, not exact topology certification.'
        if (folder / 'vertices.json').exists():
            from scipy.optimize import linear_sum_assignment
            vertices = read(folder / 'vertices.json')
            with np.load(fields / sid / 'fields.npz') as f:
                truth = f['vertex_points']
            pred = np.array([v['position'] for v in vertices]).reshape(-1, 3)
            distances = np.linalg.norm(pred[:, None] - truth[None], axis=2)
            a, b = linear_sum_assignment(distances)
            row['localized_vertices'] = dict(predicted=len(pred), target=len(truth),
                matched_at_003=int((distances[a, b] <= .003).sum()),
                assignment_distances=distances[a, b].tolist())
        rows.append(row)
        print(record['model_id'], row['status'], row.get('evaluation_error', 'scored'), flush=True)
    success = [r for r in rows if r['status'] == 'solid_validated']
    summary = dict(total=len(rows), success=len(success), success_ids=[r['model_id'] for r in success],
        previous_success=sum(r['previous']['status'] == 'solid_validated' for r in rows),
        failures=dict(Counter(r['code'] for r in rows if r['status'] != 'solid_validated')),
        failed_stages=dict(Counter(r['failed_stage'] for r in rows if r['status'] != 'solid_validated')),
        completed_stages=dict(Counter(s['stage'] for r in rows for s in r['stages'])),
        batch_wall_seconds=progress['finished_unix']-progress['started_unix'],
        per_model_median_seconds=float(np.median([r['wall_seconds'] for r in rows])),
        previous_per_model_median_seconds=float(np.median([r['previous']['wall_seconds'] for r in rows])),
        evaluation_errors=sum('evaluation_error' in r for r in rows))
    config = next((r['config'] for r in rows if r['config']), progress['config'])
    two_pass = 'face_threshold1' in config
    report = dict(summary=summary, rows=rows, baseline=str(baseline), source_fields=str(fields),
                  config=config, evaluated_unix=time.time())
    (out / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    def counts(value):
        return '/'.join(str(value[k]) for k in ('faces', 'edges', 'vertices')) if value else '—'
    text = ['# 30 个 ABC 模型：多标签隔离带方案重建评估', '',
        '## 实验设置', '',
        f'- 本轮结果：`{out}`', f'- 对照结果：`{baseline}`',
        f'- 复用数据：`{fields}`；沿用原清单中的 30 个模型，不重新筛选或采样。',
        '- 沿用原实验的连续 CAD oracle 查询器，先校验与保存三场样本的一致性；新增位置通过 oracle 求场值。不是仅对 NPZ 离散点插值，也不是生成场噪声鲁棒性测试。',
        '- 重建只读取标量场、梯度和存在性信息；真值面/边/顶点标签只用于本报告的离线评分。',
        '- 4 个并发 worker，无单模型超时；失败继续处理下一条。旧结果保留，本轮不修改重建算法、不调参救回失败样例。',
        (f"- DualBrep 两阶段面核心阈值 {config['face_threshold1']}/{config['face_threshold2']}，最小分量 {config['face_filter_size']}，平均 UDF 下限 {config['face_avg_udf_threshold']}。仅当第二阶段至少有两个合格子分量时替换父核心。顶点阈值 {config['tau_vertex']}；tau_edge={config['tau_edge']} 仅供既有几何定位和覆盖检查，不定义传播边带。" if two_pass else '- 边带阈值 0.025，顶点带阈值 0.04。'),
        '- 初始 SDF 分辨率 65，必要时重提到 129；提面后固定网格，不执行隔离带递归细分。距离均为归一化空间长度。',
        '- 使用与旧 continuation 实验一致的质量警告策略：部分场残差、方向和覆盖误差降为警告，结构错误仍失败。因此 solid_validated 表示 CAD 校验与 STEP 读回成功，不等于无场保真误差。',
        ('- 边带只包含因距离阈值排除的点；大小/均值过滤区域单独记录，不参与传播。多标签传播及后续参数化方法保持不变，未启用额外面合并/回填。' if two_pass else '- 面粘连/碎核心修复未启用；保留小核心、多标签候选，歧义在后续几何阶段报告。'), '',
        '## 整体结果', '',
        '| 指标 | 对照批次 | 当前批次 |', '|---|---:|---:|',
        f"| STEP 成功 | {summary['previous_success']}/30 | {summary['success']}/30 |",
        f"| 单模型总耗时中位数（秒） | {summary['previous_per_model_median_seconds']:.2f} | {summary['per_model_median_seconds']:.2f} |", '',
        f"本轮批次总墙钟时间 {summary['batch_wall_seconds']:.1f} 秒。耗时包括输入载入、oracle 校验和场查询，且两次运行的机器负载不受控，不能据此声称纯算法加速倍数。", '',
        '### 已完成阶段', '', '| 阶段 | 模型数 |', '|---|---:|']
    for stage in ('surface', 'bands', 'seeds', 'graph', 'vertices', 'curves', 'postprocess', 'surfaces', 'cad'):
        text.append(f"| {stage} | {summary['completed_stages'].get(stage, 0)} |")
    text += ['', 'graph 完成只表示候选实例和关联已保存，不表示关联正确；curves 完成包含覆盖校验，失败时也可能已保存部分曲线文件。', '',
             '### 首个失败原因', '', '| 错误码 | 数量 |', '|---|---:|']
    for code, n in summary['failures'].items():
        text.append(f'| `{code}` | {n} |')
    text += ['', '每个模型只记录首个阻断错误；失败数量不是各类问题的总发生数量。', '',
             '## 逐模型结果', '',
             'F/E/V 为面/边/顶点实例数量。候选数量相同不构成拓扑正确证明。', '',
             '| 模型 | 真值 F/E/V | 候选 F/E/V | 未决边 | 当前结果 | 旧结果 | 总秒数 |',
             '|---|---|---|---:|---|---|---:|']
    for r in rows:
        ev = r.get('evaluation', {})
        unresolved = len(ev.get('association_diagnostics', {}).get('unresolved_edges', []))
        text.append(f"| {r['model_id']} | {counts(r['target'])} | {counts(ev.get('association_counts'))} | {unresolved} | {r['code'] or r['status']} | {r['previous']['code'] or r['previous']['status']} | {r['wall_seconds']:.1f} |")
    text += ['', '## 成功 STEP 与离线核对', '']
    for r in success:
        ev = r.get('evaluation', {})
        text.append(f"- `{r['model_id']}`：[reconstruction.step]({out / 'reconstruction' / r['sample_id'] / 'reconstruction.step'})；实例数一致：{ev.get('count_agreement')}；边面/边点关联一致：{ev.get('edge_face_vertex_agreement')}；质量警告数：{len(r['warnings'])}。")
    text += ['', '离线面匹配每个核心最多取 128 个中心点，对真值网格做最近三角形匹配；薄层与交界附近可能有最近面误判。边距离为有限点云双向采样距离，包含真值点云间距误差。域混合诊断每个域最多取 512 个点，用最近真值边采样/顶点归属统计，不能代替精确拓扑认证。', '',
        '完整误差、面匹配混淆矩阵、域混合诊断、各阶段耗时和旧结果见同目录 `evaluation.json`；重建输入代码哈希见 `source_sha256.json`。', '',
        '## 复现与查询', '', '```bash',
        f'/miniconda/envs/brdf/bin/python {HERE / "datasets30_20261013/continuous_batch.py"} --output {out} --status',
        '```', '', '运行同一个批次脚本并指定新的空输出目录可重做实验。不要向本次已存在的输出目录重启批次。', '']
    (out / 'REPORT.md').write_text('\n'.join(text))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
