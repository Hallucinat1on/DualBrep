"""Summarize the 2026-10-17 pipeline batch: variants x model sets, from result JSONs only."""
import argparse
from collections import Counter
import json
from pathlib import Path

DEV = ('all', 'base', 'no_vertex', 'no_surface', 'no_cad')
HOLD = ('all', 'base')


def read(p): return json.loads(Path(p).read_text())


def load(run, name, manifest_ids):
    """One row per planned model; missing result = process failure / timeout, never dropped."""
    proc = {Path(r['output']).name: r for r in read(run/f'{name}_processes.json')} if (run/f'{name}_processes.json').exists() else {}
    rows = {}
    for mid in manifest_ids:
        f = run/name/mid/'result.json'
        r = read(f) if f.exists() else {}
        p = proc.get(mid, {})
        full = r.get('full')
        if not r.get('baseline'):
            fail = run/name/mid/'failure.json'
            code = read(fail)['error']['code'] if fail.exists() else p.get('status', 'missing')
            stage = 'preprocess' if p.get('status') != 'timeout' else 'timeout'
            rows[mid] = dict(graph=False, face=False, step=False, stage=stage, code=code, fidelity=None)
        else:
            b = r['baseline']
            if full is None:
                stage, code = ('timeout', 'timeout') if p.get('status') == 'timeout' else ('pipeline', p.get('status', 'missing'))
            elif full['status'] == 'solid_validated':
                stage, code = 'STEP', 'STEP'
            else:
                stage, code = full.get('failed_stage'), full.get('error', {}).get('code')
            rows[mid] = dict(graph=b['graph_good'], face=b['face_good'], step=stage == 'STEP', stage=stage, code=code,
                             fidelity=(full or {}).get('fidelity'), inferred=len((r.get('short_edges') or {}).get('inferred', [])))
    return rows


def first_cause(row):
    if row['step']: return 'STEP'
    if row['stage'] in ('preprocess', 'timeout', 'pipeline'): return row['stage']
    if not row['face']: return 'face_segmentation'
    if not row['graph']: return 'edge_vertex_instances'
    return {'vertices': 'geometry_vertices', 'curves': 'geometry_curves', 'coverage': 'geometry_curves', 'postprocess': 'postprocess',
            'surfaces': 'surfaces', 'cad': 'cad'}.get(row['stage'], row['stage'])


def stats(rows):
    f = [r['fidelity'] for r in rows.values() if r['step'] and r['fidelity'] and 'volume_rel_error' in r['fidelity']]
    def q(key, p):
        v = sorted(x[key] for x in f)
        return v[min(len(v)-1, int(p*len(v)))] if v else None
    return dict(models=len(rows), graph=sum(r['graph'] for r in rows.values()), face=sum(r['face'] for r in rows.values()),
                step=sum(r['step'] for r in rows.values()), causes=dict(Counter(first_cause(r) for r in rows.values())),
                fidelity_n=len(f), vol_err_median=q('volume_rel_error', .5), vol_err_max=q('volume_rel_error', 1.),
                hd_median=q('recon_to_truth_max', .5), hd_max=max([max(x['recon_to_truth_max'], x['truth_to_recon_max']) for x in f], default=None),
                fidelity_flag=[m for m, r in rows.items() if r['step'] and r['fidelity'] and 'volume_rel_error' in r['fidelity']
                               and (r['fidelity']['volume_rel_error'] > .01 or max(r['fidelity']['recon_to_truth_max'], r['fidelity']['truth_to_recon_max']) > .01)])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); args = p.parse_args()
    root = args.root; old = Path('/opt/yihengxu/DualBrep/band_brep_reconstruction/outputs/plan_20261016/main')
    s30 = [m['id'] for m in read(old/'manifest.json')['models']]
    h100 = read(old/'holdout_manifest.json')['selected']; h100b = read(root/'holdout_b_manifest.json')['selected']
    strata = {**{r['id']: r['stratum'] for r in h100}, **{r['id']: r['stratum'] for r in h100b}}
    out = {'dev': {}, 'h100b': {}, 'per_model': {}}
    for v in DEV:
        run = root/f'dev_{v}'
        if not (run/'holdout_processes.json').exists(): continue
        rows = {('s30', m): r for m, r in load(run, 's30', s30).items()}
        rows.update({('h100', m): r for m, r in load(run, 'holdout', [x['id'] for x in h100]).items()})
        out['dev'][v] = {'s30': stats({m: r for (s, m), r in rows.items() if s == 's30'}),
                         'h100': stats({m: r for (s, m), r in rows.items() if s == 'h100'}),
                         'h100_strata': {k: stats({m: r for (s, m), r in rows.items() if s == 'h100' and strata[m] == k}) for k in range(4)}}
        out['per_model'][f'dev_{v}'] = {f'{s}/{m}': r for (s, m), r in rows.items()}
    for v in HOLD:
        run = root/f'h100b_{v}'
        if not (run/'holdout_processes.json').exists(): continue
        rows = load(run, 'holdout', [x['id'] for x in h100b])
        out['h100b'][v] = {'all': stats(rows), 'strata': {k: stats({m: r for m, r in rows.items() if strata[m] == k}) for k in range(4)}}
        out['per_model'][f'h100b_{v}'] = rows
    # Model-level flips between the full method and the base configuration.
    for a, b in (('dev_all', 'dev_base'), ('h100b_all', 'h100b_base')):
        if a in out['per_model'] and b in out['per_model']:
            A, B = out['per_model'][a], out['per_model'][b]
            out[f'flips_{a}'] = {'gained': sorted(m for m in A if A[m]['step'] and not B[m]['step']),
                                 'lost': sorted(m for m in A if B[m]['step'] and not A[m]['step']),
                                 'graph_gained': sorted(m for m in A if A[m]['graph'] and not B[m]['graph']),
                                 'graph_lost': sorted(m for m in A if B[m]['graph'] and not A[m]['graph'])}
    (root/'summary.json').write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str)+'\n')
    for k in ('dev', 'h100b'):
        for v, s in out[k].items():
            for name, x in s.items():
                if name in ('h100_strata', 'strata'):
                    print(k, v, name, {i: (y['graph'], y['step'], y['models']) for i, y in x.items()}); continue
                print(k, v, name, 'graph', x['graph'], 'STEP', x['step'], '/', x['models'], x['causes'],
                      'vol_med', x['vol_err_median'], 'hd_max', x['hd_max'], 'flag', x['fidelity_flag'])
    for k in [k for k in out if k.startswith('flips')]: print(k, out[k])


if __name__ == '__main__': main()
