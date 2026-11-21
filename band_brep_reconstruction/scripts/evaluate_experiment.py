"""Evaluate a batch reconstruction experiment: success rates, per-stage pass rates,
V/E/F count & topology accuracy, and geometric fidelity metrics.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


STAGES = ('scout', 'wireframe', 'split_mesh', 'surfaces', 'cad')


def _stats(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(arr):
        return dict(count=0, mean=None, median=None, p95=None, max=None)
    return dict(
        count=int(len(arr)),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        p95=float(np.quantile(arr, 0.95)),
        max=float(arr.max()),
    )


def _prf(matched, missing, duplicated, unmatched):
    gt = matched + missing
    pred = matched + duplicated + unmatched
    prec = matched / pred if pred > 0 else (1.0 if gt == 0 else 0.0)
    rec = matched / gt if gt > 0 else (1.0 if pred == 0 else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, gt, pred


def _load_group_records(exp_dir: Path):
    """Discover groups and per-model folders from results.json, state.json, or directory layout."""
    manifest = {}
    manifest_file = exp_dir / 'manifest.json'
    if manifest_file.is_file():
        for row in json.loads(manifest_file.read_text()):
            manifest[row['id']] = row

    records_by_group = {}
    res_file = exp_dir / 'results.json'
    state_file = exp_dir / 'state.json'
    if res_file.is_file():
        rows = json.loads(res_file.read_text())
    elif state_file.is_file():
        rows = json.loads(state_file.read_text()).get('records', [])
    else:
        rows = []

    if rows:
        for r in rows:
            grp = r.get('group', 'default')
            mid = r['model']
            folder = exp_dir / grp / mid if (exp_dir / grp / mid).is_dir() else exp_dir / mid
            records_by_group.setdefault(grp, []).append((mid, folder, r, manifest.get(mid)))
        for grp in records_by_group:
            records_by_group[grp].sort(key=lambda x: x[0])
        return records_by_group

    # Fallback: single run directory or directory of model subdirectories.
    if (exp_dir / 'run.json').is_file():
        return {'default': [(exp_dir.name, exp_dir, {}, None)]}
    for sub in sorted(exp_dir.iterdir()):
        if not sub.is_dir() or sub.name in ('source_snapshot', 'mc_vis'):
            continue
        if (sub / 'reconstruction/run.json').is_file() or (sub / 'run.json').is_file():
            records_by_group.setdefault('default', []).append((sub.name, sub, {}, manifest.get(sub.name)))
        else:
            for mdir in sorted(sub.iterdir()):
                if mdir.is_dir() and ((mdir / 'reconstruction/run.json').is_file() or (mdir / 'run.json').is_file()):
                    records_by_group.setdefault(sub.name, []).append((mdir.name, mdir, {}, manifest.get(mdir.name)))
    return records_by_group


def _truth_counts_from_step(step_path: Path):
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from veudf.preprocess.cad import collect_targets, read_normalized
    from veudf.preprocess.config import Config as PreConfig

    cfg = PreConfig().validate()
    solid, _ = read_normalized(str(step_path), cfg)
    targets = collect_targets(solid, cfg)
    exp = TopExp_Explorer(solid, TopAbs_FACE)
    n_faces = 0
    while exp.More():
        n_faces += 1
        exp.Next()
    return dict(vertices=len(targets.vertices), edges=len(targets.edges), faces=n_faces)


def evaluate_group(entries, *, with_truth=False):
    n = len(entries)
    stage_passed = Counter()
    wf_clean = 0
    wf_zero_unres = 0
    step_ok = 0
    graph_good = 0
    timeouts = 0
    fail_breakdown = Counter()

    elapsed_list = []
    scout_pts_list = []
    scout_sec_list = []

    v_exact = e_exact = f_exact = vef_exact = 0
    v_exact_all = e_exact_all = 0
    truth_checked_all = 0

    topo_metrics = {k: dict(matched=0, missing=0, duplicated=0, unmatched=0, prec=[], rec=[], f1=[])
                    for k in ('vertices', 'edges', 'faces')}
    ev_acc, ef_acc = [], []

    geom_lists = {k: [] for k in (
        'vertex_nearest_max', 'curve_to_truth_max', 'truth_to_curve_sampled_max', 'curve_sdf_max',
        'volume_rel_error', 'recon_to_truth_mean', 'truth_to_recon_mean', 'sym_chamfer_mean',
        'recon_to_truth_p95', 'truth_to_recon_p95', 'recon_to_truth_max', 'truth_to_recon_max',
    )}

    per_model = []
    for mid, folder, rec, m_info in entries:
        run_dir = folder / 'reconstruction' if (folder / 'reconstruction').is_dir() else folder
        run_file = run_dir / 'run.json'
        run = json.loads(run_file.read_text()) if run_file.is_file() else {}
        score_file = folder / 'score.json'
        sc = json.loads(score_file.read_text()) if score_file.is_file() else None

        status = rec.get('status') or ('success' if run.get('status') == 'solid_validated' else run.get('status', 'unknown'))
        if status == 'timeout':
            timeouts += 1
        is_ok = (status == 'success') or (run.get('status') == 'solid_validated')
        if is_ok:
            step_ok += 1

        stages = run.get('stages') or rec.get('stages') or {}
        failed_stage = run.get('failed_stage')
        err = run.get('error') or rec.get('error') or {}
        err_code = err.get('code', '') if isinstance(err, dict) else str(err)

        for st in STAGES:
            if st in stages and failed_stage != st:
                stage_passed[st] += 1
            elif st == 'cad' and is_ok:
                stage_passed[st] += 1
        # If fast_faces succeeded, split_mesh & surfaces were bypassed validly.
        if run.get('face_method') == 'coarse_projection' and is_ok:
            for st in ('split_mesh', 'surfaces'):
                if st not in stages:
                    stage_passed[st] += 1

        wf = stages.get('wireframe') or {}
        if 'wireframe' in stages and failed_stage != 'wireframe':
            if not wf.get('incomplete', True):
                wf_clean += 1
            if len(wf.get('unresolved', [])) == 0:
                wf_zero_unres += 1

        if not is_ok:
            fail_breakdown[f'{failed_stage or rec.get("last_stage") or "unknown"}:{err_code or status}'] += 1

        elapsed = rec.get('elapsed_seconds') or run.get('elapsed_seconds')
        if elapsed is not None:
            elapsed_list.append(float(elapsed))
        scout_st = stages.get('scout') or {}
        if 'unique_query_points' in scout_st:
            scout_pts_list.append(int(scout_st['unique_query_points']))
        if 'seconds' in scout_st:
            scout_sec_list.append(float(scout_st['seconds']))

        row_out = dict(
            model=mid,
            status='success' if is_ok else status,
            failed_stage=failed_stage,
            error_code=err_code if not is_ok else '',
            elapsed_s=round(float(elapsed), 1) if elapsed is not None else None,
            wf_v=wf.get('vertices'),
            wf_e=wf.get('edges'),
            wf_unres=len(wf.get('unresolved', [])) if wf else None,
        )

        if sc is not None:
            topo = sc.get('topology', {})
            geom = sc.get('geometry', {})
            fid = sc.get('fidelity', {})
            gg = bool(topo.get('graph_good', False))
            if gg:
                graph_good += 1

            ok_v = ok_e = ok_f = False
            for key in ('vertices', 'edges', 'faces'):
                d = topo.get(key, {})
                m_cnt = int(d.get('matched', 0))
                mi = int(d.get('missing', 0))
                du = int(d.get('duplicated', 0))
                un = int(d.get('unmatched', 0))
                prec, rec_v, f1, gt_n, pred_n = _prf(m_cnt, mi, du, un)
                tm = topo_metrics[key]
                tm['matched'] += m_cnt
                tm['missing'] += mi
                tm['duplicated'] += du
                tm['unmatched'] += un
                tm['prec'].append(prec)
                tm['rec'].append(rec_v)
                tm['f1'].append(f1)
                exact = (gt_n == pred_n) and (mi == 0) and (du == 0) and (un == 0)
                if key == 'vertices':
                    ok_v = (gt_n == pred_n)
                    v_exact += int(ok_v)
                    row_out['v_pred_gt'] = f'{pred_n}/{gt_n}'
                elif key == 'edges':
                    ok_e = (gt_n == pred_n)
                    e_exact += int(ok_e)
                    row_out['e_pred_gt'] = f'{pred_n}/{gt_n}'
                    if gt_n > 0:
                        ev_acc.append(int(topo.get('edge_vertex_matches', 0)) / gt_n)
                        ef_acc.append(int(topo.get('edge_face_matches', 0)) / gt_n)
                elif key == 'faces':
                    ok_f = (gt_n == pred_n)
                    f_exact += int(ok_f)
                    row_out['f_pred_gt'] = f'{pred_n}/{gt_n}'
            vef_exact += int(ok_v and ok_e and ok_f)
            row_out['graph_good'] = gg

            for k in ('vertex_nearest_max', 'curve_to_truth_max', 'truth_to_curve_sampled_max', 'curve_sdf_max'):
                if k in geom and geom[k] is not None:
                    geom_lists[k].append(float(geom[k]))
            if fid:
                if 'volume_rel_error' in fid:
                    geom_lists['volume_rel_error'].append(float(fid['volume_rel_error']))
                r2t = fid.get('recon_to_truth_mean')
                t2r = fid.get('truth_to_recon_mean')
                if r2t is not None:
                    geom_lists['recon_to_truth_mean'].append(float(r2t))
                if t2r is not None:
                    geom_lists['truth_to_recon_mean'].append(float(t2r))
                if r2t is not None and t2r is not None:
                    sym = 0.5 * (float(r2t) + float(t2r))
                    geom_lists['sym_chamfer_mean'].append(sym)
                    row_out['chamfer_mean'] = sym
                for side in ('recon_to_truth', 'truth_to_recon'):
                    q = fid.get(f'{side}_quantiles') or {}
                    if 'p95' in q:
                        geom_lists[f'{side}_p95'].append(float(q['p95']))
                    if f'{side}_max' in fid:
                        geom_lists[f'{side}_max'].append(float(fid[f'{side}_max']))
                row_out['vol_rel_err'] = fid.get('volume_rel_error')

        if with_truth and m_info and m_info.get('step'):
            tc = _truth_counts_from_step(Path(m_info['step']))
            truth_checked_all += 1
            if wf.get('vertices') == tc['vertices']:
                v_exact_all += 1
            if wf.get('edges') == tc['edges']:
                e_exact_all += 1
            row_out.setdefault('v_pred_gt', f'{wf.get("vertices")}/{tc["vertices"]}')
            row_out.setdefault('e_pred_gt', f'{wf.get("edges")}/{tc["edges"]}')

        per_model.append(row_out)

    topo_summary = {}
    for key, tm in topo_metrics.items():
        mp, mr, mf1, gt_tot, pred_tot = _prf(tm['matched'], tm['missing'], tm['duplicated'], tm['unmatched'])
        topo_summary[key] = dict(
            micro_precision=mp,
            micro_recall=mr,
            micro_f1=mf1,
            macro_f1=float(np.mean(tm['f1'])) if tm['f1'] else None,
            gt_total=gt_tot,
            pred_total=pred_tot,
            matched=tm['matched'],
            missing=tm['missing'],
            duplicated=tm['duplicated'],
            unmatched=tm['unmatched'],
        )

    summary = dict(
        total=n,
        step_success=step_ok,
        step_success_rate=step_ok / n if n else 0.0,
        graph_good=graph_good,
        graph_good_rate_all=graph_good / n if n else 0.0,
        graph_good_rate_on_success=graph_good / step_ok if step_ok else 0.0,
        timeouts=timeouts,
        stage_pass_counts={st: stage_passed[st] for st in STAGES},
        stage_pass_rates={st: stage_passed[st] / n if n else 0.0 for st in STAGES},
        wireframe_clean=wf_clean,
        wireframe_clean_rate=wf_clean / n if n else 0.0,
        wireframe_zero_unresolved=wf_zero_unres,
        wireframe_zero_unresolved_rate=wf_zero_unres / n if n else 0.0,
        count_exact_on_success=dict(
            vertices=v_exact,
            edges=e_exact,
            faces=f_exact,
            all_vef=vef_exact,
            vertices_rate=v_exact / step_ok if step_ok else 0.0,
            edges_rate=e_exact / step_ok if step_ok else 0.0,
            faces_rate=f_exact / step_ok if step_ok else 0.0,
            all_vef_rate=vef_exact / step_ok if step_ok else 0.0,
        ),
        topology=topo_summary,
        incidence=dict(
            edge_vertex_match_rate=float(np.mean(ev_acc)) if ev_acc else None,
            edge_face_match_rate=float(np.mean(ef_acc)) if ef_acc else None,
        ),
        geometry={k: _stats(v) for k, v in geom_lists.items()},
        performance=dict(
            elapsed_seconds=_stats(elapsed_list),
            scout_seconds=_stats(scout_sec_list),
            scout_unique_points=_stats(scout_pts_list),
        ),
        failures=dict(fail_breakdown.most_common()),
        per_model=per_model,
    )
    if truth_checked_all:
        summary['count_exact_all_models'] = dict(
            checked=truth_checked_all,
            vertices=v_exact_all,
            edges=e_exact_all,
            vertices_rate=v_exact_all / truth_checked_all,
            edges_rate=e_exact_all / truth_checked_all,
        )
    return summary


def _fmt(val, pct=False, sci=False):
    if val is None:
        return '-'
    if pct:
        return f'{100.0 * val:.1f}%'
    if sci:
        return f'{val:.2e}'
    return f'{val:.5f}'


def print_report(exp_name, group_summaries, show_per_model=False):
    print(f'================ Experiment Evaluation: {exp_name} ================')
    for grp, s in group_summaries.items():
        n = s['total']
        ok = s['step_success']
        print(f'\n--- Group: {grp} (N={n}) ---')
        print(f'  STEP Success        : {ok}/{n} ({_fmt(s["step_success_rate"], pct=True)}) | Timeouts: {s["timeouts"]}')
        print(f'  Strict Graph Good   : {s["graph_good"]}/{n} ({_fmt(s["graph_good_rate_all"], pct=True)} overall, '
              f'{_fmt(s["graph_good_rate_on_success"], pct=True)} of STEP-valid)')
        print('  Per-Stage Pass Rate : ' + ' | '.join(
            f'{st}={s["stage_pass_counts"][st]}/{n} ({_fmt(s["stage_pass_rates"][st], pct=True)})' for st in STAGES
        ))
        print(f'  Wireframe Quality   : clean={s["wireframe_clean"]}/{n} ({_fmt(s["wireframe_clean_rate"], pct=True)}), '
              f'zero_unresolved={s["wireframe_zero_unresolved"]}/{n} ({_fmt(s["wireframe_zero_unresolved_rate"], pct=True)})')

        ce = s['count_exact_on_success']
        print(f'  Count Exact (STEP)  : V={ce["vertices"]}/{ok} ({_fmt(ce["vertices_rate"], pct=True)}) | '
              f'E={ce["edges"]}/{ok} ({_fmt(ce["edges_rate"], pct=True)}) | '
              f'F={ce["faces"]}/{ok} ({_fmt(ce["faces_rate"], pct=True)}) | '
              f'All={ce["all_vef"]}/{ok} ({_fmt(ce["all_vef_rate"], pct=True)})')
        if 'count_exact_all_models' in s:
            ca = s['count_exact_all_models']
            print(f'  Count Exact (All)   : V={ca["vertices"]}/{ca["checked"]} ({_fmt(ca["vertices_rate"], pct=True)}) | '
                  f'E={ca["edges"]}/{ca["checked"]} ({_fmt(ca["edges_rate"], pct=True)})')

        print('  Topology Precision / Recall / F1 (micro, on STEP-valid):')
        for key in ('vertices', 'edges', 'faces'):
            t = s['topology'][key]
            print(f'    {key:8s}: P={_fmt(t["micro_precision"], pct=True)} R={_fmt(t["micro_recall"], pct=True)} '
                  f'F1={_fmt(t["micro_f1"], pct=True)} (matched={t["matched"]}, miss={t["missing"]}, '
                  f'dup={t["duplicated"]}, unmat={t["unmatched"]})')
        inc = s['incidence']
        print(f'    Incidence: E-V={_fmt(inc["edge_vertex_match_rate"], pct=True)} | E-F={_fmt(inc["edge_face_match_rate"], pct=True)}')

        g = s['geometry']
        print('  Geometric & Fidelity Accuracy (median / mean / p95 / max, on STEP-valid):')
        for label, k in (
            ('Vertex Max Err    ', 'vertex_nearest_max'),
            ('Curve->Truth Max  ', 'curve_to_truth_max'),
            ('Truth->Curve Max  ', 'truth_to_curve_sampled_max'),
            ('Sym Chamfer Mean  ', 'sym_chamfer_mean'),
            ('Recon->Truth p95  ', 'recon_to_truth_p95'),
            ('Truth->Recon p95  ', 'truth_to_recon_p95'),
            ('Volume Rel Error  ', 'volume_rel_error'),
        ):
            st = g[k]
            print(f'    {label}: med={_fmt(st["median"])}  mean={_fmt(st["mean"])}  p95={_fmt(st["p95"])}  max={_fmt(st["max"])}')

        perf = s['performance']
        el = perf['elapsed_seconds']
        sq = perf['scout_unique_points']
        print(f'  Performance         : elapsed mean={el["mean"]:.1f}s (med={el["median"]:.1f}s) | '
              f'scout unique pts mean={(sq["mean"] or 0)/1e6:.2f}M' if el['mean'] is not None else '  Performance: -')
        if s['failures']:
            print(f'  Failure Breakdown   : {s["failures"]}')

        if show_per_model:
            print('\n  Per-Model Breakdown:')
            print(f'    {"Model":10s} {"Status":8s} {"GG":5s} {"V(p/g)":9s} {"E(p/g)":9s} {"F(p/g)":9s} {"Chamfer":9s} {"VolErr":9s} {"Time":6s} {"Failure"}')
            for r in s['per_model']:
                gg_s = 'True' if r.get('graph_good') else ('False' if 'graph_good' in r else '-')
                ch_s = f'{r["chamfer_mean"]:.5f}' if r.get('chamfer_mean') is not None else '-'
                ve_s = f'{r["vol_rel_err"]:.5f}' if r.get('vol_rel_err') is not None else '-'
                tm_s = f'{r["elapsed_s"]:.0f}s' if r.get('elapsed_s') is not None else '-'
                fl_s = f'{r["failed_stage"]}:{r["error_code"]}' if r['status'] != 'success' else ''
                print(f'    {r["model"]:10s} {r["status"]:8s} {gg_s:5s} '
                      f'{str(r.get("v_pred_gt", "-")):9s} {str(r.get("e_pred_gt", "-")):9s} {str(r.get("f_pred_gt", "-")):9s} '
                      f'{ch_s:9s} {ve_s:9s} {tm_s:6s} {fl_s}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment_dir', type=Path, help='Experiment output directory (e.g. .../projection_thinning_paired_20261120_v4)')
    parser.add_argument('--with-truth', action='store_true',
                        help='Also load source STEP files from manifest.json to evaluate V/E counts on failed models')
    parser.add_argument('--per-model', action='store_true', help='Print per-model detailed table')
    parser.add_argument('--out', type=Path, default=None, help='Optional path to write full JSON summary')
    args = parser.parse_args()

    groups = _load_group_records(args.experiment_dir)
    if not groups:
        raise SystemExit(f'No reconstruction runs found in {args.experiment_dir}')
    summaries = {grp: evaluate_group(entries, with_truth=args.with_truth) for grp, entries in groups.items()}
    print_report(args.experiment_dir.name, summaries, show_per_model=args.per_model)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summaries, indent=2) + '\n')


if __name__ == '__main__':
    main()
