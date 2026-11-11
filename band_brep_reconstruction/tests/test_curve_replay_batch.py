"""Reporting must retain failures and distinguish fixed topology from new STEP."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from run_curve_replay_batch import update_report, verify
from run_mesh_input_experiment import save, digest
import pytest


def test_partial_report_keeps_denominator_and_fixed_graph(tmp_path):
    group = tmp_path / 'old'
    out = tmp_path / 'new'
    rows = [dict(id=f'{i:08d}') for i in range(20)]
    save(out / 'manifest.json', rows)
    save(out / 'campaign.json', dict(input_group=str(group)))
    for i, row in enumerate(rows):
        save(group / 'records' / (row['id'] + '.json'), dict(
            status='solid_validated', joint_success=i != 2,
            score=dict(face_good=True, graph_good=i != 2)))
    for i, status, timings in [(0, 'solid_validated', dict(curve_recovery=10, global_coverage=2)),
                                (1, 'failed', {}),
                                (2, 'solid_validated', dict(curve_recovery=20, global_coverage=4))]:
        save(out / 'records' / (rows[i]['id'] + '.json'), dict(
            status=status, timings=timings, error=dict(code='curve_failed') if i == 1 else {},
            process=dict(peak_rss_bytes=1024, seconds=30)))
    summary = update_report(out)
    assert (summary['total'], summary['completed'], summary['pending']) == (20, 3, 17)
    assert (summary['cad_step'], summary['joint_success']) == (2, 1)
    assert summary['paired'] == dict(gained=[], lost=['00000001'], common=['00000000'])
    assert summary['failure_counts'] == dict(curve_failed=1)
    assert summary['median_completed_curve_and_coverage_seconds'] == 18


def test_frozen_input_change_is_rejected(tmp_path):
    source = tmp_path / 'source.json'
    source.write_text('{}')
    save(tmp_path / 'identity.json', dict(files={str(source): digest(source)}))
    verify(tmp_path)
    source.write_text('{"changed":true}')
    with pytest.raises(RuntimeError, match='Frozen batch input changed'):
        verify(tmp_path)
