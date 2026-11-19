"""Batch accounting distinguishes selected fallback from actual merge success."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from run_region_repair_experiment import report,verify
from run_mesh_input_experiment import save,digest


def test_pending_and_fallbacks_are_reported_separately(tmp_path):
    old=tmp_path/'old';comparison=tmp_path/'comparison';out=tmp_path/'batch'
    rows=[dict(id=f'{i:08d}') for i in range(20)]
    save(out/'manifest.json',rows);save(out/'campaign.json',dict(input_group=str(old),comparison=str(comparison),jobs=2))
    for row in rows:
        save(old/'records'/(row['id']+'.json'),dict(score=dict(graph_good=True)))
        save(comparison/'records'/(row['id']+'.json'),dict(status='solid_validated'))
    record=dict(status='solid_validated',joint_success=True,effective_mode='off',
        score=dict(graph_good=True,face_good=True),process=dict(seconds=10),
        repair_attempts=[dict(status='failed',mode='joint'),dict(selected=True,mode='off')])
    save(out/'records'/'00000000'/'joint.json',record)
    result=report(out)
    assert result['completed']==1
    assert result['groups']['joint']['pending']==19
    assert result['groups']['joint']['successful_fallbacks']==1
    assert result['groups']['joint']['historical_lost']==[]
    assert result['groups']['baseline']['completed']==0
    assert result['groups']['joint']['baseline_gained']==[]


def test_snapshot_tampering_is_rejected(tmp_path):
    source=tmp_path/'source_snapshot'/'library.py';source.parent.mkdir();source.write_text('x=1\n')
    save(tmp_path/'identity.json',dict(common={},cases={},code={'library.py':digest(source)}))
    verify(tmp_path);source.write_text('x=2\n')
    with pytest.raises(RuntimeError,match='Frozen input or source changed'):verify(tmp_path)
