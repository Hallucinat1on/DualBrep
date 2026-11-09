import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from run_mesh_input_experiment import configuration
from run_coarse_fine_comparison import update_report


class BenchmarkPolicyTests(unittest.TestCase):
    def test_single_pass_fine_target_is_explicit_and_other_settings_are_retained(self):
        for coarse in (.02,.03):
            cfg=configuration(coarse_edge=coarse,fine_edge=.003,
                              face_threshold1=.003,face_second_pass=False).validate()
            self.assertEqual((cfg.refinement_coarse_edge,cfg.refinement_target_edge),(coarse,.003))
            self.assertEqual((cfg.face_threshold1,cfg.face_second_pass),(.003,False))
            self.assertEqual((cfg.feature_sdf_tolerance,cfg.query_batch_size,cfg.save_queries),(.004,65536,False))
        self.assertTrue(configuration().face_second_pass)
        with self.assertRaises(ValueError):configuration(fine_edge=.04).validate()

    def test_report_uses_frozen_settings_and_pairs_with_historical_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'new';old=root/'old'
            def save(path,data):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(data))
            policy=dict(groups={'coarse_002':.02,'coarse_003':.03},dev=20,
                        fine_edge=.003,face_threshold1=.003,face_second_pass=False,
                        refinement_selection='edge_udf_only',historical_comparison=str(old))
            save(out/'campaign.json',policy)
            current=dict(id='case',status='failed',process=dict(seconds=10),joint_success=False,
                         score=dict(face_good=False,graph_good=False))
            historical=dict(id='case',joint_success=True,score=dict(face_good=True,graph_good=True))
            for group in policy['groups']:
                save(out/group/'policy.json',dict(group='mesh_coarse_fine'))
                save(out/group/'records/case.json',current)
                save(out/group/'manifest.json',[dict(id='case',cohort='dev')])
                save(old/group/'records/case.json',historical)
            metrics=dict(total=20,completed=1,prepared=1,face_good=0,graph_good=0,cad_step=0,joint_success=0)
            with patch('run_coarse_fine_comparison.report',return_value=dict(
                    groups=dict(mesh_coarse_fine=metrics),failure_counts={})):
                result=update_report(out)
            self.assertEqual(result['completed'],2)
            for group in policy['groups']:
                delta=result['groups'][group]['historical_comparison']
                self.assertEqual(delta['changes']['joint_success']['lost'],['case'])
                self.assertEqual(delta['paired_completed'],1)
            text=(out/'REPORT.md').read_text()
            self.assertIn('细目标 0.003',text)
            self.assertIn('第二阶段关闭',text)
            self.assertIn('仅E场选区',text)
