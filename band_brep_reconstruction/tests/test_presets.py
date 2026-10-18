"""The 'recommended' preset: loading/precedence rules and analytic end-to-end STEP."""
import tempfile
import unittest
from pathlib import Path
from bandbrep.config import Config, PRESETS, load_config
from bandbrep.demo_fields import box, cylinder, sphere_with_bore, split_sphere, hollow_sphere
from bandbrep.reconstruct import reconstruct


class PresetTests(unittest.TestCase):
    def test_precedence_and_validation(self):
        cfg = load_config(preset='recommended')
        for key, value in PRESETS['recommended'].items():
            self.assertEqual(getattr(cfg, key), value)
        self.assertEqual(load_config(preset='recommended', overrides={'resolution': 129, 'max_resolution': 129}).resolution, 129)
        # Library defaults stay the historical configuration.
        self.assertEqual(Config().vertex_domains, 'threshold')
        with self.assertRaises(ValueError): load_config(preset='fast')
        with self.assertRaises(ValueError): load_config(overrides={'project_face_samples': True})

    def test_analytic_shapes_reach_validated_step(self):
        # Default 0.005/0.01 thresholds at 257^3: all five analytic shapes must pass
        # kernel checks and STEP read-back (about 30 s in total).
        cfg = load_config(preset='recommended')
        with tempfile.TemporaryDirectory() as tmp:
            for name, field in (('box', box), ('cylinder', cylinder), ('sphere_bore', sphere_with_bore),
                                ('split_sphere', split_sphere), ('hollow_sphere', hollow_sphere)):
                with self.subTest(shape=name):
                    reconstruct(field(), cfg, Path(tmp)/name)
                    self.assertTrue((Path(tmp)/name/'reconstruction.step').exists())


if __name__ == '__main__': unittest.main()
