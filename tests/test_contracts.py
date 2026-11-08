"""Lightweight public-interface checks; no model weights or network needed."""
from pathlib import Path
import io, importlib.util, unittest, zipfile
import numpy as np
from longwall_surrogate.inference import field_inputs, nearest_pressure, read_npz

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_archive', ROOT/'scripts/prepare_archive.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)

class ContractChecks(unittest.TestCase):
    def test_real_example_has_ten_canonical_snapshots(self):
        x, mask, steps = field_inputs(read_npz(ROOT/'examples/field_inputs_10.npz'))
        self.assertEqual(x.shape,(10,14))
        self.assertEqual(mask.shape,(10,5504))
        self.assertEqual(set(steps),{1,2,3,4,5})
        np.testing.assert_allclose(x[:,3],.024*(x[:,0]-85),rtol=2e-6,atol=2e-6)

    def test_misordered_feature_names_rejected(self):
        data = read_npz(ROOT/'examples/field_inputs_10.npz')
        data['feature_names'] = data['feature_names'][::-1]
        with self.assertRaises(ValueError):
            field_inputs(data)

    def test_all_pressure_midpoints_choose_lower_candidate(self):
        np.testing.assert_allclose(nearest_pressure(np.array([.65,.95,1.25,1.55,1.85,2.15,2.45])),[.5,.8,1.1,1.4,1.7,2.,2.3],rtol=0,atol=1e-12)

    def test_installer_rejects_parent_traversal(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream,'w') as archive:
            archive.writestr('longwall-surrogate-v2.0.0/../../outside.txt','not extracted')
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            with self.assertRaises(ValueError):
                list(installer.safe_members(archive))

    def test_installer_rejects_other_version_layout(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream,'w') as archive:
            archive.writestr('legacy-v1.0.0/data.csv','not extracted')
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            with self.assertRaises(ValueError):
                list(installer.safe_members(archive))

if __name__ == '__main__':
    unittest.main()
