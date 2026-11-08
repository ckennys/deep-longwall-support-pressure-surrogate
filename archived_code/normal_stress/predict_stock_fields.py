from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np

from search_target import combine
from stock_models import predict_field

OUT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Frozen expanded-data field candidate inference; no solver or reference fields.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', nargs='+', default=['ux_m', 'uz_m', 'sxx_MPa', 'syy_MPa',
                                                       'szz_MPa', 'sxy_MPa', 'sxz_MPa', 'syz_MPa'])
    args = parser.parse_args()
    manifest = json.loads((OUT / 'data-manifest.json').read_text(encoding='utf-8'))
    assert set(args.target) <= set(manifest['targets'])
    assert args.input.resolve() != args.output.resolve()
    assert not args.output.exists(), 'Output already exists; choose a new path.'
    raw = np.load(args.input, allow_pickle=False)
    x = np.asarray(raw['features'], dtype=float)
    assert x.ndim == 2 and x.shape[1] == 14 and len(x) > 0 and np.isfinite(x).all()
    names = manifest['feature_names']
    if 'feature_names' in raw:
        assert raw['feature_names'].astype(str).tolist() == names, 'Feature order mismatch.'
    else:
        registered = {Path(b['input_file']).resolve() for b in manifest['holdouts']}
        assert args.input.resolve() in registered, 'Custom inputs require feature_names.'
    valid = np.asarray(raw['valid_mask'], dtype=bool).reshape(len(x), -1)
    assert valid.shape == (len(x), 5504) and np.all(valid.any(axis=1))
    step = np.rint(5 * x[:, -1]).astype(int)
    assert set(step) <= {1, 2, 3, 4, 5}
    assert np.allclose(x[:, -1], step / 5, rtol=0, atol=1e-6)
    record = {'model_type': 'Frozen stock-data research candidates', 'targets': args.target,
              'input_file': str(args.input.resolve()), 'input_sha256': digest(args.input),
              'snapshots': len(x), 'valid_points': int(valid.sum()), 'model_sha256': {},
              'reference_fields_read': False, 'new_flac3d_runs': 0,
              'adoption_record': str(OUT / 'adoption-recommendations.json')}
    fields = {}
    for target in args.target:
        folder = OUT / 'runs' / target
        completed = json.loads((folder / 'completed.json').read_text(encoding='utf-8'))
        path = folder / 'optimized_field_model.joblib'
        actual = digest(path)
        assert actual == completed['model_sha256'], 'Model hash mismatch.'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = joblib.load(path)
        components = [predict_field(package, x, valid) for package in model['models']]
        values = combine(components, model['rule'], step)
        assert np.isfinite(values[valid]).all()
        values[~valid] = np.nan
        fields[target] = values.reshape(len(x), 64, 86)
        record['model_sha256'][target] = actual
        del model, components
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=x, feature_names=np.asarray(names),
                        valid_mask=valid.reshape(len(x), 64, 86), step_index=step, **fields)
    args.output.with_suffix('.provenance.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
