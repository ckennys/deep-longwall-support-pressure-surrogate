"""Reconstruct five frozen field components from an input NPZ."""
import argparse
from pathlib import Path
import numpy as np
from longwall_surrogate.inference import ROOT, read_npz, predict_fields

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT/'examples/field_inputs_10.npz')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--targets', nargs='+', default=['ux_m', 'uz_m', 'sxx_MPa', 'syy_MPa', 'szz_MPa'])
    p.add_argument('--no-guard', action='store_true', help='Displacement diagnostic only; disable step-5 coefficient clipping.')
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new output path; existing files are not overwritten.')
    result = predict_fields(read_npz(a.input), a.targets, apply_guard=not a.no_guard)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **result)
    print(f'Saved {len(result["features"])} snapshots to {a.output}')

if __name__ == '__main__':
    main()
