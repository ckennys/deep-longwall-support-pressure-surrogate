"""Recompute field metrics from deposited predictions and qualified references."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from longwall_surrogate.inference import ROOT, read_npz

def metric(truth, pred, mask):
    a, b = np.asarray(truth, float)[mask], np.asarray(pred, float)[mask]
    if not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Metrics require nonempty finite matching valid positions.')
    error = b-a
    sse = np.sum(error**2)
    sst = np.sum((a-a.mean())**2)
    return {'MAE':float(np.mean(np.abs(error))), 'RMSE':float(np.sqrt(np.mean(error**2))),
            'R2':float(1-sse/sst) if sst > 0 else float('nan'), 'valid_positions':int(len(a))}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Choose a new output directory.')
    args.output_dir.mkdir(parents=True)
    root = ROOT/'data/new_case_evaluation'
    inputs = read_npz(root/'inputs_50.npz')
    quality = pd.read_csv(root/'quality_all_50.csv').sort_values('snapshot_index')
    if not np.array_equal(quality['snapshot_index'].to_numpy(), np.arange(50)):
        raise ValueError('Quality records do not cover the 50 input rows exactly once.')
    if not np.array_equal(quality['case_id'].to_numpy(), inputs['case_id']):
        raise ValueError('Case ordering does not match the deposited inputs.')
    if not np.array_equal(quality['step'].to_numpy(), inputs['step']):
        raise ValueError('Step ordering does not match the deposited inputs.')
    accepted = quality['quality_status'].eq('ACCEPTED_LARGE_REFERENCE').to_numpy()
    steps = inputs['step'].astype(int)
    if int(accepted.sum()) != 41 or int((accepted & (steps == 5)).sum()) != 6:
        raise ValueError('Qualified snapshot scope differs from the frozen evaluation scope.')
    disp = read_npz(root/'displacement_predictions_before_native.npz')
    disp_ref = read_npz(root/'displacement_references.npz')
    stress = read_npz(root/'supplementary_stress_predictions_50.npz')
    stress_ref = read_npz(root/'supplementary_stress_references_50.npz')
    pooled, snapshots, guard = [], [], []
    for target in ['ux_m', 'uz_m', 'sxx_MPa', 'syy_MPa', 'szz_MPa']:
        unit = 'm' if target in ['ux_m','uz_m'] else 'MPa'
        if unit == 'm':
            stem = target[:2]
            pred, truth = disp[stem+'_repaired_m'], disp_ref[stem+'_reference_m']
            valid = inputs['valid_mask'] & disp_ref['valid_mask']
        else:
            pred, truth = stress[target], stress_ref[target]
            valid = inputs['valid_mask'] & stress['valid_mask'] & stress_ref['valid_mask']
        for k in [0, 1, 2, 3, 4, 5]:
            chosen = accepted if k == 0 else accepted & (steps == k)
            pooled.append({'component':target,'unit':unit,'step':'all' if k == 0 else str(k),
                'snapshots':int(chosen.sum()),**metric(truth[chosen],pred[chosen],valid[chosen])})
        for i in np.flatnonzero(accepted):
            snapshots.append({'component':target,'unit':unit,'snapshot_index':int(i),
                'case_id':str(inputs['case_id'][i]),'step':int(steps[i]),
                **metric(truth[i],pred[i],valid[i])})
        if unit == 'm':
            chosen = accepted & (steps == 5)
            for name, values in [('before',disp[stem+'_archived_m']),('after',pred)]:
                guard.append({'component':target,'variant':name,'snapshots':int(chosen.sum()),
                    'physical_groups':int(len(set(inputs['physical_group'][chosen]))),
                    **metric(truth[chosen],values[chosen],valid[chosen])})
    pd.DataFrame(pooled).to_csv(args.output_dir/'field_metrics_pooled.csv',index=False)
    pd.DataFrame(snapshots).to_csv(args.output_dir/'field_metrics_per_snapshot.csv',index=False)
    pd.DataFrame(guard).to_csv(args.output_dir/'step5_guard_comparison.csv',index=False)
    gates = pd.read_csv(root/'acceptance_17.csv')
    failed = gates.loc[~gates['passed'].astype(str).str.lower().eq('true'),'gate'].tolist()
    summary = {'planned_snapshots':50,'qualified_snapshots':41,'qualified_step5_snapshots':6,
        'original_acceptance_conditions':17,'original_failed_conditions':failed,
        'original_overall_acceptance_passed':False,
        'note':'Archive replay checks are distinct from the original research acceptance criteria.'}
    (args.output_dir/'evaluation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
