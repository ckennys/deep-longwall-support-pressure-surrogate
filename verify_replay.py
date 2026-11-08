"""Compare portable model inference with independently archived predictions."""
import argparse, json, sys, importlib.metadata as md
from pathlib import Path
import numpy as np
import pandas as pd
from longwall_surrogate.inference import ROOT, read_npz, predict_fields, support_prediction

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new output path.')
    report = {'scope':'Frozen inference replay; no model retraining or native solver runs.',
              'environment':{'python':sys.version.split()[0], 'packages':{n:md.version(n) for n in
                ['numpy','scipy','pandas','scikit-learn','joblib','torch','catboost','threadpoolctl']}},
              'checks':[]}
    data = read_npz(ROOT/'data/new_case_evaluation/inputs_50.npz')
    valid = data['valid_mask']
    frozen = read_npz(ROOT/'data/new_case_evaluation/displacement_predictions_before_native.npz')
    frozen_stress = read_npz(ROOT/'data/new_case_evaluation/supplementary_stress_predictions_50.npz')
    predicted = predict_fields(data)
    for target in ['ux_m','uz_m','sxx_MPa','syy_MPa','szz_MPa']:
        expected = frozen[target[:2]+'_repaired_m'] if target in ['ux_m','uz_m'] else frozen_stress[target]
        got = predicted[target].reshape(50,-1)
        np.testing.assert_allclose(got[valid],expected[valid],rtol=5e-6,atol=2e-6)
        report['checks'].append({'component':target,'snapshots':50,'max_absolute_difference':float(np.max(np.abs(got[valid]-expected[valid]))),'passed':True})
        print(target,'replayed',flush=True)
    before = predict_fields(data,['ux_m','uz_m'],apply_guard=False)
    for target in ['ux_m','uz_m']:
        expected = frozen[target[:2]+'_archived_m']
        got = before[target].reshape(50,-1)
        np.testing.assert_allclose(got[valid],expected[valid],rtol=5e-6,atol=2e-6)
        unguarded = data['step'] < 5
        np.testing.assert_array_equal(predicted[target][unguarded],before[target][unguarded])
        report['checks'].append({'component':target,'diagnostic':'unguarded predictions and unchanged steps 1 to 4','max_absolute_difference':float(np.max(np.abs(got[valid]-expected[valid]))),'passed':True})
    all_frozen = pd.read_csv(ROOT/'results/support/cmcrnet_all_predictions.csv')
    for dataset,label,cache_name,tag in [('development','development_labels.csv','single_stage_stage7_oof_predictions.csv','Stage7_matched_OOF'),('archived','archived_labels.csv','single_stage_stage8_predictions.csv','Stage8_locked_ensemble')]:
        frame = pd.read_csv(ROOT/'data/support'/label)
        teacher = pd.read_csv(ROOT/'data/support'/cache_name).set_index('sample_id').loc[frame['sample_id']]
        anchors = teacher['two_stage_predicted_pressure_MPa'].to_numpy()
        folds = teacher['matched_outer_fold'].to_numpy() if dataset == 'development' else None
        for variant in ['independent_multitask','plus_censored_loss','plus_coupled_ranking','full_CMCRNet']:
            got = support_prediction(frame,anchors,folds,variant)
            expected = all_frozen[(all_frozen['dataset'] == tag) & (all_frozen['variant'] == variant)].set_index('sample_id').loc[frame['sample_id']]
            for new,old in [('predicted_latent_demand_MPa','predicted_latent_demand_MPa'),('predicted_no_pass_probability','predicted_boundary_probability')]:
                np.testing.assert_allclose(got[new],expected[old],rtol=2e-6,atol=1e-6)
            np.testing.assert_array_equal(got['predicted_no_pass'],expected['predicted_boundary'].to_numpy().astype(bool))
            report['checks'].append({'module':'support','dataset':dataset,'variant':variant,'rows':len(frame),'max_demand_difference_MPa':float(np.max(np.abs(got['predicted_latent_demand_MPa'].to_numpy()-expected['predicted_latent_demand_MPa'].to_numpy()))),'passed':True})
            print(dataset,variant,'replayed',flush=True)
    report['all_passed'] = all(row['passed'] for row in report['checks'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'all_passed':report['all_passed'],'checks':len(report['checks'])}))

if __name__ == '__main__':
    main()
