from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import joblib
import numpy as np
from scipy.optimize import minimize

from stock_models import SEED, candidates, fit_field, predict_field, scores

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent
DATA = OUT / 'data'


def write_csv(path, rows):
    if not rows:
        return
    names = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def selection_score(value, reference, displacement):
    score = (0.70 if displacement else 0.85) * np.log(max(value['rmse'], 1e-12) / max(reference['rmse'], 1e-12))
    score += (0.20 if displacement else 0.15) * np.log(max(value['mae'], 1e-12) / max(reference['mae'], 1e-12))
    if displacement:
        score += 0.10 * np.log(max(value['low_rmse'], 1e-12) / max(reference['low_rmse'], 1e-12))
    return float(score)


def convex_weights(predictions, truth, valid, sample_weights=None):
    errors = np.column_stack([(pred - truth)[valid] for pred in predictions]).astype(float)
    if sample_weights is not None:
        weight = np.broadcast_to(sample_weights[:, None], valid.shape)[valid]
        errors *= np.sqrt(weight)[:, None]
    gram = errors.T @ errors / len(errors)
    scale = max(float(np.max(np.diag(gram))), 1e-15)
    gram /= scale
    count = len(predictions)
    uniform = np.full(count, 1.0 / count)
    penalty = 0.002
    objective = lambda w: float(w @ gram @ w + penalty * np.sum((w - uniform) ** 2))
    jacobian = lambda w: 2 * gram @ w + 2 * penalty * (w - uniform)
    solved = minimize(objective, uniform, jac=jacobian, bounds=[(0.0, 1.0)] * count,
                      constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1.0,
                                   'jac': lambda w: np.ones(count)}, method='SLSQP',
                      options={'ftol': 1e-10, 'maxiter': 500})
    assert solved.success, solved.message
    weights = np.maximum(solved.x, 0)
    weights[weights < 0.025] = 0.0
    return weights / weights.sum()


def combine(predictions, rule, steps):
    result = np.zeros_like(predictions[0], dtype=np.float32)
    if rule['mode'] == 'step':
        for step in np.unique(steps):
            selected = steps == step
            weights = rule['weights'][str(int(step))]
            result[selected] = sum(float(weight) * pred[selected] for weight, pred in zip(weights, predictions))
    else:
        result = sum(float(weight) * pred for weight, pred in zip(rule['weights'], predictions)).astype(np.float32)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True)
    args = parser.parse_args()
    target = args.target
    target_out = OUT / 'runs' / target
    target_out.mkdir(parents=True, exist_ok=True)
    metadata = np.load(DATA / 'development_inputs.npz', allow_pickle=False)
    x = metadata['features']
    role = metadata['role']
    valid = np.load(DATA / 'development_valid.npy', mmap_mode='r')
    y = np.load(DATA / f'development_{target}.npy', mmap_mode='r')
    amp_uz = metadata['amplitude_uz']
    fit, screen, confirm = [np.flatnonzero(role == name) for name in ['fit', 'screen', 'confirm']]
    train_x, train_y, train_valid, train_amp = x[fit], y[fit], valid[fit], amp_uz[fit]
    screen_x, screen_y, screen_valid = x[screen], y[screen], valid[screen]
    low = amp_uz[screen] < 2.0
    configuration_list = candidates(target)
    contract = {'target': target, 'seed': SEED, 'candidate_count': len(configuration_list),
                'candidates': configuration_list,
                'selection_score': '0.70 log(RMSE/reference) + 0.20 log(MAE/reference) + 0.10 log(low-RMSE/reference) for displacement; 0.85 log(RMSE/reference) + 0.15 log(MAE/reference) for stress',
                'fit_groups': len(np.unique(metadata['physical_group'][fit])),
                'screen_groups': len(np.unique(metadata['physical_group'][screen])),
                'confirm_groups': len(np.unique(metadata['physical_group'][confirm])),
                'external_truth_read': False, 'flac3d_runs': 0}
    (target_out / 'search-contract.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    rows, failed, predictions = [], [], {}
    cache = {}
    reference = None
    for number, configuration in enumerate(configuration_list, 1):
        started = time.perf_counter()
        try:
            model = fit_field(configuration, train_x, train_y, train_valid, train_amp, cache)
            predicted = predict_field(model, screen_x, screen_valid)
            value = scores(screen_y, predicted, screen_valid, low=low)
            reference = value if reference is None else reference
            value['selection_score'] = selection_score(value, reference, target in ['ux_m', 'uz_m'])
            row = {'target': target, 'candidate': configuration['name'], 'learner': configuration['learner'],
                   'representation': configuration['representation'], 'wrapper': configuration.get('wrapper', ''),
                   **value, 'seconds': time.perf_counter() - started}
            rows.append(row)
            predictions[configuration['name']] = predicted
            write_csv(target_out / 'screen_candidates.csv', rows)
            print(json.dumps({'target': target, 'progress': f'{number}/{len(configuration_list)}',
                              'candidate': configuration['name'], 'rmse': value['rmse'],
                              'mae': value['mae'], 'score': value['selection_score'],
                              'seconds': row['seconds']}, ensure_ascii=False), flush=True)
            del model
        except Exception as exc:
            failed.append({'candidate': configuration['name'], 'error': str(exc), 'traceback': traceback.format_exc()})
            (target_out / 'failed_candidates.json').write_text(json.dumps(failed, indent=2), encoding='utf-8')
            print(json.dumps({'target': target, 'candidate_failed': configuration['name'], 'error': str(exc)}), flush=True)
        gc.collect()
    assert reference is not None and len(rows) >= 10
    ranking = sorted(rows, key=lambda row: row['selection_score'])
    shortlist = [row['candidate'] for row in ranking[:4]]
    low_best = min(rows, key=lambda row: row['low_rmse'])['candidate']
    if low_best not in shortlist:
        shortlist.append(low_best)
    if 'ET_reference_variance' not in shortlist:
        shortlist.append('ET_reference_variance')
    selected_predictions = [predictions[name] for name in shortlist]
    steps = metadata['step_index'][screen]
    rules = []
    for i, name in enumerate(shortlist):
        weights = np.eye(len(shortlist))[i].tolist()
        rules.append({'name': name, 'mode': 'global', 'weights': weights})
    weights = convex_weights(selected_predictions, screen_y, screen_valid)
    rules.append({'name': 'Convex_global', 'mode': 'global', 'weights': weights.tolist()})
    step_weights = {str(int(step)): convex_weights([p[steps == step] for p in selected_predictions],
                    screen_y[steps == step], screen_valid[steps == step]).tolist() for step in np.unique(steps)}
    rules.append({'name': 'Convex_by_step', 'mode': 'step', 'weights': step_weights})
    amp = np.maximum(amp_uz[screen], 0.2)
    balance = 1 / np.maximum(amp ** 2, 0.04)
    balanced = convex_weights(selected_predictions, screen_y, screen_valid, sample_weights=balance)
    rules.append({'name': 'Convex_response_balanced', 'mode': 'global', 'weights': balanced.tolist()})
    rule_rows = []
    for rule in rules:
        prediction = combine(selected_predictions, rule, steps)
        value = scores(screen_y, prediction, screen_valid, low=low)
        value['selection_score'] = selection_score(value, reference, target in ['ux_m', 'uz_m'])
        rule_rows.append({'rule': rule['name'], **value})
    write_csv(target_out / 'screen_ensembles.csv', rule_rows)
    best_name = min(rule_rows, key=lambda row: row['selection_score'])['rule']
    selected_rule = next(rule for rule in rules if rule['name'] == best_name)
    used = np.zeros(len(shortlist), dtype=bool)
    if selected_rule['mode'] == 'step':
        used = np.max(np.asarray(list(selected_rule['weights'].values())), axis=0) > 0
    else:
        used = np.asarray(selected_rule['weights']) > 0
    names = [name for name, keep in zip(shortlist, used) if keep]
    if selected_rule['mode'] == 'step':
        selected_rule['weights'] = {s: np.asarray(w)[used].tolist() for s, w in selected_rule['weights'].items()}
    else:
        selected_rule['weights'] = np.asarray(selected_rule['weights'])[used].tolist()
    selected_configs = [next(conf for conf in configuration_list if conf['name'] == name) for name in names]
    selection = {'target': target, 'selected_rule': selected_rule, 'selected_configs': selected_configs,
                 'screen_result': next(row for row in rule_rows if row['rule'] == best_name),
                 'baseline_screen_result': reference, 'selected_before_internal_confirmation': True,
                 'historical_holdout_truth_used': False, 'candidate_failures': len(failed)}
    (target_out / 'selection-before-confirmation.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    # Release search predictions before refitting. The confirmation partition was not scored above.
    del predictions, selected_predictions, cache
    gc.collect()
    development_fit = np.concatenate((fit, screen))
    confirmation_predictions, confirmation_models = [], []
    cache = {}
    for configuration in selected_configs:
        print(json.dumps({'target': target, 'confirmation_refit': configuration['name']}), flush=True)
        model = fit_field(configuration, x[development_fit], y[development_fit], valid[development_fit],
                          amp_uz[development_fit], cache)
        confirmation_predictions.append(predict_field(model, x[confirm], valid[confirm]))
        confirmation_models.append(model)
    prediction = combine(confirmation_predictions, selected_rule, metadata['step_index'][confirm])
    confirmation_metrics = scores(y[confirm], prediction, valid[confirm], low=amp_uz[confirm] < 2.0)
    baseline_config = configuration_list[0]
    baseline = fit_field(baseline_config, x[development_fit], y[development_fit], valid[development_fit],
                          amp_uz[development_fit], cache)
    baseline_prediction = predict_field(baseline, x[confirm], valid[confirm])
    baseline_metrics = scores(y[confirm], baseline_prediction, valid[confirm], low=amp_uz[confirm] < 2.0)
    np.savez_compressed(target_out / 'internal_confirmation_predictions.npz',
                        prediction=prediction, baseline_prediction=baseline_prediction, indices=confirm)
    confirmed = {'target': target, 'candidate': confirmation_metrics, 'same_data_reference': baseline_metrics,
                 'rmse_reduction_percent': float(100 * (1 - confirmation_metrics['rmse'] / baseline_metrics['rmse'])),
                 'evaluation_groups': len(np.unique(metadata['physical_group'][confirm])),
                 'evaluation_snapshots': len(confirm), 'one_shot_confirmation': True,
                 'selection_changed_after_confirmation': False}
    (target_out / 'internal-confirmation.json').write_text(json.dumps(confirmed, indent=2), encoding='utf-8')
    del confirmation_models, confirmation_predictions, baseline, cache
    gc.collect()
    print(json.dumps({'target': target, 'confirmation': confirmed}), flush=True)
    # Freeze full-data refit identity before any historical evaluation truth is loaded.
    selection['internal_confirmation'] = confirmed
    selection['final_training_rows'] = len(x)
    selection['final_training_groups'] = len(np.unique(metadata['physical_group']))
    (target_out / 'frozen-selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    models, cache = [], {}
    for configuration in selected_configs:
        print(json.dumps({'target': target, 'final_refit': configuration['name']}), flush=True)
        models.append(fit_field(configuration, x, y, valid, amp_uz, cache))
    package = {'target': target, 'models': models, 'rule': selected_rule, 'configs': selected_configs,
               'feature_names': metadata['feature_names'].tolist(), 'training_groups': np.unique(metadata['physical_group']),
               'selection': selection, 'flac3d_runs': 0}
    joblib.dump(package, target_out / 'optimized_field_model.joblib', compress=3)
    selection['model_sha256'] = hashlib.sha256((target_out / 'optimized_field_model.joblib').read_bytes()).hexdigest()
    (target_out / 'completed.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    print(json.dumps({'target': target, 'completed': True, 'model_file': str(target_out / 'optimized_field_model.joblib')}), flush=True)


if __name__ == '__main__':
    main()
