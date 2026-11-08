from __future__ import annotations

import hashlib
import warnings

import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEED = 20261022
THREADS = 4


def engineered_inputs(x):
    x = np.asarray(x, dtype=float)
    q, lateral, mining, p = x[:, 3], x[:, 1], x[:, 2], x[:, 8]
    span, width, advance, roof = x[:, 9], x[:, 10], x[:, 11], x[:, 12]
    t = x[:, 13]
    extra = np.column_stack((p / np.maximum(q, 1e-6), lateral * q,
                             span / np.maximum(mining, 1e-6), width / np.maximum(mining, 1e-6),
                             advance * t, advance * t / np.maximum(span, 1e-6),
                             q / np.maximum(x[:, 4], 1e-6), q / np.maximum(x[:, 5], 1e-6),
                             mining / np.maximum(roof, 1e-6), np.sin(np.deg2rad(x[:, 6])),
                             p / np.maximum(5.15 * x[:, 5], 1e-6), t ** 2,
                             t * mining, t * q, t * span))
    return np.column_stack((x, extra))


def amplitude(y, valid):
    return np.max(np.where(valid, np.abs(y), 0.0), axis=1).astype(float)


def scores(truth, predicted, valid, low=None):
    t, p = np.asarray(truth[valid], dtype=float), np.asarray(predicted[valid], dtype=float)
    assert t.shape == p.shape and np.isfinite(p).all()
    e = p - t
    sse, sst = float(e @ e), float(np.sum((t - t.mean()) ** 2))
    result = {'rmse': float(np.sqrt(sse / len(e))), 'mae': float(np.mean(np.abs(e))),
              'r2': float(1 - sse / sst) if sst > 0 else 0.0, 'valid_points': int(len(e))}
    if low is not None and np.any(low):
        le = (predicted[low] - truth[low])[valid[low]].astype(float)
        result['low_rmse'] = float(np.sqrt(np.mean(le ** 2)))
        result['low_mae'] = float(np.mean(np.abs(le)))
    return result


def fit_representation(y, valid, kind, cache, x):
    key = (kind, hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest())
    if key in cache:
        return cache[key]
    amp = np.maximum(amplitude(y, valid), 1e-8)
    scale = max(float(np.median(amp)) * 0.2, 1e-5)
    if kind == 'shape':
        transformed = y / amp[:, None]
    elif kind == 'asinh':
        transformed = np.arcsinh(y / scale)
    else:
        transformed = y
    transformed = np.asarray(transformed, dtype=np.float32)
    dimensions = min(96, len(y) - 1, y.shape[1])
    pca = PCA(n_components=dimensions, svd_solver='randomized', random_state=SEED)
    with threadpool_limits(limits=THREADS):
        coefficients = pca.fit_transform(transformed)
    package = {'kind': kind, 'pca': pca, 'coefficients': coefficients, 'amplitude': amp,
               'scale': scale, 'transform_min': float(transformed.min()), 'transform_max': float(transformed.max())}
    cache[key] = package
    return package


def fit_regressor(kind, x, y, configuration, sample_weight=None):
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    leaf = configuration.get('leaf', 2)
    feature_fraction = configuration.get('features', 0.9)
    scaler = StandardScaler().fit(x)
    xx = scaler.transform(x) if kind in ['knn', 'krr'] else x
    if kind in ['et', 'rf']:
        factory = ExtraTreesRegressor if kind == 'et' else RandomForestRegressor
        model = factory(n_estimators=configuration.get('trees', 320), min_samples_leaf=leaf,
                        max_features=feature_fraction, max_leaf_nodes=configuration.get('max_leaves', 512),
                        n_jobs=THREADS, random_state=SEED + configuration.get('seed_offset', 0))
        model.fit(xx, y.ravel() if y.shape[1] == 1 else y, sample_weight=sample_weight)
    elif kind == 'cat':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=configuration.get('iterations', 500), depth=configuration.get('depth', 6),
                                 learning_rate=0.05, l2_leaf_reg=5, random_strength=0.5,
                                 loss_function='RMSE' if y.shape[1] == 1 else 'MultiRMSE',
                                 random_seed=SEED, thread_count=THREADS, verbose=False, allow_writing_files=False)
        model.fit(xx, y.ravel() if y.shape[1] == 1 else y, sample_weight=sample_weight)
    elif kind == 'lgb':
        from lightgbm import LGBMRegressor
        model = []
        for j in range(y.shape[1]):
            estimator = LGBMRegressor(n_estimators=configuration.get('iterations', 400), num_leaves=23,
                                      learning_rate=0.04, min_child_samples=8, colsample_bytree=0.9,
                                      subsample=0.9, subsample_freq=1, reg_lambda=2.0, n_jobs=THREADS,
                                      random_state=SEED + j, verbosity=-1, deterministic=True, force_col_wise=True)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                estimator.fit(xx, y[:, j], sample_weight=sample_weight)
            model.append(estimator)
    elif kind == 'knn':
        model = KNeighborsRegressor(n_neighbors=min(configuration.get('neighbors', 5), len(xx)),
                                     weights='distance', p=configuration.get('p', 2), n_jobs=THREADS)
        model.fit(xx, y)
    elif kind == 'krr':
        model = KernelRidge(alpha=configuration.get('alpha', 0.01), kernel='rbf', gamma=configuration.get('gamma', 0.03))
        with threadpool_limits(limits=THREADS):
            model.fit(xx, y, sample_weight=sample_weight)
    else:
        raise ValueError(kind)
    return {'kind': kind, 'model': model, 'scaler': scaler}


def predict_regressor(package, x):
    xx = package['scaler'].transform(x) if package['kind'] in ['knn', 'krr'] else x
    with threadpool_limits(limits=THREADS):
        if package['kind'] == 'lgb':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = np.column_stack([model.predict(xx) for model in package['model']])
        else:
            result = package['model'].predict(xx)
    return np.asarray(result, dtype=float).reshape(len(x), -1)


def fit_field(configuration, x, y, valid, uz_amp, cache=None):
    cache = {} if cache is None else cache
    if configuration.get('wrapper') == 'step':
        step = np.rint(x[:, -1] * 5).astype(int)
        child_config = {key: value for key, value in configuration.items() if key != 'wrapper'}
        children = {str(s): fit_field(child_config, x[step == s], y[step == s], valid[step == s], uz_amp[step == s], cache)
                    for s in np.unique(step)}
        return {'wrapper': 'step', 'children': children, 'config': configuration}
    if configuration.get('wrapper') == 'regime':
        label = uz_amp >= 2.0
        child_config = {key: value for key, value in configuration.items() if key not in ['wrapper', 'gate']}
        classifier = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, max_features=0.9,
                                           class_weight='balanced', n_jobs=THREADS, random_state=SEED)
        classifier.fit(engineered_inputs(x), label)
        children = {str(int(value)): fit_field(child_config, x[label == value], y[label == value], valid[label == value],
                                              uz_amp[label == value], cache) for value in [False, True]}
        return {'wrapper': 'regime', 'classifier': classifier, 'children': children, 'config': configuration}
    inputs = engineered_inputs(x) if configuration.get('engineered', True) else x
    representation = fit_representation(y, valid, configuration['representation'], cache, x)
    if configuration.get('modes') == 'variance':
        retained = int(np.searchsorted(np.cumsum(representation['pca'].explained_variance_ratio_), 0.9995) + 1)
    else:
        retained = int(configuration.get('modes', 32))
    retained = min(retained, representation['coefficients'].shape[1])
    coefficients = representation['coefficients'][:, :retained].astype(float)
    if configuration.get('whiten', False):
        coefficient_scale = np.maximum(coefficients.std(axis=0), 1e-7)
    else:
        coefficient_scale = np.full(retained, max(float(np.sqrt(np.mean(coefficients ** 2))), 1e-7))
    weights = None
    if configuration.get('weighted'):
        amp = representation['amplitude']
        weights = 1.0 + 2.0 * np.clip(amp / max(float(np.quantile(amp, 0.8)), 1e-7), 0, 4)
    regressor = fit_regressor(configuration['learner'], inputs, coefficients / coefficient_scale,
                              configuration, sample_weight=weights)
    amp_regressor = None
    if configuration['representation'] == 'shape':
        amp_regressor = fit_regressor(configuration.get('amplitude_learner', 'et'), inputs,
                                      np.log(representation['amplitude']), configuration, sample_weight=None)
    return {'wrapper': None, 'config': configuration, 'representation': {key: value for key, value in representation.items()
                                                                       if key not in ['coefficients', 'amplitude']},
            'retained': retained, 'coefficient_scale': coefficient_scale, 'regressor': regressor,
            'amplitude_regressor': amp_regressor,
            'amplitude_log_limits': [float(np.log(representation['amplitude']).min()) - 0.5,
                                     float(np.log(representation['amplitude']).max()) + 0.5]}


def predict_field(package, x, valid):
    if package['wrapper'] == 'step':
        step = np.rint(x[:, -1] * 5).astype(int)
        result = np.zeros(valid.shape, dtype=np.float32)
        for s in np.unique(step):
            selected = step == s
            result[selected] = predict_field(package['children'][str(s)], x[selected], valid[selected])
        return result
    if package['wrapper'] == 'regime':
        probability = package['classifier'].predict_proba(engineered_inputs(x))[:, 1]
        low = predict_field(package['children']['0'], x, valid)
        high = predict_field(package['children']['1'], x, valid)
        gate = (probability >= 0.5).astype(float) if package['config'].get('gate') == 'hard' else expit((probability - 0.5) / 0.10)
        return ((1 - gate[:, None]) * low + gate[:, None] * high).astype(np.float32)
    configuration = package['config']
    inputs = engineered_inputs(x) if configuration.get('engineered', True) else x
    representation = package['representation']
    coefficients = predict_regressor(package['regressor'], inputs) * package['coefficient_scale']
    with threadpool_limits(limits=THREADS):
        transformed = coefficients @ representation['pca'].components_[:package['retained']] + representation['pca'].mean_
    if configuration['representation'] == 'shape':
        amp_log = predict_regressor(package['amplitude_regressor'], inputs)[:, 0]
        amp = np.exp(np.clip(amp_log, *package['amplitude_log_limits']))
        shape_amplitude = np.maximum(amplitude(transformed, valid), 1e-8)
        result = transformed / shape_amplitude[:, None] * amp[:, None]
    elif configuration['representation'] == 'asinh':
        result = np.sinh(np.clip(transformed, representation['transform_min'] - 0.5,
                                 representation['transform_max'] + 0.5)) * representation['scale']
    else:
        result = transformed
    result = np.asarray(result, dtype=np.float32)
    assert np.isfinite(result).all()
    return result


def candidates(target):
    displacement = target in ['ux_m', 'uz_m']
    result = []

    def add(name, representation='raw', learner='et', modes=32, **kwargs):
        result.append({'name': name, 'representation': representation, 'learner': learner,
                       'modes': modes, **kwargs})

    add('ET_reference_variance', modes='variance', engineered=False, weighted=True)
    for k, leaf, fraction in [(4, 1, 1.0), (12, 1, 0.8), (32, 2, 0.9), (64, 1, 0.8), (64, 4, 1.0), (96, 2, 0.7)]:
        add(f'ET_raw_k{k}_leaf{leaf}', modes=k, leaf=leaf, features=fraction)
    add('ET_raw_whitened', modes=48, leaf=2, whiten=True)
    add('ET_raw_weighted', modes=48, leaf=1, weighted=True)
    add('RF_raw', learner='rf', modes=32, leaf=1)
    add('ET_raw_step_experts', modes=32, leaf=1, wrapper='step')
    for k in [16, 48]:
        add(f'ET_asinh_k{k}', representation='asinh', modes=k, leaf=1)
    for neighbors in [3, 7, 15]:
        add(f'KNN_raw_{neighbors}', learner='knn', modes=64, neighbors=neighbors)
    for gamma in [0.005, 0.02, 0.08]:
        add(f'KRR_raw_g{gamma}', learner='krr', modes=64, gamma=gamma, alpha=0.01)
    add('CatBoost_raw_depth5', learner='cat', modes=24, depth=5, iterations=500)
    add('CatBoost_raw_depth7', learner='cat', modes=24, depth=7, iterations=600)
    add('LightGBM_raw', learner='lgb', modes=16, iterations=400)
    for k in [24, 64]:
        add(f'ET_shape_k{k}_ETamp', representation='shape', modes=k, leaf=1, amplitude_learner='et')
        add(f'ET_shape_k{k}_Catamp', representation='shape', modes=k, leaf=1, amplitude_learner='cat', iterations=600, depth=5)
    add('ET_shape_LGBamp', representation='shape', modes=48, leaf=1, amplitude_learner='lgb')
    add('CatBoost_shape', representation='shape', learner='cat', modes=24, amplitude_learner='cat', depth=5, iterations=600)
    add('KRR_shape', representation='shape', learner='krr', modes=48, amplitude_learner='cat', gamma=0.02, alpha=0.01, iterations=500)
    add('ET_shape_step_experts', representation='shape', modes=32, leaf=1, wrapper='step', amplitude_learner='et')
    if displacement:
        add('ET_raw_regime_experts', modes=48, leaf=1, wrapper='regime', gate='hard')
        add('ET_shape_regime_hard', representation='shape', modes=48, leaf=1,
             wrapper='regime', gate='hard', amplitude_learner='cat', iterations=600, depth=5)
        add('ET_shape_regime_soft', representation='shape', modes=48, leaf=1,
             wrapper='regime', gate='soft', amplitude_learner='cat', iterations=600, depth=5)
        add('ET_asinh_step_experts', representation='asinh', modes=32, leaf=1, wrapper='step')
    return result
