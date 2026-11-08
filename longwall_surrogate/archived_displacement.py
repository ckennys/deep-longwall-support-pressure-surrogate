from __future__ import annotations

import warnings
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEED = 20261022
THREADS = 5


def engineered(x):
    x = np.asarray(x, dtype=float)
    q, lam, m, p = x[:, 3], x[:, 1], x[:, 2], x[:, 8]
    span, width, advance, roof, t = x[:, 9], x[:, 10], x[:, 11], x[:, 12], x[:, 13]
    e, c, tensile = x[:, 4], x[:, 5], x[:, 7]
    phi = np.deg2rad(x[:, 6])
    extra = np.column_stack([
        p / q, lam * q, span / m, width / m, advance * t, advance * t / span,
        q / e, q / c, m / roof, np.sin(phi), p / (5.15 * c), t ** 2,
        t * m, t * q, t * span, 5 * advance * t / span,
        span / (17.543 * roof), (span / (17.543 * roof)) ** 2,
        q * span ** 2 / (e * (17.543 * roof) ** 2),
        q * span ** 4 / (e * (17.543 * roof) ** 3),
        (q - p) / (5.15 * c), (q * lam - p) / (5.15 * c),
        (q - p) / (1.896 * tensile),
        q * np.sin(phi) / (5.15 * c * np.cos(phi)),
        (1 + np.sin(phi)) / (1 - np.sin(phi)),
        np.log(e), np.log(c), np.log(tensile),
        t ** 3, t ** 4, t * p / q, t * q / e, t * q / c,
        t * span / (17.543 * roof), t * m / roof,
        span / advance, width / advance, m / width, p / m,
    ])
    out = np.column_stack([x, extra])
    assert np.isfinite(out).all()
    return out


def amplitude(y, valid, rms=False):
    if rms:
        return np.sqrt(np.sum(np.where(valid, y * y, 0.0), axis=1) / valid.sum(axis=1))
    return np.max(np.where(valid, np.abs(y), 0.0), axis=1)


def metrics(y, p, valid):
    a, b = np.asarray(y[valid], dtype=float), np.asarray(p[valid], dtype=float)
    assert len(a) and np.isfinite(a).all() and np.isfinite(b).all()
    error = b - a
    sse, sst = float(error @ error), float(np.sum((a - a.mean()) ** 2))
    return {'rmse': float(np.sqrt(sse / len(a))), 'mae': float(np.mean(np.abs(error))),
            'r2': float(1 - sse / sst), 'valid_points': len(a)}


def configurations():
    return [
        dict(name='Reference_ET64', kind='raw', learner='et', modes=64, trees=480, leaves=512, features=.9, leaf=2),
        dict(name='Raw_ET128', kind='raw', learner='et', modes=128, trees=720, leaves=1536, features=1., leaf=1),
        dict(name='Raw_ET96_random', kind='raw', learner='et', modes=96, trees=720, leaves=1024, features=.65, leaf=1),
        dict(name='Raw_Cat48_d6', kind='raw', learner='cat', modes=48, iterations=1500, depth=6),
        dict(name='Raw_Cat64_d8', kind='raw', learner='cat', modes=64, iterations=1200, depth=8),
        dict(name='Raw_LGB24', kind='raw', learner='lgb', modes=24, iterations=1100),
        dict(name='Sqrt_Cat48', kind='power', power=.5, learner='cat', modes=48, iterations=1400, depth=6),
        dict(name='FourthRoot_ET96', kind='power', power=.25, learner='et', modes=96, trees=600, leaves=1024, features=.9, leaf=1),
        dict(name='RMS_ET64_rawamp', kind='rms', learner='et', modes=64, trees=600, leaves=1024, features=.9, leaf=1, amp='raw'),
        dict(name='RMS_ET48_sqrtamp', kind='rms', learner='et', modes=48, trees=600, leaves=1024, features=.9, leaf=1, amp='sqrt'),
        dict(name='RMS_ET48_logamp', kind='rms', learner='et', modes=48, trees=600, leaves=1024, features=.9, leaf=1, amp='log'),
        dict(name='Raw_Neural64', kind='raw', learner='nn', modes=64, epochs=320, members=2),
        dict(name='Sqrt_Neural64', kind='power', power=.5, learner='nn', modes=64, epochs=360, members=2),
        dict(name='Raw_LocalGaussian48', kind='raw', learner='rbf', modes=48),
    ]


def fit_nn(x, y, config):
    import torch
    from torch import nn
    torch.set_num_threads(THREADS)
    scaler = StandardScaler().fit(x)
    xx = torch.from_numpy(np.clip(scaler.transform(x), -8, 8).astype(np.float32))
    scale = np.maximum(np.std(y, axis=0), max(float(np.std(y)) * .015, 1e-7))
    yy = torch.from_numpy((y / scale).astype(np.float32))
    field_weights = torch.from_numpy((scale ** 2 / np.mean(scale ** 2)).astype(np.float32))
    states = []
    for member in range(config.get('members', 2)):
        torch.manual_seed(SEED + member * 31)
        net = nn.Sequential(nn.Linear(x.shape[1], 192), nn.SiLU(), nn.Linear(192, 256), nn.SiLU(),
                            nn.Dropout(.04), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, y.shape[1]))
        optimizer = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.00015)
        epochs = config.get('epochs', 320)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=.00008)
        net.train()
        for epoch in range(epochs):
            order = torch.randperm(len(xx))
            for idx in order.split(384):
                pred = net(xx[idx])
                error = pred - yy[idx]
                loss = .8 * (error.square() * field_weights).mean() + .2 * torch.nn.functional.smooth_l1_loss(pred, yy[idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 8.)
                optimizer.step()
            scheduler.step()
        states.append({key: value.detach().cpu().numpy() for key, value in net.state_dict().items()})
    return {'kind': 'nn', 'states': states, 'scaler': scaler, 'scale': scale,
            'inputs': x.shape[1], 'outputs': y.shape[1]}


def predict_nn(package, x):
    import torch
    from torch import nn
    torch.set_num_threads(THREADS)
    xx = torch.from_numpy(np.clip(package['scaler'].transform(x), -8, 8).astype(np.float32))
    values = []
    for state in package['states']:
        net = nn.Sequential(nn.Linear(package['inputs'], 192), nn.SiLU(), nn.Linear(192, 256), nn.SiLU(),
                            nn.Dropout(.04), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, package['outputs']))
        net.load_state_dict({key: torch.from_numpy(value) for key, value in state.items()})
        net.eval()
        with torch.no_grad():
            values.append(net(xx).numpy() * package['scale'])
    return np.mean(values, axis=0)


def fit_regressor(x, y, config, sample_weight=None):
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    kind = config['learner']
    if kind == 'nn':
        return fit_nn(x, y, config)
    if kind == 'et':
        model = ExtraTreesRegressor(n_estimators=config.get('trees', 600), max_leaf_nodes=config.get('leaves', 1024),
                                    min_samples_leaf=config.get('leaf', 1), max_features=config.get('features', .9),
                                    n_jobs=THREADS, random_state=SEED)
        model.fit(x, y.ravel() if y.shape[1] == 1 else y, sample_weight=sample_weight)
    elif kind == 'cat':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(iterations=config.get('iterations', 1200), depth=config.get('depth', 6),
                                  learning_rate=.035, l2_leaf_reg=4, random_strength=.3,
                                  loss_function='MultiRMSE' if y.shape[1] > 1 else 'RMSE',
                                  random_seed=SEED, thread_count=THREADS, verbose=False, allow_writing_files=False)
        model.fit(x, y.ravel() if y.shape[1] == 1 else y, sample_weight=sample_weight)
    elif kind == 'lgb':
        from lightgbm import LGBMRegressor
        model = []
        for column in range(y.shape[1]):
            estimator = LGBMRegressor(n_estimators=config.get('iterations', 1100), num_leaves=23,
                                      learning_rate=.025, min_child_samples=12, colsample_bytree=.85,
                                      reg_lambda=5, max_depth=-1, n_jobs=THREADS, verbosity=-1,
                                      random_state=SEED + column, deterministic=True, force_col_wise=True)
            estimator.fit(x, y[:, column], sample_weight=sample_weight)
            model.append(estimator)
    elif kind == 'rbf':
        scaler = StandardScaler().fit(x[:, :14])
        active = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        xx = scaler.transform(x[:, :14])[:, active]
        model = RBFInterpolator(xx, y, neighbors=min(96, len(xx)), smoothing=.08, kernel='gaussian', epsilon=.5, degree=0)
        return {'kind': 'rbf', 'model': model, 'scaler': scaler, 'active': active}
    else:
        raise ValueError(kind)
    return {'kind': kind, 'model': model}


def predict_regressor(package, x):
    kind = package['kind']
    with threadpool_limits(limits=THREADS):
        if kind == 'nn':
            result = predict_nn(package, x)
        elif kind == 'rbf':
            result = package['model'](package['scaler'].transform(x[:, :14])[:, package['active']])
        elif kind == 'lgb':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = np.column_stack([m.predict(x) for m in package['model']])
        else:
            result = package['model'].predict(x)
    return np.asarray(result).reshape(len(x), -1)


def fit_field(config, x, y, valid, cache=None):
    cache = {} if cache is None else cache
    kind = config['kind']
    key = (kind, config.get('power'))
    if key not in cache:
        if kind == 'power':
            z = np.sign(y) * np.abs(y) ** config['power']
        elif kind == 'rms':
            amp = np.maximum(amplitude(y, valid, rms=True), 1e-8)
            z = y / amp[:, None]
        else:
            z = y
        z = np.where(valid, z, 0).astype(np.float32)
        pca = PCA(n_components=min(128, len(x) - 1), svd_solver='randomized', random_state=SEED)
        with threadpool_limits(limits=THREADS):
            coefficients = pca.fit_transform(z)
        cache[key] = (pca, coefficients)
    pca, coefs = cache[key]
    count = min(config['modes'], coefs.shape[1])
    scale = max(float(np.sqrt(np.mean(coefs[:, :count] ** 2))), 1e-7)
    regressor = fit_regressor(engineered(x), coefs[:, :count] / scale, config)
    result = {'config': config, 'pca': pca, 'modes': count, 'scale': scale, 'regressor': regressor}
    if kind == 'rms':
        amp = np.maximum(amplitude(y, valid, rms=True), 1e-8)
        response = amp if config['amp'] == 'raw' else np.sqrt(amp) if config['amp'] == 'sqrt' else np.log(amp)
        result['amplitude'] = fit_regressor(engineered(x), response,
                                           dict(learner='cat', iterations=1500, depth=6))
        result['max_amp'] = float(amp.max() * 4)
    return result


def predict_field(model, x, valid):
    coefs = predict_regressor(model['regressor'], engineered(x)) * model['scale']
    with threadpool_limits(limits=THREADS):
        z = coefs @ model['pca'].components_[:model['modes']] + model['pca'].mean_
    config = model['config']
    if config['kind'] == 'power':
        out = np.sign(z) * np.abs(z) ** (1 / config['power'])
    elif config['kind'] == 'rms':
        raw = predict_regressor(model['amplitude'], engineered(x))[:, 0]
        amp = raw if config['amp'] == 'raw' else np.maximum(raw, 0) ** 2 if config['amp'] == 'sqrt' else np.exp(np.clip(raw, -20, 20))
        amp = np.clip(amp, 1e-8, model['max_amp'])
        out = z / np.maximum(amplitude(z, valid, rms=True), 1e-8)[:, None] * amp[:, None]
    else:
        out = z
    out = np.asarray(out, dtype=np.float32)
    assert np.isfinite(out).all()
    return out


def meta_features(x, fields, valid):
    extra = []
    for f in fields:
        extra += [np.log1p(amplitude(f, valid)), np.log1p(amplitude(f, valid, rms=True)),
                  np.sum(np.where(valid, f, 0), axis=1) / valid.sum(axis=1)]
    return np.column_stack([engineered(x), *extra])


def simplex_weights(truth, fields, valid, sample_weights=None):
    errors = np.array([(np.asarray(p, float) - truth) * valid for p in fields])
    if sample_weights is not None:
        errors *= np.sqrt(sample_weights)[None, :, None]
    gram = np.einsum('kij,lij->kl', errors, errors, optimize=True)
    gram /= max(float(np.trace(gram)), 1e-12)
    count = len(fields)
    solution = minimize(lambda w: float(w @ gram @ w), np.ones(count) / count,
                        jac=lambda w: 2 * gram @ w, method='SLSQP',
                        bounds=[(0, 1)] * count, constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
                        options={'ftol': 1e-12, 'maxiter': 500})
    assert solution.success, solution.message
    return np.maximum(solution.x, 0) / np.maximum(solution.x, 0).sum()


def blend(fields, weights):
    if np.ndim(weights) == 1:
        return sum(w * f for w, f in zip(weights, fields)).astype(np.float32)
    return sum(weights[:, j, None] * f for j, f in enumerate(fields)).astype(np.float32)


def fit_corrections(x, truth, fields, valid):
    z = meta_features(x, fields, valid)
    reference = fields[0]
    residual = truth - reference
    residual_model = fit_field(dict(name='Crossfit_residual', kind='raw', learner='cat', modes=32,
                                   iterations=1000, depth=5), x, residual, valid, {})
    dot = np.sum(np.where(valid, reference * truth, 0), axis=1)
    energy = np.sum(np.where(valid, reference ** 2, 0), axis=1)
    target_scale = np.clip(dot / np.maximum(energy, 1e-12), .05, 5.)
    scale_model = fit_regressor(z, np.log(target_scale), dict(learner='cat', iterations=900, depth=5))
    error = np.stack([np.sum(np.where(valid, (f - truth) ** 2, 0), axis=1) for f in fields], axis=1)
    soft = 1 / np.maximum(error, np.median(error, axis=1, keepdims=True) * .02 + 1e-10)
    soft /= soft.sum(axis=1, keepdims=True)
    gate = fit_regressor(z, soft, dict(learner='cat', iterations=1000, depth=5))
    return {'residual': residual_model, 'scale': scale_model, 'gate': gate,
            'convex': simplex_weights(truth, fields, valid)}


def correction_predictions(package, x, fields, valid):
    z = meta_features(x, fields, valid)
    residual = predict_field(package['residual'], x, valid)
    log_scale = np.clip(predict_regressor(package['scale'], z)[:, 0], np.log(.1), np.log(5.))
    gate = np.clip(predict_regressor(package['gate'], z), 0, 1)
    gate = np.divide(gate, np.maximum(gate.sum(axis=1, keepdims=True), 1e-12))
    result = {f'Expert_{i}': field for i, field in enumerate(fields)}
    result['Convex_OOF'] = blend(fields, package['convex'])
    result['ErrorAware_gate'] = blend(fields, gate)
    for strength in [.25, .5, .75, 1., 1.5]:
        result[f'Residual_{strength:g}'] = (fields[0] + strength * residual).astype(np.float32)
    for strength in [.25, .5, 1., 1.5]:
        result[f'Amplitude_{strength:g}'] = (fields[0] * np.exp(log_scale * strength)[:, None]).astype(np.float32)
    return result


def predict_package(package, x, valid, include_experts=False):
    fields = [predict_field(head, x, valid) for head in package['heads']]
    all_fields = correction_predictions(package['corrections'], x, fields, valid)
    values = {}
    for profile, recipe in package['recipes'].items():
        names, weights = recipe['names'], recipe['weights']
        values[profile] = blend([all_fields[name] for name in names], np.asarray(weights))
    if include_experts:
        for head, field in zip(package['heads'], fields):
            values['expert_' + head['config']['name']] = field
    return values
