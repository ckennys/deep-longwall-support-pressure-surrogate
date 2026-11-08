"""Portable inference. All data and model paths resolve inside the release root."""
from pathlib import Path
import gc
import numpy as np

FIELD_FEATURES = ['H_m', 'lambda', 'M_m', 'q_top_MPa', 'alpha_E_roof',
    'alpha_c_roof', 'phi_roof_deg', 'alpha_t_roof', 'support_pressure_MPa',
    'panel_span_m', 'roadway_width_m', 'slice_width_m',
    'direct_roof_thickness_factor', 'step_fraction']
SUPPORT_FEATURES = FIELD_FEATURES[:8]
PRESSURE_LEVELS = np.array([.5, .8, 1.1, 1.4, 1.7, 2., 2.3, 2.6])
from .paths import ROOT


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def field_inputs(data):
    x = np.asarray(data['features'], dtype=float)
    if x.ndim != 2 or x.shape[1] != 14 or not len(x) or not np.isfinite(x).all():
        raise ValueError('A nonempty finite N by 14 feature matrix is required.')
    if np.asarray(data['feature_names']).astype(str).tolist() != FIELD_FEATURES:
        raise ValueError('Feature names must match the documented 14-column order.')
    mask = np.asarray(data['valid_mask'], dtype=bool).reshape(len(x), -1)
    if mask.shape != (len(x), 5504) or not mask.any(axis=1).all():
        raise ValueError('Each snapshot requires a nonempty 64 by 86 valid mask.')
    steps = np.rint(x[:, 13]*5).astype(int)
    if not np.isin(steps, [1, 2, 3, 4, 5]).all() or not np.allclose(x[:, 13], steps/5., atol=1e-7, rtol=0):
        raise ValueError('step_fraction must be 0.2, 0.4, 0.6, 0.8 or 1.0.')
    if np.any(x[:, 8] <= 0):
        raise ValueError('Support pressure must be positive and expressed in MPa.')
    if np.any(x[:, [2, 3, 4, 5, 7, 9, 10, 11, 12]] <= 0):
        raise ValueError('Thickness, load, material scale factors and geometry must be positive.')
    return x, mask, steps


def predict_displacement(bundle, x, mask, apply_guard=True):
    from . import archived_displacement as old
    from threadpoolctl import threadpool_limits
    old.THREADS = 2
    head = bundle['archived_head']
    result = old.predict_field(head, x, mask)
    counts = np.zeros(len(x), dtype=int)
    selected = np.isclose(x[:, 13], 1., rtol=0, atol=1e-7)
    if apply_guard and selected.any():
        guard = bundle['step5_guard']
        c = old.predict_regressor(head['regressor'], old.engineered(x[selected]))*head['scale']
        active = np.asarray(guard['active_modes'], dtype=int)
        counts[selected] = ((c[:, active] < guard['lower'][active]) | (c[:, active] > guard['upper'][active])).sum(axis=1)
        c[:, active] = np.clip(c[:, active], guard['lower'][active], guard['upper'][active])
        with threadpool_limits(limits=2):
            z = c@head['pca'].components_[:64] + head['pca'].mean_
        result[selected] = (z*np.abs(z)).astype(np.float32)
    return result, counts


def predict_fields(data, targets=None, root=ROOT, apply_guard=True):
    import joblib
    from . import archived_stress as stress
    from .combine import combine
    stress.THREADS = 2
    x, mask, steps = field_inputs(data)
    targets = list(targets or ['ux_m', 'uz_m', 'sxx_MPa', 'syy_MPa', 'szz_MPa'])
    if not targets or not set(targets) <= {'ux_m', 'uz_m', 'sxx_MPa', 'syy_MPa', 'szz_MPa'}:
        raise ValueError('Choose one or more of the five documented field components.')
    result = {'features': x, 'feature_names': np.asarray(FIELD_FEATURES),
              'valid_mask': mask.reshape(-1, 64, 86), 'step_index': steps}
    for key in ['case_id', 'physical_group']:
        if key in data:
            result[key] = data[key]
    for target in targets:
        if target in ['ux_m', 'uz_m']:
            model = joblib.load(Path(root)/'models/displacement'/f'{target[:2]}_guard64.joblib')
            values, counts = predict_displacement(model, x, mask, apply_guard)
            result[target[:2]+'_guarded_modes_count'] = counts
        else:
            model = joblib.load(Path(root)/'models/normal_stress'/f'{target}.joblib')
            members = [stress.predict_field(m, x, mask) for m in model['models']]
            values = combine(members, model['rule'], steps)
            del members
        if not np.isfinite(values[mask]).all():
            raise FloatingPointError(f'Nonfinite valid prediction for {target}')
        values = np.asarray(values, dtype=np.float32).copy()
        values[~mask] = np.nan
        result[target] = values.reshape(-1, 64, 86)
        del model, values
        gc.collect()
    return result


def nearest_pressure(values):
    """Nearest candidate, with lower-pressure tie breaking within roundoff."""
    values = np.asarray(values, dtype=float)
    distance = np.abs(values[:, None]-PRESSURE_LEVELS[None, :])
    tied = np.isclose(distance, distance.min(axis=1, keepdims=True), rtol=0, atol=1e-12)
    return PRESSURE_LEVELS[np.argmax(tied, axis=1)]


def support_prediction(frame, anchors, folds=None, variant='full_CMCRNet', root=ROOT):
    """Predict with a supplied teacher anchor; folds routes development OOF rows."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(2)

    class CMCRNet(nn.Module):
        def __init__(self, coupling):
            super().__init__()
            self.use_coupling = coupling
            self.encoder = nn.Sequential(nn.Linear(8, 64), nn.SiLU(), nn.LayerNorm(64),
                nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU())
            self.demand_head = nn.Linear(32, 1)
            self.risk_residual_head = nn.Linear(32, 1)
            self.link_scale_raw = nn.Parameter(torch.tensor(.5))
        def forward(self, x, base):
            latent = self.encoder(x)
            demand = base+.35*torch.tanh(self.demand_head(latent).squeeze(-1))
            logit = self.risk_residual_head(latent).squeeze(-1)
            if self.use_coupling:
                logit = logit+F.softplus(self.link_scale_raw)*(demand-2.6)
            return demand, torch.sigmoid(logit)

    x = frame[SUPPORT_FEATURES].to_numpy(np.float32)
    anchors = np.asarray(anchors, dtype=np.float32)
    if anchors.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(anchors).all():
        raise ValueError('Finite features and one finite teacher anchor per row are required.')
    if folds is not None:
        folds = np.asarray(folds, dtype=int)
        if folds.shape != (len(x),) or not np.isin(folds, np.arange(1, 6)).all():
            raise ValueError('OOF fold assignments must be integers from 1 to 5.')
    demands, probabilities = [], []
    for fold in range(1, 6):
        state = torch.load(Path(root)/'models/support'/f'{variant}_fold{fold}.pt', map_location='cpu', weights_only=False)
        if state['features'] != SUPPORT_FEATURES:
            raise ValueError('Checkpoint feature order differs from the release contract.')
        scaled = x.copy()
        scaled -= state['scaler_mean']
        scaled /= state['scaler_scale']
        model = CMCRNet(state['variant']['use_coupling'])
        model.load_state_dict(state['model_state'])
        model.eval()
        with torch.no_grad():
            d, p = model(torch.from_numpy(scaled), torch.from_numpy(anchors))
        demands.append(d.numpy())
        probabilities.append(p.numpy())
    dm, pm = np.vstack(demands), np.vstack(probabilities)
    if folds is None:
        demand, probability = dm.mean(axis=0), pm.mean(axis=0)
    else:
        row = np.arange(len(x))
        demand, probability = dm[folds-1, row], pm[folds-1, row]
    output = frame.copy()
    output['teacher_anchor_MPa'] = anchors
    output['predicted_latent_demand_MPa'] = demand
    output['predicted_no_pass_probability'] = probability
    output['predicted_no_pass'] = probability >= .5
    output['clipped_demand_MPa'] = np.clip(demand, .5, 2.6)
    # A finite demand and a passing pressure level are undefined for predicted no-pass cases.
    output['finite_demand_MPa'] = np.where(probability < .5, np.clip(demand, .5, 2.6), np.nan)
    output['decoded_pressure_level_MPa'] = np.where(probability < .5, nearest_pressure(np.clip(demand, .5, 2.6)), np.nan)
    return output
