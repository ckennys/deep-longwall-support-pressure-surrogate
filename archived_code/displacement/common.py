"""Immutable experiment identities and numeric helpers for the Step 5 repair."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
AUDIT = ROOT / "分析/数值质量统一裁决_20261024"
PREVIOUS = ROOT / "分析/第5步_大应变参考恢复与冻结模型诊断_20261025"
MODEL_DIR = ROOT / "分析/第一篇_位移激进结构优化_20261022"
INPUTS = ROOT / "分析/第一篇_PCA_存量数据广泛优化_20261022/data/historical_405_comparison_inputs.npz"
CACHE = ROOT / "图表/第一篇_第五章_D1_D2_D3初版_20261023/D3/data/Fig07_all_selected_fields.npz"
SOURCE_CASES = ROOT / "工况/flac3d_stage52_independent_ensemble_confirmation_cases"
FEATURES = ["H_m", "lambda", "M_m", "q_top_MPa", "alpha_E_roof", "alpha_c_roof",
            "phi_roof_deg", "alpha_t_roof", "support_pressure_MPa", "panel_span_m",
            "roadway_width_m", "slice_width_m", "direct_roof_thickness_factor", "step_fraction"]
HASHES = {"uz": "6e600a63a0d664dd4bbacc739a2313aaad0d74b2a5f1e4166562adfef80a03e3",
          "ux": "e1a85fbb370f9d944c0af7fdb8f0977e5b1db844c70fc468d797894dfbc3e9c3"}


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(12):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(.025*(attempt+1), .25))


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def old_module():
    sys.path.insert(0, str(MODEL_DIR))
    import aggressive_models
    aggressive_models.THREADS = 2
    return aggressive_models


def inverse(z):
    return np.asarray(z)*np.abs(z)


def metrics(truth, prediction, mask=None):
    a, p = np.asarray(truth, float), np.asarray(prediction, float)
    if mask is not None:
        a, p = a[mask], p[mask]
    a, p = a.reshape(-1), p.reshape(-1)
    if not len(a) or a.shape != p.shape or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Nonempty finite matching field arrays are required")
    e = p-a
    sse, sst = float(e@e), float(np.sum((a-a.mean())**2))
    peak, predicted_peak = float(np.max(np.abs(a))), float(np.max(np.abs(p)))
    return {"n": len(a), "r2": 1-sse/sst if sst > 0 else np.nan,
            "rmse_m": float(np.sqrt(sse/len(a))), "mae_m": float(np.mean(np.abs(e))),
            "sse_m2": sse, "truth_sum": float(a.sum()), "truth_sum_squares": float(a@a),
            "absolute_error_sum": float(np.abs(e).sum()), "reference_abs_peak_m": peak,
            "prediction_abs_peak_m": predicted_peak, "absolute_peak_error_m": abs(predicted_peak-peak),
            "peak_error_fraction_floor1mm": abs(predicted_peak-peak)/max(peak, .001),
            "reference_min_m": float(a.min()), "reference_max_m": float(a.max()),
            "prediction_min_m": float(p.min()), "prediction_max_m": float(p.max())}


def aggregate(frame):
    n, sse = int(frame.n.sum()), float(frame.sse_m2.sum())
    sst = float(frame.truth_sum_squares.sum()-frame.truth_sum.sum()**2/n)
    return {"n": n, "r2": 1-sse/sst if sst > 0 else np.nan,
            "rmse_m": float(np.sqrt(sse/n)), "mae_m": float(frame.absolute_error_sum.sum()/n),
            "snapshots": len(frame), "physical_groups": frame.physical_group.nunique(),
            "snapshot_rmse_median_m": float(frame.rmse_m.median()),
            "snapshot_rmse_p90_m": float(frame.rmse_m.quantile(.9)),
            "snapshot_rmse_p95_m": float(frame.rmse_m.quantile(.95)),
            "snapshot_rmse_max_m": float(frame.rmse_m.max()),
            "peak_error_median_fraction": float(frame.peak_error_fraction_floor1mm.median()),
            "peak_error_p90_fraction": float(frame.peak_error_fraction_floor1mm.quantile(.9))}


def protected_hashes():
    files = [CACHE, INPUTS, AUDIT / "sensitivity_policy.json",
             AUDIT / "review_completion/completion_policy.json",
             AUDIT / "review_completion/405快照_完成层质量状态.csv",
             AUDIT / "review_completion/D3_statistics/D3_最终质量合格子集.csv"]
    files += [MODEL_DIR / f"models/{t}_m_Sqrt_Neural64.joblib" for t in HASHES]
    return {str(p): sha(p) for p in files}
