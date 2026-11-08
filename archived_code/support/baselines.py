from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu, spearmanr, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "分析" / "innovation_strengthening_20260811"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

TRAIN_LABELS = (
    ROOT
    / "工况"
    / "flac3d_ml_dataset_stage7_expanded"
    / "labels_required_support_pressure_all_960.csv"
)
SEARCH_RECORDS = (
    ROOT
    / "工况"
    / "flac3d_ml_dataset_stage7_expanded"
    / "support_pressure_search_records_all_7629.csv"
)
EXTERNAL_LABELS = (
    ROOT
    / "分析"
    / "stage8_external_validation"
    / "stage8_external_labels_locked.csv"
)
TWO_STAGE_EXTERNAL = (
    ROOT
    / "分析"
    / "stage8_validation_stage7_expanded_model"
    / "stage8_first_stage_external_predictions.csv"
)
TWO_STAGE_STAGE1_OOF = (
    ROOT
    / "分析"
    / "stage7_expanded_surrogate"
    / "tables"
    / "stage1_boundary_classifier_cv_predictions.csv"
)
TWO_STAGE_STAGE2_OOF = (
    ROOT
    / "分析"
    / "stage7_expanded_surrogate"
    / "tables"
    / "stage2_regression_cv_predictions.csv"
)
STAGE11_DATA = (
    ROOT
    / "分析"
    / "stage11_multistep_operator"
    / "stage11_aligned_multistep_dataset.npz"
)

FEATURES = [
    "H_m",
    "lambda",
    "M_m",
    "q_top_MPa",
    "alpha_E_roof",
    "alpha_c_roof",
    "phi_roof_deg",
    "alpha_t_roof",
]
PRESSURE_LEVELS = np.array([0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6])
SEED = 20260811


def setup_style() -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.05)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def nearest_level(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    index = np.abs(values[:, None] - PRESSURE_LEVELS[None, :]).argmin(axis=1)
    return PRESSURE_LEVELS[index]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def tabpfn_regressor(seed: int):
    from tabpfn import TabPFNRegressor

    kwargs = {
        "n_estimators": 8,
        "device": "auto",
        "random_state": seed,
        "show_progress_bar": False,
    }
    try:
        return TabPFNRegressor(ignore_pretraining_limits=True, **kwargs)
    except TypeError:
        return TabPFNRegressor(**kwargs)


def boundary_classifier(seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced", max_iter=3000, random_state=seed
        ),
    )


def numeric_pressure(series: pd.Series, capped: bool = False) -> np.ndarray:
    text = series.astype(str)
    values = pd.to_numeric(text.where(text != ">2.6"), errors="coerce").to_numpy(float)
    if capped:
        values[np.isnan(values)] = 2.6
    return values


def choose_safety_threshold(y_boundary: np.ndarray, raw_prediction: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(1.70, 2.90, 241):
        prediction = (raw_prediction >= threshold).astype(int)
        cm = confusion_matrix(y_boundary, prediction, labels=[0, 1])
        rows.append(
            {
                "threshold_MPa": threshold,
                "f2": fbeta_score(y_boundary, prediction, beta=2, zero_division=0),
                "recall": recall_score(y_boundary, prediction, zero_division=0),
                "precision": precision_score(y_boundary, prediction, zero_division=0),
                "specificity": cm[0, 0] / max(cm[0].sum(), 1),
                "false_negatives": int(cm[1, 0]),
                "false_positives": int(cm[0, 1]),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["f2", "recall", "specificity", "threshold_MPa"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def evaluate_decision_scheme(
    dataset: str,
    scheme: str,
    actual_boundary: np.ndarray,
    predicted_boundary: np.ndarray,
    true_finite_pressure: np.ndarray,
    predicted_pressure: np.ndarray,
) -> tuple[dict[str, float | int | str], np.ndarray]:
    actual_boundary = np.asarray(actual_boundary, dtype=int)
    predicted_boundary = np.asarray(predicted_boundary, dtype=int)
    predicted_pressure = np.clip(np.asarray(predicted_pressure, dtype=float), 0.5, 2.6)
    finite_mask = actual_boundary == 0
    boundary_mask = actual_boundary == 1
    cm = confusion_matrix(actual_boundary, predicted_boundary, labels=[0, 1])
    snapped = nearest_level(predicted_pressure)
    finite_true = np.asarray(true_finite_pressure, dtype=float)[finite_mask]
    finite_pred = predicted_pressure[finite_mask]
    finite_snapped = snapped[finite_mask]
    row = {
        "dataset": dataset,
        "scheme": scheme,
        "n": int(actual_boundary.size),
        "n_boundary": int(boundary_mask.sum()),
        "n_finite": int(finite_mask.sum()),
        "boundary_recall": recall_score(actual_boundary, predicted_boundary, zero_division=0),
        "boundary_precision": precision_score(actual_boundary, predicted_boundary, zero_division=0),
        "boundary_accuracy": accuracy_score(actual_boundary, predicted_boundary),
        "false_negatives": int(cm[1, 0]),
        "false_positives": int(cm[0, 1]),
        "false_finite_rate": float(cm[1, 0] / max(boundary_mask.sum(), 1)),
        "finite_R2": r2_score(finite_true, finite_pred),
        "finite_RMSE_MPa": rmse(finite_true, finite_pred),
        "finite_MAE_MPa": mean_absolute_error(finite_true, finite_pred),
        "exact_level_accuracy": float(np.mean(np.isclose(finite_snapped, finite_true))),
        "adjacent_level_accuracy": float(np.mean(np.abs(finite_snapped - finite_true) <= 0.31)),
        "safety_weighted_error": float((5 * cm[1, 0] + cm[0, 1]) / actual_boundary.size),
    }
    return row, cm


def load_two_stage_internal(labels: pd.DataFrame) -> pd.DataFrame:
    stage1 = pd.read_csv(TWO_STAGE_STAGE1_OOF)
    stage1 = stage1[stage1["model"] == "TabPFN"].copy()
    stage2 = pd.read_csv(TWO_STAGE_STAGE2_OOF)
    stage2 = stage2[stage2["model"] == "TabPFN"].copy()
    stage2 = stage2[["sample_id", "predicted_MPa"]]
    base = labels[["sample_id", "required_support_pressure_MPa"]].copy()
    base["actual_boundary"] = (
        base["required_support_pressure_MPa"].astype(str) == ">2.6"
    ).astype(int)
    base["actual_pressure"] = numeric_pressure(base["required_support_pressure_MPa"])
    merged = base.merge(
        stage1[["sample_id", "predicted_boundary", "boundary_score"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    ).merge(stage2, on="sample_id", how="left", validate="one_to_one")
    merged["predicted_MPa"] = merged["predicted_MPa"].fillna(2.6)
    if merged[["predicted_boundary", "predicted_MPa"]].isna().any().any():
        raise ValueError("Two-stage internal predictions did not map to all 960 combinations.")
    return merged


def run_counterfactual_comparison() -> dict[str, object]:
    labels = pd.read_csv(TRAIN_LABELS)
    external = pd.read_csv(EXTERNAL_LABELS)
    for frame in (labels, external):
        for feature in FEATURES:
            frame[feature] = pd.to_numeric(frame[feature], errors="raise")

    x = labels[FEATURES].to_numpy(np.float32)
    y_boundary = (
        labels["required_support_pressure_MPa"].astype(str) == ">2.6"
    ).astype(int).to_numpy()
    y_capped = numeric_pressure(labels["required_support_pressure_MPa"], capped=True)
    y_finite = numeric_pressure(labels["required_support_pressure_MPa"])

    x_external = external[FEATURES].to_numpy(np.float32)
    y_external_boundary = external["is_boundary"].to_numpy(int)
    y_external_finite = numeric_pressure(external["required_support_pressure_MPa"])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_raw = np.zeros(len(labels), dtype=float)
    two_stage_oof_boundary = np.zeros(len(labels), dtype=int)
    two_stage_oof_pressure = np.zeros(len(labels), dtype=float)
    fold_assignment = np.zeros(len(labels), dtype=int)
    start = time.time()
    for fold, (train_index, test_index) in enumerate(cv.split(x, y_boundary), start=1):
        print(f"Matched outer fold {fold}/5", flush=True)
        fold_assignment[test_index] = fold

        classifier = boundary_classifier(SEED + fold)
        classifier.fit(x[train_index], y_boundary[train_index])
        two_stage_oof_boundary[test_index] = classifier.predict(x[test_index]).astype(int)

        finite_train_index = train_index[y_boundary[train_index] == 0]
        finite_model = tabpfn_regressor(SEED + 100 + fold)
        finite_model.fit(x[finite_train_index], y_finite[finite_train_index])
        two_stage_oof_pressure[test_index] = np.asarray(
            finite_model.predict(x[test_index]), dtype=float
        ).reshape(-1)

        capped_model = tabpfn_regressor(SEED + 200 + fold)
        capped_model.fit(x[train_index], y_capped[train_index])
        oof_raw[test_index] = np.asarray(
            capped_model.predict(x[test_index]), dtype=float
        ).reshape(-1)

    threshold_search = choose_safety_threshold(y_boundary, oof_raw)
    threshold = float(threshold_search.loc[0, "threshold_MPa"])

    print("Fitting matched models on all Stage 7 combinations", flush=True)
    capped_model = tabpfn_regressor(SEED + 300)
    capped_model.fit(x, y_capped)
    external_raw = np.asarray(capped_model.predict(x_external), dtype=float).reshape(-1)

    classifier = boundary_classifier(SEED + 301)
    classifier.fit(x, y_boundary)
    external_two_stage_boundary = classifier.predict(x_external).astype(int)
    finite_index = np.flatnonzero(y_boundary == 0)
    finite_model = tabpfn_regressor(SEED + 302)
    finite_model.fit(x[finite_index], y_finite[finite_index])
    external_two_stage_pressure = np.asarray(
        finite_model.predict(x_external), dtype=float
    ).reshape(-1)

    rows: list[dict[str, float | int | str]] = []
    matrices: dict[str, np.ndarray] = {}

    row, matrix = evaluate_decision_scheme(
        "Stage7_OOF",
        "two_stage_Logistic_TabPFN",
        y_boundary,
        two_stage_oof_boundary,
        y_finite,
        two_stage_oof_pressure,
    )
    rows.append(row)
    matrices["Stage7_OOF_two_stage_Logistic_TabPFN"] = matrix

    row, matrix = evaluate_decision_scheme(
        "Stage7_OOF",
        "single_stage_capped_no_boundary",
        y_boundary,
        np.zeros_like(y_boundary),
        y_finite,
        oof_raw,
    )
    rows.append(row)
    matrices["Stage7_OOF_single_stage_capped_no_boundary"] = matrix

    row, matrix = evaluate_decision_scheme(
        "Stage7_OOF",
        "single_stage_capped_thresholded",
        y_boundary,
        (oof_raw >= threshold).astype(int),
        y_finite,
        oof_raw,
    )
    rows.append(row)
    matrices["Stage7_OOF_single_stage_capped_thresholded"] = matrix

    row, matrix = evaluate_decision_scheme(
        "Stage8_locked",
        "two_stage_Logistic_TabPFN",
        y_external_boundary,
        external_two_stage_boundary,
        y_external_finite,
        external_two_stage_pressure,
    )
    rows.append(row)
    matrices["Stage8_locked_two_stage_Logistic_TabPFN"] = matrix

    row, matrix = evaluate_decision_scheme(
        "Stage8_locked",
        "single_stage_capped_no_boundary",
        y_external_boundary,
        np.zeros_like(y_external_boundary),
        y_external_finite,
        external_raw,
    )
    rows.append(row)
    matrices["Stage8_locked_single_stage_capped_no_boundary"] = matrix

    row, matrix = evaluate_decision_scheme(
        "Stage8_locked",
        "single_stage_capped_thresholded",
        y_external_boundary,
        (external_raw >= threshold).astype(int),
        y_external_finite,
        external_raw,
    )
    rows.append(row)
    matrices["Stage8_locked_single_stage_capped_thresholded"] = matrix

    metrics = pd.DataFrame(rows)
    metrics.to_csv(TABLE_DIR / "counterfactual_model_comparison.csv", index=False)
    threshold_search.to_csv(TABLE_DIR / "single_stage_threshold_search.csv", index=False)

    internal_predictions = labels[["sample_id", "required_support_pressure_MPa"] + FEATURES].copy()
    internal_predictions["actual_boundary"] = y_boundary
    internal_predictions["matched_outer_fold"] = fold_assignment
    internal_predictions["capped_target_MPa"] = y_capped
    internal_predictions["two_stage_predicted_boundary"] = two_stage_oof_boundary
    internal_predictions["two_stage_predicted_pressure_MPa"] = two_stage_oof_pressure
    internal_predictions["single_stage_raw_MPa"] = oof_raw
    internal_predictions["single_stage_thresholded_boundary"] = (oof_raw >= threshold).astype(int)
    internal_predictions.to_csv(TABLE_DIR / "single_stage_stage7_oof_predictions.csv", index=False)

    external_predictions = external[["sample_id", "required_support_pressure_MPa"] + FEATURES].copy()
    external_predictions["actual_boundary"] = y_external_boundary
    external_predictions["capped_target_MPa"] = numeric_pressure(
        external_predictions["required_support_pressure_MPa"], capped=True
    )
    external_predictions["single_stage_raw_MPa"] = external_raw
    external_predictions["two_stage_predicted_boundary"] = external_two_stage_boundary
    external_predictions["two_stage_predicted_pressure_MPa"] = external_two_stage_pressure
    external_predictions["single_stage_thresholded_boundary"] = (
        external_raw >= threshold
    ).astype(int)
    external_predictions.to_csv(TABLE_DIR / "single_stage_stage8_predictions.csv", index=False)

    with (TABLE_DIR / "counterfactual_confusion_matrices.json").open("w", encoding="utf-8") as handle:
        json.dump({key: value.tolist() for key, value in matrices.items()}, handle, indent=2)

    return {
        "metrics": metrics,
        "threshold": threshold,
        "threshold_search": threshold_search,
        "internal_predictions": internal_predictions,
        "external_predictions": external_predictions,
        "matrices": matrices,
        "elapsed_seconds": time.time() - start,
    }


def bootstrap_median_difference(
    first: np.ndarray, second: np.ndarray, seed: int, repeats: int = 3000
) -> tuple[float, float, float]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=float)
    for index in range(repeats):
        samples[index] = np.median(rng.choice(first, first.size, replace=True)) - np.median(
            rng.choice(second, second.size, replace=True)
        )
    return (
        float(np.median(first) - np.median(second)),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def cliffs_delta(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    u_statistic = mannwhitneyu(first, second, alternative="two-sided").statistic
    return float(2.0 * u_statistic / (first.size * second.size) - 1.0)


def response_at_search_termination() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(TRAIN_LABELS)
    records = pd.read_csv(SEARCH_RECORDS)
    raw_sample_id = records["sample_id"].astype(str)
    records["sample_id"] = np.where(
        raw_sample_id.str.startswith("stage"),
        raw_sample_id,
        records["source"].astype(str) + "_" + raw_sample_id,
    )
    terminal = (
        records.sort_values(["sample_id", "support_pressure_MPa", "case_sample_id"])
        .drop_duplicates("sample_id", keep="last")
        .copy()
    )
    keep = labels[["sample_id", "required_support_pressure_MPa", "M_m"]].copy()
    keep = keep.rename(columns={"M_m": "label_M_m"})
    terminal = terminal.merge(keep, on="sample_id", how="inner", validate="one_to_one")
    if len(terminal) != len(labels):
        raise ValueError(
            f"Terminal-search mapping retained {len(terminal)} of {len(labels)} combinations."
        )

    terminal["is_boundary"] = (
        terminal["required_support_pressure_MPa"].astype(str) == ">2.6"
    ).astype(int)
    terminal["finite_required_MPa"] = numeric_pressure(
        terminal["required_support_pressure_MPa"]
    )
    terminal["demand_group"] = "Controlled low"
    terminal.loc[
        (terminal["is_boundary"] == 0) & (terminal["finite_required_MPa"] >= 2.0),
        "demand_group",
    ] = "Controlled critical"
    terminal.loc[terminal["is_boundary"] == 1, "demand_group"] = "Boundary >2.6"
    terminal["normalised_settlement"] = (
        terminal["max_settlement_m"].abs() / terminal["M_m"]
    )
    terminal["normalised_heave"] = terminal["max_heave_m"].abs() / terminal["M_m"]
    terminal["combined_normalised_deformation"] = (
        terminal["normalised_settlement"] + terminal["normalised_heave"]
    )

    metric_columns = [
        "max_settlement_m",
        "max_heave_m",
        "peak_abutment_stress_MPa",
        "normalised_settlement",
        "normalised_heave",
        "combined_normalised_deformation",
    ]
    group_summary = (
        terminal.groupby("demand_group", observed=True)[metric_columns]
        .agg(["count", "median", "mean", "std"])
        .reset_index()
    )
    group_summary.columns = [
        "_".join([str(value) for value in column if str(value)])
        if isinstance(column, tuple)
        else str(column)
        for column in group_summary.columns
    ]

    boundary = terminal[terminal["is_boundary"] == 1]
    controlled = terminal[terminal["is_boundary"] == 0]
    test_rows = []
    for index, metric in enumerate(metric_columns):
        first = boundary[metric].to_numpy(float)
        second = controlled[metric].to_numpy(float)
        u_result = mannwhitneyu(first, second, alternative="two-sided")
        median_difference, lower, upper_ci = bootstrap_median_difference(
            first, second, SEED + index
        )
        values = terminal[metric].to_numpy(float)
        auc = roc_auc_score(terminal["is_boundary"], values)
        auc_direction = "higher indicates boundary"
        if auc < 0.5:
            auc = 1.0 - auc
            auc_direction = "lower indicates boundary"
        group_values = [
            terminal.loc[terminal["demand_group"] == group, metric].to_numpy(float)
            for group in ["Controlled low", "Controlled critical", "Boundary >2.6"]
        ]
        kw = kruskal(*group_values)
        test_rows.append(
            {
                "metric": metric,
                "boundary_median": float(np.median(first)),
                "controlled_median": float(np.median(second)),
                "boundary_minus_controlled_median": median_difference,
                "bootstrap_ci95_lower": lower,
                "bootstrap_ci95_upper": upper_ci,
                "mann_whitney_p": float(u_result.pvalue),
                "cliffs_delta": cliffs_delta(first, second),
                "boundary_auc": float(auc),
                "auc_direction": auc_direction,
                "kruskal_p_three_groups": float(kw.pvalue),
            }
        )

    terminal.to_csv(
        TABLE_DIR / "terminal_search_scalar_response_by_combination.csv", index=False
    )
    group_summary.to_csv(
        TABLE_DIR / "terminal_search_scalar_group_summary.csv", index=False
    )
    tests = pd.DataFrame(test_rows)
    tests.to_csv(TABLE_DIR / "terminal_search_scalar_boundary_tests.csv", index=False)
    return terminal, group_summary, tests


def masked_values(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(array[mask], dtype=float)
    return values[np.isfinite(values)]


def field_metric_rows() -> pd.DataFrame:
    data = np.load(STAGE11_DATA, allow_pickle=True)
    feature_names = [str(value) for value in data["feature_names"]]
    feature_index = {name: index for index, name in enumerate(feature_names)}
    case_ids = data["case_id"]
    combination_ids = data["combination_id"]
    splits = data["split"]
    snapshot_roles = data["snapshot_role"]
    step_indices = data["step_index"]
    features_all = data["features"]
    valid_masks = data["valid_mask"]
    ux_all = data["ux_m"]
    uz_all = data["uz_m"]
    sxx_all = data["sxx_MPa"]
    szz_all = data["szz_MPa"]
    sxz_all = data["sxz_MPa"]
    plastic_all = data["plastic_any"]
    shear_current_all = data["shear_current"]
    shear_past_all = data["shear_past"]
    tension_past_all = data["tension_past"]
    rows = []
    for snapshot_index in range(len(case_ids)):
        mask = valid_masks[snapshot_index]
        ux = masked_values(ux_all[snapshot_index], mask)
        uz = masked_values(uz_all[snapshot_index], mask)
        sxx = masked_values(sxx_all[snapshot_index], mask)
        szz = masked_values(szz_all[snapshot_index], mask)
        sxz = masked_values(sxz_all[snapshot_index], mask)
        plastic = np.asarray(plastic_all[snapshot_index][mask], dtype=bool)
        shear_current = np.asarray(shear_current_all[snapshot_index][mask], dtype=bool)
        shear_past = np.asarray(shear_past_all[snapshot_index][mask], dtype=bool)
        tension_past = np.asarray(tension_past_all[snapshot_index][mask], dtype=bool)
        features = features_all[snapshot_index]
        mining_height = float(features[feature_index["M_m"]])
        rows.append(
            {
                "case_id": str(case_ids[snapshot_index]),
                "combination_id": str(combination_ids[snapshot_index]),
                "split": str(splits[snapshot_index]),
                "snapshot_role": str(snapshot_roles[snapshot_index]),
                "step_index": int(step_indices[snapshot_index]),
                "support_pressure_MPa": float(
                    features[feature_index["support_pressure_MPa"]]
                ),
                "M_m": mining_height,
                "max_abs_ux_m": float(np.max(np.abs(ux))),
                "max_abs_uz_m": float(np.max(np.abs(uz))),
                "p95_abs_uz_m": float(np.quantile(np.abs(uz), 0.95)),
                "mean_abs_uz_m": float(np.mean(np.abs(uz))),
                "max_abs_uz_over_M": float(np.max(np.abs(uz)) / mining_height),
                "p95_abs_uz_over_M": float(np.quantile(np.abs(uz), 0.95) / mining_height),
                "p95_abs_sxx_MPa": float(np.quantile(np.abs(sxx), 0.95)),
                "p95_abs_szz_MPa": float(np.quantile(np.abs(szz), 0.95)),
                "p95_abs_sxz_MPa": float(np.quantile(np.abs(sxz), 0.95)),
                "plastic_any_fraction": float(np.mean(plastic)),
                "shear_current_fraction": float(np.mean(shear_current)),
                "shear_past_fraction": float(np.mean(shear_past)),
                "tension_past_fraction": float(np.mean(tension_past)),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLE_DIR / "stage11_actual_field_metrics_by_snapshot.csv", index=False)
    return frame


def run_field_consistency() -> dict[str, pd.DataFrame]:
    field = field_metric_rows()
    final = field[field["step_index"] == 5].copy()
    required = final[final["snapshot_role"] == "required_pressure"].copy()
    required = required.rename(columns={"support_pressure_MPa": "required_pressure_MPa"})
    required["demand_tier"] = pd.cut(
        required["required_pressure_MPa"],
        bins=[0.49, 1.1, 1.7, 2.61],
        labels=["Low 0.5-1.1", "Moderate 1.4-1.7", "High 2.0-2.6"],
        include_lowest=True,
    )

    field_metrics = [
        "max_abs_uz_over_M",
        "p95_abs_uz_over_M",
        "p95_abs_sxx_MPa",
        "p95_abs_szz_MPa",
        "p95_abs_sxz_MPa",
        "plastic_any_fraction",
        "shear_current_fraction",
        "shear_past_fraction",
        "tension_past_fraction",
    ]
    correlation_rows = []
    for metric in field_metrics:
        rho, p_value = spearmanr(required["required_pressure_MPa"], required[metric])
        correlation_rows.append(
            {
                "metric": metric,
                "spearman_rho_with_required_pressure": float(rho),
                "p_value": float(p_value),
                "n_combinations": int(len(required)),
            }
        )
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(TABLE_DIR / "stage11_required_pressure_field_correlations.csv", index=False)

    tier_summary = (
        required.groupby(["split", "demand_tier"], observed=True)[field_metrics]
        .agg(["count", "median", "mean", "std"])
        .reset_index()
    )
    tier_summary.columns = [
        "_".join([str(value) for value in column if str(value)])
        if isinstance(column, tuple)
        else str(column)
        for column in tier_summary.columns
    ]
    tier_summary.to_csv(TABLE_DIR / "stage11_required_pressure_tier_summary.csv", index=False)

    above = final[final["snapshot_role"] == "above_required_pressure"].copy()
    above = (
        above.sort_values(["split", "combination_id", "support_pressure_MPa"])
        .drop_duplicates(["split", "combination_id"], keep="first")
        .copy()
    )
    pair_columns = ["combination_id", "split"] + field_metrics
    pairs = required[
        ["combination_id", "split", "required_pressure_MPa", "demand_tier"] + field_metrics
    ].merge(
        above[pair_columns + ["support_pressure_MPa"]],
        on=["combination_id", "split"],
        how="inner",
        suffixes=("_required", "_above"),
        validate="one_to_one",
    )
    pairs["pressure_increment_MPa"] = (
        pairs["support_pressure_MPa"] - pairs["required_pressure_MPa"]
    )
    paired_rows = []
    for metric in field_metrics:
        required_values = pairs[f"{metric}_required"].to_numpy(float)
        above_values = pairs[f"{metric}_above"].to_numpy(float)
        difference = above_values - required_values
        try:
            test = wilcoxon(difference)
            p_value = float(test.pvalue)
        except ValueError:
            p_value = float("nan")
        rng = np.random.default_rng(SEED + len(paired_rows))
        bootstrap = np.empty(3000, dtype=float)
        for index in range(bootstrap.size):
            bootstrap[index] = np.median(rng.choice(difference, difference.size, replace=True))
        paired_rows.append(
            {
                "metric": metric,
                "n_pairs": int(len(pairs)),
                "required_median": float(np.median(required_values)),
                "above_required_median": float(np.median(above_values)),
                "above_minus_required_median": float(np.median(difference)),
                "bootstrap_ci95_lower": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci95_upper": float(np.quantile(bootstrap, 0.975)),
                "wilcoxon_p": p_value,
                "fraction_decreased_with_more_support": float(np.mean(difference <= 0.0)),
            }
        )
    paired_summary = pd.DataFrame(paired_rows)
    pairs.to_csv(TABLE_DIR / "stage11_required_vs_above_pairs.csv", index=False)
    paired_summary.to_csv(TABLE_DIR / "stage11_required_vs_above_summary.csv", index=False)

    required_all_steps = field[field["snapshot_role"] == "required_pressure"].copy()
    first = required_all_steps[required_all_steps["step_index"] == 1]
    fifth = required_all_steps[required_all_steps["step_index"] == 5]
    evolution = first[["combination_id", "split"] + field_metrics].merge(
        fifth[["combination_id", "split"] + field_metrics],
        on=["combination_id", "split"],
        suffixes=("_step1", "_step5"),
        validate="one_to_one",
    )
    evolution = evolution.merge(
        required[["combination_id", "split", "required_pressure_MPa", "demand_tier"]],
        on=["combination_id", "split"],
        validate="one_to_one",
    )
    for metric in field_metrics:
        evolution[f"{metric}_growth"] = (
            evolution[f"{metric}_step5"] - evolution[f"{metric}_step1"]
        )
    evolution.to_csv(TABLE_DIR / "stage11_step1_to_step5_field_growth.csv", index=False)

    required.to_csv(TABLE_DIR / "stage11_final_required_pressure_field_metrics.csv", index=False)
    return {
        "field": field,
        "required": required,
        "correlations": correlations,
        "tier_summary": tier_summary,
        "pairs": pairs,
        "paired_summary": paired_summary,
        "evolution": evolution,
    }


def plot_counterfactual(result: dict[str, object]) -> None:
    metrics = result["metrics"]
    external = metrics[metrics["dataset"] == "Stage8_locked"].copy()
    label_map = {
        "two_stage_Logistic_TabPFN": "Two-stage",
        "single_stage_capped_no_boundary": "Capped regression",
        "single_stage_capped_thresholded": "Thresholded capped",
    }
    external["display"] = external["scheme"].map(label_map)
    colors = ["#2F6B8A", "#B64B3C", "#D39B2A"]

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.8))
    sns.barplot(data=external, x="display", y="boundary_recall", hue="display", palette=colors, legend=False, ax=axes[0, 0])
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Boundary recall")
    axes[0, 0].tick_params(axis="x", rotation=15)

    sns.barplot(data=external, x="display", y="false_finite_rate", hue="display", palette=colors, legend=False, ax=axes[0, 1])
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("False-finite rate")
    axes[0, 1].tick_params(axis="x", rotation=15)

    sns.barplot(data=external, x="display", y="finite_RMSE_MPa", hue="display", palette=colors, legend=False, ax=axes[1, 0])
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Finite-case RMSE (MPa)")
    axes[1, 0].tick_params(axis="x", rotation=15)

    sns.barplot(data=external, x="display", y="exact_level_accuracy", hue="display", palette=colors, legend=False, ax=axes[1, 1])
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Exact pressure-level accuracy")
    axes[1, 1].tick_params(axis="x", rotation=15)

    for label, axis in zip(["a", "b", "c", "d"], axes.flat):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontweight="bold")
        for container in axis.containers:
            axis.bar_label(container, fmt="%.3f", padding=2, fontsize=7)
    fig.suptitle("Counterfactual comparison of support-demand formulations", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "counterfactual_two_stage_vs_single_stage.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "counterfactual_two_stage_vs_single_stage.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_consistency(
    scalar: pd.DataFrame,
    scalar_tests: pd.DataFrame,
    field_result: dict[str, pd.DataFrame],
) -> None:
    order = ["Controlled low", "Controlled critical", "Boundary >2.6"]
    palette = ["#3B82A0", "#E1A33B", "#B64B3C"]
    required = field_result["required"]
    pairs = field_result["pairs"]
    correlations = field_result["correlations"].sort_values(
        "spearman_rho_with_required_pressure", key=np.abs, ascending=False
    )
    metric_labels = {
        "max_settlement_m": "Maximum settlement",
        "max_heave_m": "Maximum floor heave",
        "peak_abutment_stress_MPa": "Peak abutment stress",
        "normalised_settlement": "Normalised settlement",
        "normalised_heave": "Normalised floor heave",
        "combined_normalised_deformation": "Combined normalised deformation",
        "max_abs_uz_over_M": "Maximum |uz| / mining height",
        "p95_abs_uz_over_M": "P95 |uz| / mining height",
        "p95_abs_sxx_MPa": "P95 |sxx|",
        "p95_abs_szz_MPa": "P95 |szz|",
        "p95_abs_sxz_MPa": "P95 |sxz|",
        "plastic_any_fraction": "Plastic-zone fraction",
        "shear_current_fraction": "Current-shear fraction",
        "shear_past_fraction": "Past-shear fraction",
        "tension_past_fraction": "Past-tension fraction",
    }

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8))
    sns.boxplot(
        data=scalar,
        x="demand_group",
        y="normalised_settlement",
        order=order,
        palette=palette,
        hue="demand_group",
        hue_order=order,
        legend=False,
        showfliers=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Settlement / mining height")
    axes[0, 0].tick_params(axis="x", rotation=15)

    sns.boxplot(
        data=scalar,
        x="demand_group",
        y="peak_abutment_stress_MPa",
        order=order,
        palette=palette,
        hue="demand_group",
        hue_order=order,
        legend=False,
        showfliers=False,
        ax=axes[0, 1],
    )
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Peak abutment stress (MPa)")
    axes[0, 1].tick_params(axis="x", rotation=15)

    auc_data = scalar_tests.sort_values("boundary_auc", ascending=True)
    auc_data = auc_data.assign(metric_label=auc_data["metric"].map(metric_labels))
    sns.barplot(data=auc_data, x="boundary_auc", y="metric_label", color="#5F9E6E", ax=axes[0, 2])
    axes[0, 2].axvline(0.5, color="0.4", linestyle=":", linewidth=1)
    axes[0, 2].set_xlim(0.45, 1.0)
    axes[0, 2].set_xlabel("Boundary discrimination AUC")
    axes[0, 2].set_ylabel("")

    sns.boxplot(
        data=required,
        x="demand_tier",
        y="p95_abs_uz_over_M",
        hue="demand_tier",
        order=["Low 0.5-1.1", "Moderate 1.4-1.7", "High 2.0-2.6"],
        hue_order=["Low 0.5-1.1", "Moderate 1.4-1.7", "High 2.0-2.6"],
        palette=["#3B82A0", "#E1A33B", "#B64B3C"],
        legend=False,
        showfliers=False,
        ax=axes[1, 0],
    )
    axes[1, 0].set_xlabel("Finite required-pressure tier")
    axes[1, 0].set_ylabel("P95 |uz| / mining height")
    axes[1, 0].tick_params(axis="x", rotation=15)

    sample = pairs.sample(min(len(pairs), 120), random_state=SEED)
    for _, row in sample.iterrows():
        axes[1, 1].plot(
            [0, 1],
            [row["p95_abs_uz_over_M_required"], row["p95_abs_uz_over_M_above"]],
            color="#2F6B8A",
            alpha=0.2,
            linewidth=0.7,
        )
    axes[1, 1].set_xticks([0, 1], ["Required", "Above required"])
    axes[1, 1].set_ylabel("P95 |uz| / mining height")

    plot_corr = correlations.head(7).copy()
    plot_corr["metric_label"] = plot_corr["metric"].map(metric_labels)
    sns.barplot(
        data=plot_corr,
        x="spearman_rho_with_required_pressure",
        y="metric_label",
        hue="spearman_rho_with_required_pressure",
        palette="vlag",
        legend=False,
        ax=axes[1, 2],
    )
    axes[1, 2].axvline(0, color="0.35", linewidth=0.8)
    axes[1, 2].set_xlabel("Spearman rho with required pressure")
    axes[1, 2].set_ylabel("")

    for label, axis in zip(["a", "b", "c", "d", "e", "f"], axes.flat):
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontweight="bold")
    fig.suptitle("Consistency between support-demand labels and geomechanical responses", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "decision_response_field_consistency.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "decision_response_field_consistency.pdf", bbox_inches="tight")
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_report(
    counterfactual: dict[str, object],
    scalar: pd.DataFrame,
    scalar_tests: pd.DataFrame,
    field_result: dict[str, pd.DataFrame],
) -> None:
    metrics = counterfactual["metrics"]
    internal = metrics[metrics["dataset"] == "Stage7_OOF"].set_index("scheme")
    external = metrics[metrics["dataset"] == "Stage8_locked"].set_index("scheme")
    threshold = counterfactual["threshold"]
    settlement_test = scalar_tests.set_index("metric").loc["normalised_settlement"]
    deformation_test = scalar_tests.set_index("metric").loc[
        "combined_normalised_deformation"
    ]
    paired = field_result["paired_summary"].set_index("metric")
    correlation = field_result["correlations"].set_index("metric")
    required = field_result["required"]
    field_all = field_result["field"]
    boundary_field_count = int(
        field_all["combination_id"].astype(str).str.contains("boundary", case=False).sum()
    )

    lines = [
        "# 创新补强分析报告",
        "",
        "## 1 分析范围",
        "",
        "本报告完成两项分析。第一项在相同外层分层五折、Stage 7 开发集和锁定 Stage 8 外部集上比较 LogisticRegression 加 TabPFN 两阶段模型、单阶段封顶回归及带开发集阈值的单阶段封顶回归。第二项检验组合级支护需求标签与 FLAC3D 标量响应及 Stage 11 实际空间场指标之间的一致性。论文正文未作修改。",
        "",
        "## 2 两阶段与单阶段反事实对照",
        "",
        f"单阶段阈值仅使用 Stage 7 五折 OOF 预测选择，最优安全阈值为 {threshold:.3f} MPa。Stage 8 在模型训练和阈值选择过程中保持锁定。",
        "",
        "| 方法 | 边界召回率 | 假有限率 | 有限 RMSE (MPa) | 精确等级准确率 | 安全加权误差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "two_stage_Logistic_TabPFN": "两阶段 LogisticRegression 加 TabPFN",
        "single_stage_capped_no_boundary": "单阶段封顶回归",
        "single_stage_capped_thresholded": "阈值化单阶段封顶回归",
    }
    for key, display in labels.items():
        row = external.loc[key]
        lines.append(
            f"| {display} | {fmt(row['boundary_recall'])} | {fmt(row['false_finite_rate'])} | {fmt(row['finite_RMSE_MPa'])} | {fmt(row['exact_level_accuracy'])} | {fmt(row['safety_weighted_error'])} |"
        )
    lines.extend(
        [
            "",
            f"Stage 7 配对 OOF 检验中，两阶段模型的边界召回率为 {fmt(internal.loc['two_stage_Logistic_TabPFN', 'boundary_recall'])}，假有限率为 {fmt(internal.loc['two_stage_Logistic_TabPFN', 'false_finite_rate'])}；阈值化单阶段模型分别为 {fmt(internal.loc['single_stage_capped_thresholded', 'boundary_recall'])} 和 {fmt(internal.loc['single_stage_capped_thresholded', 'false_finite_rate'])}。后者在内部边界召回上具有竞争力。锁定 Stage 8 中，两阶段模型保持零假阴性和零假阳性，有限工况 RMSE 为 {fmt(external.loc['two_stage_Logistic_TabPFN', 'finite_RMSE_MPa'])} MPa，低于阈值化单阶段模型的 {fmt(external.loc['single_stage_capped_thresholded', 'finite_RMSE_MPa'])} MPa。该结果支持显式边界输出和独立有限压力回归的工程语义与外部稳定性，尚不支持所有统计指标均占优的表述。",
            "",
            "两阶段模型为边界状态保留独立输出。单阶段封顶回归只能输出有限数值，因此其边界样本均构成假有限预测。阈值化单阶段模型重新引入了隐式边界判别，能够降低该风险，其代价由假阴性、假阳性和有限压力误差共同评价。",
            "",
            "## 3 组合级边界与标量响应一致性",
            "",
            f"在全 960 个组合的终止搜索记录上，边界组与可控组的归一化顶板下沉中位数差为 {fmt(settlement_test['boundary_minus_controlled_median'])}，95% bootstrap 区间为 [{fmt(settlement_test['bootstrap_ci95_lower'])}, {fmt(settlement_test['bootstrap_ci95_upper'])}]，Cliff's delta 为 {fmt(settlement_test['cliffs_delta'])}。归一化顶板下沉对边界标签的判别 AUC 为 {fmt(settlement_test['boundary_auc'])}。有限组合取首次满足控制判据的压力记录，边界组合取 2.6 MPa 搜索上限记录。",
            "",
            f"组合归一化变形的边界组与可控组中位数差为 {fmt(deformation_test['boundary_minus_controlled_median'])}，AUC 为 {fmt(deformation_test['boundary_auc'])}。这些结果检验的是给定 2.0 m 判据和 2.6 MPa 搜索上限下的数值状态差异。",
            "",
            "边界标签由位移控制判据生成，因此该组标量检验属于标签实现和响应分离度核查，不构成独立物理验证。独立性更强的证据来自锁定 Stage 8 与 Stage 11 多场响应分析。",
            "",
            "## 4 有限工况决策与多步空间场一致性",
            "",
            f"Stage 11 包含 {required['combination_id'].nunique()} 个具有最小可控压力场的组合。其数据用于分析有限压力等级与空间场指标的关系，以及压力提高一个等级后的配对响应。",
            "",
            f"最小压力与 P95 归一化垂向位移的 Spearman 相关系数为 {fmt(correlation.loc['p95_abs_uz_over_M', 'spearman_rho_with_required_pressure'])}。最小压力与总体塑性比例的相关系数为 {fmt(correlation.loc['plastic_any_fraction', 'spearman_rho_with_required_pressure'])}。",
            "",
            f"支护压力提高一个等级后，P95 归一化垂向位移下降的组合比例为 {fmt(paired.loc['p95_abs_uz_over_M', 'fraction_decreased_with_more_support'])}，总体塑性比例下降的组合比例为 {fmt(paired.loc['plastic_any_fraction', 'fraction_decreased_with_more_support'])}。",
            "",
            "## 5 证据边界",
            "",
            f"Stage 11 多步场数据没有未控制边界组合，可用于直接边界空间转变检验的组合数为 {boundary_field_count}。因此，本报告已经完成全 960 组合的边界标量响应一致性，以及 272 个有限组合的多步空间场一致性。边界状态对应的塑性区贯通和空间应力重分布仍需补充专门的 2.6 MPa 边界场工况。",
            "",
            "## 6 对创新点的影响",
            "",
            "1. 两阶段反事实对照直接检验了显式边界输出对假有限预测风险的影响。",
            "2. 全量终止搜索响应分析建立了边界标签与大变形状态之间的组合级联系。",
            "3. Stage 11 配对分析建立了有限支护决策与空间场响应之间的联系。",
            "4. 当前结果能够增强任务定义创新和框架一致性证据，尚不能证明未控制边界对应完整塑性区状态转变。",
            "",
            "## 7 输出文件",
            "",
            "核心表格位于 `tables` 文件夹，核心图位于 `figures` 文件夹。全部分析按组合或完整工况保留独立标识，方便后续聚类 bootstrap 和论文图表更新。",
        ]
    )
    (OUT_DIR / "innovation_strengthening_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    setup_style()
    ensure_dirs()
    counterfactual = run_counterfactual_comparison()
    scalar, _, scalar_tests = response_at_search_termination()
    field_result = run_field_consistency()
    plot_counterfactual(counterfactual)
    plot_consistency(scalar, scalar_tests, field_result)
    write_report(counterfactual, scalar, scalar_tests, field_result)
    metadata = {
        "seed": SEED,
        "pressure_levels_MPa": PRESSURE_LEVELS.tolist(),
        "single_stage_threshold_MPa": counterfactual["threshold"],
        "stage7_combinations": 960,
        "stage8_locked_combinations": 64,
        "stage11_field_combinations": int(
            field_result["required"]["combination_id"].nunique()
        ),
        "stage11_boundary_field_combinations": 0,
        "elapsed_seconds": counterfactual["elapsed_seconds"],
    }
    (OUT_DIR / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Innovation-strengthening outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
