from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tabpfn import TabPFNRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "工况" / "flac3d_ml_dataset"
OUT_DIR = ROOT / "分析" / "advanced_models"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

FEATURES = [
    "H_m",
    "lambda",
    "M_m",
    "alpha_E_roof",
    "alpha_c_roof",
    "phi_roof_deg",
    "alpha_t_roof",
]
PRESSURE_LEVELS = np.array([0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6])


def setup_style() -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    labels = pd.read_csv(DATA_DIR / "labels_required_support_pressure_regression.csv")
    for col in FEATURES + ["required_support_pressure_MPa"]:
        labels[col] = pd.to_numeric(labels[col], errors="raise")
    x = labels[FEATURES].to_numpy(dtype=np.float32)
    y = labels["required_support_pressure_MPa"].to_numpy(dtype=np.float32)
    return labels, x, y


def nearest_level(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    idx = np.abs(values.reshape(-1, 1) - PRESSURE_LEVELS.reshape(1, -1)).argmin(axis=1)
    return PRESSURE_LEVELS[idx]


def model_factories() -> dict[str, callable]:
    return {
        "SVR_RBF": lambda seed: make_pipeline(StandardScaler(), SVR(C=4.0, epsilon=0.08, gamma="scale")),
        "RandomForest": lambda seed: RandomForestRegressor(
            n_estimators=800,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        ),
        "CatBoost": lambda seed: CatBoostRegressor(
            iterations=700,
            depth=3,
            learning_rate=0.035,
            l2_leaf_reg=6.0,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        ),
        "TabPFN": lambda seed: TabPFNRegressor(
            n_estimators=8,
            device="auto",
            random_state=seed,
            show_progress_bar=False,
        ),
        "MLP": lambda seed: make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                alpha=0.02,
                learning_rate_init=0.003,
                max_iter=2500,
                early_stopping=True,
                validation_fraction=0.18,
                random_state=seed,
            ),
        ),
    }


def evaluate_model(name: str, factory: callable, labels: pd.DataFrame, x: np.ndarray, y: np.ndarray) -> tuple[dict, pd.DataFrame]:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros_like(y, dtype=float)
    started = time.time()
    fold_errors = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(x), start=1):
        try:
            model = factory(42 + fold)
            model.fit(x[train_idx], y[train_idx])
            fold_pred = model.predict(x[test_idx])
            preds[test_idx] = np.asarray(fold_pred, dtype=float).reshape(-1)
        except Exception as exc:
            fold_errors.append({"fold": fold, "error": repr(exc), "traceback": traceback.format_exc()})
            raise

    elapsed = time.time() - started
    preds = np.clip(preds, PRESSURE_LEVELS.min(), PRESSURE_LEVELS.max())
    snapped = nearest_level(preds)

    row = {
        "model": name,
        "R2": r2_score(y, preds),
        "RMSE_MPa": math.sqrt(mean_squared_error(y, preds)),
        "MAE_MPa": mean_absolute_error(y, preds),
        "MAPE_percent": mean_absolute_percentage_error(y, preds) * 100.0,
        "nearest_level_accuracy": float(np.mean(np.isclose(snapped, y))),
        "nearest_level_MAE_MPa": mean_absolute_error(y, snapped),
        "elapsed_seconds": elapsed,
    }

    pred_frame = labels[["sample_id", "source", "geo_id"] + FEATURES].copy()
    pred_frame["target_MPa"] = y
    pred_frame["model"] = name
    pred_frame["predicted_MPa"] = preds
    pred_frame["predicted_nearest_level_MPa"] = snapped
    pred_frame["absolute_error_MPa"] = np.abs(preds - y)
    return row, pred_frame


def plot_results(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    best_model = metrics.iloc[0]["model"]
    best = predictions[predictions["model"] == best_model]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=metrics, x="RMSE_MPa", y="model", color="#56B4E9", ax=axes[0])
    axes[0].set_xlabel("5-fold CV RMSE (MPa)")
    axes[0].set_ylabel("")
    axes[0].set_title("Advanced model comparison")

    axes[1].scatter(
        best["target_MPa"],
        best["predicted_MPa"],
        s=36,
        alpha=0.82,
        color="#0072B2",
        edgecolor="white",
        linewidth=0.4,
    )
    axes[1].plot([0.45, 2.65], [0.45, 2.65], linestyle=":", color="black", linewidth=1.2)
    axes[1].set_xlim(0.45, 2.65)
    axes[1].set_ylim(0.45, 2.65)
    axes[1].set_xlabel("FLAC3D-derived label (MPa)")
    axes[1].set_ylabel("Predicted pressure (MPa)")
    axes[1].set_title(f"Best model: {best_model}")
    axes[1].grid(True, alpha=0.18, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "advanced_model_comparison.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "advanced_model_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sns.boxplot(data=predictions, x="model", y="absolute_error_MPa", color="#009E73", ax=ax)
    sns.stripplot(data=predictions, x="model", y="absolute_error_MPa", color="black", alpha=0.28, size=2.5, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Absolute error (MPa)")
    ax.set_title("Cross-validated absolute error distribution")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "advanced_model_error_distribution.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "advanced_model_error_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    labels, x, y = load_data()
    rows = []
    pred_frames = []
    errors = []

    for name, factory in model_factories().items():
        print(f"Running {name} ...", flush=True)
        try:
            row, pred_frame = evaluate_model(name, factory, labels, x, y)
            rows.append(row)
            pred_frames.append(pred_frame)
            print(f"  done: RMSE={row['RMSE_MPa']:.4f}, R2={row['R2']:.4f}", flush=True)
        except Exception as exc:
            error = {"model": name, "error": repr(exc), "traceback": traceback.format_exc()}
            errors.append(error)
            print(f"  failed: {exc!r}", flush=True)

    metrics = pd.DataFrame(rows).sort_values(["RMSE_MPa", "MAE_MPa"]).reset_index(drop=True)
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()

    metrics.to_csv(TABLE_DIR / "advanced_model_metrics_cv.csv", index=False, encoding="utf-8")
    predictions.to_csv(TABLE_DIR / "advanced_model_cv_predictions.csv", index=False, encoding="utf-8")
    (OUT_DIR / "advanced_model_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

    if not metrics.empty:
        plot_results(metrics, predictions)
        lines = [
            "# Advanced Tabular Model Comparison",
            "",
            "Models: SVR, RandomForest, CatBoost, TabPFN, and MLP.",
            "",
            "## Metrics",
            "",
            metrics.to_markdown(index=False),
            "",
        ]
        if errors:
            lines.extend(["## Failed Models", "", "```json", json.dumps(errors, ensure_ascii=False, indent=2), "```"])
        (OUT_DIR / "先进模型对比总结.md").write_text("\n".join(lines), encoding="utf-8")

    print(metrics.to_string(index=False))
    if errors:
        print("Errors were written to advanced_model_errors.json")


if __name__ == "__main__":
    main()
