from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)


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


def find_project_root() -> Path:
    root = Path.cwd()
    if (root / "脚本").exists() and any(root.rglob("flac3d_ml_dataset")):
        return root
    for parent in [root, *root.parents]:
        if (parent / "脚本").exists() and any(parent.rglob("flac3d_ml_dataset")):
            return parent
    raise FileNotFoundError("Could not locate project root with flac3d_ml_dataset.")


def try_tabpfn_regressor():
    try:
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(ignore_pretraining_limits=True), "TabPFNRegressor"
    except Exception:
        return (
            RandomForestRegressor(
                n_estimators=600,
                min_samples_leaf=2,
                random_state=42,
            ),
            "RandomForestRegressor_fallback",
        )


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def classification_learning_curve(labels_all: pd.DataFrame, rng_seed: int = 42) -> pd.DataFrame:
    df = labels_all.copy()
    df["is_boundary"] = df["label_status"].astype(str).str.contains("exceedance").astype(int)
    X = df[FEATURES].astype(float)
    y = df["is_boundary"].astype(int)

    # Keep sizes large enough to contain boundary samples after stratified sampling.
    sizes = [40, 60, 80, 100, 120, 140, 150]
    rows = []
    estimator = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
    )

    for n in sizes:
        splitter = StratifiedShuffleSplit(n_splits=15, train_size=n, random_state=rng_seed + n)
        for repeat, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
            if len(np.unique(y.iloc[train_idx])) < 2 or len(np.unique(y.iloc[test_idx])) < 2:
                continue
            model = clone(estimator)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            prob = model.predict_proba(X.iloc[test_idx])[:, 1]
            rows.append(
                {
                    "task": "boundary_classification",
                    "model": "LogisticRegression",
                    "train_size": n,
                    "repeat": repeat,
                    "boundary_recall": recall_score(y.iloc[test_idx], pred, zero_division=0),
                    "roc_auc": roc_auc_score(y.iloc[test_idx], prob),
                    "test_boundary_count": int(y.iloc[test_idx].sum()),
                    "test_size": len(test_idx),
                }
            )
    return pd.DataFrame(rows)


def regression_learning_curve(labels_reg: pd.DataFrame, rng_seed: int = 42) -> tuple[pd.DataFrame, str]:
    X = labels_reg[FEATURES].astype(float)
    y = labels_reg["required_support_pressure_MPa"].astype(float)
    base_estimator, model_name = try_tabpfn_regressor()

    sizes = [30, 50, 70, 90, 110, 130]
    rows = []
    for n in sizes:
        splitter = ShuffleSplit(n_splits=8, train_size=n, random_state=rng_seed + n)
        for repeat, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
            model = clone(base_estimator)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = np.asarray(model.predict(X.iloc[test_idx]), dtype=float)
            rows.append(
                {
                    "task": "support_pressure_regression",
                    "model": model_name,
                    "train_size": n,
                    "repeat": repeat,
                    "r2": r2_score(y.iloc[test_idx], pred),
                    "rmse_MPa": rmse(y.iloc[test_idx], pred),
                    "mae_MPa": float(mean_absolute_error(y.iloc[test_idx], pred)),
                    "test_size": len(test_idx),
                }
            )
    return pd.DataFrame(rows), model_name


def summarize_curves(cls_detail: pd.DataFrame, reg_detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cls_summary = (
        cls_detail.groupby(["task", "model", "train_size"], as_index=False)
        .agg(
            boundary_recall_mean=("boundary_recall", "mean"),
            boundary_recall_sd=("boundary_recall", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sd=("roc_auc", "std"),
            mean_test_boundary_count=("test_boundary_count", "mean"),
        )
        .sort_values("train_size")
    )
    reg_summary = (
        reg_detail.groupby(["task", "model", "train_size"], as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            rmse_mean_MPa=("rmse_MPa", "mean"),
            rmse_sd_MPa=("rmse_MPa", "std"),
            mae_mean_MPa=("mae_MPa", "mean"),
            mae_sd_MPa=("mae_MPa", "std"),
        )
        .sort_values("train_size")
    )
    return cls_summary, reg_summary


def plot_learning_curve(cls_summary: pd.DataFrame, reg_summary: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "figure.dpi": 300,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4), constrained_layout=True)

    ax = axes[0]
    x = reg_summary["train_size"]
    ax.errorbar(
        x,
        reg_summary["r2_mean"],
        yerr=reg_summary["r2_sd"],
        marker="o",
        linewidth=1.4,
        capsize=2.5,
        color="#2F6B9A",
    )
    ax.set_xlabel("Regression training samples")
    ax.set_ylabel(r"$R^2$")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.22)
    ax.set_title("Regression accuracy")

    ax = axes[1]
    ax.errorbar(
        x,
        reg_summary["rmse_mean_MPa"],
        yerr=reg_summary["rmse_sd_MPa"],
        marker="o",
        linewidth=1.4,
        capsize=2.5,
        color="#B14E2D",
    )
    ax.axhline(0.3, color="0.25", linestyle="--", linewidth=0.9)
    ax.text(x.min(), 0.305, "pressure-level interval", va="bottom", fontsize=8)
    ax.set_xlabel("Regression training samples")
    ax.set_ylabel("RMSE (MPa)")
    ax.grid(alpha=0.22)
    ax.set_title("Regression error")

    ax = axes[2]
    x_cls = cls_summary["train_size"]
    ax.errorbar(
        x_cls,
        cls_summary["boundary_recall_mean"],
        yerr=cls_summary["boundary_recall_sd"],
        marker="o",
        linewidth=1.4,
        capsize=2.5,
        color="#2E8B57",
    )
    ax.set_xlabel("Classification training samples")
    ax.set_ylabel("Boundary recall")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.22)
    ax.set_title("Boundary screening")

    fig.suptitle("Learning curves for FLAC3D support-pressure surrogate models", y=1.05, fontsize=11)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def write_report(out_dir: Path, cls_summary: pd.DataFrame, reg_summary: pd.DataFrame, reg_model_name: str) -> None:
    last_reg = reg_summary.iloc[-1]
    last_cls = cls_summary.iloc[-1]
    report = f"""# Learning-curve evidence for the FLAC3D surrogate dataset

## Purpose

This analysis tests whether the current FLAC3D support-pressure dataset is large enough to support the two-stage surrogate model. It follows the learning-curve logic used in numerical-model-based surrogate studies: model quality is evaluated as the number of training samples increases.

## Data and models

- Boundary classification data: 160 geological/mining-height combinations, including 21 `>2.6 MPa` boundary samples.
- Regression data: 139 controllable samples with finite required support-pressure labels.
- Stage 1 model: LogisticRegression with class-balanced fitting.
- Stage 2 model: {reg_model_name}.

## Main observations

- At the full regression sample size of {int(last_reg['train_size'])}, the learning curve gives mean R² = {last_reg['r2_mean']:.3f} ± {last_reg['r2_sd']:.3f}, RMSE = {last_reg['rmse_mean_MPa']:.3f} ± {last_reg['rmse_sd_MPa']:.3f} MPa, and MAE = {last_reg['mae_mean_MPa']:.3f} ± {last_reg['mae_sd_MPa']:.3f} MPa.
- At the full classification sample size of {int(last_cls['train_size'])}, the boundary-screening recall is {last_cls['boundary_recall_mean']:.3f} ± {last_cls['boundary_recall_sd']:.3f}, with ROC-AUC = {last_cls['roc_auc_mean']:.3f} ± {last_cls['roc_auc_sd']:.3f}.
- The regression RMSE should be interpreted relative to the 0.3 MPa support-pressure level interval. If the full-sample RMSE remains below this interval, the surrogate is suitable for screening support-pressure levels rather than replacing detailed FLAC3D calculations.
- The boundary recall curve is more important than overall accuracy for the first stage because missing a `>2.6 MPa` case is more severe than conservatively flagging a controllable case for further FLAC3D review.

## Manuscript use

Use the generated figure as a main-text or supplementary learning-curve figure. It directly addresses whether the current 139 controllable labels and 21 boundary labels are sufficient for the proposed screening task.
"""
    (out_dir / "learning_curve_evidence_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    root = find_project_root()
    data_dir = next(p for p in root.rglob("flac3d_ml_dataset") if p.is_dir())
    analysis_dir = root / "分析" / "learning_curve_evidence"
    figures_dir = analysis_dir / "figures"
    tables_dir = analysis_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    labels_all = pd.read_csv(data_dir / "labels_required_support_pressure_all.csv")
    labels_reg = pd.read_csv(data_dir / "labels_required_support_pressure_regression.csv")

    cls_detail = classification_learning_curve(labels_all)
    reg_detail, reg_model_name = regression_learning_curve(labels_reg)
    cls_summary, reg_summary = summarize_curves(cls_detail, reg_detail)

    cls_detail.to_csv(tables_dir / "learning_curve_stage1_classification_detail.csv", index=False, encoding="utf-8-sig")
    reg_detail.to_csv(tables_dir / "learning_curve_stage2_regression_detail.csv", index=False, encoding="utf-8-sig")
    cls_summary.to_csv(tables_dir / "learning_curve_stage1_classification_summary.csv", index=False, encoding="utf-8-sig")
    reg_summary.to_csv(tables_dir / "learning_curve_stage2_regression_summary.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "features": FEATURES,
        "classification_model": "LogisticRegression",
        "regression_model": reg_model_name,
        "classification_repeats_per_size": 15,
        "regression_repeats_per_size": 8,
    }
    (analysis_dir / "learning_curve_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    out_png = figures_dir / "Fig17_learning_curve_surrogate_dataset.png"
    out_pdf = figures_dir / "Fig17_learning_curve_surrogate_dataset.pdf"
    plot_learning_curve(cls_summary, reg_summary, out_png, out_pdf)

    manuscript_fig_dir = root / "分析" / "论文图片汇总"
    if manuscript_fig_dir.exists():
        import shutil

        shutil.copy2(out_png, manuscript_fig_dir / out_png.name)

    write_report(analysis_dir, cls_summary, reg_summary, reg_model_name)
    print(f"Saved learning-curve evidence to: {analysis_dir}")
    print(f"Regression model used: {reg_model_name}")
    print(reg_summary.to_string(index=False))
    print(cls_summary.to_string(index=False))


if __name__ == "__main__":
    main()
