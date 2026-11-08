from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "工况" / "flac3d_ml_dataset"
OUT_DIR = ROOT / "分析" / "two_stage_exceedance"
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


def setup_style() -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    labels = pd.read_csv(DATA_DIR / "labels_required_support_pressure_all.csv")
    for col in FEATURES:
        labels[col] = pd.to_numeric(labels[col], errors="raise")
    labels["is_exceedance"] = (labels["label_status"] != "labeled").astype(int)
    x = labels[FEATURES].to_numpy(dtype=np.float32)
    y = labels["is_exceedance"].to_numpy(dtype=int)
    return labels, x, y


def model_factories() -> dict[str, callable]:
    return {
        "LogisticRegression": lambda seed: make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
        ),
        "RandomForest": lambda seed: RandomForestClassifier(
            n_estimators=800,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "CatBoost": lambda seed: CatBoostClassifier(
            iterations=700,
            depth=3,
            learning_rate=0.035,
            l2_leaf_reg=6.0,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        ),
        "TabPFN": lambda seed: TabPFNClassifier(
            n_estimators=8,
            device="auto",
            random_state=seed,
            show_progress_bar=False,
        ),
    }


def predict_score(model, x_test: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
        return np.asarray(proba)[:, 1]
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(x_test), dtype=float)
        return 1.0 / (1.0 + np.exp(-score))
    return np.asarray(model.predict(x_test), dtype=float)


def evaluate_model(name: str, factory: callable, labels: pd.DataFrame, x: np.ndarray, y: np.ndarray) -> tuple[dict, pd.DataFrame, np.ndarray]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = np.zeros_like(y, dtype=int)
    score = np.zeros_like(y, dtype=float)
    started = time.time()

    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        model = factory(342 + fold)
        model.fit(x[train_idx], y[train_idx])
        score[test_idx] = predict_score(model, x[test_idx])
        pred[test_idx] = (score[test_idx] >= 0.5).astype(int)

    row = {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "exceedance_precision": precision_score(y, pred, zero_division=0),
        "exceedance_recall": recall_score(y, pred, zero_division=0),
        "exceedance_f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "elapsed_seconds": time.time() - started,
    }
    frame = labels[["sample_id", "source", "geo_id", "label_status"] + FEATURES].copy()
    frame["target_exceedance"] = y
    frame["predicted_exceedance"] = pred
    frame["exceedance_score"] = score
    frame["model"] = name
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return row, frame, cm


def plot_results(metrics: pd.DataFrame, predictions: pd.DataFrame, best_name: str, cm: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=metrics, x="exceedance_recall", y="model", color="#D55E00", ax=axes[0])
    axes[0].set_xlabel("Exceedance recall")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylabel("")
    axes[0].set_title("Boundary/exceedance detection")

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        square=True,
        cbar=False,
        xticklabels=["Controllable", ">2.6 MPa"],
        yticklabels=["Controllable", ">2.6 MPa"],
        ax=axes[1],
    )
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("FLAC3D-derived label")
    axes[1].set_title(f"Best model: {best_name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exceedance_classifier_comparison.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "exceedance_classifier_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    labels, x, y = load_data()
    rows = []
    pred_frames = []
    cms = {}
    for name, factory in model_factories().items():
        print(f"Running {name} exceedance classifier ...", flush=True)
        row, pred_frame, cm = evaluate_model(name, factory, labels, x, y)
        rows.append(row)
        pred_frames.append(pred_frame)
        cms[name] = cm
        print(
            f"  done: recall={row['exceedance_recall']:.3f}, precision={row['exceedance_precision']:.3f}, pr_auc={row['pr_auc']:.3f}",
            flush=True,
        )

    metrics = pd.DataFrame(rows).sort_values(["exceedance_recall", "pr_auc", "exceedance_precision"], ascending=False).reset_index(drop=True)
    predictions = pd.concat(pred_frames, ignore_index=True)
    best_name = str(metrics.iloc[0]["model"])

    metrics.to_csv(TABLE_DIR / "exceedance_classifier_metrics_cv.csv", index=False, encoding="utf-8")
    predictions.to_csv(TABLE_DIR / "exceedance_classifier_cv_predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame(cms[best_name], index=["controllable", "exceedance"], columns=["pred_controllable", "pred_exceedance"]).to_csv(
        TABLE_DIR / "best_exceedance_confusion_matrix.csv",
        encoding="utf-8",
    )
    plot_results(metrics, predictions, best_name, cms[best_name])

    lines = [
        "# Two-stage Model: Exceedance Classifier",
        "",
        f"Total geological combinations: {len(labels)}",
        f"Exceedance samples: {int(y.sum())}",
        f"Controllable samples: {int((1-y).sum())}",
        "",
        "## Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        f"Best model by recall/PR-AUC priority: **{best_name}**.",
    ]
    (OUT_DIR / "超限识别模型总结.md").write_text("\n".join(lines), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
