from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "分析" / "evidence_chain_strengthening" / "tables" / "repeated_stage2_regression_predictions.csv"
FALLBACK = ROOT / "分析" / "advanced_models" / "tables" / "tabpfn_cv_predictions.csv"
OUT_DIR = ROOT / "分析" / "error_stratification"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
SUMMARY_DIR = ROOT / "分析" / "论文图片汇总"

PRESSURE_LEVELS = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6]


def load_predictions() -> pd.DataFrame:
    if PREDICTIONS.exists():
        df = pd.read_csv(PREDICTIONS)
        df = df[df["model"].str.contains("TabPFN", case=False, na=False)].copy()
        df["source_file"] = str(PREDICTIONS.relative_to(ROOT))
    else:
        df = pd.read_csv(FALLBACK)
        df = df[df["model"].str.contains("TabPFN", case=False, na=False)].copy()
        df["source_file"] = str(FALLBACK.relative_to(ROOT))
    df["target_MPa"] = df["target_MPa"].astype(float)
    df["predicted_MPa"] = df["predicted_MPa"].astype(float)
    df["predicted_nearest_level_MPa"] = df["predicted_nearest_level_MPa"].astype(float)
    df["target_MPa"] = df["target_MPa"].map(snap_level)
    df["predicted_nearest_level_MPa"] = df["predicted_nearest_level_MPa"].map(snap_level)
    df["error_MPa"] = df["predicted_MPa"] - df["target_MPa"]
    df["absolute_error_MPa"] = df["error_MPa"].abs()
    df["level_error_steps"] = df.apply(
        lambda row: PRESSURE_LEVELS.index(row["predicted_nearest_level_MPa"])
        - PRESSURE_LEVELS.index(row["target_MPa"]),
        axis=1,
    )
    return df


def rmse(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def snap_level(value: float) -> float:
    return min(PRESSURE_LEVELS, key=lambda level: abs(level - float(value)))


def make_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for level, group in df.groupby("target_MPa", sort=True):
        rows.append(
            {
                "target_level_MPa": level,
                "n_unique_samples": group["sample_id"].nunique(),
                "n_repeated_cv_predictions": len(group),
                "MAE_MPa": group["absolute_error_MPa"].mean(),
                "RMSE_MPa": rmse(group["error_MPa"]),
                "mean_bias_MPa": group["error_MPa"].mean(),
                "median_abs_error_MPa": group["absolute_error_MPa"].median(),
                "P90_abs_error_MPa": group["absolute_error_MPa"].quantile(0.9),
                "nearest_level_accuracy": (group["level_error_steps"] == 0).mean(),
                "adjacent_level_accuracy": (group["level_error_steps"].abs() <= 1).mean(),
                "over_prediction_rate": (group["level_error_steps"] > 0).mean(),
                "under_prediction_rate": (group["level_error_steps"] < 0).mean(),
            }
        )
    level_summary = pd.DataFrame(rows)

    confusion = pd.crosstab(
        df["target_MPa"],
        df["predicted_nearest_level_MPa"],
        rownames=["target_level_MPa"],
        colnames=["predicted_nearest_level_MPa"],
        dropna=False,
    ).reindex(index=PRESSURE_LEVELS, columns=PRESSURE_LEVELS, fill_value=0)

    return level_summary, confusion


def save_plot(df: pd.DataFrame, level_summary: pd.DataFrame, confusion: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.scatter(df["target_MPa"], df["predicted_MPa"], s=22, alpha=0.55, color="#2f80ed", edgecolor="none")
    ax.plot([0.45, 2.65], [0.45, 2.65], color="#333333", lw=1.2, label="1:1 line")
    for offset in (-0.3, 0.3):
        ax.plot([0.45, 2.65], [0.45 + offset, 2.65 + offset], color="#b0b7c3", lw=0.8, ls=":")
    ax.set_xlim(0.42, 2.68)
    ax.set_ylim(0.42, 2.75)
    ax.set_xticks(PRESSURE_LEVELS)
    ax.set_yticks(PRESSURE_LEVELS)
    ax.set_xlabel("True pressure level (MPa)")
    ax.set_ylabel("Predicted pressure (MPa)")
    ax.set_title("(a) True vs predicted support pressure", pad=12)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.18)

    ax = axes[1]
    x = np.arange(len(level_summary))
    width = 0.36
    ax.bar(x - width / 2, level_summary["MAE_MPa"], width, label="MAE", color="#5aa9e6")
    ax.bar(x + width / 2, level_summary["RMSE_MPa"], width, label="RMSE", color="#f28e2b")
    ymax = float(level_summary[["MAE_MPa", "RMSE_MPa"]].max().max())
    for i, row in level_summary.iterrows():
        y = max(row["MAE_MPa"], row["RMSE_MPa"]) + 0.012
        ax.text(i, y, f"n={int(row['n_unique_samples'])}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.1f}" for v in level_summary["target_level_MPa"]])
    ax.set_xlabel("True pressure level (MPa)")
    ax.set_ylabel("Error (MPa)")
    ax.set_ylim(0, ymax * 1.32)
    ax.set_title("(b) Stratified MAE and RMSE", pad=12)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)

    ax = axes[2]
    matrix = confusion.to_numpy()
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(PRESSURE_LEVELS)))
    ax.set_yticks(np.arange(len(PRESSURE_LEVELS)))
    ax.set_xticklabels([f"{v:.1f}" for v in PRESSURE_LEVELS], rotation=45, ha="right")
    ax.set_yticklabels([f"{v:.1f}" for v in PRESSURE_LEVELS])
    ax.set_xlabel("Nearest predicted level (MPa)")
    ax.set_ylabel("True level (MPa)")
    ax.set_title("(c) Nearest-level confusion matrix", pad=12)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            if value:
                ax.text(j, i, str(value), ha="center", va="center", color="#1f2937", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "Fig19_stratified_error_analysis.png"
    fig.savefig(fig_path, dpi=300)
    fig.savefig(FIG_DIR / "Fig19_stratified_error_analysis.pdf")
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SUMMARY_DIR / "Fig19_stratified_error_analysis.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_predictions()
    level_summary, confusion = make_tables(df)
    level_summary.to_csv(TABLE_DIR / "tabpfn_error_by_pressure_level.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(TABLE_DIR / "tabpfn_nearest_level_confusion_matrix.csv", encoding="utf-8-sig")
    df.to_csv(TABLE_DIR / "tabpfn_repeated_cv_predictions_with_errors.csv", index=False, encoding="utf-8-sig")
    save_plot(df, level_summary, confusion)
    print(level_summary.to_string(index=False))
    print()
    print(confusion.to_string())


if __name__ == "__main__":
    main()
