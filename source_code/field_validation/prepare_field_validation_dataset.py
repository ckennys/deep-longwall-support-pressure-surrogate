from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
THESIS_ROOT = SCRIPT_PATH.parents[2]

OUT_DIR = PROJECT_ROOT / "分析" / "field_validation"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

SUPPORT_STATS_FILE = THESIS_ROOT / "shujugoogle" / "final_evaluation_figures" / "real_12_support_statistics.csv"
PERIODIC_EVENTS_FILE = THESIS_ROOT / "shujugoogle" / "final_evaluation_figures" / "real_periodic_weighting_events.csv"
DAILY_MERGED_FILE = THESIS_ROOT / "shujuchul" / "support_stress_daily_merged.csv"
ML_LABELS_FILE = PROJECT_ROOT / "工况" / "flac3d_ml_dataset" / "labels_required_support_pressure_all.csv"


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


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk"]
    last_error: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            last_error = exc
    raise RuntimeError(f"Failed to read {path}: {last_error}")


def load_support_statistics() -> pd.DataFrame:
    df = read_csv_auto(SUPPORT_STATS_FILE)
    numeric_cols = ["Mean", "Std", "Max", "P95", "WarningHours_20_25", "DangerHours_gt25", "NonzeroHours"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["UsedInFigure"] = df["UsedInFigure"].astype(str).str.lower().eq("true")
    df["support_id"] = df["Support"].astype(str).str.extract(r"(\d+)").astype(float)
    df["monitoring_group"] = np.where(df["NonzeroHours"] >= 6000, "long_duration", "short_or_edge")
    df["danger_hour_ratio"] = df["DangerHours_gt25"] / df["NonzeroHours"].replace(0, np.nan)
    df["warning_or_danger_hours"] = df["WarningHours_20_25"].fillna(0) + df["DangerHours_gt25"].fillna(0)
    df["high_load_hour_ratio"] = df["warning_or_danger_hours"] / df["NonzeroHours"].replace(0, np.nan)
    return df


def load_daily_response() -> pd.DataFrame:
    df = read_csv_auto(DAILY_MERGED_FILE)
    df["date"] = pd.to_datetime(df["日期"], errors="coerce")
    support_cols = [col for col in df.columns if col.startswith("support_") or col.startswith("belt_support_")]
    stress_cols = [col for col in df.columns if col.startswith("rail_stress_") or col.startswith("belt_stress_")]
    for col in support_cols + stress_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    out = pd.DataFrame({"date": df["date"]})
    out["support_mean"] = df[support_cols].mean(axis=1, skipna=True)
    out["support_p95"] = df[support_cols].quantile(0.95, axis=1, interpolation="linear")
    out["support_max"] = df[support_cols].max(axis=1, skipna=True)
    out["active_support_count"] = df[support_cols].notna().sum(axis=1)
    out["stress_mean"] = df[stress_cols].mean(axis=1, skipna=True)
    out["stress_max"] = df[stress_cols].max(axis=1, skipna=True)
    out["active_stress_count"] = df[stress_cols].notna().sum(axis=1)
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def load_periodic_events() -> pd.DataFrame:
    df = read_csv_auto(PERIODIC_EVENTS_FILE)
    for col in ["SmoothedMeanSupport", "CycleIntervalDays", "Prominence"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["PeakTime"] = pd.to_datetime(df["PeakTime"], errors="coerce")
    return df


def load_ml_label_summary() -> dict:
    labels = read_csv_auto(ML_LABELS_FILE)
    total = int(len(labels))
    boundary = labels["label_status"].astype(str).ne("labeled").sum() if "label_status" in labels.columns else 0
    labeled = pd.to_numeric(labels["required_support_pressure_MPa"], errors="coerce")
    return {
        "flac3d_label_count": total,
        "flac3d_labeled_count": int(labeled.notna().sum()),
        "flac3d_boundary_count": int(boundary),
        "flac3d_pressure_min_MPa": float(labeled.min()),
        "flac3d_pressure_median_MPa": float(labeled.median()),
        "flac3d_pressure_p95_MPa": float(labeled.quantile(0.95)),
        "flac3d_pressure_max_MPa": float(labeled.max()),
    }


def build_summary_tables(
    support_stats: pd.DataFrame,
    daily: pd.DataFrame,
    events: pd.DataFrame,
    ml_summary: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    used = support_stats[support_stats["UsedInFigure"] & support_stats["Mean"].notna()].copy()
    long_duration = used[used["monitoring_group"] == "long_duration"].copy()

    summary_rows = [
        {
            "validation_item": "field_support_sensor_level_all_used",
            "sample_scope": "used supports with valid statistics",
            "n": len(used),
            "mean": used["Mean"].mean(),
            "p95_mean": used["P95"].mean(),
            "max": used["Max"].max(),
            "note": "Field support sensor statistics; unit follows source data.",
        },
        {
            "validation_item": "field_support_sensor_level_long_duration",
            "sample_scope": "supports with nonzero monitoring hours >= 6000",
            "n": len(long_duration),
            "mean": long_duration["Mean"].mean(),
            "p95_mean": long_duration["P95"].mean(),
            "max": long_duration["Max"].max(),
            "note": "More stable subset for magnitude validation.",
        },
        {
            "validation_item": "daily_support_response",
            "sample_scope": "daily merged support-stress data",
            "n": daily["support_mean"].notna().sum(),
            "mean": daily["support_mean"].mean(),
            "p95_mean": daily["support_p95"].mean(),
            "max": daily["support_max"].max(),
            "note": "Daily average/p95/max response across available supports.",
        },
        {
            "validation_item": "periodic_weighting_events",
            "sample_scope": "identified field pressure events",
            "n": len(events),
            "mean": events["SmoothedMeanSupport"].mean(),
            "p95_mean": events["SmoothedMeanSupport"].quantile(0.95),
            "max": events["SmoothedMeanSupport"].max(),
            "note": "Event-level smoothed mean support level and cycle intervals.",
        },
    ]
    summary = pd.DataFrame(summary_rows)

    flac3d_rows = [
        {"metric": key, "value": value}
        for key, value in ml_summary.items()
    ]
    flac3d_rows.extend(
        [
            {
                "metric": "field_to_flac3d_mapping_status",
                "value": "external_validation_ready_no_direct_unit_conversion",
            },
            {
                "metric": "required_conversion",
                "value": "hydraulic pressure/resistance -> equivalent support pressure via support parameters and canopy area",
            },
        ]
    )
    mapping = pd.DataFrame(flac3d_rows)
    return summary, mapping


def build_validation_case_template(summary: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    long_row = summary[summary["validation_item"] == "field_support_sensor_level_long_duration"].iloc[0]
    event_interval = events["CycleIntervalDays"].dropna().mean() if not events.empty else np.nan
    return pd.DataFrame(
        [
            {
                "validation_case_id": "field_7306_v1",
                "field_data_scope": "7306 long-duration supports + periodic weighting events",
                "H_m": np.nan,
                "lambda": np.nan,
                "M_m": np.nan,
                "roof_parameter_source": "to_be_filled_from_geology_or_baseline_model",
                "model_required_support_pressure_MPa": np.nan,
                "model_exceedance_gt_2p6": np.nan,
                "field_support_sensor_mean": long_row["mean"],
                "field_support_sensor_p95": long_row["p95_mean"],
                "field_support_sensor_max": long_row["max"],
                "field_periodic_event_count": len(events),
                "field_mean_cycle_interval_days": event_interval,
                "support_column_area_m2": np.nan,
                "support_column_count": np.nan,
                "support_control_area_m2": np.nan,
                "support_efficiency": np.nan,
                "field_equivalent_support_pressure_MPa": np.nan,
                "validation_status": "ready_for_mapping_after_support_geometry_is_filled",
            }
        ]
    )


def plot_support_statistics(support_stats: pd.DataFrame) -> None:
    data = support_stats[support_stats["UsedInFigure"] & support_stats["Mean"].notna()].sort_values("support_id")
    labels = ["S" + str(int(x)) if pd.notna(x) else str(s) for x, s in zip(data["support_id"], data["Support"])]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(data))
    ax.bar(x - 0.22, data["Mean"], width=0.22, label="Mean", color="#56B4E9")
    ax.bar(x, data["P95"], width=0.22, label="P95", color="#009E73")
    ax.bar(x + 0.22, data["Max"], width=0.22, label="Max", color="#D55E00")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Field support sensor value")
    ax.set_title("Field support magnitude statistics")
    ax.legend(frameon=False, ncols=3)
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    save_fig(fig, "fig01_field_support_statistics")


def plot_daily_response(daily: pd.DataFrame, events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.plot(daily["date"], daily["support_mean"], color="#0072B2", linewidth=1.4, label="Daily mean support")
    ax.plot(daily["date"], daily["support_p95"], color="#009E73", linewidth=1.2, label="Daily P95 support")
    ax.fill_between(
        daily["date"],
        daily["support_mean"],
        daily["support_p95"],
        color="#009E73",
        alpha=0.14,
        linewidth=0,
    )
    for _, row in events.dropna(subset=["PeakTime"]).iterrows():
        ax.axvline(row["PeakTime"], color="#D55E00", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_ylabel("Field support sensor value")
    ax.set_xlabel("")
    ax.set_title("Daily support response and periodic weighting events")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, alpha=0.18, linewidth=0.5)
    save_fig(fig, "fig02_daily_support_with_events")


def plot_support_stress_relation(daily: pd.DataFrame) -> None:
    data = daily.dropna(subset=["support_mean", "stress_mean"]).copy()
    if data.empty:
        return
    corr = data[["support_mean", "stress_mean"]].corr(method="spearman").iloc[0, 1]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sc = ax.scatter(
        data["stress_mean"],
        data["support_mean"],
        c=data["support_p95"],
        cmap="viridis",
        s=36,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_xlabel("Daily mean roadway stress sensor value")
    ax.set_ylabel("Daily mean support sensor value")
    ax.set_title(f"Support-stress relation (Spearman r={corr:.2f})")
    ax.grid(True, alpha=0.18, linewidth=0.5)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Daily P95 support")
    save_fig(fig, "fig03_support_stress_relation")


def write_summary_markdown(summary: pd.DataFrame, mapping: pd.DataFrame, events: pd.DataFrame) -> None:
    long_row = summary[summary["validation_item"] == "field_support_sensor_level_long_duration"].iloc[0]
    all_row = summary[summary["validation_item"] == "field_support_sensor_level_all_used"].iloc[0]
    lines = [
        "# Field validation dataset v1",
        "",
        "## Data sources",
        "",
        f"- Support statistics: `{SUPPORT_STATS_FILE}`",
        f"- Periodic weighting events: `{PERIODIC_EVENTS_FILE}`",
        f"- Daily support-stress merged data: `{DAILY_MERGED_FILE}`",
        "",
        "## Key extracted field metrics",
        "",
        f"- Valid supports used: {int(all_row['n'])}",
        f"- Long-duration supports: {int(long_row['n'])}",
        f"- Long-duration mean support sensor value: {long_row['mean']:.3f}",
        f"- Long-duration mean P95 support sensor value: {long_row['p95_mean']:.3f}",
        f"- Maximum observed support sensor value: {all_row['max']:.3f}",
        f"- Periodic weighting events: {len(events)}",
    ]
    if len(events) > 0:
        intervals = events["CycleIntervalDays"].dropna()
        if not intervals.empty:
            lines.append(f"- Mean cycle interval: {intervals.mean():.2f} days")

    lines.extend(
        [
            "",
            "## Validation use",
            "",
            "1. Use support statistics for magnitude validation of the FLAC3D-support-pressure surrogate.",
            "2. Use periodic weighting events to check whether high-risk stages correspond to field pressure peaks.",
            "3. Use daily support-stress response to support the physical interpretation of stress-driven support demand.",
            "",
            "## Important note",
            "",
            "The field support sensor values and FLAC3D equivalent support pressure are not directly interchangeable unless support geometry, column area, canopy/control area, and efficiency factors are specified. This v1 dataset is therefore prepared for external validation and mapping, not for direct supervised training.",
            "",
            "Required conversion relation:",
            "",
            "`F = p_hydraulic * A_column * n_columns`, and `p_equiv = F / A_control`.",
            "",
        ]
    )
    lines.append("## FLAC3D label summary")
    lines.append("")
    for _, row in mapping.iterrows():
        lines.append(f"- {row['metric']}: {row['value']}")
    (OUT_DIR / "field_validation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    support_stats = load_support_statistics()
    daily = load_daily_response()
    events = load_periodic_events()
    ml_summary = load_ml_label_summary()

    summary, mapping = build_summary_tables(support_stats, daily, events, ml_summary)
    validation_template = build_validation_case_template(summary, events)

    support_stats.to_csv(TABLE_DIR / "field_support_statistics_clean.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(TABLE_DIR / "field_daily_support_stress_response.csv", index=False, encoding="utf-8-sig")
    events.to_csv(TABLE_DIR / "field_periodic_weighting_events.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE_DIR / "field_validation_metric_summary.csv", index=False, encoding="utf-8-sig")
    mapping.to_csv(TABLE_DIR / "field_to_flac3d_mapping_notes.csv", index=False, encoding="utf-8-sig")
    validation_template.to_csv(TABLE_DIR / "field_validation_case_template.csv", index=False, encoding="utf-8-sig")

    plot_support_statistics(support_stats)
    plot_daily_response(daily, events)
    plot_support_stress_relation(daily)
    write_summary_markdown(summary, mapping, events)

    metadata = {
        "support_stats_file": str(SUPPORT_STATS_FILE),
        "periodic_events_file": str(PERIODIC_EVENTS_FILE),
        "daily_merged_file": str(DAILY_MERGED_FILE),
        "ml_labels_file": str(ML_LABELS_FILE),
        "outputs": str(OUT_DIR),
    }
    (OUT_DIR / "field_validation_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"\nSaved field validation outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
