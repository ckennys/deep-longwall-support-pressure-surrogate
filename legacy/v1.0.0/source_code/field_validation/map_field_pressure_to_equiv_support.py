from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = PROJECT_ROOT / "分析" / "field_validation"
FIELD_PARAM_DIR = PROJECT_ROOT / "分析" / "field_support_parameters"
OUT_DIR = PROJECT_ROOT / "分析" / "field_pressure_equivalent_mapping"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
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


def make_assumptions() -> pd.DataFrame:
    rows = []
    for model, rated_resistance_kn in [
        ("ZY8000/23/47D", 8000.0),
        ("ZY9000/15/32D", 9000.0),
    ]:
        for area_name, control_area_m2 in [
            ("small_area_high_equiv_pressure", 5.0),
            ("base_area", 6.0),
            ("large_area_low_equiv_pressure", 7.0),
        ]:
            rows.append(
                {
                    "support_model": model,
                    "rated_working_resistance_kN": rated_resistance_kn,
                    "rated_hydraulic_pressure_MPa": 40.0,
                    "equivalent_control_area_m2": control_area_m2,
                    "area_scenario": area_name,
                    "conversion_factor_eqMPa_per_fieldMPa": (rated_resistance_kn / 1000.0)
                    / (40.0 * control_area_m2),
                    "basis": (
                        "rated resistance inferred from model code; rated hydraulic pressure "
                        "temporarily set to 40 MPa from field peak/no-overlimit evidence; "
                        "control area is a sensitivity assumption until support manual gives "
                        "center distance and roof-control length."
                    ),
                }
            )
    return pd.DataFrame(rows)


def convert_value(value, factor):
    value = pd.to_numeric(value, errors="coerce")
    return value * factor


def convert_support_statistics(stats: pd.DataFrame, assumptions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, s in stats.iterrows():
        for _, a in assumptions.iterrows():
            factor = a["conversion_factor_eqMPa_per_fieldMPa"]
            rows.append(
                {
                    "Support": s.get("Support"),
                    "support_id": s.get("support_id"),
                    "monitoring_group": s.get("monitoring_group"),
                    "support_model_scenario": a["support_model"],
                    "area_scenario": a["area_scenario"],
                    "field_mean_MPa": s.get("Mean"),
                    "field_p95_MPa": s.get("P95"),
                    "field_max_MPa": s.get("Max"),
                    "equiv_mean_MPa": convert_value(s.get("Mean"), factor),
                    "equiv_p95_MPa": convert_value(s.get("P95"), factor),
                    "equiv_max_MPa": convert_value(s.get("Max"), factor),
                    "conversion_factor": factor,
                }
            )
    return pd.DataFrame(rows)


def convert_daily(daily: pd.DataFrame, assumptions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, d in daily.iterrows():
        for _, a in assumptions.iterrows():
            factor = a["conversion_factor_eqMPa_per_fieldMPa"]
            rows.append(
                {
                    "date": d.get("date"),
                    "support_model_scenario": a["support_model"],
                    "area_scenario": a["area_scenario"],
                    "field_support_mean_MPa": d.get("support_mean"),
                    "field_support_p95_MPa": d.get("support_p95"),
                    "field_support_max_MPa": d.get("support_max"),
                    "equiv_support_mean_MPa": convert_value(d.get("support_mean"), factor),
                    "equiv_support_p95_MPa": convert_value(d.get("support_p95"), factor),
                    "equiv_support_max_MPa": convert_value(d.get("support_max"), factor),
                    "active_support_count": d.get("active_support_count"),
                    "stress_mean": d.get("stress_mean"),
                    "stress_max": d.get("stress_max"),
                    "conversion_factor": factor,
                }
            )
    return pd.DataFrame(rows)


def make_mapping_curve(assumptions: pd.DataFrame) -> None:
    x = pd.Series(range(0, 46))
    plt.figure(figsize=(7.2, 4.8))
    for _, a in assumptions.iterrows():
        if a["area_scenario"] != "base_area":
            continue
        label = f"{a['support_model']}, A={a['equivalent_control_area_m2']:.1f} m2"
        plt.plot(x, x * a["conversion_factor_eqMPa_per_fieldMPa"], label=label)
    plt.axvspan(20, 25, color="#f2c94c", alpha=0.18, label="field warning 20-25 MPa")
    plt.axvline(25, color="#eb5757", linestyle="--", linewidth=1.2, label="field >25 MPa")
    plt.xlabel("Field hydraulic support pressure / MPa")
    plt.ylabel("Equivalent roof support pressure / MPa")
    plt.title("Mapping from field hydraulic pressure to FLAC3D equivalent support pressure")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "field_pressure_to_equivalent_pressure_curve.png", dpi=220)
    plt.close()


def write_report(assumptions: pd.DataFrame, converted_stats: pd.DataFrame) -> None:
    base = converted_stats[
        (converted_stats["support_model_scenario"] == "ZY8000/23/47D")
        & (converted_stats["area_scenario"] == "base_area")
    ].copy()
    long_duration = base[base["monitoring_group"] == "long_duration"]
    lines = [
        "# Field hydraulic pressure to FLAC3D equivalent support pressure mapping",
        "",
        "## Conversion relation",
        "",
        "`F = p_hydraulic * A_column_total * eta`",
        "",
        "`p_equiv = F / A_control`",
        "",
        "Because complete column geometry and canopy/control area were not found in the 7306 field files, the first usable mapping is written in rated-resistance form:",
        "",
        "`p_equiv = p_field * R_rated / (p_rated * A_control)`",
        "",
        "where `p_field` is the monitored hydraulic pressure in MPa, `R_rated` is support rated working resistance in MN, `p_rated` is the assumed rated hydraulic pressure in MPa, and `A_control` is equivalent roof-control area in m2.",
        "",
        "## Current assumptions",
        "",
        "| Support model | Rated resistance / kN | Rated hydraulic pressure / MPa | Control area / m2 | Factor eqMPa/fieldMPa |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in assumptions.to_dict("records"):
        lines.append(
            f"| {row['support_model']} | {row['rated_working_resistance_kN']:.0f} | "
            f"{row['rated_hydraulic_pressure_MPa']:.1f} | {row['equivalent_control_area_m2']:.1f} | "
            f"{row['conversion_factor_eqMPa_per_fieldMPa']:.4f} |"
        )

    if not long_duration.empty:
        lines.extend(
            [
                "",
                "## Base-case mapping result",
                "",
                "Base case: `ZY8000/23/47D`, `p_rated = 40 MPa`, `A_control = 6.0 m2`.",
                "",
                f"- Long-duration field mean: {pd.to_numeric(long_duration['field_mean_MPa']).mean():.3f} MPa -> equivalent mean {pd.to_numeric(long_duration['equiv_mean_MPa']).mean():.3f} MPa.",
                f"- Long-duration field P95 mean: {pd.to_numeric(long_duration['field_p95_MPa']).mean():.3f} MPa -> equivalent P95 mean {pd.to_numeric(long_duration['equiv_p95_MPa']).mean():.3f} MPa.",
                f"- Long-duration field max: {pd.to_numeric(long_duration['field_max_MPa']).max():.3f} MPa -> equivalent max {pd.to_numeric(long_duration['equiv_max_MPa']).max():.3f} MPa.",
            ]
        )

    lines.extend(
        [
            "",
            "## How to use this in the paper",
            "",
            "Use the mapped equivalent pressure as external validation of the FLAC3D surrogate output, not as an additional training label yet. The robust statement is trend/magnitude validation: field high-pressure periods correspond to higher equivalent support demand and should fall within the model-predicted demand range.",
            "",
            "For strict numerical validation, replace the temporary assumptions with manufacturer/manual values: column diameter and count, setting pressure, safety-valve pressure, support center distance, roof-control distance, canopy/contact area, and mechanical efficiency.",
            "",
            "## Output files",
            "",
            "- `tables/support_pressure_mapping_assumptions.csv`",
            "- `tables/field_support_statistics_equivalent_pressure.csv`",
            "- `tables/field_daily_equivalent_pressure.csv`",
            "- `figures/field_pressure_to_equivalent_pressure_curve.png`",
        ]
    )
    (OUT_DIR / "field_pressure_equivalent_mapping_report.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    setup_style()

    stats_path = VALIDATION_DIR / "tables" / "field_support_statistics_clean.csv"
    daily_path = VALIDATION_DIR / "tables" / "field_daily_support_stress_response.csv"
    stats = pd.read_csv(stats_path)
    daily = pd.read_csv(daily_path)

    assumptions = make_assumptions()
    converted_stats = convert_support_statistics(stats, assumptions)
    converted_daily = convert_daily(daily, assumptions)

    assumptions.to_csv(TABLE_DIR / "support_pressure_mapping_assumptions.csv", index=False, encoding="utf-8-sig")
    converted_stats.to_csv(TABLE_DIR / "field_support_statistics_equivalent_pressure.csv", index=False, encoding="utf-8-sig")
    converted_daily.to_csv(TABLE_DIR / "field_daily_equivalent_pressure.csv", index=False, encoding="utf-8-sig")

    make_mapping_curve(assumptions)
    write_report(assumptions, converted_stats)

    print(f"out: {OUT_DIR}")
    print(assumptions[["support_model", "area_scenario", "conversion_factor_eqMPa_per_fieldMPa"]].to_string(index=False))


if __name__ == "__main__":
    main()
