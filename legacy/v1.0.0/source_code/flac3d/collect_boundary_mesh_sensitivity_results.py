from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "工况" / "flac3d_boundary_mesh_sensitivity_cases"
ANALYSIS_DIR = ROOT / "分析" / "flac3d_model_sensitivity"


def parse_summary_tab(path: Path) -> dict[str, float]:
    values: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            idx = int(float(parts[0]))
            val = float(parts[1])
        except ValueError:
            continue
        values[idx] = val
    missing = {1, 2, 3, 4, 5}.difference(values)
    if missing:
        raise ValueError(f"{path.name} missing summary keys: {sorted(missing)}")
    return {
        "uz_min_m": values[1],
        "uz_max_m": values[2],
        "max_settlement_m": abs(values[1]),
        "max_heave_m": abs(values[2]),
        "peak_szz_MPa": values[3] / 1.0e6,
        "support_pressure_MPa_from_summary": values[4] / 1.0e6,
        "q_top_MPa_from_summary": values[5] / 1.0e6,
    }


def pct_change(value: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (value - base) / abs(base) * 100.0


def main() -> None:
    manifest_path = CASE_DIR / "boundary_mesh_sensitivity_case_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    rows = []
    for _, row in manifest.iterrows():
        out = row.to_dict()
        summary_path = CASE_DIR / str(row["expected_summary_file"])
        if summary_path.exists():
            out.update(parse_summary_tab(summary_path))
            out["run_status"] = "done"
            controllable = out["max_settlement_m"] <= 2.0 and out["max_heave_m"] <= 2.0
            out["computed_label_at_2m"] = "controllable_at_2.6MPa" if controllable else "not_controllable_at_2.6MPa"
        else:
            out["run_status"] = "missing_summary"
            out["computed_label_at_2m"] = "not_yet_available"
        rows.append(out)

    df = pd.DataFrame(rows)
    done = df[df["run_status"].eq("done")].copy()
    if not done.empty:
        for sens_case, group in done.groupby("sensitivity_case"):
            medium = group[group["mesh_level"].eq("medium")]
            if medium.empty:
                continue
            medium = medium.iloc[0]
            idx = done["sensitivity_case"].eq(sens_case)
            for col in ["max_settlement_m", "max_heave_m", "peak_szz_MPa"]:
                done.loc[idx, f"relative_change_{col}_vs_medium_pct"] = done.loc[idx, col].apply(
                    lambda x: pct_change(float(x), float(medium[col]))
                )
            medium_label = medium["computed_label_at_2m"]
            done.loc[idx, "label_change_vs_medium"] = done.loc[idx, "computed_label_at_2m"].apply(
                lambda x: "unchanged" if x == medium_label else "changed"
            )
        df = pd.concat([done, df[df["run_status"].ne("done")]], ignore_index=True)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = ANALYSIS_DIR / "Table_S2_boundary_mesh_sensitivity_results.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    md = ["# Boundary-case FLAC3D mesh-sensitivity results", ""]
    md.append("This table checks whether a near-boundary controllable case and an uncontrolled `>2.6 MPa` case keep the same displacement-control label under coarse, medium and fine x-y mesh resolutions.")
    md.append("")
    if done.empty:
        md.append("FLAC3D summaries have not been found yet. Run `工况/flac3d_boundary_mesh_sensitivity_cases/batch_run_boundary_mesh_sensitivity.dat`, then rerun this collector.")
    else:
        show_cols = [
            "sensitivity_case",
            "mesh_level",
            "source_sample_id",
            "support_pressure_MPa",
            "max_settlement_m",
            "max_heave_m",
            "peak_szz_MPa",
            "computed_label_at_2m",
            "label_change_vs_medium",
        ]
        md.append(done[show_cols].to_markdown(index=False, floatfmt=".4g"))
        md.append("")
        for sens_case, group in done.groupby("sensitivity_case"):
            changes = set(group["label_change_vs_medium"].dropna())
            if changes == {"unchanged"}:
                md.append(f"- `{sens_case}` kept the same displacement-control label across all completed mesh variants.")
            else:
                md.append(f"- `{sens_case}` showed label sensitivity across mesh variants and should be treated as a numerical boundary requiring FLAC3D re-check.")
    (ANALYSIS_DIR / "boundary_mesh_sensitivity_results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Saved: {out_csv}")
    print(f"Completed summaries: {int(df['run_status'].eq('done').sum())}/{len(df)}")


if __name__ == "__main__":
    main()
