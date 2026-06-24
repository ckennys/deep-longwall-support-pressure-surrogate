from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "工况" / "flac3d_support_search_cases"
CASE_TABLE = CASE_DIR / "cases_stage2_support_search.csv"

NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def read_summary(path: Path) -> dict[str, float]:
    values: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        nums = [float(x) for x in NUMBER_RE.findall(line)]
        if len(nums) >= 2:
            key = int(round(nums[0]))
            if 1 <= key <= 5:
                values[key] = nums[1]

    required = {1, 2, 3, 4, 5}
    missing = required.difference(values)
    if missing:
        raise ValueError(f"{path.name} missing summary keys: {sorted(missing)}")

    return {
        "max_settlement_m": -values[1],
        "max_heave_m": values[2],
        "peak_abutment_stress_MPa": values[3] / 1.0e6,
        "support_pressure_MPa_from_summary": values[4] / 1.0e6,
        "q_top_MPa_from_summary": values[5] / 1.0e6,
    }


def fnum(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or "nan")
    except ValueError:
        return math.nan


def qc_flag(row: dict[str, str], settlement_limit: float, heave_limit: float) -> str:
    settlement = fnum(row, "max_settlement_m")
    heave = fnum(row, "max_heave_m")
    flags = []
    if not math.isfinite(settlement):
        flags.append("missing_settlement")
    elif settlement < 0:
        flags.append("negative_settlement")
    elif settlement > settlement_limit:
        flags.append(f"settlement_gt_{settlement_limit:g}m")
    if math.isfinite(heave) and heave > heave_limit:
        flags.append(f"heave_gt_{heave_limit:g}m")
    return ";".join(flags) if flags else "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, default=CASE_DIR)
    parser.add_argument("--case-table", type=Path, default=CASE_TABLE)
    parser.add_argument("--results-name", default="results_stage2_support_search.csv")
    parser.add_argument("--labels-name", default="labels_required_support_pressure.csv")
    parser.add_argument("--settlement-limit", type=float, default=2.0)
    parser.add_argument("--heave-limit", type=float, default=2.0)
    args = parser.parse_args()

    case_dir = args.case_dir
    case_table = args.case_table
    results_table = case_dir / args.results_name
    labels_table = case_dir / args.labels_name

    with case_table.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        case_id = row["case_id"]
        summary_path = case_dir / f"summary_{case_id}.tab"
        if summary_path.exists():
            row.update({k: f"{v:.8g}" for k, v in read_summary(summary_path).items()})
            row["summary_file"] = summary_path.name
            row["status"] = "done"
        else:
            row["summary_file"] = ""
            row["status"] = "missing_summary"
        row["qc_flag"] = qc_flag(row, args.settlement_limit, args.heave_limit)
        row["meets_control"] = "1" if row["qc_flag"] == "ok" else "0"

    result_fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in result_fields:
                result_fields.append(key)
    with results_table.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(rows)

    geo_ids = sorted({row["geo_id"] for row in rows})
    label_rows = []
    for geo_id in geo_ids:
        geo_rows = [row for row in rows if row["geo_id"] == geo_id]
        geo_rows.sort(key=lambda row: fnum(row, "support_pressure_MPa"))
        ok_rows = [row for row in geo_rows if row["meets_control"] == "1"]
        base = geo_rows[0]
        label = {
            "geo_id": geo_id,
            "H_m": base["H_m"],
            "lambda": base["lambda"],
            "M_m": base["M_m"],
            "q_top_MPa": base["q_top_MPa"],
            "alpha_E_roof": base["alpha_E_roof"],
            "alpha_c_roof": base["alpha_c_roof"],
            "phi_roof_deg": base["phi_roof_deg"],
            "alpha_t_roof": base["alpha_t_roof"],
            "required_support_pressure_MPa": "",
            "required_case_id": "",
            "label_status": "no_level_meets_control",
        }
        if ok_rows:
            best = ok_rows[0]
            label["required_support_pressure_MPa"] = best["support_pressure_MPa"]
            label["required_case_id"] = best["case_id"]
            label["label_status"] = "labeled"
        label_rows.append(label)

    with labels_table.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(label_rows[0]))
        writer.writeheader()
        writer.writerows(label_rows)

    done = sum(1 for row in rows if row["status"] == "done")
    labeled = sum(1 for row in label_rows if row["label_status"] == "labeled")
    print(f"Collected {done}/{len(rows)} summaries -> {results_table}")
    print(f"Labeled {labeled}/{len(label_rows)} geo combinations -> {labels_table}")


if __name__ == "__main__":
    main()
