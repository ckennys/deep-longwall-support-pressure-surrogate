from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "模型模板" / "flac3d_longwall_template.dat"
OUT_DIR = ROOT / "工况" / "flac3d_support_search_cases"


def lhs_samples(n: int, bounds: dict[str, tuple[float, float]], seed: int) -> list[dict[str, float]]:
    rng = random.Random(seed)
    values: dict[str, list[float]] = {}
    for key, (lo, hi) in bounds.items():
        bins = []
        for i in range(n):
            u = (i + rng.random()) / n
            bins.append(lo + u * (hi - lo))
        rng.shuffle(bins)
        values[key] = bins
    return [{key: values[key][i] for key in bounds} for i in range(n)]


def render(template: str, mapping: dict[str, object]) -> str:
    text = template
    for key, value in mapping.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def fmt(value: float) -> str:
    return f"{value:.6g}"


def save_commands(case_id: str, save_mode: str) -> dict[str, str]:
    commands = {
        "SAVE_INIT_COMMAND": f"model save 'init_{case_id}'",
        "SAVE_ROADWAY_COMMAND": f"model save 'roadway_{case_id}'",
        "SAVE_STEP01_COMMAND": f"model save '{case_id}_step01'",
        "SAVE_STEP02_COMMAND": f"model save '{case_id}_step02'",
        "SAVE_STEP03_COMMAND": f"model save '{case_id}_step03'",
        "SAVE_STEP04_COMMAND": f"model save '{case_id}_step04'",
        "SAVE_STEP05_COMMAND": f"model save '{case_id}_step05'",
        "SAVE_FINAL_COMMAND": f"model save 'final_{case_id}'",
    }
    if save_mode == "full":
        return commands
    if save_mode == "final":
        for key in commands:
            if key != "SAVE_FINAL_COMMAND":
                commands[key] = f"; {key} skipped by save-mode final"
        return commands
    if save_mode == "summary-only":
        return {key: f"; {key} skipped by save-mode summary-only" for key in commands}
    raise ValueError(f"Unsupported save mode: {save_mode}")


def build_case_mapping(
    geo_no: int,
    pressure_no: int,
    sample: dict[str, float],
    support_pressure_mpa: float,
    save_mode: str,
) -> dict[str, object]:
    model_height = 100.0
    coal_floor = 9.0
    face_width = 20.0
    gamma_v = 2.4e4

    h = sample["H"]
    lam = sample["lambda"]
    mining_height = sample["M"]
    alpha_e_roof = sample["alpha_E_roof"]
    alpha_c_roof = sample["alpha_c_roof"]
    phi_roof = sample["phi_roof"]
    alpha_t_roof = sample["alpha_t_roof"]

    coal_top = coal_floor + mining_height
    roadway_top = coal_floor + min(4.5, mining_height)

    q_top = gamma_v * (h - 85.0)
    sigma_v_bottom = q_top + gamma_v * model_height
    gamma_h = lam * gamma_v
    sigma_h_bottom = lam * sigma_v_bottom

    roof_bulk = ((13.2e9 + 10.2e9) / 2.0) * alpha_e_roof
    roof_shear = ((5.0e9 + 4.5e9) / 2.0) * alpha_e_roof
    roof_cohesion = ((5.65e6 + 4.65e6) / 2.0) * alpha_c_roof
    roof_tension = ((4.03e6 + 3.03e6) / 2.0) * alpha_t_roof

    support_pressure = support_pressure_mpa * 1.0e6
    panel_length = 20.0
    support_area = panel_length * face_width
    p_s = support_pressure * support_area

    case_id = f"s2_g{geo_no:02d}_p{pressure_no:02d}"
    mapping = {
        "CASE_ID": case_id,
        "FACE_WIDTH_M": fmt(face_width),
        "NY": 8,
        "Y_MID_M": fmt(face_width / 2.0),
        "COAL_TOP_Z": fmt(coal_top),
        "COAL_TOP_Z_MIN": fmt(coal_top - 0.05),
        "COAL_TOP_Z_MAX": fmt(coal_top + 0.05),
        "ROADWAY_TOP_Z": fmt(roadway_top),
        "Q_TOP_PA": fmt(q_top),
        "SIGMA_V_BOTTOM_PA": fmt(sigma_v_bottom),
        "SIGMA_H_BOTTOM_PA": fmt(sigma_h_bottom),
        "GAMMA_V_PA_PER_M": fmt(gamma_v),
        "GAMMA_H_PA_PER_M": fmt(gamma_h),
        "ROOF_BULK_PA": fmt(roof_bulk),
        "ROOF_SHEAR_PA": fmt(roof_shear),
        "ROOF_COHESION_PA": fmt(roof_cohesion),
        "ROOF_TENSION_PA": fmt(roof_tension),
        "ROOF_FRICTION_DEG": fmt(phi_roof),
        "SUPPORT_PRESSURE_PA": fmt(support_pressure),
        "_geo_no": geo_no,
        "_pressure_no": pressure_no,
        "_H": h,
        "_lambda": lam,
        "_M": mining_height,
        "_alpha_E_roof": alpha_e_roof,
        "_alpha_c_roof": alpha_c_roof,
        "_phi_roof": phi_roof,
        "_alpha_t_roof": alpha_t_roof,
        "_P_s": p_s,
        "_q_top": q_top,
        "_support_pressure": support_pressure,
    }
    mapping.update(save_commands(case_id, save_mode))
    return mapping


def write_batch_files(out_dir: Path, dat_files: list[str], chunk_size: int) -> None:
    all_lines = [f"program call '{name}'" for name in dat_files]
    (out_dir / "batch_run_stage2_all.dat").write_text("\n".join(all_lines) + "\n", encoding="utf-8")

    for start in range(0, len(dat_files), chunk_size):
        end = min(start + chunk_size, len(dat_files))
        chunk = dat_files[start:end]
        lines = [f"program call '{name}'" for name in chunk]
        batch_name = f"batch_run_stage2_{start + 1:03d}_{end:03d}.dat"
        (out_dir / batch_name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-geo", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument(
        "--save-mode",
        choices=("full", "final", "summary-only"),
        default="summary-only",
        help="full saves all states; final saves only final state; summary-only exports only .tab summaries.",
    )
    args = parser.parse_args()

    bounds = {
        "H": (800.0, 1200.0),
        "lambda": (0.8, 1.5),
        "M": (3.0, 6.0),
        "alpha_E_roof": (0.6, 1.4),
        "alpha_c_roof": (0.6, 1.4),
        "phi_roof": (25.0, 40.0),
        "alpha_t_roof": (0.5, 1.3),
    }
    support_levels_mpa = [0.5, 0.8, 1.1, 1.4, 1.7]

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    geo_samples = lhs_samples(args.n_geo, bounds, args.seed)
    rows = []
    dat_files = []

    for geo_no, sample in enumerate(geo_samples, start=1):
        for pressure_no, support_pressure_mpa in enumerate(support_levels_mpa, start=1):
            mapping = build_case_mapping(geo_no, pressure_no, sample, support_pressure_mpa, args.save_mode)
            case_id = mapping["CASE_ID"]
            dat_path = OUT_DIR / f"{case_id}.dat"
            dat_path.write_text(render(template, mapping), encoding="utf-8")
            dat_files.append(dat_path.name)
            rows.append(
                {
                    "case_id": case_id,
                    "geo_id": f"g{geo_no:02d}",
                    "pressure_level_id": f"p{pressure_no:02d}",
                    "H_m": fmt(mapping["_H"]),
                    "lambda": fmt(mapping["_lambda"]),
                    "M_m": fmt(mapping["_M"]),
                    "q_top_MPa": fmt(mapping["_q_top"] / 1.0e6),
                    "alpha_E_roof": fmt(mapping["_alpha_E_roof"]),
                    "alpha_c_roof": fmt(mapping["_alpha_c_roof"]),
                    "phi_roof_deg": fmt(mapping["_phi_roof"]),
                    "alpha_t_roof": fmt(mapping["_alpha_t_roof"]),
                    "support_pressure_MPa": fmt(mapping["_support_pressure"] / 1.0e6),
                    "P_s": fmt(mapping["_P_s"]),
                    "dat_file": dat_path.name,
                }
            )

    with (OUT_DIR / "cases_stage2_support_search.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    write_batch_files(OUT_DIR, dat_files, args.chunk_size)
    print(f"Generated {len(rows)} FLAC3D support-search cases in {OUT_DIR}")


if __name__ == "__main__":
    main()
