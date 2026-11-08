"""Single evaluation of locked new physical combinations after all native jobs."""
import json
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree

from common import *
from repair_models import predict_repaired


def map_native_displacements(zones, meta, mask, collector):
    plane = zones.loc[np.isclose(zones.Y_m, meta.slice_y_m, atol=1e-5, rtol=0)].copy()
    assert len(plane) >= 2500 and plane.zone_id.is_unique
    xx, zz = np.meshgrid(np.linspace(1., 171., 86), np.linspace(.5, 99.5, 64))
    points = np.column_stack([plane.X_m, collector.align_vertical_coordinate(plane.Z_m.to_numpy(), meta.M_m)])
    query = np.column_stack([xx.ravel(), zz.ravel()])
    triangulation, tree = Delaunay(points), cKDTree(points)
    _, nearest = tree.query(query, k=1)
    result = {}
    for target in HASHES:
        values = plane[target+"_m"].to_numpy(float)
        mapped = np.asarray(LinearNDInterpolator(triangulation, values, fill_value=np.nan)(query))
        missing = ~np.isfinite(mapped)
        mapped[missing] = values[nearest[missing]]
        mapped = mapped.astype(np.float32)
        mapped[~mask] = np.nan
        assert np.isfinite(mapped[mask]).all()
        result[target] = mapped
    return result


def bootstrap_groups(frame, draws=2000):
    """Conditional cluster-resampling variability, without IID pressure claims."""
    rng = np.random.default_rng(2026102507)
    groups = sorted(frame.physical_group.unique())
    if len(groups) < 2:
        return {"groups": len(groups), "available": False}
    group_parts = [frame.loc[frame.physical_group.eq(group)] for group in groups]
    metrics_draws = []
    for _ in range(draws):
        sampled = pd.concat([group_parts[i] for i in rng.integers(0, len(groups), len(groups))], ignore_index=True)
        a = aggregate(sampled)
        metrics_draws.append([a["r2"], a["rmse_m"], a["mae_m"]])
    values = np.asarray(metrics_draws)
    return {"groups": len(groups), "available": True, "draws": draws,
        "interpretation": "conditional physical-group resampling interval on a fixed targeted test design, not a population-wide coverage claim",
        "r2_interval95": np.nanquantile(values[:, 0], [.025, .975]).tolist(),
        "rmse_m_interval95": np.quantile(values[:, 1], [.025, .975]).tolist(),
        "mae_m_interval95": np.quantile(values[:, 2], [.025, .975]).tolist()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    state = json.loads((OUT / "native_validation/state.json").read_text(encoding="utf-8"))
    assert state["phase"] in ["native_complete", "native_outcomes_adjudicated"], "Independent reference collection is still running"
    assert state.get("adjudicated_cases", state["completed_cases"]) == state["total_cases"] == 10, "Every planned case needs a completed execution or an evidence-adjudicated numerical termination"
    policy_path = OUT / "experiment_protocol.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    freezing = json.loads((OUT / "independent_design/predictions_frozen.json").read_text(encoding="utf-8"))
    frozen_models = json.loads((OUT / "models_v2/独立验证前最终模型冻结.json").read_text(encoding="utf-8"))
    assert sha(policy_path) == freezing["protocol_sha256"]
    assert sha(OUT / "development/pretest_amendment_v2.json") == freezing["pretest_amendment_sha256"]
    assert sha(OUT / "independent_design/predictions_before_native.npz") == freezing["predictions_sha256"]
    for target, model in frozen_models["models"].items():
        assert sha(model["path"]) == model["sha256"]
    launched = json.loads((OUT / "native_validation/first_launch.json").read_text(encoding="utf-8"))
    assert datetime.datetime.fromisoformat(freezing["frozen_at"]) < datetime.datetime.fromisoformat(launched["started_at"])
    predictions = load_npz(OUT / "independent_design/predictions_before_native.npz")
    metadata = pd.read_csv(OUT / "independent_design/new_snapshot_manifest.csv")
    np.testing.assert_array_equal(metadata.case_id, predictions["case_id"])
    np.testing.assert_array_equal(metadata.step_index, predictions["step"])
    collector = load_module("independent_reference_collector", ROOT / "脚本/collect_flac3d_stage11_multistep_fields.py")
    output = OUT / "validation"
    output.mkdir(exist_ok=True)
    quality_rows, snapshot_rows, phases = [], [], []
    reference = {target: np.full(predictions[target+"_repaired_m"].shape, np.nan, np.float32) for target in HASHES}
    results = {row["case_id"]: row for row in state["cases"]}
    for case, local in metadata.groupby("case_id", sort=False):
        item = results[case]
        statuses, first_failure = {}, None
        directory = None
        if item["completed"] or item.get("terminal_outcome_adjudicated"):
            directory = Path(item["evidence_directory"])
            endpoint = pd.read_csv(directory / "phases.csv").set_index("phase")
            motion = pd.read_csv(directory / "kinematics_and_motion.csv").set_index("phase")
            last_phase = 7 if item["completed"] else int(item["last_complete_phase"])
            assert endpoint.index.tolist() == list(range(1, last_phase + 1)) and motion.index.tolist() == list(range(1, last_phase + 1))
            for phase in range(1, last_phase + 1):
                q, m = endpoint.loc[phase], motion.loc[phase]
                force = bool(q.ratio_local <= policy["native_reference"]["local_force_ratio_at_most"])
                volume = bool(m.nonpositive_deformed_zones == 0 and m.min_native_volume_ratio > 0)
                movement = bool(m.motion_screen_passed)
                independently_recomputed_motion = bool(m.continuation_ratio1 <= 1e-4 and m.continuation_ratio2 <= 1e-4
                    and m.window1_drift_m <= .001 and m.window2_drift_m <= .001 and m.total_2000_cycle_drift_m <= .001)
                assert movement == independently_recomputed_motion
                own = force and volume and movement
                if not own and first_failure is None:
                    first_failure = phase
                accepted = own and first_failure is None
                statuses[phase] = accepted
                phases.append({"case_id": case, "phase": phase, "force_passed": force, "native_volume_passed": volume,
                    "motion_passed": movement, "accepted_with_all_prerequisites": accepted,
                    "ratio_local": float(q.ratio_local), "min_native_volume_ratio": float(m.min_native_volume_ratio),
                    "window1_drift_m": float(m.window1_drift_m), "window2_drift_m": float(m.window2_drift_m),
                    "total_drift_m": float(m.total_2000_cycle_drift_m), "first_failed_phase": first_failure})
            if item.get("terminal_outcome_adjudicated"):
                failed_phase = int(item["numerical_failure_phase"])
                assert failed_phase == last_phase + 1
                if first_failure is None:
                    first_failure = failed_phase
                statuses[failed_phase] = False
                phases.append({"case_id": case, "phase": failed_phase, "force_passed": False,
                    "native_volume_passed": False, "motion_passed": False,
                    "accepted_with_all_prerequisites": False, "ratio_local": np.nan,
                    "min_native_volume_ratio": np.nan, "window1_drift_m": np.nan,
                    "window2_drift_m": np.nan, "total_drift_m": np.nan,
                    "first_failed_phase": first_failure, "native_termination": item["outcome"],
                    "endpoint_or_motion_values_imputed": False})
        for index, meta in local.iterrows():
            phase = int(meta.step_index)+2
            accepted = bool(statuses.get(phase, False))
            disposition = ("ACCEPTED_LARGE_REFERENCE" if accepted else
                "NATIVE_ILLEGAL_GEOMETRY_NO_ENDPOINT" if item.get("terminal_outcome_adjudicated") and phase >= item["numerical_failure_phase"] else
                "REFERENCE_PROTOCOL_NOT_MET" if item["completed"] or item.get("terminal_outcome_adjudicated") else "PENDING_INFRASTRUCTURE_EVIDENCE")
            quality_rows.append({"snapshot_index": index, "case_id": case, "physical_group": meta.physical_group,
                "support_pressure_MPa": meta.support_pressure_MPa, "step": int(meta.step_index),
                "quality_status": disposition, "first_failed_phase": first_failure,
                "evidence_directory": str(directory) if directory else "", "model_error_used_for_quality": False})
            if directory is None or phase > last_phase:
                continue
            zones = pd.read_csv(directory / f"zones_p{phase}.csv")
            mask = predictions["valid_mask"][index]
            mapped = map_native_displacements(zones, meta, mask, collector)
            for target in HASHES:
                reference[target][index] = mapped[target]
                for variant in ["archived", "repaired"]:
                    snapshot_rows.append({"snapshot_index": index, "case_id": case, "physical_group": meta.physical_group,
                        "pressure_MPa": meta.support_pressure_MPa, "sampling_stratum": meta.sampling_stratum,
                        "step": int(meta.step_index), "target": target, "variant": variant,
                        "quality_status": disposition, **metrics(mapped[target], predictions[target+"_"+variant+"_m"][index], mask)})
    quality, snapshots = pd.DataFrame(quality_rows), pd.DataFrame(snapshot_rows)
    assert len(quality) == 50 and quality.case_id.nunique() == 10 and quality.physical_group.nunique() == 6
    quality.to_csv(output / "新独立50快照_质量状态.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(phases).to_csv(output / "新独立70阶段_质量证据.csv", index=False, encoding="utf-8-sig")
    snapshots.to_csv(output / "新独立_全部原生结果逐快照指标_含质量标记.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(output / "新独立_原材料坐标参考场.npz", **{target+"_reference_m": reference[target] for target in HASHES},
                        valid_mask=predictions["valid_mask"], case_id=predictions["case_id"], step=predictions["step"])
    accepted = snapshots.loc[snapshots.quality_status.eq("ACCEPTED_LARGE_REFERENCE")].copy()
    summaries, group_rows, intervals = [], [], []
    for (variant, target), frame in accepted.groupby(["variant", "target"]):
        for step in range(6):
            q = frame if not step else frame.loc[frame.step.eq(step)]
            if len(q):
                summaries.append({"variant": variant, "target": target, "step": step, **aggregate(q)})
                for group, g in q.groupby("physical_group"):
                    group_rows.append({"variant": variant, "target": target, "step": step,
                                       "physical_group": group, **aggregate(g)})
        fifth = frame.loc[frame.step.eq(5)]
        if len(fifth):
            intervals.append({"variant": variant, "target": target, **bootstrap_groups(fifth)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "新独立D3_整体与逐步_数值质量条件统计.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(group_rows).to_csv(output / "新独立_物理组合指标.csv", index=False, encoding="utf-8-sig")
    save_json(output / "新独立_物理组合重采样区间.json", intervals)
    gates, high_rows = [], []
    criteria = policy["independent_step5_gates"]
    q5 = quality.loc[quality.step.eq(5)]
    fraction = float(q5.quality_status.eq("ACCEPTED_LARGE_REFERENCE").mean())
    groups_n = int(q5.loc[q5.quality_status.eq("ACCEPTED_LARGE_REFERENCE"), "physical_group"].nunique())
    def gate(name, value, limit, direction="at_most"):
        passed = bool(value >= limit if direction == "at_least" else value <= limit) if value is not None and np.isfinite(value) else False
        gates.append({"gate": name, "value": value, "limit": limit, "direction": direction, "passed": passed})
    gate("qualified_pressure_case_fraction", fraction, criteria["minimum_qualified_pressure_case_fraction"], "at_least")
    gate("qualified_physical_groups", groups_n, criteria["minimum_qualified_physical_groups"], "at_least")
    vertical5 = accepted.loc[accepted.step.eq(5) & accepted.target.eq("uz") & accepted.variant.eq("repaired")]
    high_cases = vertical5.loc[vertical5.reference_abs_peak_m.ge(.5), "case_id"].tolist()
    high_group_n = int(vertical5.loc[vertical5.case_id.isin(high_cases), "physical_group"].nunique())
    gate("high_deformation_physical_groups", high_group_n, criteria["minimum_high_deformation_physical_groups"], "at_least")
    for target in HASHES:
        q = accepted.loc[accepted.step.eq(5) & accepted.target.eq(target) & accepted.variant.eq("repaired")]
        values = aggregate(q) if len(q) else {}
        gate(target+"_R2", values.get("r2"), criteria["pooled_r2_minimum_each_target"], "at_least")
        gate(target+"_RMSE_m", values.get("rmse_m"), criteria["pooled_RMSE_max_m"][target])
        gate(target+"_case_RMSE_P90_m", values.get("snapshot_rmse_p90_m"), criteria["snapshot_RMSE_P90_max_m"][target])
        gate(target+"_case_RMSE_max_m", values.get("snapshot_rmse_max_m"), criteria["snapshot_RMSE_max_m"][target])
        gate(target+"_peak_error_median_fraction", values.get("peak_error_median_fraction"), criteria["absolute_peak_relative_error_median_max"])
        gate(target+"_peak_error_P90_fraction", values.get("peak_error_p90_fraction"), criteria["absolute_peak_relative_error_P90_max"])
        high = q.loc[q.case_id.isin(high_cases)]
        if len(high):
            high_stats = aggregate(high)
            high_rows.append({"target": target, **high_stats})
        else:
            high_stats = {}
        gate(target+"_high_deformation_RMSE_m", high_stats.get("rmse_m"), criteria["high_deformation_RMSE_max_m"][target])
        early = predictions["step"] < 5
        np.testing.assert_array_equal(predictions[target+"_archived_m"][early], predictions[target+"_repaired_m"][early])
        package = joblib.load(frozen_models["models"][target]["path"])
        reloaded = predict_repaired(package, predictions["features"], predictions["valid_mask"])
        np.testing.assert_array_equal(reloaded, predictions[target+"_repaired_m"])
    pd.DataFrame(gates).to_csv(output / "新独立_预先固定阈值逐项验收.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(high_rows).to_csv(output / "新独立_高变形子集指标.csv", index=False, encoding="utf-8-sig")
    result = {"evaluated_at": now(), "selected_variant": frozen_models["selected_variant"],
        "new_physical_groups_planned": 6, "pressure_cases_planned": 10, "snapshots_planned": 50,
        "native_full_executions_completed": state["completed_cases"],
        "native_outcomes_adjudicated": state.get("adjudicated_cases", state["completed_cases"]),
        "native_numerical_termination_cases": state.get("numerical_terminal_cases", 0),
        "quality_accepted_step5": int(q5.quality_status.eq("ACCEPTED_LARGE_REFERENCE").sum()),
        "quality_accepted_groups_step5": groups_n, "qualified_step5_fraction": fraction,
        "high_deformation_physical_groups": high_group_n,
        "all_gates_passed": all(g["passed"] for g in gates), "gates": gates,
        "step5_results": summary.loc[summary.step.eq(5)].to_dict("records") if len(summary) else [],
        "models_selected_and_predictions_frozen_before_new_native_execution": True,
        "no_model_tuning_after_test_truth": True,
        "historical_53_and_g012_not_used_as_new_independent_test": True,
        "numerical_quality_acceptance_is_not_field_validation": True,
        "reference_quality_exclusion_does_not_prove_physical_collapse": True,
        "first4_predictions_bitwise_unchanged": True,
        "precomputed_predictions_bitwise_reproduced_after_reference_collection": True,
        "protocol_sha256": sha(policy_path), "amendment_sha256": freezing["pretest_amendment_sha256"],
        "frozen_prediction_sha256": freezing["predictions_sha256"],
        "test_design_sha256": sha(OUT / "independent_design/new_snapshot_manifest.csv")}
    save_json(output / "independent_results.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
