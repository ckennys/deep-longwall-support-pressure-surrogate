"""Pretest amendment: verify a fit-only coefficient envelope without case rules."""
import json
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from common import *
from repair_models import predict_repaired


def make_guard(coefficients, active_count, margin):
    lower, upper = coefficients.min(0), coefficients.max(0)
    padding = margin*np.maximum(upper-lower, .015*max(float(coefficients.std()), 1e-6))
    return {"lower": lower-padding, "upper": upper+padding, "active_modes": np.arange(active_count),
            "margin": margin, "fit_reference_coefficients_only": True}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    assert not (OUT / "independent_design/predictions_frozen.json").exists()
    assert not list((OUT / "native_validation").glob("**/execution.json"))
    base_policy = OUT / "experiment_protocol.json"
    amendment = OUT / "development/pretest_amendment_v2.json"
    variants = [{"name": f"Guard{count}_margin{int(margin*100):02d}", "active_modes": count, "margin": margin}
                for count in (3, 64) for margin in (0., .10, .25)]
    lock = {"created_at": now(), "base_protocol_sha256": sha(base_policy),
        "reason": "The first two clean-reference retraining variants miss development accuracy gates; dominant coefficient errors justify a universal fit-only output envelope control",
        "amended_before_any_new_test_reference_or_prediction_freeze": True,
        "independent_test_gates_design_and_numerical_quality_rules_unchanged": True,
        "original_two_variants_and_their_results_preserved": True,
        "new_variants": variants, "same_group_folds_and_selection_metric": True,
        "bounds_fit_on_training_group_true_PCA_coefficients_only": True,
        "pretrained_PCA_basis_and_MLP_weights_unchanged": True,
        "no_g012_or_pressure_or_case_id_specific_rule": True,
        "two_previously_compared_variants_remain_in_selection_pool": True,
        "independent_validation_is_single_use_after_final_freeze": True}
    if amendment.exists():
        existing = json.loads(amendment.read_text(encoding="utf-8"))
        lock["created_at"] = existing["created_at"]
        assert existing == lock
    else:
        save_json(amendment, lock)
    protected = protected_hashes()
    data, cached = load_npz(OUT / "development/可信候选参考_265快照.npz"), load_npz(CACHE)
    folds = json.loads((OUT / "development/物理组合交叉验证_分组.json").read_text(encoding="utf-8"))["fold_by_group"]
    old = old_module()
    rows, group_rows, packages = [], [], {}
    for target in HASHES:
        model_path = MODEL_DIR / f"models/{target}_m_Sqrt_Neural64.joblib"
        assert sha(model_path) == HASHES[target]
        head = joblib.load(model_path)["head"]
        basis, mean = head["pca"].components_[:64], head["pca"].mean_
        valid, truth = data["valid_mask"], data[target]
        transformed = np.where(valid, np.sign(truth)*np.sqrt(np.abs(truth)), 0.)
        with threadpool_limits(limits=2):
            reference_coefficients = (transformed-mean)@basis.T
        predicted_coefficients = old.predict_regressor(head["regressor"], old.engineered(data["features"]))*head["scale"]
        original = cached[target+"_prediction_m"][data["snapshot_index"]].reshape(len(truth), -1)
        with threadpool_limits(limits=2):
            reproduced = inverse(predicted_coefficients@basis+mean)
        np.testing.assert_allclose(reproduced[valid], original[valid], rtol=5e-5, atol=5e-5)
        for specification in variants:
            variant = specification["name"]
            oof_prediction = original.copy()
            clipped_counts = np.zeros(len(truth), int)
            for fold in range(5):
                held = np.array([folds[str(g)] == fold for g in data["physical_group"]])
                train = ~held
                guard = make_guard(reference_coefficients[train], specification["active_modes"], specification["margin"])
                selected = held & data["step"].astype(int).__eq__(5)
                c = predicted_coefficients[selected].copy()
                before = c.copy()
                active = guard["active_modes"]
                c[:, active] = np.clip(c[:, active], guard["lower"][active], guard["upper"][active])
                clipped_counts[selected] = np.sum(c != before, axis=1)
                with threadpool_limits(limits=2):
                    oof_prediction[selected] = inverse(c@basis+mean)
            for index in np.flatnonzero(data["step"] == 5):
                rows.append({"variant": variant, "target": target, "case_id": str(data["case_id"][index]),
                    "physical_group": str(data["physical_group"][index]), "step": 5,
                    "fold": folds[str(data["physical_group"][index])], "guarded_coefficients": int(clipped_counts[index]),
                    **metrics(truth[index], oof_prediction[index], valid[index])})
            np.savez_compressed(OUT / f"development/cv/{target}_{variant}_held_predictions.npz",
                source_rows=np.flatnonzero(data["step"] == 5), prediction=oof_prediction[data["step"] == 5])
            guard = make_guard(reference_coefficients, specification["active_modes"], specification["margin"])
            packages[target, variant] = {"target": target, "feature_names": FEATURES,
                "archived_head": head, "step5_guard": guard, "grid_shape": (64, 86),
                "routing": "Step5 universal fit-reference coefficient envelope; Steps1to4 archived head",
                "selected_variant": variant, "development_physical_groups": sorted(np.unique(data["physical_group"]).tolist()),
                "protocol_sha256": sha(base_policy), "pretest_amendment_sha256": sha(amendment),
                "reference_dataset_sha256": sha(OUT / "development/可信候选参考_265快照.npz")}
    guard_rows = pd.DataFrame(rows)
    guard_rows.to_csv(OUT / "development/逐快照_系数边界OOF指标.csv", index=False, encoding="utf-8-sig")
    previous_rows = pd.read_csv(OUT / "development/逐快照_分组OOF指标.csv")
    previous_rows = previous_rows.loc[previous_rows.step.eq(5)]
    combined = pd.concat([previous_rows, guard_rows], ignore_index=True)
    summaries, ranking = [], []
    for (variant, target), frame in combined.groupby(["variant", "target"]):
        summaries.append({"variant": variant, "target": target, **aggregate(frame)})
        for group, q in frame.groupby("physical_group"):
            group_rows.append({"variant": variant, "target": target, "physical_group": group, **aggregate(q)})
    group_scores = pd.DataFrame(group_rows)
    for variant, q in group_scores.groupby("variant"):
        normalized = q.rmse_m/np.where(q.target.eq("uz"), .05, .02)
        per_group = pd.DataFrame({"group": q.physical_group, "normalized": normalized}).groupby("group").normalized.mean()
        ranking.append({"variant": variant, "selection_score": float(per_group.mean()), "worst_group_score": float(per_group.max())})
    ranking = sorted(ranking, key=lambda x: (x["selection_score"], x["worst_group_score"]))
    selected = ranking[0]["variant"]
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "development/全部8方案_第5步OOF指标.csv", index=False, encoding="utf-8-sig")
    group_scores.to_csv(OUT / "development/全部8方案_第5步OOF物理组合指标.csv", index=False, encoding="utf-8-sig")
    folder = OUT / "models_v2"
    folder.mkdir(exist_ok=True)
    new_models = {}
    for target in HASHES:
        if (target, selected) in packages:
            package = packages[target, selected]
        else:
            package = joblib.load(OUT / f"models/{target}_step5_repaired.joblib")
            assert package["selected_variant"] == selected
        model_path = folder / f"{target}_step5_final_candidate.joblib"
        joblib.dump(package, model_path)
        prediction = predict_repaired(package, data["features"], data["valid_mask"])
        early = data["step"] != 5
        np.testing.assert_allclose(prediction[early][data["valid_mask"][early]],
            cached[target+"_prediction_m"][data["snapshot_index"][early]].reshape(early.sum(), -1)[data["valid_mask"][early]], rtol=5e-5, atol=5e-5)
        np.savez_compressed(OUT / f"development/{target}_final_candidate_fullfit_predictions.npz", prediction=prediction)
        new_models[target] = {"path": str(model_path), "sha256": sha(model_path)}
    assert protected == protected_hashes()
    record = {"frozen_at": now(), "models": new_models, "selected_variant": selected,
        "selection_ranking": ranking, "protocol_sha256": sha(base_policy), "pretest_amendment_sha256": sha(amendment),
        "prior_model_freeze_preserved": str(OUT / "models/新模型_独立验证前冻结.json"),
        "new_independent_truth_generated_or_used": False,
        "development_selection_only": True, "selected_oof": summary.loc[summary.variant.eq(selected)].to_dict("records")}
    save_json(OUT / "models_v2/独立验证前最终模型冻结.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
