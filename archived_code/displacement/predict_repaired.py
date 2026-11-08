"""Explicit, frozen research-candidate entry point with canonical input units."""
import argparse
import json
import sys

import joblib
import numpy as np

from common import *
from repair_models import predict_repaired


def validate_inputs(data, assume_canonical=False):
    features, valid = np.asarray(data["features"]), np.asarray(data["valid_mask"], bool)
    if features.ndim != 2 or features.shape[1] != 14 or not np.isfinite(features).all():
        raise ValueError("A finite N by 14 canonical feature matrix is required")
    valid = valid.reshape(len(features), -1)
    if valid.shape[1] != 5504 or np.any(valid.sum(1)==0):
        raise ValueError("Each 64 by 86 mask must contain valid field points")
    if "feature_names" in data:
        if list(data["feature_names"].astype(str)) != FEATURES:
            raise ValueError("Input feature names or order differ from the frozen 14-column contract")
    elif not assume_canonical:
        raise ValueError("Provide feature_names or explicitly use --assume-canonical-order for the audited historical input archive")
    if np.any(features[:, 8] <= 0) or np.any(features[:, 8] > 10):
        raise ValueError("The support-pressure input is positive MPa, for example 2.3, with SI conversion only in the FLAC3D source")
    if not np.all(np.isclose(features[:, 13, None], np.arange(1, 6)[None, :]/5., atol=1e-7, rtol=0).any(1)):
        raise ValueError("The excavation-step feature must be one of 0.2, 0.4, 0.6, 0.8, 1.0")
    return features, valid


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--assume-canonical-order", action="store_true")
    args = parser.parse_args()
    data = load_npz(args.input)
    features, valid = validate_inputs(data, args.assume_canonical_order)
    freezing = json.loads((OUT / "models_v2/独立验证前最终模型冻结.json").read_text(encoding="utf-8"))
    fields = {"valid_mask": valid, "feature_names": np.asarray(FEATURES), "features": features}
    for key in ("case_id", "step", "step_index"):
        if key in data:
            fields[key] = data[key]
    for target, info in freezing["models"].items():
        if sha(info["path"]) != info["sha256"]:
            raise RuntimeError("Frozen candidate checkpoint identity changed")
        package = joblib.load(info["path"])
        fields[target+"_m"] = predict_repaired(package, features, valid)
        if "step5_guard" in package:
            head, guard = package["archived_head"], package["step5_guard"]
            c = old_module().predict_regressor(head["regressor"], old_module().engineered(features))*head["scale"]
            active = guard["active_modes"]
            counts = np.sum((c[:, active] < guard["lower"][active]) | (c[:, active] > guard["upper"][active]), axis=1)
            fields[target+"_guarded_modes_count"] = np.where(np.isclose(features[:, 13], 1.), counts, 0)
    evaluated = OUT / "validation/independent_results.json"
    status = json.loads(evaluated.read_text(encoding="utf-8"))["all_gates_passed"] if evaluated.exists() else None
    fields["independent_research_gates_passed"] = np.asarray("pending" if status is None else str(status))
    fields["selected_variant"] = np.asarray(freezing["selected_variant"])
    np.savez_compressed(args.output, **fields)
    print(json.dumps({"output": str(Path(args.output).resolve()), "snapshots": len(features),
        "units": "displacement m; network pressure MPa", "candidate": freezing["selected_variant"],
        "independent_research_gates_passed": status}, ensure_ascii=False))


if __name__ == "__main__":
    main()
