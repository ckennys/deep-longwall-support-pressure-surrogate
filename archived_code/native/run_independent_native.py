"""Serial native references after immutable model and prediction freezing."""
import json
import shutil
import sys
import time

import pandas as pd

from common import *


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    design = OUT / "independent_design"
    freeze = json.loads((design / "predictions_frozen.json").read_text(encoding="utf-8"))
    assert sha(design / "predictions_before_native.npz") == freeze["predictions_sha256"]
    assert sha(OUT / "experiment_protocol.json") == freeze["protocol_sha256"]
    for target, item in freeze["models"].items():
        assert sha(item["path"]) == item["sha256"]
    sys.path.insert(0, str(AUDIT))
    import native_quality_v2 as native
    import native_completion_v3 as resilient_native
    from serial_native_gate import configured_threads
    assert configured_threads() == 10
    native.base.SOURCE = design / "cases"
    manifest = pd.read_csv(design / "new_case_manifest.csv").sort_values("run_order")
    status_path = OUT / "native_validation/state.json"
    status_path.parent.mkdir(exist_ok=True)
    completed = []
    launch_path = OUT / "native_validation/first_launch.json"
    if not launch_path.exists():
        save_json(launch_path, {"started_at": now(), "models_and_predictions_frozen_before_launch": freeze,
            "single_solver_threads": 10, "maximum_concurrent_native_processes": 1})
    for case in manifest.itertuples():
        case_state = OUT / f"native_validation/{case.case_id}/case_state.json"
        if case_state.exists():
            existing = json.loads(case_state.read_text(encoding="utf-8"))
            if existing.get("completed"):
                completed.append(existing)
                continue
        success, errors = None, []
        for attempt in range(1, 4):
            directory = case_state.parent / f"attempt_{attempt}"
            result_path = directory / "result.json"
            if result_path.exists():
                success = json.loads(result_path.read_text(encoding="utf-8"))
                break
            if (directory / "execution.json").exists():
                previous = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
                if not previous.get("completed"):
                    import psutil
                    if not psutil.pid_exists(int(previous.get("pid", -1))):
                        errors.append({"attempt": attempt, "preserved_interrupted_attempt": str(directory)})
                        continue
            if shutil.disk_usage(ROOT).free < 30*1024**3:
                raise RuntimeError("Less than locked 30 GiB reserve; preserve outputs and await storage")
            save_json(status_path, {"updated_at": now(), "phase": "running", "active_case": case.case_id,
                "attempt": attempt, "completed_cases": len(completed), "total_cases": len(manifest),
                "threads_per_native": 10, "max_concurrent_native": 1, "directory": str(directory)})
            print(f"Starting {case.case_id}, case {len(completed)+1}/{len(manifest)}, 10 threads", flush=True)
            try:
                native.prepare_replay(case.case_id, "large", directory, max_phase=7)
                resilient_native.run_native(directory)
                success = native.summarize_replay(case.case_id, "large", directory, max_phase=7)
                break
            except Exception as error:
                errors.append({"attempt": attempt, "error": repr(error), "directory": str(directory)})
                save_json(case_state, {"case_id": case.case_id, "completed": False, "errors": errors})
                if attempt < 3:
                    time.sleep(5)
        if success is None:
            item = {"case_id": case.case_id, "completed": False, "errors": errors,
                    "numerical_quality_rejection_inferred_from_execution_error": False}
        else:
            item = {"case_id": case.case_id, "completed": True, "evidence_directory": str(directory),
                "reference_summary": success, "errors": errors}
        save_json(case_state, item)
        completed.append(item)
        print(f"Finished {case.case_id}: completed={item['completed']}", flush=True)
    save_json(status_path, {"updated_at": now(), "phase": "native_complete", "active_case": None,
        "completed_cases": sum(bool(r["completed"]) for r in completed), "total_cases": len(manifest),
        "cases": completed, "evaluation_not_yet_performed": True,
        "models_and_precomputed_predictions_unchanged": True})
    print(json.dumps({"phase": "native_complete", "completed": sum(bool(r["completed"]) for r in completed),
                      "total": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
