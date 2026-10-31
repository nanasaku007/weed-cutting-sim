"""Build an auditable stage-4 summary from completed, frozen evaluations."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from environment import ACTION_NAMES, MAX_DELTA
from run import TreeBC, save_json
from stage4 import summarize


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def validation_analysis(data_path, checkpoint):
    policy = TreeBC(checkpoint)
    with np.load(data_path, allow_pickle=False) as z:
        mask = np.isin(z["episode_id"], z["validation_episode_ids"])
        x, y = z["observation"][mask], z["action"][mask]
        phases = z["teacher_phase"][mask]
        conditions = z["condition_id"][mask]
        recovery = z["recovery"][mask]
        prediction = np.asarray([policy.predict(o) for o in x])
        error = (prediction - y) ** 2
        def metrics(selected):
            return {"frames": int(selected.sum()), "action_mse": error[selected].mean(axis=0).tolist()}
        return {"actions": ACTION_NAMES, "overall": metrics(np.ones(len(x), dtype=bool)),
                "by_teacher_phase": {str(p): metrics(phases == p) for p in np.unique(phases)},
                "by_condition_id": {str(c): metrics(conditions == c) for c in np.unique(conditions)},
                "by_start": {"recovery": metrics(recovery), "normal": metrics(~recovery)},
                "note": "Teacher phase is an analysis label only, never a BC input; held-out demonstration frames, not closed-loop performance."}


def evaluation_summary(report):
    if not report["complete"] or len(report["conditions"]) != 18:
        raise ValueError("Evaluation is incomplete")
    rows = [r for c in report["conditions"] for r in c["episodes"]]
    result = {k: report[k] for k in ("seed_start", "count_per_condition", "recovery", "ood",
              "checkpoint_sha256", "source_sha256", "max_steps", "simulation_seconds_limit",
              "independent_layouts", "layout_reuse", "summary", "elapsed_seconds")}
    result["conditions"] = [{k: v for k, v in c.items() if k != "episodes"} for c in report["conditions"]]
    result["by_policy"] = {}
    for label in ("waypoint_v2", "bc_state30_v1"):
        selected = [r for r in rows if r["policy"] == label]
        result["by_policy"][label] = {
            "by_tilt": {str(t): summarize([r for r in selected if r["environment"]["tilt_degrees"] == t])
                        for t in (0, 10, 20)},
            "by_azimuth": {str(a): summarize([r for r in selected if r["environment"]["azimuth_degrees"] == a])
                           for a in (0, 90, 180, 270)},
            "failures": []}
        for row in selected:
            if row["success"]:
                continue
            failure = dict(row)
            if "failure_trace" in row:
                failure["failure_trace_sha256"] = digest(row["failure_trace"])
                with np.load(row["failure_trace"], allow_pickle=False) as trace:
                    commanded = trace["observation"][:, 3:6] + trace["action"][:, :3] * MAX_DELTA
                    bounded = np.clip(commanded, [0.28, -0.24, 0.095], [0.72, 0.24, 0.43])
                    failure["trace_diagnostics"] = {
                        "first_action": trace["action"][0].tolist(),
                        "last_action": trace["action"][-1].tolist(),
                        "final_tool_xyz_m": trace["next_observation"][-1, 3:6].tolist(),
                        "workspace_bound_command_steps": int(np.any(commanded != bounded, axis=1).sum())}
            result["by_policy"][label]["failures"].append(failure)
    pairs = {}
    for row in rows:
        config = row["environment"]
        key = (config["plant"], config["tilt_degrees"], row["seed"])
        pairs.setdefault(key, {})[row["policy"]] = row["success"]
    result["paired_success"] = {
        "both": sum(p["waypoint_v2"] and p["bc_state30_v1"] for p in pairs.values()),
        "teacher_only": sum(p["waypoint_v2"] and not p["bc_state30_v1"] for p in pairs.values()),
        "bc_only": sum(not p["waypoint_v2"] and p["bc_state30_v1"] for p in pairs.values()),
        "neither": sum(not p["waypoint_v2"] and not p["bc_state30_v1"] for p in pairs.values())}
    return result


def build(args):
    protocol_path = Path("experiments/stage4_protocol.json")
    protocol = json.loads(protocol_path.read_text())
    root = Path(args.root)
    checkpoint = root / "bc_state30_v1.npz"
    training = json.loads(checkpoint.with_suffix(".json").read_text())
    data = Path(protocol["dataset"])
    assert digest(data) == training["data_sha256"] == protocol["dataset_sha256"]
    assert digest(checkpoint) == training["checkpoint_sha256"]
    legacy = protocol["legacy_checkpoint"]
    assert digest(legacy["path"]) == digest(legacy["backup"]) == legacy["sha256"]
    with np.load(data, allow_pickle=False) as z:
        assert set(training["validation_episode_ids"]) == set(z["validation_episode_ids"])
        assert set(training["train_episode_ids"]) == set(z["episode_id"]) - set(z["validation_episode_ids"])
        dataset_seeds = set(z["seed"].tolist())
    evaluations = {}
    artifacts = [protocol_path, data, checkpoint, checkpoint.with_suffix(".json")]
    for name, seed_key in (("iid", "iid_seeds"), ("recovery", "recovery_seeds")):
        path = root / f"evaluation_{name}.json"
        report = json.loads(path.read_text())
        assert report["checkpoint_sha256"] == digest(checkpoint)
        start, end = protocol[seed_key]
        assert report["seed_start"] == start
        assert report["count_per_condition"] == end - start + 1
        assert not set(range(start, end + 1)) & dataset_seeds
        for c in report["conditions"]:
            assert [r["seed"] for r in c["episodes"]] == list(range(start, end + 1))
        evaluations[name] = evaluation_summary(report)
        artifacts.append(path)
    assert evaluations["iid"]["source_sha256"] == evaluations["recovery"]["source_sha256"]
    for source, expected in evaluations["iid"]["source_sha256"].items():
        assert digest(source) == expected, f"Evaluation source changed: {source}"
    analysis = validation_analysis(data, checkpoint)
    np.testing.assert_allclose(analysis["overall"]["action_mse"], training["validation_action_mse"], atol=1e-7)
    summary = {"date": "2026-10-25", "scope": "stage 4 fixed-split BC training and paired held-out closed-loop evaluation",
               "protocol": protocol, "training": training,
               "dataset_validation": json.loads(Path("outputs/stage3/dataset_validation_stage4.json").read_text()),
               "validation_analysis": analysis, "evaluation": evaluations,
               "legacy_checkpoint_unchanged": True,
               "artifacts_sha256": {str(p): digest(p) for p in artifacts},
               "report_source_sha256": digest(__file__)}
    save_json(args.out, summary)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/stage4")
    parser.add_argument("--out", default="experiments/stage4_summary.json")
    build(parser.parse_args())
