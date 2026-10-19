"""Validate the versioned stage-3 dataset before BC training."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from environment import OBS_SCHEMAS


def validate(path):
    path = Path(path)
    manifest = json.loads(path.with_suffix(".json").read_text())
    if sha256(path.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Dataset checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        schema = str(data["observation_schema"])
        x, y, nxt = data["observation"], data["action"], data["next_observation"]
        ids, seeds = data["episode_id"], data["seed"]
        val_ids = data["validation_episode_ids"]
        unique = np.unique(ids)
        assert schema == manifest["observation_schema"] == "state30_v2"
        assert x.shape == nxt.shape == (len(ids), len(OBS_SCHEMAS[schema]))
        assert y.shape == (len(ids), 4)
        assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(nxt).all()
        assert ((y >= [-1, -1, -1, 0]) & (y <= 1)).all()
        success = {r["episode_id"]: r for r in manifest["episodes"] if r["success"]}
        assert set(unique) == set(success)
        assert len(unique) == manifest["success_count"]
        assert len(ids) == manifest["frames"]
        assert len(manifest["episodes"]) == manifest["attempt_count"]
        assert len({r["seed"] for r in manifest["episodes"]}) == manifest["attempt_count"]
        assert set(val_ids) == {i for i, r in success.items() if r["split"] == "validation"}
        rows = []
        for episode_id in unique:
            idx = np.flatnonzero(ids == episode_id)
            row = success[int(episode_id)]
            assert np.all(np.diff(idx) == 1)
            assert np.all(seeds[idx] == row["seed"])
            assert np.all(data["condition_id"][idx] == row["condition_id"])
            assert np.all(data["recovery"][idx] == row["recovery"])
            assert len(idx) == row["steps"]
            np.testing.assert_array_equal(nxt[idx[:-1]], x[idx[1:]])
            assert x[idx[0], 7] == 0 and nxt[idx[-1], 7] == 1
            assert row["wrong_cuts"] == row["protected_intrusions"] == 0
            assert not row["tool_terrain_contact"]
            if row["recovery"]:
                assert x[idx[0], 6] == 0 and row["reopened_before_cut"]
            rows.append(row)
        train = [r for r in rows if r["split"] == "train"]
        val = [r for r in rows if r["split"] == "validation"]
        assert not {r["seed"] for r in train} & {r["seed"] for r in val}
        return {"dataset_sha256": manifest["dataset_sha256"], "observation_schema": schema,
                "frames": len(ids), "accepted_episodes": len(rows),
                "train_episodes": len(train), "validation_episodes": len(val),
                "train_frames": int((~np.isin(ids, val_ids)).sum()),
                "validation_frames": int(np.isin(ids, val_ids).sum()),
                "recovery_episodes": sum(r["recovery"] for r in rows),
                "recovery_train": sum(r["recovery"] for r in train),
                "recovery_validation": sum(r["recovery"] for r in val),
                "conditions": [{"condition_id": i,
                    "accepted": sum(r["condition_id"] == i for r in rows),
                    "train": sum(r["condition_id"] == i for r in train),
                    "validation": sum(r["condition_id"] == i for r in val),
                    "target_counts": dict(Counter(r["target_id"] for r in rows if r["condition_id"] == i))}
                    for i in range(9)],
                "checks": "checksum, schema, finite values, action bounds, episode continuity, transitions, successful-only, disjoint seed split, closed-start recovery"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data")
    parser.add_argument("--out")
    args = parser.parse_args()
    result = validate(args.data)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(payload)
    print(payload)
