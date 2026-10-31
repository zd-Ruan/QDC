"""Aggregate independent checkpoint evaluations; do not mix rollout and training-seed variance."""

import argparse
import json
from collections import defaultdict
from statistics import mean, stdev


def summarize(files):
    groups = defaultdict(list)
    seen = set()
    training_seeds = set()
    protocols = {}
    for path in files:
        with open(path, encoding="utf-8") as stream:
            result = json.load(stream)
        key = (result["env_id"], result["checkpoint"])
        if key in seen:
            raise ValueError(
                "Repeated checkpoint: provide one evaluation per independently trained model"
            )
        seen.add(key)
        env_id = result["env_id"]
        seed = result.get("training_seed")
        if seed is not None:
            seed_key = (env_id, seed)
            if seed_key in training_seeds:
                raise ValueError("Repeated training seed: supply independently trained models")
            training_seeds.add(seed_key)
        protocol = {
            key: result.get(key)
            for key in (
                "action_selection",
                "num_candidates",
                "rtg_from_q",
                "targets",
                "mode",
            )
        }
        if env_id in protocols and protocols[env_id] != protocol:
            raise ValueError(f"Incompatible inference settings for {env_id}")
        protocols[env_id] = protocol
        groups[result["env_id"]].append(result)
    summary = {}
    for env_id, results in groups.items():
        scores = [r["normalized_score_mean"] for r in results]
        sample_std = stdev(scores) if len(scores) > 1 else None
        summary[env_id] = {
            "models": len(scores),
            "rollouts": sum(r["episodes"] for r in results),
            "mean": mean(scores),
            "std_between_models": sample_std,
            "standard_error": sample_std / len(scores) ** 0.5 if sample_std is not None else None,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    print(json.dumps(summarize(args.files), indent=2))


if __name__ == "__main__":
    main()
