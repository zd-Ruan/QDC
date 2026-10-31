"""Export D4RL datasets to trajectory pickle files without dropping the final fragment."""

import argparse
import pickle
from pathlib import Path

import numpy as np

from qdc.envs import make_env, task_spec


def split_trajectories(dataset, max_ep_len=1000):
    required = ("observations", "actions", "rewards", "terminals")
    if any(key not in dataset for key in required):
        raise ValueError(f"Dataset must contain {required}")
    length = len(dataset["rewards"])
    if length == 0 or any(len(dataset[key]) != length for key in required):
        raise ValueError("Dataset arrays must be non-empty and have the same length")
    if "timeouts" in dataset and len(dataset["timeouts"]) != length:
        raise ValueError("timeouts must match the dataset length")
    keys = [*required, *(["next_observations"] if "next_observations" in dataset else [])]
    paths, start = [], 0
    for index in range(length):
        timeout = (
            bool(dataset["timeouts"][index])
            if "timeouts" in dataset
            else index - start + 1 >= max_ep_len
        )
        if bool(dataset["terminals"][index]) or timeout or index == length - 1:
            paths.append({key: np.asarray(dataset[key][start : index + 1]) for key in keys})
            start = index + 1
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    spec = task_spec(args.env, args.dataset)
    destination = args.output_dir / f"{spec.env_id}.pkl"
    if destination.exists() and not args.overwrite:
        parser.error(f"{destination} already exists; use --overwrite to replace it")
    env = make_env(spec.env_id)
    try:
        paths = split_trajectories(env.get_dataset(), spec.max_ep_len)
    finally:
        env.close()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        pickle.dump(paths, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"Saved {len(paths)} trajectories / {sum(len(p['rewards']) for p in paths)} steps to {destination}"
    )


if __name__ == "__main__":
    main()
