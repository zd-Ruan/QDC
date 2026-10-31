# SPDX-License-Identifier: Apache-2.0
# Trajectory sampling adapted from QT / Decision Transformer; see NOTICE.
"""Validation, normalization, and left-padded trajectory sampling."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Batch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    returns_to_go: torch.Tensor
    timesteps: torch.Tensor
    mask: torch.Tensor


def discount_cumsum(values, discount=1.0):
    result = np.zeros_like(values, dtype=np.float32)
    running = 0.0
    for i in range(len(values) - 1, -1, -1):
        running = float(values[i]) + discount * running
        result[i] = running
    return result


class TrajectoryDataset:
    """Sample trajectories proportional to their length, as in the original implementation."""

    def __init__(
        self, trajectories, context_length, max_ep_len, scale, seed=123, fraction=1.0, mode="normal"
    ):
        if not 0 < fraction <= 1 or context_length < 2 or scale <= 0 or max_ep_len < 1:
            raise ValueError(
                "Invalid trajectory fraction, context length, episode length, or scale"
            )
        if mode not in ("normal", "delayed"):
            raise ValueError("mode must be normal or delayed")
        self.trajectories = []
        if not isinstance(trajectories, (list, tuple)) or not trajectories:
            raise ValueError("Dataset must contain a non-empty list of trajectory dictionaries")
        for index, trajectory in enumerate(trajectories):
            path = {}
            for key in ("observations", "actions", "rewards"):
                if key not in trajectory:
                    raise ValueError(f"Trajectory {index} is missing {key}")
                path[key] = np.asarray(trajectory[key], dtype=np.float32).copy()
            terminals = trajectory.get("terminals", trajectory.get("dones"))
            if terminals is None:
                raise ValueError(f"Trajectory {index} is missing terminals (or dones)")
            path["terminals"] = np.asarray(terminals, dtype=np.float32).reshape(-1)
            path["rewards"] = path["rewards"].reshape(-1)
            length = len(path["rewards"])
            if length == 0 or any(len(x) != length for x in path.values()):
                raise ValueError(f"Trajectory {index} has empty or inconsistent arrays")
            if path["observations"].ndim != 2 or path["actions"].ndim != 2:
                raise ValueError(f"Trajectory {index}: observations and actions must be 2D")
            if not all(np.isfinite(x).all() for x in path.values()):
                raise ValueError(f"Trajectory {index} contains NaN or infinite values")
            if not np.isin(path["terminals"], [0, 1]).all():
                raise ValueError(f"Trajectory {index}: terminals must be binary")
            if path["terminals"][:-1].any():
                raise ValueError(
                    f"Trajectory {index} contains an internal terminal; split episodes first"
                )
            if mode == "delayed":
                total = path["rewards"].sum()
                path["rewards"][:] = 0
                path["rewards"][-1] = total
            path["returns_to_go"] = discount_cumsum(path["rewards"])
            self.trajectories.append(path)
        self.state_dim = self.trajectories[0]["observations"].shape[1]
        self.action_dim = self.trajectories[0]["actions"].shape[1]
        if any(
            p["observations"].shape[1] != self.state_dim or p["actions"].shape[1] != self.action_dim
            for p in self.trajectories
        ):
            raise ValueError("All trajectories must have the same state and action dimensions")
        all_states = np.concatenate([p["observations"] for p in self.trajectories])
        self.state_mean = all_states.mean(axis=0)
        self.state_std = all_states.std(axis=0) + 1e-6
        lengths = np.array([len(p["rewards"]) for p in self.trajectories])
        returns = np.array([p["rewards"].sum() for p in self.trajectories])
        ordered = [i for i in np.argsort(returns)[::-1] if lengths[i] >= 2]
        if not ordered:
            raise ValueError("Training requires at least one trajectory with two or more steps")
        budget = max(int(lengths.sum() * fraction), 1)
        selected, total = [], 0
        for i in ordered:
            if selected and total + lengths[i] > budget:
                break
            selected.append(i)
            total += lengths[i]
        self.indices = np.array(selected)
        self.probabilities = lengths[self.indices] / lengths[self.indices].sum()
        self.context_length, self.max_ep_len, self.scale = context_length, max_ep_len, scale
        self.rng = np.random.default_rng(seed)

    @classmethod
    def load(cls, path, **kwargs):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}. Run qdc.prepare_data first.")
        # Historical D4RL trajectory files use pickle. Only load data from trusted sources.
        with path.open("rb") as stream:
            return cls(pickle.load(stream), **kwargs)

    def sample(self, batch_size, device="cpu"):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        size = (batch_size, self.context_length)
        arrays = {
            "states": np.zeros((*size, self.state_dim), dtype=np.float32),
            "actions": np.zeros((*size, self.action_dim), dtype=np.float32),
            "rewards": np.zeros((*size, 1), dtype=np.float32),
            "terminals": np.zeros((*size, 1), dtype=np.float32),
            "returns_to_go": np.zeros((*size, 1), dtype=np.float32),
            "timesteps": np.zeros(size, dtype=np.int64),
            "mask": np.zeros(size, dtype=bool),
        }
        indices = self.rng.choice(self.indices, size=batch_size, p=self.probabilities)
        for row, index in enumerate(indices):
            path = self.trajectories[index]
            # A single valid timestep cannot supervise a Bellman transition.
            start = int(self.rng.integers(max(len(path["rewards"]) - 1, 1)))
            stop = min(start + self.context_length, len(path["rewards"]))
            length = stop - start
            offset = self.context_length - length
            for field, key in (("states", "observations"), ("actions", "actions")):
                arrays[field][row, offset:] = path[key][start:stop]
            for key in ("rewards", "terminals", "returns_to_go"):
                arrays[key][row, offset:, 0] = path[key][start:stop]
            arrays["timesteps"][row, offset:] = np.arange(start, stop).clip(max=self.max_ep_len - 1)
            arrays["mask"][row, offset:] = True
        # Match the original convention: normalize zero padding along with observations.
        arrays["states"] = (arrays["states"] - self.state_mean) / self.state_std
        arrays["returns_to_go"] /= self.scale
        return Batch(**{key: torch.from_numpy(value).to(device) for key, value in arrays.items()})

    def summary(self):
        return {
            "trajectories": len(self.trajectories),
            "selected_trajectories": len(self.indices),
            "timesteps": sum(len(p["rewards"]) for p in self.trajectories),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }
