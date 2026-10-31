import copy

import numpy as np
import pytest

from qdc.data import TrajectoryDataset
from qdc.prepare_data import split_trajectories


def test_sampling_has_valid_transitions_and_full_future_returns(dataset):
    batch = dataset.sample(32)
    assert (batch.mask[:, :-1] & batch.mask[:, 1:]).any(dim=1).all()
    assert batch.states.shape == (32, 5, 3)
    assert batch.timesteps.max() < 20
    # Every real token's RTG includes all rewards remaining in the original trajectory.
    for i in range(32):
        mask = batch.mask[i]
        times = batch.timesteps[i, mask].numpy()
        last_rtg = batch.returns_to_go[i, mask][-1].item() * 10
        candidates = [
            p
            for p in dataset.trajectories
            if times[-1] < len(p["rewards"])
            and abs(p["returns_to_go"][times[-1]] - last_rtg) < 1e-4
        ]
        assert candidates


def test_sampling_is_seeded(trajectories):
    first = TrajectoryDataset(trajectories, 5, 20, 10.0, seed=9).sample(4)
    second = TrajectoryDataset(trajectories, 5, 20, 10.0, seed=9).sample(4)
    np.testing.assert_array_equal(first.states.numpy(), second.states.numpy())


def test_bad_trajectory_is_rejected(trajectories):
    paths = copy.deepcopy(trajectories)
    paths[0]["actions"][0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        TrajectoryDataset(paths, 5, 20, 10.0)


def test_preparation_keeps_trailing_fragment():
    raw = {
        "observations": np.zeros((5, 3)),
        "actions": np.zeros((5, 2)),
        "rewards": np.ones(5),
        "terminals": np.zeros(5),
        "timeouts": np.array([0, 1, 0, 0, 0]),
    }
    paths = split_trajectories(raw)
    assert [len(p["rewards"]) for p in paths] == [2, 3]
    assert not paths[0]["terminals"].any()  # A timeout must not become a true terminal.
