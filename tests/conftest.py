import numpy as np
import pytest
import torch

from qdc.data import TrajectoryDataset
from qdc.models import QDC, Critic, ModelConfig

torch.set_num_threads(1)


@pytest.fixture
def trajectories():
    rng = np.random.default_rng(7)
    return [
        {
            "observations": rng.normal(size=(length, 3)).astype(np.float32),
            "actions": rng.uniform(-1, 1, size=(length, 2)).astype(np.float32),
            "rewards": np.arange(1, length + 1, dtype=np.float32),
            "terminals": np.array([0] * (length - 1) + [1], dtype=np.float32),
        }
        for length in (3, 7)
    ]


@pytest.fixture
def dataset(trajectories):
    return TrajectoryDataset(trajectories, context_length=5, max_ep_len=20, scale=10.0)


@pytest.fixture
def models():
    torch.manual_seed(5)
    config = ModelConfig(
        3, 2, hidden_size=8, n_layers=2, context_length=5, max_ep_len=20, dropout=0.0, scale=10.0
    )
    return QDC(config), Critic(3, 2, hidden_size=8)
