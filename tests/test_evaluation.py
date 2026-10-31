import numpy as np
import pytest

from qdc.evaluation import evaluate_policy


class FakeEnv:
    def __init__(self, new_api):
        self.new_api = new_api
        self.steps = 0

    def reset(self, seed=None):
        self.steps = 0
        observation = np.zeros(3, dtype=np.float32)
        return (observation, {}) if self.new_api else observation

    def step(self, action):
        assert action.shape == (2,)
        self.steps += 1
        observation = np.full(3, self.steps, dtype=np.float32)
        if self.new_api:
            return observation, 1.0, False, self.steps == 3, {}
        return observation, 1.0, self.steps == 3, {}

    def get_normalized_score(self, score):
        return score / 10


@pytest.mark.parametrize("new_api", [False, True])
def test_rollout_handles_both_step_apis_and_score_scale(models, new_api):
    actor, critic = models
    result = evaluate_policy(
        FakeEnv(new_api),
        actor,
        critic,
        np.zeros(3),
        np.ones(3),
        [10.0, 5.0],
        episodes=2,
        num_candidates=5,
    )
    assert result["returns"] == [3.0, 3.0]
    assert result["length_mean"] == 3
    assert result["normalized_score_mean"] == 30
