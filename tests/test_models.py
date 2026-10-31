import pytest
import torch

from qdc.models import ModelConfig


def test_future_tokens_cannot_change_current_action(models):
    actor, _ = models
    actor.eval()
    states, actions, rtg = torch.randn(2, 5, 3), torch.randn(2, 5, 2), torch.randn(2, 5, 1)
    times = torch.arange(5).repeat(2, 1)
    _, reference = actor(states, actions, rtg, times)
    states[:, 3:] += 100
    actions[:, 2:] -= 100  # a_2 follows s_2, so it must not affect the prediction of a_2.
    rtg[:, 3:] -= 100
    _, changed = actor(states, actions, rtg, times)
    torch.testing.assert_close(reference[:, :3], changed[:, :3])


@pytest.mark.parametrize("selection", ["sample", "greedy"])
@pytest.mark.parametrize("history_length", [1, 3])
def test_candidate_selection_runs_on_cpu(models, selection, history_length):
    actor, critic = models
    action = actor.get_action(
        critic,
        torch.randn(history_length, 3),
        torch.randn(history_length, 2),
        torch.zeros(history_length, 1),
        torch.ones(2, history_length),
        torch.arange(history_length),
        num_candidates=7,
        selection=selection,
    )
    assert action.shape == (2,)
    assert torch.isfinite(action).all()
    assert action.abs().max() <= 1


def test_model_rejects_invalid_context():
    with pytest.raises(ValueError, match="context_length"):
        ModelConfig(3, 2, context_length=1)
