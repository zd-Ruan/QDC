import copy
import json
import pickle

import numpy as np
import torch

from qdc.checkpoint import load_checkpoint, save_checkpoint
from qdc.train import build_parser, run
from qdc.trainer import TrainConfig, Trainer, multi_step_targets


def test_multi_step_targets_match_hand_computed_values():
    rewards = torch.tensor([[[1.0], [2.0], [3.0], [999.0]]])
    terminals = torch.zeros_like(rewards)
    targets = multi_step_targets(rewards, terminals, torch.tensor([[10.0]]), 0.5)
    torch.testing.assert_close(targets.flatten(), torch.tensor([4.0, 6.0, 8.0, 10.0]))
    terminals[:, 1] = 1
    targets = multi_step_targets(rewards, terminals, torch.tensor([[10.0]]), 0.5)
    torch.testing.assert_close(targets.flatten(), torch.tensor([2.0, 2.0, 8.0, 10.0]))


def test_update_and_checkpoint_roundtrip(models, dataset, tmp_path):
    actor, critic = models
    trainer = Trainer(actor, critic, TrainConfig(ema_start=0, ema_every=1, max_iters=2))
    before = copy.deepcopy(actor.state_dict())
    for _ in range(2):
        losses = trainer.train_step(dataset.sample(4))
        assert all(np.isfinite(list(losses.values())))
    assert any(not torch.equal(value, before[key]) for key, value in actor.state_dict().items())
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, trainer, dataset, {"env": "hopper", "dataset": "medium"}, 1)
    restored, restored_critic, payload = load_checkpoint(checkpoint)
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
    for key, value in trainer.target_critic.state_dict().items():
        torch.testing.assert_close(value, restored_critic.state_dict()[key])
    np.testing.assert_allclose(payload["state_mean"].numpy(), dataset.state_mean)


def test_offline_training_cli_needs_no_simulator(trajectories, tmp_path):
    path = tmp_path / "trajectories.pkl"
    with path.open("wb") as stream:
        pickle.dump(trajectories, stream)
    args = build_parser().parse_args(
        [
            "--data-file",
            str(path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--device",
            "cpu",
            "--eval-every",
            "0",
            "--max-iters",
            "1",
            "--steps-per-iter",
            "2",
            "--batch-size",
            "2",
            "--hidden-size",
            "8",
            "--n-layers",
            "1",
        ]
    )
    folder = run(args)
    assert (folder / "last.pt").is_file()
    assert not (folder / "best.pt").exists()
    config = json.loads((folder / "config.json").read_text())
    assert config["run"]["eval_every"] == 0
    load_checkpoint(folder / "last.pt")
