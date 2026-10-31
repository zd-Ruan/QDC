"""Self-contained inference checkpoints, plus an explicit historical checkpoint importer."""

import json
from dataclasses import asdict
from pathlib import Path

import torch

from qdc.models import QDC, Critic, ModelConfig


def save_checkpoint(path, trainer, dataset, run_config, iteration):
    payload = {
        "format_version": 1,
        "iteration": iteration,
        "step": trainer.step,
        "model_config": trainer.actor.config_dict(),
        "train_config": asdict(trainer.config),
        "run_config": run_config,
        "actor": trainer.actor.state_dict(),
        "critic": trainer.target_critic.state_dict(),
        "state_mean": torch.as_tensor(dataset.state_mean),
        "state_std": torch.as_tensor(dataset.state_std),
    }
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu", legacy_config=None, dataset=None):
    payload = torch.load(path, map_location=device, weights_only=True)
    if "model_config" in payload:
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format version")
        config = ModelConfig(**payload["model_config"])
    else:
        if legacy_config is None or dataset is None:
            raise ValueError(
                "Historical weights lack normalization and architecture metadata. "
                "Supply --legacy-config variant.json and --data-file original-dataset.pkl."
            )
        with Path(legacy_config).open(encoding="utf-8") as stream:
            variant = json.load(stream)
        if variant.get("sar") or variant.get("remove_act_embs") or variant.get("infer_no_q"):
            raise ValueError(
                "Historical SAR / removed-action / no-Q-inference ablations are unsupported"
            )
        config = ModelConfig(
            state_dim=dataset.state_dim,
            action_dim=dataset.action_dim,
            hidden_size=variant["embed_dim"],
            n_layers=variant["n_layer"],
            window_size=variant.get("conv_window_size", 6),
            context_length=variant["K"],
            max_ep_len=variant.get("max_ep_len", 1000),
            dropout=variant.get("dropout", 0.1),
            scale=variant["scale"],
        )
        payload["state_mean"] = torch.as_tensor(dataset.state_mean)
        payload["state_std"] = torch.as_tensor(dataset.state_std)
        payload["run_config"] = {
            "env": variant["env"],
            "dataset": variant["dataset"],
            "targets": variant.get("env_targets"),
            "mode": variant.get("mode", "normal"),
            "rtg_from_q": not variant.get("rtg_no_q", False),
            "seed": variant.get("seed"),
        }
    actor = QDC(config).to(device)
    critic = Critic(config.state_dim, config.action_dim, config.hidden_size).to(device)
    actor.load_state_dict(payload["actor"], strict=True)
    critic.load_state_dict(payload["critic"], strict=True)
    return actor.eval(), critic.eval(), payload
