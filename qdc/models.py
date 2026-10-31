# SPDX-License-Identifier: Apache-2.0
# Adapted from QT and Decision ConvFormer. See NOTICE and THIRD_PARTY_NOTICES.md.
"""Causal local filtering policy and twin Q networks, implemented in plain PyTorch."""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    state_dim: int
    action_dim: int
    hidden_size: int = 128
    n_layers: int = 4
    window_size: int = 6
    context_length: int = 8
    max_ep_len: int = 1000
    dropout: float = 0.1
    scale: float = 1000.0

    def __post_init__(self):
        for key in (
            "state_dim",
            "action_dim",
            "hidden_size",
            "n_layers",
            "window_size",
            "max_ep_len",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.context_length < 2:
            raise ValueError("context_length must be at least 2 for multi-step Q learning")
        if self.scale <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("scale must be positive and dropout must be in [0, 1)")


class Critic(nn.Module):
    """Two independently parameterized state-action value functions."""

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()

        def network():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_size),
                nn.Mish(),
                nn.Linear(hidden_size, hidden_size),
                nn.Mish(),
                nn.Linear(hidden_size, hidden_size),
                nn.Mish(),
                nn.Linear(hidden_size, 1),
            )

        self.q1_model = network()
        self.q2_model = network()

    def forward(self, states, actions):
        inputs = torch.cat((states, actions), dim=-1)
        return self.q1_model(inputs), self.q2_model(inputs)

    def q_min(self, states, actions):
        q1, q2 = self(states, actions)
        return torch.minimum(q1, q2)


class CausalConvolution(nn.Module):
    """Modality-specific depthwise filters over interleaved (RTG, state, action) tokens."""

    def __init__(self, config):
        super().__init__()
        self.window_size = config.window_size
        for name in ("rtg_conv1d", "obs_conv1d", "act_conv1d"):
            setattr(
                self,
                name,
                nn.Conv1d(
                    config.hidden_size,
                    config.hidden_size,
                    config.window_size,
                    groups=config.hidden_size,
                ),
            )

    def forward(self, tokens):
        padded = F.pad(tokens.transpose(1, 2), (self.window_size - 1, 0))
        filtered = (
            self.rtg_conv1d(padded)[:, :, 0::3],
            self.obs_conv1d(padded)[:, :, 1::3],
            self.act_conv1d(padded)[:, :, 2::3],
        )
        return torch.stack(filtered, dim=-1).flatten(2).transpose(1, 2)


class ConvBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        size = config.hidden_size
        self.ln_1 = nn.LayerNorm(size, eps=1e-5)
        self.conv = CausalConvolution(config)
        self.ln_2 = nn.LayerNorm(size, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(size, 4 * size),
            nn.GELU(),
            nn.Linear(4 * size, size),
            nn.Dropout(config.dropout),
        )

    def forward(self, tokens):
        tokens = tokens + self.conv(self.ln_1(tokens))
        return tokens + self.mlp(self.ln_2(tokens))


class ConvBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([ConvBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=1e-5)
        # Match the original backbone's GPT2-style linear initialization.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(self, tokens):
        tokens = self.drop(tokens)
        for block in self.h:
            tokens = block(tokens)
        return self.ln_f(tokens)


class QDC(nn.Module):
    """Return-conditioned convolutional policy with Q-guided candidate selection.

    Parameter names retain compatibility with the historical ``ql_DC`` state dict.
    The backbone is causal by construction; padding is excluded by the training loss.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.transformer = ConvBackbone(config)
        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_return = nn.Linear(1, config.hidden_size)
        self.embed_rewards = nn.Linear(1, config.hidden_size)
        self.embed_state = nn.Linear(config.state_dim, config.hidden_size)
        self.embed_action = nn.Linear(config.action_dim, config.hidden_size)
        self.embed_ln = nn.LayerNorm(config.hidden_size)
        self.predict_state = nn.Linear(config.hidden_size, config.state_dim)
        self.predict_action = nn.Sequential(
            nn.Linear(config.hidden_size, config.action_dim), nn.Tanh()
        )
        # Retained solely for compatibility with old weights; standard QDC does not predict rewards.
        self.predict_rewards = nn.Linear(config.hidden_size, 1)

    def forward(self, states, actions, returns_to_go, timesteps):
        batch, length, _ = states.shape
        times = self.embed_timestep(timesteps)
        tokens = torch.stack(
            (
                self.embed_return(returns_to_go) + times,
                self.embed_state(states) + times,
                self.embed_action(actions) + times,
            ),
            dim=2,
        ).reshape(batch, 3 * length, self.config.hidden_size)
        hidden = self.transformer(self.embed_ln(tokens))
        hidden = hidden.reshape(batch, length, 3, self.config.hidden_size)
        return self.predict_state(hidden[:, :, 2]), self.predict_action(hidden[:, :, 1])

    @torch.no_grad()
    def get_action(
        self,
        critic,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        num_candidates=50,
        selection="sample",
        rtg_from_q=True,
    ):
        """Generate candidate actions and select according to their minimum twin-Q value."""
        if num_candidates < 1 or selection not in ("sample", "greedy"):
            raise ValueError("num_candidates must be positive; selection must be sample or greedy")
        target_count = returns_to_go.shape[0]
        count = max(num_candidates, target_count)
        length = self.config.context_length
        history_length = states.shape[0]

        def history(tensor, width):
            tensor = tensor.reshape(1, -1, width)[:, -length:]
            return F.pad(tensor, (0, 0, length - tensor.shape[1], 0)).repeat(count, 1, 1)

        states = history(states, self.config.state_dim)
        actions = history(actions, self.config.action_dim)
        rewards = history(rewards, 1)
        steps = timesteps.reshape(1, -1)[:, -length:]
        steps = F.pad(steps, (length - steps.shape[1], 0)).repeat(count, 1)
        steps = steps.clamp(max=self.config.max_ep_len - 1)
        rtg = returns_to_go.reshape(target_count, -1, 1)[:, -length:]
        rtg = rtg.repeat_interleave(count // target_count, dim=0)
        if rtg.shape[0] < count:
            rtg = torch.cat(
                (
                    rtg,
                    torch.randn(
                        count - rtg.shape[0],
                        rtg.shape[1],
                        1,
                        device=rtg.device,
                    ),
                )
            )
        rtg = F.pad(rtg, (0, 0, length - rtg.shape[1], 0))
        rtg[target_count:, -1] += torch.randn_like(rtg[target_count:, -1]) * 0.1
        if rtg_from_q and history_length > 1:
            # Preserve the research implementation's raw-Q RTG proposal convention.
            rtg[-1, -1] = (
                critic.q_min(states[-1:, -2], actions[-1:, -2]).flatten()
                - rewards[-1, -2] / self.config.scale
            )
        _, predictions = self(states, actions, rtg, steps)
        candidates = predictions[:, -1]
        values = critic.q_min(states[:, -1], candidates).flatten()
        index = (
            values.argmax() if selection == "greedy" else torch.multinomial(values.softmax(0), 1)
        )
        return candidates[index].reshape(-1)

    def config_dict(self):
        return asdict(self.config)
