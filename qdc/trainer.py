# SPDX-License-Identifier: Apache-2.0
# Q regularization and target-policy updates adapted from QT. See NOTICE.
"""Twin-Q multi-step learning and regularized convolutional policy optimization."""

import copy
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class TrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    discount: float = 0.99
    tau: float = 0.005
    eta: float = 1.0
    bc_weight: float = 1.0
    state_loss_weight: float = 1.0
    grad_norm: float = 9.0
    ema_decay: float = 0.995
    ema_start: int = 1000
    ema_every: int = 5
    lr_decay: bool = True
    max_iters: int = 500

    def __post_init__(self):
        if self.learning_rate <= 0 or self.max_iters < 1 or self.ema_every < 1:
            raise ValueError("learning_rate, max_iters, and ema_every must be positive")
        if not 0 <= self.discount <= 1 or not 0 <= self.tau <= 1 or not 0 <= self.ema_decay <= 1:
            raise ValueError("discount, tau, and ema_decay must be in [0, 1]")
        if min(self.eta, self.bc_weight, self.state_loss_weight, self.weight_decay) < 0:
            raise ValueError("Loss weights and weight_decay must be nonnegative")


@torch.no_grad()
def multi_step_targets(rewards, terminals, bootstrap, discount):
    """Targets for each position, bootstrapping from the final state of the window.

    ``rewards[:, -1]`` belongs to an action after that final state and is excluded.
    Terminal transitions stop bootstrapping; time-limit truncations do not.
    """
    targets = torch.zeros_like(rewards)
    targets[:, -1] = bootstrap
    running = bootstrap
    for index in range(rewards.shape[1] - 2, -1, -1):
        running = rewards[:, index] + discount * (1 - terminals[:, index]) * running
        targets[:, index] = running
    return targets


class Trainer:
    def __init__(self, actor, critic, config: TrainConfig):
        self.actor, self.critic, self.config = actor, critic, config
        self.target_actor = copy.deepcopy(actor).requires_grad_(False).eval()
        self.target_critic = copy.deepcopy(critic).requires_grad_(False).eval()
        self.actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer,
            T_max=config.max_iters,
        )
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer,
            T_max=config.max_iters,
        )
        self.step = 0

    def train_step(self, batch):
        config = self.config
        self.actor.train()
        self.critic.train()
        transitions = batch.mask[:, :-1] & batch.mask[:, 1:]
        if not transitions.any():
            raise ValueError(
                "Batch has no valid transitions; use trajectories with at least two steps"
            )
        with torch.no_grad():
            _, next_actions = self.target_actor(
                batch.states,
                batch.actions,
                batch.returns_to_go,
                batch.timesteps,
            )
            bootstrap = self.target_critic.q_min(batch.states[:, -1], next_actions[:, -1])
            targets = multi_step_targets(batch.rewards, batch.terminals, bootstrap, config.discount)
        q1, q2 = self.critic(batch.states, batch.actions)
        critic_loss = F.mse_loss(
            q1[:, :-1][transitions], targets[:, :-1][transitions]
        ) + F.mse_loss(q2[:, :-1][transitions], targets[:, :-1][transitions])
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if config.grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), config.grad_norm)
        self.critic_optimizer.step()

        # The critic remains differentiable with respect to actions, but its weights are frozen.
        self.critic.requires_grad_(False)
        try:
            state_preds, action_preds = self.actor(
                batch.states,
                batch.actions,
                batch.returns_to_go,
                batch.timesteps,
            )
            action_loss = F.mse_loss(action_preds[batch.mask], batch.actions[batch.mask])
            state_loss = F.mse_loss(
                state_preds[:, :-1][transitions], batch.states[:, 1:][transitions]
            )
            bc_loss = action_loss + config.state_loss_weight * state_loss
            policy_q1, policy_q2 = self.critic(batch.states[batch.mask], action_preds[batch.mask])
            if torch.rand((), device=batch.states.device) > 0.5:
                numerator, denominator = policy_q1, policy_q2
            else:
                numerator, denominator = policy_q2, policy_q1
            q_loss = -numerator.mean() / denominator.abs().mean().detach().clamp_min(1e-6)
            actor_loss = config.bc_weight * bc_loss + config.eta * q_loss
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if config.grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), config.grad_norm)
            self.actor_optimizer.step()
        finally:
            self.critic.requires_grad_(True)

        with torch.no_grad():
            if self.step > config.ema_start and self.step % config.ema_every == 0:
                for target, source in zip(self.target_actor.parameters(), self.actor.parameters()):
                    target.lerp_(source, 1 - config.ema_decay)
            for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.lerp_(source, config.tau)
        self.step += 1
        return {
            name: float(value.detach())
            for name, value in {
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "action_loss": action_loss,
                "state_loss": state_loss,
                "bc_loss": bc_loss,
                "q_loss": q_loss,
            }.items()
        }

    def end_iteration(self):
        if self.config.lr_decay:
            self.actor_scheduler.step()
            self.critic_scheduler.step()
