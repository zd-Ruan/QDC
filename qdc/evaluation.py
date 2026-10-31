"""Return-conditioned rollouts and consistent 0-100 D4RL score reporting."""

import numpy as np
import torch


def reset_env(env, seed):
    try:
        result = env.reset(seed=seed)
    except TypeError:
        env.seed(seed)
        result = env.reset()
    return result[0] if isinstance(result, tuple) else result


@torch.no_grad()
def evaluate_policy(
    env,
    actor,
    critic,
    state_mean,
    state_std,
    targets,
    episodes=10,
    seed=123,
    selection="sample",
    num_candidates=50,
    rtg_from_q=True,
    mode="normal",
):
    if episodes < 1 or not targets:
        raise ValueError("episodes and targets must be non-empty")
    actor.eval()
    critic.eval()
    device = next(actor.parameters()).device
    mean = torch.as_tensor(state_mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(state_std, dtype=torch.float32, device=device)
    returns, lengths = [], []
    for episode in range(episodes):
        initial = reset_env(env, seed + episode)
        states = torch.as_tensor(np.asarray(initial), device=device, dtype=torch.float32).reshape(
            1, -1
        )
        actions = torch.zeros(1, actor.config.action_dim, device=device)
        rewards = torch.zeros(1, 1, device=device)
        rtg = torch.as_tensor(targets, device=device, dtype=torch.float32).reshape(-1, 1)
        rtg = rtg / actor.config.scale
        timesteps = torch.zeros(1, dtype=torch.long, device=device)
        total = 0.0
        for step in range(actor.config.max_ep_len):
            action = actor.get_action(
                critic,
                (states - mean) / std,
                actions,
                rewards,
                rtg,
                timesteps,
                num_candidates=num_candidates,
                selection=selection,
                rtg_from_q=rtg_from_q,
            )
            result = env.step(action.cpu().numpy())
            if len(result) == 5:
                observation, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                observation, reward, done, _ = result
            total += float(reward)
            if done or step + 1 == actor.config.max_ep_len:
                break
            actions[-1] = action
            rewards[-1, 0] = float(reward)
            actions = torch.cat((actions, torch.zeros_like(actions[:1])))
            rewards = torch.cat((rewards, torch.zeros_like(rewards[:1])))
            states = torch.cat(
                (
                    states,
                    torch.as_tensor(
                        np.asarray(observation),
                        dtype=torch.float32,
                        device=device,
                    ).reshape(1, -1),
                )
            )
            next_return = (
                rtg[:, -1:] if mode == "delayed" else rtg[:, -1:] - reward / actor.config.scale
            )
            rtg = torch.cat((rtg, next_return), dim=1)
            timesteps = torch.cat((timesteps, timesteps.new_tensor([step + 1])))
        returns.append(total)
        lengths.append(step + 1)
    result = {
        "returns": returns,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "length_mean": float(np.mean(lengths)),
        "episodes": episodes,
    }
    if hasattr(env, "get_normalized_score"):
        scores = [float(env.get_normalized_score(value)) * 100 for value in returns]
        result.update(
            normalized_scores=scores,
            normalized_score_mean=float(np.mean(scores)),
            normalized_score_std=float(np.std(scores)),
        )
    return result
