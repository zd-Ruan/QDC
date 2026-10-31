# Implementation notes

This release retains the QDC architecture and optimization scheme. Full D4RL benchmark scores have not been rerun after cleanup; the 23 [task presets](../configs/benchmarks.json) are starting points, not verified paper configurations.

## Research conventions

- The actor loss combines action MSE, state prediction MSE, and normalized Q regularization. Set `--state-loss-weight 0` to disable the auxiliary state loss, which is not explicit in the paper's action-only formula.
- Q normalization divides one critic's mean value by the other critic's detached mean absolute predicted-action value, with a small denominator floor.
- Inference samples 50 RTG-conditioned candidates using `softmax(min(Q1, Q2))`; `--action-selection greedy` selects the largest Q value as described in the paper.
- The historical Q-derived RTG proposal uses `Q(s_prev, a_prev) - reward_prev / scale`, leaving Q unscaled. Disable it with `--no-rtg-from-q`.

## Cleanup changes

Device handling, duplicate forward passes, short-window sampling, and dataset export were corrected. Targets now run in evaluation mode without gradients; Bellman recursion handles terminals without modifying rewards. The first-step Q proposal skips padded history. Iteration limits are explicit, sampling uses dedicated seeded generators, and scores use the 0-100 scale. These changes can affect historical training results.

## Evaluation and checkpoints

Use independent training seeds and report ten rollouts per model. `scripts/summarize_results.py` reports between-model standard deviation and standard error separately. Checkpoints include normalization and model settings, but do not support exact training resume.

Historical QDC weights require their matching configuration and dataset:

```bash
python -m qdc.evaluate --checkpoint path/to/epoch_51.pth \
  --legacy-config path/to/variant.json --data-file D4RL/halfcheetah-medium-v2.pkl
```

Transformer, SAR, removed-action-embedding, and no-Q-inference ablations are unsupported. CPU tests validate functionality; they do not establish benchmark reproduction.
