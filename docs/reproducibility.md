# Implementation and reproducibility

## Scope of the release

This is a cleaned release of the author's QDC research code for [arXiv:2409.08062](https://arxiv.org/abs/2409.08062). It focuses on the paper's Gym locomotion, Maze2D, and Adroit tasks. Unused Transformer baselines and incomplete non-paper ablation branches were removed from the public source tree.

The architecture and principal optimization scheme are retained, but this is not a bit-for-bit frozen copy of the historical training program. Published benchmark scores have not been rerun after cleanup. No bundled checkpoint or task preset is certified as the final configuration behind a paper table.

## Retained research conventions

- Interleaved RTG, state, and action tokens pass through separate depthwise local filters and residual feedforward blocks. Parameter names retain the historical `transformer` prefix solely to support loading existing QDC weights; no attention or Transformers dependency remains.
- The actor objective is `bc_weight * (action_mse + state_loss_weight * state_mse) + eta * q_loss`. Both weights default to 1. The state prediction auxiliary loss is present in the author's code but not explicitly included in the paper's action-only behavior cloning formula. Set `--state-loss-weight 0` to disable it as an ablation.
- Q regularization uses one randomly chosen critic in the numerator and the other critic's detached mean absolute predicted-action value in the denominator. This preserves the code's convention rather than replacing it with the paper's simplified normalization expression. A small denominator floor prevents division by zero.
- Inference generates 50 RTG-conditioned candidates by default and samples actions using `softmax(min(Q1, Q2))`. The paper describes selecting the largest Q value. `--action-selection greedy` exposes that alternative explicitly.
- One optional RTG proposal retains the original expression `Q(s_previous, a_previous) - reward_previous / scale`. In particular, the Q term itself is not divided by the reward scale; this is a historical convention, not a corrected interpretation of RTG units. Use `--no-rtg-from-q` to remove this proposal for an ablation.
- State statistics are computed over all provided trajectories before selecting the requested top-return fraction. RTG includes rewards beyond the sampled window. States, including zero padding, follow the original normalization convention.

## Cleanup changes that can affect numerical results

- The duplicate backbone forward pass was removed. The old program discarded the first result, consuming dropout randomness twice in training.
- All tensors follow the selected device. The old convolution output unconditionally moved to CUDA.
- Target policy and critic forward passes run without gradients and in evaluation mode. The original target policy remained in training mode and applied dropout.
- Multi-step targets use an explicit backward Bellman recursion, exclude the final state's reward, stop bootstrapping at true terminals, and distinguish timeouts from terminals. They no longer modify the input rewards in place. The old implementation used different terminal gating and padding-specific discount expressions.
- Sampling avoids windows containing only one real timestep, and trajectories shorter than two steps are excluded from training selection. Such windows previously produced empty transition losses.
- The Q-derived RTG proposal is skipped at the first real environment step instead of evaluating a critic on synthetic padded history.
- Training defaults to the requested number of iterations; the previous implicit stop after iteration 51 was removed. There is no automatic early-stop heuristic.
- NumPy's dedicated seeded generator and PyTorch randomness replace mixed global Python / NumPy sampling. Fixed seed values therefore do not imply identical historical minibatches.
- Dataset export retains a final incomplete fragment that historical scripts discarded.
- D4RL scores are consistently reported on the 0-100 scale. Historical files mixed normalized fractions with percentages.

## Task presets

`configs/benchmarks.json` contains 23 task presets recovered from historical author configurations (9 Gym, 8 Maze2D, 6 Adroit). They are practical starting points, not a verified reconstruction of every published experiment. Training budgets are explicit and may differ between tasks. All remaining settings use the CLI defaults, which are recorded in each run's `config.json`.

For a benchmark report, use at least three independent training seeds and evaluate each selected checkpoint on ten trajectories, as in the paper's 30-rollout protocol. Record the model-selection rule (this release saves the highest evaluation mean return), rollout seeds, action-selection rule, simulator versions, and training budget. The result summarizer reports between-model standard deviation and standard error separately; a single run cannot recover these statistics.

## Checkpoints

New checkpoints contain actor weights, target critic weights, model settings, task arguments, state statistics, and iteration metadata. They intentionally do not contain optimizer or random-generator state and do not implement exact training resume.

Historical `epoch_*.pth` files contain only weights and epoch metadata. To evaluate a standard historical QDC checkpoint, provide the matching `variant.json` and original trajectory dataset:

```bash
python -m qdc.evaluate --checkpoint path/to/epoch_51.pth \
  --legacy-config path/to/variant.json \
  --data-file D4RL/halfcheetah-medium-v2.pkl --episodes 10
```

The loader strictly validates parameter names and tensor dimensions. Old Transformer, SAR, removed-action-embedding, and no-Q-inference ablation checkpoints are unsupported. This importer supports historical architecture weights; it does not claim identical old rollout randomness or metrics.

## Validation boundary

CPU tests exercise causal action prediction, candidate selection on short histories, trajectory sampling, hand-calculated Bellman targets, a real gradient update, checkpoint round trips, training without simulator imports, and rollouts against both environment step API forms. Passing these tests does not establish paper benchmark reproduction. CUDA training and full D4RL rollout evaluation require separate benchmark validation.
