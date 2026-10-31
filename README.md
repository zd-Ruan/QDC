# QDC: Q-value Regularized Decision ConvFormer

Code for **[Q-value Regularized Decision ConvFormer for Offline Reinforcement Learning](https://arxiv.org/abs/2409.08062)**.

**Teng Yan, Zhendong Ruan, Yaobang Cai, Yu Han, Wenxian Li, Yang Zhang**

[Paper](https://arxiv.org/abs/2409.08062) · [Code](https://github.com/zd-Ruan/QDC) · [中文说明](README.zh-CN.md) · [Installation](docs/installation.md) · [Implementation notes](docs/reproducibility.md)

## Overview

QDC combines a return-conditioned Decision ConvFormer policy with Q-value regularization for offline reinforcement learning. Causal local convolutional filters capture dependencies in state-action trajectories, while twin Q networks guide the policy toward higher-value actions and improve trajectory stitching.

The implementation includes:

- A convolutional policy with modality-specific local filters and residual feedforward blocks.
- Multi-step Bellman targets, twin critics, and a target policy updated by exponential moving averages.
- Behavior cloning with normalized Q-value regularization.
- Q-guided action selection from multiple return-to-go (RTG) candidates.
- Training, checkpoint evaluation, and dataset preparation for D4RL Gym, Maze2D, and Adroit tasks.

## Installation

Use Python 3.10 and install a PyTorch build appropriate for your hardware. The core model and offline trainer require only PyTorch and NumPy:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate with `.venv\Scripts\activate`. D4RL rollout evaluation uses the legacy Gym / MuJoCo stack; Linux is the recommended benchmark environment. Follow [the simulator setup instructions](docs/installation.md) before preparing datasets or evaluating policies. Hugging Face Transformers is not required.

## Prepare data

After installing D4RL, export a task into a list of trajectory dictionaries:

```bash
python -m qdc.prepare_data --env hopper --dataset medium --output-dir data
```

Each trajectory contains `observations`, `actions`, `rewards`, and `terminals` (`dones` is also accepted when loading existing files). Files are named using the D4RL environment ID, for example `data/hopper-medium-v2.pkl`.

Existing trajectory files can be supplied with `--data-dir` or `--data-file`. Pickle files must come from a trusted source. Datasets and pretrained weights are not bundled with this source release.

## Train

```bash
python -m qdc.train \
  --env hopper --dataset medium --data-dir data \
  --seed 123 --eta 1.0 --grad-norm 9.0 \
  --context-length 8 --device auto --output-dir runs
```

To train from an existing trajectory file without installing a simulator:

```bash
python -m qdc.train --env hopper --dataset medium \
  --data-file data/hopper-medium-v2.pkl --eval-every 0
```

`--device auto` selects CUDA when available and CPU otherwise. Every run saves its full configuration, runtime versions, training metrics, and a self-contained `last.pt` checkpoint. When rollout evaluation is enabled, it also saves `best.pt` and the corresponding evaluation results. Checkpoints contain model settings and state normalization statistics, so evaluation does not need the original dataset. They are inference checkpoints; optimizer state and exact training resume are not included.

Run `python -m qdc.train --help` for all options. `train.py` is an equivalent convenience entry point.

## Evaluate

Replace the checkpoint path below with one produced by training:

```bash
python -m qdc.evaluate --checkpoint runs/your-run/best.pt \
  --episodes 10 --seed 123 --output evaluation.json
```

Evaluation reports episode returns and D4RL normalized scores on the conventional **0-100 scale**. The default inference rule samples candidates according to `softmax(Q)`, matching the supplied research implementation. Use `--action-selection greedy` to select the largest Q value deterministically at the selection step; candidate RTG perturbations still use the specified random seed.

## Benchmark runs

Task presets are provided in [`configs/benchmarks.json`](configs/benchmarks.json). They were recovered from the author's historical experiment configurations and are starting points for this cleaned release. They have not been revalidated as a complete set of configurations reproducing the published tables.

```bash
# Inspect commands first.
python scripts/run_benchmark.py --suite gym --seeds 0 1 2 --dry-run

# Train three independent models for each Maze2D task.
python scripts/run_benchmark.py --suite maze2d --seeds 0 1 2 --data-dir data

# Or choose a single task.
python scripts/run_benchmark.py --task walker2d-medium-replay --seeds 0 1 2
```

Evaluate each independently trained model, then aggregate its evaluation JSON:

```bash
python scripts/summarize_results.py evaluation-seed0.json evaluation-seed1.json evaluation-seed2.json
```

The summary reports the mean across models, between-model standard deviation, and standard error separately. Ten rollouts from one model do not replace three independent training runs. See [implementation and reproducibility notes](docs/reproducibility.md) for retained research conventions and changes introduced during cleanup. Full D4RL training results have not been rerun for this release.

## Repository structure

```text
qdc/
  models.py          Convolutional policy and twin Q networks
  data.py            Trajectory validation, normalization, and sampling
  trainer.py         Multi-step Q learning and policy optimization
  train.py           Training command
  evaluation.py      Rollout evaluation
  evaluate.py        Checkpoint evaluation command
  checkpoint.py      Checkpoint I/O and historical weight loading
  prepare_data.py    D4RL dataset export
configs/             Task presets
scripts/             Benchmark runs, result aggregation, and source packaging
tests/               CPU tests without simulator dependencies
docs/                Installation and implementation notes
```

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
ruff format --check .
python scripts/package_release.py
```

The packaging command creates `dist/qdc-source.zip` from an explicit source allowlist. Local datasets, checkpoints, experiment logs, environments, and backups are excluded.

## Citation

```bibtex
@article{yan2024qdc,
  title={Q-value Regularized Decision ConvFormer for Offline Reinforcement Learning},
  author={Yan, Teng and Ruan, Zhendong and Cai, Yaobang and Han, Yu and Li, Wenxian and Zhang, Yang},
  journal={arXiv preprint arXiv:2409.08062},
  year={2024}
}
```

## Acknowledgments and license

This implementation is adapted from [QT](https://github.com/charleshsc/QT) and builds on [Decision ConvFormer](https://github.com/beanie00/Decision-ConvFormer). We thank the authors for making their work available.

Released under [Apache-2.0](LICENSE), with upstream attribution and third-party license notices preserved in [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and [`third_party/`](third_party/).
