# Installation

## Core model and offline training

Use Python 3.10 as the reference interpreter. Install [PyTorch](https://pytorch.org/get-started/locally/) for the intended CPU or CUDA runtime, then run `python -m pip install -e .`. The supported dependency range is declared in `pyproject.toml`; it is not a frozen reconstruction of the 2024 training environment.

Training with `--eval-every 0`, checkpoint loading, and all CPU tests do not import Gym, D4RL, MuJoCo, MJRL, or Transformers.

## Legacy D4RL environments

The paper uses original D4RL environment IDs and datasets. Benchmark evaluation therefore targets the legacy [D4RL](https://github.com/Farama-Foundation/D4RL) stack. Newer Gymnasium / Minari replacements may have different dynamics, APIs, and datasets, so they should be treated as separate benchmarks.

Linux with Python 3.10 is the recommended simulator platform. Follow the upstream [mujoco-py installation instructions](https://github.com/openai/mujoco-py#install-mujoco), including MuJoCo 2.1 binaries and system OpenGL / compiler dependencies. Then:

```bash
python -m pip install "Cython<3"
python -m pip install -r requirements-d4rl.txt
python -m pip install -e .
```

Adroit environments also rely on MJRL, which the upstream D4RL installation declares as a dependency. No unconditional MJRL import is needed by QDC. Consult the D4RL installation guide if optional environment registration fails.

`requirements-d4rl.txt` intentionally identifies the optional legacy stack separately from the core package. Its D4RL source reference tracks upstream `master`, so it is not a reproducibility lockfile. Save `python -m pip freeze` and the D4RL revision with benchmark runs. This release's full simulator installation has not been validated in the local Windows environment.

## Existing author datasets

Existing files named like `D4RL/halfcheetah-medium-v2.pkl` can be used directly:

```bash
python -m qdc.train --env halfcheetah --dataset medium \
  --data-dir D4RL --eval-every 0
```

The `--env` and `--dataset` flags still select the task's RTG targets, reward scale, and episode limit when `--data-file` overrides its path.
