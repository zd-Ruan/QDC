# Setup

Use Python 3.10 and install [PyTorch](https://pytorch.org/get-started/locally/) for your CPU or CUDA runtime:

```bash
python -m pip install -e .
```

Offline training with `--eval-every 0` needs only PyTorch and NumPy. Existing trajectory files can be passed with `--data-dir` or `--data-file`; load pickle files only from trusted sources.

## D4RL evaluation

For the original benchmark environments, use Linux and follow the [MuJoCo 2.1 / mujoco-py setup](https://github.com/openai/mujoco-py#install-mujoco), then install:

```bash
python -m pip install "Cython<3"
python -m pip install -r requirements-d4rl.txt
python -m pip install -e .
```

See [D4RL](https://github.com/Farama-Foundation/D4RL) for simulator dependencies. The optional stack tracks upstream D4RL rather than a frozen environment; record installed versions for experiments. Full simulator setup has not been validated on Windows.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
python scripts/package_release.py
```

The source archive excludes local datasets, weights, logs, and backups.
