# QDC：Q-value Regularized Decision ConvFormer

论文 **[Q-value Regularized Decision ConvFormer for Offline Reinforcement Learning](https://arxiv.org/abs/2409.08062)** 的代码。

**Teng Yan、Zhendong Ruan、Yaobang Cai、Yu Han、Wenxian Li、Yang Zhang**

[English](README.md) · [代码仓库](https://github.com/zd-Ruan/QDC) · [安装说明](docs/installation.md) · [实现与复现说明](docs/reproducibility.md)

## 方法

QDC 将 Decision ConvFormer 的局部卷积建模与 Q 值正则化结合，用于离线强化学习。模型以回报目标（RTG）、状态和历史动作为输入，通过因果卷积捕获轨迹中的局部依赖；双 Q 网络提供价值引导，帮助策略学习更高价值的动作并提高轨迹拼接能力。

本项目包含局部卷积策略、双 Q 网络、多步 Bellman 目标、目标策略的指数移动平均更新，以及训练、数据准备和独立评估入口，覆盖论文中的 D4RL Gym、Maze2D 和 Adroit 任务。

## 安装

建议使用 Python 3.10，并根据硬件安装合适的 PyTorch。核心训练代码只依赖 PyTorch 和 NumPy，不需要 Transformers：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Windows 的激活命令为 `.venv\Scripts\activate`。D4RL 环境评估还需要旧版 Gym 和 MuJoCo，建议在 Linux 中运行完整基准实验，具体见[安装说明](docs/installation.md)。

## 数据准备

安装 D4RL 后执行：

```bash
python -m qdc.prepare_data --env hopper --dataset medium --output-dir data
```

默认生成 `data/hopper-medium-v2.pkl`。每条轨迹包括 `observations`、`actions`、`rewards` 和 `terminals`，加载时也接受 `dones`。现有数据可通过 `--data-dir D4RL` 或 `--data-file 路径` 指定，无需重新下载。只加载可信来源的 pickle 文件。源码发布包不包含数据集和历史权重。

## 训练与评估

```bash
python -m qdc.train --env hopper --dataset medium --data-dir data \
  --seed 123 --eta 1.0 --context-length 8 --device auto
```

已有轨迹文件时，可添加 `--eval-every 0`，在不安装仿真环境的情况下训练。`--device auto` 自动选择 CUDA 或 CPU。每次实验保存配置、运行环境版本、CSV 指标以及含归一化参数的 `last.pt`；开启环境评估时还会保存 `best.pt`。这些权重用于推理，不包含精确续训所需的优化器状态。

将下面的路径替换为实际实验目录：

```bash
python -m qdc.evaluate --checkpoint runs/your-run/best.pt \
  --episodes 10 --seed 123 --output evaluation.json
```

评估报告原始回报和 **0–100 标度**的 D4RL 标准化得分。默认保留研究代码中按 `softmax(Q)` 采样候选动作的方式；使用 `--action-selection greedy` 可以改为选择最大 Q 值候选。

## 批量实验

```bash
python scripts/run_benchmark.py --suite gym --seeds 0 1 2 --dry-run
python scripts/run_benchmark.py --suite maze2d --seeds 0 1 2
```

`configs/benchmarks.json` 中的参数整理自作者提供的历史实验配置，可作为新实验的起点；目前尚未重新验证它们能完整复现论文表格。三个训练种子应分别独立训练和评估，不能用同一模型的多次 rollout 代替。可使用 `scripts/summarize_results.py` 汇总各模型的评估 JSON，分别报告均值、模型间标准差和标准误差。

代码保留了原实现中的状态预测辅助损失和 Q 值采样规则；整理中涉及数值行为的修正已列在[实现与复现说明](docs/reproducibility.md)中。本次发布未重新跑完 D4RL 基准。

## 测试与打包

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
ruff format --check .
python scripts/package_release.py
```

发布包生成于 `dist/qdc-source.zip`，仅包含明确列出的源码与文档，不包含本地实验记录、数据、权重和备份。目录结构及论文 BibTeX 见 [English README](README.md)。

## 致谢与许可

本实现基于 [QT](https://github.com/charleshsc/QT) 改造，并使用 [Decision ConvFormer](https://github.com/beanie00/Decision-ConvFormer) 的模型思路，感谢原作者的开源工作。

项目沿用 [Apache-2.0](LICENSE)；上游来源及第三方许可保留于 [NOTICE](NOTICE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 `third_party/`。
