"""Train QDC on trajectory files: python -m qdc.train --help."""

import argparse
import csv
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from qdc.checkpoint import save_checkpoint
from qdc.data import TrajectoryDataset
from qdc.envs import make_env, task_spec
from qdc.evaluation import evaluate_policy
from qdc.models import QDC, Critic, ModelConfig
from qdc.trainer import TrainConfig, Trainer
from qdc.utils import resolve_device, runtime_versions, seed_everything, write_json


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="hopper")
    parser.add_argument("--dataset", default="medium")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--data-file", type=Path, help="Override the task's default trajectory file"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--context-length", "--K", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--bc-weight", type=float, default=1.0)
    parser.add_argument("--state-loss-weight", type=float, default=1.0)
    parser.add_argument("--grad-norm", type=float, default=9.0)
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--steps-per-iter", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=1, help="0 disables simulator evaluation")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--lr-decay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory-fraction", type=float, default=1.0)
    parser.add_argument("--mode", choices=("normal", "delayed"), default="normal")
    parser.add_argument("--scale", type=float)
    parser.add_argument("--targets", type=float, nargs="+")
    parser.add_argument("--action-selection", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--num-candidates", type=int, default=50)
    parser.add_argument("--rtg-from-q", action=argparse.BooleanOptionalAction, default=True)
    return parser


def run(args):
    if (
        args.steps_per_iter < 1
        or args.eval_every < 0
        or args.eval_episodes < 1
        or args.batch_size < 1
    ):
        raise ValueError(
            "Steps, episodes, and batch size must be positive; eval-every must be >= 0"
        )
    if args.num_candidates < 1:
        raise ValueError("num-candidates must be positive")
    spec = task_spec(args.env, args.dataset)
    device = resolve_device(args.device)
    seed_everything(args.seed)
    scale = args.scale if args.scale is not None else spec.scale
    dataset = TrajectoryDataset.load(
        args.data_file or args.data_dir / f"{spec.env_id}.pkl",
        context_length=args.context_length,
        max_ep_len=spec.max_ep_len,
        scale=scale,
        seed=args.seed,
        fraction=args.trajectory_fraction,
        mode=args.mode,
    )
    model_config = ModelConfig(
        dataset.state_dim,
        dataset.action_dim,
        args.hidden_size,
        args.n_layers,
        args.window_size,
        args.context_length,
        spec.max_ep_len,
        args.dropout,
        scale,
    )
    train_config = TrainConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        discount=args.discount,
        tau=args.tau,
        eta=args.eta,
        bc_weight=args.bc_weight,
        state_loss_weight=args.state_loss_weight,
        grad_norm=args.grad_norm,
        lr_decay=args.lr_decay,
        max_iters=args.max_iters,
    )
    actor = QDC(model_config).to(device)
    critic = Critic(dataset.state_dim, dataset.action_dim, args.hidden_size).to(device)
    trainer = Trainer(actor, critic, train_config)
    env = make_env(spec.env_id) if args.eval_every else None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = args.output_dir / f"qdc-{spec.env_id}-seed{args.seed}-{stamp}"
    run_config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    run_config.update(
        env_id=spec.env_id, targets=args.targets or list(spec.targets), device=str(device)
    )
    fieldnames = [
        "iteration",
        "step",
        "elapsed_seconds",
        "actor_loss",
        "critic_loss",
        "action_loss",
        "state_loss",
        "bc_loss",
        "q_loss",
        "return_mean",
        "normalized_score_mean",
    ]
    best_return = -np.inf
    started = time.monotonic()
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(
            run_dir / "config.json",
            {
                "model": asdict(model_config),
                "training": asdict(train_config),
                "run": run_config,
                "runtime": runtime_versions(),
                "data": dataset.summary(),
            },
        )
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            print(f"Run directory: {run_dir}", flush=True)
            for iteration in range(1, args.max_iters + 1):
                metrics = [
                    trainer.train_step(dataset.sample(args.batch_size, device))
                    for _ in range(args.steps_per_iter)
                ]
                row = {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]}
                trainer.end_iteration()
                row.update(
                    iteration=iteration,
                    step=trainer.step,
                    elapsed_seconds=time.monotonic() - started,
                )
                if env is not None and (
                    iteration % args.eval_every == 0 or iteration == args.max_iters
                ):
                    scores = evaluate_policy(
                        env,
                        actor,
                        trainer.target_critic,
                        dataset.state_mean,
                        dataset.state_std,
                        run_config["targets"],
                        args.eval_episodes,
                        args.seed,
                        args.action_selection,
                        args.num_candidates,
                        args.rtg_from_q,
                        args.mode,
                    )
                    row["return_mean"] = scores["return_mean"]
                    row["normalized_score_mean"] = scores.get("normalized_score_mean", "")
                    if scores["return_mean"] > best_return:
                        best_return = scores["return_mean"]
                        save_checkpoint(
                            run_dir / "best.pt", trainer, dataset, run_config, iteration
                        )
                        write_json(
                            run_dir / "best_evaluation.json", {"iteration": iteration, **scores}
                        )
                save_checkpoint(run_dir / "last.pt", trainer, dataset, run_config, iteration)
                writer.writerow(row)
                stream.flush()
                print(json.dumps(row, allow_nan=False), flush=True)
    finally:
        if env is not None:
            env.close()
    return run_dir


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f"QDC: {exc}\n")


if __name__ == "__main__":
    main()
