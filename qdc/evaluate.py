"""Evaluate a saved QDC model: python -m qdc.evaluate --help."""

import argparse
from pathlib import Path

from qdc.checkpoint import load_checkpoint
from qdc.data import TrajectoryDataset
from qdc.envs import make_env, task_spec
from qdc.evaluation import evaluate_policy
from qdc.utils import resolve_device, seed_everything, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--targets", nargs="+", type=float)
    parser.add_argument("--action-selection", choices=("sample", "greedy"))
    parser.add_argument("--num-candidates", type=int)
    parser.add_argument("--rtg-from-q", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    parser.add_argument("--legacy-config", type=Path)
    parser.add_argument(
        "--data-file", type=Path, help="Original dataset for historical weights only"
    )
    args = parser.parse_args(argv)
    if args.episodes < 1 or (args.num_candidates is not None and args.num_candidates < 1):
        parser.error("episodes and num-candidates must be positive")
    dataset = None
    if args.legacy_config:
        if not args.data_file:
            parser.error("--legacy-config requires --data-file to recover normalization")
        dataset = TrajectoryDataset.load(
            args.data_file,
            context_length=2,
            max_ep_len=1000,
            scale=1.0,
        )
    device = resolve_device(args.device)
    seed_everything(args.seed)
    actor, critic, payload = load_checkpoint(args.checkpoint, device, args.legacy_config, dataset)
    run_config = payload["run_config"]
    spec = task_spec(run_config["env"], run_config["dataset"])
    env = make_env(spec.env_id)
    selection = args.action_selection or run_config.get("action_selection", "sample")
    count = args.num_candidates or run_config.get("num_candidates", 50)
    rtg_from_q = (
        args.rtg_from_q if args.rtg_from_q is not None else run_config.get("rtg_from_q", True)
    )
    try:
        result = evaluate_policy(
            env,
            actor,
            critic,
            payload["state_mean"],
            payload["state_std"],
            args.targets or run_config.get("targets") or list(spec.targets),
            args.episodes,
            args.seed,
            selection,
            count,
            rtg_from_q,
            run_config.get("mode", "normal"),
        )
    finally:
        env.close()
    result.update(
        checkpoint=str(args.checkpoint),
        env_id=spec.env_id,
        seed=args.seed,
        action_selection=selection,
        num_candidates=count,
        rtg_from_q=rtg_from_q,
        training_seed=run_config.get("seed"),
        targets=args.targets or run_config.get("targets") or list(spec.targets),
        mode=run_config.get("mode", "normal"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(f"Return: {result['return_mean']:.2f} +/- {result['return_std']:.2f}")
    if "normalized_score_mean" in result:
        print(
            f"D4RL score: {result['normalized_score_mean']:.2f} +/- {result['normalized_score_std']:.2f}"
        )
    print(f"Saved evaluation to {args.output}")


if __name__ == "__main__":
    main()
