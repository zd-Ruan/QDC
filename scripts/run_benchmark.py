"""Run the released task presets with independent training seeds."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("gym", "maze2d", "adroit", "all"), default="gym")
    parser.add_argument("--task", help="Run a single task key from configs/benchmarks.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without launching training"
    )
    args = parser.parse_args()
    presets = json.loads((ROOT / "configs" / "benchmarks.json").read_text(encoding="utf-8"))
    if args.task and args.task not in presets:
        parser.error(f"Unknown task {args.task}; inspect configs/benchmarks.json")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    tasks = {
        key: value
        for key, value in presets.items()
        if (key == args.task if args.task else args.suite == "all" or value["suite"] == args.suite)
    }
    for task, preset in tasks.items():
        for seed in args.seeds:
            command = [
                sys.executable,
                "-m",
                "qdc.train",
                "--seed",
                str(seed),
                "--data-dir",
                args.data_dir,
                "--output-dir",
                args.output_dir,
                "--device",
                args.device,
            ]
            for key, value in preset["arguments"].items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            print(f"{task}: {subprocess.list2cmdline(command)}", flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
