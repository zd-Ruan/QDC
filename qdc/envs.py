"""D4RL environment settings used by the paper; optional simulator imports are lazy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    env_id: str
    targets: tuple[float, ...]
    scale: float
    max_ep_len: int = 1000


def task_spec(env, dataset):
    locomotion = {
        "halfcheetah": ((12000.0, 9000.0, 6000.0), 1000.0),
        "hopper": ((3600.0, 1800.0), 1000.0),
        "walker2d": ((5000.0, 4000.0, 2500.0), 1000.0),
    }
    adroit = {
        "pen": ((12000.0, 6000.0), 1000.0),
        "hammer": ((12000.0, 6000.0, 3000.0), 1000.0),
        "door": ((2000.0, 1000.0, 500.0), 100.0),
    }
    if env in locomotion and dataset in ("medium", "medium-replay", "medium-expert"):
        targets, scale = locomotion[env]
        return TaskSpec(f"{env}-{dataset}-v2", targets, scale)
    if env in adroit and dataset in ("human", "cloned"):
        targets, scale = adroit[env]
        return TaskSpec(f"{env}-{dataset}-v1", targets, scale)
    if env == "maze2d" and dataset in (
        "open",
        "umaze",
        "medium",
        "large",
        "open-dense",
        "umaze-dense",
        "medium-dense",
        "large-dense",
    ):
        version = 0 if dataset.startswith("open") else 1
        return TaskSpec(
            f"maze2d-{dataset}-v{version}", (300.0, 200.0, 150.0, 100.0, 50.0, 20.0), 10.0
        )
    raise ValueError(f"Unsupported paper task: {env}/{dataset}")


def make_env(env_id):
    try:
        import gym

        import d4rl  # noqa: F401 - registers benchmark environments
    except ImportError as exc:
        raise RuntimeError(
            "D4RL evaluation requires the optional simulator stack. "
            "See docs/installation.md; offline training works with --eval-every 0."
        ) from exc
    return gym.make(env_id)
