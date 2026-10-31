import json

import pytest

from scripts.summarize_results import summarize


def test_summary_reports_variation_between_training_seeds(tmp_path):
    files = []
    for seed, score in enumerate((80.0, 90.0, 100.0)):
        path = tmp_path / f"seed{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "env_id": "hopper-medium-v2",
                    "checkpoint": f"run{seed}/best.pt",
                    "training_seed": seed,
                    "normalized_score_mean": score,
                    "episodes": 10,
                }
            )
        )
        files.append(path)
    result = summarize(files)["hopper-medium-v2"]
    assert result["mean"] == 90
    assert result["std_between_models"] == 10
    assert result["standard_error"] == pytest.approx(10 / 3**0.5)
    assert result["rollouts"] == 30


def test_summary_rejects_repeated_checkpoint(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"env_id": "hopper-medium-v2", "checkpoint": "run/best.pt"}))
    with pytest.raises(ValueError, match="Repeated checkpoint"):
        summarize([path, path])
