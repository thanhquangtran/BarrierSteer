import json

from barriersteer_pipeline.harmbench.stages.circuit_breaker_training import (
    CircuitBreakerTrainingStage,
)


def test_circuit_breaker_dataset_normalizes_list_prompts(tmp_path):
    base = tmp_path / "results" / "PAP-top5" / "qwen_1.5b"
    (base / "results").mkdir(parents=True)
    (base / "completions").mkdir()
    (base / "results" / "qwen_1.5b_20260101.json").write_text(
        json.dumps(
            {
                "B1": [{"label": 1}],
                "B2": [{"label": 0}],
            }
        )
    )
    (base / "completions" / "qwen_1.5b_20260101.json").write_text(
        json.dumps(
            {
                "B1": [
                    {
                        "test_case": ["unsafe rationale", "unsafe prompt"],
                        "generation": "bad",
                    }
                ],
                "B2": [{"test_case": "safe prompt", "generation": "ok"}],
            }
        )
    )

    stage = CircuitBreakerTrainingStage(
        {"pipeline": {"base_save_dir": str(tmp_path / "results")}},
        str(tmp_path),
        str(tmp_path),
    )
    dataset_path = stage._build_training_dataset(
        model_name="qwen_1.5b",
        attack_methods=["PAP-top5"],
        training_cfg={},
        output_dir=str(tmp_path),
        seed=0,
    )

    payload = json.loads(
        (tmp_path / "harmbench_circuit_breaker_dataset.json").read_text()
    )
    assert dataset_path == str(tmp_path / "harmbench_circuit_breaker_dataset.json")
    assert payload["stats"]["unsafe_total"] == 1
    assert payload["unsafe_train"][0]["prompt"] == "unsafe rationale\nunsafe prompt"
