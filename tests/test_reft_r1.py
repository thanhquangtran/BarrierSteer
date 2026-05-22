import json

import torch
import torch.nn as nn

from barriersteer_pipeline.harmbench.stages.reft_r1_training import ReFTR1TrainingStage
from steer.reft_r1 import (
    ReFTR1Config,
    ReFTR1Intervention,
    attach_reft_r1_hook,
    load_reft_r1_intervention,
)
from steer.reft_r1.trainer import PromptTargetDataset, collate_prompt_targets


class _Block(nn.Module):
    def forward(self, x):
        return (x,)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block(), _Block()])

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)[0]
        return x


def test_reft_r1_hook_changes_hidden_states():
    model = _TinyModel()
    cfg = ReFTR1Config(target_layer=0, top_k=1, beta=1.0)
    intervention = ReFTR1Intervention(
        hidden_size=3, config=cfg, vector=torch.tensor([1.0, 0.0, 0.0])
    )
    handle = attach_reft_r1_hook(model, intervention)
    try:
        x = torch.tensor([[[2.0, 0.0, 0.0], [0.5, 0.0, 0.0]]])
        y = model(x)
        assert y[0, 0, 0] > x[0, 0, 0]
        assert intervention.last_scores is not None
    finally:
        handle.remove()


def test_reft_r1_checkpoint_roundtrip(tmp_path):
    ckpt = tmp_path / "reft_r1_vector.pt"
    torch.save(
        {
            "vector": torch.tensor([0.0, 3.0]),
            "config": {"target_layer": 1, "top_k": 2, "beta": 0.5},
        },
        ckpt,
    )
    intervention = load_reft_r1_intervention(str(ckpt))
    assert intervention.config.target_layer == 1
    assert intervention.config.top_k == 2
    assert torch.isclose(intervention.unit_vector.norm(), torch.tensor(1.0))


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        ids = [1] if add_special_tokens else []
        ids += [2 + (ord(c) % 10) for c in text]
        return type("Tokens", (), {"input_ids": ids})()


def test_prompt_target_dataset_masks_prompt(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"examples": [{"prompt": "ab", "target": "cd"}]}))
    ds = PromptTargetDataset(str(path), _Tokenizer(), max_length=16)
    item = ds[0]
    assert item["labels"][0].item() == -100
    batch = collate_prompt_targets([item], pad_token_id=0)
    assert batch["input_ids"].shape == batch["labels"].shape


def test_reft_r1_direct_request_falls_back_to_model_dir(tmp_path):
    base = tmp_path / "results"
    default_results = base / "DirectRequest" / "default" / "results"
    default_completions = base / "DirectRequest" / "default" / "completions"
    model_results = base / "DirectRequest" / "llama2_7b" / "results"
    model_completions = base / "DirectRequest" / "llama2_7b" / "completions"
    for path in [
        default_results,
        default_completions,
        model_results,
        model_completions,
    ]:
        path.mkdir(parents=True)

    (default_results / "llama2_7b.json").write_text("{}")
    (default_completions / "llama2_7b.json").write_text("{}")
    result_path = model_results / "llama2_7b_20260125_235140.json"
    completion_path = model_completions / "llama2_7b_20260125_235140_merged.json"
    result_path.write_text(json.dumps({"B": [{"label": 1}]}))
    completion_path.write_text(
        json.dumps({"B": [{"test_case": "x", "generation": "y"}]})
    )

    stage = ReFTR1TrainingStage(
        {"pipeline": {"base_save_dir": str(base)}}, str(tmp_path), str(tmp_path)
    )
    found_completion, found_result = stage._find_latest_completion_result_pair(
        "llama2_7b", "DirectRequest"
    )
    assert found_completion == str(completion_path)
    assert found_result == str(result_path)


def test_reft_r1_uses_all_retain_examples_by_default(tmp_path):
    base = tmp_path / "results" / "AutoDAN" / "llama2_7b"
    (base / "results").mkdir(parents=True)
    (base / "completions").mkdir()
    (base / "results" / "llama2_7b_20260118.json").write_text(
        json.dumps(
            {
                "B1": [{"label": 1}],
                "B2": [{"label": 0}],
                "B3": [{"label": 0}],
            }
        )
    )
    (base / "completions" / "llama2_7b_20260118.json").write_text(
        json.dumps(
            {
                "B1": [{"test_case": "bad", "generation": "unsafe"}],
                "B2": [{"test_case": "safe1", "generation": "ok1"}],
                "B3": [{"test_case": "safe2", "generation": "ok2"}],
            }
        )
    )

    stage = ReFTR1TrainingStage(
        {"pipeline": {"base_save_dir": str(tmp_path / "results")}},
        str(tmp_path),
        str(tmp_path),
    )
    dataset_path = stage._build_training_dataset(
        model_name="llama2_7b",
        attack_methods=["AutoDAN"],
        training_cfg={"include_safe_retain": True},
        output_dir=str(tmp_path),
        seed=0,
    )
    payload = json.loads((tmp_path / "harmbench_reft_r1_dataset.json").read_text())
    assert dataset_path == str(tmp_path / "harmbench_reft_r1_dataset.json")
    assert payload["stats"] == {"harmful_refusal": 1, "retain": 2, "total": 3}
