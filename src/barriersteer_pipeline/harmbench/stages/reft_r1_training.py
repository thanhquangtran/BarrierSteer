"""Train HarmBench-derived ReFT-r1 refusal vectors."""

from __future__ import annotations

import csv
import glob
import json
import logging
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_REFUSAL = "I'm sorry, but I can't help with that request."


class ReFTR1TrainingStage:
    """Build prompt/target pairs and train a rank-1 ReFT vector."""

    def __init__(
        self, config: Dict[str, Any], harmbench_path: str, barriersteer_path: str
    ):
        self.config = config
        self.harmbench_path = Path(harmbench_path)
        self.barriersteer_path = Path(barriersteer_path)
        self.logger = logging.getLogger(__name__)

    def run(
        self,
        model_name: str,
        attack_methods: List[str],
        model_config: Dict[str, Any],
        training_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> str:
        training_cfg = self._resolve_training_config(model_config, training_mode)
        output_dir = self._resolve_model_output_dir(training_cfg, training_mode, seed)
        os.makedirs(output_dir, exist_ok=True)
        dataset_path = self._build_training_dataset(
            model_name=model_name,
            attack_methods=attack_methods,
            training_cfg=training_cfg,
            output_dir=output_dir,
            seed=seed if seed is not None else int(training_cfg.get("seed", 42)),
        )
        self._run_training_job(model_name, dataset_path, output_dir, training_cfg, seed)
        return output_dir

    def _resolve_training_config(
        self, model_config: Dict[str, Any], training_mode: Optional[str]
    ) -> Dict[str, Any]:
        modes = model_config.get("polytope_training", {}).get("modes", {})
        if not training_mode or training_mode not in modes:
            raise ValueError(
                f"Missing model.polytope_training.modes.{training_mode} configuration"
            )
        cfg = (modes[training_mode] or {}).copy()
        if cfg.get("model_type") != "reft_r1":
            raise ValueError(
                f"Mode {training_mode!r} is not reft_r1 (model_type={cfg.get('model_type')!r})"
            )
        return cfg

    def _resolve_model_output_dir(
        self,
        training_cfg: Dict[str, Any],
        training_mode: Optional[str],
        seed: Optional[int],
    ) -> str:
        explicit = training_cfg.get("reft_r1_model_path") or training_cfg.get(
            "vector_dir"
        )
        if explicit:
            return os.path.abspath(os.path.expandvars(explicit))
        base = training_cfg.get("base_output_dir")
        if not base:
            raise ValueError(
                "ReFT-r1 training requires base_output_dir or reft_r1_model_path"
            )
        suffix = training_cfg.get("base_exp_ident_suffix", training_mode or "reft_r1")
        effective_seed = seed if seed is not None else training_cfg.get("seed")
        dirname = suffix if effective_seed is None else f"{suffix}_seed{effective_seed}"
        return os.path.abspath(os.path.expandvars(os.path.join(base, dirname)))

    def _build_training_dataset(
        self,
        model_name: str,
        attack_methods: List[str],
        training_cfg: Dict[str, Any],
        output_dir: str,
        seed: int,
    ) -> str:
        behavior_filter = self._load_behavior_filter(training_cfg)
        refusal = training_cfg.get("refusal_target", DEFAULT_REFUSAL)
        examples: List[Dict[str, str]] = []
        retain_examples: List[Dict[str, str]] = []

        for method in attack_methods:
            completions_path, results_path = self._find_latest_completion_result_pair(
                model_name, method
            )
            if completions_path is None or results_path is None:
                self.logger.warning(
                    "Skipping %s: no matching completions/results found", method
                )
                continue
            with open(completions_path, "r", encoding="utf-8") as f:
                completions = json.load(f)
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            for behavior_id, completions_list in completions.items():
                if behavior_filter is not None and behavior_id not in behavior_filter:
                    continue
                result_list = results.get(behavior_id) or []
                for completion_entry, result_entry in zip(
                    completions_list, result_list
                ):
                    prompt = completion_entry.get("test_case")
                    generation = completion_entry.get("generation")
                    if not prompt:
                        continue
                    label = int(result_entry.get("label", 0))
                    if label == 1:
                        examples.append(
                            {"prompt": prompt, "target": refusal, "source": "harmful"}
                        )
                    elif generation and training_cfg.get("include_safe_retain", True):
                        retain_examples.append(
                            {"prompt": prompt, "target": generation, "source": "retain"}
                        )

        if not examples:
            raise ValueError(
                "No unsafe HarmBench examples were found for ReFT-r1 training"
            )

        rng = random.Random(seed)
        rng.shuffle(examples)
        rng.shuffle(retain_examples)
        max_harmful = training_cfg.get(
            "max_harmful_examples", training_cfg.get("max_examples")
        )
        max_retain = training_cfg.get("max_retain_examples")
        if max_harmful:
            examples = examples[: int(max_harmful)]
        if max_retain is not None:
            retain_examples = retain_examples[: int(max_retain)] if max_retain else []
        final_examples = examples + retain_examples
        rng.shuffle(final_examples)

        payload = {
            "examples": final_examples,
            "stats": {
                "harmful_refusal": len(examples),
                "retain": len(retain_examples),
                "total": len(final_examples),
            },
        }
        path = os.path.join(output_dir, "harmbench_reft_r1_dataset.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.logger.info(
            "Wrote ReFT-r1 dataset to %s with stats %s", path, payload["stats"]
        )
        return path

    def _load_behavior_filter(self, training_cfg: Dict[str, Any]) -> Optional[set]:
        behavior_dataset = training_cfg.get("behavior_dataset")
        if not behavior_dataset:
            return None
        behavior_dataset = os.path.expandvars(behavior_dataset)
        if not os.path.isabs(behavior_dataset):
            behavior_dataset = os.path.join(
                self.harmbench_path, behavior_dataset.lstrip("./")
            )
        behavior_ids = set()
        with open(behavior_dataset, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                behavior_ids.add(row["BehaviorID"])
        return behavior_ids

    def _find_latest_completion_result_pair(
        self, model_name: str, method: str
    ) -> Tuple[Optional[str], Optional[str]]:
        results_base_dir = self.config.get("pipeline", {}).get(
            "base_save_dir", "results"
        )
        if not os.path.isabs(results_base_dir):
            results_base_dir = os.path.join(self.harmbench_path, results_base_dir)

        run_dirs = [
            os.path.join(
                results_base_dir,
                method,
                "default" if method == "DirectRequest" else model_name,
            )
        ]
        if method == "DirectRequest":
            # Older local HarmBench runs store DirectRequest under the model name,
            # while newer canonical runs may use DirectRequest/default.
            model_run_dir = os.path.join(results_base_dir, method, model_name)
            if model_run_dir not in run_dirs:
                run_dirs.append(model_run_dir)

        candidates: List[Tuple[str, str]] = []
        for results_dir in run_dirs:
            completions_dir = os.path.join(results_dir, "completions")
            results_dir_only = os.path.join(results_dir, "results")
            result_candidates = glob.glob(
                os.path.join(results_dir_only, f"{model_name}.json")
            )
            result_candidates.extend(
                glob.glob(os.path.join(results_dir_only, f"{model_name}_*.json"))
            )
            for result_path in result_candidates:
                candidates.append((completions_dir, result_path))

        candidates = sorted(
            candidates, key=lambda item: os.path.getmtime(item[1]), reverse=True
        )
        for completions_dir, result_path in candidates:
            if os.path.getsize(result_path) <= 10:
                continue
            suffix = os.path.basename(result_path)[len(model_name) : -5]
            for name in (
                f"{model_name}{suffix}_merged.json",
                f"{model_name}{suffix}.json",
            ):
                comp = os.path.join(completions_dir, name)
                if os.path.exists(comp) and os.path.getsize(comp) > 10:
                    return comp, result_path
        return None, None

    def _lookup_model_path(self, model_name: str) -> str:
        for model_cfg in self.config.get("models", []):
            if model_cfg.get("name") == model_name:
                return model_cfg["path"]
        raise ValueError(f"Could not determine base model path for {model_name}")

    def _run_training_job(
        self,
        model_name: str,
        dataset_path: str,
        output_dir: str,
        training_cfg: Dict[str, Any],
        seed: Optional[int],
    ) -> None:
        model_path = training_cfg.get(
            "model_name_or_path", self._lookup_model_path(model_name)
        )
        cmd = [
            sys.executable,
            "-m",
            "barriersteer_pipeline.harmbench.train_reft_r1",
            "--model_name_or_path",
            model_path,
            "--dataset_path",
            dataset_path,
            "--output_dir",
            output_dir,
            "--target_layer",
            str(training_cfg["target_layer"]),
            "--top_k",
            str(training_cfg.get("top_k", 5)),
            "--beta",
            str(training_cfg.get("beta", 1.0)),
            "--l1_coeff",
            str(training_cfg.get("l1_coeff", 0.0)),
            "--learning_rate",
            str(training_cfg.get("learning_rate", 1e-3)),
            "--max_steps",
            str(training_cfg.get("max_steps", 500)),
            "--batch_size",
            str(
                training_cfg.get(
                    "batch_size", training_cfg.get("per_device_train_batch_size", 4)
                )
            ),
            "--gradient_accumulation_steps",
            str(training_cfg.get("gradient_accumulation_steps", 1)),
            "--warmup_steps",
            str(training_cfg.get("warmup_steps", 0)),
            "--max_length",
            str(training_cfg.get("max_length", 1024)),
            "--dtype",
            str(training_cfg.get("dtype", "auto")),
            "--seed",
            str(seed if seed is not None else training_cfg.get("seed", 42)),
            "--log_every",
            str(training_cfg.get("logging_steps", 10)),
        ]
        env = os.environ.copy()
        src_path = str(self.barriersteer_path / "src")
        root_path = str(self.barriersteer_path)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{src_path}:{root_path}:{existing}"
            if existing
            else f"{src_path}:{root_path}"
        )
        self.logger.info(
            "Training ReFT-r1 vector for %s: %s", model_name, " ".join(cmd)
        )
        subprocess.run(cmd, check=True, cwd=self.barriersteer_path, env=env)


def generate_reft_r1_vector_path(
    base_output_dir: str, base_exp_ident_suffix: str, seed: Optional[int] = None
) -> str:
    dirname = (
        base_exp_ident_suffix if seed is None else f"{base_exp_ident_suffix}_seed{seed}"
    )
    return str(Path(base_output_dir) / dirname / "reft_r1_vector.pt")
