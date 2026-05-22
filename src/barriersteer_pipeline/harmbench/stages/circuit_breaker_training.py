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

import torch


class CircuitBreakerTrainingStage:
    """Train a HarmBench-derived CircuitBreaker baseline from labeled completions."""

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
            seed=seed or training_cfg.get("seed", 42),
        )
        self._run_training_job(
            model_name=model_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            training_cfg=training_cfg,
            seed=seed,
        )
        return output_dir

    def _resolve_training_config(
        self, model_config: Dict[str, Any], training_mode: Optional[str]
    ) -> Dict[str, Any]:
        polytope_cfg = model_config.get("polytope_training", {})
        modes = polytope_cfg.get("modes", {}) if isinstance(polytope_cfg, dict) else {}
        if not training_mode:
            raise ValueError(
                "CircuitBreaker training requires an explicit training_mode under model.polytope_training.modes"
            )
        if not isinstance(modes, dict) or training_mode not in modes:
            raise ValueError(
                f"Missing model.polytope_training.modes.{training_mode} configuration"
            )

        training_cfg = (modes.get(training_mode) or {}).copy()
        if not training_cfg:
            raise ValueError(
                f"Empty model.polytope_training.modes.{training_mode} configuration"
            )
        if training_cfg.get("model_type") != "circuit_breaker":
            raise ValueError(
                f"Mode '{training_mode}' is not a circuit_breaker mode (model_type={training_cfg.get('model_type')!r})"
            )
        return training_cfg

    def _resolve_model_output_dir(
        self,
        training_cfg: Dict[str, Any],
        training_mode: Optional[str],
        seed: Optional[int],
    ) -> str:
        explicit_path = training_cfg.get("circuit_breaker_model_path")
        if explicit_path:
            return os.path.abspath(os.path.expandvars(explicit_path))

        base_output_dir = training_cfg.get("base_output_dir")
        if not base_output_dir:
            raise ValueError("circuit_breaker_training.base_output_dir is required")
        base_output_dir = os.path.abspath(os.path.expandvars(base_output_dir))

        suffix = training_cfg.get(
            "base_exp_ident_suffix", training_mode or "circuit_breaker"
        )
        effective_seed = seed if seed is not None else training_cfg.get("seed")
        if effective_seed is None:
            return os.path.join(base_output_dir, suffix)
        return os.path.join(base_output_dir, f"{suffix}_seed{effective_seed}")

    def _build_training_dataset(
        self,
        model_name: str,
        attack_methods: List[str],
        training_cfg: Dict[str, Any],
        output_dir: str,
        seed: int,
    ) -> str:
        behavior_filter = self._load_behavior_filter(training_cfg)
        safe_pairs: List[Dict[str, str]] = []
        unsafe_pairs: List[Dict[str, str]] = []

        for method in attack_methods:
            completions_path, results_path = self._find_latest_completion_result_pair(
                model_name=model_name, method=method
            )
            if completions_path is None or results_path is None:
                self.logger.warning(
                    f"Skipping {method}: no matching completions/results found"
                )
                continue

            with open(completions_path, "r", encoding="utf-8") as f:
                completions = json.load(f)
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)

            for behavior_id, completions_list in completions.items():
                if behavior_filter is not None and behavior_id not in behavior_filter:
                    continue
                result_list = results.get(behavior_id)
                if not result_list:
                    continue

                for completion_entry, result_entry in zip(
                    completions_list, result_list
                ):
                    prompt = self._normalize_prompt(completion_entry.get("test_case"))
                    generation = completion_entry.get("generation")
                    if not prompt or not generation:
                        continue

                    pair = {
                        "prompt": prompt,
                        "output": generation,
                        "behavior_id": behavior_id,
                        "attack_method": method,
                    }
                    is_unsafe = int(result_entry.get("label", 0)) == 1
                    if is_unsafe:
                        unsafe_pairs.append(pair)
                    else:
                        safe_pairs.append(pair)

        if not unsafe_pairs:
            raise ValueError(
                "No unsafe HarmBench completions were found for CircuitBreaker training"
            )
        if not safe_pairs:
            raise ValueError(
                "No safe HarmBench completions were found for retain training"
            )

        random.Random(seed).shuffle(safe_pairs)
        random.Random(seed).shuffle(unsafe_pairs)

        max_safe = training_cfg.get("max_safe_examples")
        max_unsafe = training_cfg.get("max_unsafe_examples")
        if max_safe:
            safe_pairs = safe_pairs[:max_safe]
        if max_unsafe:
            unsafe_pairs = unsafe_pairs[:max_unsafe]

        val_fraction = float(training_cfg.get("val_fraction", 0.1))
        safe_train, safe_val = self._train_val_split(safe_pairs, val_fraction)
        unsafe_train, unsafe_val = self._train_val_split(unsafe_pairs, val_fraction)

        payload = {
            "safe_train": [
                {"prompt": p["prompt"], "output": p["output"]} for p in safe_train
            ],
            "safe_val": [
                {"prompt": p["prompt"], "output": p["output"]} for p in safe_val
            ],
            "unsafe_train": [
                {"prompt": p["prompt"], "output": p["output"]} for p in unsafe_train
            ],
            "unsafe_val": [
                {"prompt": p["prompt"], "output": p["output"]} for p in unsafe_val
            ],
            "stats": {
                "safe_total": len(safe_pairs),
                "unsafe_total": len(unsafe_pairs),
                "safe_train": len(safe_train),
                "safe_val": len(safe_val),
                "unsafe_train": len(unsafe_train),
                "unsafe_val": len(unsafe_val),
            },
        }

        dataset_path = os.path.join(
            output_dir, "harmbench_circuit_breaker_dataset.json"
        )
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        self.logger.info(f"Wrote CircuitBreaker dataset to {dataset_path}")
        self.logger.info(f"CircuitBreaker dataset stats: {payload['stats']}")
        return dataset_path

    @staticmethod
    def _normalize_prompt(prompt: Any) -> Optional[str]:
        if prompt is None:
            return None
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            return "\n".join(str(item) for item in prompt if item is not None)
        return str(prompt)

    def _load_behavior_filter(self, training_cfg: Dict[str, Any]) -> Optional[set]:
        behavior_dataset = training_cfg.get("behavior_dataset")
        if not behavior_dataset:
            return None
        behavior_dataset = os.path.expandvars(behavior_dataset)
        if not os.path.isabs(behavior_dataset):
            behavior_dataset = os.path.join(
                self.harmbench_path, behavior_dataset.lstrip("./")
            )
        if not os.path.exists(behavior_dataset):
            raise FileNotFoundError(f"Behavior dataset not found: {behavior_dataset}")

        behavior_ids = set()
        with open(behavior_dataset, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                behavior_ids.add(row["BehaviorID"])
        return behavior_ids

    def _train_val_split(
        self, pairs: List[Dict[str, str]], val_fraction: float
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        if len(pairs) < 2:
            return pairs, pairs
        val_count = max(1, int(math.ceil(len(pairs) * val_fraction)))
        val_count = min(val_count, len(pairs) - 1)
        return pairs[val_count:], pairs[:val_count]

    def _find_latest_completion_result_pair(
        self, model_name: str, method: str
    ) -> Tuple[Optional[str], Optional[str]]:
        results_base_dir = self.config.get("pipeline", {}).get(
            "base_save_dir", "results"
        )
        if not os.path.isabs(results_base_dir):
            results_base_dir = os.path.join(self.harmbench_path, results_base_dir)

        if method == "DirectRequest":
            results_dir = os.path.join(results_base_dir, method, "default")
        else:
            results_dir = os.path.join(results_base_dir, method, model_name)

        completions_dir = os.path.join(results_dir, "completions")
        results_dir_only = os.path.join(results_dir, "results")

        result_candidates = sorted(
            glob.glob(os.path.join(results_dir_only, f"{model_name}_*.json")),
            key=os.path.getmtime,
            reverse=True,
        )
        for result_path in result_candidates:
            if os.path.getsize(result_path) <= 10:
                continue
            suffix = os.path.basename(result_path)[len(model_name) : -5]
            completions_candidate = os.path.join(
                completions_dir, f"{model_name}{suffix}_merged.json"
            )
            if not os.path.exists(completions_candidate):
                completions_candidate = os.path.join(
                    completions_dir, f"{model_name}{suffix}.json"
                )
            if (
                os.path.exists(completions_candidate)
                and os.path.getsize(completions_candidate) > 10
            ):
                return completions_candidate, result_path
        return None, None

    def _get_num_processes(self, training_cfg: Dict[str, Any]) -> int:
        configured = training_cfg.get("num_processes")
        if configured is not None:
            return max(1, int(configured))
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            return max(
                1,
                len(
                    [
                        device
                        for device in cuda_visible_devices.split(",")
                        if device.strip()
                    ]
                ),
            )
        if torch.cuda.is_available():
            return max(1, torch.cuda.device_count())
        return 1

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
        num_processes = self._get_num_processes(training_cfg)
        use_accelerate = bool(training_cfg.get("use_accelerate", num_processes > 1))

        cmd = [sys.executable]
        if use_accelerate:
            cmd.extend(["-m", "accelerate.commands.launch"])
            if num_processes > 1:
                cmd.extend(
                    [
                        "--multi_gpu",
                        "--num_processes",
                        str(num_processes),
                        "--num_machines",
                        "1",
                    ]
                )
            config_file = training_cfg.get("accelerate_config_file")
            if config_file:
                cmd.extend(
                    ["--config_file", os.path.abspath(os.path.expandvars(config_file))]
                )
            main_process_port = training_cfg.get("main_process_port")
            if main_process_port:
                cmd.extend(["--main_process_port", str(main_process_port)])
            cmd.extend(["-m", "barriersteer_pipeline.harmbench.train_circuit_breaker"])
        else:
            cmd.extend(["-m", "barriersteer_pipeline.harmbench.train_circuit_breaker"])

        cmd.extend(
            [
                "--model_name_or_path",
                model_path,
                "--dataset_path",
                dataset_path,
                "--output_dir",
                output_dir,
                "--target_layers",
                str(training_cfg["target_layers"]),
                "--transform_layers",
                str(training_cfg.get("transform_layers", "-1")),
                "--lorra_alpha",
                str(training_cfg.get("lorra_alpha", 10.0)),
                "--lora_r",
                str(training_cfg.get("lora_r", 16)),
                "--lora_alpha",
                str(training_cfg.get("lora_alpha", 16)),
                "--lora_dropout",
                str(training_cfg.get("lora_dropout", 0.05)),
                "--learning_rate",
                str(training_cfg.get("learning_rate", 1e-4)),
                "--weight_decay",
                str(training_cfg.get("weight_decay", 0.0)),
                "--max_steps",
                str(training_cfg.get("max_steps", 150)),
                "--per_device_train_batch_size",
                str(training_cfg.get("per_device_train_batch_size", 8)),
                "--gradient_accumulation_steps",
                str(training_cfg.get("gradient_accumulation_steps", 1)),
                "--warmup_steps",
                str(training_cfg.get("warmup_steps", 0)),
                "--logging_steps",
                str(training_cfg.get("logging_steps", 10)),
                "--schedule_steps",
                str(training_cfg.get("schedule_steps", 300)),
                "--max_length",
                str(training_cfg.get("max_length", 1024)),
                "--seed",
                str(seed if seed is not None else training_cfg.get("seed", 42)),
            ]
        )
        if training_cfg.get("bf16", True):
            cmd.append("--bf16")
        if training_cfg.get("gradient_checkpointing", True):
            cmd.append("--gradient_checkpointing")

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{self.barriersteer_path}:{env['PYTHONPATH']}"
            if env.get("PYTHONPATH")
            else str(self.barriersteer_path)
        )

        self.logger.info(
            "Training CircuitBreaker model for %s with %d process(es)%s",
            model_name,
            num_processes,
            " via accelerate" if use_accelerate else "",
        )
        self.logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=self.barriersteer_path, env=env)

    def _lookup_model_path(self, model_name: str) -> str:
        for model_cfg in self.config.get("models", []):
            if model_cfg.get("name") == model_name:
                return model_cfg["path"]
        raise ValueError(f"Could not determine base model path for {model_name}")
