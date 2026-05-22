#!/usr/bin/env python3
"""
HarmBench Pipeline Orchestrator

This script orchestrates the complete HarmBench experimental pipeline:
1. Attack Generation: Generate adversarial test cases
2. Hidden State Extraction and Processing: Extract and process hidden states
3. Polytope Training: Train safety polytope constraints
4. Steering Evaluation: Evaluate polytope effectiveness

Usage:
python src/barriersteer_pipeline/harmbench/run_harmbench_pipeline.py --config src/barriersteer_pipeline/harmbench/config/pipeline_config.yaml --model qwen_1.5b --stages 1
"""

import argparse
import copy
import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

# Add the barriersteer_pipeline package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from barriersteer_pipeline.harmbench.stages.attack_generation import (  # noqa: E402
    AttackGenerationStage,
)
from barriersteer_pipeline.harmbench.stages.circuit_breaker_training import (  # noqa: E402
    CircuitBreakerTrainingStage,
)
from barriersteer_pipeline.harmbench.stages.hidden_state_extraction_and_processing import (  # noqa: E402
    HiddenStateExtractionAndProcessingStage,
)
from barriersteer_pipeline.harmbench.stages.polytope_training import (  # noqa: E402
    PolytopeTrainingStage,
)
from barriersteer_pipeline.harmbench.stages.reft_r1_training import (  # noqa: E402
    ReFTR1TrainingStage,
    generate_reft_r1_vector_path,
)
from barriersteer_pipeline.harmbench.stages.steering_evaluation import (  # noqa: E402
    SteeringEvaluationStage,
)
from barriersteer_pipeline.harmbench.utils.config_utils import (  # noqa: E402
    expand_modes_with_num_phi,
    generate_circuit_breaker_model_path,
    generate_polytope_model_path,
    get_model_config,
    load_config,
    setup_logging,
    validate_config,
)


def _run_training_task_worker(
    barriersteer_path: str,
    model_name: str,
    processed_data_path: str,
    model_config: Dict[str, Any],
    training_mode: str,
    config: Dict[str, Any],
    gpu_id: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    """
    Worker function for parallel training execution.
    This is a standalone function (not a method) so it can be pickled for ProcessPoolExecutor.
    """
    from barriersteer_pipeline.harmbench.stages.polytope_training import (
        PolytopeTrainingStage,
    )

    training_stage = PolytopeTrainingStage(config, barriersteer_path)
    training_stage.run(
        model_name,
        processed_data_path,
        model_config,
        training_mode=training_mode,
        gpu_id=gpu_id,
        seed=seed,
    )


def _normalize_selection(value: Optional[Any]) -> Optional[List[str]]:
    """Normalize comma-separated/list mode filters while preserving user spelling."""
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = []
        for entry in value:
            if isinstance(entry, str):
                items.extend(item.strip() for item in entry.split(","))
            else:
                items.append(str(entry).strip())
    normalized = [item for item in items if item]
    return normalized or None


def _mode_type_from_training(mode_cfg: Dict[str, Any]) -> str:
    if not isinstance(mode_cfg, dict):
        return "polytope"
    return str(mode_cfg.get("model_type") or "polytope")


def _mode_type_from_steering(
    mode_name: str,
    mode_cfg: Dict[str, Any],
    training_modes: Dict[str, Any],
) -> str:
    """Infer the runnable type for a Stage 4 steering mode."""
    if not isinstance(mode_cfg, dict):
        return _mode_type_from_training(training_modes.get(mode_name, {}))

    circuit_breaker_cfg = mode_cfg.get("circuit_breaker", {})
    if isinstance(circuit_breaker_cfg, dict) and circuit_breaker_cfg.get("use", False):
        return "circuit_breaker"

    reft_r1_cfg = mode_cfg.get("reft_r1", {})
    if isinstance(reft_r1_cfg, dict) and reft_r1_cfg.get("use", False):
        return "reft_r1"

    defense_cfg = mode_cfg.get("defense", {})
    if isinstance(defense_cfg, dict) and defense_cfg.get("use", False):
        return str(defense_cfg.get("method") or "defense")

    if mode_cfg.get("polytope_model_path", None) == "":
        return "base_model"

    return _mode_type_from_training(training_modes.get(mode_name, {}))


def _mode_matches_selection(
    mode_name: str,
    mode_type: str,
    selected_modes: Optional[List[str]] = None,
    selected_mode_types: Optional[List[str]] = None,
) -> bool:
    selected_mode_set = set(selected_modes or [])
    selected_type_set = {item.lower() for item in (selected_mode_types or [])}

    if selected_mode_set and mode_name not in selected_mode_set:
        return False
    if selected_type_set and mode_type.lower() not in selected_type_set:
        return False
    return True


def _filter_modes_by_selection(
    modes: Dict[str, Any],
    type_resolver,
    selected_modes: Optional[List[str]] = None,
    selected_mode_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not selected_modes and not selected_mode_types:
        return dict(modes)
    return {
        mode_name: mode_cfg
        for mode_name, mode_cfg in modes.items()
        if _mode_matches_selection(
            mode_name,
            type_resolver(mode_name, mode_cfg),
            selected_modes=selected_modes,
            selected_mode_types=selected_mode_types,
        )
    }


def _normalize_seeds(seed_value) -> List[int]:
    """
    Normalize a seed value to a list of ints.
    Accepts a single int or a list of ints. Returns a list.
    """
    if seed_value is None:
        return [None]
    if isinstance(seed_value, (list, tuple)):
        return list(seed_value)
    return [seed_value]


def _stage4_mode_uses_generation_seed(
    mode_name: str,
    mode_cfg: Dict[str, Any],
    training_modes: Dict[str, Any],
) -> bool:
    """Return whether a Stage 4 mode should fan out over generation seeds.

    Some baselines are deterministic with respect to the configured
    HarmBench generation seed: base-model evaluation, prompt-only
    Self-Reminder, and fixed activation-vector baselines (ActAdd and
    directional ablation).  Those modes should run once even when the
    model config has a historical ``generation_seed: [42, 43, 44]`` list
    for seed-specific steering artifacts.
    """
    mode_type = _mode_type_from_steering(mode_name, mode_cfg, training_modes)
    if mode_type in {"base_model", "self_reminder"}:
        return False

    training_cfg = training_modes.get(mode_name, {})
    steering_vector_cfg = (
        mode_cfg.get("steering_vector", {}) if isinstance(mode_cfg, dict) else {}
    )
    vector_method = None
    if isinstance(steering_vector_cfg, dict):
        vector_method = steering_vector_cfg.get("method")
    if vector_method is None and isinstance(training_cfg, dict):
        vector_method = training_cfg.get("steering_method") or training_cfg.get(
            "method"
        )

    deterministic_vector_methods = {"actadd", "dirablate"}
    if mode_type == "steering_vector" and (
        mode_name in deterministic_vector_methods
        or vector_method in deterministic_vector_methods
    ):
        return False

    return True


class HarmBenchPipeline:
    """Main pipeline orchestrator for HarmBench experiments"""

    def __init__(self, config_path: str):
        """
        Initialize pipeline

        Args:
            config_path: Path to pipeline configuration file
        """
        self.config = load_config(config_path)
        validate_config(self.config)
        setup_logging(self.config)

        self.logger = logging.getLogger(__name__)
        self.harmbench_path = self.config["pipeline"]["harmbench_path"]
        self.barriersteer_path = self.config["pipeline"]["barriersteer_path"]

        # Initialize stages
        self.attack_generation = AttackGenerationStage(self.config, self.harmbench_path)
        self.hidden_state_extraction_and_processing = (
            HiddenStateExtractionAndProcessingStage(
                self.config, self.harmbench_path, self.barriersteer_path
            )
        )
        self.polytope_training = PolytopeTrainingStage(
            self.config, self.barriersteer_path
        )
        self.circuit_breaker_training = CircuitBreakerTrainingStage(
            self.config, self.harmbench_path, self.barriersteer_path
        )
        self.reft_r1_training = ReFTR1TrainingStage(
            self.config, self.harmbench_path, self.barriersteer_path
        )
        self.steering_evaluation = SteeringEvaluationStage(
            self.config, self.harmbench_path
        )

    def run(
        self,
        model_name: str,
        stages: Optional[List[int]] = None,
        selected_modes: Optional[List[str]] = None,
        selected_mode_types: Optional[List[str]] = None,
    ) -> None:
        """
        Run the complete pipeline for a specific model

        Args:
            model_name: Name of the model to process
            stages: List of stage numbers to run (1-4). If None, runs all stages.
            selected_modes: Optional steering/training mode names to include.
            selected_mode_types: Optional mode types to include (for example, circuit_breaker).
        """
        if stages is None:
            stages = [1, 2, 3, 4]

        selected_modes = _normalize_selection(
            selected_modes
            if selected_modes is not None
            else self.config.get("pipeline", {}).get("modes")
        )
        selected_mode_types = _normalize_selection(
            selected_mode_types
            if selected_mode_types is not None
            else self.config.get("pipeline", {}).get("mode_types")
        )

        self.logger.info(
            f"Starting HarmBench pipeline for {model_name} with stages {stages}"
        )
        if selected_modes or selected_mode_types:
            self.logger.info(
                "Mode selection active: modes=%s mode_types=%s",
                selected_modes or "all",
                selected_mode_types or "all",
            )

        # Get model configuration
        model_config = get_model_config(self.config, model_name)
        attack_methods = model_config.get("attack_methods", [])
        # Default to DirectRequest if using category data and no methods specified
        # This handles cases like WildGuard/BeaverTails where we want implicitly "DirectRequest" for stage 1
        use_category_data = (
            self.config.get("polytope_training", {}).get("use_category_data", False)
            or self.config.get("polytope_training", {})
            .get("dataset", {})
            .get("use_category_data", False)
            or model_config.get("polytope_training", {})
            .get("dataset", {})
            .get("use_category_data", False)
            or model_config.get("polytope_training", {}).get("use_category_data", False)
        )

        if not attack_methods and use_category_data:
            self.logger.info(
                "No attack_methods specified but use_category_data=True. Defaulting to ['DirectRequest'] for attack generation/stage 1."
            )
            attack_methods = ["DirectRequest"]

        eval_methods = model_config.get("steering", {}).get("evaluation_methods", [])
        # Ensure stages 1/2 cover everything we plan to evaluate in stage 4
        combined_methods = list(dict.fromkeys(attack_methods + eval_methods))
        hidden_state_layer = model_config.get("hidden_state_layer", 20)

        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Attack methods (for training): {attack_methods}")
        if eval_methods:
            self.logger.info(f"Evaluation methods (for Stage 4): {eval_methods}")
        if combined_methods != attack_methods:
            self.logger.info(
                f"Combined methods for Stage 1 (attack + evaluation): {combined_methods}"
            )
        else:
            self.logger.info(
                f"All evaluation methods are already in attack methods. Using: {combined_methods}"
            )
        methods_for_generation = combined_methods
        self.logger.info(f"Hidden state layer: {hidden_state_layer}")

        # Stage 1: Attack Generation
        if 1 in stages:
            self.logger.info("=== Stage 1: Attack Generation ===")
            try:
                self.attack_generation.run(model_name, methods_for_generation)
                self.logger.info("Stage 1 completed successfully")
            except Exception as e:
                self.logger.error(f"Stage 1 failed: {e}")
                raise

        # Stage 2: Hidden State Extraction and Processing
        # Only use attack_methods for training (not evaluation_methods)
        processed_data_path = None
        if 2 in stages:
            self.logger.info("=== Stage 2: Hidden State Extraction and Processing ===")
            self.logger.info(
                f"Using only attack_methods for training: {attack_methods}"
            )
            try:
                processed_data_path = self.hidden_state_extraction_and_processing.run(
                    model_name, attack_methods, hidden_state_layer, model_config
                )
                self.logger.info(
                    f"Stage 2 completed successfully: {processed_data_path}"
                )
            except Exception as e:
                self.logger.error(f"Stage 2 failed: {e}")
                raise

        # Stage 3: Polytope Training
        if 3 in stages:
            self.logger.info("=== Stage 3: Polytope Training ===")
            try:
                if processed_data_path is None:
                    # Assume default path if not from previous stage
                    try:
                        processed_data_path = model_config["polytope_training"][
                            "dataset"
                        ]["hidden_states_path"]
                    except KeyError:
                        # Fallback to a default structure if not explicitly defined like in the pipeline config
                        # This handles cases where Llama2_7b default config is used without explicit polytope section
                        self.logger.warning(
                            "Could not find 'hidden_states_path' in model config. Checking default pipeline config location."
                        )
                        processed_data_path = (
                            self.config["pipeline"]
                            .get("polytope_training", {})
                            .get("dataset", {})
                            .get("hidden_states_path", None)
                        )

                        if processed_data_path is None:
                            # Try to construct it if we know the base dirs
                            # This is a bit of a guess but helps unblock
                            base_dir = self.config["pipeline"].get(
                                "barriersteer_path", "."
                            )
                            processed_data_path = (
                                f"{base_dir}/data/harmbench_hidden_states/{model_name}"
                            )
                            self.logger.warning(
                                f"Constructed fallback path: {processed_data_path}"
                            )

                training_cfg = model_config.get("polytope_training", {})
                training_modes_dict = {}
                if isinstance(training_cfg, dict) and isinstance(
                    training_cfg.get("modes", None), dict
                ):
                    training_modes_dict = training_cfg["modes"]

                training_modes_dict = _filter_modes_by_selection(
                    training_modes_dict,
                    lambda _mode_name, mode_cfg: _mode_type_from_training(mode_cfg),
                    selected_modes=selected_modes,
                    selected_mode_types=selected_mode_types,
                )
                if selected_modes or selected_mode_types:
                    self.logger.info(
                        "Stage 3 selected training modes: %s",
                        list(training_modes_dict.keys()) or "none",
                    )

                # Resolve seeds: support both single int and list
                steering_cfg = model_config.get("steering", {})
                raw_seed = steering_cfg.get(
                    "generation_seed",
                    training_cfg.get(
                        "seed", self.config.get("polytope_training", {}).get("seed", 5)
                    ),
                )
                seeds = _normalize_seeds(raw_seed)
                self.logger.info(f"Stage 3 will iterate over seeds: {seeds}")

                circuit_breaker_modes = {
                    mode_name: mode_cfg
                    for mode_name, mode_cfg in training_modes_dict.items()
                    if isinstance(mode_cfg, dict)
                    and mode_cfg.get("model_type") == "circuit_breaker"
                }
                reft_r1_modes = {
                    mode_name: mode_cfg
                    for mode_name, mode_cfg in training_modes_dict.items()
                    if isinstance(mode_cfg, dict)
                    and mode_cfg.get("model_type") == "reft_r1"
                }
                polytope_modes_dict = {
                    mode_name: mode_cfg
                    for mode_name, mode_cfg in training_modes_dict.items()
                    if not (
                        isinstance(mode_cfg, dict)
                        and mode_cfg.get("model_type") in {"circuit_breaker", "reft_r1"}
                    )
                }

                if circuit_breaker_modes:
                    for seed in seeds:
                        for mode_name in circuit_breaker_modes:
                            self.logger.info(
                                f"Stage 3 circuit breaker training (mode={mode_name}, seed={seed}) starting"
                            )
                            self.circuit_breaker_training.run(
                                model_name=model_name,
                                attack_methods=attack_methods,
                                model_config=model_config,
                                training_mode=mode_name,
                                seed=seed,
                            )
                else:
                    self.logger.info(
                        "Stage 3 circuit breaker training skipped: no circuit_breaker modes configured"
                    )

                if reft_r1_modes:
                    for seed in seeds:
                        for mode_name in reft_r1_modes:
                            self.logger.info(
                                f"Stage 3 ReFT-r1 training (mode={mode_name}, seed={seed}) starting"
                            )
                            self.reft_r1_training.run(
                                model_name=model_name,
                                attack_methods=attack_methods,
                                model_config=model_config,
                                training_mode=mode_name,
                                seed=seed,
                            )
                else:
                    self.logger.info(
                        "Stage 3 ReFT-r1 training skipped: no reft_r1 modes configured"
                    )

                if not polytope_modes_dict:
                    self.logger.info(
                        "Stage 3 polytope training skipped: no non-specialized polytope modes configured"
                    )
                else:
                    # Expand modes with num_phi lists
                    expanded_modes = expand_modes_with_num_phi(polytope_modes_dict)

                    # Get execution mode
                    execution_mode = self.config.get("polytope_training", {}).get(
                        "execution_mode", "local"
                    )

                    for seed in seeds:
                        self.logger.info(f"--- Stage 3: Training with seed={seed} ---")

                        # Separate modes into two groups:
                        # 1. Modes that need training (no reuse_weights_from)
                        # 2. Modes that reuse weights (have reuse_weights_from)
                        training_tasks = []
                        reuse_weight_tasks = []

                        for base_mode_name, num_phi, mode_cfg in expanded_modes:
                            mode_label = f"{base_mode_name}_phi{num_phi}"

                            # Check if this mode reuses weights from another mode
                            reuse_from = mode_cfg.get("reuse_weights_from", None)
                            reuse_steering_from = mode_cfg.get(
                                "reuse_steering_vector_from", None
                            )
                            if (
                                reuse_from and reuse_from in polytope_modes_dict
                            ) or reuse_steering_from:
                                # Store for later processing (after training)
                                reuse_weight_tasks.append(
                                    (
                                        mode_label,
                                        base_mode_name,
                                        num_phi,
                                        mode_cfg,
                                        reuse_from,
                                    )
                                )
                                continue

                            # Prepare training task
                            modified_model_config = model_config.copy()
                            modified_training_cfg = training_cfg.copy()
                            modified_training_cfg["modes"] = {base_mode_name: mode_cfg}
                            modified_model_config["polytope_training"] = (
                                modified_training_cfg
                            )

                            training_tasks.append(
                                (
                                    mode_label,
                                    model_name,
                                    processed_data_path,
                                    modified_model_config,
                                    base_mode_name,
                                )
                            )

                        # Execute training tasks
                        if (
                            execution_mode == "local_parallel"
                            and len(training_tasks) > 1
                        ):
                            # Parallel execution with GPU assignment
                            self.logger.info(
                                f"Stage 3 running {len(training_tasks)} training tasks in parallel (seed={seed})"
                            )

                            # Automatically detect available GPUs
                            # Check CUDA_VISIBLE_DEVICES environment variable first
                            cuda_visible_devices = os.environ.get(
                                "CUDA_VISIBLE_DEVICES", None
                            )
                            if cuda_visible_devices:
                                # Parse the actual physical GPU IDs from CUDA_VISIBLE_DEVICES
                                # e.g., "2,3,4,5,6" -> [2, 3, 4, 5, 6]
                                gpu_ids = [
                                    int(gid.strip())
                                    for gid in cuda_visible_devices.split(",")
                                    if gid.strip()
                                ]
                                self.logger.info(
                                    f"Using GPU IDs from CUDA_VISIBLE_DEVICES: {gpu_ids}"
                                )
                            else:
                                # If CUDA_VISIBLE_DEVICES not set, detect all available GPUs
                                try:
                                    import torch

                                    if torch.cuda.is_available():
                                        num_gpus = torch.cuda.device_count()
                                        gpu_ids = list(range(num_gpus))
                                        self.logger.info(
                                            f"Detected {num_gpus} available GPU(s): {gpu_ids}"
                                        )
                                    else:
                                        self.logger.warning(
                                            "No CUDA GPUs available, falling back to sequential execution"
                                        )
                                        gpu_ids = [0]  # Fallback, will use CPU
                                except ImportError:
                                    self.logger.warning(
                                        "PyTorch not available, cannot detect GPUs. Falling back to sequential execution."
                                    )
                                    gpu_ids = [0]

                            if not gpu_ids:
                                raise ValueError(
                                    "No GPUs available for parallel training"
                                )

                            # Always 1 job per GPU
                            jobs_per_gpu = 1

                            # Assign GPUs to tasks in round-robin fashion
                            tasks_with_gpus = []
                            for i, task in enumerate(training_tasks):
                                (
                                    mode_label,
                                    model_name,
                                    processed_data_path,
                                    modified_model_config,
                                    base_mode_name,
                                ) = task
                                # Round-robin GPU assignment (1 job per GPU)
                                # Use the actual physical GPU ID from the list
                                gpu_id = gpu_ids[i % len(gpu_ids)]
                                tasks_with_gpus.append((task, gpu_id))
                                self.logger.info(
                                    f"Assigned GPU {gpu_id} to {mode_label}"
                                )

                            # Determine max workers (1 per GPU)
                            max_workers = min(
                                len(training_tasks), len(gpu_ids), mp.cpu_count()
                            )

                            with ProcessPoolExecutor(
                                max_workers=max_workers
                            ) as executor:
                                futures = {}
                                for task, gpu_id in tasks_with_gpus:
                                    (
                                        mode_label,
                                        model_name,
                                        processed_data_path,
                                        modified_model_config,
                                        base_mode_name,
                                    ) = task
                                    future = executor.submit(
                                        _run_training_task_worker,
                                        self.barriersteer_path,
                                        model_name,
                                        processed_data_path,
                                        modified_model_config,
                                        base_mode_name,
                                        self.config,
                                        gpu_id,
                                        seed,
                                    )
                                    futures[future] = mode_label

                                # Wait for all tasks to complete
                                for future in as_completed(futures):
                                    mode_label = futures[future]
                                    try:
                                        future.result()
                                        self.logger.info(
                                            f"Stage 3 training completed for {mode_label} (seed={seed})"
                                        )
                                    except Exception as e:
                                        self.logger.error(
                                            f"Stage 3 training failed for {mode_label} (seed={seed}): {e}"
                                        )
                                        raise
                        else:
                            # Sequential execution
                            for (
                                mode_label,
                                model_name,
                                processed_data_path,
                                modified_model_config,
                                base_mode_name,
                            ) in training_tasks:
                                self.logger.info(
                                    f"Stage 3 training run (mode={mode_label}, seed={seed}) starting"
                                )
                                self.polytope_training.run(
                                    model_name,
                                    processed_data_path,
                                    modified_model_config,
                                    training_mode=base_mode_name,
                                    seed=seed,
                                )

                        # After training is complete, verify source weights exist for reuse tasks
                        # Note: We don't copy weights - stage 4 will use the source weights path directly
                        if reuse_weight_tasks:
                            self.logger.info(
                                f"Stage 3 verifying {len(reuse_weight_tasks)} weight reuse task(s) (seed={seed})"
                            )
                            for (
                                mode_label,
                                base_mode_name,
                                num_phi,
                                mode_cfg,
                                reuse_from,
                            ) in reuse_weight_tasks:
                                self.logger.info(
                                    f"Stage 3 verifying/building reused weights for {mode_label} (seed={seed})"
                                )
                                if mode_cfg.get(
                                    "model_type"
                                ) == "conditional_steering_vector" and mode_cfg.get(
                                    "reuse_steering_vector_from"
                                ):
                                    self.polytope_training.run(
                                        model_name,
                                        processed_data_path,
                                        model_config,
                                        training_mode=base_mode_name,
                                        seed=seed,
                                    )
                                    continue
                                # Get source mode config
                                source_mode_cfg = polytope_modes_dict[reuse_from]
                                source_base_output_dir = source_mode_cfg.get(
                                    "base_output_dir", None
                                )
                                if not source_base_output_dir:
                                    source_base_output_dir = training_cfg.get(
                                        "base_output_dir", None
                                    )

                                source_base_exp_ident_suffix = source_mode_cfg.get(
                                    "base_exp_ident_suffix", reuse_from
                                )

                                if source_base_output_dir:
                                    source_weights_path = generate_polytope_model_path(
                                        source_base_output_dir,
                                        source_base_exp_ident_suffix,
                                        num_phi,
                                        seed=seed,
                                    )

                                    # Verify source weights exist (they should after training)
                                    if not os.path.exists(source_weights_path):
                                        raise FileNotFoundError(
                                            f"Source weights not found for reuse: {source_weights_path}. "
                                            f"Please train {reuse_from} mode first (seed={seed})."
                                        )

                                    self.logger.info(
                                        f"Source weights verified for {mode_label} at {source_weights_path} (seed={seed})"
                                    )
                                else:
                                    self.logger.warning(
                                        f"Could not determine source path for weight reuse for {mode_label}. "
                                        f"Source base_output_dir not configured."
                                    )

                self.logger.info("Stage 3 completed successfully.")
            except Exception as e:
                self.logger.error(f"Stage 3 failed: {e}")
                raise

        # Stage 4: Steering Evaluation
        if 4 in stages:
            self.logger.info("=== Stage 4: Steering Evaluation ===")
            try:
                steering_cfg = model_config.get("steering", {})
                steering_modes_dict = {}
                if isinstance(steering_cfg, dict) and isinstance(
                    steering_cfg.get("modes", None), dict
                ):
                    steering_modes_dict = steering_cfg["modes"]
                had_configured_steering_modes = bool(steering_modes_dict)

                training_cfg = model_config.get("polytope_training", {})
                polytope_modes_dict = {}
                if isinstance(training_cfg, dict) and isinstance(
                    training_cfg.get("modes", None), dict
                ):
                    polytope_modes_dict = training_cfg["modes"]

                if selected_modes or selected_mode_types:
                    steering_modes_dict = _filter_modes_by_selection(
                        steering_modes_dict,
                        lambda mode_name, mode_cfg: _mode_type_from_steering(
                            mode_name, mode_cfg, polytope_modes_dict
                        ),
                        selected_modes=selected_modes,
                        selected_mode_types=selected_mode_types,
                    )
                    self.logger.info(
                        "Stage 4 selected steering modes: %s",
                        list(steering_modes_dict.keys()) or "none",
                    )

                # Resolve seeds: support both single int and list
                raw_seed = steering_cfg.get("generation_seed", None)
                seeds = _normalize_seeds(raw_seed)
                self.logger.info(f"Stage 4 will iterate over seeds: {seeds}")

                if not steering_modes_dict:
                    if (
                        selected_modes or selected_mode_types
                    ) and had_configured_steering_modes:
                        self.logger.info(
                            "Stage 4 skipped: mode selection matched no steering modes"
                        )
                    else:
                        # No modes defined, run with default
                        training_cfg = model_config.get("polytope_training", {})
                        polytope_model_path = None
                        # Try to find a default model path
                        if isinstance(training_cfg, dict):
                            training_modes = training_cfg.get("modes", {})
                            if training_modes:
                                # Use first available mode
                                first_mode_cfg = next(iter(training_modes.values()))
                                polytope_model_path = first_mode_cfg.get(
                                    "polytope_model_path", None
                                )

                        if polytope_model_path and os.path.exists(polytope_model_path):
                            for seed in seeds:
                                self.logger.info(
                                    f"Stage 4 default mode evaluation (seed={seed}) starting"
                                )
                                self.steering_evaluation.run(
                                    model_name,
                                    polytope_model_path,
                                    model_config,
                                    steering_mode=None,
                                    generation_seed=seed,
                                )
                else:
                    # Expand modes with num_phi lists
                    expanded_modes = expand_modes_with_num_phi(steering_modes_dict)

                    training_cfg = model_config.get("polytope_training", {})
                    training_modes_dict = (
                        training_cfg.get("modes", {})
                        if isinstance(training_cfg, dict)
                        else {}
                    )
                    if expanded_modes and all(
                        not _stage4_mode_uses_generation_seed(
                            base_mode_name, mode_cfg, polytope_modes_dict
                        )
                        for base_mode_name, _num_phi, mode_cfg in expanded_modes
                    ):
                        if seeds != [None]:
                            self.logger.info(
                                "Stage 4 selected modes are seed-invariant; running once without generation seed"
                            )
                        seeds = [None]

                    skipped_unseeded_modes = set()
                    for seed_index, configured_seed in enumerate(seeds):
                        self.logger.info(
                            f"--- Stage 4: Evaluating with seed={configured_seed} ---"
                        )

                        for base_mode_name, num_phi, mode_cfg in expanded_modes:
                            mode_uses_generation_seed = (
                                _stage4_mode_uses_generation_seed(
                                    base_mode_name,
                                    mode_cfg,
                                    polytope_modes_dict,
                                )
                            )
                            if not mode_uses_generation_seed and seed_index > 0:
                                skipped_unseeded_modes.add(base_mode_name)
                                continue
                            seed = (
                                configured_seed if mode_uses_generation_seed else None
                            )
                            mode_label = f"{base_mode_name}_phi{num_phi}"
                            self.logger.info(
                                f"Stage 4 steering evaluation run (mode={mode_label}, seed={seed}) starting"
                            )

                            # Find polytope model path for this (base_mode, num_phi) combination
                            polytope_model_path = None
                            model_name_or_path_override = None
                            model_adapter_path_override = None
                            reft_r1_vector_path_override = None

                            # Check if this is a defense mode (doesn't need polytope model)
                            defense_cfg = mode_cfg.get("defense", {})
                            is_defense_mode = (
                                defense_cfg.get("use", False)
                                if isinstance(defense_cfg, dict)
                                else False
                            )

                            # Check if polytope_model_path is explicitly set to empty (for base model)
                            explicit_empty_path = (
                                mode_cfg.get("polytope_model_path", None) == ""
                            )
                            if explicit_empty_path:
                                polytope_model_path = ""
                                self.logger.info(
                                    f"Using base model (no polytope) for mode '{mode_label}'"
                                )

                            circuit_breaker_eval_cfg = mode_cfg.get(
                                "circuit_breaker", {}
                            )
                            use_circuit_breaker = isinstance(
                                circuit_breaker_eval_cfg, dict
                            ) and circuit_breaker_eval_cfg.get("use", False)
                            if use_circuit_breaker:
                                cb_modes = model_config.get(
                                    "polytope_training", {}
                                ).get("modes", {})
                                training_mode_name = circuit_breaker_eval_cfg.get(
                                    "training_mode"
                                )
                                cb_mode_cfg = (
                                    cb_modes.get(training_mode_name, {})
                                    if training_mode_name
                                    else {}
                                )
                                explicit_cb_path = circuit_breaker_eval_cfg.get(
                                    "model_path",
                                    cb_mode_cfg.get("circuit_breaker_model_path"),
                                )
                                if explicit_cb_path:
                                    model_adapter_path_override = os.path.abspath(
                                        explicit_cb_path
                                    )
                                else:
                                    base_output_dir = cb_mode_cfg.get("base_output_dir")
                                    base_exp_ident_suffix = cb_mode_cfg.get(
                                        "base_exp_ident_suffix",
                                        training_mode_name or "circuit_breaker",
                                    )
                                    if not base_output_dir:
                                        raise ValueError(
                                            f"Could not resolve CircuitBreaker output dir for mode '{mode_label}'"
                                        )
                                    model_adapter_path_override = (
                                        generate_circuit_breaker_model_path(
                                            base_output_dir,
                                            base_exp_ident_suffix,
                                            seed=seed,
                                        )
                                    )
                                polytope_model_path = ""
                                self.logger.info(
                                    f"Using CircuitBreaker adapter for mode '{mode_label}': {model_adapter_path_override}"
                                )

                            reft_r1_eval_cfg = mode_cfg.get("reft_r1", {})
                            use_reft_r1 = isinstance(
                                reft_r1_eval_cfg, dict
                            ) and reft_r1_eval_cfg.get("use", False)
                            if use_reft_r1:
                                reft_modes = model_config.get(
                                    "polytope_training", {}
                                ).get("modes", {})
                                training_mode_name = reft_r1_eval_cfg.get(
                                    "training_mode"
                                )
                                reft_mode_cfg = (
                                    reft_modes.get(training_mode_name, {})
                                    if training_mode_name
                                    else {}
                                )
                                explicit_reft_path = reft_r1_eval_cfg.get(
                                    "vector_path",
                                    reft_mode_cfg.get("reft_r1_vector_path"),
                                )
                                if explicit_reft_path:
                                    reft_r1_vector_path_override = os.path.abspath(
                                        explicit_reft_path
                                    )
                                else:
                                    base_output_dir = reft_mode_cfg.get(
                                        "base_output_dir"
                                    )
                                    base_exp_ident_suffix = reft_mode_cfg.get(
                                        "base_exp_ident_suffix",
                                        training_mode_name or "reft_r1",
                                    )
                                    if not base_output_dir:
                                        raise ValueError(
                                            f"Could not resolve ReFT-r1 vector path for mode '{mode_label}'"
                                        )
                                    reft_r1_vector_path_override = (
                                        generate_reft_r1_vector_path(
                                            base_output_dir,
                                            base_exp_ident_suffix,
                                            seed=seed,
                                        )
                                    )
                                polytope_model_path = ""
                                self.logger.info(
                                    f"Using ReFT-r1 vector for mode '{mode_label}': {reft_r1_vector_path_override}"
                                )

                            if (
                                not is_defense_mode
                                and not explicit_empty_path
                                and not use_circuit_breaker
                                and not use_reft_r1
                            ):
                                # Check if the mode config itself overrides the polytope path (e.g. from steering config)
                                explicit_polytope_path = mode_cfg.get(
                                    "polytope_model_path", None
                                )

                                if explicit_polytope_path:
                                    polytope_model_path = explicit_polytope_path
                                else:
                                    # Check if this mode has reuse_weights_from in the training config
                                    # (reuse_weights_from is in training config, not steering config)
                                    mode_training_cfg = polytope_modes_dict.get(
                                        base_mode_name, {}
                                    )
                                    reuse_from = mode_training_cfg.get(
                                        "reuse_weights_from", None
                                    )
                                    if reuse_from and reuse_from in polytope_modes_dict:
                                        if mode_training_cfg.get(
                                            "model_type"
                                        ) == "conditional_steering_vector" and mode_training_cfg.get(
                                            "reuse_steering_vector_from"
                                        ):
                                            base_output_dir = mode_training_cfg.get(
                                                "base_output_dir", None
                                            ) or training_cfg.get(
                                                "base_output_dir", None
                                            )
                                            base_exp_ident_suffix = (
                                                mode_training_cfg.get(
                                                    "base_exp_ident_suffix",
                                                    base_mode_name,
                                                )
                                            )
                                            if base_output_dir:
                                                seed_to_load = seed
                                                if seed_to_load is None:
                                                    global_polytope_config = (
                                                        self.config.get(
                                                            "polytope_training", {}
                                                        )
                                                    )
                                                    default_seed = (
                                                        global_polytope_config.get(
                                                            "seed", 5
                                                        )
                                                    )
                                                    seed_to_load = (
                                                        mode_training_cfg.get(
                                                            "seed",
                                                            training_cfg.get(
                                                                "seed", default_seed
                                                            ),
                                                        )
                                                    )
                                                polytope_model_path = (
                                                    generate_polytope_model_path(
                                                        base_output_dir,
                                                        base_exp_ident_suffix,
                                                        num_phi,
                                                        seed=seed_to_load,
                                                    )
                                                )
                                            else:
                                                polytope_model_path = (
                                                    mode_training_cfg.get(
                                                        "polytope_model_path", None
                                                    )
                                                )
                                        else:
                                            # Reuse weights from another mode
                                            source_mode_cfg = polytope_modes_dict[
                                                reuse_from
                                            ]
                                            # Try to get base_output_dir from source mode config first, then fall back to parent config
                                            base_output_dir = source_mode_cfg.get(
                                                "base_output_dir", None
                                            )
                                            if not base_output_dir:
                                                base_output_dir = training_cfg.get(
                                                    "base_output_dir", None
                                                )
                                            base_exp_ident_suffix = source_mode_cfg.get(
                                                "base_exp_ident_suffix", reuse_from
                                            )
                                            if base_output_dir:
                                                # Use the current loop seed for weight path resolution
                                                seed_to_load = seed
                                                if seed_to_load is None:
                                                    global_polytope_config = (
                                                        self.config.get(
                                                            "polytope_training", {}
                                                        )
                                                    )
                                                    default_seed = (
                                                        global_polytope_config.get(
                                                            "seed", 5
                                                        )
                                                    )
                                                    seed_to_load = source_mode_cfg.get(
                                                        "seed",
                                                        training_cfg.get(
                                                            "seed", default_seed
                                                        ),
                                                    )

                                                polytope_model_path = (
                                                    generate_polytope_model_path(
                                                        base_output_dir,
                                                        base_exp_ident_suffix,
                                                        num_phi,
                                                        seed=seed_to_load,
                                                    )
                                                )
                                            else:
                                                # Try to get explicit polytope_model_path from source mode
                                                polytope_model_path = (
                                                    source_mode_cfg.get(
                                                        "polytope_model_path", None
                                                    )
                                                )
                                    else:
                                        # Use this mode's own weights
                                        if base_mode_name in polytope_modes_dict:
                                            polytope_model_path = mode_training_cfg.get(
                                                "polytope_model_path", None
                                            )

                                        # If not explicitly set, try to generate from base_output_dir
                                        if not polytope_model_path:
                                            # Try to get base_output_dir from mode config first, then fall back to parent config
                                            base_output_dir = mode_training_cfg.get(
                                                "base_output_dir", None
                                            )
                                            if not base_output_dir:
                                                base_output_dir = training_cfg.get(
                                                    "base_output_dir", None
                                                )

                                            # Default suffix from training config if available for this mode name
                                            base_exp_ident_suffix = (
                                                mode_training_cfg.get(
                                                    "base_exp_ident_suffix",
                                                    base_mode_name,
                                                )
                                            )

                                            # Override for Multi-CBF: Use training_mode config if specified
                                            cbf_cfg_check = mode_cfg.get("cbf", {})
                                            if isinstance(cbf_cfg_check, dict):
                                                multi_cbf_check = cbf_cfg_check.get(
                                                    "multi_cbf", {}
                                                )
                                                if multi_cbf_check.get(
                                                    "enabled", False
                                                ):
                                                    training_mode_ref = (
                                                        multi_cbf_check.get(
                                                            "training_mode", None
                                                        )
                                                    )
                                                    print(
                                                        f"DEBUG: Multi-CBF enabled for {base_mode_name}. Requesting training mode: {training_mode_ref}"
                                                    )
                                                    print(
                                                        f"DEBUG: Available training modes: {list(polytope_modes_dict.keys())}"
                                                    )

                                                    if (
                                                        training_mode_ref
                                                        and training_mode_ref
                                                        in polytope_modes_dict
                                                    ):
                                                        ref_cfg = training_modes_dict[
                                                            training_mode_ref
                                                        ]
                                                        base_exp_ident_suffix = (
                                                            ref_cfg.get(
                                                                "base_exp_ident_suffix",
                                                                training_mode_ref,
                                                            )
                                                        )
                                                        print(
                                                            f"DEBUG: Found ref config. Suffix override: {base_exp_ident_suffix}"
                                                        )
                                                        # Also prefer explicit output dir from referenced mode if set
                                                        if ref_cfg.get(
                                                            "base_output_dir"
                                                        ):
                                                            base_output_dir = (
                                                                ref_cfg.get(
                                                                    "base_output_dir"
                                                                )
                                                            )
                                                    else:
                                                        print(
                                                            f"DEBUG: Training mode {training_mode_ref} NOT FOUND in training_modes_dict"
                                                        )

                                            if base_output_dir:
                                                # Use the current loop seed for weight path resolution
                                                seed_to_load = seed
                                                if seed_to_load is None:
                                                    # Look up the training config for this mode to find the seed
                                                    global_polytope_config = (
                                                        self.config.get(
                                                            "polytope_training", {}
                                                        )
                                                    )
                                                    default_seed = (
                                                        global_polytope_config.get(
                                                            "seed", 5
                                                        )
                                                    )
                                                    target_training_mode = (
                                                        training_mode_ref
                                                        if training_mode_ref
                                                        else base_mode_name
                                                    )
                                                    train_cfg_for_seed = (
                                                        polytope_modes_dict.get(
                                                            target_training_mode, {}
                                                        )
                                                    )
                                                    seed_to_load = (
                                                        train_cfg_for_seed.get(
                                                            "seed",
                                                            training_cfg.get(
                                                                "seed", default_seed
                                                            ),
                                                        )
                                                    )

                                                polytope_model_path = (
                                                    generate_polytope_model_path(
                                                        base_output_dir,
                                                        base_exp_ident_suffix,
                                                        num_phi,
                                                        seed=seed_to_load,
                                                    )
                                                )

                                    if (
                                        mode_training_cfg.get("model_type")
                                        == "conditional_steering_vector"
                                        and mode_training_cfg.get(
                                            "reuse_steering_vector_from"
                                        )
                                        and polytope_model_path
                                        and not os.path.exists(polytope_model_path)
                                    ):
                                        self.logger.info(
                                            f"CAST artifact missing for mode '{mode_label}' (seed={seed}); composing from reused weights"
                                        )
                                        self.polytope_training._compose_cast_artifact(
                                            model_name=model_name,
                                            model_config=model_config,
                                            training_config=mode_training_cfg,
                                            training_mode=base_mode_name,
                                            polytope_model_path=polytope_model_path,
                                            seed=seed,
                                        )

                                    # Check if multi-CBF is enabled for this mode
                                    cbf_cfg = mode_cfg.get("cbf", {})
                                    multi_cbf_enabled = False
                                    if isinstance(cbf_cfg, dict):
                                        multi_cbf_enabled = cbf_cfg.get(
                                            "multi_cbf", {}
                                        ).get("enabled", False)

                                    # Validate polytope model path only for non-defense modes
                                    path_exists = False
                                    if polytope_model_path:
                                        if multi_cbf_enabled:
                                            # For multi-CBF, we expect a directory of models, not a single file
                                            # If generate_polytope_model_path returned a file path, take use its directory
                                            if polytope_model_path.endswith(".pth"):
                                                polytope_model_path = os.path.dirname(
                                                    polytope_model_path
                                                )
                                            path_exists = os.path.isdir(
                                                polytope_model_path
                                            )
                                        else:
                                            # Standard single-model check
                                            path_exists = os.path.exists(
                                                polytope_model_path
                                            )

                                    if polytope_model_path is None or (
                                        polytope_model_path != "" and not path_exists
                                    ):
                                        error_msg = f"Polytope model path not found or does not exist for mode '{mode_label}' (seed={seed}): {polytope_model_path}. "
                                        if multi_cbf_enabled:
                                            error_msg += f" (Multi-CBF enabled, looked for directory)"
                                        error_msg += f" Please ensure the model was trained for this (mode, num_phi) combination."
                                        raise ValueError(error_msg)
                            else:
                                # For defense modes, use empty string as placeholder if needed
                                polytope_model_path = ""

                            # Logic to handle list of k values for CBF
                            # Resolve cbf config to check for k list
                            cbf_cfg = mode_cfg.get("cbf", {})
                            if not isinstance(cbf_cfg, dict):
                                cbf_cfg = {}

                            # Check global steering config if not in mode config (fallback)
                            global_cbf_cfg = steering_cfg.get("cbf", {})
                            if not cbf_cfg and isinstance(global_cbf_cfg, dict):
                                # Only use global for detection if mode specific is empty/missing
                                # Note: Config merging usually happens later, but we need to know if we should iterate NOW.
                                # If 'k' is in global but not local, we should probably check it.
                                pass

                            # Better approach: Check if 'k' is a list in either mode cbf or global cbf (if mode doesn't override)
                            k_val = cbf_cfg.get("k", None)

                            # If k not in mode config, check global
                            if k_val is None:
                                k_val = steering_cfg.get("cbf", {}).get("k", None)

                            tasks_to_run = []
                            if isinstance(k_val, list):
                                self.logger.info(
                                    f"Iterating over k values: {k_val} for mode {mode_label}"
                                )
                                for k in k_val:
                                    # Deep copy the mode config to avoid shared state
                                    new_mode_cfg = copy.deepcopy(mode_cfg)

                                    # Ensure cbf dict exists (might have been missing in mode_cfg)
                                    if "cbf" not in new_mode_cfg:
                                        new_mode_cfg["cbf"] = {}

                                    # If cbf was just a boolean or something, make it a dict (unlikely if we found k list, but safe)
                                    if not isinstance(new_mode_cfg["cbf"], dict):
                                        new_mode_cfg["cbf"] = (
                                            {"use": new_mode_cfg["cbf"]}
                                            if new_mode_cfg["cbf"]
                                            else {}
                                        )

                                    new_mode_cfg["cbf"]["k"] = k

                                    # Update suffix
                                    base_suffix = new_mode_cfg.get(
                                        "exp_ident_suffix", mode_label
                                    )
                                    new_suffix = f"{base_suffix}_k_{k}"
                                    new_mode_cfg["exp_ident_suffix"] = new_suffix
                                    # Also update base_exp_ident_suffix because SteeringEvaluationStage prefers it
                                    if "base_exp_ident_suffix" in new_mode_cfg:
                                        new_mode_cfg["base_exp_ident_suffix"] = (
                                            new_suffix
                                        )

                                    tasks_to_run.append(new_mode_cfg)
                            else:
                                # Single run (k is float/int or missing)
                                tasks_to_run.append(mode_cfg)

                            for final_mode_cfg in tasks_to_run:
                                # Temporarily update the mode config with the expanded version
                                modified_model_config = model_config.copy()
                                modified_steering_cfg = steering_cfg.copy()
                                modified_steering_cfg["modes"] = {
                                    base_mode_name: final_mode_cfg
                                }
                                if not mode_uses_generation_seed:
                                    # Prevent SteeringEvaluationStage from falling back to the
                                    # model-level generation_seed list for deterministic baselines.
                                    modified_steering_cfg.pop("generation_seed", None)
                                modified_model_config["steering"] = (
                                    modified_steering_cfg
                                )

                                self.steering_evaluation.run(
                                    model_name,
                                    polytope_model_path,
                                    modified_model_config,
                                    steering_mode=base_mode_name,
                                    generation_seed=seed,
                                    model_name_or_path_override=model_name_or_path_override,
                                    model_adapter_path_override=model_adapter_path_override,
                                    reft_r1_vector_path_override=reft_r1_vector_path_override,
                                )
                    if skipped_unseeded_modes:
                        self.logger.info(
                            "Stage 4 seed-invariant modes ran once without generation seed: %s",
                            sorted(skipped_unseeded_modes),
                        )
                self.logger.info("Stage 4 completed successfully")
            except Exception as e:
                self.logger.error(f"Stage 4 failed: {e}")
                raise

        self.logger.info(f"Pipeline completed successfully for {model_name}")

    def list_modes(self, model_name: str) -> List[Dict[str, Any]]:
        """Return configured Stage 3/4 runnable modes for a model."""
        model_config = get_model_config(self.config, model_name)
        training_cfg = model_config.get("polytope_training", {})
        training_modes = (
            training_cfg.get("modes", {})
            if isinstance(training_cfg, dict)
            and isinstance(training_cfg.get("modes", {}), dict)
            else {}
        )
        steering_cfg = model_config.get("steering", {})
        steering_modes = (
            steering_cfg.get("modes", {})
            if isinstance(steering_cfg, dict)
            and isinstance(steering_cfg.get("modes", {}), dict)
            else {}
        )

        rows: List[Dict[str, Any]] = []
        for mode_name, mode_cfg in training_modes.items():
            rows.append(
                {
                    "stage": 3,
                    "mode": mode_name,
                    "type": _mode_type_from_training(mode_cfg),
                }
            )
        for mode_name, mode_cfg in steering_modes.items():
            rows.append(
                {
                    "stage": 4,
                    "mode": mode_name,
                    "type": _mode_type_from_steering(
                        mode_name, mode_cfg, training_modes
                    ),
                }
            )
        return rows

    def run_all_models(
        self,
        stages: Optional[List[int]] = None,
        selected_modes: Optional[List[str]] = None,
        selected_mode_types: Optional[List[str]] = None,
    ) -> None:
        """
        Run pipeline for all configured models

        Args:
            stages: List of stage numbers to run (1-4). If None, runs all stages.
        """
        models = [model["name"] for model in self.config["models"]]
        self.logger.info(f"Running pipeline for all models: {models}")

        for model_name in models:
            try:
                self.run(
                    model_name,
                    stages,
                    selected_modes=selected_modes,
                    selected_mode_types=selected_mode_types,
                )
            except Exception as e:
                self.logger.error(f"Pipeline failed for {model_name}: {e}")
                # Continue with next model
                continue


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="HarmBench Pipeline Orchestrator")
    parser.add_argument(
        "--config", required=True, help="Path to pipeline configuration file"
    )
    parser.add_argument(
        "--model",
        required=True,
        help='Model name to process, or "all" for all models',
    )
    parser.add_argument(
        "--stages",
        help="Comma-separated list of stages to run (1-4). Default: all stages",
    )
    parser.add_argument(
        "--mode-types",
        help=(
            "Comma-separated Stage 3/4 mode types to run, e.g. circuit_breaker,reft_r1,polytope. "
            "Defaults to pipeline.mode_types from the config, or all modes if unset."
        ),
    )
    parser.add_argument(
        "--modes",
        help=(
            "Comma-separated Stage 3/4 mode names to run, e.g. circuit_breaker,cbf_qp_nonlinear. "
            "Defaults to pipeline.modes from the config, or all modes if unset."
        ),
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="List configured Stage 3/4 modes and inferred types for --model, then exit.",
    )

    args = parser.parse_args()

    # Parse stages
    stages = None
    if args.stages:
        try:
            stages = [int(s.strip()) for s in args.stages.split(",")]
            for stage in stages:
                if stage not in [1, 2, 3, 4]:
                    raise ValueError(f"Invalid stage number: {stage}")
        except ValueError as e:
            print(f"Error parsing stages: {e}")
            sys.exit(1)

    # Initialize pipeline
    try:
        pipeline = HarmBenchPipeline(args.config)
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        sys.exit(1)

    selected_modes = _normalize_selection(args.modes)
    selected_mode_types = _normalize_selection(args.mode_types)

    if args.list_modes:
        models = (
            [model["name"] for model in pipeline.config["models"]]
            if args.model.lower() == "all"
            else [args.model]
        )
        for model_name in models:
            print(f"# {model_name}")
            for row in pipeline.list_modes(model_name):
                print(f"stage={row['stage']}\tmode={row['mode']}\ttype={row['type']}")
        return

    # Run pipeline
    try:
        if args.model.lower() == "all":
            pipeline.run_all_models(
                stages,
                selected_modes=selected_modes,
                selected_mode_types=selected_mode_types,
            )
        else:
            pipeline.run(
                args.model,
                stages,
                selected_modes=selected_modes,
                selected_mode_types=selected_mode_types,
            )
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        sys.exit(1)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
