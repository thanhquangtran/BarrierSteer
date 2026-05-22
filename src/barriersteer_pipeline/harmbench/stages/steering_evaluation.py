"""
Stage 5: Steering Evaluation
Evaluates polytope effectiveness through steering experiments using HarmBench's step_2_and_3.sh script.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils.config_utils import (
    generate_polytope_model_path,
    get_harmbench_model_name,
    get_results_directory,
)


class SteeringEvaluationStage:
    """Stage 5: Evaluate polytope effectiveness with steering"""

    def __init__(self, config: Dict[str, Any], harmbench_path: str):
        self.config = config
        self.harmbench_path = Path(harmbench_path)
        self.logger = logging.getLogger(__name__)
        self.barriersteer_path = Path(self.config["pipeline"]["barriersteer_path"])

        # Get execution mode and configuration
        steering_config = config.get("steering_evaluation", {})
        self.part_size = steering_config.get("part_size", 500)
        self.execution_mode = steering_config.get("execution_mode", "local")
        # vLLM memory settings
        self.vllm_gpu_memory_utilization = steering_config.get(
            "vllm_gpu_memory_utilization", 0.7
        )
        self.vllm_enforce_eager = steering_config.get("vllm_enforce_eager", True)
        self.vllm_max_num_seqs = steering_config.get("vllm_max_num_seqs", 64)

    def run(
        self,
        model_name: str,
        polytope_model_path: str,
        model_config: Dict[str, Any],
        steering_mode: Optional[str] = None,
        generation_seed: Optional[int] = None,
        model_name_or_path_override: Optional[str] = None,
        model_adapter_path_override: Optional[str] = None,
        reft_r1_vector_path_override: Optional[str] = None,
    ) -> None:
        """
        Evaluate polytope effectiveness through steering

        Args:
            model_name: Name of the model
            polytope_model_path: Path to trained polytope model
            model_config: Model-specific configuration
            generation_seed: Optional seed for generation (overrides config if provided)
        """
        self.logger.info(f"Starting steering evaluation for {model_name}")

        try:
            # Validate inputs
            self._validate_inputs(model_name, polytope_model_path, model_config)

            # Get HarmBench model name
            harmbench_model_name = get_harmbench_model_name(model_name)

            # Get evaluation methods from steering config
            steering_config = model_config.get("steering", {})
            evaluation_methods = steering_config.get("evaluation_methods", [])

            # Optional per-mode overrides (supports cbf vs no_cbf)
            mode_cfg: Dict[str, Any] = {}
            modes = steering_config.get("modes", {})
            if steering_mode and isinstance(modes, dict) and steering_mode in modes:
                mode_cfg = modes.get(steering_mode, {}) or {}
                if mode_cfg:
                    self.logger.info(
                        f"Using steering evaluation mode '{steering_mode}' overrides"
                    )

            cbf_cfg = mode_cfg.get("cbf", steering_config.get("cbf", None))

            # Defense configuration (e.g., self_reminder, smoothllm, response_check)
            defense_cfg = mode_cfg.get("defense", steering_config.get("defense", None))

            if "evaluation_methods" in mode_cfg:
                evaluation_methods = mode_cfg.get(
                    "evaluation_methods", evaluation_methods
                )

            if not evaluation_methods:
                raise ValueError(f"No evaluation methods configured for {model_name}")

            # Create timestamp for unique identification (for HarmBench script)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            exp_suffix = mode_cfg.get(
                "base_exp_ident_suffix", mode_cfg.get("exp_ident_suffix", None)
            )

            # Use provided generation_seed, otherwise fall back to config
            if generation_seed is None:
                generation_seed = steering_config.get("generation_seed", None)

            exp_ident = f"steering_{timestamp}"
            if exp_suffix:
                exp_ident = f"{exp_ident}_{exp_suffix}"

            # Auto-append seed suffix if generation_seed is set
            if generation_seed is not None:
                exp_ident = f"{exp_ident}_seed{generation_seed}"
                self.logger.info(f"Using generation seed: {generation_seed}")

            # Execute the step_2_and_3.sh command
            self._run_harmbench_script(
                harmbench_model_name,
                polytope_model_path,
                evaluation_methods,
                exp_ident,
                cbf_cfg=cbf_cfg,
                defense_cfg=defense_cfg,
                generation_seed=generation_seed,
                model_name_or_path_override=model_name_or_path_override,
                model_adapter_path_override=model_adapter_path_override,
                reft_r1_vector_path_override=reft_r1_vector_path_override,
            )

            # Consolidate per-run artifacts into SafeLLM/outputs for easy browsing
            out_dir = self._consolidate_run_outputs(
                model_name=model_name,
                evaluation_methods=evaluation_methods,
                exp_ident=exp_ident,
            )
            self.logger.info(f"Consolidated run outputs to: {out_dir}")

            self.logger.info("Steering evaluation completed")

        except Exception as e:
            self.logger.error(f"Steering evaluation failed: {e}")
            raise

    def _validate_inputs(
        self,
        model_name: str,
        polytope_model_path: str,
        model_config: Dict[str, Any],
    ) -> None:
        """Validate inputs before running evaluation."""
        # Polytope model path is optional for defense-only runs
        if polytope_model_path and not Path(polytope_model_path).exists():
            raise FileNotFoundError(f"Polytope model not found: {polytope_model_path}")

        if not self.harmbench_path.exists():
            raise FileNotFoundError(f"HarmBench path not found: {self.harmbench_path}")

        if "steering" not in model_config:
            raise ValueError(f"Steering configuration missing for model: {model_name}")

        if "attack_methods" not in model_config:
            raise ValueError(f"Attack methods not configured for model: {model_name}")

    def _run_harmbench_script(
        self,
        harmbench_model_name: str,
        polytope_model_path: str,
        evaluation_methods: list,
        exp_ident: str,
        cbf_cfg: Optional[Dict[str, Any]] = None,
        defense_cfg: Optional[Dict[str, Any]] = None,
        generation_seed: Optional[int] = None,
        model_name_or_path_override: Optional[str] = None,
        model_adapter_path_override: Optional[str] = None,
        reft_r1_vector_path_override: Optional[str] = None,
    ) -> None:
        """
        Run HarmBench's step_2_and_3.sh script to perform generation and evaluation.
        The script handles both local and slurm execution modes internally.

        Args:
            harmbench_model_name: HarmBench-compatible model name
            polytope_model_path: Path to polytope weights
            evaluation_methods: List of evaluation methods
            exp_ident: Experiment identifier for output files
            cbf_cfg: Optional CBF configuration
            defense_cfg: Optional defense method configuration (self_reminder, smoothllm, etc.)
        """
        if polytope_model_path and not os.path.isabs(polytope_model_path):
            polytope_model_path = os.path.abspath(polytope_model_path)
        if model_name_or_path_override and not os.path.isabs(
            model_name_or_path_override
        ):
            model_name_or_path_override = os.path.abspath(model_name_or_path_override)
        if model_adapter_path_override and not os.path.isabs(
            model_adapter_path_override
        ):
            model_adapter_path_override = os.path.abspath(model_adapter_path_override)
        if reft_r1_vector_path_override and not os.path.isabs(
            reft_r1_vector_path_override
        ):
            reft_r1_vector_path_override = os.path.abspath(reft_r1_vector_path_override)

        # Convert evaluation methods list to comma-separated string
        algorithms = ",".join(evaluation_methods)

        # Build command for step_2_and_3.sh
        script_path = self.harmbench_path / "scripts" / "step_2_and_3.sh"

        # Parameters for the script:
        # $1: MODEL
        # $2: MODE (local or sbatch)
        # $3: ALGORITHMS (comma-separated)
        # $4: EXP_IDENT (optional, for filename)
        # $5: PART_SIZE (default 500)
        # $6: POLYTOPE_PATH (path to polytope weights)
        # $7: USE_DEFENSE (True for defense methods, False otherwise)
        # $8: DEFENSE_METHOD (self_reminder, smoothllm, response_check, etc.)

        # Determine defense settings from defense_cfg
        use_defense = "False"
        defense_method = "None"
        if defense_cfg is not None:
            use_defense = str(defense_cfg.get("use", False))
            defense_method = defense_cfg.get("method", "None")
            self.logger.info(f"Using defense method: {defense_method}")

        cmd = [
            "bash",
            str(script_path),
            harmbench_model_name,
            self.execution_mode,  # Pass execution mode to the script
            algorithms,
            exp_ident,
            str(self.part_size),
            (
                polytope_model_path if polytope_model_path else ""
            ),  # Can be empty for defense-only runs
            use_defense,
            defense_method,
        ]

        # Get behavior dataset for evaluation from config
        eval_config = self.config.get("evaluation", {})
        behavior_dataset = eval_config.get(
            "behavior_dataset",
            "./data/behavior_datasets/harmbench_behaviors_text_all.csv",
        )
        # Convert to absolute path relative to HarmBench directory
        if not os.path.isabs(behavior_dataset):
            behavior_dataset = os.path.join(
                self.harmbench_path, behavior_dataset.lstrip("./")
            )
        self.logger.info(f"Using behavior dataset for evaluation: {behavior_dataset}")

        # Optional CBF args (positions 9..19); always pass (empty if not configured)
        if cbf_cfg is not None:
            use_cbf = cbf_cfg.get("use_cbf", cbf_cfg.get("use", False))
            control_radius = str(
                cbf_cfg.get("control_radius", 1.0)
            )  # Extract for all CBF modes
            cbf_num_steps = str(
                cbf_cfg.get("num_steps", cbf_cfg.get("cbf_num_steps", 1))
            )
            cmd.extend(
                [
                    str(bool(use_cbf)),  # USE_CBF (position 9)
                    str(
                        cbf_cfg.get("mode", cbf_cfg.get("cbf_mode", "estimated"))
                    ),  # 10
                    str(cbf_cfg.get("dt", cbf_cfg.get("cbf_dt", 1.0))),  # 11
                    str(cbf_cfg.get("k", cbf_cfg.get("cbf_k", 1.0))),  # 12
                    str(cbf_cfg.get("w", cbf_cfg.get("cbf_w", 1.0))),  # 13
                    str(cbf_cfg.get("p", cbf_cfg.get("cbf_p", 10.0))),  # 14
                    str(
                        cbf_cfg.get(
                            "max_constraints",
                            cbf_cfg.get(
                                "cbf_max_constraints", cbf_cfg.get("cbf_max", 2)
                            ),
                        )
                    ),  # 15
                    str(
                        cbf_cfg.get(
                            "constraint_mode",
                            cbf_cfg.get("cbf_constraint_mode", "topk"),
                        )
                    ),  # 16
                    str(
                        cbf_cfg.get(
                            "kappa",
                            cbf_cfg.get("cbf_kappa", 10.0),
                        )
                    ),  # 17
                    control_radius,  # 18 (for all CBF modes)
                    cbf_num_steps,  # 19
                ]
            )
        else:
            # Pass empty strings for CBF args to reach position 19 (num_steps)
            control_radius = "1.0"
            cmd.extend([""] * 9 + [control_radius, "1"])  # positions 9-19

        # Add behavior dataset (position 20)
        cmd.append(behavior_dataset)

        # Add starting batch size (position 21)
        starting_batch_size = self.config.get("pipeline", {}).get(
            "starting_batch_size", 16
        )
        cmd.append(str(starting_batch_size))

        # Add multi-CBF arguments (positions 22-24)
        multi_cbf_enabled = "False"
        multi_cbf_models_dir = ""

        if cbf_cfg is not None:
            multi_cbf = cbf_cfg.get("multi_cbf", {})
            if multi_cbf.get("enabled", False):
                multi_cbf_enabled = "True"
                # If polytope_model_path is passed (resolved by pipeline), use it as models directory
                if polytope_model_path:
                    multi_cbf_models_dir = polytope_model_path
                else:
                    # Fallback to direct config if not passed
                    multi_cbf_models_dir = multi_cbf.get("models_dir", "")

                if not multi_cbf_models_dir:
                    raise ValueError(
                        "multi_cbf.models_dir could not be resolved. Please ensure training_mode or models_dir is specified."
                    )

                # Ensure models dir is absolute (relative to CWD/SafeLLM if not already)
                if multi_cbf_models_dir and not os.path.isabs(multi_cbf_models_dir):
                    multi_cbf_models_dir = os.path.abspath(multi_cbf_models_dir)

                self.logger.info(
                    f"Multi-CBF enabled. Using models from: {multi_cbf_models_dir}"
                )

        multi_cbf_load_attacks = ""
        if cbf_cfg is not None:
            multi_cbf = cbf_cfg.get("multi_cbf", {})
            if multi_cbf.get("enabled", False):
                attacks = multi_cbf.get("load_attacks", [])
                if attacks:
                    multi_cbf_load_attacks = json.dumps(attacks)
                    self.logger.info(
                        f"Loading specific attacks: {multi_cbf_load_attacks}"
                    )

        cmd.append(multi_cbf_enabled)
        cmd.append(multi_cbf_models_dir)
        cmd.append(multi_cbf_load_attacks)

        # Add generation_seed (position 25)
        cmd.append(str(generation_seed) if generation_seed is not None else "")

        # Optional custom model path (position 26)
        cmd.append(model_name_or_path_override or "")
        # Optional LoRA adapter path (position 27)
        cmd.append(model_adapter_path_override or "")
        # Optional ReFT-r1 vector path (position 28)
        cmd.append(reft_r1_vector_path_override or "")

        # Set vLLM memory environment variables EARLY (before HarmBench imports vLLM)
        # These must be set in os.environ (not just subprocess env) because vLLM reads them at import time
        os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = str(
            self.vllm_gpu_memory_utilization
        )
        os.environ["VLLM_ENFORCE_EAGER"] = "1" if self.vllm_enforce_eager else "0"
        os.environ["VLLM_MAX_NUM_SEQS"] = str(self.vllm_max_num_seqs)

        # Set HARMBENCH_RESULTS_DIR from pipeline config
        base_save_dir = self.config.get("pipeline", {}).get("base_save_dir", "results")
        # Ensure it's absolute path
        if not os.path.isabs(base_save_dir):
            base_save_dir = os.path.join(self.harmbench_path, base_save_dir)
        if not os.path.isabs(base_save_dir):
            base_save_dir = os.path.join(self.harmbench_path, base_save_dir)
        os.environ["HARMBENCH_RESULTS_DIR"] = base_save_dir
        os.environ["PYTHON_EXECUTABLE"] = sys.executable
        self.logger.info(f"Setting HARMBENCH_RESULTS_DIR={base_save_dir}")

        # Note: Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here
        # as it's incompatible with vLLM's multi-process workers

        self.logger.info(
            f"vLLM memory settings (set in os.environ): "
            f"GPU_MEMORY_UTILIZATION={os.environ.get('VLLM_GPU_MEMORY_UTILIZATION')}, "
            f"ENFORCE_EAGER={os.environ.get('VLLM_ENFORCE_EAGER')}, "
            f"MAX_NUM_SEQS={os.environ.get('VLLM_MAX_NUM_SEQS')}"
        )

        # If CUDA_VISIBLE_DEVICES not set, warn user to use GPU 0
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            self.logger.warning(
                "CUDA_VISIBLE_DEVICES not set. GPU 2 is almost full (38.93 GiB used)."
            )
            self.logger.warning(
                "RECOMMENDED: Set CUDA_VISIBLE_DEVICES=0 before running Stage 4"
            )
            self.logger.warning(
                "Example: CUDA_VISIBLE_DEVICES=0 python .../run_harmbench_pipeline.py --stages 4"
            )
        else:
            self.logger.info(
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )

        # Change to HarmBench directory for execution
        original_dir = os.getcwd()
        os.chdir(self.harmbench_path)

        # Copy environment for subprocess (includes all the vLLM settings we just set)
        env = os.environ.copy()

        try:
            # Run the script - it handles local vs slurm internally
            self.logger.info(f"Running HarmBench script: {' '.join(cmd)}")
            # Don't capture output so we can see errors in real-time
            result = subprocess.run(cmd, check=True, env=env)

            self.logger.info("HarmBench script completed successfully")

        finally:
            os.chdir(original_dir)

    def _consolidate_run_outputs(
        self,
        model_name: str,
        evaluation_methods: List[str],
        exp_ident: str,
    ) -> Path:
        """
        Create a per-run folder in SafeLLM/outputs and copy in ONLY:
        - merged completions json
        - results json
        for each evaluated method.
        """
        date_dir = time.strftime("%Y-%m-%d")

        # Use configured output directory if available, else default to outputs/harmbench
        configured_out_dir = self.config.get("pipeline", {}).get("output_dir", None)
        if configured_out_dir:
            # If configured path is absolute, use it. If relative, make it relative to barriersteer_path
            if os.path.isabs(configured_out_dir):
                out_base = Path(configured_out_dir)
            else:
                out_base = self.barriersteer_path / configured_out_dir
        else:
            out_base = self.barriersteer_path / "outputs" / "harmbench"

        out_root = out_base / date_dir

        # Extract mode name and timestamp from exp_ident (format: steering_YYYYMMDD_HHMMSS_MODE_NAME)
        # We want directory to end with MODE_NAME-TIMESTAMP to match training format (like training: -HH-MM-SS)
        mode_name = exp_ident
        dir_timestamp = ""
        if exp_ident.startswith("steering_"):
            # Extract timestamp and mode name (format: steering_YYYYMMDD_HHMMSS_MODE_NAME)
            parts = exp_ident.split(
                "_", 3
            )  # Split into: ["steering", "YYYYMMDD", "HHMMSS", "MODE_NAME"]
            if len(parts) >= 4:
                mode_name = parts[3]  # Get the mode name part
                # Convert HHMMSS to HH-MM-SS format (matching Hydra timestamp format)
                time_str = parts[2]  # HHMMSS
                if len(time_str) == 6:
                    dir_timestamp = f"{time_str[0:2]}-{time_str[2:4]}-{time_str[4:6]}"

        # Create directory name ending with mode_name-timestamp (matching training format)
        if dir_timestamp:
            run_dir = (
                out_root
                / f"harmbench_eval_{get_harmbench_model_name(model_name)}_{mode_name}-{dir_timestamp}"
            )
        else:
            # Fallback if we can't parse the format
            run_dir = (
                out_root
                / f"harmbench_eval_{get_harmbench_model_name(model_name)}_{mode_name}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)

        missing: List[Tuple[str, str]] = []

        for method in evaluation_methods:
            method_dir = run_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)

            results_dir = get_results_directory(
                self.harmbench_path, method, model_name, transform_model_name=True
            )
            completions_dir = results_dir / "completions"
            eval_results_dir = results_dir / "results"

            merged_completion, merged_timing = self._find_merged_completion_files(
                completions_dir, exp_ident
            )
            results_file = self._find_results_file(eval_results_dir, exp_ident)

            if merged_completion is None:
                missing.append(
                    (method, f"merged completions for exp_ident={exp_ident}")
                )
            else:
                shutil.copy2(merged_completion, method_dir / "merged_completions.json")

            if merged_timing is None:
                self.logger.warning(
                    f"Missing merged timing for method {method} (exp_ident={exp_ident}). "
                    "Skipping timing artifact."
                )
            else:
                shutil.copy2(merged_timing, method_dir / "merged_timing.json")

            if results_file is None:
                missing.append((method, f"results for exp_ident={exp_ident}"))
            else:
                shutil.copy2(results_file, method_dir / "results.json")

        if missing:
            details = ", ".join([f"{m}:{what}" for m, what in missing])
            raise RuntimeError(
                "Could not consolidate all HarmBench artifacts for this run. "
                f"Missing: {details}"
            )

        return run_dir

    def _find_merged_completion_files(
        self, completions_dir: Path, exp_ident: str
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Find both '*_merged.json' and '*_merged_timing.json' (distinct artifacts).
        Match by exp_ident substring.
        """
        if not completions_dir.exists():
            return None, None

        candidates = [p for p in completions_dir.glob("*.json") if exp_ident in p.name]
        merged = [
            p for p in candidates if "_merged" in p.stem and "_part" not in p.stem
        ]

        merged_json = [p for p in merged if p.name.endswith("_merged.json")]
        merged_timing_json = [
            p for p in merged if p.name.endswith("_merged_timing.json")
        ]

        merged_out = (
            max(merged_json, key=lambda p: p.stat().st_mtime) if merged_json else None
        )
        timing_out = (
            max(merged_timing_json, key=lambda p: p.stat().st_mtime)
            if merged_timing_json
            else None
        )
        return merged_out, timing_out

    def _find_results_file(self, results_dir: Path, exp_ident: str) -> Optional[Path]:
        """
        Match by exp_ident substring, exclude merged/part/timing artifacts if any.
        """
        if not results_dir.exists():
            return None

        candidates = [p for p in results_dir.glob("*.json") if exp_ident in p.name]
        candidates = [
            p
            for p in candidates
            if (
                "_merged" not in p.stem
                and "_part" not in p.stem
                and "timing" not in p.stem
            )
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)
