"""
Stage 1: Complete HarmBench Pipeline
Runs the complete HarmBench evaluation pipeline using run_pipeline.py.
"""

import logging
import os
import subprocess
import sys
from typing import Any, Dict, List


class AttackGenerationStage:
    """Stage 1: Run complete HarmBench evaluation pipeline"""

    def __init__(self, config: Dict[str, Any], harmbench_path: str):
        self.config = config
        self.harmbench_path = harmbench_path
        self.logger = logging.getLogger(__name__)

    def run(self, model_name: str, attack_methods: List[str]) -> None:
        """
        Run complete HarmBench evaluation pipeline using run_pipeline.py

        Args:
            model_name: Name of the model (e.g., 'qwen_1.5b')
            attack_methods: List of methods to use (includes both attack methods and evaluation methods)
        """
        self.logger.info(
            f"Starting complete HarmBench pipeline for {model_name} with methods {attack_methods}"
        )

        # Change to HarmBench directory
        original_dir = os.getcwd()
        os.chdir(self.harmbench_path)

        try:
            # Step 1: Generate test cases
            self._run_harmbench_pipeline(model_name, attack_methods, step="1")

            # Step 1.5: Merge test cases
            self._run_harmbench_pipeline(model_name, attack_methods, step="1.5")

            self._validate_pipeline_results(model_name, attack_methods)

        finally:
            os.chdir(original_dir)

    def _run_harmbench_pipeline(
        self, model_name: str, attack_methods: List[str], step: str = "1"
    ) -> None:
        """
        Run complete HarmBench pipeline using run_pipeline.py (all steps: test cases, completions, evaluation)

        Args:
            model_name: Name of the model
            attack_methods: List of methods (includes both attack methods and evaluation methods)
            step: Step to run (1 or 1.5)
        """
        self.logger.info(
            f"Running HarmBench pipeline (step={step}) for {model_name} with methods {attack_methods}"
        )

        # Prepare arguments for run_pipeline.py
        methods_str = ",".join(attack_methods)
        attack_config = self.config.get("attack_generation", {})
        pipeline_config = self.config.get("pipeline", {})
        slurm_config = self.config.get("slurm", {})

        # Convert relative paths to absolute paths
        base_save_dir = pipeline_config.get("base_save_dir", "results")
        if not os.path.isabs(base_save_dir):
            base_save_dir = os.path.join(self.harmbench_path, base_save_dir)

        base_log_dir = pipeline_config.get("base_log_dir", "slurm_logs")
        if not os.path.isabs(base_log_dir):
            base_log_dir = os.path.join(self.harmbench_path, base_log_dir)

        behaviors_path = attack_config.get(
            "behavior_dataset",
            "data/behavior_datasets/harmbench_behaviors_text_val.csv",
        )
        if not os.path.isabs(behaviors_path):
            behaviors_path = os.path.join(self.harmbench_path, behaviors_path)

        cmd = [
            "python",
            os.path.join(self.harmbench_path, "scripts/run_pipeline.py"),
            "--methods",
            methods_str,
            "--models",
            model_name,
            "--step",
            step,
            "--mode",
            attack_config.get("execution_mode", "slurm"),
            "--base_save_dir",
            base_save_dir,
            "--base_log_dir",
            base_log_dir,
            "--behaviors_path",
            behaviors_path,
            "--partition",
            slurm_config.get("partition", "gpu"),
            "--cls_path",
            pipeline_config.get("cls_path", "cais/HarmBench-Llama-2-13b-cls"),
            "--starting_batch_size",
            str(pipeline_config.get("starting_batch_size", 16)),
        ]

        # Add vLLM evaluation settings if present in config
        if "vllm_gpu_memory_utilization" in pipeline_config:
            cmd.extend(
                [
                    "--gpu_memory_utilization",
                    str(pipeline_config["vllm_gpu_memory_utilization"]),
                ]
            )
        if "vllm_enforce_eager" in pipeline_config:
            cmd.extend(["--enforce_eager", str(pipeline_config["vllm_enforce_eager"])])
        if "vllm_max_num_seqs" in pipeline_config:
            cmd.extend(["--max_num_seqs", str(pipeline_config["vllm_max_num_seqs"])])

        # Add verbosity when running locally to see progress
        if attack_config.get("execution_mode", "slurm") == "local":
            cmd.append("--verbose")

        # Add overwrite flag if configured
        if attack_config.get("overwrite", False):
            cmd.append("--overwrite")
        else:
            # If not overwriting, enable incremental update for completions
            cmd.append("--incremental_update")

        # Add reverse flag if configured
        if attack_config.get("reverse", False):
            cmd.append("--reverse")
            self.logger.info("Enabling reverse generation order")

        self.logger.info(f"Executing HarmBench pipeline: {' '.join(cmd)}")

        # Ensure CUDA_VISIBLE_DEVICES is passed to subprocess
        env = os.environ.copy()
        if "CUDA_VISIBLE_DEVICES" in env:
            self.logger.info(
                f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} is set"
            )
        # Run the complete pipeline, streaming stdout/stderr to terminal
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            self.logger.error("HarmBench pipeline failed. See above logs for details.")
            raise RuntimeError("HarmBench pipeline failed") from e

        self.logger.info(f"Successfully completed HarmBench pipeline for {model_name}")

    def _validate_pipeline_results(
        self, model_name: str, attack_methods: List[str]
    ) -> None:
        """
        Validate that HarmBench pipeline completed successfully by checking
        that actual result files (not just directories) were created.

        Args:
            model_name: Name of the model
            attack_methods: List of attack methods
        """
        pipeline_config = self.config.get("pipeline", {})
        base_save_dir = pipeline_config.get("base_save_dir", "results")

        # Convert to absolute path
        if not os.path.isabs(base_save_dir):
            base_save_dir = os.path.join(self.harmbench_path, base_save_dir)

        failed_methods = []

        # Load run_pipeline.yaml to get experiment_name_template and class_name for each method
        import yaml

        pipeline_yaml_path = os.path.join(
            self.harmbench_path, "configs", "pipeline_configs", "run_pipeline.yaml"
        )
        method_templates = {}
        method_class_names = {}
        if os.path.exists(pipeline_yaml_path):
            with open(pipeline_yaml_path, "r") as f:
                run_pipeline_config = yaml.safe_load(f)
            for method_name, method_cfg in run_pipeline_config.items():
                if isinstance(method_cfg, dict):
                    if "experiment_name_template" in method_cfg:
                        method_templates[method_name] = method_cfg[
                            "experiment_name_template"
                        ]
                    if "class_name" in method_cfg:
                        method_class_names[method_name] = method_cfg["class_name"]

        for method in attack_methods:
            # Determine the correct subdirectory based on experiment_name_template
            template = method_templates.get(method, "<model_name>")
            if "<model_name>" in template:
                subdir = template.replace("<model_name>", model_name)
            else:
                subdir = template

            # HarmBench stores results under class_name folder when it differs
            # from the method key (e.g., PAP-top5 -> class_name PAP -> results/PAP/top_5/)
            method_folder = method_class_names.get(method, method)

            method_failed = False

            # Check for test cases file
            test_case_file = (
                f"{base_save_dir}/{method_folder}/{subdir}/test_cases/test_cases.json"
            )
            if not os.path.exists(test_case_file):
                self.logger.warning(
                    f"{method}: Test cases file not found at {test_case_file}"
                )
                method_failed = True
            else:
                # Check if test cases file has content
                try:
                    with open(test_case_file, "r") as f:
                        import json

                        test_cases = json.load(f)
                        total_cases = sum(len(cases) for cases in test_cases.values())
                        if total_cases == 0:
                            self.logger.warning(
                                f"{method}: Test cases file is empty (0 test cases)"
                            )
                            method_failed = True
                except Exception as e:
                    self.logger.warning(f"{method}: Error reading test cases: {e}")
                    method_failed = True

            # # Check for completions file
            # completions_file = (
            #     f"{base_save_dir}/{method}/{subdir}/completions/{model_name}.json"
            # )
            # if not os.path.exists(completions_file):
            #     self.logger.warning(
            #         f"{method}: Completions file not found at {completions_file}"
            #     )
            #     method_failed = True

            # # Check for evaluation results file
            # results_file = f"{base_save_dir}/{method}/{subdir}/results/{model_name}.json"
            # if not os.path.exists(results_file):
            #     self.logger.warning(
            #         f"{method}: Evaluation results file not found at {results_file}"
            #     )
            #     method_failed = True

            if method_failed:
                failed_methods.append(method)
            else:
                self.logger.info(
                    f"Validated HarmBench pipeline results for {method}/{model_name}"
                )

        if failed_methods:
            raise RuntimeError(
                f"HarmBench pipeline failed for methods: {', '.join(failed_methods)}. "
                "Check logs above for details on missing files."
            )
