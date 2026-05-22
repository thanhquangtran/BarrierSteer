"""
Merged Stage 2&3: Hidden State Extraction and Processing
Directly calls HarmBench's save_reps.sh and process_hb_states.py for exact reproducibility.
"""

import glob
import logging
import os
import subprocess
import time
from typing import Any, Dict, List


class HiddenStateExtractionAndProcessingStage:
    """Merged stage that calls save_reps.sh and process_hb_states.py directly"""

    def __init__(
        self,
        config: Dict[str, Any],
        harmbench_path: str,
        barriersteer_path: str,
    ):
        self.config = config
        self.harmbench_path = harmbench_path
        self.barriersteer_path = barriersteer_path
        self.logger = logging.getLogger(__name__)

    def run(
        self,
        model_name: str,
        attack_methods: List[str],
        layer_num: int = 20,
        model_config: Dict[str, Any] = None,
    ) -> str:
        """
        Run hidden state extraction and processing using exact previous commands

        Args:
            model_name: Name of the model (e.g., 'qwen_1.5b')
            attack_methods: List of attack methods from config
            layer_num: Layer number to extract hidden states from
            model_config: Model-specific configuration

        Returns:
            Path to processed data file
        """
        self.logger.info(
            f"Starting hidden state extraction and processing for {model_name}"
        )
        self.logger.info(f"Attack methods: {attack_methods}")
        self.logger.info(f"Layer number: {layer_num}")

        # Step 1: Run save_reps.sh
        self._run_save_reps(model_name, attack_methods, layer_num)

        # Step 2: Run process_hb_states.py
        output_path = self._run_process_hb_states(
            model_name, attack_methods, model_config
        )

        return output_path

    def _run_save_reps(
        self, model_name: str, attack_methods: List[str], layer_num: int
    ) -> None:
        """
        Run save_reps.sh script from HarmBench

        Command: scripts/save_reps.sh qwen_1.5b sbatch AutoPrompt,DirectRequest,GBDA,GCG 20
        """
        self.logger.info("Running save_reps.sh...")

        # Change to HarmBench directory
        original_dir = os.getcwd()
        os.chdir(self.harmbench_path)

        try:
            # Prepare command
            methods_to_run = []
            hidden_state_config = self.config.get("hidden_state_extraction", {})
            overwrite = hidden_state_config.get("overwrite", False)

            for method in attack_methods:
                # Use model-specific directory for all method except DirectRequest
                if method == "DirectRequest":
                    subdir = "default"
                else:
                    subdir = model_name

                # Check if hidden states already exist
                pipeline_config = self.config.get("pipeline", {})
                base_save_dir = pipeline_config.get("base_save_dir", "results")
                if not os.path.isabs(base_save_dir):
                    base_save_dir = os.path.join(
                        self.harmbench_path, base_save_dir.lstrip("./")
                    )

                hidden_states_dir = os.path.join(
                    base_save_dir, method, subdir, "hidden_states"
                )

                # We look for any file matching the pattern because timestamps vary
                pattern = os.path.join(
                    hidden_states_dir,
                    f"hidden_states_{model_name}_layer{layer_num}_*.pth",
                )
                existing_files = glob.glob(pattern)

                # Filter out partial files
                existing_files = [f for f in existing_files if "_part" not in f]

                # Also check if results (labels) exist and are valid (non-empty)
                results_dir = os.path.join(base_save_dir, method, subdir, "results")
                results_pattern = os.path.join(results_dir, f"{model_name}*.json")
                existing_results = glob.glob(results_pattern)

                results_valid = False
                if existing_results:
                    for res_file in existing_results:
                        # Check size > 10 bytes (an empty JSON "{}" is ~2 bytes)
                        if os.path.getsize(res_file) > 10:
                            results_valid = True
                            break

                if existing_files and results_valid and not overwrite:
                    self.logger.info(
                        f"Skipping hidden state generation for {method} (found {len(existing_files)} existing files)"
                    )
                else:
                    methods_to_run.append(method)

            if not methods_to_run:
                self.logger.info(
                    "All methods have existing hidden states. Skipping generation."
                )
                return

            methods_str = ",".join(methods_to_run)
            execution_mode = hidden_state_config.get("execution_mode", "sbatch")

            # Map 'slurm' to 'sbatch' for compatibility with save_reps.sh
            if execution_mode == "slurm":
                execution_mode = "sbatch"

            cmd = [
                "scripts/save_reps.sh",
                model_name,
                execution_mode,
                methods_str,
                str(layer_num),
                str(
                    self.config.get("hidden_state_extraction", {}).get(
                        "part_size", None
                    )
                ),
                str(self.config.get("pipeline", {}).get("starting_batch_size", 8)),
            ]

            self.logger.info(f"Running command: {' '.join(cmd)}")

            # Prepare environment with WORKSPACE_ROOT
            env = os.environ.copy()

            # Infer WORKSPACE_ROOT from harmbench_path (parent of HarmBench directory)
            # harmbench_path is e.g. /path/to/workspace/HarmBench
            workspace_root = os.path.dirname(self.harmbench_path.rstrip("/"))
            env["WORKSPACE_ROOT"] = workspace_root

            # Pass custom results directory to save_reps.sh
            pipeline_config = self.config.get("pipeline", {})
            base_save_dir = pipeline_config.get("base_save_dir", "results")
            if not os.path.isabs(base_save_dir):
                base_save_dir = os.path.join(
                    self.harmbench_path, base_save_dir.lstrip("./")
                )

            env["HARM_BENCH_RESULTS_DIR"] = base_save_dir

            # Pass custom behaviors/test-case paths to save_reps.sh. WildGuard
            # uses separate train/val behavior files, so Stage 2 can point at a
            # train-only DirectRequest JSON while Stage 4 keeps the default val
            # test_cases.json for evaluation.
            hidden_state_config = self.config.get("hidden_state_extraction", {})
            behaviors_path = hidden_state_config.get("training_behavior_dataset")
            if behaviors_path:
                if not os.path.isabs(behaviors_path):
                    behaviors_path = os.path.join(
                        self.harmbench_path, behaviors_path.lstrip("./")
                    )
                env["HARM_BENCH_BEHAVIORS_PATH"] = behaviors_path
                self.logger.info(
                    f"Setting HARM_BENCH_BEHAVIORS_PATH to: {behaviors_path}"
                )

            test_cases_path = hidden_state_config.get("test_cases_path")
            if test_cases_path:
                if not os.path.isabs(test_cases_path):
                    test_cases_path = os.path.join(
                        self.harmbench_path, test_cases_path.lstrip("./")
                    )
                env["HARM_BENCH_TEST_CASES_PATH"] = test_cases_path
                self.logger.info(
                    f"Setting HARM_BENCH_TEST_CASES_PATH to: {test_cases_path}"
                )

            self.logger.info(f"Setting WORKSPACE_ROOT to: {workspace_root}")
            self.logger.info(f"Setting HARM_BENCH_RESULTS_DIR to: {base_save_dir}")

            # Run the script
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)

            if result.returncode != 0:
                error_msg = f"save_reps.sh failed with return code {result.returncode}"
                if result.stdout:
                    error_msg += f"\nSTDOUT:\n{result.stdout}"
                if result.stderr:
                    error_msg += f"\nSTDERR:\n{result.stderr}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            self.logger.info("save_reps.sh completed successfully")

            # If using sbatch mode, wait for jobs to complete
            if execution_mode == "sbatch":
                self._wait_for_all_jobs_completion()

        finally:
            os.chdir(original_dir)

    def _wait_for_all_jobs_completion(self) -> None:
        """
        Wait for all SLURM jobs with name starting with "generate_save_" or "evaluate_" to complete
        """
        job_patterns = ["generate_save_*", "evaluate_*"]
        self.logger.info(
            f"Waiting for SLURM jobs with names matching {job_patterns} to complete..."
        )

        # First check if squeue is available
        squeue_check = subprocess.run(
            ["which", "squeue"], capture_output=True, text=True
        )

        if squeue_check.returncode != 0:
            self.logger.warning("squeue command not found. This might be because:")
            self.logger.warning("1. You're not on a SLURM cluster node")
            self.logger.warning("2. SLURM is not installed or not in PATH")
            self.logger.warning(
                "Skipping job monitoring. Please check job status manually."
            )
            # Give some time for jobs to be submitted
            time.sleep(30)
            return

        timeout_minutes = self.config.get("slurm", {}).get("job_timeout_minutes", 240)
        check_interval = self.config.get("slurm", {}).get("check_interval_seconds", 60)

        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        user = os.getenv("USER")
        if not user:
            raise RuntimeError("USER environment variable not set")

        while time.time() - start_time < timeout_seconds:
            any_jobs_running = False
            total_job_count = 0

            for pattern in job_patterns:
                # Check SLURM queue for jobs with matching pattern
                result = subprocess.run(
                    [
                        "squeue",
                        "-u",
                        user,
                        "-h",
                        "--name",
                        pattern,
                    ],  # -h for no header, filter by job name
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    self.logger.warning(
                        f"Failed to check SLURM queue for {pattern}: {result.stderr}"
                    )
                    continue

                output = result.stdout.strip()
                if output:
                    any_jobs_running = True
                    job_lines = output.split("\n")
                    count = len([line for line in job_lines if line.strip()])
                    total_job_count += count

            if not any_jobs_running:
                self.logger.info(f"All monitored SLURM jobs ({job_patterns}) completed")
                # Give a bit more time for file system sync
                time.sleep(10)
                return

            self.logger.info(
                f"Still waiting for {total_job_count} jobs matching {job_patterns} to complete..."
            )
            time.sleep(check_interval)

        # Timeout reached
        raise TimeoutError(
            f"SLURM jobs did not complete within {timeout_minutes} minutes"
        )

    def _run_process_hb_states(
        self,
        model_name: str,
        attack_methods: List[str],
        model_config: Dict[str, Any] = None,
    ) -> str:
        """
        Run process_hb_states.py script

        Command: python src/barriersteer_pipeline/data/process_hb_states.py
                 --methods AutoPrompt,DirectRequest,GBDA,GCG
                 --model qwen_1.5b
                 --output ./hs_data/qwen_1.5b/
        """
        self.logger.info("Running process_hb_states.py...")

        # Change to BarrierSteer directory
        original_dir = os.getcwd()
        os.chdir(self.barriersteer_path)

        try:
            # Prepare output path
            # Allow overriding the base hidden state directory via config
            # Priority:
            # 1. model_config.hidden_state_extraction.hs_save_dir
            # 2. global_config.hidden_state_extraction.hs_save_dir
            # 3. default (./hs_data/{model_name})

            output_path = None
            hidden_state_config = self.config.get("hidden_state_extraction", {})

            # Check model config first
            if model_config:
                model_hidden_config = model_config.get("hidden_state_extraction", {})
                if model_hidden_config and "hs_save_dir" in model_hidden_config:
                    output_path = model_hidden_config["hs_save_dir"]

            # Check global config if not found in model config
            if not output_path:
                output_path = hidden_state_config.get("hs_save_dir", None)

            if not output_path:
                output_path = f"./hs_data/{model_name}"

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Prepare command
            methods_str = ",".join(attack_methods)

            # Get training behavior dataset from config to filter behaviors for training
            hidden_state_config = self.config.get("hidden_state_extraction", {})
            training_behavior_dataset = hidden_state_config.get(
                "training_behavior_dataset",
                "data/behavior_datasets/harmbench_behaviors_text_test.csv",
            )
            training_behavior_dataset = os.path.join(
                self.harmbench_path, training_behavior_dataset.lstrip("./")
            )
            self.logger.info(
                f"Filtering to training behaviors from: {training_behavior_dataset}"
            )

            # Determine root directory for results (matches Stage 1 output)
            pipeline_config = self.config.get("pipeline", {})
            base_save_dir = pipeline_config.get("base_save_dir", "results")
            if not os.path.isabs(base_save_dir):
                base_save_dir = os.path.join(
                    self.harmbench_path, base_save_dir.lstrip("./")
                )

            cmd = [
                "python",
                "src/barriersteer_pipeline/data/process_hb_states.py",
                "--root",
                base_save_dir,
                "--methods",
                methods_str,
                "--model",
                model_name,
                "--output",
                output_path,
                "--training_behavior_dataset",
                training_behavior_dataset,
            ]

            if hidden_state_config.get("split_by_category", False):
                cmd.append("--split_by_category")
                behaviors_path = hidden_state_config.get("behaviors_path")
                if behaviors_path:
                    # Resolve path relative to harmbench or workspace
                    if not os.path.isabs(behaviors_path):
                        behaviors_path = os.path.join(
                            self.harmbench_path, behaviors_path.lstrip("./")
                        )
                    cmd.extend(["--behaviors_path", behaviors_path])

                category_column = hidden_state_config.get("category_column")
                if category_column:
                    cmd.extend(["--category_column", category_column])
                else:
                    self.logger.warning(
                        "split_by_category is True but behaviors_path is missing"
                    )

            self.logger.info(f"Running command: {' '.join(cmd)}")

            # Run the script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode != 0:
                error_msg = (
                    f"process_hb_states.py failed with return code {result.returncode}"
                )
                if result.stdout:
                    error_msg += f"\nSTDOUT:\n{result.stdout}"
                if result.stderr:
                    error_msg += f"\nSTDERR:\n{result.stderr}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            self.logger.info("process_hb_states.py completed successfully")
            self.logger.info(f"Output saved to: {output_path}")

            # Validate output file
            full_output_path = os.path.join(self.barriersteer_path, output_path)
            if not os.path.isdir(full_output_path):
                raise RuntimeError(
                    f"Expected output directory not found: {full_output_path}"
                )

            # Check if directory is not empty and contains .pt files
            pt_files = glob.glob(os.path.join(full_output_path, "*.pt"))
            if not pt_files:
                raise RuntimeError(
                    f"Output directory is empty or contains no .pt files: {full_output_path}"
                )

            self.logger.info(
                f"Validated output directory: {full_output_path} (contains {len(pt_files)} .pt files)"
            )

            return output_path

        finally:
            os.chdir(original_dir)
