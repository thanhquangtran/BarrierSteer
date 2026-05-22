import argparse
import os
import subprocess
import sys

# Configuration
# Assuming we are running from workspace root
WORKSPACE_ROOT = os.environ.get("WORKSPACE_ROOT", os.getcwd())
HARMBENCH_PIPELINE_SCRIPT = os.path.join(
    WORKSPACE_ROOT, "src/barriersteer_pipeline/harmbench/run_harmbench_pipeline.py"
)
CONFIG_DIR = os.path.join(WORKSPACE_ROOT, "src/barriersteer_pipeline/harmbench/config")

MODEL_NAME = "llama2_7b"
DATASETS = ["beavertails", "wildguard"]


def run_command(cmd, env=None):
    print(f"[RUNNING] {' '.join(cmd)}")
    if env is None:
        env = os.environ.copy()

    # Ensure WORKSPACE_ROOT is set in environment for config expansion
    env["WORKSPACE_ROOT"] = WORKSPACE_ROOT

    # Ensure PYTHONPATH includes necessary directories
    env["PYTHONPATH"] = (
        f"{os.path.join(WORKSPACE_ROOT, 'src')}:{WORKSPACE_ROOT}:{os.path.join(WORKSPACE_ROOT, 'HarmBench')}:{env.get('PYTHONPATH', '')}"
    )

    result = subprocess.run(cmd, env=env, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with code {result.returncode}")
        raise RuntimeError("Command failed")


def main():
    parser = argparse.ArgumentParser(
        description="Run adaptation pipeline for specific datasets"
    )
    parser.add_argument(
        "--dataset", type=str, default="all", help="beavertails, wildguard, or all"
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="1,2,3",
        help="Comma-separated stages (1=Gen, 2=Extract, 3=Train). Default: 1,2,3",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama2_7b",
        help="Model name to run (e.g., llama2_7b, mistral_8b, qwen_1.5b, gemma_2_9b)",
    )
    args = parser.parse_args()

    target_datasets = DATASETS if args.dataset == "all" else [args.dataset]

    for dataset in target_datasets:
        print(f"\n{'='*50}\nPipeline Run for Dataset: {dataset}\n{'='*50}")

        config_path = os.path.join(CONFIG_DIR, f"{dataset}.yaml")
        if not os.path.exists(config_path):
            print(f"[ERROR] Config not found: {config_path}")
            continue

        cmd = [
            "python",
            HARMBENCH_PIPELINE_SCRIPT,
            "--config",
            config_path,
            "--model",
            args.model,
            "--stages",
            args.stages,
        ]

        try:
            run_command(cmd)
            print(f"[{dataset}] Pipeline completed successfully.")
        except RuntimeError:
            print(f"[{dataset}] Pipeline failed.")


if __name__ == "__main__":
    main()
