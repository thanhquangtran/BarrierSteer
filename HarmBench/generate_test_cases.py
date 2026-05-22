import argparse
import csv
import os
from os.path import join
import yaml
import sys
import json

# Define fast path helper BEFORE heavy imports
def get_output_file_path_fast(save_dir, behavior_id, file_type, run_id=None, method_name=None):
    """
    Fast version of get_output_file_path that doesn't require importing the method class.
    Must align with the logic in the respective method classes.
    """
    if method_name in ["DirectRequest", "HumanJailbreaks"]:
        return os.path.join(save_dir, f"{file_type}.json")
    
    # Default logic (used by PEZ, GCG, AutoDAN, etc.)
    run_id_str = f"_{run_id}" if run_id else ""
    return os.path.join(save_dir, 'test_cases_individual_behaviors', behavior_id, f"{file_type}{run_id_str}.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Running red teaming with baseline methods."
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default="GCG",
        help="The name of the red teaming method (options are in baselines/__init__.py)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="The name of the experiment (options are in configs/methods_config/{{method_name}}_config.yaml)",
    )
    parser.add_argument(
        "--method_config_file",
        type=str,
        default=None,
        help="The path to the config file with method hyperparameters. This will default to configs/method_configs/{{method_name}}_config.yaml",
    )
    parser.add_argument(
        "--models_config_file",
        type=str,
        default="./configs/model_configs/models.yaml",
        help="The path to the config file with model hyperparameters",
    )
    parser.add_argument(
        "--behaviors_path",
        type=str,
        default="./data/behavior_datasets/harmbench_behaviors_text_all.csv",
        help="The path to the behaviors file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./test_cases",
        help="The directory used for saving test cases",
    )
    parser.add_argument(
        "--behavior_start_idx",
        type=int,
        default=None,
        help="Start index for behaviors_path data (inclusive)",
    )
    parser.add_argument(
        "--behavior_end_idx",
        type=int,
        default=None,
        help="End index for behaviors_path data (exclusive)",
    )
    parser.add_argument(
        "--behavior_ids_subset",
        type=str,
        default="",
        help="An optional comma-separated list of behavior IDs, or a path to a newline-separated list of behavior IDs. If provided, this will override behavior_start_idx and behavior_end_idx for selecting a subset of behaviors.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="An optional run id for output files, will be appended to output filenames. Must be an integer.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing saved test cases",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print out intermediate results",
    )

    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)

    method_name = args.method_name
    experiment_name = args.experiment_name
    save_dir = os.path.abspath(args.save_dir)

    # Load model config file
    config_file = (
        f"configs/model_configs/models.yaml"
        if not args.models_config_file
        else args.models_config_file
    )
    with open(config_file) as file:
        model_configs = yaml.full_load(file)

    # Load method config file
    config_file = (
        f"configs/method_configs/{method_name}_config.yaml"
        if not args.method_config_file
        else args.method_config_file
    )
    with open(config_file) as file:
        method_configs = yaml.full_load(file)

    # Load default method parameters
    if "default_method_hyperparameters" not in method_configs:
        raise ValueError(
            f"Default method hyperparameters not found in {config_file}"
        )
    method_config = method_configs.get("default_method_hyperparameters") or {}

    # Update method parameters with experiment parameters
    # Note: We can't use get_experiment_config check yet because it's in eval_utils which imports heavily
    # But we can do basic filtering first. 
    # Actually, we need to load behaviors first to filter them.

    if args.run_id:
        assert (
            args.run_id.isdigit()
        ), f"epxected run_id to be an integer; got {args.run_id}"

    # ========== generate test cases ========== #
    # load behaviors csv
    with open(args.behaviors_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        behaviors = list(reader)

    if args.behavior_ids_subset:
        print(
            f"Selecting subset from provided behavior IDs: {args.behavior_ids_subset}"
        )
        if os.path.exists(args.behavior_ids_subset):
            with open(args.behavior_ids_subset, "r") as f:
                behavior_ids_subset = f.read().splitlines()
        else:
            behavior_ids_subset = args.behavior_ids_subset.split(",")
        behaviors = [
            b for b in behaviors if b["BehaviorID"] in behavior_ids_subset
        ]
    else:
        behavior_start_idx, behavior_end_idx = (
            args.behavior_start_idx,
            args.behavior_end_idx,
        )
        if (
            args.behavior_start_idx is not None
            and args.behavior_end_idx is not None
        ):
            print(
                f"Selecting subset from [{behavior_start_idx}, {behavior_end_idx}]"
            )
            behaviors = behaviors[
                args.behavior_start_idx : args.behavior_end_idx
            ]

    # ==== Filtering existed runs (FAST PATH) =====
    
    existing_merged_behaviors = set()
    # Check for merged file for methods that use individual files
    if method_name not in ["DirectRequest", "HumanJailbreaks"]:
        merged_file = os.path.join(save_dir, "test_cases.json")
        if os.path.exists(merged_file):
            try:
                with open(merged_file, 'r') as f:
                    data = json.load(f)
                    existing_merged_behaviors = set(data.keys())
            except Exception as e:
                print(f"Warning: Failed to load existing test_cases.json: {e}")

    if not args.overwrite:
        filtered_behaviors = []
        for behavior in behaviors:
            behavior_id = behavior["BehaviorID"]
            
            # Check merged file first
            if behavior_id in existing_merged_behaviors:
                continue

            test_case_file = get_output_file_path_fast(
                save_dir,
                behavior_id,
                file_type="test_cases",
                run_id=args.run_id,
                method_name=method_name
            )
            if not os.path.exists(test_case_file):
                filtered_behaviors.append(behavior)
            # No print here to keep fast path clean/quiet, but strictly speaking we could print skipping
            # if args.verbose: print(f"===>Skipping {behavior_id}") 
        
        skipped_count = len(behaviors) - len(filtered_behaviors)
        if skipped_count > 0:
            print(f"Skipping {skipped_count} behaviors that already have test cases.")
        
        behaviors = filtered_behaviors

        if len(behaviors) == 0:
            print(
                "Found existing test cases for all current behaviors. Exiting."
            )
            return

    # ========== HEAVY IMPORTS START HERE ========== #
    print("Loading heavy dependencies...")
    import pandas as pd
    import torch
    import transformers
    from baselines import get_method_class, init_method
    from eval_utils import get_experiment_config
    
    # Set this to disable warning messages in the generation mode.
    transformers.utils.logging.set_verbosity_error()
    os.system("module load eth_proxy")
    # ========== HEAVY IMPORTS END HERE ========== #

    # Update method parameters with experiment parameters (Now safe to call)
    experiment_config = get_experiment_config(
        experiment_name, model_configs, method_configs
    )
    method_config.update(experiment_config)

    world_size = torch.cuda.device_count()

    print(
        f"============= {method_name}, {experiment_name} Config ============="
    )
    print("\n".join([f"{k}: {v}" for k, v in method_config.items()]))
    print("============= GPU Allocation Configs =============")
    print(f"Total GPUs: {world_size}")
    print("=============")

    # ==== Init Method ====
    method_class = get_method_class(method_name)
    method = init_method(method_class, method_config)

    # ====== Run attack ======
    print("#Behaviors to run:", len(behaviors))
    test_cases, logs = method.generate_test_cases(
        behaviors=behaviors, verbose=args.verbose
    )

    # ====== Save test cases and logs ======
    print("Saving outputs to: ", save_dir)
    method.save_test_cases(
        save_dir,
        test_cases,
        logs,
        method_config=method_config,
        run_id=args.run_id,
    )
    print("Finished")


if __name__ == "__main__":
    main()
