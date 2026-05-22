import argparse
import csv
import json
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
# Force offline for library calls as well
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# NumPy 2.0 compatibility fix
# Handle deprecated numpy types that cupy might try to access
if not hasattr(np, 'float_'):
    np.float_ = float
if not hasattr(np, 'int_'):
    np.int_ = int
if not hasattr(np, 'complex_'):
    np.complex_ = complex
if not hasattr(np, 'bool_'):
    np.bool_ = bool

# Additional NumPy 2.0 compatibility for cupy and other packages
if not hasattr(np, 'AxisError'):
    # AxisError was introduced in NumPy 1.13, but may be missing in some configurations
    try:
        from numpy import AxisError
        np.AxisError = AxisError
    except ImportError:
        # Create a dummy AxisError class if it doesn't exist
        class AxisError(Exception):
            """Axis supplied was invalid."""
            pass
        np.AxisError = AxisError

from eval_utils import (
    compute_results_advbench,
    compute_results_classifier,
    compute_results_hashing,
)

os.system("module load eth_proxy")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Running red teaming with baseline methods."
    )
    parser.add_argument(
        "--cls_path",
        type=str,
        default="cais/HarmBench-Llama-2-13b-cls",
        help="The name or path of the classifier for computing ASR",
    )
    parser.add_argument(
        "--behaviors_path",
        type=str,
        default="./data/behavior_datasets/harmbench_behaviors_text_all.csv",
        help="The path to the behaviors file",
    )
    parser.add_argument(
        "--completions_path",
        type=str,
        help="The path to the completions to evaluate",
    )
    parser.add_argument(
        "--save_path", type=str, help="The path for saving results"
    )
    parser.add_argument(
        "--include_advbench_metric",
        action="store_true",
        help="Whether to include the AdvBench refusal metric",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=512,
        help="The number of tokens to evaluate",
    )
    # vLLM memory configuration arguments
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=None,
        help="GPU memory utilization ratio (0.0-1.0). If not provided, reads from VLLM_GPU_MEMORY_UTILIZATION env var or defaults to 0.9",
    )
    parser.add_argument(
        "--enforce_eager",
        type=str,
        default=None,
        help="Whether to enforce eager mode (disable CUDA graphs). Use 'true' or '1' for True. If not provided, reads from VLLM_ENFORCE_EAGER env var or defaults to False",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=None,
        help="Maximum number of sequences to process in parallel. If not provided, reads from VLLM_MAX_NUM_SEQS env var or defaults to 256",
    )
    parser.add_argument(
        "--batch_config",
        type=str,
        default=None,
        help="Path to a JSON config file containing a list of evaluation tasks. If provided, ignores other arguments.",
    )
    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)

    # If batch_config is provided, load tasks from it
    if args.batch_config:
        with open(args.batch_config, "r") as f:
            tasks = json.load(f)
        print(f"Loaded {len(tasks)} tasks from {args.batch_config}")
    else:
        # Create a single task from command line arguments
        tasks = [{
            "behaviors_path": args.behaviors_path,
            "completions_path": args.completions_path,
            "save_path": args.save_path,
            "include_advbench_metric": args.include_advbench_metric,
            "num_tokens": args.num_tokens,
        }]

    # Load tokenizer and classifier once
    tokenizer = AutoTokenizer.from_pretrained(args.cls_path)
    tokenizer.truncation_side = "right"
    
    # Initialize vLLM model once
    num_gpus = torch.cuda.device_count()
    gpu_memory_utilization = args.gpu_memory_utilization
    if gpu_memory_utilization is None:
        gpu_memory_utilization = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
    
    enforce_eager = args.enforce_eager
    if enforce_eager is None:
        enforce_eager_str = os.environ.get("VLLM_ENFORCE_EAGER", "0")
        enforce_eager = str(enforce_eager_str).lower() in ("1", "true", "yes")
    else:
        enforce_eager = str(enforce_eager).lower() in ("1", "true", "yes")
    
    max_num_seqs = args.max_num_seqs
    if max_num_seqs is None:
        max_num_seqs = int(os.environ.get("VLLM_MAX_NUM_SEQS", "256"))
    
    print(f"vLLM memory settings: gpu_memory_utilization={gpu_memory_utilization}, "
          f"enforce_eager={enforce_eager}, max_num_seqs={max_num_seqs}")
    
    # Load model config to get number of attention heads
    config = transformers.AutoConfig.from_pretrained(args.cls_path)
    num_heads = config.num_attention_heads
    
    # Find the largest divisor of num_heads that is <= num_gpus
    tp_size = num_gpus
    while tp_size > 0:
        if num_heads % tp_size == 0:
            break
        tp_size -= 1
        
    if tp_size != num_gpus:
        print(f"Adjusting tensor_parallel_size from {num_gpus} to {tp_size} to divide num_heads ({num_heads})")

    cls = LLM(
        model=args.cls_path,
        tensor_parallel_size=tp_size,
        dtype=torch.float16,
        distributed_executor_backend="mp",
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        max_num_seqs=max_num_seqs,
    )
    cls.llm_engine.tokenizer.tokenizer.truncation_side = "left"
    cls_params = SamplingParams(temperature=0.0, max_tokens=1)

    # Process each task
    for i, task in enumerate(tasks):
        print(f"Processing evaluation task {i+1}/{len(tasks)}")
        behaviors_path = task.get("behaviors_path", args.behaviors_path)
        completions_path = task.get("completions_path", args.completions_path)
        save_path = task.get("save_path", args.save_path)
        include_advbench_metric = task.get("include_advbench_metric", args.include_advbench_metric)
        num_tokens = task.get("num_tokens", args.num_tokens)

        # ========== load behaviors (for tags and context strings) ========== #
        with open(behaviors_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            behaviors = list(reader)

        # convert to dictionary mapping behavior ID field to behavior dict
        behaviors = {b["BehaviorID"]: b for b in behaviors}

        # ========== initialize results ========== #
        with open(completions_path, "r") as f:
            completions = json.load(f)

        # Clip the 'generation' field of the completions to have a maximum of num_tokens tokens
        for behavior_id, completion_list in completions.items():
            for completion in completion_list:
                generation = completion["generation"]
                if generation is None or len(generation) == 0:
                    generation = "None"
                tokenized_text = tokenizer.encode(
                    generation, max_length=num_tokens, truncation=True
                )
                clipped_generation = tokenizer.decode(
                    tokenized_text, skip_special_tokens=True
                )
                completion["generation"] = clipped_generation

        # ========== evaluate completions ========== #
        results = {}
        skipped_behaviors = 0
        for behavior_id, data in tqdm(completions.items()):
            if behavior_id not in behaviors:
                skipped_behaviors += 1
                continue
            behavior_dict = behaviors[behavior_id]
            tags = behavior_dict["Tags"].split(", ")

            if "hash_check" in tags:
                current_results = compute_results_hashing(behavior_dict, data)
            else:
                current_results = compute_results_classifier(
                    behavior_dict, data, cls, cls_params
                )

            if include_advbench_metric:
                current_results = compute_results_advbench(current_results, data)

            results[behavior_id] = current_results

        if skipped_behaviors > 0:
            print(f"Skipped {skipped_behaviors} behaviors not found in behaviors file")

        success_rates = []

        for j, (behavior_id, data) in enumerate(results.items()):
            success_mean = np.mean([d["label"] == 1 for d in data])
            # print(f"{j}. {behavior_id} === average asr: {success_mean}")
            success_rates.append(success_mean)

        print("Average ASR: ", np.mean(success_rates))

        # Make dirs to output_file if not yet exist
        (
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if os.path.dirname(save_path)
            else None
        )
        with open(save_path, "w") as file:
            json.dump(results, file, indent=4)



if __name__ == "__main__":
    main()
