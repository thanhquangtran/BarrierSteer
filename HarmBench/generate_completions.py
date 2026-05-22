import argparse
import csv
import gc
import json
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import random
import sys
import time
from functools import partial
from typing import Optional

import numpy as np
import torch
import transformers
import yaml
from accelerate.utils import find_executable_batch_size

# Add src to path for barriersteer_pipeline module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from steer.common.defense_utils import setup_defense, setup_defense_object, detect_model_name_from_path
from llm_jailbreaking_defense import TargetLM
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from vllm import SamplingParams

import multimodalmodels
from api_models import api_models_map
from baselines import get_template, load_model_and_tokenizer, load_vllm_model

# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()

class _TimingRecorder:
    """Accumulates wall-clock generation timing and token counts."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_generate_s = 0.0
        self.total_new_tokens = 0
        self.total_prompt_tokens = 0
        self.num_batches = 0
        
        # New split timing: Monitor (Check) vs Steer (Solve)
        self.total_check_s = 0.0
        self.total_solve_s = 0.0
        
        # Legacy/Total overhead tracking
        self.total_forward_pass_s = 0.0
        self.total_steering_overhead_s = 0.0
        
        # Separate timing for violated vs not violated
        self.total_violated_tokens = 0
        self.total_violated_time_s = 0.0
        self.total_not_violated_tokens = 0
        self.total_not_violated_time_s = 0.0
        self.max_steering_norm = 0.0

    def add_batch(
        self,
        generate_s: float,
        new_tokens: int,
        prompt_tokens: int,
        forward_pass_s: Optional[float] = None,
        steering_overhead_s: Optional[float] = None,
        check_overhead_s: Optional[float] = None,
        solve_overhead_s: Optional[float] = None,
        violated_tokens: Optional[int] = None,
        violated_time_s: Optional[float] = None,
        not_violated_tokens: Optional[int] = None,
        not_violated_time_s: Optional[float] = None,
        max_steering_norm: Optional[float] = None,
    ):
        self.total_generate_s += float(generate_s)
        self.total_new_tokens += int(new_tokens)
        self.total_prompt_tokens += int(prompt_tokens)
        self.num_batches += 1
        
        if forward_pass_s is not None:
            self.total_forward_pass_s += float(forward_pass_s)
        if steering_overhead_s is not None:
            self.total_steering_overhead_s += float(steering_overhead_s)
            
        if check_overhead_s is not None:
            self.total_check_s += float(check_overhead_s)
        if solve_overhead_s is not None:
            self.total_solve_s += float(solve_overhead_s)
            
        if violated_tokens is not None:
            self.total_violated_tokens += int(violated_tokens)
        if violated_time_s is not None:
            self.total_violated_time_s += float(violated_time_s)
        if not_violated_tokens is not None:
            self.total_not_violated_tokens += int(not_violated_tokens)
        if not_violated_time_s is not None:
            self.total_not_violated_time_s += float(not_violated_time_s)
            
        if max_steering_norm is not None:
            if max_steering_norm > self.max_steering_norm:
                self.max_steering_norm = float(max_steering_norm)

    def summary(self) -> dict:
        t = self.total_generate_s
        nt = self.total_new_tokens
        pt = self.total_prompt_tokens
        result = {
            "total_generate_s": t,
            "total_new_tokens": nt,
            "total_prompt_tokens": pt,
            "num_batches": self.num_batches,
            "sec_per_new_token": (t / nt) if nt > 0 else None,
            "tokens_per_sec_new": (nt / t) if t > 0 else None,
        }
        
        # Add forward pass vs steering overhead timing
        if self.total_forward_pass_s > 0 or self.total_steering_overhead_s > 0:
            result["total_forward_pass_s"] = self.total_forward_pass_s
            result["total_steering_overhead_s"] = self.total_steering_overhead_s
            result["forward_pass_ratio"] = (
                self.total_forward_pass_s / t if t > 0 else None
            )
            result["steering_overhead_ratio"] = (
                self.total_steering_overhead_s / t if t > 0 else None
            )
        
        # New Detailed Metrics: Monitor vs Steer
        if self.total_check_s > 0 or self.total_solve_s > 0:
            result["total_check_s"] = self.total_check_s
            result["total_solve_s"] = self.total_solve_s
            
            # avg_monitor_ms: Average cost of checking per generated token (runs always)
            if nt > 0:
                result["avg_monitor_ms"] = (self.total_check_s / nt) * 1000.0
            
            # avg_steer_ms: Average cost of solving per UNSAFE token (runs only on steering)
            # This is "just the cost of steering"
            if self.total_violated_tokens > 0:
                result["avg_steer_ms"] = (self.total_solve_s / self.total_violated_tokens) * 1000.0
            else:
                result["avg_steer_ms"] = 0.0

        if self.max_steering_norm > 0:
            result["max_steering_norm"] = self.max_steering_norm
        
        # Add violated vs not violated timing (Legacy / Consolidated)
        if self.total_violated_tokens > 0 or self.total_not_violated_tokens > 0:
            result["total_violated_tokens"] = self.total_violated_tokens
            result["total_violated_time_s"] = self.total_violated_time_s
            result["total_not_violated_tokens"] = self.total_not_violated_tokens
            result["total_not_violated_time_s"] = self.total_not_violated_time_s
            
            # ms_per_violated_token effectively usually includes monitor + steer latency
            result["ms_per_violated_token"] = (
                (self.total_violated_time_s / self.total_violated_tokens) * 1000.0
                if self.total_violated_tokens > 0 else None
            )
            result["ms_per_not_violated_token"] = (
                (self.total_not_violated_time_s / self.total_not_violated_tokens) * 1000.0
                if self.total_not_violated_tokens > 0 else None
            )
            result["violation_rate"] = (
                self.total_violated_tokens / nt if nt > 0 else None
            )
        
        return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Running red teaming with baseline methods."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model in the models config file",
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
        "--test_cases_path",
        type=str,
        help="The path to the test cases file to generate completions for",
    )
    parser.add_argument(
        "--save_path", type=str, help="The path for saving completions"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens for completions",
    )
    parser.add_argument(
        "--generate_with_vllm",
        action="store_true",
        default=False,
        help="Whether to generate completions with vLLM (if applicable)",
    )
    parser.add_argument(
        "--incremental_update",
        action="store_true",
        help="Whether to incrementally update completions or generate a new completions file from scratch",
    )
    parser.add_argument(
        "--part_size",
        type=int,
        default=None,
        help="Number of test cases to process in this part. If None, process all.",
    )
    parser.add_argument(
        "--part_number",
        type=int,
        default=1,
        help="Which part to process (starts from 1)",
    )
    parser.add_argument(
        "--use_defense",
        action="store_true",
        help="Whether to use defense method",
    )
    parser.add_argument(
        "--defense_method",
        type=str,
        default="backtranslation",
        choices=[
            "backtranslation",
            "ICL",
            "semantic_smoothing",
            "response_check",
            "smoothllm",
            "self_reminder",
        ],
        help="Type of defense method to use",
    )
    parser.add_argument(
        "--starting_batch_size",
        type=int,
        default=16,
        help="Starting batch size for finding optimal batch size",
    )
    parser.add_argument(
        "--batch_config",
        type=str,
        default=None,
        help="Path to a JSON config file containing a list of generation tasks. If provided, ignores other arguments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility across multiple runs",
    )
    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)
    
    # Set random seeds if provided
    if args.seed is not None:
        print(f"Setting random seed: {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Load model config file
    config_file = (
        f"configs/model_configs/models.yaml"
        if not args.models_config_file
        else args.models_config_file
    )
    with open(config_file) as file:
        model_configs = yaml.full_load(file)

    num_gpus = model_configs[args.model_name].get("num_gpus", 0)
    # check num gpus available to see if greater than num_gpus in config
    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available != num_gpus:
        print(
            f"Warning: num_gpus in config ({num_gpus}) does not match num_gpus available ({num_gpus_available}). Using {num_gpus_available} GPUs."
        )
        num_gpus = num_gpus_available
    model_config = model_configs[args.model_name]["model"]
    model_config["num_gpus"] = num_gpus

    print("model_config", model_config)

    # If batch_config is provided, load tasks from it
    if args.batch_config:
        with open(args.batch_config, "r") as f:
            tasks = json.load(f)
        print(f"Loaded {len(tasks)} tasks from {args.batch_config}")
    else:
        # Create a single task from command line arguments
        tasks = [{
            "behaviors_path": args.behaviors_path,
            "test_cases_path": args.test_cases_path,
            "save_path": args.save_path,
            "max_new_tokens": args.max_new_tokens,
            "incremental_update": args.incremental_update,
            "use_defense": args.use_defense,
            "defense_method": args.defense_method,
            "save_hidden_states_path": None,
            "part_size": args.part_size,
            "part_number": args.part_number,
        }]

    # Load model once
    # Use global args for model loading logic, though explicit task args might differ slightly?
    # For now assuming model config is consistent.
    # We use the first task's test cases path for potential initialization needs (like images)
    # providing a default if tasks is empty (shouldn't happen)
    first_task_test_cases_path = tasks[0]["test_cases_path"] if tasks else args.test_cases_path
    
    # Determine defense settings from FIRST task (assuming consistent batch) or args as fallback
    use_defense = args.use_defense
    defense_method = args.defense_method
    if tasks:
       use_defense = tasks[0].get("use_defense", args.use_defense)
       defense_method = tasks[0].get("defense_method", args.defense_method)

    generation_function = load_generation_function(
        model_config,
        args.max_new_tokens,
        test_cases_path=first_task_test_cases_path, # used for finding methods/images
        generate_with_vllm=args.generate_with_vllm,
        use_defense=use_defense,
        defense_method=defense_method,
        starting_batch_size=args.starting_batch_size,
    )

    for i, task in enumerate(tasks):
        print(f"Processing task {i+1}/{len(tasks)}")
        behaviors_path = task.get("behaviors_path", args.behaviors_path)
        test_cases_path = task.get("test_cases_path", args.test_cases_path)
        save_path = task.get("save_path", args.save_path)
        # Note: max_new_tokens is effectively fixed by load_generation_function currently
        incremental_update = task.get("incremental_update", args.incremental_update)
        part_size = task.get("part_size", args.part_size)
        part_number = task.get("part_number", args.part_number)

        # ========== generate test cases ========== #
        # load behaviors csv
        with open(behaviors_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            behaviors_set = {row["BehaviorID"] for row in reader}

        with open(test_cases_path, "r") as f:
            test_cases_data = json.load(f)

        # Filter test cases to only include those in behaviors_set
        test_cases_data = {
            k: v for k, v in test_cases_data.items() if k in behaviors_set
        }



        returned_data = {}

        if incremental_update and os.path.exists(save_path):
            with open(save_path, "r") as file:
                returned_data = json.load(file)

            # Remove test cases that have already been generated
            new_test_cases_data = {}
            new_returned_data = {}
            for behavior_id, _test_cases in test_cases_data.items():
                # Check if behavior_id is in returned_data and completions have been generated for all test cases
                if behavior_id in returned_data and len(
                    returned_data[behavior_id]
                ) == len(_test_cases):
                    continue
                # print(behavior_id, len(_test_cases))
                new_test_cases_data[behavior_id] = _test_cases
                new_returned_data[behavior_id] = (
                    []
                )  # If not all completions are present, we will regenerate all completions
            test_cases_data = new_test_cases_data

        # ==== Flatten dict to array ======
        test_cases = []
        for behavior_id, _test_cases in test_cases_data.items():
            test_cases.extend(
                [dict(behavior_id=behavior_id, test_case=t) for t in _test_cases]
            )

        # Calculate part indices if part_size is specified
        total_cases = len(test_cases)
        current_save_path = save_path
        if part_size is not None:
            start_idx = (part_number - 1) * part_size
            end_idx = min(start_idx + part_size, total_cases)
            test_cases = test_cases[start_idx:end_idx]
            print(
                f"Processing part {part_number}: cases {start_idx} to {end_idx-1} (total cases: {total_cases})"
            )
            # Modify save path to include part number if part_size is specified
            base_save_path = save_path
            save_dir = os.path.dirname(base_save_path)
            filename = os.path.basename(base_save_path)
            name, ext = os.path.splitext(filename)
            current_save_path = os.path.join(
                save_dir, f"{name}_part{part_number}{ext}"
            )

        if len(test_cases) == 0:
            print("No test cases to generate completions for")
            continue
        else:
            print(f"Generating completions for {len(test_cases)} test cases")

        # ==== Generate ====
        print("Generating completions...")
        generation_result = generation_function(
            test_cases=[t["test_case"] for t in test_cases]
        )
        print("Done")

        # Handle return value: could be (generations, timing_recorder) or just generations
        if isinstance(generation_result, tuple) and len(generation_result) == 2:
            generations, timing = generation_result
        else:
            generations = generation_result
            timing = None

        # ---- Save timing stats (HF path only) ---------------------------------
        if timing is not None and hasattr(timing, "summary"):
            timing_path = None
            base, ext = os.path.splitext(current_save_path)
            timing_path = f"{base}_timing.json"
            payload = {
                "model_name": args.model_name,
                "save_path": current_save_path,
                "timing": timing.summary(),
                "model_config": model_config,
            }
            try:
                with open(timing_path, "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"[timing] Saved timing stats to {timing_path}")
                # print(f"[timing] {payload['timing']}")
            except Exception as e:
                print(f"[timing] Failed to save timing stats: {e}")

        # ==== Unflatten into dict of {behavior_id: [{test_case: t, generation: g}, ..], ...}
        for t, g in zip(test_cases, generations):
            returned_data.setdefault(t["behavior_id"], [])
            returned_data[t["behavior_id"]].append(
                dict(test_case=t["test_case"], generation=g)
            )

        # Create directories for save_path
        print(f"Saving completions to {current_save_path}...")
        (
            os.makedirs(os.path.dirname(current_save_path), exist_ok=True)
            if os.path.dirname(current_save_path)
            else None
        )
        with open(current_save_path, "w") as file:
            json.dump(returned_data, file, indent=4)
        print("Saved")


def _vllm_generate(model, test_cases, template, **generation_kwargs):
    inputs = [template["prompt"].format(instruction=s) for s in test_cases]
    outputs = model.generate(inputs, **generation_kwargs)
    generations = [o.outputs[0].text.strip() for o in outputs]
    return generations


def _hf_generate_with_batching(
    model,
    tokenizer,
    test_cases,
    template,
    timing_recorder: _TimingRecorder | None = None,
    starting_batch_size: int = 1,
    **generation_kwargs,
):
    # Remove batch_size from generation_kwargs if it exists
    generation_kwargs.pop(
        "batch_size", None
    )  # Safely remove batch_size if present

    # Force initial CUDA memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def inner_generation_loop(batch_size):
        nonlocal model, tokenizer, test_cases, template, generation_kwargs

        # Force CUDA memory cleanup before starting new batch size attempt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Starting generation with batch_size={batch_size}")
            print_memory_stats("Before generation")

        generations = []
        skipped_indices = []  # Track indices of skipped test cases

        for i in tqdm(range(0, len(test_cases), batch_size)):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            batched_test_cases = test_cases[i : i + batch_size]
            inputs = [
                template["prompt"].format(instruction=s)
                for s in batched_test_cases
            ]

            try:
                with torch.cuda.amp.autocast():
                    inputs = tokenizer(
                        inputs, return_tensors="pt", padding=True
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        # Initialize timing info storage in model if it's SafeRepModel
                        if hasattr(model, '_timing_info'):
                            model._timing_info = {}
                        
                        # Time only the model.generate call (this includes
                        # SafeRepModel's steering/optimization overhead).
                        t0 = time.perf_counter()
                        outputs = model.generate(
                            inputs=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            **generation_kwargs,
                        )
                        t1 = time.perf_counter()

                        outputs = outputs.cpu()
                        generated_tokens = outputs[
                            :, inputs["input_ids"].shape[1] :
                        ]

                        # Record timing + token counts
                        if timing_recorder is not None:
                            pad_id = tokenizer.pad_token_id
                            if pad_id is None:
                                # Fall back: treat all positions as tokens
                                new_tok = int(generated_tokens.numel())
                            else:
                                new_tok = int((generated_tokens != pad_id).sum().item())
                            prompt_tok = int(inputs["input_ids"].numel())
                            
                            # Extract timing info from model if available
                            timing_info = {}
                            if hasattr(model, '_timing_info'):
                                timing_info = model._timing_info.copy()
                            
                            # Calculate forward pass time (total - steering overhead)
                            total_time = t1 - t0
                            steering_overhead = timing_info.get('steering_overhead', 0.0) # Used for fallback if individual not available?
                            # Use new check+solve if available, else fallback to total steering
                            check_overhead = timing_info.get('check_overhead', 0.0)
                            solve_overhead = timing_info.get('solve_overhead', 0.0)
                            
                            # If new metrics available, use them. Else if only old steering_overhead matches, use it.
                            # Since we updated SafeRepModel, check/solve should be there.
                            # But if running old model wrapper, maybe not.
                            if check_overhead + solve_overhead > 0:
                                total_steering = check_overhead + solve_overhead
                            else:
                                total_steering = steering_overhead

                            forward_pass_time = total_time - total_steering
                            
                            # Get violated/not violated counts and max_steering_norm
                            violated_tokens = timing_info.get('violated_tokens', 0)
                            not_violated_tokens = timing_info.get('not_violated_tokens', 0)
                            max_steering_norm = timing_info.get('max_steering_norm', 0.0)
                            print(f"[DEBUG] Global Max Steering Norm: {max_steering_norm:.4f}")
                            
                            # Refined accounting:
                            # Monitor time = check_overhead (distributed across ALL tokens roughly, but mostly safe ones count big)
                            # Verify assumption: We want to attribute cost properly. 'violated_time' was used for overhead?
                            # Using new breakdown:
                            # violated_time ~= check_cost_per_token + solve_cost_per_token
                            # not_violated_time ~= check_cost_per_token
                            
                            # Backward compat calc for "violated_time_s"
                            # check_per_tok = check_overhead / new_tok
                            # solve_per_vio = solve_overhead / violated_tokens if violated_tokens > 0 else 0
                            
                            # violated_time = (check_per_tok * violated_tokens) + solve_overhead
                            # not_violated_time = check_per_tok * not_violated_tokens
                            
                            violated_time = 0.0
                            not_violated_time = 0.0
                            if new_tok > 0:
                                check_per_tok = check_overhead / new_tok
                                violated_time = (check_per_tok * violated_tokens) + solve_overhead
                                not_violated_time = check_per_tok * not_violated_tokens
                            
                            # If using old metric only:
                            if check_overhead == 0 and solve_overhead == 0 and total_steering > 0 and new_tok > 0:
                                violated_time = total_steering * (violated_tokens / new_tok)
                                not_violated_time = total_steering * (not_violated_tokens / new_tok)
                            
                            timing_recorder.add_batch(
                                generate_s=total_time,
                                new_tokens=new_tok,
                                prompt_tokens=prompt_tok,
                                forward_pass_s=forward_pass_time,
                                steering_overhead_s=total_steering,
                                check_overhead_s=check_overhead,
                                solve_overhead_s=solve_overhead,
                                violated_tokens=violated_tokens,
                                violated_time_s=violated_time,
                                not_violated_tokens=not_violated_tokens,
                                not_violated_time_s=not_violated_time,
                                max_steering_norm=max_steering_norm,
                            )

                        batch_generations = [
                            tokenizer.decode(
                                o, skip_special_tokens=True
                            ).strip()
                            for o in generated_tokens
                        ]
                        generations.extend(batch_generations)
                        print()
                        print(f"Generated batch {i}:{i + len(batched_test_cases)} ({len(batch_generations)} completions)")

                    del outputs, generated_tokens
                    for v in inputs.values():
                        del v
                    del inputs

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(
                        f"\nOOM error for batch at index {i}. Skipping this batch."
                    )
                    generations.extend(["None"] * len(batched_test_cases))
                    skipped_indices.extend(
                        range(i, min(i + batch_size, len(test_cases)))
                    )
                else:
                    raise e

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        if skipped_indices:
            print(
                f"\nSkipped {len(skipped_indices)} test cases due to OOM errors at indices: {skipped_indices}"
            )

        return generations, skipped_indices

    generations, skipped_indices = inner_generation_loop()
    # Return timing recorder along with generations if it was used
    if timing_recorder is not None:
        return generations, timing_recorder
    return generations


def print_memory_stats(prefix=""):
    """Print current GPU memory usage with an optional prefix."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(
            f"{prefix} GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )


def _api_model_generate(model, test_cases, **generation_kwargs):
    # MultiModal test cases are in saved in pair of [[img, text], ...]
    if isinstance(test_cases[0], (tuple, list)):
        images, prompts = zip(*test_cases)
        return model.generate(
            prompts=list(prompts), images=list(images), **generation_kwargs
        )
    return model.generate(prompts=test_cases, **generation_kwargs)


def _defended_generate_with_batching(
    model, test_cases, batch_size=1, **generation_kwargs
):
    """Batched generation for defended models with adaptive OOM backoff.

    The upstream defense wrapper is safe to call with lists, but some defenses
    expand prompts enough that a large batch can OOM. Do not skip data on OOM;
    retry the same slice with half the batch size until it fits, only failing if
    a single prompt cannot run.
    """
    generations = []
    batch_size = max(1, int(batch_size or 1))
    i = 0
    pbar = tqdm(total=len(test_cases))

    try:
        while i < len(test_cases):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            current_batch_size = min(batch_size, len(test_cases) - i)
            batch = test_cases[i : i + current_batch_size]
            try:
                batch_generations = model.get_response(batch, **generation_kwargs)
                generations.extend(batch_generations)

                print(f"\nBatch {i}:{i + current_batch_size} generations:")
                print(batch_generations)

                i += current_batch_size
                pbar.update(current_batch_size)

            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                if current_batch_size <= 1:
                    raise RuntimeError(
                        f"OOM while generating a single defended prompt at index {i}"
                    ) from e
                new_batch_size = max(1, current_batch_size // 2)
                print(
                    f"\nOOM for defended batch {i}:{i + current_batch_size}; "
                    f"retrying with batch_size={new_batch_size}"
                )
                batch_size = new_batch_size
    finally:
        pbar.close()

    return generations




def load_generation_function(
    model_config,
    max_new_tokens,
    test_cases_path,
    generate_with_vllm=False,
    use_defense=False,
    defense_method="backtranslation",
    starting_batch_size=16,
):

    model_name_or_path = model_config["model_name_or_path"]

    if model_name_or_path in dir(multimodalmodels):
        model = getattr(multimodalmodels, model_name_or_path)()
        generation_kwargs = dict(
            max_new_tokens=max_new_tokens, do_sample=False, num_beams=1
        )
        image_dir = os.path.join(os.path.dirname(test_cases_path), "images")
        return partial(
            model.generate, image_dir=image_dir, **generation_kwargs
        )
    if model := api_models_map(**model_config):
        generation_kwargs = dict(
            max_new_tokens=max_new_tokens, temperature=0.0, use_tqdm=True
        )
        return partial(_api_model_generate, model=model, **generation_kwargs)
    elif generate_with_vllm:
        print("Using vLLM generation")
        sp = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        model = load_vllm_model(**model_config)
        generation_kwargs = dict(sampling_params=sp, use_tqdm=True)
        TEMPLATE = get_template(
            model_name_or_path,
            chat_template=model_config.get("chat_template", None),
        )
        return partial(
            _vllm_generate, model=model, template=TEMPLATE, **generation_kwargs
        )
    else:
        print("Using HF generation")
        if use_defense and defense_method == "self_reminder":
            print("Setting up native HF self_reminder defense")
            model, tokenizer = load_model_and_tokenizer(**model_config)

            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(tokenizer))

            defense = setup_defense_object("self_reminder")
            generation_kwargs = dict(
                max_new_tokens=max_new_tokens, do_sample=False
            )
            TEMPLATE = get_template(
                model_name_or_path,
                chat_template=model_config.get("chat_template", None),
            )

            def _self_reminder_hf_generate(test_cases, **kwargs):
                reminded = [defense._remind_suffix(prompt) for prompt in test_cases]
                prefix_template = defense._remind_prefix()
                if prefix_template:
                    prefix = prefix_template.format(original_system_message="")
                    reminded = [f"{prefix}\n\n{prompt}" for prompt in reminded]
                return _hf_generate_with_batching(
                    model=model,
                    tokenizer=tokenizer,
                    test_cases=reminded,
                    template=TEMPLATE,
                    starting_batch_size=starting_batch_size,
                    **generation_kwargs,
                )

            return _self_reminder_hf_generate

        if use_defense:
            print(f"Setting up model with {defense_method} defense")
            detected_name = detect_model_name_from_path(model_name_or_path)
            print(f"Detected model name from '{model_name_or_path}': {detected_name}")
            
            # Initialize target model using llm_jailbreaking_defense
            target_model = TargetLM(
                model_name=detected_name,
                max_n_tokens=max_new_tokens,
                batch_size=starting_batch_size,
            )
            # Setup defense
            defended_model = setup_defense(target_model, defense_method)

            return partial(
                _defended_generate_with_batching,
                model=defended_model,
                batch_size=starting_batch_size,
            )
        else:
            model, tokenizer = load_model_and_tokenizer(**model_config)
            
            # Add this block to add a new padding token if needed
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(tokenizer))
            
            generation_kwargs = dict(
                max_new_tokens=max_new_tokens, do_sample=False
            )
            TEMPLATE = get_template(
                model_name_or_path,
                chat_template=model_config.get("chat_template", None),
            )
            timing_recorder = _TimingRecorder()
            return partial(
                _hf_generate_with_batching,
                model=model,
                tokenizer=tokenizer,
                template=TEMPLATE,
                timing_recorder=timing_recorder,
                starting_batch_size=starting_batch_size,
                **generation_kwargs,
            )


if __name__ == "__main__":
    main()
