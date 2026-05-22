import argparse
import glob
import json

import torch


def merge_completion_parts(input_pattern, output_file):
    # Get all part files, excluding timing files
    all_files = sorted(glob.glob(input_pattern))
    part_files = [f for f in all_files if "_timing" not in f]

    if not part_files:
        raise ValueError(f"No completion files found matching pattern: {input_pattern} (excluded timing files)")

    print(f"Found {len(part_files)} completion files to merge")
    # Merge all parts
    merged_data = {}
    for part_file in part_files:
        print(f"Processing {part_file}")
        with open(part_file, "r") as f:
            part_data = json.load(f)
            for behavior_id, completions in part_data.items():
                if behavior_id not in merged_data:
                    merged_data[behavior_id] = []
                merged_data[behavior_id].extend(completions)

    print(f"Saving merged completion data to {output_file}")
    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=2)


def merge_timing_parts(input_pattern, output_file):
    # Get all timing part files
    all_files = sorted(glob.glob(input_pattern))
    part_files = [f for f in all_files if "_timing" in f]

    print(f"Found {len(part_files)} timing files to merge")
    if not part_files:
        print(f"Warning: No timing files found match pattern: {input_pattern}. Skipping timing merge.")
        return

    # Aggregate timing data across all parts
    total_generate_s = 0.0
    total_new_tokens = 0
    total_prompt_tokens = 0
    num_batches = 0
    # New detailed timing fields
    total_forward_pass_s = 0.0
    total_steering_overhead_s = 0.0
    
    total_check_s = 0.0
    total_solve_s = 0.0
    
    total_violated_tokens = 0
    total_violated_time_s = 0.0
    total_not_violated_tokens = 0
    total_not_violated_time_s = 0.0
    model_name = None
    save_path = None
    model_config = None

    for part_file in part_files:
        print(f"Processing {part_file}")
        with open(part_file, "r") as f:
            part_data = json.load(f)
            timing = part_data.get("timing", {})
            total_generate_s += timing.get("total_generate_s", 0.0)
            total_new_tokens += timing.get("total_new_tokens", 0)
            total_prompt_tokens += timing.get("total_prompt_tokens", 0)
            num_batches += timing.get("num_batches", 0)
            # Aggregate new detailed timing fields
            total_forward_pass_s += timing.get("total_forward_pass_s", 0.0)
            total_steering_overhead_s += timing.get("total_steering_overhead_s", 0.0)
            
            total_check_s += timing.get("total_check_s", 0.0)
            total_solve_s += timing.get("total_solve_s", 0.0)
            
            total_violated_tokens += timing.get("total_violated_tokens", 0)
            total_violated_time_s += timing.get("total_violated_time_s", 0.0)
            total_not_violated_tokens += timing.get("total_not_violated_tokens", 0)
            total_not_violated_time_s += timing.get("total_not_violated_time_s", 0.0)

            if model_name is None:
                model_name = part_data.get("model_name")
            if save_path is None:
                save_path = part_data.get("save_path")
            if model_config is None:
                model_config = part_data.get("model_config")

    # Compute aggregate metrics
    sec_per_new_token = total_generate_s / total_new_tokens if total_new_tokens > 0 else 0.0
    tokens_per_sec_new = total_new_tokens / total_generate_s if total_generate_s > 0 else 0.0

    merged_timing = {
        "total_generate_s": total_generate_s,
        "total_new_tokens": total_new_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "num_batches": num_batches,
        "sec_per_new_token": sec_per_new_token,
        "tokens_per_sec_new": tokens_per_sec_new,
    }
    
    # Add new detailed timing metrics if available
    if total_forward_pass_s > 0 or total_steering_overhead_s > 0:
        merged_timing["total_forward_pass_s"] = total_forward_pass_s
        merged_timing["total_steering_overhead_s"] = total_steering_overhead_s
        merged_timing["forward_pass_ratio"] = (
            total_forward_pass_s / total_generate_s if total_generate_s > 0 else None
        )
        merged_timing["steering_overhead_ratio"] = (
            total_steering_overhead_s / total_generate_s if total_generate_s > 0 else None
        )
    
    if total_check_s > 0 or total_solve_s > 0:
        merged_timing["total_check_s"] = total_check_s
        merged_timing["total_solve_s"] = total_solve_s
        
        # avg_monitor_ms: Average cost of checking per generated token
        if total_new_tokens > 0:
            merged_timing["avg_monitor_ms"] = (total_check_s / total_new_tokens) * 1000.0
        
        # avg_steer_ms: Average cost of solving per UNSAFE token
        if total_violated_tokens > 0:
            merged_timing["avg_steer_ms"] = (total_solve_s / total_violated_tokens) * 1000.0
        else:
            merged_timing["avg_steer_ms"] = 0.0
    
    if total_violated_tokens > 0 or total_not_violated_tokens > 0:
        # Recalculate violated/not_violated times from steering overhead only
        # (The part files might have been generated with old code that included forward pass time)
        # Check if the sum equals total_generate_s (wrong) vs total_steering_overhead_s (correct)
        sum_violated_not_violated = total_violated_time_s + total_not_violated_time_s
        if abs(sum_violated_not_violated - total_generate_s) < 0.01:
            # The values are wrong - they sum to total_generate_s instead of steering_overhead_s
            # Recalculate from steering_overhead_s proportionally
            print(f"Warning: Detected incorrect timing calculation in part files. Recalculating from steering_overhead_s.")
            if total_new_tokens > 0:
                total_violated_time_s = total_steering_overhead_s * (total_violated_tokens / total_new_tokens)
                total_not_violated_time_s = total_steering_overhead_s * (total_not_violated_tokens / total_new_tokens)
        
        merged_timing["total_violated_tokens"] = total_violated_tokens
        merged_timing["total_violated_time_s"] = total_violated_time_s
        merged_timing["total_not_violated_tokens"] = total_not_violated_tokens
        merged_timing["total_not_violated_time_s"] = total_not_violated_time_s
        merged_timing["ms_per_violated_token"] = (
            (total_violated_time_s / total_violated_tokens) * 1000.0
            if total_violated_tokens > 0 else None
        )
        merged_timing["ms_per_not_violated_token"] = (
            (total_not_violated_time_s / total_not_violated_tokens) * 1000.0
            if total_not_violated_tokens > 0 else None
        )
        merged_timing["violation_rate"] = (
            total_violated_tokens / total_new_tokens if total_new_tokens > 0 else None
        )

    payload = {
        "model_name": model_name,
        "save_path": save_path,
        "timing": merged_timing,
        "model_config": model_config,
    }

    print(f"Saving merged timing data to {output_file}")
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)


def merge_hidden_states_parts(input_pattern, output_file):
    # Get all part files
    part_files = sorted(glob.glob(input_pattern))

    if not part_files:
        raise ValueError(f"No files found matching pattern: {input_pattern}")

    # Load and merge all parts
    merged_states = None
    for part_file in part_files:
        print(f"Processing {part_file}")
        part_states = torch.load(part_file)

        if merged_states is None:
            merged_states = part_states
        else:
            # Concatenate along the appropriate dimension (usually dim=0)
            if isinstance(part_states, torch.Tensor):
                merged_states = torch.cat([merged_states, part_states], dim=0)
            elif isinstance(part_states, dict):
                for key in part_states:
                    if key not in merged_states:
                        merged_states[key] = []
                    if isinstance(part_states[key], torch.Tensor):
                        if key not in merged_states:
                            merged_states[key] = part_states[key]
                        else:
                            merged_states[key] = torch.cat(
                                [merged_states[key], part_states[key]], dim=0
                            )
                    else:
                        if key not in merged_states:
                            merged_states[key] = []
                        merged_states[key].extend(part_states[key])

    # Save merged states
    print(f"Saving merged hidden states to {output_file}")
    torch.save(merged_states, output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_pattern",
        type=str,
        required=True,
        help='Glob pattern for input files (e.g., "path/to/completions_part*.json" or "path/to/hidden_states_part*.pth")',
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save merged file",
    )
    args = parser.parse_args()

    # Determine the type of files to merge based on file extension and content
    if args.input_pattern.endswith(".json"):
        if "_timing" in args.input_pattern:
            merge_timing_parts(args.input_pattern, args.output_file)
        else:
            merge_completion_parts(args.input_pattern, args.output_file)
    elif args.input_pattern.endswith(".pth"):
        merge_hidden_states_parts(args.input_pattern, args.output_file)
    else:
        raise ValueError("Unsupported file type. Must be either .json or .pth")


if __name__ == "__main__":
    main()
