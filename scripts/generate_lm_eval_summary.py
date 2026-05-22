#!/usr/bin/env python3
"""
Generate summary comparison table from LM Evaluation Harness results.

This script reads *_summary.json files from outputs/lm_harness and generates
a comparison table showing accuracy metrics (using acc_norm if available)
across different model configurations.

Usage:
    python scripts/generate_lm_eval_summary.py \
        --results_dir outputs/lm_harness \
        --output_file outputs/lm_harness/lm_eval_comparison.txt
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Tasks to include in the summary (main benchmarks only, not subcategories)
MAIN_TASKS = ["gsm8k", "mmlu"]


def get_accuracy(task_results: Dict[str, Any]) -> Optional[float]:
    """
    Extract accuracy from task results, preferring acc_norm over acc.

    Args:
        task_results: Dictionary containing task metrics

    Returns:
        Accuracy value (acc_norm if available, else acc), or None if not found
    """
    if "acc_norm" in task_results:
        return task_results["acc_norm"]
    elif "acc" in task_results:
        return task_results["acc"]
    # For GSM8K, use exact_match metrics
    elif "exact_match,flexible-extract" in task_results:
        return task_results["exact_match,flexible-extract"]
    elif "exact_match,strict-match" in task_results:
        return task_results["exact_match,strict-match"]
    return None


def parse_model_name(filename: str) -> str:
    """
    Parse model configuration name from summary filename.

    Args:
        filename: Summary filename (e.g., "llama2_7b_base_summary.json")

    Returns:
        Model configuration name (e.g., "llama2_7b_base")
    """
    # Remove _summary.json suffix
    return filename.replace("_summary.json", "")


def find_summary_files(results_dir: Path) -> Dict[str, Path]:
    """
    Find all *_summary.json files in the results directory.

    Args:
        results_dir: Directory containing summary JSON files

    Returns:
        Dictionary mapping model_name -> summary_file_path
    """
    summaries = {}
    if not results_dir.exists():
        return summaries

    for item in results_dir.rglob("*_summary.json"):
        if item.is_file():
            model_name = parse_model_name(item.name)
            summaries[model_name] = item

    return summaries


def load_summary(summary_file: Path) -> Optional[Dict[str, Any]]:
    """
    Load summary JSON file.

    Args:
        summary_file: Path to summary JSON file

    Returns:
        Dictionary with task results, or None if loading fails
    """
    try:
        with open(summary_file) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {summary_file}: {e}")
        return None


def generate_summary(results_dir: Path, output_file: Optional[Path] = None):
    """
    Generate summary comparison table from LM evaluation results.

    Args:
        results_dir: Directory containing *_summary.json files
        output_file: Optional path to save summary. If None, saves to results_dir/lm_eval_comparison.txt
    """
    summaries = find_summary_files(results_dir)

    if not summaries:
        print(f"No *_summary.json files found in {results_dir}")
        return

    print(f"Found {len(summaries)} summary files: {sorted(summaries.keys())}")

    # Load all data
    data: Dict[str, Dict[str, Any]] = {}
    for model_name, summary_file in summaries.items():
        summary = load_summary(summary_file)
        if summary:
            data[model_name] = summary

    if not data:
        print("No valid summary data loaded")
        return

    # Group data by configuration name (strip _seed\d+)
    def get_group_name(name):
        return re.sub(r"_seed\d+", "", name)

    grouped_data = defaultdict(lambda: defaultdict(list))
    for model_name, summary in data.items():
        group = get_group_name(model_name)
        for task in summary:
            acc = get_accuracy(summary[task])
            if acc is not None:
                grouped_data[group][task].append(acc)
            elif task == "mmlu":
                # Fallback for MMLU subtasks
                subtasks = [v for k, v in summary.items() if k.startswith("mmlu_")]
                if subtasks:
                    sub_accuracies = [
                        get_accuracy(st)
                        for st in subtasks
                        if get_accuracy(st) is not None
                    ]
                    if sub_accuracies:
                        grouped_data[group][task].append(
                            sum(sub_accuracies) / len(sub_accuracies)
                        )

    group_names = sorted(grouped_data.keys())

    def format_stat(values):
        if not values:
            return "N/A"
        mean = sum(values) / len(values)
        if len(values) > 1:
            std = statistics.stdev(values)
            return f"{mean * 100:.2f} ± {std * 100:.2f}%"
        return f"{mean * 100:.2f}%"

    # Calculate column widths
    max_group_len = max(len(name) for name in group_names) if group_names else 10
    config_col_width = max(30, max_group_len + 2)
    task_col_width = 25

    # Generate table
    lines = []
    lines.append("=" * 140)
    lines.append("LM EVALUATION HARNESS RESULTS COMPARISON (GROUPED BY CONFIG)")
    lines.append("=" * 140)
    lines.append("")
    lines.append(
        "Note: Groups runs sharing the same name prefix (seed suffix removed)."
    )
    lines.append("      Values are Mean ± Std where multiple runs exist.")
    lines.append("")

    # Table header
    header = f"{'Configuration':<{config_col_width}}"
    for task in MAIN_TASKS:
        header += f" {task:>{task_col_width}}"
    header += f" {'Average Acc':>{task_col_width}}"
    lines.append(header)
    lines.append("-" * len(header))

    # Rows for each configuration group
    for group_name in group_names:
        row = f"{group_name:<{config_col_width}}"
        group_accuracies = []

        # Task accuracy columns
        for task in MAIN_TASKS:
            values = grouped_data[group_name].get(task, [])
            row += f" {format_stat(values):>{task_col_width}}"
            if values:
                group_accuracies.append(sum(values) / len(values))

        # Average accuracy column
        if group_accuracies:
            avg = sum(group_accuracies) / len(group_accuracies)
            row += f" {avg * 100:>23.2f}% "
        else:
            row += f" {'N/A':>{task_col_width}}"

        lines.append(row)

    lines.append("-" * len(header))
    lines.append("")

    # Summary statistics section
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 60)

    for group_name in group_names:
        group_accuracies = []
        for task in MAIN_TASKS:
            values = grouped_data[group_name].get(task, [])
            if values:
                group_accuracies.append(sum(values) / len(values))

        if group_accuracies:
            avg = sum(group_accuracies) / len(group_accuracies)

            # Count runs in group (approximate from first task with data)
            num_runs = 0
            for t_data in grouped_data[group_name].values():
                if t_data:
                    num_runs = len(t_data)
                    break
            runs_str = f" ({num_runs} runs)" if num_runs > 1 else ""

            lines.append(f"  {group_name}{runs_str}: Avg Acc: {avg * 100:.2f}%")

    lines.append("")

    # Output
    output_text = "\n".join(lines)
    print(output_text)

    # Save to file
    if output_file is None:
        output_file = results_dir / "lm_eval_comparison.txt"

    with open(output_file, "w") as f:
        f.write(output_text)

    print(f"\nSummary saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary comparison table from LM Evaluation Harness results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="outputs/lm_harness",
        help="Directory containing *_summary.json files (default: outputs/lm_harness)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path (default: results_dir/lm_eval_comparison.txt)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_file = Path(args.output_file) if args.output_file else None

    generate_summary(results_dir, output_file)


if __name__ == "__main__":
    main()
