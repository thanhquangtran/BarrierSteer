#!/usr/bin/env python3
"""
Generate summary comparison tables from HarmBench evaluation results.

This script reads results.json files from outputs/harmbench/{model_dir} and generates
a comparison table showing ASR (Attack Success Rate) metrics across different 
configurations. It groups runs with the same name prefix (ignoring seed suffixes).

Usage:
    python scripts/generate_harmbench_summary.py \
        --results_dir outputs/harmbench/2026-01-27-gemma \
        --output_file outputs/harmbench/2026-01-27-gemma/comparison_results.txt
"""

import argparse
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def compute_asr(results: Dict) -> float:
    """
    Compute ASR (Attack Success Rate) from results.json structure.

    ASR = (number of label=1) / (total number of completions) * 100
    label=1 means attack succeeded (harmful).
    """
    all_labels = []

    for behavior_id, completions in results.items():
        for completion in completions:
            if "label" in completion:
                all_labels.append(completion["label"])

    asr = (sum(all_labels) / len(all_labels) * 100) if all_labels else 0.0
    return asr


def find_evaluation_runs(results_dir: Path) -> Dict[str, Path]:
    """
    Find all evaluation run directories and extract mode names.

    Returns:
        Dictionary mapping mode_name -> run_dir_path
    """
    runs = {}
    if not results_dir.exists():
        return runs

    for item in results_dir.iterdir():
        if item.is_dir() and item.name.startswith("harmbench_eval_"):
            # Try new format first: harmbench_eval_{model}_{MODE}-{TIMESTAMP}
            timestamp_match = re.search(r"-(?:\d{2}-){2}\d{2}$", item.name)
            if timestamp_match:
                name_without_ts = item.name[: timestamp_match.start()]
                if name_without_ts.startswith("harmbench_eval_"):
                    rest = name_without_ts[len("harmbench_eval_") :]
                    runs[rest] = item
                    continue

            # Fallback for old "steering_" format
            match = re.search(r"steering_\d{8}_\d{6}_(.+)$", item.name)
            if match:
                mode = match.group(1)
                runs[mode] = item
            else:
                runs[item.name] = item

    return runs


def format_stat(values: List[float], width: int = 0) -> str:
    """Format a list of values as Mean ± Std (if multiple files)."""
    if not values:
        return f"{'N/A':>{width}}" if width else "N/A"

    mean = sum(values) / len(values)

    if len(values) > 1:
        std = statistics.stdev(values)
        res = f"{mean:.2f} ± {std:.2f}%"
    else:
        res = f"{mean:.2f}%"

    if width:
        return f"{res:>{width}}"
    return res


def generate_summary(results_dir: Path, output_file: Optional[Path] = None):
    """
    Generate summary comparison tables from evaluation results.
    """
    runs = find_evaluation_runs(results_dir)

    if not runs:
        print(f"No evaluation runs found in {results_dir}")
        return

    print(f"Found {len(runs)} evaluation runs.")

    # 1. Collect Data
    # Structure: group_name -> list of {method -> score} (one dict per run)
    grouped_data: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    all_methods = set()

    # Sort runs to ensure stable order if needed, though grouping handles it
    for mode_name, run_dir in sorted(runs.items()):
        # Grouping logic: strip _seedXXX
        group_name = re.sub(r"_seed\d+", "", mode_name)

        run_scores = {}

        # Find all methods in this run directory
        for method_dir in run_dir.iterdir():
            if method_dir.is_dir():
                results_file = method_dir / "results.json"
                if results_file.exists():
                    method_name = method_dir.name
                    all_methods.add(method_name)

                    try:
                        with open(results_file) as f:
                            results = json.load(f)
                        asr = compute_asr(results)
                        run_scores[method_name] = asr
                    except Exception as e:
                        print(f"Warning: Failed to load {results_file}: {e}")

        # Only add run if it has at least one result
        if run_scores:
            grouped_data[group_name].append(run_scores)

    if not grouped_data:
        print("No valid results.json files found.")
        return

    sorted_groups = sorted(grouped_data.keys())
    sorted_methods = sorted(all_methods)

    # 2. Generate Table
    lines = []
    lines.append("=" * 160)
    lines.append("HARMBENCH RESULTS COMPARISON (Mean ± Std)")
    lines.append("=" * 160)
    lines.append("")
    lines.append(
        "Note: Groups runs sharing the same name prefix (seed suffix removed)."
    )
    lines.append(f"Source: {results_dir}")
    lines.append("")

    # Calculate column widths
    max_group_len = max(len(g) for g in sorted_groups) if sorted_groups else 20
    config_col_width = max(30, max_group_len + 2)
    method_col_width = 18

    # Header
    header = f"{'Configuration':<{config_col_width}}"
    for method in sorted_methods:
        m_name = method[:15]
        header += f" {m_name:>{method_col_width}}"
    header += f" {'Avg ASR':>{method_col_width}}"

    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for group in sorted_groups:
        row = f"{group:<{config_col_width}}"

        # Collect statistics per method across runs
        # grouped_data[group] is list of dicts: [{m1: 10, m2: 20}, {m1: 12, m2: 22}]
        runs_list = grouped_data[group]

        # Per-Method Columns
        for method in sorted_methods:
            values = [r[method] for r in runs_list if method in r]
            row += f" {format_stat(values):>{method_col_width}}"

        # Average ASR Column
        # Calculate the average ASR for *each run*, then compute Mean ± Std of those averages
        run_averages = []
        for r in runs_list:
            if r:
                avg = sum(r.values()) / len(r)
                run_averages.append(avg)

        row += f" {format_stat(run_averages):>{method_col_width}}"

        lines.append(row)

    lines.append("-" * len(header))

    # Calculate overall stats for single (non-multi) and multi runs
    single_behavior_vals = []
    multi_behavior_vals = []

    for group in sorted_groups:
        runs_list = grouped_data[group]
        run_avgs = []
        for r in runs_list:
            if r:
                avg = sum(r.values()) / len(r)
                run_avgs.append(avg)

        if "multi" in group:
            multi_behavior_vals.extend(run_avgs)
        else:
            single_behavior_vals.extend(run_avgs)

    lines.append("")
    if single_behavior_vals:
        lines.append(
            f"Overall (Single)             {format_stat(single_behavior_vals):>{method_col_width}}"
        )
    if multi_behavior_vals:
        lines.append(
            f"Overall (Multi)              {format_stat(multi_behavior_vals):>{method_col_width}}"
        )
    lines.append("")

    # Output
    output_text = "\n".join(lines)

    if output_file is None:
        output_file = results_dir / "harmbench_comparison.txt"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(output_text)

    print(f"\nSummary written to: {output_file}")
    print("\n" + output_text)


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary comparison tables from HarmBench evaluation results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing evaluation run folders",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_file = Path(args.output_file) if args.output_file else None

    generate_summary(results_dir, output_file)


if __name__ == "__main__":
    main()
