import argparse
import os
import random
import uuid
from collections import defaultdict

import pandas as pd
from datasets import load_dataset


def save_splits(rows, output_dir, dataset_name, split_ratio=0.8):
    if not rows:
        print(f"No rows to save for {dataset_name}")
        return

    # To stratify, we group by SemanticCategory first
    cat_buckets = defaultdict(list)
    for row in rows:
        cat_buckets[row["SemanticCategory"]].append(row)

    train_rows = []
    val_rows = []

    print(f"Splitting {dataset_name} (Train/Val ratio: {split_ratio})...")

    for cat, items in cat_buckets.items():
        random.shuffle(items)
        n_train = int(len(items) * split_ratio)
        # Ensure at least one val sample if possible and total > 1
        if n_train == len(items) and len(items) > 1:
            n_train -= 1

        train_rows.extend(items[:n_train])
        val_rows.extend(items[n_train:])

    # Shuffle final lists
    random.shuffle(train_rows)
    random.shuffle(val_rows)

    # Save Train
    df_train = pd.DataFrame(train_rows)
    train_path = os.path.join(
        output_dir, f"harmbench_behaviors_{dataset_name}_train.csv"
    )
    df_train.to_csv(train_path, index=False)
    print(f"  Train: {len(df_train)} behaviors saved to {train_path}")

    # Save Val
    df_val = pd.DataFrame(val_rows)
    val_path = os.path.join(output_dir, f"harmbench_behaviors_{dataset_name}_val.csv")
    df_val.to_csv(val_path, index=False)
    print(f"  Val:   {len(df_val)} behaviors saved to {val_path}")


def convert_beavertails(output_dir, n_samples=50, split="330k_train"):
    print(f"Loading BeaverTails ({split})...")
    try:
        ds = load_dataset(
            "PKU-Alignment/BeaverTails", split=split, trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading BeaverTails: {e}")
        return

    unsafe_ds = ds.filter(lambda x: not x["is_safe"])

    # Bucketing strategy:
    # 1. Identify all valid categories for each prompt.
    # 2. Assign prompt to the category that needs samples the most?
    #    Or just simple first-match for now, but shuffle dataset first to avoid bias.

    # Let's collect ALL indices for each category
    cat_indices = defaultdict(list)

    # We iterate once
    for i, item in enumerate(unsafe_ds):
        for cat, is_true in item["category"].items():
            if is_true:
                cat_indices[cat].append(i)

    final_rows = []
    seen_prompts = set()

    # Sampling
    # We want N per category.
    # We iterate categories.

    for cat, indices in cat_indices.items():
        # Shuffle indices for this category
        random.shuffle(indices)

        count = 0
        for idx in indices:
            if count >= n_samples:
                break

            prompt = unsafe_ds[idx]["prompt"]

            # Avoid duplicate prompts across the entire dataset if desired
            # But specific prompt might be good for multiple categories?
            # Usually HarmBench behaviors are unique.
            if prompt in seen_prompts:
                continue

            seen_prompts.add(prompt)

            row = {
                "Behavior": prompt,
                "FunctionalCategory": "standard",
                "SemanticCategory": cat,
                "Tags": f"beaver_tails,{cat}",
                "ContextString": "",
                "BehaviorID": f"beaver_{uuid.uuid4().hex[:8]}",
            }
            final_rows.append(row)
            count += 1

        print(f"  Category '{cat}': collected {count} samples")

    save_splits(final_rows, output_dir, "beavertails")


def convert_wildguard(output_dir, n_samples=50):
    print("Loading WildGuard (wildguardtrain)...")
    try:
        ds = load_dataset(
            "allenai/wildguardmix",
            name="wildguardtrain",
            split="train",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error loading WildGuard: {e}")
        return

    prompt_col = "prompt" if "prompt" in ds.column_names else "instruction"
    harmful_ds = ds.filter(lambda x: x["prompt_harm_label"] == "harmful")

    # Bucket by subcategory
    cat_items = defaultdict(list)
    for item in harmful_ds:
        cat_items[item["subcategory"]].append(item[prompt_col])

    final_rows = []

    for cat, prompts in cat_items.items():
        # Dedup prompts just in case
        prompts = list(set(prompts))
        random.shuffle(prompts)

        selected = prompts[:n_samples]
        for p in selected:
            row = {
                "Behavior": p,
                "FunctionalCategory": "standard",
                "SemanticCategory": cat,
                "Tags": f"wildguard,{cat}",
                "ContextString": "",
                "BehaviorID": f"wildguard_{uuid.uuid4().hex[:8]}",
            }
            final_rows.append(row)
        print(f"  Category '{cat}': collected {len(selected)} samples")

    save_splits(final_rows, output_dir, "wildguard")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./HarmBench/data/behavior_datasets")
    parser.add_argument("--n_samples", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Set seed
    random.seed(42)

    convert_beavertails(args.output_dir, args.n_samples)
    convert_wildguard(args.output_dir, args.n_samples)
