
import argparse
import csv
import glob
import json
import os
import re
import sys

import torch
from torch.utils.data import random_split


def load_hidden_states(hidden_states_file):
    return torch.load(hidden_states_file, weights_only=False)


def load_labels(labels_file):
    with open(labels_file, "r") as f:
        labels_dict = json.load(f)
    return labels_dict


def load_behavior_ids_from_dataset(behavior_dataset_path):
    """Load behavior IDs from a behavior dataset CSV file"""
    behavior_ids = set()
    if behavior_dataset_path and os.path.exists(behavior_dataset_path):
        with open(behavior_dataset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                behavior_ids.add(row['BehaviorID'])
    return behavior_ids


def load_behavior_categories(behavior_dataset_path, category_column="SemanticCategory"):
    """Load mapping of BehaviorID to Category"""
    behavior_categories = {}
    if behavior_dataset_path and os.path.exists(behavior_dataset_path):
        with open(behavior_dataset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Use specified column as the grouping key
                # Sanitize category name for filename safety
                category = row.get(category_column, 'unknown')
                category = re.sub(r'[^\w\-_]', '_', category)
                behavior_categories[row['BehaviorID']] = category
    return behavior_categories


def match_hidden_states_and_labels(
    hidden_states, labels_dict, max_tokens=None, filter_behavior_ids=None
):
    """
    Match hidden states with labels, optionally filtering by behavior IDs
    
    Args:
        hidden_states: List of hidden state tensors
        labels_dict: Dictionary mapping behavior_id to list of entries
        max_tokens: Maximum tokens to keep per sentence
        filter_behavior_ids: Optional set of behavior IDs to include (if None, includes all)
    """
    all_hidden_states = []
    all_labels = []
    all_behavior_ids = [] # Track behavior IDs for splitting later if needed
    
    # Filter labels_dict if filter_behavior_ids is provided
    if filter_behavior_ids is not None:
        labels_dict = {k: v for k, v in labels_dict.items() if k in filter_behavior_ids}
        print(f"Filtered to {len(labels_dict)} behaviors")

    for idx, (key, entries) in enumerate(labels_dict.items()):
        for entry in entries:
            # Safety label is the opposite of "attack success" label
            sentence_label = int(not entry["label"])
            hidden_state = hidden_states[idx]
            # Truncate if max_tokens is specified
            if max_tokens is not None:
                hidden_state = hidden_state[:max_tokens]
            all_hidden_states.append(hidden_state)
            all_labels.extend([sentence_label] * hidden_state.shape[0])
            all_behavior_ids.extend([key] * hidden_state.shape[0])

    if not all_hidden_states:
        return torch.tensor([]), torch.tensor([]), []

    all_hidden_states = torch.cat(all_hidden_states)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    assert all_hidden_states.shape[0] == all_labels.shape[0]
    return all_hidden_states, all_labels, all_behavior_ids


def split_dataset(all_hidden_states, all_labels, test_size=0.2):
    """Split dataset into train and test sets"""
    if len(all_hidden_states) == 0:
        return [], []
        
    dataset = list(zip(all_hidden_states, all_labels))
    train_size = int((1 - test_size) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    return train_dataset, test_dataset


def process_method(root, method, model, max_tokens=None, filter_behavior_ids=None):
    hidden_states_pattern = f"hidden_states_{model}_layer*_*.pth"
    # Note: Layer number is wildcarded to be more robust, though usually 20
    
    model_specific_labels_dir = os.path.join(root, method, model, "results")

    # Handle special directory structures for certain methods
    if method == "HumanJailbreaks":
        # Check if model specific directory exists first
        if os.path.exists(model_specific_labels_dir):
            labels_dir = model_specific_labels_dir
        else:
            # HumanJailbreaks uses shared results directory
            labels_dir = os.path.join(root, method, "random_subset_5", "results")
    elif method == "DirectRequest":
        if os.path.exists(model_specific_labels_dir):
            labels_dir = model_specific_labels_dir
        else:
            # DirectRequest uses shared results directory
            labels_dir = os.path.join(root, method, "default", "results")
    else:
        # Other methods use model-specific results directory
        labels_dir = model_specific_labels_dir

    # Find the matching hidden states file
    # Look in method/model/hidden_states first
    if method == "DirectRequest":
        hidden_states_dir = os.path.join(root, method, "default", "hidden_states")
    else:
        hidden_states_dir = os.path.join(root, method, model, "hidden_states")
        
    hidden_states_files = glob.glob(
        os.path.join(hidden_states_dir, hidden_states_pattern)
    )

    # Filter part files and sort by modification time (newest first)
    hidden_states_files = [f for f in hidden_states_files if "_part" not in f]
    hidden_states_files.sort(key=os.path.getmtime, reverse=True)
    
    selected_hidden_states_file = None
    selected_labels_file = None

    for hs_file in hidden_states_files:
        # Check size > 1KB to filter empty files
        if os.path.getsize(hs_file) < 1024:
            continue
            
        basename = os.path.basename(hs_file)
        # Match format: hidden_states_{model}_layer{layer}_{timestamp}.pth
        match = re.search(f"hidden_states_{re.escape(model)}_layer\d+_(.*)\.pth", basename)
        if match:
            timestamp = match.group(1)
            # Check for matching label file
            label_candidate = os.path.join(labels_dir, f"{model}_{timestamp}.json")
            if os.path.exists(label_candidate) and os.path.getsize(label_candidate) > 10:
                selected_hidden_states_file = hs_file
                selected_labels_file = label_candidate
                print(f"Found valid matching pair: {basename} and {os.path.basename(label_candidate)}")
                break
    
    if not selected_hidden_states_file:
        print(f"Error: No valid hidden state and result pair found for {method} in {hidden_states_dir}")
        print("Checked candidate files (excluding parts):")
        for f in hidden_states_files[:5]: # Show top 5
             print(f"  {os.path.basename(f)} (size: {os.path.getsize(f)} bytes)")
        return None, None, [], None

    hidden_states_file = selected_hidden_states_file
    labels_file = selected_labels_file

    hidden_states = load_hidden_states(hidden_states_file)
    labels_dict = load_labels(labels_file)

    # Filter labels_dict keys to match hidden_states keys (behavior IDs) if needed
    # Assuming hidden_states is a dictionary or list that aligns. 
    # Current implementation assumes implicit alignment by index/order or dictionary keys.
    # The original implementation implies hidden_states is a dictionary of {behavior_id: tensor} 
    # OR a list aligned with labels_dict. 
    # Looking at load_hidden_states usage in original code (not shown fully), let's assume it returns a dict-like object or list.
    # If hidden_states is a dict, we can be more precise.
    
    if isinstance(hidden_states, dict):
         # Align by keys
         common_keys = set(hidden_states.keys()) & set(labels_dict.keys())
         if filter_behavior_ids:
             common_keys &= filter_behavior_ids
         
         # Re-construct aligned lists
         aligned_hidden_states = []
         aligned_labels_dict = {}
         for k in common_keys:
             aligned_hidden_states.append(hidden_states[k])
             aligned_labels_dict[k] = labels_dict[k]
         
         hidden_states = aligned_hidden_states
         labels_dict = aligned_labels_dict
    
    features, labels, behavior_ids = match_hidden_states_and_labels(
        hidden_states, labels_dict, max_tokens, filter_behavior_ids
    )
    
    return features, labels, behavior_ids, labels_dict # Return labels_dict for inspection if needed


def save_dataset(dataset, output_path):
    if not dataset:
        print(f"Warning: Empty dataset, skipping save to {output_path}")
        return
        
    hidden_states, labels = zip(*dataset)
    hidden_states = torch.stack(hidden_states)
    labels = torch.stack(labels)
    torch.save(
        {"hidden_states": hidden_states, "labels": labels},
        output_path,
    )
    print(f"Saved {len(dataset)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process HarmBench hidden states")
    parser.add_argument("--root", type=str, required=True, help="Root directory of HarmBench results")
    parser.add_argument("--methods", type=str, required=True, help="Comma-separated list of attack methods")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens per sentence")
    parser.add_argument("--training_behavior_dataset", type=str, default=None, help="Path to training behavior CSV to filter IDs")
    
    # New arguments for category splitting
    parser.add_argument("--split_by_category", action="store_true", help="Split output by FunctionalCategory")
    parser.add_argument("--behaviors_path", type=str, default=None, help="Path to behavior CSV for category mapping (required if split_by_category is True)")
    parser.add_argument("--category_column", type=str, default="FunctionalCategory", help="Column name to use for categorization (default: FunctionalCategory)")

    args = parser.parse_args()

    methods = args.methods.split(",")
    os.makedirs(args.output, exist_ok=True)
    
    # Load training behavior IDs if provided
    training_behavior_ids = None
    if args.training_behavior_dataset:
        print(f"Loading training behavior IDs from {args.training_behavior_dataset}")
        training_behavior_ids = load_behavior_ids_from_dataset(args.training_behavior_dataset)
        print(f"Loaded {len(training_behavior_ids)} training behavior IDs")

    # Load category mapping if splitting by category
    behavior_category_map = {}
    if args.split_by_category:
        if not args.behaviors_path:
             # Fallback to training_behavior_dataset if behaviors_path not explicit
             if args.training_behavior_dataset:
                 args.behaviors_path = args.training_behavior_dataset
             else:
                 raise ValueError("--behaviors_path (or --training_behavior_dataset) is required when --split_by_category is used")
        
        print(f"Loading behavior categories from {args.behaviors_path} using column '{args.category_column}'")
        behavior_category_map = load_behavior_categories(args.behaviors_path, args.category_column)
        print(f"Loaded categories for {len(behavior_category_map)} behaviors. Unique categories: {len(set(behavior_category_map.values()))}")


    for method in methods:
        print(f"Processing method: {method}")
        features, labels, behavior_ids, _ = process_method(
            args.root, method, args.model, args.max_tokens, filter_behavior_ids=training_behavior_ids
        )
        
        if features is None or len(features) == 0:
            print(f"No data found for {method}, failing.")
            sys.exit(1)

        if args.split_by_category:
            # Group by category
            category_data = {}
            
            for i in range(len(features)):
                bid = behavior_ids[i]
                # If behavior ID not in map (e.g. from a different split), use 'unknown' or skip
                # Assuming process_method filtered by training_behavior_ids which matches behaviors_path usually
                cat = behavior_category_map.get(bid, "unknown")
                
                if cat not in category_data:
                    category_data[cat] = {"features": [], "labels": []}
                
                category_data[cat]["features"].append(features[i])
                category_data[cat]["labels"].append(labels[i])
            
            # Save per category
            for cat, data in category_data.items():
                cat_features = torch.stack(data["features"])
                cat_labels = torch.stack(data["labels"])
                
                train_dataset, test_dataset = split_dataset(cat_features, cat_labels)
                
                # Use category name as the "method" part of filename
                # e.g. {category}_train.pt
                train_output = os.path.join(args.output, f"{cat}_train.pt")
                test_output = os.path.join(args.output, f"{cat}_test.pt")
                
                print(f"  Category '{cat}': {len(train_dataset)+len(test_dataset)} samples")
                save_dataset(train_dataset, train_output)
                save_dataset(test_dataset, test_output)
                
        else:
            # Standard single-file output
            train_dataset, test_dataset = split_dataset(features, labels)
            
            train_output = os.path.join(args.output, f"{method}_train.pt")
            test_output = os.path.join(args.output, f"{method}_test.pt")
    
            save_dataset(train_dataset, train_output)
            save_dataset(test_dataset, test_output)


if __name__ == "__main__":
    main()
