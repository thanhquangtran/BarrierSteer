import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_representations(rep_save_dir):
    # Walk through the directory and load all representations
    directions = {}
    signs = {}
    for root, _, files in os.walk(rep_save_dir):
        for file in files:
            if file.endswith("directions.pth"):
                rep = torch.load(os.path.join(root, file))
                category = file[: file.index("_directions.pth")]
                directions[category] = rep
            elif file.endswith("signs.pth"):
                sign = torch.load(os.path.join(root, file))
                category = file[: file.index("_signs.pth")]
                signs[category] = sign
    return directions, signs


def load_model_and_tokenizer(model_path, device=None, **kwargs):
    # Check if device_map is explicitly provided in kwargs
    device_map = kwargs.get("device_map", None)

    # If no device_map is specified, load on a single device to avoid device mismatch
    if device_map is None:
        if device is not None:
            # Use the specified device
            target_device = torch.device(device)
        elif torch.cuda.is_available():
            target_device = torch.device("cuda:0")
        else:
            target_device = torch.device("cpu")

        # Remove device from kwargs to avoid passing it to from_pretrained
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            **filtered_kwargs,
        ).eval()
        model = model.to(target_device)
    else:
        # Use the provided device_map (e.g., "auto" or custom mapping)
        # Remove device from kwargs to avoid passing it to from_pretrained
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            **filtered_kwargs,
        ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        # use_fast=False, padding_side="left", padding="max_length"
    )
    tokenizer.pad_token = (
        tokenizer.unk_token if tokenizer.pad_token is None else tokenizer.pad_token
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from sklearn.utils import check_random_state

    check_random_state(seed)
